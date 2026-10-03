"""Metrics. Every function takes plain numpy arrays so results can be recomputed from saved predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(-1) == labels).mean()) if len(labels) else float("nan")


def macro_f1(probs: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    pred = probs.argmax(-1)
    f1s = []
    for c in range(num_classes):
        tp = float(((pred == c) & (labels == c)).sum())
        fp = float(((pred == c) & (labels != c)).sum())
        fn = float(((pred != c) & (labels == c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared distance between the probability vector and the one-hot truth."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(-1).mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top prediction, equal-width confidence bins."""
    conf = probs.max(-1)
    correct = (probs.argmax(-1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, n = 0.0, len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def binary_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Calibration error of a probability of a binary event (not of the top class)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, n = 0.0, len(y)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        if m.any():
            total += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """Temperature minimising NLL (Guo et al., 2017). NLL is convex in 1/T, so a bounded scalar search finds it."""
    from scipy.optimize import minimize_scalar

    def f(log_t: float) -> float:
        return nll(softmax(logits, float(np.exp(log_t))), labels)

    r = minimize_scalar(f, bounds=(np.log(lo), np.log(hi)), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(r.x))


def contrast_set_accuracy(probs: np.ndarray, labels: np.ndarray, groups: Sequence[str]) -> dict:
    """Fraction of contrast sets answered entirely correctly.

    A contrast set is every row sharing a group id, e.g. one VitaminC case: the same claims judged
    against the evidence before and after a revision. Getting one row right and its twin wrong
    means the model did not track the revision, so only fully correct sets count.
    Sets with a single row carry no contrast and are excluded.
    """
    by = defaultdict(list)
    pred = probs.argmax(-1)
    for i, g in enumerate(groups):
        by[g].append(i)
    sets = [ix for ix in by.values() if len(ix) >= 2]
    if not sets:
        return {"n_sets": 0, "set_accuracy": float("nan")}
    ok = [all(pred[i] == labels[i] for i in ix) for ix in sets]
    return {"n_sets": len(sets), "set_accuracy": float(np.mean(ok))}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                 clusters: Optional[Sequence] = None) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap interval. If `clusters` is given, whole clusters are resampled,
    which is the right unit when rows within a task or contrast set are not independent."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    if clusters is None:
        n = len(values)
        idx = rng.integers(0, n, size=(n_boot, n))
        means = values[idx].mean(1)
    else:
        by = defaultdict(list)
        for v, c in zip(values, clusters):
            by[c].append(v)
        sums = np.array([np.sum(v) for v in by.values()])
        cnts = np.array([len(v) for v in by.values()])
        k = len(sums)
        idx = rng.integers(0, k, size=(n_boot, k))
        means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarise(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0,
              groups: Optional[Sequence[str]] = None) -> dict:
    p = softmax(logits, temperature)
    out = {
        "n": int(len(labels)),
        "accuracy": accuracy(p, labels),
        "macro_f1": macro_f1(p, labels),
        "nll": nll(p, labels),
        "brier": brier(p, labels),
        "ece": ece(p, labels),
    }
    if groups is not None:
        out.update(contrast_set_accuracy(p, labels, groups))
    return out
