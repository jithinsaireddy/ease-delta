"""Dependency proposals: which predicates should a record be compared with?

Comparing every record with every predicate never misses a dependency and costs one edge per
predicate. A proposer compares a record only with the predicates it judges related. That is
cheaper, and it can be wrong: a missed link means a record that bears on a predicate is never read
against it. The engine then stays efficient and consistently wrong, which is the failure the
blueprint warns about.

So the threshold is not tuned by eye. It is set by conformal risk control (Angelopoulos, Bates,
Fisch, Lei & Schuster, ICLR 2024):

    Let L_i(lam) be the fraction of true links of calibration task i that the proposer misses at
    threshold lam. L_i is non-increasing as lam decreases and bounded by B = 1. Choose

        lam_hat = max{ lam : (n / (n + 1)) * mean_i L_i(lam) + B / (n + 1) <= alpha }.

    If calibration tasks and the next task are exchangeable, then
        E[ L_new(lam_hat) ] <= alpha.

What this does and does not give:
  * It bounds the *expected* miss rate over tasks like the calibration tasks. It is not a bound for
    each individual task, and it says nothing about tasks of a different kind.
  * "True links" are those the calibration data knows about. Links that require inference the
    similarity model cannot represent (a leg injury bearing on a cycling commute) are missed by any
    threshold that saves work. Measured on STALE in docs/RESULTS.md.

The proposer is off by default. `Engine(linker=None)` compares everything with everything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ease.util import get_device, text_hash

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Mean-pooled, L2-normalised sentence embeddings from a small frozen encoder, with a cache."""

    def __init__(self, name: str = DEFAULT_EMBEDDER, device: Optional[str] = None, max_len: int = 256,
                 batch_size: int = 128):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = name
        self.device = get_device(device)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.max_len, self.batch_size = max_len, batch_size
        self.cache: dict[str, np.ndarray] = {}
        self._torch = torch
        self.version = "emb-" + text_hash(name)[:12]
        self.texts_encoded = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        keys = [text_hash(t) for t in texts]
        todo = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if todo:
            by_key = {k: t for k, t in zip(keys, texts)}
            order = sorted(todo, key=lambda k: len(by_key[k]))
            with torch.inference_mode():
                for s in range(0, len(order), self.batch_size):
                    ks = order[s : s + self.batch_size]
                    enc = self.tok([by_key[k] for k in ks], padding=True, truncation=True, max_length=self.max_len,
                                   return_tensors="pt").to(self.device)
                    h = self.model(**enc).last_hidden_state
                    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                    v = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
                    v = torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy()
                    for k, x in zip(ks, v):
                        self.cache[k] = x
            self.texts_encoded += len(todo)
        return np.stack([self.cache[k] for k in keys])


def crc_threshold(miss_by_task: np.ndarray, thresholds: np.ndarray, alpha: float, bound: float = 1.0) -> dict:
    """miss_by_task[i, j]: miss rate of task i at thresholds[j]; thresholds ascending, so miss rates
    are non-decreasing along j. Returns the largest threshold whose corrected risk is <= alpha."""
    n = miss_by_task.shape[0]
    corrected = (n / (n + 1.0)) * miss_by_task.mean(0) + bound / (n + 1.0)
    ok = np.where(corrected <= alpha)[0]
    if len(ok) == 0:
        return {"threshold": float("-inf"), "feasible": False, "corrected_risk": float(corrected[0]), "n": int(n),
                "note": "no threshold meets alpha with this many calibration tasks; link everything"}
    j = int(ok.max())
    return {"threshold": float(thresholds[j]), "feasible": True, "corrected_risk": float(corrected[j]),
            "empirical_risk": float(miss_by_task.mean(0)[j]), "n": int(n)}


def task_similarities(embedder: Embedder, episode: dict) -> dict:
    """For one episode: similarity of every record text to every predicate, and which pairs are true
    links according to the generator."""
    preds = episode["schema"]["predicates"]
    pv = embedder.encode([p["text"] for p in preds])
    texts, owner = [], []
    seen = set()
    for s in episode["steps"]:
        t = s["event"]["text"]
        if not t or (s["event"]["record_id"], t) in seen:
            continue
        seen.add((s["event"]["record_id"], t))
        texts.append(t)
        owner.append(episode["links"].get(s["event"]["record_id"]))
    tv = embedder.encode(texts)
    sim = tv @ pv.T
    pid_ix = {p["id"]: i for i, p in enumerate(preds)}
    linked = np.zeros_like(sim, dtype=bool)
    for r, o in enumerate(owner):
        if o is not None:
            linked[r, pid_ix[o]] = True
    return {"sim": sim, "linked": linked}


def calibrate(embedder: Embedder, episodes: Sequence[dict], alpha: float = 0.05,
              thresholds: Optional[np.ndarray] = None) -> dict:
    thresholds = np.linspace(-0.2, 0.9, 221) if thresholds is None else np.asarray(thresholds)
    rows, fan = [], []
    for ep in episodes:
        ts = task_similarities(embedder, ep)
        if not ts["linked"].any():
            continue
        s_true = ts["sim"][ts["linked"]]
        rows.append([(s_true < t).mean() for t in thresholds])
        fan.append([(ts["sim"] >= t).mean() for t in thresholds])
    miss = np.asarray(rows)
    res = crc_threshold(miss, thresholds, alpha)
    res["alpha"] = alpha
    res["embedder"] = embedder.name
    if res["feasible"]:
        j = int(np.argmin(np.abs(thresholds - res["threshold"])))
        res["fraction_of_pairs_kept"] = float(np.mean(fan, 0)[j])
    return res


@dataclass
class ProposerConfig:
    embedder: str = DEFAULT_EMBEDDER
    threshold: float = float("-inf")
    alpha: float = 0.05
    min_links: int = 1  # always compare with at least the closest predicate


class Proposer:
    """A Linker for ease.engine.Engine."""

    def __init__(self, cfg: ProposerConfig, embedder: Optional[Embedder] = None):
        self.cfg = cfg
        self.embedder = embedder or Embedder(cfg.embedder)
        self.version = f"prop-{self.embedder.version}-{cfg.threshold:.4f}"
        self.calls = 0
        self.kept = 0
        self.considered = 0

    def __call__(self, record_id: str, text: str, schema) -> list[str]:
        preds = schema.predicates
        if not preds:
            return []
        sims = (self.embedder.encode([text]) @ self.embedder.encode([p.text for p in preds]).T)[0]
        keep = [i for i, s in enumerate(sims) if s >= self.cfg.threshold]
        if len(keep) < self.cfg.min_links:
            keep = list(np.argsort(-sims)[: self.cfg.min_links])
        self.calls += 1
        self.kept += len(keep)
        self.considered += len(preds)
        return [preds[i].id for i in sorted(keep)]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"embedder": self.cfg.embedder, "threshold": self.cfg.threshold,
                                          "alpha": self.cfg.alpha, "min_links": self.cfg.min_links}, indent=2))

    @classmethod
    def load(cls, path: str | Path, embedder: Optional[Embedder] = None) -> "Proposer":
        return cls(ProposerConfig(**json.loads(Path(path).read_text())), embedder)
