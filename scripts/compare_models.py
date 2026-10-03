"""Paired comparison of two edge models on development sets.

Written for one question: would more training of the same model on the same corpora help? Model A
is the released edge model, model B the same model trained further
(`configs/diag_more_training.yaml`). Both read the same items, so the comparison is paired.

Development sets only. No test split is read, so nothing here spends test data.

    python scripts/compare_models.py --a runs/stage_a/final --b runs/diag_more_training/final \
        --out runs/results/more_training.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from ease.data.evalsets import load_eval_set
from ease.eval.metrics import softmax
from ease.util import atomic_write_json, read_json


def read(model_dir: str, rows: list[dict]) -> np.ndarray:
    from ease.scorer import ModelScorer

    scorer = ModelScorer(model_dir, batch_size=128)
    return softmax(np.stack([v[0] for v in scorer.score([(r["claim"], r["evidence"]) for r in rows])]))


def interval(x: np.ndarray, n_boot: int, rng) -> dict:
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    means = x[idx].mean(1)
    return {"value": float(x.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="runs/stage_a/final")
    ap.add_argument("--b", default="runs/diag_more_training/final")
    ap.add_argument("--eval-cache", default="runs/eval_cache")
    ap.add_argument("--sets", nargs="+", default=["vitaminc.dev", "mnli.dev", "unrelated.dev", "snli.dev"])
    ap.add_argument("--out", default="runs/results/more_training.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    ma, mb = read_json(Path(a.a) / "training_manifest.json"), read_json(Path(a.b) / "training_manifest.json")
    out = {"note": "Diagnostic; exploratory. Development sets only. B is A trained further on the same mixture.",
           "a": {"dir": a.a, "examples": ma["examples"]},
           "b": {"dir": a.b, "additional_examples": mb["examples"], "by_source": mb["examples_by_source"]},
           "sets": {}}
    rng = np.random.default_rng(0)
    for name in a.sets:
        if not name.endswith(".dev"):
            raise SystemExit(f"{name}: this script reads development sets only")
        rows = load_eval_set(a.eval_cache, name)
        y = np.asarray([r["label"] for r in rows])
        pa, pb = read(a.a, rows), read(a.b, rows)
        ca, cb = (pa.argmax(1) == y).astype(float), (pb.argmax(1) == y).astype(float)
        la = -np.log(np.clip(pa[np.arange(len(y)), y], 1e-12, 1))
        lb = -np.log(np.clip(pb[np.arange(len(y)), y], 1e-12, 1))
        only_a, only_b = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
        out["sets"][name] = {
            "n": len(rows),
            "accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()),
            "accuracy_difference": interval(cb - ca, a.n_boot, rng),
            "nll_a": float(la.mean()), "nll_b": float(lb.mean()),
            "nll_difference": interval(lb - la, a.n_boot, rng),
            "only_a_right": only_a, "only_b_right": only_b,
            "mcnemar_p": float(binomtest(only_b, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0,
        }
        s = out["sets"][name]
        print(f"{name:14s} n={s['n']:5d}  A {s['accuracy_a']:.4f}  B {s['accuracy_b']:.4f}  "
              f"diff {100*s['accuracy_difference']['value']:+.2f} pts [{100*s['accuracy_difference']['lo']:+.2f}, "
              f"{100*s['accuracy_difference']['hi']:+.2f}]  only A right {only_a}, only B right {only_b}, p={s['mcnemar_p']:.3f}  "
              f"NLL {s['nll_a']:.4f} -> {s['nll_b']:.4f} [{s['nll_difference']['lo']:+.4f}, {s['nll_difference']['hi']:+.4f}]",
              flush=True)
    atomic_write_json(a.out, out)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
