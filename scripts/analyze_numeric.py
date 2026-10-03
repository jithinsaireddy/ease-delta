"""Is the edge model weaker on claims that compare quantities?

Splits an evaluation set by the form of the claim and reports accuracy for each part. Uses the
logits saved by eval_edge.py, so it needs no model.

    python scripts/analyze_numeric.py --logits runs/results/edge_logits --out runs/results/numeric.json
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from ease.data.evalsets import load_eval_set
from ease.util import atomic_write_json

COMPARATIVE = re.compile(
    r"\b(more than|less than|fewer than|greater than|higher than|lower than|at least|at most|no more than|"
    r"no less than|over|under|above|below|exceeds?|exceeded|before|after|prior to|earlier than|later than|"
    r"up to|nearly|almost|approximately|about|around)\b", re.I)
DIGIT = re.compile(r"\d")


def kind(claim: str) -> str:
    if DIGIT.search(claim) and COMPARATIVE.search(claim):
        return "number with comparison"
    if DIGIT.search(claim):
        return "number, no comparison"
    return "no number"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", default="runs/results/edge_logits")
    ap.add_argument("--eval-cache", default="runs/eval_cache")
    ap.add_argument("--sets", nargs="+", default=["vitaminc.dev", "vitaminc.test"])
    ap.add_argument("--out", default="runs/results/numeric.json")
    a = ap.parse_args()
    out = {"note": "A claim counts as a comparison if it contains a digit and a comparative phrase.", "sets": {}}
    rng = np.random.default_rng(0)
    for name in a.sets:
        p = Path(a.logits) / f"{name}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        rows = load_eval_set(a.eval_cache, name)[: len(z["labels"])]
        kinds = np.asarray([kind(r["claim"]) for r in rows])
        hit = (z["logits"].argmax(1) == z["labels"]).astype(float)
        entry = {}
        for k in ("number with comparison", "number, no comparison", "no number"):
            m = kinds == k
            if not m.any():
                continue
            h = hit[m]
            b = h[rng.integers(0, len(h), size=(2000, len(h)))].mean(1)
            entry[k] = {"n": int(m.sum()), "accuracy": float(h.mean()), "lo": float(np.quantile(b, 0.025)),
                        "hi": float(np.quantile(b, 0.975))}
        out["sets"][name] = entry
        print(name, {k: (v["n"], round(v["accuracy"], 4)) for k, v in entry.items()})
    atomic_write_json(a.out, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
