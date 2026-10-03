"""Ship a regression suite with the model: readings of fixed, labelled pairs that any later
change must not make worse. The consolidation gate refuses to adopt anything without it.

    python scripts/build_regression_suite.py --edge runs/stage_a/final
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ease.data.atoms import load_pool
from ease.data.evalsets import load_eval_set
from ease.evolve.consolidate import save_readings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--eval-cache", default="runs/eval_cache")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--replay", type=int, default=6000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_evolution import readings_of, replay_rows

    from ease.scorer import ModelScorer

    scorer = ModelScorer(a.edge, batch_size=128)
    sets = {}
    for name in ("vitaminc.dev", "mnli.dev", "unrelated.dev"):
        sets[name] = readings_of(scorer, load_eval_set(a.eval_cache, name)[: a.rows])
        scorer.cache.clear()
    sets["replay"] = readings_of(scorer, replay_rows(load_pool(a.atoms, "stage_b"), a.replay, 1))
    out = Path(a.out) if a.out else Path(a.edge) / "regression_suite.npz"
    save_readings(out, sets)
    print({k: len(v) for k, v in sets.items()}, "->", out, f"({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
