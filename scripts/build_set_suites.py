"""Set suites for the consolidation gate: whole predicates from validation episodes.

All episodes come from the Stage B atom pool (never the test pool):
    standard   the Stage B validation episodes (the refiner's own validation distribution)
    wide       larger tasks, generated with a seed no training run used
    small      smaller tasks, likewise

Each suite is capped so that the file stays small enough to ship with the model.

    python scripts/build_set_suites.py --edge runs/stage_a/final --stage-b runs/stage_b
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, WIDE, generate
from ease.eval.trace import EdgeTable, build_table, trace
from ease.evolve.consolidate import SetSuite, save_set_suites
from ease.util import read_json


def small_config():
    spec = importlib.util.spec_from_file_location("eval_small", Path(__file__).parent / "eval_small.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.SMALL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--work", default="runs/set_suites_work")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from ease.scorer import ModelScorer

    sb = read_json(Path(a.stage_b) / "summary.json")
    pool = load_pool(a.atoms, "stage_b")
    eps = generate(pool, sb["episodes"], sb.get("episode_seed", 20260901), STANDARD, tag="stageb")
    plan = {
        "standard": (eps[-sb.get("val_episodes", 400):], EdgeTable.load(Path(a.stage_b) / "table"), 3000),
        "wide": (generate(pool, 60, 20261002, WIDE, tag="suite-wide"), None, 1000),
        "small": (generate(pool, 400, 20261003, small_config(), tag="suite-small"), None, 3000),
    }
    suites = {}
    scorer = None
    for name, (episodes, table, cap) in plan.items():
        if table is None:
            d = Path(a.work) / f"table_{name}"
            if (d / "table.json").exists():
                table = EdgeTable.load(d)
            else:
                scorer = scorer or ModelScorer(a.edge, batch_size=128)
                table = build_table(episodes, scorer)
                table.save(d)
        tr = trace(episodes, table)
        suites[name] = SetSuite.from_trace(tr, table, limit=cap, seed=7)
        su = suites[name]
        print(f"{name:9s} from {len(episodes)} episodes: examples={len(su)} readings={len(su.logits)} "
              f"label counts={np.bincount(su.labels, minlength=4).tolist()}", flush=True)
    out = Path(a.out) if a.out else Path(a.edge) / "regression_sets.npz"
    save_set_suites(out, suites)
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
