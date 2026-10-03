"""X2 (exploratory, not pre-registered): small tasks.

The hand-off demo has four predicates and at most six records. Training and test episodes have
about eleven records per task, and every record is compared with every predicate, so the learned
refiner almost never sees a predicate with only one or two records. In the demo the refiner
weakened correct readings exactly then. This measures whether that is general.

Episodes: 2-4 predicates, at most one distractor at setup, 4-8 events, from the test atom pool with
a seed no other experiment uses.

    python scripts/eval_small.py --edge runs/stage_a/final --stage-b runs/stage_b --out runs/results/small.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ease.aggregate import Calibration, RuleAggregator, load_aggregator
from ease.data.atoms import load_pool
from ease.data.episodes import GenConfig, describe, generate
from ease.eval.harness import compare, run_engine, summarise
from ease.eval.trace import EdgeTable, build_table
from ease.util import atomic_write_json, read_json

SMALL = GenConfig(n_pred=(2, 4), n_distractors=(0, 1), n_events=(4, 8), n_actions=(1, 2), max_children=3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--work", default="runs/results/work")
    ap.add_argument("--out", default="runs/results/small.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    from ease.scorer import ModelScorer

    episodes = generate(load_pool(a.atoms, "test"), a.n, 20261001, SMALL, tag="small")
    d = Path(a.work) / "table_SMALL_edge"
    if (d / "table.json").exists():
        table = EdgeTable.load(d)
    else:
        table = build_table(episodes, ModelScorer(a.edge, batch_size=128))
        table.save(d)
    sb = read_json(Path(a.stage_b) / "summary.json")
    cal = read_json(Path(a.stage_b) / "rule_calibration.json")
    results = {"B1-soft": run_engine("B1-soft", episodes, table, RuleAggregator(Calibration(cal["temperature"], tuple(cal["bias"])))),
               "B1-hard": run_engine("B1-hard", episodes, table, RuleAggregator(hard=True))}
    for r in sb["runs"]:
        tag = ("E" if r["kind"] == "refined" else "N") + f"-seed{r['seed']}"
        results[tag] = run_engine(tag, episodes, table, load_aggregator(r["dir"]))
    out = {"experiment": "X2 (exploratory, not pre-registered): small tasks", "episodes": describe(episodes),
           "systems": {k: summarise(v, a.n_boot) for k, v in results.items()},
           "comparisons": {f"{k} vs B1-soft": compare(results[k], results["B1-soft"], a.n_boot)
                           for k in results if k != "B1-soft"}}
    atomic_write_json(a.out, out)
    for k, s in out["systems"].items():
        print(f"{k:10s} disposition {s['disposition_accuracy']['value']:.4f} status {s['status_accuracy']['value']:.4f} "
              f"nll {s['status_nll']['value']:.4f} false READY {s['false_ready_rate']['value']:.4f}", flush=True)
    for k, c in out["comparisons"].items():
        print(f"{k:22s} status NLL diff {c['status_nll']['difference']:+.4f} [{c['status_nll']['lo']:+.4f}, {c['status_nll']['hi']:+.4f}] "
              f"disposition diff {100*c['disposition_accuracy']['difference']:+.2f} pts", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
