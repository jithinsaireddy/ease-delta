"""X3b (exploratory, not pre-registered): the dependency proposer with calibrated rules.

H8 measured the proposer with the learned refiner and found it cost 2.3 points on STANDARD. X3
then showed the refiner degrades when predicates have few readings, which is what the proposer
produces, while calibrated rules improve. This measures the proposer with rules, at the threshold
already chosen by conformal risk control for H8 (runs/results/proposer.json); nothing is re-tuned.

    python scripts/eval_linking_rules.py --out runs/results/linking_rules.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ease.aggregate import Calibration, RuleAggregator, load_aggregator
from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, WIDE, generate
from ease.eval.harness import compare, run_engine, summarise
from ease.eval.trace import EdgeTable
from ease.proposer import Embedder, Proposer, ProposerConfig
from ease.util import atomic_write_json, read_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--work", default="runs/results/work")
    ap.add_argument("--proposer", default="runs/results/proposer.json")
    ap.add_argument("--out", default="runs/results/linking_rules.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    sb = read_json(Path(a.stage_b) / "summary.json")
    cal = read_json(Path(a.stage_b) / "rule_calibration.json")
    thr = read_json(a.proposer)["calibration"]["threshold"]
    rules = RuleAggregator(Calibration(cal["temperature"], tuple(cal["bias"])))
    refined = load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"])
    emb = Embedder()
    atoms = load_pool(a.atoms, "test")
    out = {"experiment": "X3b (exploratory): proposer with calibrated rules", "threshold": thr, "sets": {}}
    for name, gen, n, seed in (("STANDARD", STANDARD, 1000, 20260929), ("WIDE", WIDE, 300, 20260930)):
        episodes = generate(atoms, n, seed, gen, tag=name.lower())
        table = EdgeTable.load(Path(a.work) / f"table_{name}_edge")
        res = {
            "rules, all links": run_engine("rules", episodes, table, rules),
            "rules + proposer": run_engine("rules+p", episodes, table, rules,
                                           linker=Proposer(ProposerConfig(threshold=thr), emb)),
            "E + proposer": run_engine("E+p", episodes, table, refined,
                                       linker=Proposer(ProposerConfig(threshold=thr), emb)),
        }
        entry = {"systems": {k: summarise(v, a.n_boot) for k, v in res.items()},
                 "comparisons": {"rules + proposer vs rules, all links": compare(res["rules + proposer"], res["rules, all links"], a.n_boot),
                                 "rules + proposer vs E + proposer": compare(res["rules + proposer"], res["E + proposer"], a.n_boot)}}
        out["sets"][name] = entry
        for k, s in entry["systems"].items():
            print(f"{name:8s} {k:18s} disposition {s['disposition_accuracy']['value']:.4f} false READY "
                  f"{s['false_ready_rate']['value']:.4f} tokens/changed event {s['tokens_per_changed_event']['value']:.0f}",
                  flush=True)
        for k, c in entry["comparisons"].items():
            t = c.get("tokens_per_changed_event", {})
            print(f"{name:8s} {k:40s} disposition {100*c['disposition_accuracy']['difference']:+.2f} pts "
                  f"[{100*c['disposition_accuracy']['lo']:+.2f}, {100*c['disposition_accuracy']['hi']:+.2f}]"
                  + (f" tokens x{t['ratio']:.3f}" if "ratio" in t else ""), flush=True)
        atomic_write_json(a.out, out)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
