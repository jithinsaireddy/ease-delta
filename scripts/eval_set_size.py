"""X3 (exploratory, not pre-registered): does the learned refiner depend on how many records a
predicate is compared with?

Designed after the STANDARD results showed E-declared (each record compared only with the
predicate it was written for) scoring below E, the opposite of what the extra information should
do. Declared linking makes each predicate's set of readings small. So does the dependency
proposer, and so do small tasks. If the refiner has learned something about set size that does not
transfer, calibrated rules should not show the drop.

Reuses the edge tables built by eval_episodes.py; runs on the CPU.

    python scripts/eval_set_size.py --stage-b runs/stage_b --work runs/results/work --out runs/results/set_size.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from ease.aggregate import Calibration, RuleAggregator, load_aggregator
from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, WIDE, generate
from ease.eval.harness import compare, run_engine, summarise
from ease.eval.trace import EdgeTable, trace
from ease.util import atomic_write_json, read_json

SETS = {"STANDARD": (STANDARD, 1000, 20260929), "WIDE": (WIDE, 300, 20260930)}


def status_accuracy_by_size(episodes, table, aggregators, max_size=16):
    """Accuracy of the predicate status as a function of how many readings the predicate has."""
    import torch

    from ease.aggregate import EdgeView

    tr = trace(episodes, table)
    sizes = np.asarray([len(e) for e in tr["edges"]])
    out = {}
    for name, agg in aggregators.items():
        pred = np.zeros(len(sizes), np.int64)
        for i, (idx, auth, vf, rids) in enumerate(zip(tr["edges"], tr["authority"], tr["valid_from"], tr["record_ids"])):
            edges = [EdgeView(r, table.logits[j], table.msgs[j], int(a), float(v)) for r, j, a, v in zip(rids, idx, auth, vf)]
            with torch.no_grad():
                pred[i] = int(np.argmax(agg.aggregate(edges, float(tr["prior"][i])).status))
        hit = pred == tr["label"]
        rows = {}
        for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, max_size), (max_size + 1, 10**6)):
            m = (sizes >= lo) & (sizes <= hi)
            if m.sum():
                rows[f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"] = {"n": int(m.sum()), "accuracy": float(hit[m].mean())}
        out[name] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--work", default="runs/results/work")
    ap.add_argument("--out", default="runs/results/set_size.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    sb = read_json(Path(a.stage_b) / "summary.json")
    cal = read_json(Path(a.stage_b) / "rule_calibration.json")
    rules = RuleAggregator(Calibration(cal["temperature"], tuple(cal["bias"])))
    refined = load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"])
    atoms = load_pool(a.atoms, "test")
    out = {"experiment": "X3 (exploratory, not pre-registered): readings per predicate", "sets": {}}
    for name, (gen, n, seed) in SETS.items():
        episodes = generate(atoms, n, seed, gen, tag=name.lower())
        table = EdgeTable.load(Path(a.work) / f"table_{name}_edge")
        res = {
            "E, all links": run_engine("E", episodes, table, refined),
            "E, declared links": run_engine("E-declared", episodes, table, refined, declared_links=True),
            "rules, all links": run_engine("rules", episodes, table, rules),
            "rules, declared links": run_engine("rules-declared", episodes, table, rules, declared_links=True),
        }
        entry = {"systems": {k: summarise(v, a.n_boot) for k, v in res.items()},
                 "comparisons": {
                     "E vs rules, declared links": compare(res["E, declared links"], res["rules, declared links"], a.n_boot),
                     "E vs rules, all links": compare(res["E, all links"], res["rules, all links"], a.n_boot),
                     "rules: declared vs all links": compare(res["rules, declared links"], res["rules, all links"], a.n_boot),
                     "E: declared vs all links": compare(res["E, declared links"], res["E, all links"], a.n_boot)},
                 "status_accuracy_by_readings_per_predicate": status_accuracy_by_size(
                     episodes[:300], table, {"E": refined, "rules": rules})}
        out["sets"][name] = entry
        for k, s in entry["systems"].items():
            print(f"{name:8s} {k:22s} disposition {s['disposition_accuracy']['value']:.4f} status {s['status_accuracy']['value']:.4f} "
                  f"nll {s['status_nll']['value']:.4f}", flush=True)
        for k, c in entry["comparisons"].items():
            print(f"{name:8s} {k:30s} disposition {100*c['disposition_accuracy']['difference']:+.2f} pts "
                  f"[{100*c['disposition_accuracy']['lo']:+.2f}, {100*c['disposition_accuracy']['hi']:+.2f}]  status NLL "
                  f"{c['status_nll']['difference']:+.4f} [{c['status_nll']['lo']:+.4f}, {c['status_nll']['hi']:+.4f}]", flush=True)
        print(f"{name:8s} status accuracy by readings per predicate: {entry['status_accuracy_by_readings_per_predicate']}",
              flush=True)
        atomic_write_json(a.out, out)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
