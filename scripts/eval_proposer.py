"""H8: dependency proposals with a conformal bound on missed links.

Calibrate the similarity threshold on Stage B episodes, then measure on test episodes:
the fraction of true links missed, the work saved, and what it does to decisions.

    python scripts/eval_proposer.py --stage-b runs/stage_b --work runs/results/work --out runs/results/proposer.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from ease.aggregate import load_aggregator
from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, WIDE, generate
from ease.eval.harness import compare, run_engine, summarise
from ease.eval.trace import EdgeTable
from ease.proposer import Embedder, Proposer, ProposerConfig, calibrate, task_similarities
from ease.util import atomic_write_json, read_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--work", default="runs/results/work")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--calibration-episodes", type=int, default=1000)
    ap.add_argument("--out", default="runs/results/proposer.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pool", default="test")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    emb = Embedder()
    # calibration tasks come from the Stage B pool, with a seed no other component uses
    cal_eps = generate(load_pool(a.atoms, "stage_b"), a.calibration_episodes, 20260915, STANDARD, tag="propcal")
    cal = calibrate(emb, cal_eps, a.alpha)
    print("calibration:", cal, flush=True)
    out = {"hypothesis": "H8", "calibration": cal, "sets": {}}
    if not cal["feasible"]:
        out["H8_supported"] = False
        atomic_write_json(a.out, out)
        return 0

    sb = read_json(Path(a.stage_b) / "summary.json")
    agg = load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"])
    atoms = load_pool(a.atoms, a.pool)
    out["pool"] = a.pool
    for name, gen, n, seed in (("STANDARD", STANDARD, 1000, 20260929), ("WIDE", WIDE, 300, 20260930)):
        episodes = generate(atoms, a.limit or n, seed, gen, tag=name.lower())
        miss, kept = [], []
        for ep in episodes:
            ts = task_similarities(emb, ep)
            if ts["linked"].any():
                miss.append(float((ts["sim"][ts["linked"]] < cal["threshold"]).mean()))
            kept.append(float((ts["sim"] >= cal["threshold"]).mean()))
        miss = np.asarray(miss)
        rng = np.random.default_rng(0)
        boot = miss[rng.integers(0, len(miss), size=(a.n_boot, len(miss)))].mean(1)
        table = EdgeTable.load(Path(a.work) / f"table_{name}_edge")
        prop = Proposer(ProposerConfig(threshold=cal["threshold"], alpha=a.alpha), emb)
        with_prop = run_engine("E+proposer", episodes, table, agg, linker=prop)
        all_links = run_engine("E", episodes, table, agg)
        cmp_ = compare(with_prop, all_links, a.n_boot)
        entry = {"tasks": len(episodes),
                 "missed_link_rate": {"mean": float(miss.mean()), "lo": float(np.quantile(boot, 0.025)),
                                      "hi": float(np.quantile(boot, 0.975)),
                                      "tasks_with_any_miss": float((miss > 0).mean())},
                 "fraction_of_pairs_compared": float(np.mean(kept)),
                 "E+proposer": summarise(with_prop, a.n_boot), "E": summarise(all_links, a.n_boot),
                 "comparison": cmp_}
        t = cmp_["tokens_per_changed_event"]
        entry["work_reduction_factor"] = 1.0 / max(t["ratio"], 1e-12)
        entry["disposition_accuracy_change"] = cmp_["disposition_accuracy"]["difference"]
        out["sets"][name] = entry
        print(f"{name}: missed={miss.mean():.4f} [{entry['missed_link_rate']['lo']:.4f}, {entry['missed_link_rate']['hi']:.4f}] "
              f"pairs compared={np.mean(kept):.3f} work reduction x{entry['work_reduction_factor']:.2f} "
              f"accuracy change={entry['disposition_accuracy_change']:+.4f}", flush=True)
    out["H8_supported"] = bool(
        all(out["sets"][s]["missed_link_rate"]["mean"] <= a.alpha for s in out["sets"])
        and out["sets"]["WIDE"]["work_reduction_factor"] >= 2.0)
    out["keep_off_by_default"] = bool(any(out["sets"][s]["disposition_accuracy_change"] < -0.01 for s in out["sets"]))
    atomic_write_json(a.out, out)
    print("H8 supported:", out["H8_supported"], "| keep off by default:", out["keep_off_by_default"], flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
