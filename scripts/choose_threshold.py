"""Choose the initial confidence required before an action is called READY.

Chosen on the Stage B validation episodes, never on test data: the smallest threshold whose
false-READY rate has a 95% upper bound at or below alpha. After deployment the threshold tracker
(ease/evolve/threshold.py) moves it according to verdicts.

    python scripts/choose_threshold.py --stage-b runs/stage_b --out runs/results/threshold_choice.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ease.aggregate import Calibration, RuleAggregator, load_aggregator
from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, generate
from ease.engine import EngineConfig
from ease.eval.harness import run_engine, summarise
from ease.eval.trace import EdgeTable
from ease.util import atomic_write_json, read_json

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="runs/results/threshold_choice.json")
    a = ap.parse_args()
    sb = read_json(Path(a.stage_b) / "summary.json")
    cal = read_json(Path(a.stage_b) / "rule_calibration.json")
    table = EdgeTable.load(Path(a.stage_b) / "table")
    eps = generate(load_pool(a.atoms, "stage_b"), sb["episodes"], sb.get("episode_seed", 20260901), STANDARD, tag="stageb")
    val = eps[-sb.get("val_episodes", 400):]
    aggs = {"refined": load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"]),
            "rules": RuleAggregator(Calibration(cal["temperature"], tuple(cal["bias"])))}
    out = {"alpha": a.alpha, "episodes": len(val), "source": "Stage B validation episodes", "by_aggregator": {}}
    for name, agg in aggs.items():
        rows = {}
        for t in THRESHOLDS:
            m = summarise(run_engine(name, val, table, agg, config=EngineConfig(ready_threshold=t)), 1000)
            rows[str(t)] = {k: m[k] for k in ("false_ready_rate", "missed_ready_rate", "disposition_accuracy")}
            print(f"{name:8s} threshold {t:.2f}: false READY {m['false_ready_rate']['value']:.4f} "
                  f"[{m['false_ready_rate']['lo']:.4f}, {m['false_ready_rate']['hi']:.4f}]  missed READY "
                  f"{m['missed_ready_rate']['value']:.4f}  accuracy {m['disposition_accuracy']['value']:.4f}", flush=True)
        ok = [float(t) for t in rows if rows[t]["false_ready_rate"]["hi"] <= a.alpha]
        out["by_aggregator"][name] = {"sweep": rows, "chosen": min(ok) if ok else max(THRESHOLDS),
                                      "met_alpha": bool(ok)}
    atomic_write_json(a.out, out)
    print({k: v["chosen"] for k, v in out["by_aggregator"].items()})
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
