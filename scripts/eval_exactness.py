"""H1: after every event, does the cached state equal an independent full rebuild, bit for bit?

The rebuild bypasses the scorer's cache, so every edge is recomputed by the model.

    python scripts/eval_exactness.py --edge runs/stage_a/final --stage-b runs/stage_b --out runs/results/exactness.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


from ease.aggregate import load_aggregator
from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, generate
from ease.engine import Engine
from ease.eval.trace import event_of
from ease.schema import TaskSchema
from ease.scorer import ModelScorer
from ease.util import atomic_write_json, read_json


def run(episodes, scorer, agg, label):
    from ease.graph import INPUT, max_abs_difference, values_equal

    t0 = time.time()
    events = nodes = mismatched = flips = 0
    worst = 0.0
    worst_where = None
    for ei, ep in enumerate(episodes):
        scorer.cache.clear()
        eng = Engine(TaskSchema.from_dict(ep["schema"]), scorer, agg)
        for step in ep["steps"]:
            eng.deliver(event_of(step))
            saved, scorer.cache = scorer.cache, {}  # the rebuild must recompute edges, not look them up
            try:
                fresh = eng.g.full_recompute()
            finally:
                scorer.cache = saved
            events += 1
            for nid, node in eng.g.nodes.items():
                if node.kind == INPUT:
                    continue
                nodes += 1
                if not values_equal(node.value, fresh[nid]):
                    mismatched += 1
                    d = max_abs_difference(node.value, fresh[nid])
                    if d > worst:
                        worst, worst_where = d, (ep["episode_id"], step["t"], nid)
            flips += sum(1 for a in eng.schema.actions
                         if fresh[f"action:{a.id}"]["disposition"] != eng.g.value(f"action:{a.id}")["disposition"])
        if (ei + 1) % 20 == 0:
            print(f"  [{label}] {ei+1}/{len(episodes)} episodes; nodes checked={nodes} not equal={mismatched} "
                  f"max diff={worst:.3e} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return {"mode": label, "episodes": len(episodes), "events_checked": events, "nodes_checked": nodes,
            "nodes_not_bitwise_equal": mismatched, "max_abs_difference": worst, "worst": worst_where,
            "dispositions_that_differ_from_rebuild": flips, "minutes": round((time.time() - t0) / 60, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--dynamic-episodes", type=int, default=50)
    ap.add_argument("--out", default="runs/results/exactness.json")
    ap.add_argument("--pool", default="test")
    a = ap.parse_args()
    sb = read_json(Path(a.stage_b) / "summary.json")
    agg = load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"])
    episodes = generate(load_pool(a.atoms, a.pool), a.episodes, 20260929, STANDARD, tag="standard")
    out = {"hypothesis": "H1", "aggregator": agg.version, "pool": a.pool, "runs": []}
    canonical = ModelScorer(a.edge, canonical=True)
    out["edge_version"] = canonical.version
    out["runs"].append(run(episodes, canonical, agg, "canonical"))
    atomic_write_json(a.out, out)
    del canonical
    dynamic = ModelScorer(a.edge, canonical=False, batch_size=64)
    out["runs"].append(run(episodes[: a.dynamic_episodes], dynamic, agg, "dynamic"))
    out["H1_supported"] = out["runs"][0]["nodes_not_bitwise_equal"] == 0
    atomic_write_json(a.out, out)
    for r in out["runs"]:
        print(r, flush=True)
    print("H1 supported:", out["H1_supported"], flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
