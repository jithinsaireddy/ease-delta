"""Does the shipped configuration run without a GPU? Update time and memory on a chosen device.

Times the release (edge model, calibrated rules, canonical shapes) on the first test episodes of
each kind, once per call, and merges the result into the output file under `--label`. Run it once
per configuration, on an idle machine:

    python scripts/eval_cpu.py --device mps --label "Apple GPU"
    python scripts/eval_cpu.py --device cpu --label "CPU, all cores"
    python scripts/eval_cpu.py --device cpu --threads 4 --label "CPU, 4 threads"

Timing only. No accuracy is computed, so no test labels are read.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


def stats(x) -> dict:
    x = np.asarray(x, float) * 1e3
    return {"n": int(len(x)), "mean_ms": float(x.mean()), "p50_ms": float(np.median(x)),
            "p95_ms": float(np.percentile(x, 95)), "max_ms": float(x.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["cpu", "mps", "cuda"])
    ap.add_argument("--threads", type=int, default=0, help="0 leaves the library default")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="release/edge")
    ap.add_argument("--aggregator", default="release/aggregator")
    ap.add_argument("--rules-from", default=None,
                    help="a Stage B directory: use its calibrated rules instead of --aggregator")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--standard", type=int, default=40)
    ap.add_argument("--wide", type=int, default=15)
    ap.add_argument("--out", default="runs/results/cpu_timing.json")
    a = ap.parse_args()
    os.environ["EASE_DEVICE"] = a.device

    import torch

    if a.threads:
        torch.set_num_threads(a.threads)

    from ease.aggregate import load_aggregator
    from ease.data.atoms import load_pool
    from ease.data.episodes import STANDARD, WIDE, generate
    from ease.engine import Engine
    from ease.eval.trace import event_of
    from ease.schema import TaskSchema
    from ease.scorer import ModelScorer
    from ease.util import atomic_write_json, device_sync, get_device, read_json

    device = get_device()
    t0 = time.perf_counter()
    scorer = ModelScorer(a.model, canonical=True)
    if a.rules_from:
        from ease.aggregate import Calibration, RuleAggregator

        cal = read_json(Path(a.rules_from) / "rule_calibration.json")
        agg = RuleAggregator(Calibration(cal["temperature"], tuple(cal["bias"])))
    else:
        agg = load_aggregator(a.aggregator)
    load_s = time.perf_counter() - t0
    atoms = load_pool(a.atoms, "test")
    for ep in generate(atoms, 2, 7, STANDARD, tag="warm"):  # warm every shape family
        eng = Engine(TaskSchema.from_dict(ep["schema"]), scorer, agg)
        for s in ep["steps"]:
            eng.deliver(event_of(s))
    device_sync(device)

    entry = {"device": str(device), "threads": torch.get_num_threads(), "model_load_seconds": load_s, "sets": {}}
    for name, gen, n, seed in (("STANDARD", STANDARD, a.standard, 20260929), ("WIDE", WIDE, a.wide, 20260930)):
        t_inc, t_first, pairs = [], [], []
        for ep in generate(atoms, n, seed, gen, tag=name.lower())[:n]:
            scorer.cache.clear()
            eng = Engine(TaskSchema.from_dict(ep["schema"]), scorer, agg)
            setup = 0.0
            for step in ep["steps"]:
                device_sync(device)
                t = time.perf_counter()
                rep = eng.deliver(event_of(step))
                device_sync(device)
                dt = time.perf_counter() - t
                if step["setup"]:
                    setup += dt
                elif rep.receipt.changed:
                    t_inc.append(dt)
                    pairs.append(rep.edges_scored)
            t_first.append(setup)
        entry["sets"][name] = {"episodes": n, "update": stats(t_inc), "pairs_per_update": float(np.mean(pairs)),
                               "reading_a_new_task_in_full": stats(t_first)}
        print(f"{a.label:16s} {name:8s} update p50 {entry['sets'][name]['update']['p50_ms']:.1f} ms, p95 "
              f"{entry['sets'][name]['update']['p95_ms']:.1f} ms ({entry['sets'][name]['update']['n']} updates, "
              f"{entry['sets'][name]['pairs_per_update']:.1f} pairs each); first full reading of a task p50 "
              f"{entry['sets'][name]['reading_a_new_task_in_full']['p50_ms']/1e3:.2f} s", flush=True)
    entry["peak_memory_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    print(f"{a.label:16s} model load {load_s:.1f} s, peak memory {entry['peak_memory_gb']:.2f} GB, threads {entry['threads']}", flush=True)

    out = read_json(a.out) if Path(a.out).exists() else {
        "note": "Timing of the shipped configuration by device; exploratory. Canonical shapes. Apple M4 Max, "
                "12 performance and 4 efficiency cores. Times include ledger write, propagation, tokenisation and "
                "the model's forward passes.", "runs": {}}
    out["runs"][a.label] = entry
    atomic_write_json(a.out, out)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
