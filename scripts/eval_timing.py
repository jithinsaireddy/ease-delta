"""H2, wall-clock part: seconds per event that changed the ledger, measured with the real models.

Run on an idle machine. Each system is warmed up before timing. Three systems are timed on the
same episodes and the same events:

    E         incremental update
    E-full    the same model re-reading everything
    B2        the dense reader re-reading every predicate's context

    python scripts/eval_timing.py --edge runs/stage_a/final --stage-b runs/stage_b --dense runs/dense/final
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from ease.aggregate import load_aggregator
from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, WIDE, generate
from ease.engine import Engine
from ease.eval.dense_data import active_records
from ease.eval.trace import event_of
from ease.model.dense import DenseModel, predict_contexts, render_context
from ease.schema import TaskSchema
from ease.scorer import ModelScorer
from ease.util import atomic_write_json, device_sync, get_device, read_json


def stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean_ms": float(x.mean() * 1e3), "p50_ms": float(np.median(x) * 1e3),
            "p95_ms": float(np.percentile(x, 95) * 1e3), "max_ms": float(x.max() * 1e3)}


def ratio_ci(num, den, ep, n_ep, n_boot=2000, seed=0):
    num, den, ep = np.asarray(num, float), np.asarray(den, float), np.asarray(ep)
    a = np.bincount(ep, weights=num, minlength=n_ep)
    b = np.bincount(ep, weights=den, minlength=n_ep)
    idx = np.random.default_rng(seed).integers(0, n_ep, size=(n_boot, n_ep))
    r = a[idx].sum(1) / np.maximum(b[idx].sum(1), 1e-12)
    return {"ratio": float(a.sum() / b.sum()), "lo": float(np.quantile(r, 0.025)), "hi": float(np.quantile(r, 0.975))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--dense", default="runs/dense/final")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--standard", type=int, default=100)
    ap.add_argument("--wide", type=int, default=40)
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--dense-max-len", type=int, default=4096)
    ap.add_argument("--out", default="runs/results/timing.json")
    ap.add_argument("--pool", default="test")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    device = get_device()
    sb = read_json(Path(a.stage_b) / "summary.json")
    agg = load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"])
    scorer = ModelScorer(a.edge, canonical=a.canonical, batch_size=64)
    dense = DenseModel.load(a.dense, device).eval()
    dtok = AutoTokenizer.from_pretrained(a.dense)
    atoms = load_pool(a.atoms, a.pool)
    out = {"hypothesis": "H2 (wall clock)", "device": str(device), "canonical": a.canonical, "pool": a.pool, "sets": {},
           "note": "Times include ledger write, graph propagation, tokenisation and model forward passes."}

    # warm up every code path and shape family
    warm = generate(atoms, 3, 7, STANDARD, tag="warm")
    for ep in warm:
        eng = Engine(TaskSchema.from_dict(ep["schema"]), scorer, agg)
        for s in ep["steps"]:
            eng.deliver(event_of(s))
        eng.verify(bypass_cache=True)
        ctx = render_context(active_records(eng.ledger))
        predict_contexts(dense, dtok, [p.text for p in eng.schema.predicates], [ctx] * len(eng.schema.predicates),
                         device, a.dense_max_len)
    device_sync(device)

    for name, gen, n, seed in (("STANDARD", STANDARD, a.standard, 20260929), ("WIDE", WIDE, a.wide, 20260930)):
        episodes = generate(atoms, n, seed, gen, tag=name.lower())
        t_inc, t_full, t_dense, ep_ix, kinds, unchanged = [], [], [], [], [], []
        pairs_inc, pairs_full = [], []
        for ei, ep in enumerate(episodes):
            scorer.cache.clear()
            eng = Engine(TaskSchema.from_dict(ep["schema"]), scorer, agg)
            claims = [p.text for p in eng.schema.predicates]
            for step in ep["steps"]:
                device_sync(device)
                t0 = time.perf_counter()
                rep = eng.deliver(event_of(step))
                device_sync(device)
                dt = time.perf_counter() - t0
                if step["setup"]:
                    continue
                if not rep.receipt.changed:
                    unchanged.append(dt)
                    continue
                # full rebuild of the same model on the same state
                saved = scorer.cache
                scorer.cache = {}
                before = scorer.stats.snapshot()
                device_sync(device)
                t0 = time.perf_counter()
                eng.g.full_recompute()
                device_sync(device)
                tf = time.perf_counter() - t0
                n_full = scorer.stats.since(before)["pairs_computed"]
                scorer.cache = saved
                # dense reader on the same state
                ctx = render_context(active_records(eng.ledger))
                device_sync(device)
                t0 = time.perf_counter()
                predict_contexts(dense, dtok, claims, [ctx] * len(claims), device, a.dense_max_len)
                device_sync(device)
                td = time.perf_counter() - t0
                t_inc.append(dt); t_full.append(tf); t_dense.append(td); ep_ix.append(ei); kinds.append(step["type"])
                pairs_inc.append(rep.edges_scored); pairs_full.append(n_full)
            if (ei + 1) % 10 == 0:
                print(f"  {name} {ei+1}/{len(episodes)} inc p50={np.median(t_inc)*1e3:.1f}ms "
                      f"full p50={np.median(t_full)*1e3:.1f}ms dense p50={np.median(t_dense)*1e3:.1f}ms", flush=True)
        entry = {"episodes": len(episodes), "changed_events": len(t_inc),
                 "E": stats(t_inc), "E-full": stats(t_full), "B2": stats(t_dense),
                 "unchanged_events(duplicate or stale)": stats(unchanged),
                 "pairs_per_event": {"E": float(np.mean(pairs_inc)), "E-full": float(np.mean(pairs_full))},
                 "time_ratio_E_over_E-full": ratio_ci(t_inc, t_full, ep_ix, len(episodes)),
                 "time_ratio_E_over_B2": ratio_ci(t_inc, t_dense, ep_ix, len(episodes)),
                 "by_event_type_p50_ms": {k: {"E": float(np.median([t for t, kk in zip(t_inc, kinds) if kk == k]) * 1e3),
                                              "E-full": float(np.median([t for t, kk in zip(t_full, kinds) if kk == k]) * 1e3),
                                              "B2": float(np.median([t for t, kk in zip(t_dense, kinds) if kk == k]) * 1e3)}
                                          for k in sorted(set(kinds))}}
        out["sets"][name] = entry
        print(name, {k: entry[k] for k in ("E", "E-full", "B2", "time_ratio_E_over_E-full", "time_ratio_E_over_B2")}, flush=True)
        atomic_write_json(a.out, out)

    # broad changes, where no advantage is expected
    ep = generate(atoms, 1, 99, WIDE, tag="broad")[0]
    scorer.cache.clear()
    eng = Engine(TaskSchema.from_dict(ep["schema"]), scorer, agg)
    for s in ep["steps"]:
        eng.deliver(event_of(s))
    n_edges = eng.counts().get("edge", 0)
    scorer.cache = {}
    device_sync(device); t0 = time.perf_counter(); eng.g.full_recompute(); device_sync(device)
    rebuild = time.perf_counter() - t0
    scorer.cache = {}
    eng.scorer.version = eng.scorer.version + "-changed"
    device_sync(device); t0 = time.perf_counter(); rep = eng.set_scorer(eng.scorer); device_sync(device)
    weights_change = time.perf_counter() - t0
    out["broad_change"] = {"edges_in_task": n_edges, "full_rebuild_ms": rebuild * 1e3,
                           "weights_change_ms": weights_change * 1e3,
                           "edges_recomputed_on_weights_change": rep.propagation["recomputed"].get("edge", 0),
                           "ratio": weights_change / rebuild}
    print("broad change:", out["broad_change"], flush=True)
    atomic_write_json(a.out, out)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
