"""Independent replication of the blueprint's mechanism probe, using this repository's ledger and executor.

The original probe's code was not supplied, only its results file, so this is a re-implementation
from the written description: 64 numeric inputs, three layers of 64 tanh nodes, 16 outputs (208
computed nodes); a sparse topology with eight local groups and a dense control; versioned events
(replacement, retraction, duplicate, stale delivery) over five seeds.

What this can and cannot show is unchanged from the original: it counts node evaluations in fixed
random circuits. It says nothing about language, accuracy, latency or any other model.

    python scripts/mechanism_probe.py --out runs/mechanism_probe.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ease.graph import IncrementalGraph
from ease.ledger import Ledger

N_IN, WIDTH, LAYERS, N_OUT, GROUPS = 64, 64, 3, 16, 8


def build(topology: str, rng: np.random.Generator, fan_in: int = 4):
    weights: dict[str, tuple[np.ndarray, float]] = {}

    def tanh(ids, parent_values):
        out = []
        for i, pv in zip(ids, parent_values):
            w, b = weights[i]
            out.append(float(np.tanh(float(np.dot(w, np.asarray(pv, dtype=np.float64))) + b)))
        return out

    g = IncrementalGraph({"tanh": tanh})
    for i in range(N_IN):
        g.add_input(f"in{i}", 0.0)
    prev = [f"in{i}" for i in range(N_IN)]
    per_group = WIDTH // GROUPS
    for layer in range(LAYERS):
        cur = []
        for j in range(WIDTH):
            nid = f"L{layer}_{j}"
            if topology == "dense":
                parents = list(prev)
            else:
                grp = j // per_group
                pool = prev[grp * per_group : (grp + 1) * per_group]
                parents = list(rng.choice(pool, size=min(fan_in, len(pool)), replace=False))
            weights[nid] = (rng.standard_normal(len(parents)) / np.sqrt(len(parents)), float(rng.standard_normal() * 0.1))
            g.add_node(nid, "tanh", parents)
            cur.append(nid)
        prev = cur
    outs_per_group = N_OUT // GROUPS
    for o in range(N_OUT):
        nid = f"out{o}"
        if topology == "dense":
            parents = list(prev)
        else:
            grp = o // outs_per_group
            pool = prev[grp * per_group : (grp + 1) * per_group]
            parents = list(rng.choice(pool, size=min(fan_in, len(pool)), replace=False))
        weights[nid] = (rng.standard_normal(len(parents)) / np.sqrt(len(parents)), 0.0)
        g.add_node(nid, "tanh", parents)
    g.propagate()
    return g


def run(topology: str, seeds: int, events_per_seed: int) -> dict:
    kinds = Counter()
    outcomes = Counter()
    recomputed_all, recomputed_on_change = [], []
    mismatches, worst = 0, 0.0
    total_nodes = None
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        g = build(topology, np.random.default_rng(seed))
        total_nodes = g.count_computed()
        led = Ledger()
        top = {i: 0 for i in range(N_IN)}
        history: dict[int, list[tuple[int, str | None]]] = {i: [] for i in range(N_IN)}
        for _ in range(events_per_seed):
            i = int(rng.integers(0, N_IN))
            rid = f"in{i}"
            r = rng.random()
            if r < 0.54 or not history[i]:
                kind = "replacement"
                top[i] += 1
                text = repr(float(rng.standard_normal()))
                rec = led.put(rid, top[i], text)
                history[i].append((top[i], text))
            elif r < 0.71:
                kind = "retraction"
                top[i] += 1
                rec = led.withdraw(rid, top[i])
                history[i].append((top[i], None))
            elif r < 0.90:
                kind = "duplicate"
                rev, text = history[i][-1]
                rec = led.put(rid, rev, text) if text is not None else led.withdraw(rid, rev)
            else:
                kind = "stale_delivery"
                rev, text = history[i][int(rng.integers(0, len(history[i])))]
                rec = led.put(rid, rev, text) if text is not None else led.withdraw(rid, rev)
            kinds[kind] += 1
            outcomes[rec.outcome.value] += 1
            active = led.active()
            value = float(active[rid].text) if rid in active else 0.0
            changed = g.set_input(rid, value)
            rep = g.propagate()
            recomputed_all.append(rep.total_recomputed)
            if changed:
                recomputed_on_change.append(rep.total_recomputed)
            elif rep.total_recomputed:
                raise AssertionError("an event that changed no input caused recomputation")
            v = g.verify()
            mismatches += v["not_bitwise_equal"]
            worst = max(worst, v["max_abs_difference"])
    n = len(recomputed_all)
    return {
        "topology": topology,
        "events": n,
        "event_types": dict(kinds),
        "ledger_outcomes": dict(outcomes),
        "events_with_numeric_input_changes": len(recomputed_on_change),
        "computed_nodes_in_full_forward_pass": total_nodes,
        "nodes_not_bitwise_equal_to_full_recompute": mismatches,
        "max_absolute_error": worst,
        "mean_nodes_recomputed_all_events": float(np.mean(recomputed_all)),
        "mean_nodes_recomputed_when_numeric_input_changes": float(np.mean(recomputed_on_change)),
        "mean_fraction_recomputed_when_numeric_input_changes": float(np.mean(recomputed_on_change)) / total_nodes,
        "node_evaluations_incremental": int(np.sum(recomputed_all)),
        "node_evaluations_full_rebuild_each_event": int(n * total_nodes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/mechanism_probe.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--events", type=int, default=300)
    a = ap.parse_args()
    res = {
        "title": "EASE-Delta mechanism probe (independent replication)",
        "scope": "Fixed random tanh circuits, float64, CPU. Counts node evaluations only. Sparse and dense "
                 "circuits are different random functions, not matched-accuracy models.",
        "sparse_fan_in_sweep": [],
        "experiments": [run("sparse", a.seeds, a.events), run("dense", a.seeds, a.events)],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    for e in res["experiments"]:
        print(f"{e['topology']:6s} events={e['events']} changed={e['events_with_numeric_input_changes']} "
              f"mean_recomputed_on_change={e['mean_nodes_recomputed_when_numeric_input_changes']:.2f}/"
              f"{e['computed_nodes_in_full_forward_pass']} "
              f"({100*e['mean_fraction_recomputed_when_numeric_input_changes']:.2f}%) "
              f"not_bitwise_equal={e['nodes_not_bitwise_equal_to_full_recompute']} max_err={e['max_absolute_error']}")
        print("        outcomes:", e["ledger_outcomes"], "types:", e["event_types"])


if __name__ == "__main__":
    main()
