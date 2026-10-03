"""H7: does the system improve from verified feedback without being retrained, and does the gate
keep out changes that would harm it?

Stream: SNLI test, a corpus absent from training whose labelling convention differs from the
training corpora. Items arrive in random order. After each prediction a verified label is returned
for 30% of items. Consolidation is attempted whenever 300 new verified readings have accumulated.

Three runs on the same stream and the same feedback draws:
    frozen      the trained system, never changed
    evolving    feedback is truthful
    corrupted   half of the returned labels are wrong

    python scripts/eval_evolution.py --edge runs/stage_a/final --stage-b runs/stage_b --out runs/results/evolution.json
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from ease.aggregate import load_aggregator
from ease.data.atoms import load_pool
from ease.data.evalsets import load_eval_set
from ease.engine import pair_key
from ease.evolve.consolidate import Bundle, GateConfig, Readings, VersionStore, consolidate, nll_rows, score
from ease.evolve.memory import MemoryConfig
from ease.labels import NEI
from ease.util import atomic_write_json, read_json


def readings_of(scorer, rows) -> Readings:
    got = scorer.score([(r["claim"], r["evidence"]) for r in rows])
    return Readings([pair_key(r["claim"], r["evidence"]) for r in rows],
                    np.asarray([r["label"] for r in rows], np.int64),
                    np.stack([g[0] for g in got]).astype(np.float32), np.stack([g[1] for g in got]).astype(np.float32))


def replay_rows(atoms, n, seed):
    """Labelled pairs from the Stage B pool: true pairs of atoms, plus unrelated crossings."""
    rng = random.Random(seed)
    atoms = [a for a in atoms if a["source"] != "snli"]
    rows = []
    while len(rows) < n:
        a = rng.choice(atoms)
        if rng.random() < 0.2:
            b = rng.choice(atoms)
            if b["page"] == a["page"]:
                continue
            rows.append({"claim": a["claim"], "evidence": rng.choice(b["versions"])["text"], "label": NEI})
        else:
            v = rng.choice(a["versions"])
            rows.append({"claim": a["claim"], "evidence": v["text"], "label": v["label"]})
    return rows


def run_stream(name, stream: Readings, order, feedback_mask, champion: Bundle, regression, replay, gate: GateConfig,
               corrupt: float, block: int, consolidate_every: int, seed: int, store_dir: Path, evolve: bool,
               set_suites=None):
    rng = np.random.default_rng(seed)
    bundle = champion
    store = VersionStore(store_dir)
    store.save(champion, note="trained champion")
    n = len(order)
    correct = np.zeros(n, bool)
    nll = np.zeros(n)
    fb_ix: list[int] = []
    fb_labels: list[int] = []
    since = 0
    events = []
    base_reg = {k: score(champion.read(r, None), r.labels) for k, r in regression.items()}
    if set_suites:
        from ease.evolve.consolidate import score_sets, set_status_probs
        for k, su in set_suites.items():
            base_reg[f"sets:{k}"] = score_sets(set_status_probs(champion, su, None), su.labels)
    for s in range(0, n, block):
        ix = order[s : s + block]
        part = stream.take(ix)
        mem = None
        if evolve and fb_ix:
            m = stream.take(fb_ix)
            mem = Readings(m.pairs, np.asarray(fb_labels, np.int64), m.logits, m.msgs)
        p = bundle.read(part, mem)
        correct[s : s + len(ix)] = p.argmax(1) == part.labels
        nll[s : s + len(ix)] = nll_rows(p, part.labels)
        if not evolve:
            continue
        for j, i in enumerate(ix):
            if feedback_mask[s + j]:
                y = int(stream.labels[i])
                if corrupt > 0 and rng.uniform() < corrupt:
                    y = int((y + rng.integers(1, 3)) % 3)
                fb_ix.append(int(i)); fb_labels.append(y); since += 1
        if since >= consolidate_every:
            since = 0
            m = stream.take(fb_ix)
            fb = Readings(m.pairs, np.asarray(fb_labels, np.int64), m.logits, m.msgs)
            new, rep = consolidate(bundle, fb, regression, replay, gate, set_suites=set_suites)
            reg_after = {k: score(new.read(r, fb), r.labels) for k, r in regression.items()}
            if set_suites:
                from ease.evolve.consolidate import score_sets, set_status_probs
                for k, su in set_suites.items():
                    reg_after[f"sets:{k}"] = score_sets(set_status_probs(new, su, fb), su.labels)
            ev = {"after_items": int(s + len(ix)), "feedback": len(fb), "adopted": rep.adopted, "chosen": rep.chosen,
                  "reason": rep.reason, "memory": dict(new.memory.__dict__),
                  "regression_accuracy": {k: v["accuracy"] for k, v in reg_after.items()},
                  "regression_accuracy_change_vs_trained": {k: reg_after[k]["accuracy"] - base_reg[k]["accuracy"]
                                                            for k in reg_after},
                  "holdout_gain": rep.candidates.get(rep.chosen, {}).get("gain_over_champion"),
                  "seconds": rep.seconds}
            events.append(ev)
            print(f"  [{name}] after {ev['after_items']} items, {ev['feedback']} verified: adopted={rep.adopted} "
                  f"({rep.chosen}) regression change={ {k: round(v, 4) for k, v in ev['regression_accuracy_change_vs_trained'].items()} }",
                  flush=True)
            if rep.adopted:
                bundle = new
                store.save(new, rep, note=f"{name}: after {len(fb)} verified readings")
    return {"correct": correct, "nll": nll, "events": events, "versions": store.lineage(),
            "regression_trained": {k: v["accuracy"] for k, v in base_reg.items()}}


def halves(correct_a, correct_b, n_boot=2000, seed=0):
    n = len(correct_a)
    h = n // 2
    a, b = correct_a[h:].astype(float), correct_b[h:].astype(float)
    d = a - b
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return {"second_half_items": int(len(d)), "accuracy_a": float(a.mean()), "accuracy_b": float(b.mean()),
            "difference": float(d.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
            "first_half_accuracy_a": float(correct_a[:h].mean()), "first_half_accuracy_b": float(correct_b[:h].mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--eval-cache", default="runs/eval_cache")
    ap.add_argument("--streams", nargs="+", default=["snli.test", "wanli.test"])
    ap.add_argument("--feedback-rate", type=float, default=0.30)
    ap.add_argument("--block", type=int, default=250)
    ap.add_argument("--consolidate-every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--out", default="runs/results/evolution.json")
    ap.add_argument("--work", default="runs/results/work/evolution")
    ap.add_argument("--limit", type=int, default=None, help="dry run: cap every set at this many rows")
    ap.add_argument("--champion", choices=("refined", "rules"), default="refined",
                    help="start from the trained refiner (H7) or from calibrated rules written as a refiner (X4)")
    ap.add_argument("--set-suites", default=None, help="regression_sets.npz for the set-level gate check")
    ap.add_argument("--train-context", action="store_true")
    a = ap.parse_args()
    cap = (lambda rows: rows[: a.limit]) if a.limit else (lambda rows: rows)

    from ease.scorer import ModelScorer

    scorer = ModelScorer(a.edge, batch_size=128)
    sb = read_json(Path(a.stage_b) / "summary.json")
    agg = load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"])
    if a.champion == "rules":
        from ease.aggregate import Calibration, RefinedAggregator, rule_refiner

        cal = read_json(Path(a.stage_b) / "rule_calibration.json")
        agg = RefinedAggregator(rule_refiner(Calibration(cal["temperature"], tuple(cal["bias"]))))
    champion = Bundle(agg.refiner, MemoryConfig(), a.champion)
    set_suites = None
    if a.set_suites:
        from ease.evolve.consolidate import load_set_suites

        set_suites = load_set_suites(a.set_suites)

    regression = {}
    for name in ("vitaminc.dev", "mnli.dev", "unrelated.dev"):
        rows = load_eval_set(a.eval_cache, name)
        regression[name] = readings_of(scorer, cap(rows[:5000]))
        scorer.cache.clear()
    replay = readings_of(scorer, replay_rows(load_pool(a.atoms, "stage_b"), a.limit or 6000, 1))
    scorer.cache.clear()
    print("regression sets:", {k: len(v) for k, v in regression.items()}, "replay:", len(replay), flush=True)

    out = {"hypothesis": "H7" if a.champion == "refined" and not a.set_suites else "X4 (exploratory)",
           "edge_version": scorer.version, "champion": a.champion, "champion_version": agg.version,
           "set_suites": a.set_suites, "train_context": a.train_context,
           "feedback_rate": a.feedback_rate, "streams": {}}
    gate = GateConfig(min_feedback=60 if a.limit else 200, train_context=a.train_context)
    for sname in a.streams:
        rows = cap(load_eval_set(a.eval_cache, sname))
        stream = readings_of(scorer, rows)
        scorer.cache.clear()
        rng = np.random.default_rng(a.seed)
        order = rng.permutation(len(stream))
        mask = rng.uniform(size=len(stream)) < a.feedback_rate
        t0 = time.time()
        runs = {}
        for rname, corrupt, evolve in (("frozen", 0.0, False), ("evolving", 0.0, True), ("corrupted", 0.5, True)):
            runs[rname] = run_stream(f"{sname}/{rname}", stream, order, mask, champion, regression, replay, gate,
                                     corrupt, a.block, a.consolidate_every, a.seed + 1,
                                     Path(a.work) / sname / rname, evolve, set_suites)
        entry = {"items": len(stream), "verified": int(mask.sum()), "seconds": round(time.time() - t0, 1)}
        for rname, r in runs.items():
            c = r["correct"]
            k = 10
            entry[rname] = {"accuracy": float(c.mean()), "nll": float(r["nll"].mean()),
                            "accuracy_by_tenth": [float(x.mean()) for x in np.array_split(c, k)],
                            "consolidations": r["events"], "versions": r["versions"],
                            "adopted": int(sum(e["adopted"] for e in r["events"]))}
        entry["evolving_vs_frozen"] = halves(runs["evolving"]["correct"], runs["frozen"]["correct"])
        entry["corrupted_vs_frozen"] = halves(runs["corrupted"]["correct"], runs["frozen"]["correct"])
        worst_drop = lambda r: max([max(-v for v in e["regression_accuracy_change_vs_trained"].values())
                                    for e in r["events"] if e["adopted"]] or [0.0])
        entry["H7a_improves"] = bool(entry["evolving_vs_frozen"]["lo"] > 0)
        entry["H7b_max_regression_drop_evolving"] = float(worst_drop(runs["evolving"]))
        entry["H7c_max_regression_drop_corrupted"] = float(worst_drop(runs["corrupted"]))
        entry["H7b_holds"] = bool(entry["H7b_max_regression_drop_evolving"] <= 0.005 + 1e-9)
        entry["H7c_holds"] = bool(entry["H7c_max_regression_drop_corrupted"] <= 0.005 + 1e-9)
        out["streams"][sname] = entry
        print(sname, {k: entry[k] for k in ("evolving_vs_frozen", "corrupted_vs_frozen", "H7a_improves", "H7b_holds",
                                            "H7c_holds")}, flush=True)
        atomic_write_json(a.out, out)
    primary = out["streams"][a.streams[0]]
    out["H7_supported"] = bool(primary["H7a_improves"] and primary["H7b_holds"] and primary["H7c_holds"])
    out["primary_stream"] = a.streams[0]
    atomic_write_json(a.out, out)
    print("H7 supported:", out["H7_supported"], flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
