"""Where would more accuracy come from? Three diagnostics, none of which trains anything.

A  Fit.      Accuracy and loss of the edge model on a sample of its own training data against
             held-out data. Train much better than held-out: the model memorises and more of the
             same training will not help. Train about equal to held-out: the model has not yet
             fitted its data, so more training or more capacity can help.
B  Errors.   Confusion of the three labels on each evaluation set.
C  Headroom. Re-run the test episodes with a share of the reader's mistakes corrected, to see how
             task-level accuracy responds. Mistakes are split into misreading the record that was
             written for the requirement, and reading something into an unrelated record.

    python scripts/diagnose_headroom.py --out runs/results/headroom.json

Part C replays test episodes that were already evaluated. It is analysis, not a new test.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from ease.aggregate import Calibration, RuleAggregator
from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, WIDE, generate
from ease.data.sources import get_source
from ease.eval.harness import run_engine, summarise
from ease.eval.metrics import softmax
from ease.eval.trace import EdgeTable
from ease.labels import NEI, RELATION_NAMES
from ease.util import atomic_write_json, read_json


def sample_train(key: str, every: int, limit: int) -> list[dict]:
    """Every `every`-th row of the training split, in file order, so the whole file is covered."""
    from datasets import load_dataset

    spec = get_source(key)
    ds = load_dataset(spec.hf_id, split=spec.train_split, streaming=True)
    if spec.columns:
        ds = ds.select_columns(list(spec.columns))
    rows = []
    for i, raw in enumerate(ds):
        if i % every:
            continue
        r = spec.mapper(raw)
        if r is not None:
            rows.append(r)
        if len(rows) >= limit:
            break
    return rows


def fit_check(edge_dir: str, results_dir: Path) -> dict:
    from ease.scorer import ModelScorer

    scorer = ModelScorer(edge_dir, batch_size=128)
    edge = read_json(results_dir / "edge.json")
    out = {}
    for key, every, heldout in (("vitaminc", 60, "vitaminc.test"), ("mnli", 80, "mnli_mm.test"), ("wanli", 25, "wanli.test")):
        rows = sample_train(key, every, 6500)
        lg = np.stack([v[0] for v in scorer.score([(r["claim"], r["evidence"]) for r in rows])])
        scorer.cache.clear()
        y = np.asarray([r["label"] for r in rows])
        p = softmax(lg)
        tr = {"n": len(rows), "accuracy": float((p.argmax(1) == y).mean()),
              "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())}
        ho = edge["sets"][heldout]["raw"]
        out[key] = {"train_sample": tr, "held_out": {"set": heldout, "n": ho["n"], "accuracy": ho["accuracy"], "nll": ho["nll"]},
                    "accuracy_gap": tr["accuracy"] - ho["accuracy"], "nll_gap": ho["nll"] - tr["nll"]}
        print(f"A {key:9s} train sample acc {tr['accuracy']:.4f} nll {tr['nll']:.4f} (n={tr['n']}) | held-out {heldout} "
              f"acc {ho['accuracy']:.4f} nll {ho['nll']:.4f} | gap {100*out[key]['accuracy_gap']:+.2f} pts", flush=True)
    return out


def confusion(results_dir: Path) -> dict:
    out = {}
    for p in sorted((results_dir / "edge_logits").glob("*.npz")):
        z = np.load(p)
        pred, y = z["logits"].argmax(1), z["labels"]
        m = np.zeros((3, 3), int)
        for a, b in zip(y, pred):
            m[a, b] += 1
        rec = {RELATION_NAMES[i]: float(m[i, i] / max(1, m[i].sum())) for i in range(3)}
        out[p.stem] = {"rows_true_cols_pred": m.tolist(), "recall": rec, "n": int(len(y))}
    return out


def gold_of(episodes) -> dict:
    """(claim, record text) -> gold relation, from the generator's own bookkeeping."""
    gold, linked = {}, set()
    for ep in episodes:
        claims = {p["id"]: p["text"] for p in ep["schema"]["predicates"]}
        for s in ep["steps"]:
            if s["gold_relation"] is not None and s["target_pred"] and s["event"]["text"]:
                k = (claims[s["target_pred"]], s["event"]["text"])
                gold[k] = int(s["gold_relation"])
                linked.add(k)
    return gold, linked


def headroom(work: Path, stage_b: Path, atoms_dir: str, n_boot: int) -> dict:
    cal = read_json(stage_b / "rule_calibration.json")
    rules = RuleAggregator(Calibration(cal["temperature"], tuple(cal["bias"])))
    atoms = load_pool(atoms_dir, "test")
    out = {}
    for name, gen, n, seed in (("STANDARD", STANDARD, 1000, 20260929), ("WIDE", WIDE, 300, 20260930)):
        episodes = generate(atoms, n, seed, gen, tag=name.lower())
        table = EdgeTable.load(work / f"table_{name}_edge")
        gold, linked = gold_of(episodes)
        key_gold = np.full(len(table.keys), NEI, np.int64)
        is_linked = np.zeros(len(table.keys), bool)
        for (c, t), lab in gold.items():
            i = table.keys.get(table.key(c, t))
            if i is not None:
                key_gold[i] = lab
        for (c, t) in linked:
            i = table.keys.get(table.key(c, t))
            if i is not None:
                is_linked[i] = True
        pred = table.logits.argmax(1)
        wrong = pred != key_gold
        entry = {"pairs": int(len(pred)), "linked_pairs": int(is_linked.sum()),
                 "reading_accuracy": {"all": float((~wrong).mean()), "linked": float((~wrong[is_linked]).mean()),
                                      "unrelated": float((~wrong[~is_linked]).mean())},
                 "errors": {"on_linked_pairs": int((wrong & is_linked).sum()),
                            "on_unrelated_pairs": int((wrong & ~is_linked).sum())},
                 "conditions": {}}
        order = np.random.default_rng(0).permutation(np.where(wrong)[0])
        masks = {"as measured": np.zeros(len(pred), bool)}
        for q in (0.25, 0.5, 0.75):
            m = np.zeros(len(pred), bool)
            m[order[: int(round(q * len(order)))]] = True
            masks[f"{int(q*100)}% of reading errors corrected"] = m
        masks["errors on unrelated records corrected"] = wrong & ~is_linked
        masks["errors on the linked record corrected"] = wrong & is_linked
        masks["all reading errors corrected"] = wrong
        # Readings that were right but hesitant stay hesitant above. Here every reading is also
        # made confident, which isolates what the exact core does with correct input.
        masks["every reading correct and confident"] = np.ones(len(pred), bool)
        for label, fix_mask in masks.items():
            entry["conditions"][label] = replay(name, label, episodes, table, rules, key_gold, wrong, fix_mask, n_boot)
        out[name] = entry
    return out


def replay(name, label, episodes, table, rules, key_gold, wrong, fix_mask, n_boot) -> dict:
    """Run the episodes with the readings in `fix_mask` replaced by confident correct ones."""
    lg = table.logits.copy()
    fixed = np.where(fix_mask)[0]
    lg[fixed] = 0.0
    lg[fixed, key_gold[fixed]] = 8.0
    t2 = EdgeTable(table.version, table.msg_dim, table.keys, lg, table.msgs, table.tokens)
    s = summarise(run_engine(label, episodes, t2, rules), n_boot)
    print(f"C {name:8s} {label:34s} corrected {len(fixed):6d}  disposition {s['disposition_accuracy']['value']:.4f} "
          f"stale {s['stale_decision_rate']['value']:.4f} false READY {s['false_ready_rate']['value']:.4f} "
          f"missed READY {s['missed_ready_rate']['value']:.4f}", flush=True)
    return {"readings_corrected": int(len(fixed)),
            "reading_accuracy_after": float(1.0 - (wrong & ~fix_mask).mean()),
            **{k: s[k] for k in ("disposition_accuracy", "stale_decision_rate", "false_ready_rate", "missed_ready_rate",
                                 "status_accuracy")}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--results", default="runs/results")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--out", default="runs/results/headroom.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--skip-fit", action="store_true")
    a = ap.parse_args()
    R = Path(a.results)
    out = {"note": "Diagnostics; exploratory. Part C replays already-evaluated test episodes with some reading "
                   "errors corrected from the generator's gold labels."}
    earlier = read_json(a.out) if Path(a.out).exists() else {}
    out["errors"] = confusion(R)
    out["headroom"] = headroom(R / "work", Path(a.stage_b), a.atoms, a.n_boot)
    if a.skip_fit and "fit" in earlier:
        out["fit"] = earlier["fit"]
    atomic_write_json(a.out, out)
    if not a.skip_fit:
        out["fit"] = fit_check(a.edge, R)
        atomic_write_json(a.out, out)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
