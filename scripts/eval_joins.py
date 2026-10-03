"""X1 (exploratory, not pre-registered): requirements that need two records read together.

    claim     "The client has approved the latest version of the logo."
    record A  "The client approved version 4 of the logo."
    record B  "The latest version of the logo is version 5."

Neither record settles the claim alone. Together they do: satisfied if the versions match,
violated if they differ. Reading each record separately cannot see that.

The question is what each way of reading does instead. Leaving the claim open is safe: the person
is asked. Calling it satisfied when the versions differ is not.

Wording is templated, so this measures behaviour on one clean pattern, not on real documents.

    python scripts/eval_joins.py --edge runs/stage_a/final --stage-b runs/stage_b --dense runs/dense/final
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from pathlib import Path

import numpy as np

from ease.aggregate import EdgeView, load_aggregator
from ease.model.dense import render_context
from ease.util import atomic_write_json, get_device, read_json

THINGS = ["logo", "brochure", "contract", "floor plan", "press release", "price list", "slide deck", "user manual",
          "packaging design", "website mock-up"]
CLAIMS = ["The client has approved the latest version of the {t}.",
          "The most recent version of the {t} has the client's approval.",
          "The current version of the {t} is approved by the client."]
A_TEXT = ["The client approved version {k} of the {t}.",
          "We have the client's sign-off on version {k} of the {t}.",
          "Client confirmed by email that version {k} of the {t} is approved."]
B_TEXT = ["The latest version of the {t} is version {j}.",
          "Version {j} is now the most recent version of the {t}.",
          "The {t} was updated; the current version is version {j}."]
STATUS = ("SATISFIED", "VIOLATED", "UNRESOLVED", "CONFLICT")


def cases(n: int, seed: int):
    rng = random.Random(seed)
    combos = list(itertools.product(THINGS, range(3), range(3), range(3)))
    rng.shuffle(combos)
    out = []
    for t, ci, ai, bi in combos:
        k = rng.randint(2, 9)
        for kind in ("match", "differ", "only_approval", "only_latest"):
            j = k if kind == "match" else rng.choice([x for x in range(2, 10) if x != k])
            recs = []
            if kind != "only_latest":
                recs.append({"record_id": "approval", "text": A_TEXT[ai].format(k=k, t=t), "authority": 2, "valid_from": 1.0})
            if kind != "only_approval":
                recs.append({"record_id": "latest", "text": B_TEXT[bi].format(j=j, t=t), "authority": 1, "valid_from": 2.0})
            truth = {"match": "SATISFIED", "differ": "VIOLATED"}.get(kind, "UNRESOLVED")
            out.append({"kind": kind, "claim": CLAIMS[ci].format(t=t), "records": recs, "truth": truth})
        if len(out) >= n:
            break
    return out[:n]


def tabulate(rows, preds):
    table = {}
    for kind in ("match", "differ", "only_approval", "only_latest"):
        ix = [i for i, r in enumerate(rows) if r["kind"] == kind]
        got = [preds[i] for i in ix]
        table[kind] = {"n": len(ix), "truth": rows[ix[0]]["truth"],
                       **{s: float(np.mean([g == s for g in got])) for s in STATUS},
                       "correct": float(np.mean([g == rows[i]["truth"] for g, i in zip(got, ix)]))}
    unsafe = [preds[i] == "SATISFIED" for i, r in enumerate(rows) if r["truth"] != "SATISFIED"]
    table["unsafe_rate"] = float(np.mean(unsafe))
    table["unsafe_rate_when_versions_differ"] = table["differ"]["SATISFIED"]
    table["overall_correct"] = float(np.mean([p == r["truth"] for p, r in zip(preds, rows)]))
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--dense", default="runs/dense/final")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", default="runs/results/joins.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from ease.model.dense import DenseModel, predict_contexts
    from ease.scorer import ModelScorer

    rows = cases(a.n, 0)
    device = get_device()
    scorer = ModelScorer(a.edge, batch_size=128)
    sb = read_json(Path(a.stage_b) / "summary.json")
    agg = load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"])

    # separate reading: each record against the claim, combined by the policy
    flat = [(r["claim"], rec["text"]) for r in rows for rec in r["records"]]
    got = iter(scorer.score(flat))
    separate = []
    for r in rows:
        edges = []
        for rec in r["records"]:
            lg, msg = next(got)
            edges.append(EdgeView(rec["record_id"], lg, msg, rec["authority"], rec["valid_from"]))
        separate.append(STATUS[int(np.argmax(agg.aggregate(edges, 0.5).status))])

    # joint reading with the same edge model: the records concatenated into one passage
    joint_pairs = [(r["claim"], " ".join(rec["text"] for rec in r["records"])) for r in rows]
    joint = []
    for (lg, msg) in scorer.score(joint_pairs):
        joint.append(STATUS[int(np.argmax(agg.aggregate([EdgeView("joint", lg, msg, 2, 2.0)], 0.5).status))])

    out = {"experiment": "X1 (exploratory, not pre-registered)", "n": len(rows),
           "wording": "templated; 10 subjects x 3 claim, 3 approval and 3 latest-version phrasings",
           "systems": {"separate (E)": tabulate(rows, separate), "joint, same edge model": tabulate(rows, joint)}}

    if Path(a.dense, "model.safetensors").exists():
        dm = DenseModel.load(a.dense, device).eval()
        tok = AutoTokenizer.from_pretrained(a.dense)
        lg, _ = predict_contexts(dm, tok, [r["claim"] for r in rows], [render_context(r["records"]) for r in rows],
                                 device, 1024)
        out["systems"]["dense reader (B2)"] = tabulate(rows, [STATUS[int(i)] for i in lg.argmax(1)])

    out["examples"] = [{"kind": r["kind"], "truth": r["truth"], "claim": r["claim"],
                        "records": [x["text"] for x in r["records"]], "separate": s, "joint": j}
                       for r, s, j in list(zip(rows, separate, joint))[:8]]
    atomic_write_json(a.out, out)
    for k, t in out["systems"].items():
        print(f"{k:24s} correct={t['overall_correct']:.3f} unsafe={t['unsafe_rate']:.3f} "
              f"| versions differ -> " + ", ".join(f"{s} {t['differ'][s]:.2f}" for s in STATUS[:3])
              + " | versions match -> " + ", ".join(f"{s} {t['match'][s]:.2f}" for s in STATUS[:3]), flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
