"""The hand-off scenario, scored against what each message actually says.

One scenario of twelve events, written by hand before the model was trained. It is an
illustration of behaviour on email-like text, which the training data does not contain. It is not
a test set: one scenario cannot estimate an error rate.

    python scripts/demo_trace.py --edge runs/stage_a/final --stage-b runs/stage_b --out runs/results/demo_trace.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


from ease import demo
from ease.aggregate import Calibration, RuleAggregator, load_aggregator
from ease.engine import Engine, EngineConfig
from ease.scorer import ModelScorer
from ease.util import atomic_write_json, read_json

S, V, U = "SATISFIED", "VIOLATED", "UNRESOLVED"

# What the messages establish after each event, by the declared policy. Written from the scenario's
# text, independently of any model output.
TRUTH = [
    {"nda_signed": S, "design_approved": U, "date_confirmed": U, "invoice_paid": U},
    {"nda_signed": S, "design_approved": S, "date_confirmed": U, "invoice_paid": U},
    {"nda_signed": S, "design_approved": S, "date_confirmed": S, "invoice_paid": U},
    {"nda_signed": S, "design_approved": S, "date_confirmed": S, "invoice_paid": U},
    {"nda_signed": S, "design_approved": S, "date_confirmed": S, "invoice_paid": S},
    {"nda_signed": S, "design_approved": S, "date_confirmed": S, "invoice_paid": S},
    {"nda_signed": S, "design_approved": V, "date_confirmed": S, "invoice_paid": S},
    {"nda_signed": S, "design_approved": V, "date_confirmed": S, "invoice_paid": S},
    {"nda_signed": S, "design_approved": V, "date_confirmed": V, "invoice_paid": S},
    {"nda_signed": S, "design_approved": S, "date_confirmed": V, "invoice_paid": S},
    {"nda_signed": S, "design_approved": S, "date_confirmed": S, "invoice_paid": S},
    {"nda_signed": S, "design_approved": S, "date_confirmed": S, "invoice_paid": U},
]


def disposition(req: list[str], statuses: dict, negate: bool = False) -> str:
    vals = [statuses[r] for r in req]
    if negate:
        v = vals[0]
        return {S: "BLOCKED", V: "READY", U: "NEEDS_INFO"}[v]
    if all(v == S for v in vals):
        return "READY"
    if any(v == V for v in vals):
        return "BLOCKED"
    return "NEEDS_INFO"


def run(name, scorer, agg, threshold):
    schema = demo.packet_schema()
    eng = Engine(schema, scorer, agg, config=EngineConfig(ready_threshold=threshold))
    rows = []
    for (title, ev), truth in zip(demo.scenario(), TRUTH):
        eng.deliver(ev)
        preds = {p.id: eng.belief(p.id) for p in schema.predicates}
        want_actions = {"send_packet": disposition(["design_approved", "date_confirmed", "nda_signed"], truth),
                        "send_payment_reminder": disposition(["invoice_paid"], truth, negate=True)}
        got_actions = {a: eng.assessment(a).disposition for a in want_actions}
        rows.append({"event": title,
                     "predicates": {pid: {"model": b["status"], "truth": truth[pid],
                                          "p": {k: round(v, 3) for k, v in b["probabilities"].items()}}
                                    for pid, b in preds.items()},
                     "actions": {a: {"model": got_actions[a], "truth": want_actions[a]} for a in want_actions}})
    n_p = sum(len(r["predicates"]) for r in rows)
    ok_p = sum(v["model"] == v["truth"] for r in rows for v in r["predicates"].values())
    n_a = sum(len(r["actions"]) for r in rows)
    ok_a = sum(v["model"] == v["truth"] for r in rows for v in r["actions"].values())
    false_ready = sum(v["model"] == "READY" and v["truth"] != "READY" for r in rows for v in r["actions"].values())
    return {"system": name, "threshold": threshold, "predicate_status_correct": f"{ok_p}/{n_p}",
            "action_disposition_correct": f"{ok_a}/{n_a}", "false_ready": false_ready, "events": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--out", default="runs/results/demo_trace.json")
    a = ap.parse_args()
    scorer = ModelScorer(a.edge, canonical=True)
    sb = read_json(Path(a.stage_b) / "summary.json")
    cal = read_json(Path(a.stage_b) / "rule_calibration.json")
    systems = {"E (refined, seed 1)": load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"]),
               "B1-soft (calibrated rules)": RuleAggregator(Calibration(cal["temperature"], tuple(cal["bias"]))),
               "B1-hard (top label per record)": RuleAggregator(hard=True)}
    out = {"note": "One hand-written scenario; an illustration, not an estimate of error rates.", "runs": []}
    for thr in (0.5, 0.9):
        for name, agg in systems.items():
            r = run(name, scorer, agg, thr)
            out["runs"].append(r)
            print(f"threshold {thr}: {name:32s} predicates {r['predicate_status_correct']:>6s}  "
                  f"actions {r['action_disposition_correct']:>6s}  false READY {r['false_ready']}", flush=True)
    atomic_write_json(a.out, out)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
