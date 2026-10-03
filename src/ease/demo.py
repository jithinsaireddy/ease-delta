"""A client hand-off packet that stays current.

The scenario is fixed; what the system says about it is not. Every status printed below is
produced by the trained model reading the messages. Nothing is scripted to look right, so if the
model misreads a message the demo shows the misreading.

Each passage says who is speaking. The reader sees only the text of a record, never its metadata,
so a message that says "we approve" without saying who "we" is cannot establish that the client
approved. Whatever prepares passages for the ledger has to keep that attribution in the text.
"""

from __future__ import annotations

from typing import Optional

from ease.ledger import Event
from ease.schema import AND, NOT, Action, Gate, Predicate, TaskSchema

DAY = 86_400.0


def packet_schema() -> TaskSchema:
    return TaskSchema(
        "client-handoff",
        [
            Predicate("design_approved", "The client has approved the design that is being delivered.", prior=0.3,
                      ask_cost=0.2),
            Predicate("date_confirmed", "The delivery date has been confirmed by the client.", prior=0.5, ask_cost=0.2),
            Predicate("nda_signed", "The non-disclosure agreement has been signed by both parties.", prior=0.6,
                      ask_cost=0.5),
            Predicate("invoice_paid", "The deposit invoice has been paid.", prior=0.5, ask_cost=0.3),
        ],
        [
            Gate("packet_ready", AND, ("design_approved", "date_confirmed", "nda_signed")),
            Gate("deposit_outstanding", NOT, ("invoice_paid",)),
        ],
        [
            Action("send_packet", "Send the delivery packet to the client", "packet_ready",
                   value_success=10.0, cost_failure=25.0),
            Action("send_payment_reminder", "Send a reminder about the deposit invoice", "deposit_outstanding",
                   value_success=2.0, cost_failure=5.0),
        ],
    )


def scenario() -> list[tuple[str, Optional[Event]]]:
    c, t, s = ("client", 2), ("team", 1), ("system", 0)

    def m(rid, rev, text, src, day, **kw):
        return Event(rid, rev, text, source_id=src[0], authority=src[1], valid_from=day * DAY, **kw)

    return [
        ("The signed NDA is filed.",
         m("nda", 1, "Note from the project team: both parties have signed the non-disclosure agreement; the countersigned copy is attached.", t, 1,
           span="contracts/nda-countersigned.pdf")),
        ("The client approves the design.",
         m("approval", 1, "Email from the client: Thanks for sending this over. We approve the design as presented, please go ahead.", c, 2,
           span="email 14, paragraph 1")),
        ("The client confirms the date.",
         m("date", 1, "Email from the client: Confirming that delivery on 12 October works for us.", c, 3, span="email 15")),
        ("An unrelated message arrives.",
         m("lunch", 1, "Office notice: the kitchen will be closed on Friday for maintenance.", s, 3)),
        ("The deposit is paid.",
         m("payment", 1, "Note from accounts: payment has been received for the deposit invoice.", t, 4, span="ledger entry 2291")),
        ("The same approval email is delivered a second time.",
         m("approval", 1, "Email from the client: Thanks for sending this over. We approve the design as presented, please go ahead.", c, 2,
           span="email 14, paragraph 1")),
        ("The client withdraws the approval.",
         m("approval", 2, "Email from the client: We need to pause. After internal review we are withdrawing our "
                          "approval of the design until the colour palette is revised.", c, 5, span="email 19")),
        ("A late copy of the original approval arrives from an archive.",
         m("approval", 1, "Email from the client: Thanks for sending this over. We approve the design as presented, please go ahead.", c, 2,
           span="email 14, paragraph 1")),
        ("The delivery date moves.",
         m("date", 2, "Email from the client: Given the redesign, 12 October no longer works for delivery. We have "
                      "not settled on a new date yet.", c, 6,
           span="email 21")),
        ("The revised design is approved.",
         m("approval", 3, "Email from the client: The revised palette looks right. We approve the updated design for delivery.", c, 8,
           span="email 25")),
        ("A new date is confirmed.",
         m("date", 3, "Email from the client: We confirm 19 October as the delivery date.", c, 9, span="email 26")),
        ("The payment record is withdrawn: the transfer bounced.",
         Event("payment", 2, None, source_id="team", authority=1)),
    ]


def render(rt_state: dict, schema: TaskSchema) -> str:
    lines = []
    for p in schema.predicates:
        b = rt_state["beliefs"][p.id]
        pr = b["probabilities"]
        lines.append(f"    {p.id:16s} {b['status']:11s} "
                     f"(satisfied {pr['SATISFIED']:.2f}, violated {pr['VIOLATED']:.2f}, open {pr['UNRESOLVED']:.2f})"
                     + (f"  <- {', '.join(b['decisive'])}" if b["decisive"] else ""))
    for a in schema.actions:
        s = rt_state["assessments"][a.id]
        lines.append(f"  > {a.id:22s} {s['disposition']:10s} confidence {s['confidence']:.2f}, "
                     f"P(success) {s['p_success']:.2f}" + ("  [needs your approval]" if s["disposition"] == "READY" else ""))
    return "\n".join(lines)


def run(runtime, out=print, task_id: Optional[str] = None) -> dict:
    schema = packet_schema()
    if task_id:
        schema = TaskSchema.from_dict({**schema.to_dict(), "task_id": task_id})
    runtime.create_task(schema)
    tid = schema.task_id
    out(f"Task: {tid}. Endorsement threshold {runtime.tracker.tau:.2f}.")
    out(render(runtime.state(tid), schema))
    totals = {"events": 0, "edges_scored": 0, "unchanged": 0, "seconds": 0.0}
    for title, ev in scenario():
        rep = runtime.deliver(tid, ev)
        totals["events"] += 1
        totals["edges_scored"] += rep["edges_scored"]
        totals["seconds"] += rep["seconds"]
        totals["unchanged"] += int(not rep["ledger"]["changed"])
        out("")
        out(f"[{ev.record_id} rev {ev.revision}] {title}")
        out(f"    ledger: {rep['ledger']['outcome']}; pairs read: {rep['edges_scored']}; "
            f"nodes recomputed: {rep['propagation']['total_recomputed']}; {rep['seconds']*1000:.0f} ms")
        for a, old, new in rep["changed_actions"]:
            out(f"    {a}: {old} -> {new}")
        if not rep["changed_actions"]:
            out("    no proposal changed")
        out(render(runtime.state(tid), schema))
        q = runtime.question(tid)
        if q["question"]:
            out(f"    worth asking (value {q['question']['value']:.2f}): {q['question']['texts'][0]}")
        skip = {k: v for k, v in q["unnecessary"].items() if v}
        if skip:
            out(f"    cannot change the proposal, so not asked: {skip}")
    v = runtime.verify(tid)
    out("")
    out(f"Check: {v['computed_nodes']} cached nodes compared with a full rebuild; "
        f"{v['not_bitwise_equal']} differ (max difference {v['max_abs_difference']:.2e}).")
    out(f"Totals: {totals['events']} events, {totals['unchanged']} changed nothing, "
        f"{totals['edges_scored']} pairs read, {totals['seconds']*1000:.0f} ms.")
    return {"verify": v, "totals": totals, "state": runtime.state(tid)}
