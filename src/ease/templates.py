"""Task templates: ordinary workflows written as requirements, gates and actions.

A template is a function of a few parameters (a client's name, a list of people) that returns a
task definition. The texts follow the rule the reader needs: say who and what ("The client has
approved the final artwork"), never "approved" alone. Dates and amounts are not compared by the
reader; keep requirements about whether something was confirmed, not when.

    from ease.templates import TEMPLATES, render
    schema = render("client-onboarding", "acme-onboarding", {"client": "Acme"})
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ease.schema import AND, ATLEAST, OR, TaskSchema

_ID = re.compile(r"[^a-z0-9]+")


def _slug(s: str) -> str:
    return _ID.sub("_", s.lower()).strip("_") or "x"


def _pred(pid: str, text: str, prior: float = 0.3, ask_cost: float = 0.3) -> dict:
    return {"id": pid, "text": text, "prior": prior, "ask_cost": ask_cost}


def _gate(gid: str, op: str, children: list[str], k: int = 0) -> dict:
    return {"id": gid, "op": op, "children": children, "k": k}


def _action(aid: str, description: str, requires: str, value: float = 10.0, cost: float = 25.0) -> dict:
    return {"id": aid, "description": description, "requires": requires, "value_success": value, "cost_failure": cost}


@dataclass
class Template:
    id: str
    name: str
    description: str
    params: dict = field(default_factory=dict)  # name -> {"type": "text" | "list" | "int", "default": ..., "help": ...}
    build: Callable[[str, dict], dict] = None

    def describe(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description, "params": self.params}


def _client(params: dict) -> str:
    c = str(params.get("client", "")).strip()
    return c if c else "the client"


def client_handoff(task_id: str, p: dict) -> dict:
    c = _client(p)
    return {"task_id": task_id, "predicates": [
        _pred("design_approved", f"{c} has approved the design that is being delivered."),
        _pred("date_confirmed", f"{c} has confirmed the delivery date.", 0.5),
        _pred("nda_signed", "The non-disclosure agreement has been signed by both parties.", 0.6, 0.5),
        _pred("invoice_paid", f"{c} has paid the invoice.", 0.4, 0.5)],
        "gates": [_gate("packet_ready", AND, ["design_approved", "date_confirmed", "nda_signed"])],
        "actions": [_action("send_packet", f"Send the delivery packet to {c}", "packet_ready"),
                    _action("send_payment_reminder", f"Remind {c} about the unpaid invoice", "invoice_paid_missing", 2.0, 5.0)],
        "_extra_gates": [_gate("invoice_paid_missing", "NOT", ["invoice_paid"])]}


def client_onboarding(task_id: str, p: dict) -> dict:
    c = _client(p)
    return {"task_id": task_id, "predicates": [
        _pred("brief_received", f"{c} has sent the project brief."),
        _pred("files_supplied", f"{c} has supplied the files that were requested.", 0.3, 0.3),
        _pred("access_confirmed", f"{c} has given access to the website or systems needed for the work.", 0.3, 0.3),
        _pred("kickoff_agreed", f"{c} has agreed a date for the kickoff meeting.", 0.4, 0.2)],
        "gates": [_gate("ready_to_start", AND, ["brief_received", "files_supplied", "access_confirmed"])],
        "actions": [_action("start_work", f"Start the work for {c}", "ready_to_start"),
                    _action("hold_kickoff", f"Hold the kickoff meeting with {c}", "kickoff_agreed", 5.0, 5.0)]}


def campaign_launch(task_id: str, p: dict) -> dict:
    c = _client(p)
    return {"task_id": task_id, "predicates": [
        _pred("copy_approved", f"{c} has approved the final advertisement copy."),
        _pred("artwork_approved", f"{c} has approved the final artwork."),
        _pred("materials_received", "All materials required for the launch have been received.", 0.3, 0.3),
        _pred("legal_cleared", "The legal or compliance review has cleared the campaign.", 0.5, 0.5)],
        "gates": [_gate("launch_ready", AND, ["copy_approved", "artwork_approved", "materials_received", "legal_cleared"])],
        "actions": [_action("launch", "Launch the campaign", "launch_ready", 20.0, 60.0)]}


def event_coordination(task_id: str, p: dict) -> dict:
    speakers = [str(s).strip() for s in (p.get("speakers") or []) if str(s).strip()] or ["the keynote speaker"]
    k = int(p.get("min_speakers") or len(speakers))
    k = max(1, min(k, len(speakers)))
    preds = [_pred("venue_confirmed", "The venue has confirmed the booking in writing.", 0.4, 0.3),
             _pred("equipment_confirmed", "The audio-visual equipment has been confirmed by the supplier.", 0.4, 0.3),
             _pred("schedule_published", "The final schedule has been sent to all speakers.", 0.3, 0.2)]
    sp = []
    for s in speakers:
        pid = f"speaker_{_slug(s)}"
        preds.append(_pred(pid, f"{s} has confirmed they will speak at the event.", 0.5, 0.2))
        sp.append(pid)
    return {"task_id": task_id, "predicates": preds,
            "gates": [_gate("enough_speakers", ATLEAST, sp, k),
                      _gate("announce_ready", AND, ["venue_confirmed", "enough_speakers"]),
                      _gate("run_ready", AND, ["venue_confirmed", "enough_speakers", "equipment_confirmed", "schedule_published"])],
            "actions": [_action("announce", "Announce the event publicly", "announce_ready", 10.0, 30.0),
                        _action("run_event", "Run the event", "run_ready", 30.0, 80.0)]}


def support_followup(task_id: str, p: dict) -> dict:
    c = str(p.get("customer", "")).strip() or "the customer"
    return {"task_id": task_id, "predicates": [
        _pred("fix_delivered", f"The fix has been delivered to {c}.", 0.4, 0.3),
        _pred("replacement_received", f"{c} has received the replacement.", 0.3, 0.3),
        _pred("customer_confirmed", f"{c} has confirmed that the problem is resolved.", 0.3, 0.2)],
        "gates": [_gate("remedy", OR, ["fix_delivered", "replacement_received"]),
                  _gate("resolved", AND, ["remedy", "customer_confirmed"])],
        "actions": [_action("close_ticket", "Close the ticket", "resolved", 5.0, 20.0),
                    _action("ask_customer", f"Ask {c} whether the problem is resolved", "remedy", 2.0, 2.0)]}


def software_release(task_id: str, p: dict) -> dict:
    v = str(p.get("version", "")).strip() or "the release build"
    return {"task_id": task_id, "predicates": [
        _pred("tests_passed", f"The test report for {v} shows that all required tests passed.", 0.4, 0.3),
        _pred("notes_complete", f"The release notes for {v} are complete.", 0.4, 0.2),
        _pred("qa_signoff", f"QA has signed off {v}.", 0.3, 0.3),
        _pred("product_signoff", f"The product owner has signed off {v}.", 0.3, 0.3),
        _pred("rollback_ready", f"A rollback plan for {v} has been written.", 0.5, 0.3)],
        "gates": [_gate("signoffs", AND, ["qa_signoff", "product_signoff"]),
                  _gate("ship_ready", AND, ["tests_passed", "notes_complete", "signoffs", "rollback_ready"])],
        "actions": [_action("ship", f"Ship {v}", "ship_ready", 20.0, 100.0)]}


def group_trip(task_id: str, p: dict) -> dict:
    people = [str(s).strip() for s in (p.get("participants") or []) if str(s).strip()] or ["everyone"]
    k = int(p.get("min_participants") or len(people))
    k = max(1, min(k, len(people)))
    preds = [_pred("accommodation_booked", "The accommodation has been booked and confirmed.", 0.3, 0.3),
             _pred("transport_booked", "The transport has been booked and confirmed.", 0.3, 0.3)]
    conf, contrib = [], []
    for s in people:
        a, b = f"confirmed_{_slug(s)}", f"contribution_{_slug(s)}"
        preds.append(_pred(a, f"{s} has confirmed they are coming.", 0.5, 0.2))
        preds.append(_pred(b, f"{s} has paid their share.", 0.4, 0.2))
        conf.append(a)
        contrib.append(b)
    return {"task_id": task_id, "predicates": preds,
            "gates": [_gate("enough_people", ATLEAST, conf, k),
                      _gate("all_paid", AND, contrib),
                      _gate("book_ready", AND, ["enough_people"]),
                      _gate("done", AND, ["enough_people", "all_paid", "accommodation_booked", "transport_booked"])],
            "actions": [_action("book", "Make the bookings", "book_ready", 10.0, 30.0),
                        _action("send_itinerary", "Send the itinerary to everyone", "done", 5.0, 5.0)]}


TEMPLATES: dict[str, Template] = {t.id: t for t in [
    Template("client-handoff", "Client hand-off",
             "Deliver a packet once the design is approved, the date confirmed and the NDA signed; chase an unpaid invoice.",
             {"client": {"type": "text", "default": "the client", "help": "how the client is named in messages"}}, client_handoff),
    Template("client-onboarding", "Client onboarding",
             "Start work once the brief, the files and access have arrived; hold the kickoff once a date is agreed.",
             {"client": {"type": "text", "default": "the client"}}, client_onboarding),
    Template("campaign-launch", "Marketing campaign launch",
             "Launch once copy and artwork are approved, materials are in and the compliance review is clear.",
             {"client": {"type": "text", "default": "the client"}}, campaign_launch),
    Template("event-coordination", "Event coordination",
             "Announce when the venue and enough speakers are confirmed; run when equipment and schedule are too.",
             {"speakers": {"type": "list", "default": [], "help": "names of the speakers"},
              "min_speakers": {"type": "int", "default": 0, "help": "how many must confirm; all if 0"}}, event_coordination),
    Template("support-followup", "Customer-support follow-up",
             "Close the ticket once a fix or replacement has reached the customer and the customer confirms.",
             {"customer": {"type": "text", "default": "the customer"}}, support_followup),
    Template("software-release", "Software release",
             "Ship once the test report, release notes, both sign-offs and a rollback plan are in.",
             {"version": {"type": "text", "default": "the release build"}}, software_release),
    Template("group-trip", "Group trip or shared project",
             "Book once enough people have confirmed; send the itinerary once everyone has paid and bookings are made.",
             {"participants": {"type": "list", "default": []},
              "min_participants": {"type": "int", "default": 0, "help": "how many must confirm; all if 0"}}, group_trip),
]}


def render(template_id: str, task_id: str, params: dict | None = None) -> dict:
    """The task definition for a template, validated as a schema."""
    if template_id not in TEMPLATES:
        raise KeyError(f"no template {template_id!r}; known: {sorted(TEMPLATES)}")
    d = TEMPLATES[template_id].build(task_id, params or {})
    d["gates"] = d.get("gates", []) + d.pop("_extra_gates", [])
    TaskSchema.from_dict(d)  # raises on an invalid definition
    return d
