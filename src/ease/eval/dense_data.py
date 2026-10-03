"""Inputs for the dense baseline, derived from the same episodes every other system uses."""

from __future__ import annotations

from typing import Optional, Sequence

from ease.eval.trace import event_of
from ease.ledger import Ledger
from ease.model.dense import render_context
from ease.schema import TaskSchema


def active_records(ledger: Ledger, now: Optional[float] = None) -> list[dict]:
    out = []
    for rid, p in ledger.active(now).items():
        if p.text:
            out.append({"record_id": rid, "text": p.text, "authority": int(p.authority),
                        "valid_from": float(p.valid_from) if p.valid_from is not None else float("-inf")})
    return out


def contexts_of(episodes: Sequence[dict], scored_only: bool = False) -> list[dict]:
    """One row per (step at which the ledger changed, predicate): the claim, the rendered state and
    the ground-truth status. Steps that leave the ledger unchanged reuse the previous answer and
    therefore contribute no row."""
    rows = []
    for ei, ep in enumerate(episodes):
        schema = TaskSchema.from_dict(ep["schema"])
        led = Ledger(task_id=ep["episode_id"])
        for step in ep["steps"]:
            receipt = led.deliver(event_of(step))
            if not receipt.changed:
                continue
            if scored_only and step["setup"]:
                continue
            ctx = render_context(active_records(led))
            for p in schema.predicates:
                rows.append({"episode": ei, "t": step["t"], "pid": p.id, "claim": p.text, "context": ctx,
                             "label": step["truth"]["pred"][p.id], "setup": step["setup"]})
        led.close()
    return rows
