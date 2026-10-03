"""What is worth asking, and when asking cannot matter.

Value of information (Howard, 1966) applied to a task:

    VOI(Q) = E_answers[ U(state after the answers) ] - U(state now) - cost(Q)

where Q is a set of predicates put to a person, answers are assumed truthful and decisive, and
U is the utility of the assistant's own policy: it proposes an action only when that action is
READY, and a proposed action is worth  p_success * value - (1 - p_success) * cost_of_failure.

Two known limits, both handled explicitly rather than hidden:

  * Asking one predicate at a time is myopic. If an action needs two open predicates, settling
    either alone changes nothing, so every single question has negative value although the pair
    has positive value. Sets of up to `max_set` predicates are therefore evaluated jointly.
  * Everything here is relative to the model's beliefs. A certificate says the *model's* proposal
    cannot change; it does not say the proposal is right.

Computations enumerate answers exactly. Nothing is sampled or approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Optional, Sequence

import numpy as np

from ease.engine import READY, Engine, disposition_of
from ease.logic import evaluate

YES = np.array([1.0, 0.0, 0.0])
NO = np.array([0.0, 1.0, 0.0])
OPEN = np.array([0.0, 0.0, 1.0])


def policy_utility(engine: Engine, leaf_status: dict, leaf_belief: dict) -> tuple[float, dict[str, str]]:
    total, disp = 0.0, {}
    for a in engine.schema.actions:
        ev = evaluate(engine.schema, a.requires, leaf_status, leaf_belief)
        d = disposition_of(ev.status, engine.cfg)
        disp[a.id] = d
        if d == READY:
            p = ev.belief * a.reliability + (1.0 - ev.belief) * a.leak
            total += p * a.value_success - (1.0 - p) * a.cost_failure
    return total, disp


@dataclass
class Question:
    predicates: tuple[str, ...]
    texts: tuple[str, ...]
    value: float  # expected utility gained, net of the cost of asking
    gross_gain: float
    cost: float
    ask_first: str
    could_change: tuple[str, ...]  # actions whose proposal changes under at least one answer

    def as_dict(self) -> dict:
        return {"predicates": list(self.predicates), "texts": list(self.texts), "value": self.value,
                "gross_gain": self.gross_gain, "cost": self.cost, "ask_first": self.ask_first,
                "could_change": list(self.could_change)}


def open_predicates(engine: Engine, min_open: float = 0.2) -> list[str]:
    """Predicates the model does not consider settled."""
    ls, _ = engine.leaf_state()
    return [p.id for p in engine.schema.predicates if ls[p.id][2] >= min_open]


def value_of_asking(engine: Engine, predicates: Sequence[str]) -> Question:
    ls, lb = engine.leaf_state()
    base, base_disp = policy_utility(engine, ls, lb)
    expected, changed = 0.0, set()
    for answers in product((True, False), repeat=len(predicates)):
        w = 1.0
        s2, b2 = dict(ls), dict(lb)
        for pid, yes in zip(predicates, answers):
            w *= lb[pid] if yes else 1.0 - lb[pid]
            s2[pid], b2[pid] = (YES, 1.0) if yes else (NO, 0.0)
        if w == 0.0:
            continue
        u, d = policy_utility(engine, s2, b2)
        expected += w * u
        changed.update(a for a in d if d[a] != base_disp[a])
    cost = sum(engine.schema.predicate(p).ask_cost for p in predicates)
    # ask first whatever most cheaply might make the remaining questions unnecessary:
    # lowest cost per unit of probability that the answer is "no"
    first = min(predicates, key=lambda p: (engine.schema.predicate(p).ask_cost / max(1e-9, 1.0 - lb[p]), p))
    return Question(tuple(predicates), tuple(engine.schema.predicate(p).text for p in predicates),
                    expected - base - cost, expected - base, cost, first, tuple(sorted(changed)))


def best_question(engine: Engine, max_set: int = 3, candidates: Optional[Sequence[str]] = None,
                  max_candidates: int = 12) -> Optional[Question]:
    """The set of at most `max_set` open predicates with the highest positive value, or None if
    nothing is worth its cost."""
    cands = list(candidates) if candidates is not None else open_predicates(engine)
    if len(cands) > max_candidates:
        ls, _ = engine.leaf_state()
        cands = sorted(cands, key=lambda p: -ls[p][2])[:max_candidates]
    best: Optional[Question] = None
    for k in range(1, max_set + 1):
        for combo in combinations(sorted(cands), k):
            q = value_of_asking(engine, combo)
            if q.value > 1e-12 and (best is None or q.value > best.value + 1e-12):
                best = q
    return best


@dataclass
class Certificate:
    action_id: str
    free: tuple[str, ...]
    stable: bool
    disposition: str
    settlements_checked: int
    counterexample: Optional[dict] = None
    note: str = "Certifies the model's proposal, not its correctness."

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def certify_stable(engine: Engine, action_id: str, free: Sequence[str], max_free: int = 10) -> Certificate:
    """Does the proposal for `action_id` stay the same however the predicates in `free` are settled?

    Each free predicate may turn out satisfied, violated, or stay open. All 3^k settlements are
    evaluated. If the disposition is the same in every one, no answer to those questions can
    change what is proposed, so asking them is unnecessary for this action.
    """
    a = engine._action(action_id)
    leaves = set(engine.schema.leaves_of(a.requires))
    free = tuple(f for f in dict.fromkeys(free) if f in leaves)
    current = engine.assessment(action_id).disposition
    if len(free) > max_free:
        return Certificate(action_id, free, False, current, 0,
                           note="Too many free predicates to enumerate; no certificate was attempted.")
    ls, lb = engine.leaf_state()
    checked = 0
    for combo in product((0, 1, 2), repeat=len(free)):
        s2, b2 = dict(ls), dict(lb)
        for pid, v in zip(free, combo):
            if v == 0:
                s2[pid], b2[pid] = YES, 1.0
            elif v == 1:
                s2[pid], b2[pid] = NO, 0.0
            else:
                s2[pid], b2[pid] = OPEN, float(engine.schema.predicate(pid).prior)
        ev = evaluate(engine.schema, a.requires, s2, b2)
        checked += 1
        d = disposition_of(ev.status, engine.cfg)
        if d != current:
            names = ("satisfied", "violated", "open")
            return Certificate(action_id, free, False, current, checked,
                               {"settlement": {p: names[v] for p, v in zip(free, combo)}, "disposition": d})
    return Certificate(action_id, free, True, current, checked)


def unnecessary_questions(engine: Engine) -> dict[str, list[str]]:
    """For each action, the open predicates under it that provably cannot change its proposal."""
    out: dict[str, list[str]] = {}
    opens = set(open_predicates(engine))
    for a in engine.schema.actions:
        leaves = [l for l in dict.fromkeys(engine.schema.leaves_of(a.requires)) if l in opens]
        out[a.id] = [l for l in leaves if certify_stable(engine, a.id, [l]).stable]
    return out
