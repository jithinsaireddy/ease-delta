"""Exact evaluation of declared requirements.

Two questions are asked of every requirement, and they are different questions:

  status   What does the current evidence establish?   SATISFIED / VIOLATED / UNRESOLVED
           (three-valued Kleene logic; "unresolved" is a real answer, not an error)
  belief   How likely is it that the requirement truly holds?   a probability
           (two-valued logic over the hidden truth; an unresolved predicate contributes its prior)

Status drives what the assistant proposes. Belief drives the predicted chance an action succeeds.

Leaf uncertainty comes from the learned reader. Everything above the leaves is arithmetic, and
under the stated assumption it is exact:

  A1  Given the evidence, distinct predicates are independent.

A formula that uses one predicate in several places would violate A1 if its occurrences were
multiplied as though independent. `evaluate` therefore conditions on each repeated predicate and
sums over its values, so reuse is handled exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np

from ease.schema import AND, NOT, OR, Gate, TaskSchema

SATISFIED, VIOLATED, UNRESOLVED = 0, 1, 2
STATUS_NAMES = ("SATISFIED", "VIOLATED", "UNRESOLVED")


# ---------------------------------------------------------------------------
# Deterministic three-valued logic (used for ground truth and for tests)
# ---------------------------------------------------------------------------


def kleene(op: str, vals: Sequence[int], k: int = 0) -> int:
    if op == NOT:
        return {SATISFIED: VIOLATED, VIOLATED: SATISFIED, UNRESOLVED: UNRESOLVED}[vals[0]]
    n_s = sum(1 for v in vals if v == SATISFIED)
    n_v = sum(1 for v in vals if v == VIOLATED)
    n = len(vals)
    if op == AND:
        k = n
    elif op == OR:
        k = 1
    if n_s >= k:
        return SATISFIED
    if n_v >= n - k + 1:  # too many failures for k successes to remain possible
        return VIOLATED
    return UNRESOLVED


def status_of(schema: TaskSchema, node_id: str, leaf_status: Mapping[str, int]) -> int:
    by_id = {g.id: g for g in schema.gates}
    if node_id not in by_id:
        return leaf_status[node_id]
    g = by_id[node_id]
    return kleene(g.op, [status_of(schema, c, leaf_status) for c in g.children], g.k)


def truth_of(schema: TaskSchema, node_id: str, leaf_truth: Mapping[str, bool]) -> bool:
    by_id = {g.id: g for g in schema.gates}
    if node_id not in by_id:
        return bool(leaf_truth[node_id])
    g = by_id[node_id]
    vals = [truth_of(schema, c, leaf_truth) for c in g.children]
    if g.op == NOT:
        return not vals[0]
    k = len(vals) if g.op == AND else 1 if g.op == OR else g.k
    return sum(vals) >= k


# ---------------------------------------------------------------------------
# Probabilistic evaluation
# ---------------------------------------------------------------------------


def _combine_status(op: str, dists: Sequence[np.ndarray], k: int) -> np.ndarray:
    """Distribution over (S, V, U) of a gate whose children are independent."""
    if op == NOT:
        d = dists[0]
        return np.array([d[VIOLATED], d[SATISFIED], d[UNRESOLVED]], dtype=np.float64)
    n = len(dists)
    if op == AND:
        k = n
    elif op == OR:
        k = 1
    # joint distribution of (#satisfied, #violated) by dynamic programming
    table = np.zeros((n + 1, n + 1), dtype=np.float64)
    table[0, 0] = 1.0
    for d in dists:
        nxt = np.zeros_like(table)
        nxt[1:, :] += table[:-1, :] * d[SATISFIED]
        nxt[:, 1:] += table[:, :-1] * d[VIOLATED]
        nxt += table * d[UNRESOLVED]
        table = nxt
    p_s = float(table[k:, :].sum())
    p_v = float(table[:k, n - k + 1 :].sum())
    return np.array([p_s, p_v, max(0.0, 1.0 - p_s - p_v)], dtype=np.float64)


def _combine_truth(op: str, ps: Sequence[float], k: int) -> float:
    if op == NOT:
        return 1.0 - ps[0]
    n = len(ps)
    if op == AND:
        k = n
    elif op == OR:
        k = 1
    counts = np.zeros(n + 1, dtype=np.float64)
    counts[0] = 1.0
    for p in ps:
        nxt = counts * (1.0 - p)
        nxt[1:] += counts[:-1] * p
        counts = nxt
    return float(counts[k:].sum())


def _repeated_leaves(schema: TaskSchema, node_id: str) -> list[str]:
    seen, rep = set(), []
    for leaf in schema.leaves_of(node_id):
        if leaf in seen and leaf not in rep:
            rep.append(leaf)
        seen.add(leaf)
    return rep


@dataclass(frozen=True)
class Evaluation:
    status: np.ndarray  # P(SATISFIED), P(VIOLATED), P(UNRESOLVED)
    belief: float  # P(requirement truly holds)
    exact: bool  # False only if repeated predicates were too many to condition on


def evaluate(
    schema: TaskSchema,
    node_id: str,
    leaf_status: Mapping[str, np.ndarray],
    leaf_belief: Mapping[str, float],
    max_conditioned: int = 8,
) -> Evaluation:
    by_id = {g.id: g for g in schema.gates}

    def st(n: str, ls) -> np.ndarray:
        if n not in by_id:
            return np.asarray(ls[n], dtype=np.float64)
        g: Gate = by_id[n]
        return _combine_status(g.op, [st(c, ls) for c in g.children], g.k)

    def tr(n: str, lb) -> float:
        if n not in by_id:
            return float(lb[n])
        g: Gate = by_id[n]
        return _combine_truth(g.op, [tr(c, lb) for c in g.children], g.k)

    rep = _repeated_leaves(schema, node_id)
    if not rep:
        return Evaluation(st(node_id, leaf_status), tr(node_id, leaf_belief), True)
    if len(rep) > max_conditioned:
        return Evaluation(st(node_id, leaf_status), tr(node_id, leaf_belief), False)

    status = np.zeros(3, dtype=np.float64)
    for combo in product(range(3), repeat=len(rep)):
        w = 1.0
        ls = dict(leaf_status)
        for leaf, v in zip(rep, combo):
            w *= float(leaf_status[leaf][v])
            ls[leaf] = np.eye(3)[v]
        if w > 0:
            status += w * st(node_id, ls)
    belief = 0.0
    for combo in product((0, 1), repeat=len(rep)):
        w = 1.0
        lb = dict(leaf_belief)
        for leaf, v in zip(rep, combo):
            w *= leaf_belief[leaf] if v else 1.0 - leaf_belief[leaf]
            lb[leaf] = float(v)
        if w > 0:
            belief += w * tr(node_id, lb)
    return Evaluation(status, float(belief), True)


# ---------------------------------------------------------------------------
# Intervals: what could this requirement become if some predicates were settled?
# ---------------------------------------------------------------------------


def belief_interval(
    schema: TaskSchema,
    node_id: str,
    leaf_belief: Mapping[str, float],
    free: Sequence[str],
    max_enumerate: int = 12,
) -> tuple[float, float, bool]:
    """Range of the requirement's belief as the predicates in `free` range over [0, 1].

    The belief is multilinear in the leaf beliefs (each leaf enters through conditioning), so its
    extremes over a box are attained at corners. Enumerating the corners is therefore exact.
    Returns (low, high, exact). With more than `max_enumerate` free predicates the corners are not
    enumerated and the trivial sound bound (0, 1) is returned with exact=False.
    """
    free = [f for f in dict.fromkeys(free) if f in set(schema.leaves_of(node_id))]
    if not free:
        b = evaluate(schema, node_id, {k: np.array([v, 1 - v, 0.0]) for k, v in leaf_belief.items()}, leaf_belief).belief
        return b, b, True
    if len(free) > max_enumerate:
        return 0.0, 1.0, False
    lo, hi = 1.0, 0.0
    dummy_status = {k: np.array([v, 1 - v, 0.0]) for k, v in leaf_belief.items()}
    for corner in product((0.0, 1.0), repeat=len(free)):
        lb = dict(leaf_belief)
        for f, v in zip(free, corner):
            lb[f] = v
        b = evaluate(schema, node_id, dummy_status, lb).belief
        lo, hi = min(lo, b), max(hi, b)
    # a belief is a probability; rounding in the arithmetic can put a corner a few ulps outside [0, 1]
    return min(max(lo, 0.0), 1.0), min(max(hi, 0.0), 1.0), True
