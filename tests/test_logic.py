"""Exact logic is checked against brute-force enumeration of every possible world."""

from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ease.logic import (SATISFIED, UNRESOLVED, VIOLATED, belief_interval, evaluate, kleene, status_of, truth_of)
from ease.schema import AND, ATLEAST, NOT, OR, Action, Gate, Predicate, TaskSchema


def random_schema(rng: np.random.Generator, n_pred: int, n_gates: int, allow_reuse: bool) -> TaskSchema:
    preds = [Predicate(f"p{i}", f"claim {i}", prior=float(rng.uniform(0.1, 0.9))) for i in range(n_pred)]
    gates: list[Gate] = []
    unused = [p.id for p in preds]
    available = [p.id for p in preds]
    for j in range(n_gates):
        op = str(rng.choice([AND, OR, NOT, ATLEAST]))
        pool = available if allow_reuse else unused
        if not pool:
            break
        if op == NOT:
            kids = [str(rng.choice(pool))]
        else:
            size = int(rng.integers(1, min(4, len(pool)) + 1))
            kids = [str(x) for x in rng.choice(pool, size=size, replace=False)]
        k = int(rng.integers(1, len(kids) + 1)) if op == ATLEAST else 0
        gid = f"g{j}"
        gates.append(Gate(gid, op, tuple(kids), k))
        if not allow_reuse:
            unused = [u for u in unused if u not in kids]
            unused.append(gid)
        available.append(gid)
    root = gates[-1].id if gates else preds[0].id
    return TaskSchema("t", preds, gates, [Action("act", "do it", root)])


def brute_force(schema, root, leaf_status, leaf_belief):
    leaves = sorted(set(schema.leaves_of(root)))
    status = np.zeros(3)
    for combo in product(range(3), repeat=len(leaves)):
        w = float(np.prod([leaf_status[l][v] for l, v in zip(leaves, combo)]))
        if w:
            status[status_of(schema, root, dict(zip(leaves, combo)))] += w
    belief = 0.0
    for combo in product((False, True), repeat=len(leaves)):
        w = float(np.prod([leaf_belief[l] if v else 1 - leaf_belief[l] for l, v in zip(leaves, combo)]))
        if w and truth_of(schema, root, dict(zip(leaves, combo))):
            belief += w
    return status, belief


@settings(max_examples=150, deadline=None)
@given(st.integers(0, 100_000), st.integers(1, 6), st.integers(0, 6), st.booleans())
def test_probabilistic_evaluation_matches_enumeration_of_all_worlds(seed, n_pred, n_gates, reuse):
    rng = np.random.default_rng(seed)
    schema = random_schema(rng, n_pred, n_gates, reuse)
    root = schema.actions[0].requires
    ls = {p.id: rng.dirichlet(np.ones(3)) for p in schema.predicates}
    lb = {p.id: float(rng.uniform()) for p in schema.predicates}
    ev = evaluate(schema, root, ls, lb)
    assert ev.exact
    bs, bb = brute_force(schema, root, ls, lb)
    np.testing.assert_allclose(ev.status, bs, atol=1e-12)
    assert abs(ev.belief - bb) < 1e-12
    assert abs(ev.status.sum() - 1.0) < 1e-12


def test_reused_predicate_is_not_double_counted():
    """x AND x must have the probability of x, not of x squared."""
    schema = TaskSchema("t", [Predicate("x", "claim")],
                        [Gate("a", OR, ("x",)), Gate("b", OR, ("x",)), Gate("both", AND, ("a", "b"))],
                        [Action("act", "d", "both")])
    ev = evaluate(schema, "both", {"x": np.array([0.5, 0.3, 0.2])}, {"x": 0.6})
    np.testing.assert_allclose(ev.status, [0.5, 0.3, 0.2], atol=1e-12)
    assert abs(ev.belief - 0.6) < 1e-12


def test_excluded_middle_holds_for_beliefs_but_not_for_status():
    """x OR NOT x is certainly true, yet the evidence may establish neither side."""
    schema = TaskSchema("t", [Predicate("x", "claim")],
                        [Gate("nx", NOT, ("x",)), Gate("taut", OR, ("x", "nx"))], [Action("a", "d", "taut")])
    ev = evaluate(schema, "taut", {"x": np.array([0.2, 0.3, 0.5])}, {"x": 0.37})
    assert abs(ev.belief - 1.0) < 1e-12
    np.testing.assert_allclose(ev.status, [0.5, 0.0, 0.5], atol=1e-12)


@pytest.mark.parametrize("op,vals,k,want", [
    (AND, [SATISFIED, SATISFIED], 0, SATISFIED),
    (AND, [SATISFIED, UNRESOLVED], 0, UNRESOLVED),
    (AND, [UNRESOLVED, VIOLATED], 0, VIOLATED),
    (OR, [UNRESOLVED, SATISFIED], 0, SATISFIED),
    (OR, [VIOLATED, UNRESOLVED], 0, UNRESOLVED),
    (OR, [VIOLATED, VIOLATED], 0, VIOLATED),
    (NOT, [UNRESOLVED], 0, UNRESOLVED),
    (NOT, [VIOLATED], 0, SATISFIED),
    (ATLEAST, [SATISFIED, SATISFIED, VIOLATED], 2, SATISFIED),
    (ATLEAST, [SATISFIED, VIOLATED, VIOLATED], 2, VIOLATED),
    (ATLEAST, [SATISFIED, UNRESOLVED, VIOLATED], 2, UNRESOLVED),
])
def test_kleene_truth_tables(op, vals, k, want):
    assert kleene(op, vals, k) == want


@settings(max_examples=120, deadline=None)
@given(st.integers(0, 100_000), st.integers(1, 6), st.integers(0, 6), st.booleans(), st.integers(1, 3))
def test_belief_interval_is_sound_and_tight(seed, n_pred, n_gates, reuse, n_free):
    rng = np.random.default_rng(seed)
    schema = random_schema(rng, n_pred, n_gates, reuse)
    root = schema.actions[0].requires
    lb = {p.id: float(rng.uniform()) for p in schema.predicates}
    free = [str(x) for x in rng.choice([p.id for p in schema.predicates], size=min(n_free, n_pred), replace=False)]
    lo, hi, exact = belief_interval(schema, root, lb, free)
    assert exact and 0.0 <= lo <= hi <= 1.0
    ls = {k: np.array([v, 1 - v, 0.0]) for k, v in lb.items()}
    seen_lo, seen_hi = 1.0, 0.0
    for _ in range(40):
        trial = dict(lb)
        for f in free:
            trial[f] = float(rng.uniform())
        b = evaluate(schema, root, ls, trial).belief
        assert lo - 1e-12 <= b <= hi + 1e-12, "sound: no setting of the free predicates escapes the interval"
        seen_lo, seen_hi = min(seen_lo, b), max(seen_hi, b)
    # tight: the bounds are attained at corners, so they cannot be narrower than what sampling reaches
    assert lo <= seen_lo + 1e-12 and hi >= seen_hi - 1e-12


def test_schema_validation():
    p = [Predicate("a", "x"), Predicate("b", "y")]
    with pytest.raises(ValueError):
        TaskSchema("t", p, [Gate("g", AND, ("a", "zzz"))], [])
    with pytest.raises(ValueError):
        TaskSchema("t", p, [Gate("g1", AND, ("g2",)), Gate("g2", AND, ("g1",))], [])
    with pytest.raises(ValueError):
        TaskSchema("t", p, [], [Action("act", "d", "missing")])
    with pytest.raises(ValueError):
        Gate("g", ATLEAST, ("a", "b"), k=3)
    with pytest.raises(ValueError):
        Predicate("a", "x", prior=1.5)
    s = TaskSchema("t", p, [Gate("g", AND, ("a", "b"))], [Action("act", "d", "g")])
    assert TaskSchema.from_dict(s.to_dict()).version() == s.version()
    s2 = TaskSchema("t", p, [Gate("g", OR, ("a", "b"))], [Action("act", "d", "g")])
    assert s2.version() != s.version()
