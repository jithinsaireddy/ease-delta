"""Incremental execution equals full recomputation, on random graphs and random edit sequences."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ease.graph import CycleError, IncrementalGraph, values_equal


def tanh_fn(weights):
    def fn(ids, parent_values):
        out = []
        for i, pv in zip(ids, parent_values):
            w, b = weights[i]
            out.append(float(np.tanh(float(np.dot(w[: len(pv)], np.asarray(pv, dtype=np.float64))) + b)))
        return out
    return fn


def random_dag(rng, n_inputs, n_computed, max_parents):
    weights = {}
    g = IncrementalGraph(early_cutoff=True)
    g.register("tanh", tanh_fn(weights))
    ids = []
    for i in range(n_inputs):
        g.add_input(f"x{i}", float(rng.standard_normal()))
        ids.append(f"x{i}")
    for j in range(n_computed):
        k = int(rng.integers(1, max_parents + 1))
        parents = list(rng.choice(ids, size=min(k, len(ids)), replace=False))
        nid = f"h{j}"
        weights[nid] = (rng.standard_normal(max_parents), float(rng.standard_normal()))
        g.add_node(nid, "tanh", parents)
        ids.append(nid)
    return g, weights


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10_000), st.integers(2, 12), st.integers(1, 40), st.integers(1, 4), st.integers(1, 25))
def test_T2_incremental_equals_full_recompute(seed, n_in, n_comp, max_par, n_edits):
    rng = np.random.default_rng(seed)
    g, _ = random_dag(rng, n_in, n_comp, max_par)
    g.propagate()
    assert g.verify()["not_bitwise_equal"] == 0
    for _ in range(n_edits):
        target = f"x{int(rng.integers(0, n_in))}"
        # a third of edits re-send the current value: they must schedule nothing
        value = g.nodes[target].value if rng.random() < 0.33 else float(rng.standard_normal())
        changed = g.set_input(target, value)
        reachable = g.descendants(target)
        rep = g.propagate()
        if not changed:
            assert rep.total_recomputed == 0
        assert set(rep.recomputed_ids) <= reachable, "T1: only descendants of the change may be recomputed"
        v = g.verify()
        assert v["not_bitwise_equal"] == 0 and v["max_abs_difference"] == 0.0


def test_T1_unrelated_branch_is_not_touched():
    calls = []

    def f(ids, pv):
        calls.extend(ids)
        return [sum(p) for p in pv]

    g = IncrementalGraph({"sum": f})
    g.add_input("approval", 1.0)
    g.add_input("doc_check", 5.0)
    g.add_node("approval_branch", "sum", ["approval"])
    g.add_node("doc_branch", "sum", ["doc_check"])
    g.add_node("packet", "sum", ["approval_branch", "doc_branch"])
    g.propagate()
    calls.clear()
    g.set_input("approval", 0.0)
    g.propagate()
    assert sorted(calls) == ["approval_branch", "packet"]
    assert g.value("packet") == 5.0


def test_early_cutoff_stops_at_unchanged_value_and_stays_exact():
    calls = []

    def sign(ids, pv):
        calls.extend(ids)
        return [1.0 if p[0] > 0 else -1.0 for p in pv]

    def double(ids, pv):
        calls.extend(ids)
        return [2.0 * p[0] for p in pv]

    for cutoff in (True, False):
        g = IncrementalGraph({"sign": sign, "double": double}, early_cutoff=cutoff)
        g.add_input("x", 0.4)
        g.add_node("s", "sign", ["x"])
        g.add_node("d1", "double", ["s"])
        g.add_node("d2", "double", ["d1"])
        g.propagate()
        calls.clear()
        g.set_input("x", 0.9)  # sign unchanged
        rep = g.propagate()
        assert calls == (["s"] if cutoff else ["s", "d1", "d2"])
        assert rep.unchanged_after_recompute["sign"] == 1
        assert g.verify()["not_bitwise_equal"] == 0
        calls.clear()
        g.set_input("x", -0.9)  # sign flips: everything downstream must follow
        g.propagate()
        assert calls == ["s", "d1", "d2"] and g.value("d2") == -4.0


def test_weights_version_is_an_input_and_forces_recomputation():
    """A model revision is a change to an input node, so everything computed by that model is redone
    and nothing computed by other models is."""
    calls = []
    state = {"scale": 1.0}

    def edge(ids, pv):
        calls.extend(ids)
        return [state["scale"] * p[1] for p in pv]

    def other(ids, pv):
        calls.extend(ids)
        return [p[0] + 1 for p in pv]

    g = IncrementalGraph({"edge": edge, "other": other})
    g.add_input("weights:edge", "v1")
    g.add_input("a", 2.0); g.add_input("b", 3.0); g.add_input("c", 4.0)
    g.add_node("ea", "edge", ["weights:edge", "a"])
    g.add_node("eb", "edge", ["weights:edge", "b"])
    g.add_node("oc", "other", ["c"])
    g.propagate()
    calls.clear()
    state["scale"] = 10.0
    g.set_input("weights:edge", "v2")
    g.propagate()
    assert sorted(calls) == ["ea", "eb"]
    assert g.value("ea") == 20.0 and g.value("oc") == 5.0
    assert g.verify()["not_bitwise_equal"] == 0


def test_naive_cache_without_dependency_tracking_is_wrong():
    """Counterexample: reusing an output because 'the question is the same' returns a stale winner."""
    def score(ids, pv):
        return [p[0] - p[1] for p in pv]

    g = IncrementalGraph({"score": score})
    g.add_input("evidence_for_A", 0.9); g.add_input("evidence_for_B", 0.2)
    g.add_node("margin", "score", ["evidence_for_A", "evidence_for_B"])
    g.propagate()
    naive_cached_winner = "A" if g.value("margin") > 0 else "B"
    g.set_input("evidence_for_A", 0.1)
    g.propagate()
    true_winner = "A" if g.value("margin") > 0 else "B"
    assert naive_cached_winner == "A" and true_winner == "B"


def test_topology_change_adds_dependency_and_repairs_levels():
    def s(ids, pv):
        return [float(sum(p)) for p in pv]

    g = IncrementalGraph({"s": s})
    g.add_input("x", 1.0); g.add_input("y", 10.0)
    g.add_node("a", "s", ["x"])
    g.add_node("b", "s", ["a"])
    g.add_node("c", "s", ["y"])
    g.propagate()
    assert g.value("c") == 10.0
    g.set_parents("c", ["y", "b"])  # c now also depends on the x-branch
    g.propagate()
    assert g.value("c") == 11.0 and g.nodes["c"].level == 3
    g.set_input("x", 2.0)
    rep = g.propagate()
    assert "c" in rep.recomputed_ids and g.value("c") == 12.0
    assert g.verify()["not_bitwise_equal"] == 0


def test_cycles_are_refused():
    def s(ids, pv):
        return [float(sum(p)) for p in pv]

    g = IncrementalGraph({"s": s})
    g.add_input("x", 1.0)
    g.add_node("a", "s", ["x"])
    g.add_node("b", "s", ["a"])
    with pytest.raises(CycleError):
        g.set_parents("a", ["x", "b"])


def test_stale_values_cannot_be_read():
    def s(ids, pv):
        return [float(sum(p)) for p in pv]

    g = IncrementalGraph({"s": s})
    g.add_input("x", 1.0)
    g.add_node("a", "s", ["x"])
    with pytest.raises(RuntimeError):
        g.value("a")
    g.propagate()
    g.set_input("x", 2.0)
    with pytest.raises(RuntimeError):
        g.value("a")
    g.propagate()
    assert g.value("a") == 2.0


def test_batch_function_receives_all_dirty_nodes_of_a_level_at_once():
    sizes = []

    def f(ids, pv):
        sizes.append(len(ids))
        return [float(sum(p)) for p in pv]

    g = IncrementalGraph({"f": f})
    g.add_input("w", 1.0)
    for i in range(20):
        g.add_input(f"x{i}", float(i))
        g.add_node(f"e{i}", "f", ["w", f"x{i}"])
    g.propagate()
    assert sizes == [20]
    sizes.clear()
    g.set_input("w", 2.0)
    assert g.propagate().batches == 1 and sizes == [20]


def test_values_equal_is_exact():
    a = np.array([1.0, 2.0], dtype=np.float32)
    assert values_equal(a, a.copy())
    assert not values_equal(a, a + np.float32(1e-7) * 8)
    assert not values_equal(a, a.astype(np.float64))
    assert values_equal({"p": a, "k": (1, "x")}, {"p": a.copy(), "k": (1, "x")})
    assert values_equal(float("nan"), float("nan"))
    assert not values_equal(None, 0.0)
