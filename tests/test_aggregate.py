"""The precedence policy is checked against enumeration of every possible reading of every edge."""

from itertools import product

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from ease.aggregate import (C, S, U, V, Calibration, EdgeRefiner, EdgeView, NeuralAggregator, NeuralConfig,
                            RankedReader, RefinedAggregator, RefinerConfig, RuleAggregator, dense_rank, pack,
                            precedence)
from ease.labels import NEI, SUPPORTS

D = 8


def policy(labels, ranks):
    decisive = [(r, l) for l, r in zip(labels, ranks) if l != NEI]
    if not decisive:
        return U
    top = min(r for r, _ in decisive)  # dense rank 0 is highest
    seen = {l for r, l in decisive if r == top}
    if len(seen) > 1:
        return C
    return S if seen.pop() == SUPPORTS else V


def random_edges(rng, n):
    return [EdgeView(f"r{i}", rng.standard_normal(3).astype(np.float32) * 2, rng.standard_normal(D).astype(np.float32),
                     int(rng.integers(0, 3)), float(rng.integers(0, 3))) for i in range(n)]


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10**6), st.integers(0, 6))
def test_precedence_matches_enumeration(seed, n):
    rng = np.random.default_rng(seed)
    edges = random_edges(rng, n)
    agg = RuleAggregator(msg_dim=D)
    status, probs, order = agg.status_batch([edges])
    got = status[0].numpy()
    if n == 0:
        np.testing.assert_allclose(got, [0, 0, 1, 0], atol=1e-12)
        return
    p = probs[0].numpy()
    ranks = dense_rank(edges)
    ranks_sorted = [ranks[i] for i in order[0]]
    want = np.zeros(4)
    for labels in product(range(3), repeat=n):
        w = float(np.prod([p[j, l] for j, l in enumerate(labels)]))
        want[policy(labels, ranks_sorted)] += w
    np.testing.assert_allclose(got, want, atol=1e-10)
    assert abs(got.sum() - 1) < 1e-12


def test_higher_authority_prevails_and_nei_does_not_block():
    sup = np.array([9.0, 0, 0], np.float32); ref = np.array([0, 9.0, 0], np.float32); nei = np.array([0, 0, 9.0], np.float32)
    z = np.zeros(D, np.float32)
    agg = RuleAggregator(msg_dim=D)
    client_supports = EdgeView("client", sup, z, 2, 1.0)
    team_refutes_later = EdgeView("team", ref, z, 1, 5.0)
    b = agg.aggregate([client_supports, team_refutes_later], prior=0.5)
    assert b.status.argmax() == S and b.decisive[0] == "client"
    # a high-authority record that settles nothing must not hide a lower-authority one that does
    b = agg.aggregate([EdgeView("client", nei, z, 2, 9.0), team_refutes_later], prior=0.5)
    assert b.status.argmax() == V and b.decisive == ("team",)
    # same authority: the later statement prevails
    b = agg.aggregate([EdgeView("a", sup, z, 1, 1.0), EdgeView("b", ref, z, 1, 2.0)], prior=0.5)
    assert b.status.argmax() == V
    # same rank, opposite readings: conflict, and the belief falls back on the prior
    b = agg.aggregate([EdgeView("a", sup, z, 1, 2.0), EdgeView("b", ref, z, 1, 2.0)], prior=0.3)
    assert b.status.argmax() == C and abs(b.belief - 0.3) < 1e-3
    b = agg.aggregate([], prior=0.8)
    assert b.status.argmax() == U and b.belief == 0.8 and b.decisive == ()


def test_untrained_refiner_is_exactly_the_rule_aggregator():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    ref = RefinedAggregator(EdgeRefiner(RefinerConfig(msg_dim=D, hidden=16)))
    rule = RuleAggregator(msg_dim=D)
    for n in range(0, 7):
        edges = random_edges(rng, n)
        a, b = ref.aggregate(edges, 0.4), rule.aggregate(edges, 0.4)
        np.testing.assert_allclose(a.status, b.status, atol=1e-12)
        assert abs(a.belief - b.belief) < 1e-12


def test_gradients_pass_through_the_exact_policy():
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    refiner = EdgeRefiner(RefinerConfig(msg_dim=D, hidden=16)).double().train()
    sets = [random_edges(rng, n) for n in (3, 5, 1, 4)]
    logits, msg, rank, auth, mask, _ = pack(sets, D)
    status = precedence(refiner(logits, msg, mask), rank, mask)
    loss = -torch.log(status[torch.arange(4), torch.tensor([S, V, U, S])].clamp_min(1e-12)).mean()
    loss.backward()
    g = refiner.out.weight.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0
    assert torch.isfinite(refiner.log_t.grad) and float(refiner.log_t.grad.abs()) > 0


def test_padding_does_not_change_a_set():
    """A set evaluated alone and the same set padded next to a longer one give the same answer."""
    rng = np.random.default_rng(2)
    torch.manual_seed(2)
    short, long = random_edges(rng, 2), random_edges(rng, 6)
    for agg in (RuleAggregator(msg_dim=D),
                RefinedAggregator(EdgeRefiner(RefinerConfig(msg_dim=D, hidden=16))),
                NeuralAggregator(RankedReader(NeuralConfig(msg_dim=D, hidden=16)))):
        alone = agg.status_batch([short])[0][0]
        padded = agg.status_batch([long, short])[0][1]
        np.testing.assert_allclose(alone.detach().numpy(), padded.detach().numpy(), atol=1e-12)


def test_single_evaluation_is_bitwise_repeatable():
    rng = np.random.default_rng(3)
    torch.manual_seed(3)
    edges = random_edges(rng, 5)
    for agg in (RuleAggregator(msg_dim=D), RefinedAggregator(EdgeRefiner(RefinerConfig(msg_dim=D, hidden=16))),
                NeuralAggregator(RankedReader(NeuralConfig(msg_dim=D, hidden=16)))):
        a, b = agg.aggregate(edges, 0.5), agg.aggregate(list(edges), 0.5)
        assert np.array_equal(a.status, b.status) and a.belief == b.belief


def test_record_order_in_the_input_does_not_matter():
    rng = np.random.default_rng(4)
    torch.manual_seed(4)
    edges = random_edges(rng, 6)
    shuffled = [edges[i] for i in rng.permutation(6)]
    for agg in (RuleAggregator(msg_dim=D), RefinedAggregator(EdgeRefiner(RefinerConfig(msg_dim=D, hidden=16))),
                NeuralAggregator(RankedReader(NeuralConfig(msg_dim=D, hidden=16)))):
        a, b = agg.aggregate(edges, 0.5), agg.aggregate(shuffled, 0.5)
        assert np.array_equal(a.status, b.status), type(agg).__name__


def test_calibration_changes_version_and_probabilities():
    a, b = Calibration(1.0), Calibration(2.0)
    assert a.version() != b.version()
    lg = np.array([[2.0, 0.0, 0.0]], np.float32)
    assert a.apply_np(lg)[0, 0] > b.apply_np(lg)[0, 0]
    assert RuleAggregator(a).version != RuleAggregator(b).version


def test_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    torch.manual_seed(5)
    edges = random_edges(rng, 4)
    ref = EdgeRefiner(RefinerConfig(msg_dim=D, hidden=16))
    torch.nn.init.normal_(ref.out.weight, std=0.3)
    a = RefinedAggregator(ref)
    a.save(tmp_path / "r")
    b = RefinedAggregator.load(tmp_path / "r")
    assert a.version == b.version
    assert np.array_equal(a.aggregate(edges, 0.5).status, b.aggregate(edges, 0.5).status)
    n = NeuralAggregator(RankedReader(NeuralConfig(msg_dim=D, hidden=16)))
    n.save(tmp_path / "n")
    m = NeuralAggregator.load(tmp_path / "n")
    assert n.version == m.version
    assert np.array_equal(n.aggregate(edges, 0.5).status, m.aggregate(edges, 0.5).status)


def test_calibrated_rules_written_as_a_refiner_are_the_same_function():
    from ease.aggregate import rule_refiner
    rng = np.random.default_rng(9)
    cal = Calibration(1.37, (0.2, -0.5, 0.3))
    torch.manual_seed(9)
    as_refiner = RefinedAggregator(rule_refiner(cal, msg_dim=D, hidden=16))
    rules = RuleAggregator(cal, msg_dim=D)
    for n in range(0, 7):
        edges = random_edges(rng, n)
        a, b = as_refiner.aggregate(edges, 0.4), rules.aggregate(edges, 0.4)
        np.testing.assert_allclose(a.status, b.status, atol=1e-12)
        assert abs(a.belief - b.belief) < 1e-12


def test_rules_as_a_refiner_are_built_the_same_way_every_time(tmp_path):
    """Found when a release was packaged twice and its aggregator file changed: the hidden layers,
    which do not affect the output, came from the global random generator."""
    from ease.aggregate import rule_refiner
    cal = Calibration(1.37, (0.2, -0.5, 0.3))
    torch.manual_seed(1)
    a = RefinedAggregator(rule_refiner(cal, msg_dim=D, hidden=16))
    torch.manual_seed(2)
    b = RefinedAggregator(rule_refiner(cal, msg_dim=D, hidden=16))
    assert a.version == b.version
    a.save(tmp_path / "a")
    b.save(tmp_path / "b")
    assert (tmp_path / "a" / "aggregator.safetensors").read_bytes() == (tmp_path / "b" / "aggregator.safetensors").read_bytes()
    assert float(a.refiner.inp.weight.detach().abs().sum()) > 0, "hidden layers are not zero: a correction can be learned"
    assert RefinedAggregator(rule_refiner(cal, msg_dim=D, hidden=16, seed=1)).version != a.version
