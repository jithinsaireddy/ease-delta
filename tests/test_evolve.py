"""Self-evolution: the error bound on adversarial sequences, retractable memory, and the gate."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from ease.aggregate import EdgeRefiner, RefinerConfig, RuleAggregator
from ease.engine import Engine, pair_key
from ease.evolve.aggregator import EvolvingAggregator
from ease.evolve.consolidate import (Bundle, GateConfig, Readings, VersionStore, consolidate, in_holdout,
                                     refiner_probs, score)
from ease.evolve.memory import CorrectionMemory, MemoryConfig
from ease.evolve.threshold import ThresholdTracker
from ease.labels import NEI, REFUTES, SUPPORTS
from ease.ledger import Event
from ease.schema import Action, Predicate, TaskSchema
from ease.scorer import OracleScorer

D = 16


# ----------------------------------------------------------------------------- threshold bound
def run_tracker(tr, confidences, wrong, delays):
    """Feed a sequence; `delays[i]` is how many later cases pass before verdict i arrives."""
    pending = []
    worst_gap = -1.0
    for i, (c, w, d) in enumerate(zip(confidences, wrong, delays)):
        if tr.endorse(c):
            pending.append([d, w])
        still = []
        for item in pending:
            if item[0] <= 0:
                tr.verdict(item[1])
                assert tr.error_rate <= tr.bound + 1e-12, (tr.state(), i)
                worst_gap = max(worst_gap, tr.error_rate - tr.bound)
            else:
                item[0] -= 1
                still.append(item)
        pending = still
    for item in pending:
        tr.verdict(item[1])
        assert tr.error_rate <= tr.bound + 1e-12
    return tr


@settings(max_examples=300, deadline=None)
@given(st.integers(0, 10**6), st.integers(1, 400), st.floats(0.01, 0.3), st.floats(0.005, 0.2),
       st.floats(0.0, 1.0), st.integers(0, 6))
def test_T5_bound_holds_for_every_sequence(seed, n, alpha, eta, tau0, max_delay):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0, 1, n)
    mode = seed % 4
    if mode == 0:
        wrong = rng.uniform(0, 1, n) < 0.5                       # coin flips
    elif mode == 1:
        wrong = np.ones(n, bool)                                 # always wrong when endorsed
    elif mode == 2:
        wrong = conf > 0.7                                       # confidence is anti-informative
    else:
        wrong = (np.arange(n) // 25) % 2 == 0                    # long runs of errors, then none
    delays = rng.integers(0, max_delay + 1, n)
    tr = run_tracker(ThresholdTracker(alpha=alpha, eta=eta, tau=tau0, tau_initial=tau0), conf, wrong, delays)
    assert tr.tau <= 1.0 + tr.max_outstanding * eta * (1 - alpha) + 1e-12


def test_always_wrong_model_stalls_instead_of_erring_forever():
    tr = ThresholdTracker(alpha=0.05, eta=0.05, tau=0.5, tau_initial=0.5)
    endorsed = 0
    for _ in range(10_000):
        if tr.endorse(1.0):
            endorsed += 1
            tr.verdict(True)
    assert tr.stalled and endorsed < 20, "it must stop endorsing after a handful of errors"


def test_long_run_error_rate_approaches_alpha_when_the_model_is_informative():
    rng = np.random.default_rng(0)
    tr = ThresholdTracker(alpha=0.1, eta=0.02, tau=0.5, tau_initial=0.5)
    for _ in range(40_000):
        c = float(rng.uniform())
        if tr.endorse(c):
            tr.verdict(rng.uniform() > c)  # calibrated: right with probability c
    assert tr.verdicts > 2000
    assert abs(tr.error_rate - 0.1) < 0.01
    assert tr.error_rate <= tr.bound


def test_tracker_round_trip(tmp_path):
    tr = ThresholdTracker(alpha=0.05, eta=0.1, tau=0.8, tau_initial=0.8)
    tr.endorse(0.9); tr.verdict(True)
    tr.save(tmp_path / "t.json")
    assert ThresholdTracker.load(tmp_path / "t.json").state() == tr.state()
    with pytest.raises(RuntimeError):
        ThresholdTracker().verdict(False)


# ----------------------------------------------------------------------------- memory
def test_verified_pair_is_honoured_at_once_and_forgotten_on_request():
    mem = CorrectionMemory()
    rng = np.random.default_rng(0)
    probs = np.array([[0.7, 0.2, 0.1]])
    msg = rng.standard_normal((1, D)).astype(np.float32)
    before, how = mem.adjust(probs, msg, ["pairA"])
    assert np.array_equal(before, probs) and how[0] == 0
    v0 = mem.version
    item = mem.add("pairA", REFUTES, msg[0])
    assert mem.version != v0
    after, how = mem.adjust(probs, msg, ["pairA"])
    assert how[0] == 2 and after[0].argmax() == REFUTES and after[0, REFUTES] > 0.99
    assert mem.remove(item)
    restored, how = mem.adjust(probs, msg, ["pairA"])
    assert np.array_equal(restored, probs) and how[0] == 0, "deleting a correction restores the earlier reading exactly"
    assert mem.version == v0


def test_nothing_is_generalised_until_a_gate_enables_it():
    rng = np.random.default_rng(1)
    mem = CorrectionMemory()
    base = rng.standard_normal(D).astype(np.float32)
    for i in range(20):
        mem.add(f"p{i}", REFUTES, base + 0.01 * rng.standard_normal(D).astype(np.float32))
    probs = np.array([[0.6, 0.2, 0.2]])
    near = (base + 0.01 * rng.standard_normal(D)).astype(np.float32)[None]
    out, how = mem.adjust(probs, near, ["unseen"])
    assert mem.cfg.lam_max == 0.0 and np.array_equal(out, probs) and how[0] == 0
    mem.set_config(MemoryConfig(tau=0.8, lam_max=0.9))
    out, how = mem.adjust(probs, near, ["unseen"])
    assert how[0] == 1 and out[0].argmax() == REFUTES
    far = rng.standard_normal((1, D)).astype(np.float32)
    out, how = mem.adjust(probs, far, ["unseen2"])
    assert how[0] == 0, "an unrelated pair is left alone"


def test_memory_survives_restart(tmp_path):
    path = str(tmp_path / "m.sqlite")
    mem = CorrectionMemory(path)
    rng = np.random.default_rng(2)
    a = mem.add("x", SUPPORTS, rng.standard_normal(D)); mem.add("y", NEI, rng.standard_normal(D))
    mem.remove(a)
    v = mem.version
    mem.close()
    again = CorrectionMemory(path)
    assert again.version == v and len(again) == 1 and again.exact == {"y": NEI}


def test_a_correction_changes_beliefs_without_rerunning_the_reader():
    schema = TaskSchema("t", [Predicate("approved", "The client approved the design.")], [],
                        [Action("send", "send it", "approved")])
    text = "Client: looks fine I guess, but hold on."
    scorer = OracleScorer({(schema.predicates[0].text, text): SUPPORTS}, msg_dim=D)
    mem = CorrectionMemory()
    agg = EvolvingAggregator(RuleAggregator(msg_dim=D), mem)
    eng = Engine(schema, scorer, agg)
    eng.deliver(Event("mail", 1, text, source_id="client", authority=2, valid_from=1.0))
    assert eng.assessment("send").disposition == "READY"
    edge = eng.g.value("edge:mail|approved")
    mem.add(pair_key(schema.predicates[0].text, text), REFUTES, edge["msg"], source="user")
    rep = eng.set_aggregator(agg)  # same object, new version because the memory changed
    assert rep.propagation["recomputed"].get("edge", 0) == 0 and rep.edges_scored == 0
    assert rep.propagation["recomputed"]["belief"] == 1
    assert eng.assessment("send").disposition == "BLOCKED"
    assert ("send", "READY", "BLOCKED") in rep.changed_actions
    assert eng.verify()["not_bitwise_equal"] == 0
    mem.remove_pair(pair_key(schema.predicates[0].text, text))
    eng.set_aggregator(agg)
    assert eng.assessment("send").disposition == "READY", "withdrawing the correction undoes it"


# ----------------------------------------------------------------------------- consolidation gate
def synth(rng, n, shift=0.0, flip=0.0, tag="f"):
    """Messages carry the label; logits are biased against REFUTES by `shift`, which is what a model
    meeting a new domain looks like. `flip` is the fraction of labels replaced by a wrong one."""
    y = rng.integers(0, 3, n)
    centres = np.eye(3, D) * 4.0
    msgs = (centres[y] + rng.standard_normal((n, D))).astype(np.float32)
    logits = (np.eye(3)[y] * 3.0 + rng.standard_normal((n, 3))).astype(np.float32)
    logits[:, REFUTES] -= shift
    labels = y.copy()
    wrong = rng.uniform(size=n) < flip
    labels[wrong] = (labels[wrong] + rng.integers(1, 3, wrong.sum())) % 3
    return Readings([f"{tag}{i}" for i in range(n)], labels, logits, msgs)


def champion():
    torch.manual_seed(0)
    return Bundle(EdgeRefiner(RefinerConfig(msg_dim=D, hidden=32)).double().eval(), MemoryConfig())


def regression_sets(rng):
    return {"in_domain": synth(rng, 1500, shift=0.0, tag="r")}


def test_gate_adopts_an_improvement_that_does_not_regress():
    rng = np.random.default_rng(3)
    fb = synth(rng, 1200, shift=4.0)
    reg = regression_sets(rng)
    replay = synth(rng, 1500, shift=0.0, tag="rep")
    ch = champion()
    new, rep = consolidate(ch, fb, reg, replay, GateConfig(finetune_steps=200))
    assert rep.adopted, rep.reason
    c = rep.candidates[rep.chosen]
    assert c["gain_over_champion"]["lo"] > 0 and c["passes_regression"]
    fresh = synth(np.random.default_rng(33), 2000, shift=4.0, tag="new")
    before = score(ch.read(fresh, None), fresh.labels)
    after = score(new.read(fresh, fb), fresh.labels)
    assert after["accuracy"] > before["accuracy"] + 0.05, (before, after)


def test_gate_refuses_learning_from_mostly_wrong_feedback():
    rng = np.random.default_rng(4)
    fb = synth(rng, 1200, shift=0.0, flip=0.5)
    reg = regression_sets(rng)
    ch = champion()
    new, rep = consolidate(ch, fb, reg, None, GateConfig(finetune_steps=200))
    base = rep.candidates["champion"]["regression"]["in_domain"]
    chosen = rep.candidates[rep.chosen]["regression"]["in_domain"]
    assert base["accuracy"] - chosen["accuracy"] <= 0.005 + 1e-12
    assert chosen["nll"] - base["nll"] <= 0.01 + 1e-12
    if not rep.adopted:
        assert new is ch
    # whatever was or was not adopted, clean data must not be read worse
    clean = synth(np.random.default_rng(44), 2000, shift=0.0, tag="clean")
    assert score(new.read(clean, fb), clean.labels)["accuracy"] >= score(ch.read(clean, None), clean.labels)["accuracy"] - 0.01


def test_gate_waits_for_enough_feedback():
    rng = np.random.default_rng(5)
    ch = champion()
    new, rep = consolidate(ch, synth(rng, 50, shift=4.0), regression_sets(rng), None, GateConfig())
    assert not rep.adopted and new is ch and "required" in rep.reason


def test_holdout_assignment_is_stable():
    assert [in_holdout(f"p{i}", 0.3) for i in range(200)] == [in_holdout(f"p{i}", 0.3) for i in range(200)]
    frac = np.mean([in_holdout(f"q{i}", 0.3) for i in range(5000)])
    assert 0.27 < frac < 0.33


def test_versions_can_be_rolled_back(tmp_path):
    rng = np.random.default_rng(6)
    store = VersionStore(tmp_path / "versions")
    ch = champion()
    v0 = store.save(ch, note="initial")
    new, rep = consolidate(ch, synth(rng, 1200, shift=4.0), regression_sets(rng), None, GateConfig(finetune_steps=150))
    assert rep.adopted
    v1 = store.save(new, rep, note="after feedback")
    assert store.current() == v1 and [l["version"] for l in store.lineage()] == [v1, v0]
    r = synth(rng, 50)
    assert np.allclose(refiner_probs(store.load().refiner, r), refiner_probs(new.refiner, r))
    assert store.rollback() == v0 and store.current() == v0
    assert np.allclose(refiner_probs(store.load().refiner, r), refiner_probs(ch.refiner, r))
    assert (tmp_path / "versions" / v1).exists(), "the abandoned version is kept for audit"


# ----------------------------------------------------------------------------- set-level regression
from ease.aggregate import RefinedAggregator  # noqa: E402
from ease.evolve.consolidate import SetSuite, finetune, score_sets, set_status_probs  # noqa: E402


def policy_label(labels, ranks):
    """Correct status of a set from the true label of each reading (ranks: 0 is highest)."""
    decisive = [(r, l) for l, r in zip(labels, ranks) if l != NEI]
    if not decisive:
        return 2
    top = min(r for r, _ in decisive)
    seen = {l for r, l in decisive if r == top}
    return 3 if len(seen) > 1 else (0 if seen.pop() == SUPPORTS else 1)


def synth_sets(rng, n, shift=0.0, min_size=2, max_size=5):
    lg, mg, rk, off, lab = [], [], [], [0], []
    centres = np.eye(3, D) * 4.0
    for _ in range(n):
        k = int(rng.integers(min_size, max_size + 1))
        y = rng.integers(0, 3, k)
        ranks = np.sort(rng.integers(0, 3, k))
        m = (centres[y] + rng.standard_normal((k, D))).astype(np.float32)
        l = (np.eye(3)[y] * 3.0 + rng.standard_normal((k, 3))).astype(np.float32)
        l[:, REFUTES] -= shift
        dense = {v: i for i, v in enumerate(sorted(set(ranks.tolist())))}
        lg.append(l); mg.append(m); rk.append(np.asarray([dense[v] for v in ranks], np.int64))
        off.append(off[-1] + k)
        lab.append(policy_label(y.tolist(), [dense[v] for v in ranks]))
    return SetSuite(np.concatenate(lg), np.concatenate(mg), np.concatenate(rk), np.asarray(off, np.int64),
                    np.asarray(lab, np.int64))


def test_set_probabilities_match_the_engine_aggregator():
    from ease.aggregate import EdgeView
    rng = np.random.default_rng(10)
    torch.manual_seed(10)
    ref = EdgeRefiner(RefinerConfig(msg_dim=D, hidden=32)).double()
    torch.nn.init.normal_(ref.out.weight, std=0.5)
    suite = synth_sets(rng, 60)
    got = set_status_probs(Bundle(ref, MemoryConfig()), suite, None)
    agg = RefinedAggregator(ref)
    for i in range(len(suite)):
        a, z = suite.offsets[i], suite.offsets[i + 1]
        # authority descending with rank so that the aggregator's own ranking reproduces suite.rank
        edges = [EdgeView(f"r{j:02d}", suite.logits[j], suite.msgs[j], 10 - int(suite.rank[j]), 0.0)
                 for j in range(a, z)]
        np.testing.assert_allclose(got[i], agg.aggregate(edges, 0.5).status, atol=1e-10)


def test_context_is_frozen_unless_allowed():
    rng = np.random.default_rng(11)
    torch.manual_seed(11)
    champ = EdgeRefiner(RefinerConfig(msg_dim=D, hidden=32)).double()
    fb = synth(rng, 600, shift=4.0)
    frozen = finetune(champ, fb, None, GateConfig(finetune_steps=60))
    free = finetune(champ, fb, None, GateConfig(finetune_steps=60, train_context=True))
    h = champ.cfg.hidden
    assert torch.equal(frozen.ctx.weight, champ.ctx.weight) and torch.equal(frozen.ctx.bias, champ.ctx.bias)
    assert torch.equal(frozen.mid.weight[:, h:], champ.mid.weight[:, h:])
    assert not torch.equal(frozen.mid.weight[:, :h], champ.mid.weight[:, :h]), "per-edge part still learns"
    assert not torch.equal(free.ctx.weight, champ.ctx.weight)


class Saboteur(EdgeRefiner):
    """Reads single edges like the model it wraps, and reverses supports and refutes in larger sets."""

    def __init__(self, inner):
        super().__init__(inner.cfg)
        self.load_state_dict(inner.state_dict())

    def forward(self, logits, msg, mask):
        p = super().forward(logits, msg, mask)
        big = (mask.sum(1) > 1).view(-1, 1, 1)
        return torch.where(big, p[..., [1, 0, 2]], p)


def test_gate_catches_a_candidate_that_is_only_wrong_on_sets(monkeypatch):
    import ease.evolve.consolidate as C
    rng = np.random.default_rng(12)
    fb = synth(rng, 1200, shift=4.0)
    reg = regression_sets(rng)
    real_finetune = C.finetune
    monkeypatch.setattr(C, "finetune", lambda *a, **k: Saboteur(real_finetune(*a, **k)).double().eval())
    # memory candidates disabled (lam = 0 only), so every candidate but the champion is the saboteur
    cfg = GateConfig(finetune_steps=200, lams=(0.0,))
    suites = {"validation": synth_sets(np.random.default_rng(13), 400, shift=0.0)}  # in-domain, like real suites
    adopted_without, without_sets = consolidate(champion(), fb, reg, None, cfg)
    assert without_sets.adopted and without_sets.chosen.startswith("refiner"), without_sets.reason
    assert score_sets(set_status_probs(adopted_without, suites["validation"], None),
                      suites["validation"].labels)["accuracy"] < 0.5, "without set suites a set-breaking model gets in"
    kept, with_sets = consolidate(champion(), fb, reg, None, cfg, set_suites=suites)
    assert not with_sets.adopted and with_sets.chosen == "champion", "the set-level check must reject the saboteur"
    assert not with_sets.candidates["refiner"]["regression"]["sets:validation"]["pass"]
    assert "sets:validation" in with_sets.reason


def test_whatever_the_gate_adopts_passes_every_set_suite(monkeypatch):
    import ease.evolve.consolidate as C
    rng = np.random.default_rng(15)
    fb = synth(rng, 1200, shift=4.0)
    real_finetune = C.finetune
    monkeypatch.setattr(C, "finetune", lambda *a, **k: Saboteur(real_finetune(*a, **k)).double().eval())
    suites = {"validation": synth_sets(np.random.default_rng(16), 400, shift=0.0)}
    new, rep = consolidate(champion(), fb, regression_sets(rng), None, GateConfig(finetune_steps=200), set_suites=suites)
    if rep.adopted:
        assert rep.candidates[rep.chosen]["regression"]["sets:validation"]["pass"]


def test_rules_as_refiner_have_no_context_to_learn_from():
    from ease.aggregate import Calibration, rule_refiner
    ref = rule_refiner(Calibration(1.2, (0.1, -0.3, 0.2)), msg_dim=D, hidden=16)
    assert float(ref.ctx.weight.detach().abs().sum()) == 0.0 and float(ref.ctx.bias.detach().abs().sum()) == 0.0
    tuned = finetune(ref, synth(np.random.default_rng(14), 400, shift=4.0), None, GateConfig(finetune_steps=40))
    assert float(tuned.ctx.weight.detach().abs().sum()) == 0.0, "a correction learned later still ignores the rest of the set"


def test_gate_reports_negative_flips_and_can_refuse_them():
    """A candidate can keep a suite's accuracy while breaking many readings the champion got right.
    The gate reports those flips, and refuses the candidate when told to."""
    rng = np.random.default_rng(21)
    fb = synth(rng, 1200, shift=4.0)
    reg = regression_sets(rng)
    ch = champion()
    _, rep = consolidate(ch, fb, reg, synth(rng, 1500, shift=0.0, tag="rep"), GateConfig(finetune_steps=200))
    for k, c in rep.candidates.items():
        r = c["regression"]["in_domain"]
        assert "negative_flips" in r and 0.0 <= r["negative_flip_rate"] <= 1.0
        if k == "champion":
            assert r["negative_flips"] == 0, "the champion cannot flip against itself"
    _, strict = consolidate(ch, fb, reg, synth(rng, 1500, shift=0.0, tag="rep"),
                            GateConfig(finetune_steps=200, max_negative_flip_rate=0.0))
    for k, c in strict.candidates.items():
        r = c["regression"]["in_domain"]
        assert r["pass"] == (r["negative_flips"] == 0 and r["accuracy_drop"] <= 0.005 + 1e-12 and r["nll_rise"] <= 0.01 + 1e-12), k
    if strict.adopted:
        assert strict.candidates[strict.chosen]["regression"]["in_domain"]["negative_flips"] == 0
