"""Engine bookkeeping, tested with an oracle reader so that reading ability is not what is measured.

The simulator (ease/data/episodes.py) computes ground truth from gold labels with its own code.
The engine computes statuses through ledger -> graph -> aggregator -> logic. If the two agree on
every step of every episode, the engine's handling of revisions, withdrawals, duplicates, stale
deliveries, conflicts and precedence is correct.
"""

import random

import pytest

from ease.aggregate import RuleAggregator
from ease.data.episodes import GenConfig, AtomIndex, PRED_STATUS_NAMES, generate_episode
from ease.engine import BLOCKED, NEEDS_INFO, READY, Engine, EngineConfig
from ease.labels import NEI, REFUTES, SUPPORTS
from ease.ledger import Event, Ledger
from ease.schema import AND, Action, Gate, Predicate, TaskSchema
from ease.scorer import OracleScorer


def synthetic_atoms(n=400, seed=0):
    rng = random.Random(seed)
    atoms = []
    for i in range(n):
        if i % 3:
            la = rng.choice([SUPPORTS, REFUTES, NEI])
            lb = rng.choice([l for l in (SUPPORTS, REFUTES, NEI) if l != la])
            atoms.append({"atom_id": f"a{i}", "kind": "revision", "claim": f"claim number {i}",
                          "versions": [{"text": f"evidence {i} version A", "label": la},
                                       {"text": f"evidence {i} version B", "label": lb}],
                          "page": f"page{i}", "source": "synthetic"})
        else:
            atoms.append({"atom_id": f"a{i}", "kind": "static", "claim": f"claim number {i}",
                          "versions": [{"text": f"evidence {i} only", "label": rng.choice([SUPPORTS, REFUTES, NEI])}],
                          "page": f"page{i}", "source": "synthetic"})
    return atoms


ATOMS = synthetic_atoms()
GOLD = {(a["claim"], v["text"]): v["label"] for a in ATOMS for v in a["versions"]}


def run_episode(ep, linker=None, early_cutoff=True):
    from ease.schema import TaskSchema as TS
    schema = TS.from_dict(ep["schema"])
    eng = Engine(schema, OracleScorer(GOLD), RuleAggregator(msg_dim=8), linker=linker, early_cutoff=early_cutoff)
    reports = []
    for step in ep["steps"]:
        e = step["event"]
        rep = eng.deliver(Event(e["record_id"], e["revision"], e["text"], source_id=e["source_id"],
                                authority=e["authority"], valid_from=e["valid_from"]))
        reports.append(rep)
        truth = step["truth"]
        for pid, want in truth["pred"].items():
            got = eng.belief(pid)["status"]
            assert got == PRED_STATUS_NAMES[want], (ep["episode_id"], step["t"], step["type"], pid, got, want)
        for aid, want in truth["actions"].items():
            assert eng.assessment(aid).disposition == want["disposition"], (ep["episode_id"], step["t"], aid)
        assert rep.receipt.outcome.value == step["ledger_outcome"]
    return eng, reports


@pytest.mark.parametrize("seed", range(40))
def test_engine_reproduces_simulator_ground_truth(seed):
    ep = generate_episode(AtomIndex(ATOMS, exclude_sources=()), seed, GenConfig(), f"t{seed}")
    eng, reports = run_episode(ep)
    v = eng.verify()
    assert v["not_bitwise_equal"] == 0 and v["max_abs_difference"] == 0.0


@pytest.mark.parametrize("seed", range(10))
def test_wide_episodes(seed):
    cfg = GenConfig(n_pred=(9, 14), n_distractors=(8, 20), n_events=(10, 18), n_actions=(2, 4), max_children=4)
    ep = generate_episode(AtomIndex(ATOMS, exclude_sources=()), 1000 + seed, cfg, f"w{seed}")
    run_episode(ep)


@pytest.mark.parametrize("seed", range(10))
def test_early_cutoff_changes_cost_not_answers(seed):
    ep = generate_episode(AtomIndex(ATOMS, exclude_sources=()), 500 + seed, GenConfig(), f"c{seed}")
    a, ra = run_episode(ep, early_cutoff=True)
    b, rb = run_episode(ep, early_cutoff=False)
    assert a.beliefs() == b.beliefs()
    assert sum(r.propagation["total_recomputed"] for r in ra) <= sum(r.propagation["total_recomputed"] for r in rb)


def packet_schema():
    return TaskSchema("packet", [
        Predicate("approved", "The client approved the final design.", prior=0.3),
        Predicate("date", "The delivery date is confirmed.", prior=0.5),
        Predicate("invoice", "The invoice has been issued.", prior=0.5),
    ], [Gate("ready", AND, ("approved", "date"))], [
        Action("send_packet", "Send the delivery packet to the client", "ready", cost_failure=10.0),
        Action("bill", "Send the invoice reminder", "invoice"),
    ])


def packet_engine(**kw):
    gold = {("The client approved the final design.", "Client: approved, go ahead."): SUPPORTS,
            ("The client approved the final design.", "Client: we withdraw approval pending changes."): REFUTES,
            ("The delivery date is confirmed.", "Delivery confirmed for 12 October."): SUPPORTS,
            ("The invoice has been issued.", "Invoice 881 issued."): SUPPORTS}
    return Engine(packet_schema(), OracleScorer(gold), RuleAggregator(msg_dim=8), **kw)


def test_only_affected_branch_is_recomputed():
    eng = packet_engine()
    eng.deliver(Event("mail1", 1, "Client: approved, go ahead.", source_id="client", authority=2, valid_from=1.0))
    eng.deliver(Event("mail2", 1, "Delivery confirmed for 12 October.", source_id="team", authority=1, valid_from=2.0))
    eng.deliver(Event("inv", 1, "Invoice 881 issued.", source_id="team", authority=1, valid_from=3.0))
    assert eng.assessment("send_packet").disposition == READY
    assert eng.assessment("bill").disposition == READY
    rep = eng.deliver(Event("mail1", 2, "Client: we withdraw approval pending changes.", source_id="client",
                            authority=2, valid_from=4.0))
    assert eng.assessment("send_packet").disposition == BLOCKED
    assert eng.assessment("bill").disposition == READY
    assert ("send_packet", READY, BLOCKED) in rep.changed_actions
    assert rep.propagation["recomputed"]["edge"] == 3, "one record against three predicates"
    assert rep.propagation["recomputed"].get("action", 0) == 1, "the billing action must not be re-evaluated"
    assert rep.edges_scored == 3


def test_duplicate_and_stale_deliveries_cost_nothing():
    eng = packet_engine()
    ev = Event("mail1", 2, "Client: approved, go ahead.", source_id="client", authority=2, valid_from=1.0)
    eng.deliver(ev)
    for again in (ev, Event("mail1", 1, "Client: we withdraw approval pending changes.", source_id="client",
                            authority=2, valid_from=0.5)):
        rep = eng.deliver(again)
        assert not rep.receipt.changed
        assert rep.propagation["total_recomputed"] == 0 and rep.edges_scored == 0
        assert rep.changed_actions == [] and rep.changed_predicates == []
    assert eng.belief("approved")["status"] == "SATISFIED"


def test_withdrawal_leaves_the_question_open_rather_than_answered():
    eng = packet_engine()
    eng.deliver(Event("mail1", 1, "Client: approved, go ahead.", source_id="client", authority=2, valid_from=1.0))
    eng.deliver(Event("mail2", 1, "Delivery confirmed for 12 October.", source_id="team", authority=1, valid_from=2.0))
    assert eng.assessment("send_packet").disposition == READY
    eng.deliver(Event("mail1", 2, None, source_id="client", authority=2))
    a = eng.assessment("send_packet")
    assert a.disposition == NEEDS_INFO
    # not exactly the prior: the remaining record is read as "settles nothing" with probability
    # 0.9993 rather than 1, and that residue is carried honestly into the belief
    assert abs(eng.belief("approved")["belief"] - 0.3) < 1e-3, "falls back on the declared prior"


def test_explanation_names_the_records_used():
    eng = packet_engine()
    eng.deliver(Event("mail1", 1, "Client: approved, go ahead.", source_id="client", authority=2, valid_from=1.0,
                      span="message 42, lines 3-4"))
    ex = eng.explain("send_packet")
    approved = next(p for p in ex["predicates"] if p["predicate"] == "approved")
    assert approved["records"][0]["record_id"] == "mail1"
    assert approved["records"][0]["span"] == "message 42, lines 3-4"
    assert ex["assessment"]["needs_approval"] is True


def test_validity_window_expires_without_any_event():
    eng = packet_engine(now=5.0)
    eng.deliver(Event("mail1", 1, "Client: approved, go ahead.", source_id="client", authority=2, valid_from=1.0,
                      valid_until=10.0))
    assert eng.belief("approved")["status"] == "SATISFIED"
    assert eng.ledger.next_transition_after(5.0) == 10.0
    rep = eng.set_now(10.0)
    assert eng.belief("approved")["status"] == "UNRESOLVED"
    assert ("approved", "SATISFIED", "UNRESOLVED") in rep.changed_predicates
    assert eng.verify()["not_bitwise_equal"] == 0


def test_model_revision_recomputes_edges_and_only_edges_reread():
    eng = packet_engine()
    eng.deliver(Event("mail1", 1, "Client: approved, go ahead.", source_id="client", authority=2, valid_from=1.0))
    eng.deliver(Event("inv", 1, "Invoice 881 issued.", source_id="team", authority=1, valid_from=3.0))
    n_edges = eng.counts()["edge"]
    flipped = OracleScorer({("The client approved the final design.", "Client: approved, go ahead."): REFUTES},
                           version="oracle-2")
    rep = eng.set_scorer(flipped)
    assert rep.propagation["recomputed"]["edge"] == n_edges
    assert eng.belief("approved")["status"] == "VIOLATED"
    rep = eng.set_aggregator(RuleAggregator(hard=True, msg_dim=8))
    assert rep.propagation["recomputed"].get("edge", 0) == 0, "a new aggregator must not rerun the reader"
    assert rep.propagation["recomputed"]["belief"] == 3
    assert eng.verify()["not_bitwise_equal"] == 0


def test_schema_change_keeps_unaffected_edges():
    eng = packet_engine()
    eng.deliver(Event("mail1", 1, "Client: approved, go ahead.", source_id="client", authority=2, valid_from=1.0))
    eng.deliver(Event("inv", 1, "Invoice 881 issued.", source_id="team", authority=1, valid_from=3.0))
    s = packet_schema()
    s2 = TaskSchema("packet", s.predicates + [Predicate("nda", "The NDA is signed.")],
                    [Gate("ready", AND, ("approved", "date", "nda"))], s.actions)
    rep = eng.update_schema(s2)
    assert rep.propagation["recomputed"]["edge"] == 2, "two records against the one new predicate"
    assert eng.assessment("send_packet").disposition == NEEDS_INFO
    assert eng.verify()["not_bitwise_equal"] == 0
    s3 = TaskSchema("packet", [p for p in s2.predicates if p.id != "invoice"], s2.gates,
                    [a for a in s2.actions if a.id != "bill"])
    eng.update_schema(s3)
    assert "invoice" not in eng.beliefs() and eng.verify()["not_bitwise_equal"] == 0


def test_declared_links_restrict_what_is_compared():
    eng = packet_engine()
    rep = eng.deliver(Event("inv", 1, "Invoice 881 issued.", source_id="team", authority=1, valid_from=3.0,
                            meta={"about": ["invoice"]}))
    assert rep.edges_scored == 1 and rep.links_added == 1
    assert eng.belief("invoice")["status"] == "SATISFIED"


def test_linker_controls_fan_out_and_can_miss():
    """A linker that omits the right predicate makes the engine cheap and wrong. This is the risk of
    learned sparsity, shown deliberately."""
    miss = lambda rid, text, schema: ["date"]
    eng = packet_engine(linker=miss)
    rep = eng.deliver(Event("mail1", 1, "Client: approved, go ahead.", source_id="client", authority=2, valid_from=1.0))
    assert rep.edges_scored == 1
    assert eng.belief("approved")["status"] == "UNRESOLVED", "the dependency was never examined"


def test_thresholds_make_ready_harder_to_reach():
    eng = packet_engine(config=EngineConfig(ready_threshold=0.999999))
    eng.scorer.margin = 3.0  # a less confident reader
    eng.deliver(Event("mail1", 1, "Client: approved, go ahead.", source_id="client", authority=2, valid_from=1.0))
    eng.deliver(Event("mail2", 1, "Delivery confirmed for 12 October.", source_id="team", authority=1, valid_from=2.0))
    assert eng.assessment("send_packet").disposition == NEEDS_INFO
    eng.set_config(EngineConfig(ready_threshold=0.5))
    assert eng.assessment("send_packet").disposition == READY


def test_engine_rebuilds_from_a_persisted_ledger(tmp_path):
    path = str(tmp_path / "l.sqlite")
    eng = Engine(packet_schema(), OracleScorer({("The client approved the final design.", "Client: approved, go ahead."): SUPPORTS}),
                 RuleAggregator(msg_dim=8), ledger=Ledger(path, "packet"))
    eng.deliver(Event("mail1", 1, "Client: approved, go ahead.", source_id="client", authority=2, valid_from=1.0))
    before = eng.beliefs()
    eng.ledger.close()
    again = Engine(packet_schema(), OracleScorer({("The client approved the final design.", "Client: approved, go ahead."): SUPPORTS}),
                   RuleAggregator(msg_dim=8), ledger=Ledger(path, "packet"))
    assert again.beliefs() == before


# ----------------------------------------------------------------------------- joint reading
JOIN_CLAIM = "The client has approved the latest version of the logo."
A4 = "The client approved version 4 of the logo."
L4 = "The latest version of the logo is version 4."
L5 = "The latest version of the logo is version 5."


def join_engine(join=True):
    schema = TaskSchema("j", [Predicate("latest_approved", JOIN_CLAIM, prior=0.4, join=join),
                              Predicate("date", "The delivery date is confirmed.")],
                        [Gate("ready", AND, ("latest_approved", "date"))], [Action("send", "send", "ready")])
    gold = {(JOIN_CLAIM, f"{A4} {L4}"): SUPPORTS, (JOIN_CLAIM, f"{A4} {L5}"): REFUTES,
            ("The delivery date is confirmed.", "Delivery confirmed for 12 October."): SUPPORTS}
    return Engine(schema, OracleScorer(gold), RuleAggregator(msg_dim=8))


def test_join_predicate_is_settled_only_by_reading_records_together():
    eng = join_engine()
    eng.deliver(Event("date", 1, "Delivery confirmed for 12 October.", source_id="client", authority=2, valid_from=1.0,
                      meta={"about": ["date"]}))
    rep = eng.deliver(Event("approval", 1, A4, source_id="client", authority=2, valid_from=2.0,
                            meta={"about": ["latest_approved"]}))
    assert eng.belief("latest_approved")["status"] == "UNRESOLVED", "one record alone settles nothing"
    assert rep.edges_scored == 1
    rep = eng.deliver(Event("latest", 1, L5, source_id="team", authority=1, valid_from=3.0,
                            meta={"about": ["latest_approved"]}))
    assert eng.belief("latest_approved")["status"] == "VIOLATED"
    assert eng.assessment("send").disposition == BLOCKED
    assert rep.edges_scored == 1, "one joint reading, not one per record"
    rep = eng.deliver(Event("latest", 2, L4, source_id="team", authority=1, valid_from=4.0))
    assert eng.belief("latest_approved")["status"] == "SATISFIED"
    assert eng.assessment("send").disposition == READY
    assert rep.propagation["recomputed"] == {"jedge": 1, "belief": 1, "action": 1}
    ex = eng.explain("send")
    p = next(x for x in ex["predicates"] if x["predicate"] == "latest_approved")
    assert p["read_jointly"] and [r["record_id"] for r in p["records"]] == ["approval", "latest"]
    assert eng.verify()["not_bitwise_equal"] == 0


def test_without_join_the_same_records_leave_the_claim_open():
    eng = join_engine(join=False)
    eng.deliver(Event("approval", 1, A4, source_id="client", authority=2, valid_from=2.0))
    eng.deliver(Event("latest", 1, L5, source_id="team", authority=1, valid_from=3.0))
    assert eng.belief("latest_approved")["status"] == "UNRESOLVED"


def test_join_flag_can_be_switched_by_a_schema_update():
    eng = join_engine(join=False)
    eng.deliver(Event("approval", 1, A4, source_id="client", authority=2, valid_from=2.0))
    eng.deliver(Event("latest", 1, L5, source_id="team", authority=1, valid_from=3.0))
    s = eng.schema
    s2 = TaskSchema("j", [Predicate("latest_approved", JOIN_CLAIM, prior=0.4, join=True), s.predicates[1]],
                    s.gates, s.actions)
    eng.update_schema(s2)
    assert eng.belief("latest_approved")["status"] == "VIOLATED"
    assert "jedge" in eng.counts() and eng.verify()["not_bitwise_equal"] == 0
    eng.update_schema(TaskSchema("j", [Predicate("latest_approved", JOIN_CLAIM, prior=0.4, join=False),
                                      s.predicates[1]], s.gates, s.actions))
    assert eng.belief("latest_approved")["status"] == "UNRESOLVED" and "jedge" not in eng.counts()
    assert eng.verify()["not_bitwise_equal"] == 0


def test_withdrawing_a_member_reopens_a_join_predicate():
    eng = join_engine()
    eng.deliver(Event("approval", 1, A4, source_id="client", authority=2, valid_from=2.0, meta={"about": ["latest_approved"]}))
    eng.deliver(Event("latest", 1, L4, source_id="team", authority=1, valid_from=3.0, meta={"about": ["latest_approved"]}))
    assert eng.belief("latest_approved")["status"] == "SATISFIED"
    eng.deliver(Event("approval", 2, None, source_id="client", authority=2))
    assert eng.belief("latest_approved")["status"] == "UNRESOLVED"
    assert eng.verify()["not_bitwise_equal"] == 0


@pytest.mark.slow
def test_canonical_readings_on_the_cpu_do_not_depend_on_the_batch():
    """Found when the service was run on the CPU: with batches of eight, a pair's reading depended on
    the other pairs in its batch by up to 6e-6. The CPU default is now a batch of one. Needs the
    released model; skipped without it."""
    import os
    from pathlib import Path

    import numpy as np

    from ease.scorer import ModelScorer

    model = Path(os.environ.get("EASE_TEST_MODEL", "release/edge"))
    if not (model / "model.safetensors").exists():
        pytest.skip("no released model")
    claim = "The client has approved the design."
    texts = ["Email from the client: we approve the design as presented, please go ahead.",
             "The venue has confirmed the booking in writing for 14 November.", "Short note.",
             "Email from accounts: the invoice will follow next week once the finance team, closed on Monday, has it."]
    pairs = [(claim, t) for t in texts]
    s = ModelScorer(str(model), device="cpu", canonical=True)
    assert s.canonical_batch == 1
    s.cache = {}
    alone = [np.asarray(s.score([p])[0][0]) for p in pairs]
    s.cache = {}
    together = [np.asarray(v[0]) for v in s.score(pairs)]
    assert all(np.array_equal(a, b) for a, b in zip(alone, together))
