"""Planner checked by brute force: certificates against sampling, value of information against
direct simulation of asking."""

import random

import numpy as np
import pytest

from ease.aggregate import RuleAggregator
from ease.engine import BLOCKED, READY, Engine, disposition_of
from ease.labels import REFUTES, SUPPORTS
from ease.ledger import Event
from ease.logic import evaluate
from ease.planner import best_question, certify_stable, open_predicates, unnecessary_questions, value_of_asking
from ease.schema import AND, ATLEAST, OR, Action, Gate, Predicate, TaskSchema
from ease.scorer import OracleScorer


def engine_for(schema, facts):
    """facts: {predicate_id: SUPPORTS | REFUTES}; one record per settled predicate."""
    gold = {}
    eng_events = []
    for i, (pid, lab) in enumerate(facts.items()):
        text = f"record about {pid}"
        gold[(schema.predicate(pid).text, text)] = lab
        eng_events.append(Event(f"r_{pid}", 1, text, source_id="s", authority=1, valid_from=float(i),
                                meta={"about": [pid]}))
    eng = Engine(schema, OracleScorer(gold, margin=30.0), RuleAggregator(msg_dim=8))
    for e in eng_events:
        eng.deliver(e)
    return eng


def test_or_gate_already_satisfied_makes_other_question_unnecessary():
    s = TaskSchema("t", [Predicate("a", "claim a"), Predicate("b", "claim b")],
                   [Gate("g", OR, ("a", "b"))], [Action("act", "d", "g")])
    eng = engine_for(s, {"a": SUPPORTS})
    assert eng.assessment("act").disposition == READY
    c = certify_stable(eng, "act", ["b"])
    assert c.stable and c.settlements_checked == 3
    assert unnecessary_questions(eng) == {"act": ["b"]}
    assert best_question(eng) is None, "nothing is worth asking"


def test_and_gate_already_violated_makes_other_question_unnecessary():
    s = TaskSchema("t", [Predicate("a", "claim a"), Predicate("b", "claim b")],
                   [Gate("g", AND, ("a", "b"))], [Action("act", "d", "g")])
    eng = engine_for(s, {"a": REFUTES})
    assert eng.assessment("act").disposition == BLOCKED
    assert certify_stable(eng, "act", ["b"]).stable
    assert best_question(eng) is None


def test_certificate_reports_a_counterexample_when_the_answer_matters():
    s = TaskSchema("t", [Predicate("a", "claim a"), Predicate("b", "claim b")],
                   [Gate("g", AND, ("a", "b"))], [Action("act", "d", "g")])
    eng = engine_for(s, {"a": SUPPORTS})
    c = certify_stable(eng, "act", ["b"])
    assert not c.stable and c.counterexample["disposition"] in (READY, BLOCKED)


def test_single_questions_are_worthless_but_the_pair_is_not():
    """The myopia case: an AND of two open predicates."""
    s = TaskSchema("t", [Predicate("a", "claim a", prior=0.9, ask_cost=0.1),
                         Predicate("b", "claim b", prior=0.9, ask_cost=0.1)],
                   [Gate("g", AND, ("a", "b"))], [Action("act", "d", "g", value_success=5.0, cost_failure=1.0)])
    eng = engine_for(s, {})
    assert value_of_asking(eng, ["a"]).value < 0 and value_of_asking(eng, ["b"]).value < 0
    q = best_question(eng)
    assert q is not None and set(q.predicates) == {"a", "b"}
    assert abs(q.gross_gain - 0.81 * 5.0) < 1e-6 and abs(q.value - (0.81 * 5.0 - 0.2)) < 1e-6
    assert q.could_change == ("act",)
    assert best_question(eng, max_set=1) is None


def test_expensive_questions_are_not_asked():
    s = TaskSchema("t", [Predicate("a", "claim a", prior=0.5, ask_cost=100.0)], [],
                   [Action("act", "d", "a", value_success=1.0)])
    assert best_question(engine_for(s, {})) is None


def test_cheapest_likely_stopper_is_asked_first():
    s = TaskSchema("t", [Predicate("a", "claim a", prior=0.95, ask_cost=1.0),
                         Predicate("b", "claim b", prior=0.40, ask_cost=1.0)],
                   [Gate("g", AND, ("a", "b"))], [Action("act", "d", "g", value_success=20.0)])
    q = best_question(engine_for(s, {}))
    assert q.ask_first == "b", "b is far more likely to come back 'no' and end the matter"


def random_schema(rng, n):
    preds = [Predicate(f"p{i}", f"claim {i}", prior=rng.choice([0.2, 0.5, 0.8]), ask_cost=rng.choice([0.1, 0.5, 1.0]))
             for i in range(n)]
    nodes, gates = [p.id for p in preds], []
    while len(nodes) > 1:
        rng.shuffle(nodes)
        k = rng.randint(2, min(3, len(nodes)))
        kids, nodes = nodes[:k], nodes[k:]
        op = rng.choice([AND, OR, ATLEAST])
        gates.append(Gate(f"g{len(gates)}", op, tuple(kids), rng.randint(1, k) if op == ATLEAST else 0))
        nodes.append(gates[-1].id)
    return TaskSchema("t", preds, gates, [Action("act", "d", nodes[0], value_success=rng.choice([1.0, 5.0]),
                                                 cost_failure=rng.choice([1.0, 5.0]))])


@pytest.mark.parametrize("seed", range(30))
def test_certificates_are_never_contradicted(seed):
    """Soundness: a certified-stable action never changes under any settlement, including ones drawn
    at random after the fact."""
    rng = random.Random(seed)
    s = random_schema(rng, rng.randint(2, 6))
    facts = {p.id: rng.choice([SUPPORTS, REFUTES]) for p in s.predicates if rng.random() < 0.5}
    eng = engine_for(s, facts)
    opens = open_predicates(eng)
    if not opens:
        return
    free = rng.sample(opens, rng.randint(1, len(opens)))
    cert = certify_stable(eng, "act", free)
    ls, lb = eng.leaf_state()
    a = eng._action("act")
    seen = set()
    for _ in range(60):
        s2, b2 = dict(ls), dict(lb)
        for pid in free:
            v = rng.choice([0, 1, 2])
            s2[pid] = np.eye(3)[v]
            b2[pid] = 1.0 if v == 0 else 0.0 if v == 1 else s.predicate(pid).prior
        seen.add(disposition_of(evaluate(s, a.requires, s2, b2).status, eng.cfg))
    if cert.stable:
        assert seen == {cert.disposition}
    else:
        assert cert.counterexample is not None


@pytest.mark.parametrize("seed", range(20))
def test_value_of_asking_matches_simulation_of_actually_asking(seed):
    rng = random.Random(100 + seed)
    s = random_schema(rng, rng.randint(2, 5))
    facts = {p.id: rng.choice([SUPPORTS, REFUTES]) for p in s.predicates if rng.random() < 0.4}
    eng = engine_for(s, facts)
    opens = open_predicates(eng)
    if not opens:
        return
    ask = rng.sample(opens, min(len(opens), rng.randint(1, 2)))
    q = value_of_asking(eng, ask)

    def utility(e):
        a = e.assessment("act")
        return a.expected_utility if a.disposition == READY else 0.0

    base = utility(eng)
    _, lb = eng.leaf_state()
    expected = 0.0
    for bits in range(2 ** len(ask)):
        answers = {p: bool(bits >> i & 1) for i, p in enumerate(ask)}
        w = float(np.prod([lb[p] if yes else 1 - lb[p] for p, yes in answers.items()]))
        e2 = engine_for(s, {**facts, **{p: SUPPORTS if yes else REFUTES for p, yes in answers.items()}})
        expected += w * utility(e2)
    assert abs(q.gross_gain - (expected - base)) < 1e-6
    assert abs(q.cost - sum(s.predicate(p).ask_cost for p in ask)) < 1e-12


def test_a_confirmation_is_proposed_when_requirements_are_likely_but_not_sure_enough():
    """Found in use: each requirement was read as satisfied at about 0.86, so none was 'open', the
    action stayed below its threshold, and no question was proposed. A person would ask to confirm."""
    from ease.engine import NEEDS_INFO, EngineConfig

    s = TaskSchema("t", [Predicate("a", "claim a", ask_cost=0.2), Predicate("b", "claim b", ask_cost=0.2)],
                   [Gate("g", AND, ("a", "b"))], [Action("act", "d", "g", value_success=10.0, cost_failure=25.0)])
    gold = {("claim a", "record about a"): SUPPORTS, ("claim b", "record about b"): SUPPORTS}
    eng = Engine(s, OracleScorer(gold, margin=2.5), RuleAggregator(msg_dim=8), config=EngineConfig(ready_threshold=0.9))
    for i, pid in enumerate("ab"):
        eng.deliver(Event(f"r_{pid}", 1, f"record about {pid}", source_id="s", authority=1, valid_from=float(i),
                          meta={"about": [pid]}))
    ls, _ = eng.leaf_state()
    assert all(0.8 < ls[p][0] < 0.9 and ls[p][2] < 0.2 for p in "ab"), "likely, not open, not sure enough"
    assert eng.assessment("act").disposition == NEEDS_INFO
    assert open_predicates(eng) == []
    q = best_question(eng)
    assert q is not None and set(q.predicates) == {"a", "b"} and q.could_change == ("act",)
    assert q.value > 0 and abs(q.value - value_of_asking(eng, ["a", "b"]).value) < 1e-12
    assert best_question(eng, max_set=1) is None, "confirming one alone leaves the other below the threshold"


def test_settled_predicates_are_not_candidates():
    from ease.planner import askable_predicates

    s = TaskSchema("t", [Predicate("a", "claim a"), Predicate("b", "claim b")],
                   [Gate("g", AND, ("a", "b"))], [Action("act", "d", "g")])
    assert askable_predicates(engine_for(s, {"a": SUPPORTS})) == ["b"]


def test_unnecessary_means_unnecessary_in_every_combination():
    """Seen on the page: a speaker's confirmation was listed as unable to change the proposal while
    the same confirmation was part of the question worth asking. Alone it changes nothing (the venue
    is open too); together with the venue it does. So it is not unnecessary."""
    s = TaskSchema("t", [Predicate("venue", "venue confirmed", prior=0.4), Predicate("ann", "Ann confirmed", prior=0.5),
                         Predicate("bo", "Bo confirmed", prior=0.5), Predicate("extra", "unrelated", prior=0.5)],
                   [Gate("some", ATLEAST, ("ann", "bo"), 1), Gate("go", AND, ("venue", "some"))],
                   [Action("announce", "d", "go", value_success=10.0, cost_failure=30.0),
                    Action("other", "d", "extra")])
    eng = engine_for(s, {})
    assert eng.assessment("announce").disposition == "NEEDS_INFO"
    assert certify_stable(eng, "announce", ["ann"]).stable, "alone, Ann's answer changes nothing"
    assert unnecessary_questions(eng)["announce"] == [], "but with the venue it does, so it is not unnecessary"
    assert unnecessary_questions(eng)["other"] == [], "extra is the only requirement of 'other' and open"
    # a predicate that truly cannot matter: an OR already satisfied by another branch
    s2 = TaskSchema("t", [Predicate("a", "claim a"), Predicate("b", "claim b")],
                    [Gate("g", OR, ("a", "b"))], [Action("act", "d", "g")])
    eng2 = engine_for(s2, {"a": SUPPORTS})
    assert unnecessary_questions(eng2) == {"act": ["b"]}
