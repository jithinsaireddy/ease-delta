"""The short API: requirement expressions, task definitions, and a tracker driven by an oracle reader."""

import pytest

from ease.aggregate import Calibration, RefinedAggregator, rule_refiner
from ease.easy import EvidenceReader, Reading, Tracker, define, parse_requirement
from ease.labels import NEI, REFUTES, SUPPORTS
from ease.scorer import OracleScorer

D = 8
APPROVED = "Acme has approved the design that is being delivered."
DATE = "Acme has confirmed the delivery date."
NDA = "The non-disclosure agreement has been signed by both parties."
GOLD = {(APPROVED, "Email from Acme: we approve the design, please go ahead."): SUPPORTS,
        (APPROVED, "Email from Acme: we withdraw our approval, the colours are wrong."): REFUTES,
        (DATE, "Email from Acme: delivery on 12 March is confirmed."): SUPPORTS,
        (NDA, "Signed NDA received from Acme, countersigned by us."): SUPPORTS,
        (APPROVED, "Lunch is at noon."): NEI}


def oracle_reader(margin=12.0):
    s = OracleScorer(GOLD, margin=margin, msg_dim=D)
    s.cache = {}
    agg = RefinedAggregator(rule_refiner(Calibration(1.0, (0.0, 0.0, 0.0)), msg_dim=D, hidden=16))
    return EvidenceReader(s, agg, {"alpha": 0.05, "eta": 0.05, "tau_initial": 0.9}, Calibration(1.0, (0.0, 0.0, 0.0)))


@pytest.mark.parametrize("text,tree", [
    ("a", ("pred", "a")),
    ("a and b", ("and", [("pred", "a"), ("pred", "b")])),
    ("a AND b or c", ("or", [("and", [("pred", "a"), ("pred", "b")]), ("pred", "c")])),
    ("a and (b or c)", ("and", [("pred", "a"), ("or", [("pred", "b"), ("pred", "c")])])),
    ("not a", ("not", ("pred", "a"))),
    ("at least 2 of (a, b, c)", ("atleast", 2, [("pred", "a"), ("pred", "b"), ("pred", "c")])),
])
def test_requirement_expressions_parse(text, tree):
    assert parse_requirement(text, ["a", "b", "c"]) == tree


@pytest.mark.parametrize("bad", ["a and", "a b", "(a", "d", "at least 4 of (a, b, c)", "at least x of (a)", "a and and b", ""])
def test_malformed_requirements_are_refused_with_a_reason(bad):
    with pytest.raises(ValueError):
        parse_requirement(bad, ["a", "b", "c"])


def test_define_builds_a_valid_schema_and_flattens_chains():
    from ease.schema import TaskSchema

    d = define("t", {"a": "A holds.", "b": "B holds.", "c": "C holds."},
               {"go": "a and b and c", "maybe": {"requires": "at least 2 of (a, b, c) and not c", "value": 3, "cost": 1}})
    s = TaskSchema.from_dict(d)
    go = next(a for a in s.actions if a.id == "go")
    gate = next(g for g in s.gates if g.id == go.requires)
    assert gate.op == "AND" and set(gate.children) == {"a", "b", "c"}, "a chain of ANDs becomes one gate"
    maybe = next(a for a in s.actions if a.id == "maybe")
    assert maybe.value_success == 3 and maybe.cost_failure == 1
    assert s.leaves_of(maybe.requires).count("c") == 2


def test_reader_reads_with_the_systems_calibration():
    r = oracle_reader().read(APPROVED, "Email from Acme: we approve the design, please go ahead.")
    assert isinstance(r, Reading) and r.label == "SUPPORTS" and r.supports > 0.99
    assert "SUPPORTS" in repr(r)


def test_tracker_follows_a_hand_off_through_approval_withdrawal_and_correction():
    reader = oracle_reader()
    with Tracker.define({"approved": APPROVED, "date": DATE, "nda": NDA},
                        {"send_packet": {"requires": "approved and date and nda", "description": "Send the packet"}},
                        reader, task_id="handoff") as t:
        assert t.status().needs_info == ["send_packet"]
        t.add("mail-1", "Email from Acme: we approve the design, please go ahead.")
        t.add("mail-2", "Email from Acme: delivery on 12 March is confirmed.")
        u = t.add("nda", "Signed NDA received from Acme, countersigned by us.")
        assert ("send_packet", "NEEDS_INFO", "READY") in u.changed
        st = t.status()
        assert st.ready == ["send_packet"] and st.requirements["approved"].rests_on == ["mail-1"]
        # a new revision of the same record replaces it
        u = t.update("mail-1", "Email from Acme: we withdraw our approval, the colours are wrong.")
        assert ("send_packet", "READY", "BLOCKED") in u.changed and t.records()["mail-1"]["revision"] == 2
        assert "withdraw our approval" in t.explain("send_packet")
        # a person says the reader got it wrong; undoing restores the earlier state exactly
        before = t.status().requirements["approved"]
        cid = t.correct("mail-1", "approved", "supports")
        assert t.status().ready == ["send_packet"]
        assert t.undo(cid) and t.status().requirements["approved"] == before
        # withdrawing the record leaves the requirement open again
        u = t.withdraw("mail-1")
        assert t.status().requirements["approved"].status == "UNRESOLVED"
        assert t.status().question and APPROVED in t.status().question
        assert t.verify()
        assert "send_packet" in str(t.status())


def test_tracker_keeps_its_state_in_a_directory_and_reopens_it(tmp_path):
    reader = oracle_reader()
    t = Tracker.define({"approved": APPROVED}, {"go": "approved"}, reader, task_id="keep", data_dir=tmp_path)
    t.add("mail-1", "Email from Acme: we approve the design, please go ahead.")
    t.close()
    again = Tracker.define({"approved": APPROVED}, {"go": "approved"}, reader, task_id="keep", data_dir=tmp_path)
    assert again.status().ready == ["go"] and again.records()["mail-1"]["revision"] == 1
    again.close()


def test_templates_work_through_the_tracker():
    t = Tracker.from_template("client-handoff", reader=oracle_reader(), client="Acme")
    st = t.status()
    assert "send_packet" in st.actions and st.requirements["design_approved"].text.startswith("Acme has approved")
    t.close()


def test_unknown_names_fail_clearly():
    t = Tracker.define({"a": APPROVED}, {"go": "a"}, oracle_reader())
    with pytest.raises(KeyError):
        t.withdraw("never-added")
    with pytest.raises(ValueError):
        t.correct("x", "a", "maybe")
    t.close()


def test_import_ease_is_light_and_exposes_the_short_api():
    import subprocess
    import sys

    code = "import sys, ease; assert 'torch' not in sys.modules; from ease import Tracker, EvidenceReader; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr


def test_a_confirmation_is_exact_and_a_later_message_can_still_overturn_it():
    reader = oracle_reader(margin=2.0)  # hesitant readings, like real ones on email text
    with Tracker.define({"approved": APPROVED, "date": DATE}, {"go": "approved and date"}, reader) as t:
        t.add("mail-1", "Email from Acme: we approve the design, please go ahead.", about=["approved"])
        t.add("mail-2", "Email from Acme: delivery on 12 March is confirmed.", about=["date"])
        assert t.status().needs_info == ["go"], "two likely requirements are not a sure thing at 0.9"
        t.confirm("approved", by="Dana at Acme")
        u = t.confirm("date", by="Dana at Acme")
        assert ("go", "NEEDS_INFO", "READY") in u.changed
        assert t.status().requirements["approved"].satisfied > 0.99
        assert "Dana at Acme confirmed that this holds" in t.records()["confirmation-approved-1"]["text"]
        t.add("mail-3", "Email from Acme: we withdraw our approval, the colours are wrong.", about=["approved"])
        assert t.status().blocked == ["go"], "the later message wins at equal authority"
        with pytest.raises(KeyError):
            t.confirm("nonexistent", by="x")
