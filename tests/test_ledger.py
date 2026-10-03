"""Ledger properties, checked on randomly generated event sets.

Each test names the property from ease/ledger.py it checks.
"""

import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ease.ledger import Event, Ledger, Outcome, Status, merge, replay

record_ids = st.sampled_from(["a", "b", "c", "d"])
texts = st.one_of(st.none(), st.sampled_from(["x", "y", "z", "approved v4", "approved v5"]))
events = st.builds(
    Event,
    record_id=record_ids,
    revision=st.integers(min_value=0, max_value=6),
    text=texts,
    source_id=st.sampled_from(["s1", "s2"]),
    authority=st.integers(min_value=0, max_value=2),
)
event_lists = st.lists(events, min_size=0, max_size=30)


def canon(state):
    return {rid: (s.revision, tuple(p.key for p in s.payloads)) for rid, s in sorted(state.items())}


@settings(max_examples=300, deadline=None)
@given(event_lists, st.randoms(use_true_random=False))
def test_P2_delivery_order_does_not_matter(evs, rnd):
    shuffled = list(evs)
    rnd.shuffle(shuffled)
    assert canon(replay(evs)) == canon(replay(shuffled))


@settings(max_examples=300, deadline=None)
@given(event_lists, st.randoms(use_true_random=False))
def test_P1_duplicates_do_not_matter(evs, rnd):
    noisy = list(evs) + [rnd.choice(evs) for _ in range(len(evs))] if evs else []
    rnd.shuffle(noisy)
    assert canon(replay(evs)) == canon(replay(noisy))


@settings(max_examples=300, deadline=None)
@given(event_lists, events)
def test_P1_redelivery_is_a_noop(evs, extra):
    state = replay(evs + [extra])
    again, outcome = merge(state[extra.record_id], extra)
    assert again == state[extra.record_id]
    assert outcome in (Outcome.DUPLICATE, Outcome.STALE)


@settings(max_examples=300, deadline=None)
@given(event_lists, events)
def test_P3_lower_revision_never_changes_a_record(evs, late):
    state = replay(evs)
    cur = state.get(late.record_id)
    if cur is None or late.revision >= cur.revision:
        return
    after, outcome = merge(cur, late)
    assert after == cur and outcome is Outcome.STALE


def test_P3_stale_copy_cannot_undo_a_withdrawal():
    led = Ledger()
    led.put("approval", 1, "client approves design")
    led.withdraw("approval", 2)
    r = led.put("approval", 1, "client approves design")
    assert r.outcome is Outcome.STALE and not r.changed
    assert led.status("approval") is Status.WITHDRAWN
    assert "approval" not in led.active()


def test_P4_same_revision_conflict_is_order_independent_and_blocks_use():
    a = Event("date", 3, "October 12")
    b = Event("date", 3, "October 19")
    s1, s2 = replay([a, b]), replay([b, a])
    assert canon(s1) == canon(s2)
    assert s1["date"].stored_status is Status.CONFLICTED
    led = Ledger()
    assert led.deliver(a).outcome is Outcome.APPLIED
    assert led.deliver(b).outcome is Outcome.CONFLICT
    assert "date" not in led.active(), "a conflicted record must not be used as evidence"
    led.put("date", 4, "October 19")
    assert led.status("date") is Status.ACTIVE and led.active()["date"].text == "October 19"


def test_higher_revision_replaces_and_reactivates():
    led = Ledger()
    led.put("r", 1, "v1")
    assert led.put("r", 2, "v2").outcome is Outcome.APPLIED
    led.withdraw("r", 3)
    assert led.status("r") is Status.WITHDRAWN
    assert led.put("r", 4, "v4").changed
    assert led.active()["r"].text == "v4"


def test_arrival_time_is_not_authority():
    led = Ledger()
    led.put("doc", 5, "newer authorised text", observed_at=100.0)
    r = led.put("doc", 2, "older text that arrived later", observed_at=999.0)
    assert r.outcome is Outcome.STALE
    assert led.active()["doc"].text == "newer authorised text"


def test_metadata_and_arrival_time_do_not_create_conflicts():
    led = Ledger()
    led.put("r", 1, "same", observed_at=1.0, meta={"via": "email"})
    r = led.put("r", 1, "same", observed_at=2.0, meta={"via": "upload"})
    assert r.outcome is Outcome.DUPLICATE and not r.changed


def test_validity_window_uses_the_supplied_clock():
    led = Ledger()
    led.put("quote", 1, "price valid this week", valid_from=10.0, valid_until=20.0)
    assert led.status("quote", now=5.0) is Status.NOT_YET_VALID
    assert led.status("quote", now=10.0) is Status.ACTIVE
    assert led.status("quote", now=19.999) is Status.ACTIVE
    assert led.status("quote", now=20.0) is Status.EXPIRED
    assert "quote" not in led.active(now=25.0)
    assert led.next_transition_after(0.0) == 10.0
    assert led.next_transition_after(10.0) == 20.0
    assert led.next_transition_after(20.0) is None


def test_fingerprint_changes_when_status_changes_by_clock_alone():
    led = Ledger()
    led.put("q", 1, "t", valid_until=20.0)
    st_ = led.get("q")
    assert st_.fingerprint(now=1.0) != st_.fingerprint(now=21.0)
    assert st_.fingerprint(now=1.0) == st_.fingerprint(now=2.0)


@settings(max_examples=60, deadline=None)
@given(event_lists)
def test_state_survives_restart(evs):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.sqlite")
        led = Ledger(path)
        for e in evs:
            led.deliver(e)
        snap = led.snapshot()
        led.close()
        reopened = Ledger(path)
        assert reopened.snapshot() == snap
        assert canon(replay(evs)) == {k: v for k, v in snap.items()}
        reopened.close()


def test_purge_withdraws_and_erases_text_and_survives_restart():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.sqlite")
        led = Ledger(path)
        led.put("pii", 1, "home address 12 Example Road")
        led.put("pii", 2, "home address 14 Example Road")
        led.put("other", 1, "unrelated")
        led.purge("pii")
        assert led.status("pii") is Status.WITHDRAWN
        rows = led._db.execute("SELECT text FROM events WHERE record_id='pii' AND text IS NOT NULL").fetchall()
        assert rows and all(t == "" for (t,) in rows), "no stored copy of the text may remain"
        snap = led.snapshot()
        led.close()
        reopened = Ledger(path)
        assert reopened.snapshot() == snap
        assert reopened.status("pii") is Status.WITHDRAWN
        assert reopened.active()["other"].text == "unrelated"
        reopened.close()


def test_tasks_are_isolated_in_one_database():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.sqlite")
        a, b = Ledger(path, "task-a"), Ledger(path, "task-b")
        a.put("r", 1, "for a")
        b.put("r", 7, "for b")
        assert a.active()["r"].text == "for a" and b.active()["r"].text == "for b"
        a.close(); b.close()


def test_invalid_events_are_rejected():
    with pytest.raises(ValueError):
        Event("", 1, "x")
    with pytest.raises(ValueError):
        Event("r", -1, "x")
    with pytest.raises(ValueError):
        Event("r", 1, "x", valid_from=5.0, valid_until=4.0)
    with pytest.raises(ValueError):
        Ledger().put("r", 1, None)


def test_why_the_ledger_keeps_records_instead_of_a_summary():
    """The information argument behind the design (docs/PROOFS.md, L0).

    Two histories share the same compressed state (the sum) and receive the same correction, yet
    require different answers. An updater that sees only the compressed state and the correction
    gets identical inputs in both cases, so it returns one answer and is wrong at least once.
    Keeping addressable records removes the ambiguity.
    """
    h1 = {"r1": 1, "r2": 2}
    h2 = {"r1": 2, "r2": 1}
    compress = lambda h: sum(h.values())
    correction = ("delete", "r1")
    assert (compress(h1), correction) == (compress(h2), correction), "updater inputs are identical"
    truth1 = sum(v for k, v in h1.items() if k != "r1")
    truth2 = sum(v for k, v in h2.items() if k != "r1")
    assert truth1 != truth2, "required outputs differ"
    for candidate_answer in range(0, 4):  # any single answer fails on at least one history
        assert not (candidate_answer == truth1 and candidate_answer == truth2)

    led1, led2 = Ledger(), Ledger()
    for rid, v in h1.items():
        led1.put(rid, 1, str(v))
    for rid, v in h2.items():
        led2.put(rid, 1, str(v))
    led1.withdraw("r1", 2); led2.withdraw("r1", 2)
    assert sum(int(p.text) for p in led1.active().values()) == truth1
    assert sum(int(p.text) for p in led2.active().values()) == truth2
