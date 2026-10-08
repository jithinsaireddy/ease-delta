"""Runtime and HTTP API, with an oracle reader so no model is needed."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ease.aggregate import EdgeRefiner, RefinedAggregator, RefinerConfig
from ease.evolve.consolidate import Readings, save_readings
from ease.labels import REFUTES, SUPPORTS
from ease.runtime import Limits, PersistentEdgeCache, Runtime
from ease.scorer import OracleScorer
from ease.service.api import create_app

D = 8
CLAIM_A = "The client approved the final design."
CLAIM_D = "The delivery date is confirmed."
GOLD = {(CLAIM_A, "Client: approved, go ahead."): SUPPORTS,
        (CLAIM_A, "Client: we withdraw approval pending changes."): REFUTES,
        (CLAIM_D, "Delivery confirmed for 12 October."): SUPPORTS}
SCHEMA = {"task_id": "packet",
          "predicates": [{"id": "approved", "text": CLAIM_A, "prior": 0.3, "ask_cost": 0.1},
                         {"id": "date", "text": CLAIM_D}],
          "gates": [{"id": "ready", "op": "AND", "children": ["approved", "date"]}],
          "actions": [{"id": "send_packet", "description": "Send the packet", "requires": "ready",
                       "value_success": 5, "cost_failure": 10}]}


def scorer():
    s = OracleScorer(GOLD, margin=12.0, msg_dim=D)
    s.cache = {}
    return s


def refined():
    import torch
    torch.manual_seed(0)
    return RefinedAggregator(EdgeRefiner(RefinerConfig(msg_dim=D, hidden=16)))


@pytest.fixture
def client(tmp_path):
    rt = Runtime(tmp_path / "data", scorer(), refined(), tau=0.9)
    with TestClient(create_app(rt)) as c:
        c.runtime = rt
        yield c
    rt.close()


def ev(record_id, revision, text, **kw):
    return {"record_id": record_id, "revision": revision, "text": text, "source_id": "client", "authority": 2,
            "valid_from": float(revision), **kw}


def test_full_lifecycle_over_http(client):
    assert client.get("/health").json()["ok"]
    r = client.post("/tasks", json=SCHEMA)
    assert r.status_code == 201 and r.json()["assessments"]["send_packet"]["disposition"] == "NEEDS_INFO"
    assert client.post("/tasks", json=SCHEMA).status_code == 409

    r = client.post("/tasks/packet/events", json=ev("mail1", 1, "Client: approved, go ahead.")).json()
    assert r["change"]["ledger"]["outcome"] == "applied" and r["change"]["edges_scored"] == 2
    r = client.post("/tasks/packet/events", json=ev("mail2", 1, "Delivery confirmed for 12 October.")).json()
    assert r["assessments"]["send_packet"]["disposition"] == "READY"
    assert r["assessments"]["send_packet"]["needs_approval"] is True

    # the same message again, and an older copy: nothing happens
    for body in (ev("mail1", 1, "Client: approved, go ahead."), ev("mail1", 0, "an older draft")):
        r = client.post("/tasks/packet/events", json=body).json()
        assert r["change"]["propagation"]["total_recomputed"] == 0 and r["change"]["changed_actions"] == []

    # approval withdrawn
    r = client.post("/tasks/packet/events", json=ev("mail1", 2, "Client: we withdraw approval pending changes.")).json()
    assert ["send_packet", "READY", "BLOCKED"] in r["change"]["changed_actions"]
    ex = client.get("/tasks/packet/explain/send_packet").json()
    approved = next(p for p in ex["predicates"] if p["predicate"] == "approved")
    assert approved["status"] == "VIOLATED" and approved["records"][0]["record_id"] == "mail1"
    assert approved["records"][0]["revision"] == 2

    # record withdrawn altogether: the question is open again, and worth asking
    r = client.post("/tasks/packet/events", json={"record_id": "mail1", "revision": 3, "text": None}).json()
    assert r["assessments"]["send_packet"]["disposition"] == "NEEDS_INFO"
    q = client.get("/tasks/packet/question").json()
    assert q["question"]["predicates"] == ["approved"]
    assert client.get("/tasks/packet/verify").json()["not_bitwise_equal"] == 0


def test_validation_and_errors(client):
    client.post("/tasks", json=SCHEMA)
    assert client.get("/tasks/nope").status_code == 404
    assert client.post("/tasks/packet/events", json={"record_id": "x", "revision": -1, "text": "t"}).status_code == 422
    assert client.post("/tasks/packet/events", json=ev("x", 1, "t", valid_from=5.0, valid_until=1.0)).status_code == 422
    bad = dict(SCHEMA, task_id="../../etc")
    assert client.post("/tasks", json=bad).status_code == 422
    cyc = dict(SCHEMA, task_id="c", gates=[{"id": "g1", "op": "AND", "children": ["g2"]},
                                           {"id": "g2", "op": "AND", "children": ["g1"]}])
    assert client.post("/tasks", json=cyc).status_code == 422
    assert client.post("/tasks/packet/readings", json={"record_id": "a", "predicate_id": "approved",
                                                       "establishes": "maybe"}).status_code == 422
    assert client.post("/tasks/packet/readings", json={"record_id": "a", "predicate_id": "approved",
                                                       "establishes": "refutes"}).status_code == 404


def test_text_and_record_limits(tmp_path):
    rt = Runtime(tmp_path / "d", scorer(), refined(), limits=Limits(max_text_chars=50, max_records_per_task=2))
    with TestClient(create_app(rt)) as c:
        c.post("/tasks", json=SCHEMA)
        assert c.post("/tasks/packet/events", json=ev("a", 1, "x" * 51)).status_code == 422
        assert c.post("/tasks/packet/events", json=ev("a", 1, "short")).status_code == 200
        assert c.post("/tasks/packet/events", json=ev("b", 1, "short")).status_code == 200
        assert c.post("/tasks/packet/events", json=ev("c", 1, "short")).status_code == 422
        assert c.post("/tasks/packet/events", json=ev("a", 2, "revision of an existing record")).status_code == 200
    rt.close()


def test_correction_is_immediate_retractable_and_survives_restart(tmp_path):
    rt = Runtime(tmp_path / "d", scorer(), refined())
    with TestClient(create_app(rt)) as c:
        c.post("/tasks", json=SCHEMA)
        c.post("/tasks/packet/events", json=ev("mail1", 1, "Client: approved, go ahead."))
        c.post("/tasks/packet/events", json=ev("mail2", 1, "Delivery confirmed for 12 October."))
        assert c.get("/tasks/packet").json()["assessments"]["send_packet"]["disposition"] == "READY"
        r = c.post("/tasks/packet/readings", json={"record_id": "mail1", "predicate_id": "approved",
                                                   "establishes": "refutes"}).json()
        assert ["send_packet", "READY", "BLOCKED"] in r["changed_actions"]
        item = r["memory_item"]
    rt.close()

    rt2 = Runtime(tmp_path / "d", scorer(), refined())
    with TestClient(create_app(rt2)) as c:
        st = c.get("/tasks/packet").json()
        assert st["assessments"]["send_packet"]["disposition"] == "BLOCKED", "the correction survived the restart"
        assert st["records"]["mail1"]["revision"] == 1
        assert c.delete(f"/readings/{item}").json()["removed"]
        assert c.get("/tasks/packet").json()["assessments"]["send_packet"]["disposition"] == "READY"
        assert c.get("/tasks/packet/verify").json()["not_bitwise_equal"] == 0
    rt2.close()


def test_threshold_rises_after_a_wrong_endorsement(client):
    client.post("/tasks", json=SCHEMA)
    client.post("/tasks/packet/events", json=ev("mail1", 1, "Client: approved, go ahead."))
    client.post("/tasks/packet/events", json=ev("mail2", 1, "Delivery confirmed for 12 October."))
    assert client.post("/tasks/packet/verdicts", json={"action_id": "send_packet", "was_wrong": True}).status_code == 404
    e = client.post("/tasks/packet/endorsements/send_packet").json()
    assert e["endorsed"] and e["needs_approval"]
    before = e["threshold"]
    t = client.post("/tasks/packet/verdicts", json={"action_id": "send_packet", "was_wrong": True}).json()
    assert t["tau"] > before and t["errors"] == 1 and t["error_rate"] <= t["bound"]


def test_consolidation_without_a_regression_suite_changes_nothing(client):
    r = client.post("/evolve/consolidate").json()
    assert r["adopted"] is False and "regression suite" in r["reason"]


def test_consolidation_and_rollback_through_the_runtime(tmp_path):
    rng = np.random.default_rng(0)

    def synth(n, shift, tag):
        y = rng.integers(0, 3, n)
        msgs = (np.eye(3, D)[y] * 4 + rng.standard_normal((n, D))).astype(np.float32)
        lg = (np.eye(3)[y] * 3 + rng.standard_normal((n, 3))).astype(np.float32)
        lg[:, REFUTES] -= shift
        return Readings([f"{tag}{i}" for i in range(n)], y, lg, msgs)

    save_readings(tmp_path / "suite.npz", {"in_domain": synth(1200, 0.0, "r"), "replay": synth(1200, 0.0, "p")})
    rt0 = Runtime(tmp_path / "d0", scorer(), refined(), regression_suite=tmp_path / "suite.npz")
    refused = rt0.consolidate_now()
    assert refused["adopted"] is False and "regression_sets.npz" in refused["reason"]
    rt0.close()
    # a set suite: small sets of in-domain readings with their correct status under the policy
    from ease.evolve.consolidate import SetSuite, save_set_suites
    one = synth(900, 0.0, "s")
    offsets = np.arange(0, 901, 3)
    labels = []
    for i in range(len(offsets) - 1):
        ys = one.labels[offsets[i]:offsets[i + 1]]
        decisive = [int(y) for y in ys if y != 2]
        labels.append(2 if not decisive else 3 if len(set(decisive)) > 1 else (0 if decisive[0] == 0 else 1))
    save_set_suites(tmp_path / "regression_sets.npz", {"in_domain": SetSuite(
        one.logits, one.msgs, np.zeros(900, np.int64), offsets.astype(np.int64), np.asarray(labels, np.int64))})
    rt = Runtime(tmp_path / "d", scorer(), refined(), regression_suite=tmp_path / "suite.npz")
    v0 = rt.versions.current()
    fb = synth(900, 4.0, "f")
    for i in range(len(fb)):
        item = rt.memory.add(fb.pairs[i], int(fb.labels[i]), fb.msgs[i])
        rt.audit.log(kind="correct_reading", item=item, logits=fb.logits[i].tolist())
    rep = rt.consolidate_now()
    assert rep["adopted"], rep["reason"]
    assert rt.versions.current() != v0 and len(rt.info()["lineage"]) == 2
    assert rt.rollback()["version"] == v0
    rt.close()


def test_edge_cache_persists(tmp_path):
    c = PersistentEdgeCache(str(tmp_path / "e.sqlite"))
    c["k"] = (np.array([1, 2, 3], np.float32), np.arange(D, dtype=np.float32))
    c.clear()
    assert "k" in c and np.array_equal(c["k"][0], [1, 2, 3])
    again = PersistentEdgeCache(str(tmp_path / "e.sqlite"))
    assert again.stored() == 1 and np.array_equal(again["k"][1], np.arange(D))
    assert "missing" not in again


def test_ui_is_served_and_self_contained(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "default-src 'self'" in r.headers["content-security-policy"]
    html = r.text
    assert "<title>EASE-Delta</title>" in html
    for external in ("http://", "https://", "//cdn", "@import"):
        assert external not in html, f"the page must not load anything from elsewhere ({external})"
    assert "innerHTML" not in html, "record text is inserted as text, never as markup"


def test_export_feedback_leaves_out_records_that_changed_or_were_purged(tmp_path):
    import json
    rt = Runtime(tmp_path / "d", scorer(), refined())
    with TestClient(create_app(rt)) as c:
        c.post("/tasks", json=SCHEMA)
        c.post("/tasks/packet/events", json=ev("mail1", 1, "Client: approved, go ahead."))
        c.post("/tasks/packet/events", json=ev("mail2", 1, "Delivery confirmed for 12 October."))
        c.post("/tasks/packet/readings", json={"record_id": "mail1", "predicate_id": "approved", "establishes": "refutes"})
        c.post("/tasks/packet/readings", json={"record_id": "mail2", "predicate_id": "date", "establishes": "supports"})
    out = rt.export_feedback(tmp_path / "fb.jsonl")
    rows = [json.loads(l) for l in open(tmp_path / "fb.jsonl")]
    assert out["rows"] == 2 and {r["evidence"] for r in rows} == {"Client: approved, go ahead.",
                                                                  "Delivery confirmed for 12 October."}
    assert next(r for r in rows if r["claim"] == CLAIM_A)["label"] == REFUTES
    rt.engine("packet").ledger.purge("mail1")
    out = rt.export_feedback(tmp_path / "fb.jsonl")
    rows = [json.loads(l) for l in open(tmp_path / "fb.jsonl")]
    assert out["rows"] == 1 and out["skipped_because_record_changed_or_gone"] == 1
    assert all("approved, go ahead" not in r["evidence"] for r in rows), "purged text must not be exported"
    rt.close()


def test_state_names_requirements_and_actions_in_words(client):
    client.post("/tasks", json=SCHEMA)
    st = client.get("/tasks/packet").json()
    assert st["beliefs"]["approved"]["text"] == CLAIM_A and st["beliefs"]["date"]["text"] == CLAIM_D
    assert st["assessments"]["send_packet"]["description"] == "Send the packet"


def test_records_show_what_each_message_says_and_follow_edits_and_withdrawals(client):
    client.post("/tasks", json=SCHEMA)
    assert client.get("/tasks/packet/records").json() == {"records": []}
    client.post("/tasks/packet/events", json=ev("msg-1", 1, "Client: approved, go ahead."))
    assert client.get("/tasks/packet/records").json()["records"] == [
        {"record_id": "msg-1", "revision": 1, "status": "active", "text": "Client: approved, go ahead.",
         "source_id": "client", "authority": 2, "valid_from": 1.0, "valid_until": None, "span": None}]

    client.post("/tasks/packet/events", json=ev("msg-1", 2, "Client: we withdraw approval pending changes."))
    (row,) = client.get("/tasks/packet/records").json()["records"]
    assert row["revision"] == 2 and row["text"] == "Client: we withdraw approval pending changes."
    client.post("/tasks/packet/events", json={"record_id": "msg-1", "revision": 3, "text": None, "source_id": "client"})
    (row,) = client.get("/tasks/packet/records").json()["records"]
    assert (row["revision"], row["status"], row["text"]) == (3, "withdrawn", None)

    client.post("/tasks/packet/events", json=ev("msg-2", 1, "Delivery confirmed for 12 October."))
    client.post("/tasks/packet/events", json=ev("msg-2", 1, "Delivery moved to 19 October."))
    row = {r["record_id"]: r for r in client.get("/tasks/packet/records").json()["records"]}["msg-2"]
    assert row["status"] == "conflicted" and row["text"] is None
    assert sorted(v["text"] for v in row["versions"]) == ["Delivery confirmed for 12 October.", "Delivery moved to 19 October."]
    assert client.get("/tasks/nope/records").status_code == 404
