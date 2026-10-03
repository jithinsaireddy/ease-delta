"""Workspaces behind one model: keys, isolation, quotas, templates, imports, confirmations, backups.

The reader is an oracle, so no model is needed.
"""

import base64
import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ease.aggregate import Calibration, RefinedAggregator, rule_refiner
from ease.importers import parse_eml, parse_file, parse_mbox
from ease.labels import NEI, REFUTES, SUPPORTS
from ease.scorer import OracleScorer
from ease.service.api import create_app
from ease.service.workspaces import AuthError, Quota, Service
from ease.templates import TEMPLATES, render

D = 8
CLAIM_A = "The client approved the final design."
CLAIM_D = "The delivery date is confirmed."
GOLD = {(CLAIM_A, "Client: approved, go ahead."): SUPPORTS,
        (CLAIM_A, "Client: we withdraw approval pending changes."): REFUTES,
        (CLAIM_D, "Delivery confirmed for 12 October."): SUPPORTS,
        (CLAIM_A, "Email from Dana Ortiz on 2026-03-02, subject: Design v5: Approved, please proceed with v5."): SUPPORTS,
        (CLAIM_D, "Email from Dana Ortiz on 2026-03-02, subject: Design v5: Approved, please proceed with v5."): NEI}
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


def rules():
    return RefinedAggregator(rule_refiner(Calibration(1.0, (0.0, 0.0, 0.0)), msg_dim=D, hidden=16))


@pytest.fixture
def multi(tmp_path):
    svc = Service(tmp_path / "svc", scorer(), rules(), local=False, admin_key="admin-secret")
    _, key_a = svc.create_workspace("acme", "Acme agency")
    _, key_b = svc.create_workspace("bolt", "Bolt events")
    with TestClient(create_app(svc)) as c:
        c.svc, c.key_a, c.key_b = svc, key_a, key_b
        yield c
    svc.close()


def auth(key):
    return {"authorization": f"Bearer {key}"}


def ev(record_id, revision, text, **kw):
    return {"record_id": record_id, "revision": revision, "text": text, "source_id": "client", "authority": 2,
            "valid_from": float(revision), **kw}


def test_keys_are_required_hashed_and_rotatable(multi):
    c = multi
    assert c.get("/health").json()["mode"] == "workspaces"
    assert c.get("/tasks").status_code == 401
    assert c.get("/tasks", headers=auth("ease_acme_wrong")).status_code == 401
    assert c.get("/tasks", headers=auth(c.key_a)).json() == {"tasks": []}
    stored = json.loads((c.svc.dir / "workspaces" / "acme" / "workspace.json").read_text())
    assert c.key_a not in json.dumps(stored), "the key itself is never stored"
    with pytest.raises(AuthError):
        c.svc.authenticate("ease_nobody_" + "0" * 48)
    new = c.post("/admin/workspaces/acme/rotate-key", headers=auth("admin-secret")).json()["key"]
    assert c.get("/tasks", headers=auth(c.key_a)).status_code == 401
    assert c.get("/tasks", headers=auth(new)).status_code == 200
    assert c.post("/admin/workspaces", json={"id": "cora", "name": "Cora"}, headers=auth(c.key_b)).status_code == 401


def test_workspaces_are_isolated_including_corrections(multi):
    c = multi
    for key in (c.key_a, c.key_b):
        assert c.post("/tasks", json=SCHEMA, headers=auth(key)).status_code == 201
        c.post("/tasks/packet/events", json=ev("mail1", 1, "Client: approved, go ahead."), headers=auth(key))
        c.post("/tasks/packet/events", json=ev("mail2", 1, "Delivery confirmed for 12 October."), headers=auth(key))
        assert c.get("/tasks/packet", headers=auth(key)).json()["assessments"]["send_packet"]["disposition"] == "READY"
    # a correction in acme changes acme and nothing in bolt
    r = c.post("/tasks/packet/readings", json={"record_id": "mail1", "predicate_id": "approved", "establishes": "refutes"},
               headers=auth(c.key_a)).json()
    assert ["send_packet", "READY", "BLOCKED"] in r["changed_actions"]
    assert c.get("/tasks/packet", headers=auth(c.key_b)).json()["assessments"]["send_packet"]["disposition"] == "READY"
    assert c.get("/info", headers=auth(c.key_b)).json()["memory_items"] == 0
    assert c.get("/info", headers=auth(c.key_a)).json()["memory_items"] == 1
    # bolt cannot retract acme's correction
    assert c.delete(f"/readings/{r['memory_item']}", headers=auth(c.key_b)).json()["removed"] is False
    assert c.get("/tasks/packet", headers=auth(c.key_a)).json()["assessments"]["send_packet"]["disposition"] == "BLOCKED"
    # the files live apart
    assert (c.svc.dir / "workspaces" / "acme" / "tasks" / "packet" / "ledger.sqlite").exists()
    assert (c.svc.dir / "workspaces" / "bolt" / "tasks" / "packet" / "ledger.sqlite").exists()
    assert (c.svc.dir / "edge_cache.sqlite").exists(), "readings are shared; they hold no text"


def test_quotas_count_per_workspace_and_per_day(tmp_path):
    svc = Service(tmp_path / "svc", scorer(), rules(), local=False)
    ws, key = svc.create_workspace("tiny", "Tiny", Quota(max_events_per_day=2, max_tasks=1, max_requests_per_minute=1000))
    _, other = svc.create_workspace("big", "Big")
    with TestClient(create_app(svc)) as c:
        assert c.post("/tasks", json=SCHEMA, headers=auth(key)).status_code == 201
        assert c.post("/tasks", json=dict(SCHEMA, task_id="second"), headers=auth(key)).status_code == 422
        assert c.post("/tasks/packet/events", json=ev("a", 1, "x"), headers=auth(key)).status_code == 200
        assert c.post("/tasks/packet/events", json=ev("b", 1, "x"), headers=auth(key)).status_code == 200
        r = c.post("/tasks/packet/events", json=ev("c", 1, "x"), headers=auth(key))
        assert r.status_code == 429 and "daily limit" in r.json()["detail"]
        assert c.get("/workspace", headers=auth(key)).json()["usage"]["events"] == 2
        assert c.post("/tasks", json=SCHEMA, headers=auth(other)).status_code == 201
        assert c.post("/tasks/packet/events", json=ev("c", 1, "x"), headers=auth(other)).status_code == 200
        assert c.put("/admin/workspaces/tiny/quota", json={"changes": {"max_events_per_day": 5}}).status_code == 401
    svc.close()


def test_rate_limit_answers_429(tmp_path):
    svc = Service(tmp_path / "svc", scorer(), rules(), local=False)
    _, key = svc.create_workspace("slow", "Slow", Quota(max_requests_per_minute=3))
    with TestClient(create_app(svc)) as c:
        codes = [c.get("/tasks", headers=auth(key)).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    svc.close()


def test_templates_make_valid_tasks_from_forms(multi):
    c = multi
    listed = c.get("/templates").json()["templates"]
    assert {t["id"] for t in listed} == set(TEMPLATES)
    r = c.post("/tasks/from-template", json={"template": "event-coordination", "task_id": "launch-day",
                                             "params": {"speakers": ["Ann Lee", "Bo Chen", "Cy Dee"], "min_speakers": 2}},
               headers=auth(c.key_a))
    assert r.status_code == 201
    d = r.json()["definition"]
    assert sum(p["id"].startswith("speaker_") for p in d["predicates"]) == 3
    assert next(g for g in d["gates"] if g["id"] == "enough_speakers")["k"] == 2
    assert "launch-day" in c.get("/tasks", headers=auth(c.key_a)).json()["tasks"]
    assert c.post("/tasks/from-template", json={"template": "nope", "task_id": "x"}, headers=auth(c.key_a)).status_code == 404
    for tid in TEMPLATES:
        render(tid, "t", {})  # every template renders with defaults


EML = b"""From: Dana Ortiz <dana@example.com>
To: team@example.com
Subject: Design v5
Date: Mon, 02 Mar 2026 10:15:00 +0000
Message-ID: <abc123@example.com>
Content-Type: text/plain; charset="utf-8"

Approved, please proceed with v5.

On Sun, 1 Mar 2026, team@example.com wrote:
> Here is version 5 for your review.
> Thanks
"""


def test_email_import_attributes_and_drops_quoted_text(multi):
    msgs = parse_eml(EML)
    assert len(msgs) == 1
    m = msgs[0]
    assert m.attribution == "Email from Dana Ortiz on 2026-03-02, subject: Design v5"
    assert m.text == "Approved, please proceed with v5." and "version 5 for your review" not in m.text
    assert m.doc_id == "abc123@example.com" and abs(m.valid_from - 1772446500.0) < 1
    box = b"From dana@example.com Mon Mar  2 10:15:00 2026\n" + EML + b"\nFrom dana@example.com Tue Mar  3 09:00:00 2026\n" + \
        EML.replace(b"abc123", b"def456").replace(b"Approved, please proceed with v5.", b"One more change to the footer.")
    assert [m.doc_id for m in parse_mbox(box)] == ["abc123@example.com", "def456@example.com"]
    with pytest.raises(ValueError):
        parse_file(b"x", "notes.pdf")

    c = multi
    c.post("/tasks", json=SCHEMA, headers=auth(c.key_a))
    r = c.post("/tasks/packet/import", json={"filename": "approval.eml", "content_base64": base64.b64encode(EML).decode()},
               headers=auth(c.key_a))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["messages"][0]["doc_id"] == "abc123@example.com" and out["messages"][0]["passages"] == 1
    st = c.get("/tasks/packet", headers=auth(c.key_a)).json()
    assert st["beliefs"]["approved"]["status"] == "SATISFIED" and "abc123@example.com#0" in st["records"]
    assert st["beliefs"]["date"]["status"] == "UNRESOLVED"
    assert c.post("/tasks/packet/import", json={"filename": "a.eml"}, headers=auth(c.key_a)).status_code == 422


def test_confirmation_link_is_single_use_and_read_exactly(multi):
    c = multi
    c.post("/tasks", json=SCHEMA, headers=auth(c.key_a))
    c.post("/tasks/packet/events", json=ev("mail2", 1, "Delivery confirmed for 12 October."), headers=auth(c.key_a))
    assert c.get("/tasks/packet", headers=auth(c.key_a)).json()["assessments"]["send_packet"]["disposition"] == "NEEDS_INFO"
    r = c.post("/tasks/packet/confirmations", json={"predicate_id": "approved", "to_name": "Dana Ortiz"}, headers=auth(c.key_a))
    assert r.status_code == 201
    link = r.json()
    assert link["asks"] == CLAIM_A and link["path"].startswith("/confirm/acme/")
    assert c.get("/tasks/packet/confirmations", headers=auth(c.key_b)).status_code == 404, "bolt has no such task"
    page = c.get(link["path"])
    assert page.status_code == 200 and CLAIM_A in page.text and "Dana Ortiz" in page.text
    assert "packet" not in page.text and "mail2" not in page.text, "only the question is shared"
    # the reviewer answers without any key
    a = c.post(link["path"], json={"answer": "yes", "note": "confirmed on the call"})
    assert a.status_code == 200 and a.json()["applied"]["record_id"].startswith("confirmation-")
    st = c.get("/tasks/packet", headers=auth(c.key_a)).json()
    assert st["assessments"]["send_packet"]["disposition"] == "READY"
    assert st["beliefs"]["approved"]["probabilities"]["SATISFIED"] > 0.99, "a person's answer is exact, not interpreted"
    assert c.post(link["path"], json={"answer": "no"}).status_code == 422, "single use"
    assert c.get("/confirm/acme/not-a-token").status_code == 404
    assert c.post("/confirm/acme/not-a-token", json={"answer": "yes"}).status_code == 404
    listed = c.get("/tasks/packet/confirmations", headers=auth(c.key_a)).json()["confirmations"]
    assert listed[0]["answer"] == "yes" and "token" not in listed[0]
    # a "no" blocks; an "unsure" changes nothing
    r2 = c.post("/tasks/packet/confirmations", json={"predicate_id": "date", "to_name": "Lee"}, headers=auth(c.key_a)).json()
    assert c.post(r2["path"], json={"answer": "no"}).status_code == 200
    assert c.get("/tasks/packet", headers=auth(c.key_a)).json()["beliefs"]["date"]["status"] == "VIOLATED"
    r3 = c.post("/tasks/packet/confirmations", json={"predicate_id": "date", "to_name": "Kim"}, headers=auth(c.key_a)).json()
    before = c.get("/tasks/packet", headers=auth(c.key_a)).json()["beliefs"]
    assert c.post(r3["path"], json={"answer": "unsure"}).json()["applied"] is None
    assert c.get("/tasks/packet", headers=auth(c.key_a)).json()["beliefs"] == before
    assert c.get("/tasks/packet/verify", headers=auth(c.key_a)).json()["not_bitwise_equal"] == 0


def test_backup_is_a_consistent_copy_while_the_service_runs(multi, tmp_path):
    c = multi
    c.post("/tasks", json=SCHEMA, headers=auth(c.key_a))
    c.post("/tasks/packet/events", json=ev("mail1", 1, "Client: approved, go ahead."), headers=auth(c.key_a))
    out = c.post("/admin/backup", json={"out_dir": str(tmp_path / "bk")}, headers=auth("admin-secret")).json()
    copied = tmp_path / "bk"
    ledgers = list(copied.rglob("ledger.sqlite"))
    assert out["files"] >= 4 and len(ledgers) == 1
    db = sqlite3.connect(ledgers[0])
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    db.close()
    assert not list(copied.rglob("*.sqlite-wal"))
    assert c.post("/admin/backup", json={}).status_code == 401


def test_local_mode_needs_no_key_and_keeps_the_old_layout(tmp_path):
    svc = Service(tmp_path / "one", scorer(), rules(), local=True)
    with TestClient(create_app(svc)) as c:
        assert c.get("/health").json()["mode"] == "local"
        assert c.post("/tasks", json=SCHEMA).status_code == 201
        assert c.get("/tasks").json()["tasks"] == ["packet"]
        assert c.get("/workspace").json()["id"] == "local"
    assert (tmp_path / "one" / "tasks" / "packet" / "schema.json").exists()
    assert not (tmp_path / "one" / "workspaces").exists()
    svc.close()


def test_webhook_is_signed_and_never_blocks(multi, monkeypatch):
    import hashlib
    import hmac
    import threading
    import urllib.request

    c = multi
    sent, done = {}, threading.Event()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=10):
        sent["url"], sent["body"], sent["sig"] = req.full_url, req.data, req.get_header("X-ease-signature")
        done.set()
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert c.put("/workspace/webhook", json={"url": "http://127.0.0.1/hook"}, headers=auth(c.key_a)).status_code == 422
    secret = c.put("/workspace/webhook", json={"url": "https://example.com/hook"}, headers=auth(c.key_a)).json()["secret"]
    c.post("/tasks", json=SCHEMA, headers=auth(c.key_a))
    c.post("/tasks/packet/events", json=ev("mail1", 1, "Client: approved, go ahead."), headers=auth(c.key_a))
    c.post("/tasks/packet/events", json=ev("mail2", 1, "Delivery confirmed for 12 October."), headers=auth(c.key_a))
    assert done.wait(5), "the webhook was called"
    assert sent["url"] == "https://example.com/hook"
    assert sent["sig"] == "sha256=" + hmac.new(secret.encode(), sent["body"], hashlib.sha256).hexdigest()
    payload = json.loads(sent["body"])
    assert payload["task"] == "packet" and ["send_packet", "NEEDS_INFO", "READY"] in payload["changed_actions"]
    assert np.isfinite(payload["at"])
