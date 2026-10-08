"""The path from a misreading issue to the public evaluation file, and the scoring of a reader on it.

No network: issues are given as the JSON `gh issue list` prints, and the reader is a stub.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("misreadings", ROOT / "scripts" / "misreadings.py")
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)

FORM = yaml.safe_load((ROOT / ".github" / "ISSUE_TEMPLATE" / "misreading.yml").read_text())
FIELDS = {e["id"]: e for e in FORM["body"] if "id" in e}

# An issue body exactly as GitHub writes it for this form: one "### <label>" heading per field, the answer
# below it, and "_No response_" for an optional field left empty. The intro text is not submitted.
BODY = """### Requirement (claim)

The client has approved the final design.

### Message or passage

Email from the client: we like the direction, but please hold off until the colours are revised.

### What it said

supports

### What it should have said

refutes

### Model

_No response_

### May this example be added to a public evaluation set (CC BY 4.0)?

Yes"""


def issue(number, body=BODY, labels=(), login="reporter", created="2026-10-14T15:03:22Z"):
    return {"number": number, "url": f"https://github.com/jithinsaireddy/ease-delta/issues/{number}",
            "author": {"login": login}, "createdAt": created,
            "labels": [{"name": n} for n in ("misreading", *labels)], "body": body}


def answer(body, heading, value):
    """The same body with one answer changed."""
    old = mr.sections(body)[heading] or mr.NO_RESPONSE
    head = f"### {heading}\n\n"
    return body.replace(head + old, head + value, 1)


def test_headings_are_the_forms_labels():
    assert {k: v["attributes"]["label"] for k, v in FIELDS.items()} == mr.HEADINGS


def test_every_label_option_maps_to_a_reader_label():
    for field in ("got", "should"):
        options = FIELDS[field]["attributes"]["options"]
        assert {mr.to_label(o) for o in options} == {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}
    assert mr.to_label("something else") is None


def test_only_the_yes_option_is_consent():
    options = FIELDS["licence"]["attributes"]["options"]
    assert [mr.consented(o) for o in options] == [True, False]
    assert not mr.consented("")


def test_a_report_becomes_a_row():
    rows, counts = mr.build([issue(12)])
    assert counts["public"] == 1
    assert rows == [{
        "id": "gh-12", "issue": 12, "url": "https://github.com/jithinsaireddy/ease-delta/issues/12",
        "author": "reporter", "created": "2026-10-14",
        "claim": "The client has approved the final design.",
        "passage": "Email from the client: we like the direction, but please hold off until the colours are revised.",
        "reported_label": "SUPPORTS", "gold_label": "REFUTES", "model": "", "licence": "CC-BY-4.0"}]


def test_windows_line_endings_and_a_pasted_heading():
    pasted = answer(BODY, mr.HEADINGS["message"], "First line.\n### notes\nSecond line.")
    rows, _ = mr.build([issue(3, pasted.replace("\n", "\r\n"))])
    assert rows[0]["passage"] == "First line.\n### notes\nSecond line."
    assert rows[0]["claim"] == "The client has approved the final design."


def test_without_consent_nothing_is_kept(tmp_path):
    secret = "Email from Globex: the merger closes Friday."
    body = answer(answer(BODY, mr.HEADINGS["message"], secret), mr.HEADINGS["licence"],
                  "No, use it only to understand the problem")
    rows, counts = mr.build([issue(7, body)])
    assert rows == [] and counts["no_consent"] == 1
    src = tmp_path / "issues.json"
    src.write_text(json.dumps([issue(7, body), issue(8)]))
    out = tmp_path / "evals" / "misreadings.jsonl"
    assert mr.main(["sync", "--issues-json", str(src), "--out", str(out)]) == 0
    assert secret not in out.read_text() and [json.loads(x)["issue"] for x in out.read_text().splitlines()] == [8]


@pytest.mark.parametrize("why, body, labels", [
    ("kept_out", BODY, ("invalid",)),
    ("kept_out", BODY, ("duplicate",)),
    ("same_label", answer(BODY, mr.HEADINGS["should"], "supports"), ()),
    ("unreadable", answer(BODY, mr.HEADINGS["requirement"], ""), ()),
    ("unreadable", answer(BODY, mr.HEADINGS["got"], "it was wrong"), ()),
])
def test_reports_that_stay_out(why, body, labels):
    rows, counts = mr.build([issue(5, body, labels)])
    assert rows == [] and counts[why] == 1


def test_a_case_reported_twice_is_kept_once_and_a_contested_case_not_at_all():
    again = answer(BODY, mr.HEADINGS["message"],
                   "Email from the client:  we like the direction, but please hold off until the colours are revised.")
    rows, counts = mr.build([issue(30, again), issue(4)])
    assert [r["issue"] for r in rows] == [4] and counts["repeat"] == 1
    contested = answer(BODY, mr.HEADINGS["should"], "not enough info")
    rows, counts = mr.build([issue(4), issue(9, contested)])
    assert rows == [] and counts["conflict"] == 2


def test_sync_is_idempotent_and_follows_the_issues(tmp_path, capsys):
    src, out = tmp_path / "issues.json", tmp_path / "misreadings.jsonl"
    other = answer(BODY, mr.HEADINGS["message"], "Email from the client: please wait for the revised palette.")

    def sync(*issues, dry=False):
        src.write_text(json.dumps(list(issues)))
        mr.main(["sync", "--issues-json", str(src), "--out", str(out)] + (["--dry-run"] if dry else []))
        return capsys.readouterr().out

    assert "nothing written" in sync() and not out.exists()
    assert "would write 2" in sync(issue(9, other), issue(2), dry=True) and not out.exists()
    assert "wrote 2" in sync(issue(9, other), issue(2))
    first = out.read_text()
    assert [json.loads(x)["issue"] for x in first.splitlines()] == [2, 9]
    assert "unchanged, 2" in sync(issue(2), issue(9, other)) and out.read_text() == first
    edited = answer(BODY, mr.HEADINGS["requirement"], "The client has approved the revised design.")
    sync(issue(2, edited), issue(9, other))
    assert json.loads(out.read_text().splitlines()[0])["claim"] == "The client has approved the revised design."
    withdrawn = answer(other, mr.HEADINGS["licence"], "No, use it only to understand the problem")
    sync(issue(2), issue(9, withdrawn))
    assert [json.loads(x)["issue"] for x in out.read_text().splitlines()] == [2]


class StubTokenizer:
    """Encodes each pair as the index of the answer the stub reader should give, and records the order."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, first, second, **kw):
        self.calls.append((list(first), list(second), kw))
        return {"input_ids": torch.tensor([[self.answers[p]] for p in second])}


class StubReader:
    config = SimpleNamespace(id2label={"0": "SUPPORTS", "1": "REFUTES", "2": "NOT_ENOUGH_INFO"})

    def __call__(self, input_ids):
        return SimpleNamespace(logits=4.0 * torch.nn.functional.one_hot(input_ids[:, 0], 3).float())


def test_scoring_reads_claim_first_and_counts_against_the_expected_answer():
    rows = [{"id": "gh-1", "claim": "c1", "passage": "p1", "reported_label": "SUPPORTS", "gold_label": "REFUTES"},
            {"id": "gh-2", "claim": "c2", "passage": "p2", "reported_label": "REFUTES", "gold_label": "NOT_ENOUGH_INFO"},
            {"id": "gh-3", "claim": "c3", "passage": "p3", "reported_label": "SUPPORTS", "gold_label": "NOT_ENOUGH_INFO"}]
    tok = StubTokenizer({"p1": 1, "p2": 1, "p3": 2})
    preds = mr.predict(StubReader(), tok, rows, temperature=1.0, max_length=256, device="cpu", batch_size=2)
    assert [p["predicted"] for p in preds] == ["REFUTES", "REFUTES", "NOT_ENOUGH_INFO"]
    assert tok.calls[0][:2] == (["c1", "c2"], ["p1", "p2"])
    assert tok.calls[0][2]["truncation"] is True and tok.calls[0][2]["max_length"] == 256
    assert mr.summarize(rows, preds) == {"cases": 3, "correct": 2, "accuracy": 2 / 3, "repeats_reported": 1}

    sharp = preds[0]["probabilities"]["REFUTES"]
    soft = mr.predict(StubReader(), tok, rows[:1], temperature=2.0, max_length=256, device="cpu")[0]
    assert soft["predicted"] == "REFUTES" and soft["probabilities"]["REFUTES"] < sharp
    assert sum(soft["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_each_released_reader_has_a_recorded_temperature():
    for name, source in mr.READERS.values():
        assert name.startswith("jithinpothireddy21/ease-delta-reader")
        t = json.loads((ROOT / source).read_text())["temperature"]
        assert 0.5 < t < 3.0


def test_scoring_an_empty_set_needs_no_model(tmp_path, capsys):
    assert mr.main(["score", "--cases", str(tmp_path / "none.jsonl")]) == 0
    assert "no cases to score" in capsys.readouterr().out
