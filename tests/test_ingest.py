"""Document ingestion: splitting, attribution, revision and withdrawal of passages."""

from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from ease.aggregate import RuleAggregator
from ease.ingest import split_passages
from ease.labels import REFUTES, SUPPORTS
from ease.runtime import Runtime
from ease.scorer import OracleScorer
from ease.service.api import create_app

D = 8
CLAIM = "The client approved the final design."
P_APPROVE = "Email from the client: We approve the design as presented, please go ahead with production."
P_WITHDRAW = "Email from the client: We withdraw our approval of the design until the palette is revised."


def test_paragraphs_become_passages_and_fragments_are_attached():
    text = "Hi team,\n\nWe approve the design as presented, please go ahead with production.\n\n" \
           "Separately, the kitchen is closed on Friday for maintenance work.\n\nThanks,\nDana"
    got = split_passages(text)
    assert got == ["Hi team, We approve the design as presented, please go ahead with production.",
                   "Separately, the kitchen is closed on Friday for maintenance work. Thanks, Dana"]


def test_long_paragraph_is_cut_at_sentence_boundaries():
    para = " ".join(f"Sentence number {i} says something of moderate length about the project." for i in range(30))
    got = split_passages(para, max_chars=300)
    assert len(got) > 5 and all(len(p) <= 300 for p in got)
    assert all(p.endswith(".") for p in got), "cuts fall between sentences"
    assert " ".join(got) == para, "nothing is lost or duplicated"


@settings(max_examples=200, deadline=None)
@given(st.text(alphabet=st.sampled_from(list("abcdefg .!?\n")), min_size=0, max_size=1500), st.integers(60, 400))
def test_splitting_is_deterministic_bounded_and_lossless(text, max_chars):
    a, b = split_passages(text, max_chars=max_chars), split_passages(text, max_chars=max_chars)
    assert a == b
    assert all(0 < len(p) <= max_chars for p in a)
    assert "".join(" ".join(a).split()) == "".join(text.split()), "every non-space character survives, in order"


def runtime(tmp_path):
    gold = {(CLAIM, P_APPROVE): SUPPORTS, (CLAIM, P_WITHDRAW): REFUTES}
    sc = OracleScorer(gold, margin=12.0, msg_dim=D)
    sc.cache = {}
    return Runtime(tmp_path / "d", sc, RuleAggregator(msg_dim=D), tau=0.9)


SCHEMA = {"task_id": "t", "predicates": [{"id": "approved", "text": CLAIM}],
          "actions": [{"id": "send", "description": "send", "requires": "approved"}]}


def test_document_revision_updates_and_withdraws_passages(tmp_path):
    rt = runtime(tmp_path)
    with TestClient(create_app(rt)) as c:
        c.post("/tasks", json=SCHEMA)
        v1 = "We approve the design as presented, please go ahead with production.\n\n" \
             "The invoice will follow next week once accounts have processed it.\n\n" \
             "Our office is closed on the first Monday of the month for training."
        r = c.post("/tasks/t/documents", json={"doc_id": "mail-7", "revision": 1, "text": v1, "source_id": "client",
                                               "authority": 2, "attribution": "Email from the client"}).json()
        assert r["ingest"]["passages"] == 3 and r["ingest"]["applied"] == 3 and r["ingest"]["withdrawn"] == 0
        assert r["assessments"]["send"]["disposition"] == "READY"
        st_ = c.get("/tasks/t").json()
        assert sorted(st_["records"]) == ["mail-7#0", "mail-7#1", "mail-7#2"]
        ex = c.get("/tasks/t/explain/send").json()
        rec = ex["predicates"][0]["records"][0]
        assert rec["text"] == P_APPROVE and rec["span"] == "mail-7, passage 1 of 3"

        # the same document again: nothing to do
        r = c.post("/tasks/t/documents", json={"doc_id": "mail-7", "revision": 1, "text": v1, "source_id": "client",
                                               "authority": 2, "attribution": "Email from the client"}).json()
        assert r["ingest"]["applied"] == 0 and r["ingest"]["unchanged"] == 3 and r["ingest"]["edges_scored"] == 0

        # revised and shorter: first passage changes, third disappears
        v2 = "We withdraw our approval of the design until the palette is revised.\n\n" \
             "The invoice will follow next week once accounts have processed it."
        r = c.post("/tasks/t/documents", json={"doc_id": "mail-7", "revision": 2, "text": v2, "source_id": "client",
                                               "authority": 2, "attribution": "Email from the client"}).json()
        assert r["ingest"]["passages"] == 2 and r["ingest"]["withdrawn"] == 1
        assert ["send", "READY", "BLOCKED"] in r["ingest"]["changed_actions"]
        st_ = c.get("/tasks/t").json()
        assert st_["records"]["mail-7#2"]["status"] == "withdrawn"
        assert st_["records"]["mail-7#0"]["revision"] == 2
        assert c.get("/tasks/t/verify").json()["not_bitwise_equal"] == 0
        assert c.post("/tasks/t/documents", json={"doc_id": "a#b", "revision": 1, "text": "x"}).status_code == 422
    rt.close()


def test_a_passage_that_only_moved_is_not_read_again(tmp_path):
    rt = runtime(tmp_path)
    rt.scorer.cache = {}
    calls = {"n": 0}
    real = rt.scorer.score

    def counting(pairs):
        # emulate the model scorer's cache: only unseen texts cost a reading
        new = [p for p in pairs if p not in rt.scorer.cache]
        calls["n"] += len(new)
        for p in new:
            rt.scorer.cache[p] = True
        return real(pairs)

    rt.scorer.score = counting
    with TestClient(create_app(rt)) as c:
        c.post("/tasks", json=SCHEMA)
        a = "The invoice will follow next week once accounts have processed it."
        b = "Our office is closed on the first Monday of the month for training."
        c.post("/tasks/t/documents", json={"doc_id": "m", "revision": 1, "text": f"{a}\n\n{b}"})
        first = calls["n"]
        assert first == 2
        new_top = "A new opening paragraph was added above everything else in the message."
        c.post("/tasks/t/documents", json={"doc_id": "m", "revision": 2, "text": f"{new_top}\n\n{a}\n\n{b}"})
        assert calls["n"] - first == 1, "only the new paragraph is read; the two that moved are looked up"
    rt.close()
