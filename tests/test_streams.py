"""The training data stream, tested offline against a fake Hub.

The fake serves deterministic rows and can be told to fail, which is how the recovery and resume
logic is exercised without a network.
"""

import random

import pytest
import torch

from ease.data import streams
from ease.data.streams import (UNRELATED, BatchStream, MixtureConfig, ResilientStream, StreamMixture, content_words,
                               make_unrelated)
from ease.labels import NEI


class FakeHub:
    """Rows are generated from (source, epoch seed, index), so any position can be reproduced."""

    def __init__(self, sizes):
        self.sizes = sizes
        self.fail_at = {}  # source -> set of absolute call counts at which to raise
        self.calls = {k: 0 for k in sizes}
        self.opens = []

    def open(self, spec, split, seed, shuffle_buffer, skip):
        self.opens.append((spec.key, seed, skip))
        order = list(range(self.sizes[spec.key]))
        random.Random(seed).shuffle(order)
        hub = self

        def gen():
            for i in order[skip:]:
                hub.calls[spec.key] += 1
                if hub.calls[spec.key] in hub.fail_at.get(spec.key, ()):
                    raise ConnectionError("simulated network failure")
                yield hub.row(spec.key, i)
        return gen()

    @staticmethod
    def row(key, i):
        if key == "vitaminc":
            return {"claim": f"claim {key} {i} about topic{i % 50}", "evidence": f"evidence {key} {i} mentions topic{i % 50}",
                    "label": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"][i % 3], "case_id": f"c{i}", "page": f"page{i % 50}"}
        if key == "mnli":
            return {"hypothesis": f"hyp {key} {i}", "premise": f"prem {key} {i}", "label": i % 3, "pairID": f"{i}x",
                    "promptID": i}
        return {"hypothesis": f"hyp {key} {i}", "premise": f"prem {key} {i}", "gold": ["entailment", "neutral", "contradiction"][i % 3],
                "pairID": f"{i}"}


@pytest.fixture
def hub(monkeypatch):
    h = FakeHub({"vitaminc": 300, "mnli": 200, "wanli": 120})
    monkeypatch.setattr(streams, "_open", h.open)
    monkeypatch.setattr(streams.time, "sleep", lambda s: None)
    return h


class Tok:
    """Minimal tokenizer: one token per word, plus [CLS] and two [SEP]."""

    pad_token_id = 0

    def __call__(self, a, b, truncation, max_length, padding, return_attention_mask, return_token_type_ids):
        out = []
        for x, y in zip(a, b):
            ids = [1] + [2 + hash(w) % 1000 for w in x.split()] + [3] + [2 + hash(w) % 1000 for w in y.split()] + [3]
            out.append(ids[:max_length])
        return {"input_ids": out}


W = {"vitaminc": 0.4, "mnli": 0.27, "wanli": 0.13, "unrelated": 0.2}


def test_stream_recovers_from_failures_without_losing_or_repeating_rows(hub):
    clean = ResilientStream("vitaminc", seed=5, shuffle_buffer=10)
    want = [clean.next()["claim"] for _ in range(100)]
    hub.calls["vitaminc"] = 0
    hub.fail_at["vitaminc"] = {7, 8, 40, 41, 42, 90}
    flaky = ResilientStream("vitaminc", seed=5, shuffle_buffer=10)
    got = [flaky.next()["claim"] for _ in range(100)]
    assert got == want
    assert flaky.reopen_count == 6


def test_stream_gives_up_after_repeated_failures(hub):
    hub.fail_at["mnli"] = set(range(1, 1000))
    s = ResilientStream("mnli", seed=1, max_retries=4)
    with pytest.raises(RuntimeError, match="failed 5 times"):
        s.next()


def test_epochs_roll_over_with_a_new_order(hub):
    s = ResilientStream("wanli", seed=3, shuffle_buffer=10)
    first = [s.next()["claim"] for _ in range(120)]
    second = [s.next()["claim"] for _ in range(120)]
    assert s.epoch == 1 and sorted(first) == sorted(second) and first != second


def test_state_round_trip_continues_the_same_sequence(hub):
    a = ResilientStream("vitaminc", seed=9, shuffle_buffer=10)
    [a.next() for _ in range(350)]  # crosses an epoch boundary
    st = a.state()
    want = [a.next()["claim"] for _ in range(50)]
    b = ResilientStream("vitaminc", seed=9, shuffle_buffer=10)
    b.load_state(st)
    assert [b.next()["claim"] for _ in range(50)] == want


def test_untrainable_sources_are_refused():
    with pytest.raises(ValueError, match="not trainable"):
        ResilientStream("anli", seed=0)


def test_mixture_is_deterministic_and_matches_its_weights(hub):
    a = StreamMixture(MixtureConfig(dict(W), seed=11, shuffle_buffer=10))
    b = StreamMixture(MixtureConfig(dict(W), seed=11, shuffle_buffer=10))
    ra, rb = [a.next() for _ in range(4000)], [b.next() for _ in range(4000)]
    assert ra == rb
    frac = {k: a.drawn[k] / 4000 for k in W}
    for k, w in W.items():
        assert abs(frac[k] - w) < 0.03, (k, frac[k])
    c = StreamMixture(MixtureConfig(dict(W), seed=12, shuffle_buffer=10))
    assert [c.next() for _ in range(50)] != ra[:50]


def test_unrelated_pairs_are_nei_cross_topic_and_partly_mined(hub):
    m = StreamMixture(MixtureConfig(dict(W), seed=2, shuffle_buffer=10))
    rows = [m.next() for _ in range(3000)]
    un = [r for r in rows if r["source"] == UNRELATED]
    assert len(un) > 400 and all(r["label"] == NEI for r in un)
    for r in un:
        a, b = r["page"].split(":", 1)[1].split("|")
        assert a != b, "claim and evidence must come from different topics"
    assert 0.35 < m.mined / len(un) < 0.65


def test_mined_negatives_share_more_words_than_random_ones():
    claim = {"claim": "the delivery date for the client project is confirmed", "evidence": "x", "page": "a", "group": "a"}
    cands = [{"claim": f"c{i}", "evidence": e, "page": f"p{i}", "group": f"g{i}"} for i, e in enumerate([
        "kitchen closed friday", "the client project delivery was discussed", "weather is mild", "invoice issued"])]
    words = content_words(claim["claim"])
    best = max(cands, key=lambda b: len(words & content_words(b["evidence"])))
    assert best["evidence"] == "the client project delivery was discussed"
    assert make_unrelated(claim, best)["label"] == NEI
    assert make_unrelated(claim, {**best, "page": "a"}) is None, "same topic: might really be related"


def test_batches_are_bucketed_padded_and_complete(hub):
    bs = BatchStream(StreamMixture(MixtureConfig(dict(W), seed=4, shuffle_buffer=10)), Tok(), batch_size=8, max_len=32,
                     pool_batches=5)
    it = iter(bs)
    pool = [next(it) for _ in range(5)]
    assert {b.pool_id for b in pool} == {0} and sorted(b.index_in_pool for b in pool) == [0, 1, 2, 3, 4]
    for b in pool:
        assert b.input_ids.shape == b.attention_mask.shape and b.input_ids.shape[0] == 8
        lens = b.attention_mask.sum(1)
        assert int(lens.max()) == b.input_ids.shape[1], "padded to the longest row in the batch, no further"
        assert int(lens.max() - lens.min()) <= 4, "rows in a batch have similar length"
        assert ((b.input_ids == 0) == (b.attention_mask == 0)).all()
        assert b.n_tokens == int(b.attention_mask.sum())


def test_resume_mid_pool_reproduces_the_remaining_batches_exactly(hub):
    def make():
        return BatchStream(StreamMixture(MixtureConfig(dict(W), seed=6, shuffle_buffer=10)), Tok(), 8, 32, pool_batches=5)

    it = iter(make())
    ref = [next(it) for _ in range(23)]
    for k in (2, 4, 12, 19):  # inside a pool, at a pool's last batch, and later
        fresh = make()
        fresh.resume(ref[k].pool_id, ref[k].state_before_pool, ref[k].index_in_pool + 1)
        it2 = iter(fresh)
        for j in range(k + 1, 23):
            b = next(it2)
            assert torch.equal(b.input_ids, ref[j].input_ids) and torch.equal(b.labels, ref[j].labels), (k, j)
            assert b.sources == ref[j].sources and (b.pool_id, b.index_in_pool) == (ref[j].pool_id, ref[j].index_in_pool)


def test_resume_is_exact_even_when_the_network_fails_afterwards(hub):
    def make():
        return BatchStream(StreamMixture(MixtureConfig(dict(W), seed=8, shuffle_buffer=10)), Tok(), 8, 32, pool_batches=4)

    it = iter(make())
    ref = [next(it) for _ in range(12)]
    hub.fail_at = {"vitaminc": {hub.calls["vitaminc"] + 3, hub.calls["vitaminc"] + 30},
                   "mnli": {hub.calls["mnli"] + 5}}
    fresh = make()
    fresh.resume(ref[5].pool_id, ref[5].state_before_pool, ref[5].index_in_pool + 1)
    it2 = iter(fresh)
    for j in range(6, 12):
        assert torch.equal(next(it2).input_ids, ref[j].input_ids)


def test_local_rows_can_be_mixed_in_and_resume_exactly(hub, tmp_path):
    import json
    from ease.data.streams import LocalStream
    path = tmp_path / "fb.jsonl"
    path.write_text("\n".join(json.dumps({"claim": f"verified claim {i}", "evidence": f"verified evidence {i}",
                                          "label": i % 3}) for i in range(7)) + "\n")
    s = LocalStream("feedback", str(path), seed=1)
    first = [s.next()["claim"] for _ in range(7)]
    second = [s.next()["claim"] for _ in range(7)]
    assert sorted(first) == sorted(second) and s.epoch == 1
    w = {"vitaminc": 0.5, "feedback": 0.3, "unrelated": 0.2}
    a = StreamMixture(MixtureConfig(dict(w), seed=3, shuffle_buffer=10, local={"feedback": str(path)}))
    rows = [a.next() for _ in range(600)]
    assert 0.24 < sum(r["source"] == "feedback" for r in rows) / 600 < 0.36
    assert a.licences()["feedback"].startswith("local file")
    st = a.state()
    want = [a.next() for _ in range(100)]
    b = StreamMixture(MixtureConfig(dict(w), seed=3, shuffle_buffer=10, local={"feedback": str(path)}))
    b.load_state(st)
    assert [b.next() for _ in range(100)] == want


def test_local_file_is_validated(tmp_path):
    from ease.data.streams import LocalStream
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"claim": "c", "evidence": "e", "label": 7}\n')
    with pytest.raises(ValueError, match="label in 0..2"):
        LocalStream("feedback", str(bad), seed=0)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="no rows"):
        LocalStream("feedback", str(empty), seed=0)


# ----------------------------------------------------------------------------- ordered files
def ordered_corpus(n_real=3000, n_synth=1500):
    """A corpus shaped like VitaminC's train file: one kind of row first, then another kind that
    lacks a label entirely."""
    from datasets import Dataset
    rows = {"claim": [], "evidence": [], "label": [], "case_id": [], "page": [], "revision_type": [], "unused": []}
    for i in range(n_real + n_synth):
        real = i < n_real
        rows["claim"].append(f"claim {i}"); rows["evidence"].append(f"evidence {i}")
        rows["label"].append(["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"][i % 3] if real else ["SUPPORTS", "REFUTES"][i % 2])
        rows["case_id"].append(f"c{i}"); rows["page"].append(f"p{i}")
        rows["revision_type"].append("real" if real else "synthetic")
        rows["unused"].append("x" * 50)
    return Dataset.from_dict(rows).to_iterable_dataset()


def nei_share_by_third(monkeypatch, buffer, n=4500):
    monkeypatch.setattr(streams, "_load", lambda spec, split: ordered_corpus())
    s = ResilientStream("vitaminc", seed=7, shuffle_buffer=buffer)
    labels = [s.next()["label"] for _ in range(n)]
    third = n // 3
    return [sum(l == NEI for l in labels[i * third:(i + 1) * third]) / third for i in range(3)]


def test_a_small_buffer_leaves_an_ordered_file_ordered(monkeypatch):
    """The failure that spoiled the first full training run, reproduced."""
    shares = nei_share_by_third(monkeypatch, buffer=200)
    assert shares[0] > 0.30 and shares[2] < 0.02, shares


def test_a_buffer_spanning_the_corpus_mixes_it(monkeypatch):
    shares = nei_share_by_third(monkeypatch, buffer=streams.GLOBAL_SHUFFLE)
    overall = 1000 / 4500
    assert all(abs(x - overall) < 0.04 for x in shares), shares


def test_global_shuffle_resumes_exactly_and_drops_unused_columns(monkeypatch):
    seen = {}

    def load(spec, split):
        ds = ordered_corpus(600, 300)
        seen["columns_before"] = list(ds.features) if ds.features else None
        return ds

    monkeypatch.setattr(streams, "_load", load)
    a = ResilientStream("vitaminc", seed=3)
    first = [a.next()["claim"] for _ in range(500)]
    st = a.state()
    want = [a.next()["claim"] for _ in range(700)]  # crosses into the second pass
    b = ResilientStream("vitaminc", seed=3)
    b.load_state(st)
    assert [b.next()["claim"] for _ in range(700)] == want
    assert len(set(first)) == 500, "no row is repeated within a pass"
    raw = next(streams._open(streams.get_source("vitaminc"), "train", 1, 10, 0))
    assert "unused" not in raw and "claim" in raw


def test_a_checkpoint_written_by_the_trainer_loads_with_the_restricted_loader_and_resumes_exactly(hub, tmp_path):
    """Resuming must not need pickle's full power: a checkpoint is data, never code."""
    from collections import Counter

    from ease.train.stage_a import latest_checkpoint, save_checkpoint

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(4, 3)

        def save(self, d):
            torch.save(self.state_dict(), d / "weights.pt")

    def make():
        return BatchStream(StreamMixture(MixtureConfig(dict(W), seed=11, shuffle_buffer=10)), Tok(), 8, 32, pool_batches=5)

    it = iter(make())
    ref = [next(it) for _ in range(14)]
    model = Model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.w(torch.ones(2, 4)).sum().backward()
    opt.step()  # gives the optimizer moment tensors to store
    save_checkpoint(tmp_path, model, opt, 7, 56, ref[6], Counter(ref[6].sources), 1.5)

    ckpt = latest_checkpoint(tmp_path)
    state = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=True)
    assert (state["step"], state["examples"], state["elapsed"]) == (7, 56, 1.5)
    opt2 = torch.optim.AdamW(Model().parameters(), lr=1e-3)
    opt2.load_state_dict(state["optimizer"])
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]
    fresh = make()
    s = state["stream"]
    fresh.resume(s["pool_id"], s["state_before_pool"], s["next_index_in_pool"])
    it2 = iter(fresh)
    for j in range(7, 14):
        b = next(it2)
        assert torch.equal(b.input_ids, ref[j].input_ids) and torch.equal(b.labels, ref[j].labels), j
