"""Streaming training data from the Hugging Face Hub.

Nothing is downloaded in full: rows are read over the network as training consumes them.
Three properties matter for a multi-hour run on a laptop:

  resilient   a dropped connection reopens the stream and skips the rows already used
  resumable   the mixture state is a small JSON object; restoring it reproduces the
              exact same sequence of batches
  honest      every row records which corpus it came from, so the training manifest can
              state how many examples of each licence the weights have seen
"""

from __future__ import annotations

import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch

from ease.data.sources import SourceSpec, get_source
from ease.labels import NEI

UNRELATED = "unrelated"


GLOBAL_SHUFFLE = 1_000_000  # larger than any corpus used here, so the buffer spans the whole file


def _load(spec: SourceSpec, split: str):
    from datasets import load_dataset

    if spec.config:
        return load_dataset(spec.hf_id, spec.config, split=split, streaming=True)
    return load_dataset(spec.hf_id, split=split, streaming=True)


def _open(spec: SourceSpec, split: str, seed: Optional[int], shuffle_buffer: int, skip: int):
    """A shuffled iterator over one split, starting `skip` rows in.

    The shuffle buffer must span the corpus. A buffer smaller than the file shuffles only locally:
    rows that are far apart in the file stay far apart in the stream. VitaminC's train file holds
    every real-revision row before every synthetic row, and the synthetic rows have no NEI label.
    With a 10,000-row buffer a model trained on that stream saw only synthetic VitaminC rows for
    the last third of training, and its accuracy on real revisions fell from 78% to 72% while its
    training loss kept improving (runs/stage_a_v1_ordered_stream).

    With the buffer larger than the corpus, every row is read into memory and the rows are then
    yielded in a random permutation. Nothing is written to disk.
    """
    ds = _load(spec, split)
    if spec.columns:
        ds = ds.select_columns(list(spec.columns))
    if seed is not None and shuffle_buffer > 1:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if skip:
        ds = ds.skip(skip)
    return iter(ds)


class ResilientStream:
    """An endless, reshuffled-each-epoch view of one training split.

    State is (epoch, consumed): `consumed` counts raw rows pulled in the current epoch,
    including rows the mapper rejected, so that skip(consumed) lands on the same row.
    """

    def __init__(self, key: str, seed: int, shuffle_buffer: int = GLOBAL_SHUFFLE, max_retries: int = 12):
        self.spec = get_source(key)
        if not self.spec.trainable or not self.spec.train_split:
            raise ValueError(
                f"source {key!r} is not trainable (licence: {self.spec.license}); it can only be evaluated on"
            )
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.max_retries = max_retries
        self.epoch = 0
        self.consumed = 0
        self.yielded = 0
        self.reopen_count = 0
        self._it = None

    # -- state -------------------------------------------------------------
    def state(self) -> dict:
        return {"epoch": self.epoch, "consumed": self.consumed, "yielded": self.yielded}

    def load_state(self, st: dict) -> None:
        self.epoch, self.consumed, self.yielded = int(st["epoch"]), int(st["consumed"]), int(st.get("yielded", 0))
        self._it = None

    # -- iteration ---------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._it is None:
            self._it = _open(
                self.spec, self.spec.train_split, self.seed + 1000003 * self.epoch, self.shuffle_buffer, self.consumed
            )

    def next(self) -> dict:
        failures = 0
        while True:
            try:
                self._ensure_open()
                raw = next(self._it)
                self.consumed += 1
                row = self.spec.mapper(raw)
                if row is None:
                    continue
                self.yielded += 1
                return row
            except StopIteration:
                self.epoch += 1
                self.consumed = 0
                self._it = None
                failures = 0
            except Exception as e:  # network, parsing, hub hiccups
                failures += 1
                self.reopen_count += 1
                self._it = None
                if failures > self.max_retries:
                    raise RuntimeError(
                        f"stream {self.spec.key} failed {failures} times in a row; last error: {type(e).__name__}: {e}"
                    ) from e
                time.sleep(min(60.0, 1.5 ** failures))


_STOP = frozenset("a an the of in on at to for from by with and or not no is are was were be been being it its "
                  "this that these those as has have had he she they we you i his her their our your do does did "
                  "there than then so if but which who whom what when where will would can could may might".split())


def content_words(text: str) -> frozenset:
    return frozenset(w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
                     if w not in _STOP and len(w) > 1)


class LocalStream:
    """Rows from a local JSONL file ({"claim", "evidence", "label"}), reshuffled each pass.
    Used to fold verified corrections into a continued training run."""

    def __init__(self, key: str, path: str, seed: int):
        import json

        self.key, self.path, self.seed = key, str(path), seed
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("label") not in (0, 1, 2) or not r.get("claim") or not r.get("evidence"):
                    raise ValueError(f"{path}, line {i + 1}: need claim, evidence and label in 0..2")
                self.rows.append({"claim": " ".join(str(r["claim"]).split()),
                                  "evidence": " ".join(str(r["evidence"]).split()), "label": int(r["label"]),
                                  "source": key, "group": f"{key}:{i}", "page": f"{key}:{r.get('page', i)}"})
        if not self.rows:
            raise ValueError(f"{path} contains no rows")
        self.epoch = self.consumed = self.yielded = 0
        self.reopen_count = 0
        self._order: list[int] = []

    class spec:  # the mixture asks a stream for its licence
        license = "local file supplied by the operator"

    def state(self) -> dict:
        return {"epoch": self.epoch, "consumed": self.consumed, "yielded": self.yielded}

    def load_state(self, st: dict) -> None:
        self.epoch, self.consumed, self.yielded = int(st["epoch"]), int(st["consumed"]), int(st.get("yielded", 0))
        self._order = []

    def next(self) -> dict:
        if self.consumed >= len(self.rows):
            self.epoch, self.consumed, self._order = self.epoch + 1, 0, []
        if not self._order:
            self._order = list(range(len(self.rows)))
            random.Random(self.seed + 1000003 * self.epoch).shuffle(self._order)
        row = self.rows[self._order[self.consumed]]
        self.consumed += 1
        self.yielded += 1
        return dict(row)


def make_unrelated(a: dict, b: dict) -> Optional[dict]:
    """Claim of one example with evidence of another, labelled NEI.

    Training corpora contain only topically related pairs, so without these a model never learns
    that an unrelated document establishes nothing. Pairs from the same topic key are rejected
    because they might genuinely bear on each other.
    """
    if a["page"] == b["page"] or a["claim"] == b["claim"] or a["evidence"] == b["evidence"]:
        return None
    return {"claim": a["claim"], "evidence": b["evidence"], "label": NEI, "source": UNRELATED,
            "group": f"{a['group']}|{b['group']}", "page": f"{UNRELATED}:{a['page']}|{b['page']}"}


@dataclass
class MixtureConfig:
    """`weights` maps source keys to sampling weights.

    The key "unrelated" is synthetic. It crosses the claim of one recently seen example with the
    evidence of another from a different topic. Half of these are random; the other half take, out
    of `mine_candidates` random candidates, the evidence sharing the most content words with the
    claim, because distractors met in practice (other messages in the same project) share
    vocabulary with the claim and are harder to dismiss than random text.
    """

    weights: dict[str, float]
    seed: int = 1234
    shuffle_buffer: int = GLOBAL_SHUFFLE
    local: dict = field(default_factory=dict)  # key -> path of a JSONL file; the key must appear in weights
    reservoir: int = 512
    mine_candidates: int = 8
    mined_fraction: float = 0.5

    def normalised(self) -> dict[str, float]:
        tot = sum(self.weights.values())
        if tot <= 0:
            raise ValueError("mixture weights must sum to a positive number")
        return {k: v / tot for k, v in self.weights.items() if v > 0}


class StreamMixture:
    """Draws each example from a source chosen with fixed probabilities and a seeded RNG."""

    MIN_RESERVOIR = 64

    def __init__(self, cfg: MixtureConfig):
        self.cfg = cfg
        w = cfg.normalised()
        self.keys = sorted(w)
        self.probs = [w[k] for k in self.keys]
        self.real_keys = [k for k in self.keys if k != UNRELATED]
        if not self.real_keys:
            raise ValueError("the mixture needs at least one real source")
        real_total = sum(w[k] for k in self.real_keys)
        self.real_probs = [w[k] / real_total for k in self.real_keys]
        self.rng = random.Random(cfg.seed)
        self.streams = {
            k: (LocalStream(k, cfg.local[k], seed=cfg.seed + 7919 * i) if k in cfg.local
                else ResilientStream(k, seed=cfg.seed + 7919 * i, shuffle_buffer=cfg.shuffle_buffer))
            for i, k in enumerate(self.real_keys)
        }
        self.drawn = {k: 0 for k in self.keys}
        self.recent: list[dict] = []  # ring buffer of real examples; material for unrelated pairs
        self._recent_pos = 0
        self.mined = 0

    # -- state -------------------------------------------------------------
    def state(self) -> dict:
        st = self.rng.getstate()
        return {
            "rng": [st[0], list(st[1]), st[2]],
            "streams": {k: s.state() for k, s in self.streams.items()},
            "drawn": dict(self.drawn),
            "recent": [dict(r) for r in self.recent],
            "recent_pos": self._recent_pos,
            "mined": self.mined,
        }

    def load_state(self, st: dict) -> None:
        r = st["rng"]
        self.rng.setstate((r[0], tuple(r[1]), r[2]))
        for k, s in st["streams"].items():
            self.streams[k].load_state(s)
        self.drawn = {k: int(v) for k, v in st["drawn"].items()}
        self.recent = [dict(x) for x in st.get("recent", [])]
        self._recent_pos = int(st.get("recent_pos", 0))
        self.mined = int(st.get("mined", 0))

    # -- sampling ----------------------------------------------------------
    def _remember(self, row: dict) -> None:
        if len(self.recent) < self.cfg.reservoir:
            self.recent.append(row)
        else:
            self.recent[self._recent_pos] = row
            self._recent_pos = (self._recent_pos + 1) % self.cfg.reservoir

    def _real(self) -> dict:
        k = self.rng.choices(self.real_keys, weights=self.real_probs, k=1)[0]
        row = self.streams[k].next()
        self._remember(row)
        return row

    def _unrelated(self) -> Optional[dict]:
        if len(self.recent) < self.MIN_RESERVOIR:
            return None
        a = self.recent[self.rng.randrange(len(self.recent))]
        cands = [self.recent[self.rng.randrange(len(self.recent))] for _ in range(self.cfg.mine_candidates)]
        cands = [b for b in cands if make_unrelated(a, b) is not None]
        if not cands:
            return None
        if self.rng.random() < self.cfg.mined_fraction:
            words = content_words(a["claim"])
            # ties broken by position, so the choice is a function of the RNG state alone
            b = max(enumerate(cands), key=lambda ib: (len(words & content_words(ib[1]["evidence"])), -ib[0]))[1]
            self.mined += 1
        else:
            b = cands[0]
        return make_unrelated(a, b)

    def next(self) -> dict:
        k = self.rng.choices(self.keys, weights=self.probs, k=1)[0]
        if k == UNRELATED:
            row = self._unrelated()
            if row is not None:
                self.drawn[k] += 1
                return row
            k = self.rng.choices(self.real_keys, weights=self.real_probs, k=1)[0]
        self.drawn[k] += 1
        row = self.streams[k].next()
        self._remember(row)
        return row

    def licences(self) -> dict[str, str]:
        out = {k: self.streams[k].spec.license for k in self.real_keys}
        if UNRELATED in self.keys:
            out[UNRELATED] = "derived: texts drawn from the real sources above"
        return out


# ---------------------------------------------------------------------------
# Tokenisation and length-bucketed batching
# ---------------------------------------------------------------------------


def encode_pairs(tokenizer, claims: list[str], evidences: list[str], max_len: int) -> list[list[int]]:
    """Claim first, evidence second. 'longest_first' trims whichever side is longer, so a very long
    evidence passage is shortened before a short claim is touched."""
    enc = tokenizer(
        claims, evidences, truncation="longest_first", max_length=max_len, padding=False,
        return_attention_mask=False, return_token_type_ids=False,
    )
    return enc["input_ids"]


def pad_batch(ids: list[list[int]], pad_id: int, pad_to: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(x) for x in ids) if pad_to is None else pad_to
    input_ids = torch.full((len(ids), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        n = min(len(x), L)
        input_ids[i, :n] = torch.tensor(x[:n], dtype=torch.long)
        mask[i, :n] = 1
    return input_ids, mask


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sources: list[str]
    pool_id: int
    index_in_pool: int
    batches_in_pool: int
    state_before_pool: dict = field(repr=False, default_factory=dict)
    n_tokens: int = 0


class BatchStream:
    """Turns the mixture into padded batches.

    Examples are gathered into a pool of `pool_batches * batch_size`, sorted by token length and cut
    into batches, so a batch contains similar lengths and little padding. Batch order inside a pool
    is shuffled with an RNG derived from the pool id. The pool is the unit of resumption: a batch
    carries the mixture state from before its pool was drawn, which is enough to regenerate the pool
    and continue from the next batch.
    """

    def __init__(self, mixture: StreamMixture, tokenizer, batch_size: int, max_len: int, pool_batches: int = 50):
        self.mixture = mixture
        self.tok = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.pool_batches = pool_batches
        self.pool_id = 0
        self._resume_skip = 0

    def resume(self, pool_id: int, state_before_pool: dict, next_index_in_pool: int) -> None:
        self.mixture.load_state(state_before_pool)
        self.pool_id = pool_id
        self._resume_skip = next_index_in_pool

    def _make_pool(self) -> list[Batch]:
        state_before = self.mixture.state()
        n = self.batch_size * self.pool_batches
        rows = [self.mixture.next() for _ in range(n)]
        ids = encode_pairs(self.tok, [r["claim"] for r in rows], [r["evidence"] for r in rows], self.max_len)
        order = sorted(range(n), key=lambda i: len(ids[i]))
        chunks = [order[i : i + self.batch_size] for i in range(0, n, self.batch_size)]
        random.Random(self.mixture.cfg.seed * 1_000_003 + self.pool_id).shuffle(chunks)
        out = []
        for j, ch in enumerate(chunks):
            b_ids = [ids[i] for i in ch]
            input_ids, mask = pad_batch(b_ids, self.tok.pad_token_id)
            out.append(
                Batch(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=torch.tensor([rows[i]["label"] for i in ch], dtype=torch.long),
                    sources=[rows[i]["source"] for i in ch],
                    pool_id=self.pool_id,
                    index_in_pool=j,
                    batches_in_pool=len(chunks),
                    state_before_pool=state_before,
                    n_tokens=int(mask.sum()),
                )
            )
        self.pool_id += 1
        return out

    def __iter__(self) -> Iterator[Batch]:
        while True:
            pool = self._make_pool()
            skip, self._resume_skip = self._resume_skip, 0
            for b in pool[skip:]:
                yield b


class Prefetcher:
    """Runs a BatchStream in a background thread so the GPU never waits on the network."""

    def __init__(self, batch_stream: BatchStream, capacity: int = 96):
        self.q: queue.Queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._err: Optional[BaseException] = None
        self._bs = batch_stream
        self._t = threading.Thread(target=self._run, name="ease-prefetch", daemon=True)
        self._t.start()

    def _run(self) -> None:
        try:
            for b in self._bs:
                while not self._stop.is_set():
                    try:
                        self.q.put(b, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
        except BaseException as e:  # surfaced to the trainer on the next get()
            self._err = e

    def get(self, timeout: float = 900.0) -> Batch:
        waited = 0.0
        while True:
            if self._err is not None:
                raise RuntimeError(f"data thread died: {type(self._err).__name__}: {self._err}") from self._err
            try:
                return self.q.get(timeout=1.0)
            except queue.Empty:
                waited += 1.0
                if waited >= timeout:
                    raise TimeoutError(f"no batch arrived in {timeout:.0f}s; the stream may be stalled")

    def close(self) -> None:
        self._stop.set()
