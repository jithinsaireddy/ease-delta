"""Edge scorers: the only place where the large model runs.

A scorer maps (claim, evidence) to raw relation logits and a message vector. The value depends on
the weights and the two texts and on nothing else, so results are cached under
hash(weights version, claim, evidence). A revised record has new text, hence a new key; every
other pair keeps its cached value.

Calibration is deliberately *not* applied here. It is applied by the aggregator, so that adjusting
calibration later invalidates the cheap nodes downstream and never forces the encoder to rerun.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from ease.labels import NEI, NUM_RELATIONS
from ease.util import device_sync, get_device, text_hash


@dataclass
class ScorerStats:
    calls: int = 0
    pairs_requested: int = 0
    pairs_computed: int = 0
    cache_hits: int = 0
    tokens: int = 0
    seconds: float = 0.0

    def snapshot(self) -> dict:
        return dict(self.__dict__)

    def since(self, before: dict) -> dict:
        return {k: getattr(self, k) - before[k] for k in before}


class EdgeScorer(Protocol):
    version: str
    msg_dim: int
    stats: ScorerStats

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[np.ndarray, np.ndarray]]: ...


def file_digest(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class ModelScorer:
    """Runs the trained edge model.

    With dynamic batching a pair's value depends slightly on the pairs it was batched with: on the
    development machine (M4 Max, MPS, torch 2.14) regrouping changed values by up to 4.8e-6.
    `canonical=True` removes that dependence. Every forward pass then has a shape determined by the
    pair alone: the batch always holds `canonical_batch` rows (short batches are filled with copies
    of their first row) and the length is the smallest bucket that fits the pair. In that mode
    regrouping changed nothing, bit for bit, on both MPS and CPU (runs/determinism_probe.json).
    """

    def __init__(self, model_dir: str | Path, device: Optional[str] = None, batch_size: int = 64,
                 canonical: bool = False, canonical_len: Optional[int] = None, canonical_batch: int = 8,
                 canonical_buckets: tuple[int, ...] = (64, 128, 256),
                 cache: Optional[dict] = None, max_cache: int = 500_000):
        import torch
        from transformers import AutoTokenizer

        from ease.model.edge import EdgeModel

        self.dir = Path(model_dir)
        self.device = get_device(device)
        self.model = EdgeModel.load(self.dir, self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.max_len = self.model.cfg.max_len
        self.msg_dim = self.model.cfg.msg_dim
        self.batch_size = batch_size
        self.canonical = canonical
        self.canonical_batch = canonical_batch
        buckets = (canonical_len,) if canonical_len else canonical_buckets
        self.buckets = tuple(sorted({min(b, self.max_len) for b in buckets} | {self.max_len}))
        self.version = "edge-" + file_digest(self.dir / "model.safetensors")[:16]
        self.cache: dict = cache if cache is not None else {}
        self.max_cache = max_cache
        self.stats = ScorerStats()
        self._torch = torch

    def key(self, claim: str, evidence: str) -> str:
        return text_hash(self.version, claim, evidence)

    def _run(self, claims: list[str], evidences: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
        from ease.data.streams import encode_pairs, pad_batch
        from ease.model.infer import round_up

        torch = self._torch
        ids = encode_pairs(self.tok, claims, evidences, self.max_len)
        n = len(ids)
        logits = np.zeros((n, NUM_RELATIONS), np.float32)
        msgs = np.zeros((n, self.msg_dim), np.float32)
        tokens = sum(len(x) for x in ids)
        order = sorted(range(n), key=lambda i: len(ids[i]))
        if self.canonical:
            bs = self.canonical_batch
            groups: dict[int, list[int]] = {}
            for i in order:
                groups.setdefault(next(b for b in self.buckets if b >= len(ids[i])), []).append(i)
            plan = [(L, g[s : s + bs]) for L, g in sorted(groups.items()) for s in range(0, len(g), bs)]
        else:
            bs = self.batch_size
            plan = []
            for s in range(0, n, bs):
                ix = order[s : s + bs]
                plan.append((min(self.max_len, round_up(max(len(ids[i]) for i in ix), 8)), ix))
        with torch.inference_mode():
            for L, ix in plan:
                chunk = [ids[i] for i in ix]
                real = len(chunk)
                if self.canonical:
                    chunk = chunk + [chunk[0]] * (bs - real)  # fixed batch shape
                inp, mask = pad_batch(chunk, self.tok.pad_token_id, pad_to=L)
                lg, mg = self.model(inp.to(self.device), mask.to(self.device))
                lg = lg.float().cpu().numpy()[:real]
                mg = mg.float().cpu().numpy()[:real]
                for j, i in enumerate(ix):
                    logits[i] = lg[j]
                    msgs[i] = mg[j]
        return logits, msgs, tokens

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[np.ndarray, np.ndarray]]:
        t0 = time.perf_counter()
        self.stats.calls += 1
        self.stats.pairs_requested += len(pairs)
        keys = [self.key(c, e) for c, e in pairs]
        todo: dict[str, tuple[str, str]] = {}
        for k, p in zip(keys, pairs):
            if k in self.cache:
                self.stats.cache_hits += 1
            else:
                todo.setdefault(k, p)
        if todo:
            ks = list(todo)
            logits, msgs, tokens = self._run([todo[k][0] for k in ks], [todo[k][1] for k in ks])
            device_sync(self.device)
            if len(self.cache) + len(ks) > self.max_cache:
                self.cache.clear()
            for k, lg, mg in zip(ks, logits, msgs):
                self.cache[k] = (lg.copy(), mg.copy())
            self.stats.pairs_computed += len(ks)
            self.stats.tokens += tokens
        out = [self.cache[k] for k in keys]
        self.stats.seconds += time.perf_counter() - t0
        return out

    def drop(self, claim: str, evidence: str) -> None:
        self.cache.pop(self.key(claim, evidence), None)


class TableScorer:
    """Serves precomputed edge values. Used to train and evaluate downstream parts without the GPU."""

    def __init__(self, table: dict[str, tuple[np.ndarray, np.ndarray]], version: str, msg_dim: int,
                 fallback: Optional[EdgeScorer] = None):
        self.table = table
        self.version = version
        self.msg_dim = msg_dim
        self.fallback = fallback
        self.stats = ScorerStats()

    def key(self, claim: str, evidence: str) -> str:
        return text_hash(self.version, claim, evidence)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[np.ndarray, np.ndarray]]:
        self.stats.calls += 1
        self.stats.pairs_requested += len(pairs)
        out: list = [None] * len(pairs)
        missing = []
        for i, (c, e) in enumerate(pairs):
            v = self.table.get(self.key(c, e))
            if v is None:
                missing.append(i)
            else:
                out[i] = v
                self.stats.pairs_computed += 1
        if missing:
            if self.fallback is None:
                c, e = pairs[missing[0]]
                raise KeyError(f"{len(missing)} pairs are not in the table, e.g. claim={c[:60]!r} evidence={e[:60]!r}")
            got = self.fallback.score([pairs[i] for i in missing])
            for i, v in zip(missing, got):
                out[i] = v
                self.table[self.key(*pairs[i])] = v
        return out


class OracleScorer:
    """Returns a supplied gold relation with a fixed margin, NEI for anything unknown.
    It exists to test the engine's bookkeeping separately from any model's reading ability."""

    def __init__(self, gold: dict[tuple[str, str], int], margin: float = 8.0, msg_dim: int = 8,
                 version: str = "oracle-1"):
        self.gold = gold
        self.margin = margin
        self.msg_dim = msg_dim
        self.version = version
        self.stats = ScorerStats()

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[np.ndarray, np.ndarray]]:
        self.stats.calls += 1
        self.stats.pairs_requested += len(pairs)
        self.stats.pairs_computed += len(pairs)
        out = []
        for c, e in pairs:
            lab = self.gold.get((c, e), NEI)
            lg = np.zeros(NUM_RELATIONS, np.float32)
            lg[lab] = self.margin
            msg = np.zeros(self.msg_dim, np.float32)
            msg[lab] = 1.0
            out.append((lg, msg))
        return out
