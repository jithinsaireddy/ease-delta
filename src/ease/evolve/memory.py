"""Correction memory: verified readings that take effect immediately and can be withdrawn.

When a person confirms or corrects how a record bears on a claim, that judgement is stored with
the message vector the edge model produced for the pair. It is then used in two ways:

  exact     the same pair is read as verified from now on
  similar   a new pair whose message vector is close to stored ones has its reading pulled
            towards theirs, more strongly the closer it is

Nothing is written into model weights. Deleting an item restores the earlier behaviour exactly,
which is the property the blueprint's information argument asks for: a correction that must be
retractable has to be stored where it can be found.

Blending rule for a pair with model probabilities p and neighbours (s_j, y_j), s_j = cosine
similarity, among the k nearest with s_j >= tau:

    w_j   = (s_j - tau) / (1 - tau)                      in [0, 1]
    q     = sum_j w_j onehot(y_j) / sum_j w_j            neighbours' label distribution
    lam   = lam_max * (1 - prod_j (1 - w_j))             grows with closeness and with agreement in number
    p'    = (1 - lam) p + lam q

lam_max and tau are fitted on held-out feedback by the consolidation gate, not set by hand.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ease.util import text_hash


@dataclass
class MemoryConfig:
    """`lam_max` starts at 0: a verified pair is honoured at once, but nothing is generalised to
    *similar* pairs until a consolidation gate has measured, on held-out feedback, that doing so
    helps. Generalisation has to be earned."""

    k: int = 8
    tau: float = 0.80
    lam_max: float = 0.0
    exact_confidence: float = 0.995
    min_votes: int = 1


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_blend(probs: np.ndarray, q_unit: np.ndarray, mem_unit: np.ndarray, mem_labels: np.ndarray,
              cfg: MemoryConfig, exclude_self: bool = False, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised form of the blending rule in the module docstring.
    `exclude_self` is for leave-one-out evaluation, when queries are the memory items themselves."""
    out = np.asarray(probs, np.float64).copy()
    touched = np.zeros(len(out), bool)
    M = mem_unit.shape[0]
    if M == 0 or len(out) == 0 or cfg.lam_max <= 0.0:
        return out, touched
    k = min(cfg.k, M - (1 if exclude_self else 0))
    if k <= 0:
        return out, touched
    for s in range(0, len(out), block):
        q = q_unit[s : s + block]
        sims = q @ mem_unit.T
        if exclude_self:
            ix = np.arange(s, s + len(q))
            sims[np.arange(len(q)), ix] = -np.inf
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        ts = np.take_along_axis(sims, top, axis=1).astype(np.float64)
        w = np.clip((ts - cfg.tau) / max(1e-9, 1.0 - cfg.tau), 0.0, 1.0)
        votes = (w > 0).sum(1)
        wsum = w.sum(1)
        ok = (votes >= cfg.min_votes) & (wsum > 0)
        if not ok.any():
            continue
        q_lab = np.zeros((len(q), 3))
        lab = mem_labels[top]
        for c in range(3):
            q_lab[:, c] = (w * (lab == c)).sum(1)
        q_lab = q_lab / np.maximum(wsum[:, None], 1e-12)
        lam = cfg.lam_max * (1.0 - np.prod(1.0 - w, axis=1))
        blended = (1.0 - lam[:, None]) * out[s : s + len(q)] + lam[:, None] * q_lab
        out[s : s + len(q)][ok] = blended[ok]
        touched[s : s + len(q)] = ok
    return out / out.sum(1, keepdims=True), touched


@dataclass
class MemoryItem:
    item_id: int
    pair: str
    label: int
    msg: np.ndarray
    source: str
    created: float
    weight: float = 1.0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    label INTEGER NOT NULL,
    msg BLOB NOT NULL,
    dim INTEGER NOT NULL,
    source TEXT NOT NULL,
    created REAL NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS corrections_pair ON corrections(pair);
"""


class CorrectionMemory:
    def __init__(self, path: str = ":memory:", cfg: Optional[MemoryConfig] = None):
        self.cfg = cfg or MemoryConfig()
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.executescript(_SCHEMA)
        self._load()

    def _load(self) -> None:
        rows = self._db.execute(
            "SELECT item_id, pair, label, msg, dim, source, created, weight FROM corrections WHERE active=1 "
            "ORDER BY item_id").fetchall()
        self.items = [MemoryItem(i, p, l, np.frombuffer(m, np.float32, d).copy(), s, c, w)
                      for i, p, l, m, d, s, c, w in rows]
        self._reindex()

    def _reindex(self) -> None:
        if self.items:
            m = np.stack([it.msg for it in self.items]).astype(np.float32)
            self.unit = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
            self.labels = np.asarray([it.label for it in self.items], np.int64)
        else:
            self.unit = np.zeros((0, 1), np.float32)
            self.labels = np.zeros(0, np.int64)
        # the latest verdict for a pair wins; earlier ones remain in the log
        self.exact = {it.pair: it.label for it in self.items}
        self._version = text_hash(
            json.dumps([(it.item_id, it.pair, it.label) for it in self.items]),
            json.dumps(self.cfg.__dict__, sort_keys=True))[:16]

    @property
    def version(self) -> str:
        return f"mem-{self._version}"

    def __len__(self) -> int:
        return len(self.items)

    # -- writes ------------------------------------------------------------
    def add(self, pair: str, label: int, msg: np.ndarray, source: str = "user", weight: float = 1.0) -> int:
        if label not in (0, 1, 2):
            raise ValueError("label must be SUPPORTS, REFUTES or NEI")
        msg = np.asarray(msg, np.float32).reshape(-1)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO corrections(pair, label, msg, dim, source, created, weight) VALUES (?,?,?,?,?,?,?)",
                (pair, int(label), msg.tobytes(), int(msg.shape[0]), source, time.time(), float(weight)))
            self.items.append(MemoryItem(cur.lastrowid, pair, int(label), msg, source, time.time(), weight))
            self._reindex()
            return cur.lastrowid

    def add_many(self, rows: Sequence[tuple[str, int, np.ndarray]], source: str = "user") -> int:
        with self._lock:
            self._db.execute("BEGIN")
            for pair, label, msg in rows:
                msg = np.asarray(msg, np.float32).reshape(-1)
                cur = self._db.execute(
                    "INSERT INTO corrections(pair, label, msg, dim, source, created, weight) VALUES (?,?,?,?,?,?,1.0)",
                    (pair, int(label), msg.tobytes(), int(msg.shape[0]), source, time.time()))
                self.items.append(MemoryItem(cur.lastrowid, pair, int(label), msg, source, time.time()))
            self._db.execute("COMMIT")
            self._reindex()
            return len(rows)

    def remove(self, item_id: int) -> bool:
        with self._lock:
            n = self._db.execute("UPDATE corrections SET active=0 WHERE item_id=? AND active=1", (item_id,)).rowcount
            if n:
                self.items = [it for it in self.items if it.item_id != item_id]
                self._reindex()
            return bool(n)

    def remove_pair(self, pair: str) -> int:
        with self._lock:
            n = self._db.execute("UPDATE corrections SET active=0 WHERE pair=? AND active=1", (pair,)).rowcount
            if n:
                self.items = [it for it in self.items if it.pair != pair]
                self._reindex()
            return n

    def set_config(self, cfg: MemoryConfig) -> None:
        self.cfg = cfg
        self._reindex()

    # -- reads -------------------------------------------------------------
    def adjust(self, probs: np.ndarray, msgs: np.ndarray, pairs: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """probs [n,3], msgs [n,d]. Returns (adjusted probs, how: 0 untouched, 1 similar, 2 exact)."""
        out = np.asarray(probs, np.float64).copy()
        how = np.zeros(len(out), np.int64)
        if not self.items or len(out) == 0:
            return out, how
        out, touched = knn_blend(out, unit(msgs), self.unit, self.labels, self.cfg)
        how[touched] = 1
        c = self.cfg.exact_confidence
        for i, pair in enumerate(pairs):
            lab = self.exact.get(pair) if pair else None
            if lab is not None:
                out[i] = (1.0 - c) / 2.0
                out[i, lab] = c
                how[i] = 2
        return out, how

    def close(self) -> None:
        self._db.close()
