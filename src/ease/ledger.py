"""The evidence ledger: exact, versioned, order-independent.

Nothing here is learned. The ledger answers "what is the current evidence?" and it must give the
same answer however the events were delivered.

Model
-----
Each record has an identity (`record_id`) that persists across revisions. An event carries a
revision number and either content or a withdrawal. The state of a record is

    (top revision r, set of distinct payloads seen at r)

and merging an event with revision q does exactly one thing:

    q <  r   ignored                       STALE
    q == r   payload added to the set      DUPLICATE if already present, else CONFLICT
    q >  r   set replaced by {payload}     APPLIED

Status follows from the set: one content payload is ACTIVE, one withdrawal is WITHDRAWN, more than
one payload is CONFLICTED.

Properties (proved in docs/PROOFS.md, checked by tests/test_ledger.py on random event sets)
-------------------------------------------------------------------------------------------
P1  Idempotence     delivering an event twice leaves the state unchanged.
P2  Commutativity   any delivery order of the same events yields the same state.
P3  No resurrection an event with a lower revision never changes a record, so a late copy of an
                    old document cannot undo a withdrawal or a correction.
P4  No silent win   two different payloads claiming the same revision never resolve by arrival
                    order; the record becomes CONFLICTED until a higher revision arrives.

Arrival time is stored and never used to decide which payload is current. Whether one *record*
outranks a different record about the same question is a policy decision made downstream from
declared authority and validity, not by the ledger.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from ease.util import text_hash


class Status(str, Enum):
    ACTIVE = "active"
    WITHDRAWN = "withdrawn"
    CONFLICTED = "conflicted"
    EXPIRED = "expired"  # derived at read time from validity and the supplied clock
    NOT_YET_VALID = "not_yet_valid"
    ABSENT = "absent"


class Outcome(str, Enum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"
    STALE = "stale"
    CONFLICT = "conflict"


WITHDRAWAL = "__withdrawn__"


@dataclass(frozen=True)
class Event:
    """One delivery. `text=None` is a withdrawal of the record at this revision."""

    record_id: str
    revision: int
    text: Optional[str]
    source_id: str = ""
    authority: int = 0
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    span: Optional[str] = None
    observed_at: Optional[float] = None
    meta: dict = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self):
        if not self.record_id:
            raise ValueError("record_id must be non-empty")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if self.valid_from is not None and self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError("valid_until precedes valid_from")

    @property
    def is_withdrawal(self) -> bool:
        return self.text is None

    def payload_key(self) -> str:
        """Identity of what this event asserts. Arrival time and free-form metadata are excluded:
        the same document delivered twice must compare equal."""
        if self.is_withdrawal:
            return WITHDRAWAL
        return text_hash(
            self.text, self.source_id, str(self.authority),
            "" if self.valid_from is None else repr(float(self.valid_from)),
            "" if self.valid_until is None else repr(float(self.valid_until)),
        )


@dataclass(frozen=True)
class Payload:
    key: str
    text: Optional[str]
    source_id: str
    authority: int
    valid_from: Optional[float]
    valid_until: Optional[float]
    span: Optional[str]

    @property
    def is_withdrawal(self) -> bool:
        return self.key == WITHDRAWAL


@dataclass(frozen=True)
class RecordState:
    record_id: str
    revision: int
    payloads: tuple[Payload, ...]  # sorted by key, so equal states compare equal

    @property
    def stored_status(self) -> Status:
        if len(self.payloads) > 1:
            return Status.CONFLICTED
        return Status.WITHDRAWN if self.payloads[0].is_withdrawal else Status.ACTIVE

    def status(self, now: Optional[float] = None) -> Status:
        s = self.stored_status
        if s is not Status.ACTIVE or now is None:
            return s
        p = self.payloads[0]
        if p.valid_from is not None and now < p.valid_from:
            return Status.NOT_YET_VALID
        if p.valid_until is not None and now >= p.valid_until:
            return Status.EXPIRED
        return Status.ACTIVE

    @property
    def payload(self) -> Optional[Payload]:
        return self.payloads[0] if len(self.payloads) == 1 else None

    def fingerprint(self, now: Optional[float] = None) -> str:
        """Changes exactly when something a decision could depend on changes."""
        return text_hash(self.record_id, str(self.revision), "|".join(p.key for p in self.payloads),
                         self.status(now).value)


def merge(state: Optional[RecordState], ev: Event, key: Optional[str] = None) -> tuple[RecordState, Outcome]:
    """Pure merge function. All ledger semantics live here.

    `key` lets a rebuild reuse the payload identity recorded when the event was first delivered,
    which stays correct after the stored text has been erased.
    """
    pl = Payload(key or ev.payload_key(), ev.text, ev.source_id, ev.authority, ev.valid_from, ev.valid_until, ev.span)
    if state is None or ev.revision > state.revision:
        return RecordState(ev.record_id, ev.revision, (pl,)), Outcome.APPLIED
    if ev.revision < state.revision:
        return state, Outcome.STALE
    if any(p.key == pl.key for p in state.payloads):
        return state, Outcome.DUPLICATE
    merged = tuple(sorted(state.payloads + (pl,), key=lambda p: p.key))
    return RecordState(ev.record_id, ev.revision, merged), Outcome.CONFLICT


@dataclass(frozen=True)
class Receipt:
    seq: int
    record_id: str
    revision: int
    outcome: Outcome
    changed: bool  # True when the record's state differs from before this delivery
    status: Status


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    payload_key TEXT NOT NULL,
    text TEXT,
    source_id TEXT NOT NULL,
    authority INTEGER NOT NULL,
    valid_from REAL,
    valid_until REAL,
    span TEXT,
    observed_at REAL NOT NULL,
    meta TEXT NOT NULL,
    outcome TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_record ON events(task_id, record_id, revision);
"""


class Ledger:
    """Append-only event log plus the merged state, for one task.

    Backed by SQLite so a crash cannot leave a half-applied event. The merged state is rebuilt from
    the log on open; because merge is order-independent, the rebuild cannot disagree with the
    state that was live before the restart.
    """

    def __init__(self, path: str = ":memory:", task_id: str = "default"):
        self.task_id = task_id
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_SCHEMA)
        self._state: dict[str, RecordState] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        cur = self._db.execute(
            "SELECT record_id, revision, payload_key, text, source_id, authority, valid_from, valid_until, span, "
            "observed_at, meta FROM events WHERE task_id=? ORDER BY seq", (self.task_id,))
        self._state = {}
        for rid, rev, key, text, src, auth, vf, vu, span, obs, meta in cur:
            if key == WITHDRAWAL:
                text = None
            ev = Event(rid, rev, text, src, auth, vf, vu, span, obs, json.loads(meta))
            self._state[rid], _ = merge(self._state.get(rid), ev, key=key)

    # -- writes ------------------------------------------------------------
    def deliver(self, ev: Event) -> Receipt:
        with self._lock:
            before = self._state.get(ev.record_id)
            after, outcome = merge(before, ev)
            observed = time.time() if ev.observed_at is None else ev.observed_at
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cur = self._db.execute(
                    "INSERT INTO events(task_id, record_id, revision, payload_key, text, source_id, authority, "
                    "valid_from, valid_until, span, observed_at, meta, outcome) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self.task_id, ev.record_id, ev.revision, ev.payload_key(), ev.text, ev.source_id, ev.authority,
                     ev.valid_from, ev.valid_until, ev.span, observed, json.dumps(ev.meta, default=str),
                     outcome.value))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._state[ev.record_id] = after
            return Receipt(cur.lastrowid, ev.record_id, ev.revision, outcome, after != before, after.stored_status)

    def put(self, record_id: str, revision: int, text: str, **kw) -> Receipt:
        if text is None:
            raise ValueError("put() needs text; use withdraw() to retract a record")
        return self.deliver(Event(record_id, revision, text, **kw))

    def withdraw(self, record_id: str, revision: int, **kw) -> Receipt:
        return self.deliver(Event(record_id, revision, None, **kw))

    def purge(self, record_id: str) -> Optional[Receipt]:
        """Honour a deletion request: withdraw the record, then erase every stored copy of its text.

        The withdrawal is an ordinary event at the next revision, so every decision that used the
        record is recomputed through the normal path. Event rows and payload hashes remain for the
        audit trail; the words do not. This removes the text from this ledger only. It does not
        reach copies held elsewhere, and it does not alter model weights.
        """
        with self._lock:
            st = self._state.get(record_id)
            if st is None:
                return None
            receipt = None
            if st.stored_status is not Status.WITHDRAWN:
                receipt = self.withdraw(record_id, st.revision + 1, source_id="purge")
            self._db.execute(
                "UPDATE events SET text='', span=NULL WHERE task_id=? AND record_id=? AND text IS NOT NULL",
                (self.task_id, record_id))
            return receipt

    # -- reads -------------------------------------------------------------
    def get(self, record_id: str) -> Optional[RecordState]:
        return self._state.get(record_id)

    def status(self, record_id: str, now: Optional[float] = None) -> Status:
        st = self._state.get(record_id)
        return Status.ABSENT if st is None else st.status(now)

    def records(self) -> dict[str, RecordState]:
        return dict(self._state)

    def active(self, now: Optional[float] = None) -> dict[str, Payload]:
        """Records that may currently be used as evidence."""
        return {rid: st.payloads[0] for rid, st in self._state.items() if st.status(now) is Status.ACTIVE}

    def snapshot(self) -> dict[str, tuple[int, tuple[str, ...]]]:
        """Canonical state, independent of delivery order. Two ledgers agree iff snapshots are equal."""
        return {rid: (st.revision, tuple(p.key for p in st.payloads)) for rid, st in sorted(self._state.items())}

    def next_transition_after(self, now: float) -> Optional[float]:
        """Earliest future instant at which some record's status changes because of validity alone.
        The engine must re-evaluate no later than this, even if no event arrives."""
        times = []
        for st in self._state.values():
            p = st.payload
            if p is None or p.is_withdrawal:
                continue
            for t in (p.valid_from, p.valid_until):
                if t is not None and t > now:
                    times.append(t)
        return min(times) if times else None

    def history(self, record_id: Optional[str] = None) -> list[dict]:
        q = "SELECT seq, record_id, revision, payload_key, source_id, authority, observed_at, outcome FROM events WHERE task_id=?"
        args: list = [self.task_id]
        if record_id is not None:
            q += " AND record_id=?"
            args.append(record_id)
        cols = ["seq", "record_id", "revision", "payload_key", "source_id", "authority", "observed_at", "outcome"]
        return [dict(zip(cols, row)) for row in self._db.execute(q + " ORDER BY seq", args)]

    def close(self) -> None:
        self._db.close()


def replay(events: Iterable[Event]) -> dict[str, RecordState]:
    """Merged state of a set of events with no storage. Used by tests and proofs."""
    state: dict[str, RecordState] = {}
    for ev in events:
        state[ev.record_id], _ = merge(state.get(ev.record_id), ev)
    return state
