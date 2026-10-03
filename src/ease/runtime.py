"""The deployed system: models, tasks, feedback and versions, persisted under one directory.

    data_dir/
      tasks/<task_id>/schema.json       the declared task
      tasks/<task_id>/ledger.sqlite     its evidence (append-only log)
      edge_cache.sqlite                 edge readings, keyed by weights version and texts
      memory.sqlite                     verified readings (retractable)
      versions/                         adopted refiner/memory bundles, with lineage
      tracker.json                      endorsement threshold and its error bound
      audit.jsonl                       every change made through the runtime

Everything the runtime does to a task goes through `Engine`, so the guarantees in docs/PROOFS.md
apply. The runtime never executes an action. It reports READY only when the requirement's
probability of being satisfied reaches the tracker's current threshold.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ease.aggregate import RefinedAggregator, RuleAggregator
from ease.engine import Engine, EngineConfig, pair_key
from ease.evolve.aggregator import EvolvingAggregator
from ease.evolve.consolidate import Bundle, GateConfig, Readings, VersionStore, consolidate, load_readings
from ease.evolve.memory import CorrectionMemory, MemoryConfig
from ease.evolve.threshold import ThresholdTracker
from ease.ledger import Event, Ledger
from ease.planner import best_question, certify_stable, unnecessary_questions
from ease.schema import TaskSchema
from ease.util import JsonlLogger, atomic_write_json, read_json

TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass
class Limits:
    max_text_chars: int = 20_000
    max_records_per_task: int = 5_000
    max_predicates: int = 200
    max_tasks: int = 1_000


class PersistentEdgeCache(dict):
    """dict backed by SQLite. Values are (logits float32[3], msg float32[d])."""

    def __init__(self, path: str):
        super().__init__()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS edges (k TEXT PRIMARY KEY, logits BLOB NOT NULL, msg BLOB NOT NULL)")
        self._lock = threading.Lock()

    def __contains__(self, k) -> bool:
        if dict.__contains__(self, k):
            return True
        with self._lock:
            row = self._db.execute("SELECT logits, msg FROM edges WHERE k=?", (k,)).fetchone()
        if row is None:
            return False
        dict.__setitem__(self, k, (np.frombuffer(row[0], np.float32).copy(), np.frombuffer(row[1], np.float32).copy()))
        return True

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return dict.__getitem__(self, k)

    def __setitem__(self, k, v) -> None:
        dict.__setitem__(self, k, v)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO edges(k, logits, msg) VALUES (?,?,?)",
                             (k, np.asarray(v[0], np.float32).tobytes(), np.asarray(v[1], np.float32).tobytes()))

    def pop(self, k, default=None):
        with self._lock:
            self._db.execute("DELETE FROM edges WHERE k=?", (k,))
        return dict.pop(self, k, default)

    def clear(self) -> None:
        """Drops the in-memory layer only. Stored readings stay valid: they are keyed by weights version."""
        dict.clear(self)

    def stored(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]


class Runtime:
    def __init__(self, data_dir: str | Path, scorer, base_aggregator=None, limits: Optional[Limits] = None,
                 gate: Optional[GateConfig] = None, alpha: float = 0.05, eta: float = 0.02, tau: float = 0.9,
                 regression_suite: Optional[str | Path] = None):
        self.dir = Path(data_dir)
        self.regression_suite = Path(regression_suite) if regression_suite else None
        (self.dir / "tasks").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.limits = limits or Limits()
        self.gate = gate or GateConfig()
        self.audit = JsonlLogger(self.dir / "audit.jsonl")
        self.scorer = scorer
        if hasattr(scorer, "cache") and not isinstance(scorer.cache, PersistentEdgeCache):
            scorer.cache = PersistentEdgeCache(str(self.dir / "edge_cache.sqlite"))
        self.versions = VersionStore(self.dir / "versions")
        self.memory = CorrectionMemory(str(self.dir / "memory.sqlite"))
        if self.versions.current() is None:
            if isinstance(base_aggregator, RefinedAggregator):
                self.versions.save(Bundle(base_aggregator.refiner, MemoryConfig(), "trained"), note="trained model")
            elif base_aggregator is None or isinstance(base_aggregator, RuleAggregator):
                self.rule = base_aggregator or RuleAggregator(msg_dim=scorer.msg_dim)
        self._load_bundle()
        tp = self.dir / "tracker.json"
        self.tracker = ThresholdTracker.load(tp) if tp.exists() else ThresholdTracker(alpha=alpha, eta=eta, tau=tau,
                                                                                       tau_initial=tau)
        self.engines: dict[str, Engine] = {}
        self.pending: dict[tuple[str, str], float] = {}  # (task, action) -> confidence at endorsement

    # ------------------------------------------------------------------ models
    def _load_bundle(self) -> None:
        if self.versions.current() is not None:
            self.bundle = self.versions.load()
            self.memory.set_config(self.bundle.memory)
            base = RefinedAggregator(self.bundle.refiner)
        else:
            self.bundle = None
            base = self.rule
        self.aggregator = EvolvingAggregator(base, self.memory)

    def _config(self) -> EngineConfig:
        return EngineConfig(ready_threshold=max(0.5, self.tracker.tau), block_threshold=0.5)

    def _refresh_all(self) -> None:
        for eng in self.engines.values():
            eng.set_aggregator(self.aggregator)
            eng.set_config(self._config())

    # ------------------------------------------------------------------ tasks
    def _task_dir(self, task_id: str) -> Path:
        if not TASK_ID.match(task_id):
            raise ValueError("task id must be 1-64 characters: letters, digits, '_', '.', '-'")
        return self.dir / "tasks" / task_id

    def list_tasks(self) -> list[str]:
        return sorted(p.name for p in (self.dir / "tasks").iterdir() if (p / "schema.json").exists())

    def create_task(self, schema: TaskSchema, now: Optional[float] = None) -> Engine:
        with self.lock:
            d = self._task_dir(schema.task_id)
            if (d / "schema.json").exists():
                raise FileExistsError(f"task {schema.task_id!r} already exists")
            if len(self.list_tasks()) >= self.limits.max_tasks:
                raise ValueError("task limit reached")
            if len(schema.predicates) > self.limits.max_predicates:
                raise ValueError("too many predicates")
            d.mkdir(parents=True, exist_ok=True)
            atomic_write_json(d / "schema.json", schema.to_dict())
            self.audit.log(kind="create_task", task=schema.task_id, schema_version=schema.version())
            return self.engine(schema.task_id, now)

    def engine(self, task_id: str, now: Optional[float] = None) -> Engine:
        with self.lock:
            if task_id in self.engines:
                return self.engines[task_id]
            d = self._task_dir(task_id)
            if not (d / "schema.json").exists():
                raise KeyError(f"no task {task_id!r}")
            schema = TaskSchema.from_dict(read_json(d / "schema.json"))
            eng = Engine(schema, self.scorer, self.aggregator, ledger=Ledger(str(d / "ledger.sqlite"), task_id),
                         config=self._config(), now=now)
            self.engines[task_id] = eng
            return eng

    def update_schema(self, schema: TaskSchema) -> dict:
        with self.lock:
            eng = self.engine(schema.task_id)
            if len(schema.predicates) > self.limits.max_predicates:
                raise ValueError("too many predicates")
            rep = eng.update_schema(schema)
            atomic_write_json(self._task_dir(schema.task_id) / "schema.json", schema.to_dict())
            self.audit.log(kind="update_schema", task=schema.task_id, schema_version=schema.version())
            return rep.as_dict()

    # ------------------------------------------------------------------ evidence
    def deliver(self, task_id: str, event: Event) -> dict:
        with self.lock:
            if event.text is not None and len(event.text) > self.limits.max_text_chars:
                raise ValueError(f"text longer than {self.limits.max_text_chars} characters; split it into passages")
            eng = self.engine(task_id)
            if event.record_id not in eng.ledger.records() and len(eng.ledger.records()) >= self.limits.max_records_per_task:
                raise ValueError("record limit reached for this task")
            rep = eng.deliver(event)
            self.audit.log(kind="deliver", task=task_id, record=event.record_id, revision=event.revision,
                           outcome=rep.receipt.outcome.value, changed=rep.receipt.changed,
                           edges_scored=rep.edges_scored, changed_actions=rep.changed_actions)
            return rep.as_dict()

    def ingest_document(self, task_id: str, doc_id: str, revision: int, text: str, source_id: str = "",
                        authority: int = 0, attribution: Optional[str] = None,
                        valid_from: Optional[float] = None) -> dict:
        """Split a document into passages and deliver each as a record. See ease/ingest.py."""
        from ease.ingest import ingest

        with self.lock:
            if len(text) > 50 * self.limits.max_text_chars:
                raise ValueError("document too long")
            eng = self.engine(task_id)
            rep = ingest(lambda ev: self.deliver(task_id, ev), eng.ledger.records(), doc_id, revision, text,
                         source_id, authority, attribution, valid_from)
            return rep.as_dict()

    def set_now(self, task_id: str, now: float) -> dict:
        with self.lock:
            return self.engine(task_id).set_now(now).as_dict()

    # ------------------------------------------------------------------ queries
    def state(self, task_id: str) -> dict:
        with self.lock:
            eng = self.engine(task_id)
            out = {"task_id": task_id, "schema_version": eng.schema.version(),
                   "assessments": {k: self._assessment(eng, k) for k in eng.assessments()},
                   "beliefs": eng.beliefs(),
                   "records": {rid: {"revision": st.revision, "status": st.status(eng.now).value}
                               for rid, st in sorted(eng.ledger.records().items())},
                   "threshold": self.tracker.state(),
                   "next_validity_transition": eng.ledger.next_transition_after(eng.now) if eng.now is not None else None}
            return out

    def _assessment(self, eng: Engine, aid: str) -> dict:
        a = eng.assessment(aid).as_dict()
        a["endorsed"] = a["disposition"] == "READY"
        a["confidence"] = a["status"]["satisfied"]
        if self.tracker.stalled:
            a["note"] = ("The error target cannot currently be met, so nothing is endorsed. "
                         "Candidates are shown for review only.")
        return a

    def explain(self, task_id: str, action_id: str) -> dict:
        with self.lock:
            return self.engine(task_id).explain(action_id)

    def question(self, task_id: str, max_set: int = 3) -> dict:
        with self.lock:
            eng = self.engine(task_id)
            q = best_question(eng, max_set=max_set)
            return {"question": None if q is None else q.as_dict(), "unnecessary": unnecessary_questions(eng)}

    def certificate(self, task_id: str, action_id: str, free: list[str]) -> dict:
        with self.lock:
            return certify_stable(self.engine(task_id), action_id, free).as_dict()

    def verify(self, task_id: str) -> dict:
        with self.lock:
            return self.engine(task_id).verify(bypass_cache=True)

    # ------------------------------------------------------------------ learning from people
    def correct_reading(self, task_id: str, record_id: str, predicate_id: str, label: int, source: str = "user") -> dict:
        """A person states what a record establishes about a predicate. Takes effect immediately."""
        with self.lock:
            eng = self.engine(task_id)
            node = f"edge:{record_id}|{predicate_id}"
            if node not in eng.g.nodes or eng.g.value(node) is None:
                raise KeyError("that record is not currently compared with that predicate")
            e = eng.g.value(node)
            item = self.memory.add(e["pair"], int(label), e["msg"], source=source)
            before = {k: v.disposition for k, v in eng.assessments().items()}
            self._refresh_all()
            after = {k: v.disposition for k, v in eng.assessments().items()}
            self.audit.log(kind="correct_reading", task=task_id, record=record_id, predicate=predicate_id,
                           label=int(label), item=item, logits=e["logits"].tolist())
            return {"memory_item": item, "aggregator_version": self.aggregator.version,
                    "changed_actions": [(k, before[k], after[k]) for k in after if before[k] != after[k]]}

    def retract_correction(self, item_id: int) -> dict:
        with self.lock:
            ok = self.memory.remove(item_id)
            if ok:
                self._refresh_all()
            self.audit.log(kind="retract_correction", item=item_id, removed=ok)
            return {"removed": ok, "aggregator_version": self.aggregator.version}

    def endorse(self, task_id: str, action_id: str) -> dict:
        """Register that a READY action was shown to a person, so that a verdict can follow."""
        with self.lock:
            a = self.engine(task_id).assessment(action_id)
            conf = float(a.status[0])
            if a.disposition != "READY" or not self.tracker.endorse(conf):
                return {"endorsed": False, "confidence": conf, "threshold": self.tracker.tau}
            self.pending[(task_id, action_id)] = conf
            self.tracker.save(self.dir / "tracker.json")
            return {"endorsed": True, "confidence": conf, "threshold": self.tracker.tau,
                    "needs_approval": a.needs_approval}

    def verdict(self, task_id: str, action_id: str, was_wrong: bool) -> dict:
        with self.lock:
            if (task_id, action_id) not in self.pending:
                raise KeyError("no endorsement of that action is awaiting a verdict")
            self.pending.pop((task_id, action_id))
            self.tracker.verdict(bool(was_wrong))
            self.tracker.save(self.dir / "tracker.json")
            self._refresh_all()
            self.audit.log(kind="verdict", task=task_id, action=action_id, was_wrong=bool(was_wrong),
                           tau=self.tracker.tau)
            return self.tracker.state()

    # ------------------------------------------------------------------ consolidation
    def feedback_readings(self) -> Readings:
        items = self.memory.items
        if not items:
            return Readings([], np.zeros(0, np.int64), np.zeros((0, 3), np.float32), np.zeros((0, 1), np.float32))
        latest = {}
        for it in items:
            latest[it.pair] = it
        rows = list(latest.values())
        # the logits of each verified pair were written to the audit log when it was corrected
        logits = {}
        for line in open(self.dir / "audit.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("kind") == "correct_reading":
                logits[r["item"]] = r["logits"]
        keep = [it for it in rows if it.item_id in logits]
        return Readings([it.pair for it in keep], np.asarray([it.label for it in keep], np.int64),
                        np.asarray([logits[it.item_id] for it in keep], np.float32).reshape(-1, 3),
                        np.stack([it.msg for it in keep]).astype(np.float32) if keep else np.zeros((0, 1), np.float32))

    def export_feedback(self, path: str | Path) -> dict:
        """Write verified readings as JSONL ({"claim", "evidence", "label"}) for a continued training
        run. Texts are taken from the ledger as it is now, so a correction about a record that has
        since been purged or revised is left out. The file contains people's documents: treat it
        accordingly and delete it after use."""
        with self.lock:
            active = {it.item_id: it for it in self.memory.items}
            rows, skipped = [], 0
            for line in open(self.dir / "audit.jsonl", encoding="utf-8"):
                r = json.loads(line)
                if r.get("kind") != "correct_reading" or r.get("item") not in active or "task" not in r:
                    continue
                it = active[r["item"]]
                try:
                    eng = self.engine(r["task"])
                    claim = eng.schema.predicate(r["predicate"]).text
                    payload = eng.ledger.active(eng.now).get(r["record"])
                except KeyError:
                    payload = None
                if payload is None or not payload.text or pair_key(claim, payload.text) != it.pair:
                    skipped += 1
                    continue
                rows.append({"claim": claim, "evidence": payload.text, "label": it.label, "page": f"{r['task']}:{r['record']}"})
            latest = {(x["claim"], x["evidence"]): x for x in rows}
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for x in latest.values():
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            self.audit.log(kind="export_feedback", rows=len(latest), skipped=skipped)
            return {"rows": len(latest), "skipped_because_record_changed_or_gone": skipped, "path": str(path)}

    def consolidate(self, regression: dict[str, Readings], replay: Optional[Readings] = None) -> dict:
        """Try to learn from accumulated feedback. Adopts the result only if the gate passes."""
        with self.lock:
            if self.bundle is None:
                return {"adopted": False, "reason": "no trained refiner is loaded; nothing to consolidate"}
            new, rep = consolidate(self.bundle, self.feedback_readings(), regression, replay, self.gate)
            if rep.adopted:
                self.versions.save(new, rep, note=f"adopted after {rep.feedback} verified readings")
                self._load_bundle()
                self.tracker = ThresholdTracker(alpha=self.tracker.alpha, eta=self.tracker.eta,
                                                tau=self.tracker.tau_initial, tau_initial=self.tracker.tau_initial)
                self.tracker.save(self.dir / "tracker.json")
                self._refresh_all()
            self.audit.log(kind="consolidate", adopted=rep.adopted, chosen=rep.chosen, reason=rep.reason,
                           feedback=rep.feedback, version=self.versions.current())
            return rep.as_dict()

    def consolidate_now(self) -> dict:
        """Consolidate against the regression suite shipped with the model."""
        if self.regression_suite is None or not self.regression_suite.exists():
            return {"adopted": False, "reason": "no regression suite is configured, so no change can be verified "
                                                "as safe; nothing was changed"}
        sets = load_readings(self.regression_suite)
        replay = sets.pop("replay", None)
        return self.consolidate(sets, replay)

    def rollback(self) -> dict:
        with self.lock:
            v = self.versions.rollback()
            if v is not None:
                self._load_bundle()
                self._refresh_all()
            self.audit.log(kind="rollback", version=v)
            return {"version": v, "lineage": self.versions.lineage()}

    def info(self) -> dict:
        return {"edge_version": self.scorer.version, "aggregator_version": self.aggregator.version,
                "bundle_version": self.versions.current(), "lineage": self.versions.lineage(),
                "memory_items": len(self.memory), "memory": dict(self.memory.cfg.__dict__),
                "threshold": self.tracker.state(), "tasks": self.list_tasks(),
                "edge_cache_rows": self.scorer.cache.stored() if isinstance(getattr(self.scorer, "cache", None),
                                                                           PersistentEdgeCache) else None}

    def close(self) -> None:
        with self.lock:
            for eng in self.engines.values():
                eng.ledger.close()
            self.engines.clear()
            self.memory.close()
