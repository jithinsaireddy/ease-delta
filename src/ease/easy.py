"""The short way in.

    from ease import EvidenceReader, Tracker

    reader = EvidenceReader.from_pretrained()        # downloads jithinpothireddy21/ease-delta once
    reader.read("The client has approved the design.",
                "Email from the client: we approve the design, please go ahead.")

    task = Tracker.from_template("client-handoff", reader=reader, client="Acme")
    task.add("mail-1", "Email from Acme: we approve the final design, please go ahead.")
    print(task.status())

`EvidenceReader` is the trained reader with the calibration the system uses. `Tracker` keeps one task
current: you add, change and withdraw records, and it reports which actions are ready, blocked or waiting,
why, and which question is worth asking. Both are thin layers over `ease.runtime.Runtime` and
`ease.engine.Engine`, so the guarantees in docs/PROOFS.md hold unchanged.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ease.util import read_json

DEFAULT_MODEL = "jithinpothireddy21/ease-delta"
NAMES = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
ESTABLISHES = {"supports": 0, "refutes": 1, "settles_nothing": 2, "not_enough_info": 2}


# ---------------------------------------------------------------------------
# Reading one claim against one passage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Reading:
    """Calibrated probabilities that a passage supports a claim, refutes it, or does not settle it."""

    supports: float
    refutes: float
    not_enough_info: float

    @property
    def label(self) -> str:
        return NAMES[int(np.argmax([self.supports, self.refutes, self.not_enough_info]))]

    def __repr__(self) -> str:
        return (f"Reading({self.label}: supports {self.supports:.2f}, refutes {self.refutes:.2f}, "
                f"not enough info {self.not_enough_info:.2f})")


class EvidenceReader:
    """The trained reader of an EASE-Delta release, with the calibration the system applies to it."""

    def __init__(self, scorer, aggregator=None, settings: Optional[dict] = None, calibration=None,
                 source: Optional[str] = None):
        self.scorer = scorer
        self.aggregator = aggregator
        self.settings = dict(settings or {})
        self.calibration = calibration
        self.source = source
        # One reading cache for every tracker built on this reader. Readings are keyed by the weights
        # version and hashes of the texts, so sharing them is safe; keeping them in memory means no tracker
        # owns a file another one still uses.
        from ease.runtime import PersistentEdgeCache

        if isinstance(getattr(scorer, "cache", None), dict) and not isinstance(scorer.cache, PersistentEdgeCache):
            cache = PersistentEdgeCache(":memory:")
            for k, v in scorer.cache.items():
                cache[k] = v
            scorer.cache = cache

    @classmethod
    def from_pretrained(cls, name_or_path: Optional[str] = None, device: Optional[str] = None, exact: bool = True,
                        revision: Optional[str] = None) -> "EvidenceReader":
        """A release directory, or a Hugging Face model id whose files are downloaded once and cached.

        Without an argument: the `EASE_MODEL` environment variable if set, else jithinpothireddy21/ease-delta.
        jithinpothireddy21/ease-delta-base is the smaller, faster variant.

        `exact=True` fixes the shape of every forward pass, so a reading never depends on what else was read
        with it and cached values equal a full rebuild bit for bit (docs/RESULTS.md, section 4); `exact=False`
        batches freely and is faster, with values that vary by about 1e-5.
        """
        from ease.aggregate import Calibration, load_aggregator
        from ease.scorer import ModelScorer

        name_or_path = name_or_path or os.environ.get("EASE_MODEL") or DEFAULT_MODEL
        path = Path(name_or_path).expanduser()
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(name_or_path, revision=revision))
        edge = path / "edge" if (path / "edge" / "edge_config.json").exists() else path
        if not (edge / "edge_config.json").exists():
            raise FileNotFoundError(f"{name_or_path} is not an EASE-Delta release (no edge/edge_config.json). "
                                    "For the plain transformers classifier use jithinpothireddy21/ease-delta-reader "
                                    "with transformers' pipeline instead.")
        scorer = ModelScorer(str(edge), device=device, canonical=exact)
        agg_dir = path / "aggregator"
        aggregator = load_aggregator(agg_dir) if (agg_dir / "aggregator.json").exists() else None
        calibration = None
        if (agg_dir / "rule_calibration.json").exists():
            c = read_json(agg_dir / "rule_calibration.json")
            calibration = Calibration(float(c["temperature"]), tuple(float(x) for x in c["bias"]))
        settings = read_json(path / "settings.json") if (path / "settings.json").exists() else {}
        return cls(scorer, aggregator, settings, calibration, source=str(name_or_path))

    def read_many(self, pairs: Iterable[tuple[str, str]]) -> list[Reading]:
        """(claim, passage) pairs, read in one go. The claim comes first."""
        pairs = [(str(c), str(e)) for c, e in pairs]
        out = []
        for logits, _ in self.scorer.score(pairs):
            z = np.asarray(logits, np.float64)[None]
            if self.calibration is not None:
                p = self.calibration.apply_np(z)[0]
            else:
                e = np.exp(z - z.max())
                p = (e / e.sum())[0]
            out.append(Reading(float(p[0]), float(p[1]), float(p[2])))
        return out

    def read(self, claim: str, passage: str) -> Reading:
        """What does `passage` establish about `claim`?"""
        return self.read_many([(claim, passage)])[0]

    def __repr__(self) -> str:
        return f"EvidenceReader({self.source or 'custom'}, version {getattr(self.scorer, 'version', '?')})"


# ---------------------------------------------------------------------------
# Requirements written as expressions
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:(\()|(\))|(,)|(\d+)|([A-Za-z_][A-Za-z0-9_]*))")
_KEYWORDS = {"and", "or", "not", "at", "least", "of"}


def parse_requirement(text: str, known: Sequence[str]):
    """Parse e.g. "approved and (paid or waived) and not on_hold" or "at least 2 of (ann, bo, cy)".

    Returns a tree of tuples: ("pred", id) | ("and", [..]) | ("or", [..]) | ("not", x) | ("atleast", k, [..]).
    Keywords are case-insensitive; requirement ids are matched exactly.
    """
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot read {text[pos:]!r} in requirement {text!r}")
        tokens.append(next(g for g in m.groups() if g is not None))
        pos = m.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    known = set(known)
    i = 0

    def peek(k=0):
        return tokens[i + k].lower() if i + k < len(tokens) else None

    def take(expected=None):
        nonlocal i
        if i >= len(tokens):
            raise ValueError(f"requirement {text!r} ends too early")
        t = tokens[i]
        if expected is not None and t.lower() != expected:
            raise ValueError(f"expected {expected!r} but found {t!r} in requirement {text!r}")
        i += 1
        return t

    def expr():
        parts = [term()]
        while peek() == "or":
            take()
            parts.append(term())
        return parts[0] if len(parts) == 1 else ("or", parts)

    def term():
        parts = [factor()]
        while peek() == "and":
            take()
            parts.append(factor())
        return parts[0] if len(parts) == 1 else ("and", parts)

    def factor():
        t = peek()
        if t == "not":
            take()
            return ("not", factor())
        if t == "(":
            take()
            e = expr()
            take(")")
            return e
        if t == "at" and peek(1) == "least":
            take(), take()
            k = take()
            if not k.isdigit():
                raise ValueError(f"'at least' needs a number in requirement {text!r}")
            take("of")
            take("(")
            items = [expr()]
            while peek() == ",":
                take()
                items.append(expr())
            take(")")
            if not 1 <= int(k) <= len(items):
                raise ValueError(f"'at least {k} of' needs between 1 and {len(items)} in requirement {text!r}")
            return ("atleast", int(k), items)
        name = take()
        if name.lower() in _KEYWORDS or name in "(),":
            raise ValueError(f"unexpected {name!r} in requirement {text!r}")
        if name not in known:
            raise ValueError(f"unknown requirement {name!r} in {text!r}; known: {sorted(known)}")
        return ("pred", name)

    tree = expr()
    if i != len(tokens):
        raise ValueError(f"unexpected {tokens[i]!r} in requirement {text!r}")
    return tree


def _flatten(tree):
    kind = tree[0]
    if kind in ("and", "or"):
        out = []
        for c in tree[1]:
            c = _flatten(c)
            out.extend(c[1] if c[0] == kind else [c])
        return (kind, out)
    if kind == "not":
        return ("not", _flatten(tree[1]))
    if kind == "atleast":
        return ("atleast", tree[1], [_flatten(c) for c in tree[2]])
    return tree


def compile_requirement(action_id: str, text: str, known: Sequence[str], gates: list[dict]) -> str:
    """Add the gates for `text` to `gates` and return the id the action requires."""
    counter = [0]

    def emit(node) -> str:
        if node[0] == "pred":
            return node[1]
        counter[0] += 1
        gid = f"{action_id}__{counter[0]}"
        if node[0] == "not":
            gates.append({"id": gid, "op": "NOT", "children": [emit(node[1])]})
        elif node[0] == "atleast":
            gates.append({"id": gid, "op": "ATLEAST", "children": [emit(c) for c in node[2]], "k": node[1]})
        else:
            gates.append({"id": gid, "op": node[0].upper(), "children": [emit(c) for c in node[1]]})
        return gid

    return emit(_flatten(parse_requirement(text, known)))


def define(task_id: str, requirements: dict[str, str], actions: dict[str, object]) -> dict:
    """A task definition from plain parts.

    requirements: {"approved": "The client has approved the final design.", ...}
    actions:      {"send": "approved and paid"} or
                  {"send": {"requires": "approved and paid", "description": "Send the packet",
                            "value": 10, "cost": 25}}
    `value` is what doing the action is worth when its requirements hold and `cost` what it costs when
    they do not; together they decide which questions are worth asking.
    """
    preds = []
    for pid, spec in requirements.items():
        if isinstance(spec, str):
            preds.append({"id": pid, "text": spec})
        else:
            preds.append({"id": pid, **dict(spec)})
    gates: list[dict] = []
    acts = []
    for aid, spec in actions.items():
        if isinstance(spec, str):
            spec = {"requires": spec}
        spec = dict(spec)
        req = compile_requirement(aid, spec.pop("requires"), list(requirements), gates)
        acts.append({"id": aid, "description": spec.pop("description", aid.replace("_", " ")), "requires": req,
                     "value_success": float(spec.pop("value", 10.0)), "cost_failure": float(spec.pop("cost", 25.0)),
                     **spec})
    return {"task_id": task_id, "predicates": preds, "gates": gates, "actions": acts}


# ---------------------------------------------------------------------------
# What a tracker reports
# ---------------------------------------------------------------------------

@dataclass
class ActionStatus:
    id: str
    description: str
    disposition: str  # READY | BLOCKED | NEEDS_INFO
    confidence: float  # probability that its requirements are satisfied
    p_success: float


@dataclass
class RequirementStatus:
    id: str
    text: str
    status: str  # SATISFIED | VIOLATED | UNRESOLVED | CONFLICT
    satisfied: float
    violated: float
    open: float
    rests_on: list[str]


@dataclass
class Status:
    actions: dict[str, ActionStatus]
    requirements: dict[str, RequirementStatus]
    question: Optional[list[str]] = None  # requirement texts worth asking about together
    unnecessary: dict[str, list[str]] = field(default_factory=dict)

    def _ids(self, d):
        return [a for a, s in self.actions.items() if s.disposition == d]

    @property
    def ready(self) -> list[str]:
        return self._ids("READY")

    @property
    def blocked(self) -> list[str]:
        return self._ids("BLOCKED")

    @property
    def needs_info(self) -> list[str]:
        return self._ids("NEEDS_INFO")

    def __str__(self) -> str:
        w = max([len(k) for k in list(self.actions) + list(self.requirements)] + [12])
        lines = ["Actions"]
        for a in self.actions.values():
            lines.append(f"  {a.id:<{w}}  {a.disposition.replace('_', ' '):<10}  {100 * a.confidence:5.1f}% likely ready")
        lines.append("Requirements")
        for r in self.requirements.values():
            src = f"  rests on {', '.join(r.rests_on)}" if r.rests_on else ""
            lines.append(f"  {r.id:<{w}}  {r.status:<10}  satisfied {r.satisfied:.2f}  violated {r.violated:.2f}  "
                         f"open {r.open:.2f}{src}")
        if self.question:
            lines.append("Worth asking: " + "  /  ".join(self.question))
        return "\n".join(lines)

    __repr__ = __str__


@dataclass
class Update:
    record_id: str
    outcome: str  # applied | duplicate | stale | ...
    changed: list[tuple[str, str, str]]  # (action, before, after)
    pairs_read: int
    milliseconds: float

    def __str__(self) -> str:
        head = f"{self.record_id}: {self.outcome}, {self.pairs_read} pairs read, {self.milliseconds:.0f} ms"
        if not self.changed:
            return head + "; no proposal changed"
        return head + "; " + "; ".join(f"{a}: {b} -> {c}" for a, b, c in self.changed)

    __repr__ = __str__


# ---------------------------------------------------------------------------
# Keeping one task current
# ---------------------------------------------------------------------------

class Tracker:
    """One task, kept current as its records are added, changed and withdrawn.

    Records you add are read against every requirement (or only those named in `about`). Adding a record
    id that already exists replaces it with a new revision; `withdraw` removes it. Nothing is ever executed:
    a READY action is a proposal for a person to approve.
    """

    def __init__(self, definition, reader: EvidenceReader, data_dir: Optional[str | Path] = None,
                 ready_at: Optional[float] = None):
        from ease.runtime import Runtime
        from ease.schema import TaskSchema

        schema = definition if isinstance(definition, TaskSchema) else TaskSchema.from_dict(definition)
        self.reader = reader
        self._tmp = None
        if data_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="ease-")
            data_dir = self._tmp.name
        data_dir = Path(data_dir).expanduser()
        s = reader.settings
        tau = float(ready_at) if ready_at is not None else float(s.get("tau_initial", 0.9))
        self.runtime = Runtime(data_dir, reader.scorer, reader.aggregator, alpha=float(s.get("alpha", 0.05)),
                               eta=float(s.get("eta", 0.05)), tau=tau)
        self.task_id = schema.task_id
        if self.task_id in self.runtime.list_tasks():
            if self._engine().schema.version() != schema.version():
                self.runtime.update_schema(schema)
        else:
            self.runtime.create_task(schema)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_template(cls, template: str, reader: EvidenceReader, task_id: Optional[str] = None,
                      data_dir: Optional[str | Path] = None, ready_at: Optional[float] = None, **params) -> "Tracker":
        """A task from one of `ease.templates.TEMPLATES`, e.g. "client-onboarding", client="Acme"."""
        from ease.templates import render

        return cls(render(template, task_id or template, params), reader, data_dir, ready_at)

    @classmethod
    def define(cls, requirements: dict[str, str], actions: dict[str, object], reader: EvidenceReader,
               task_id: str = "task", data_dir: Optional[str | Path] = None,
               ready_at: Optional[float] = None) -> "Tracker":
        """A task from plain requirements and actions; see `ease.easy.define` for the format."""
        return cls(define(task_id, requirements, actions), reader, data_dir, ready_at)

    # -- evidence ----------------------------------------------------------
    def _engine(self):
        return self.runtime.engine(self.task_id)

    def _update(self, rep: dict, record_id: str) -> Update:
        led = rep.get("ledger") or {}
        return Update(record_id, led.get("outcome", "applied"), [tuple(c) for c in rep.get("changed_actions", [])],
                      int(rep.get("edges_scored", 0)), 1000.0 * float(rep.get("seconds", 0.0)))

    def add(self, record_id: str, text: str, *, source: str = "", authority: int = 0,
            about: Optional[Sequence[str]] = None, valid_from: Optional[float] = None) -> Update:
        """Add a record, or replace it if `record_id` exists. Say who is speaking in `text` itself
        ("Email from the client: ..."): the reader sees only the text. `about` limits which requirements
        the record is read against; `authority` (0-9) decides between records that disagree."""
        from ease.ledger import Event

        rec = self._engine().ledger.records().get(record_id)
        revision = rec.revision + 1 if rec is not None else 1
        ev = Event(record_id, revision, str(text), source_id=source, authority=int(authority),
                   valid_from=time.time() if valid_from is None else float(valid_from),
                   meta={"about": list(about)} if about else {})
        return self._update(self.runtime.deliver(self.task_id, ev), record_id)

    update = add

    def withdraw(self, record_id: str) -> Update:
        """Take a record back: it no longer counts for or against anything."""
        from ease.ledger import Event

        rec = self._engine().ledger.records().get(record_id)
        if rec is None:
            raise KeyError(f"no record {record_id!r}")
        ev = Event(record_id, rec.revision + 1, None)
        return self._update(self.runtime.deliver(self.task_id, ev), record_id)

    def add_document(self, doc_id: str, text: str, *, attribution: Optional[str] = None, source: str = "",
                     authority: int = 0) -> dict:
        """Cut a document into passages, each carrying `attribution` ("Email from the client"), and add them.
        Adding the same `doc_id` again replaces the document; passages that are gone are withdrawn."""
        recs = self._engine().ledger.records()
        revs = [st.revision for rid, st in recs.items() if rid.startswith(f"{doc_id}#")]
        return self.runtime.ingest_document(self.task_id, doc_id, (max(revs) + 1) if revs else 1, text, source,
                                            authority, attribution, time.time())

    def add_file(self, path: str | Path, *, attribution: Optional[str] = None, source: str = "",
                 authority: int = 0) -> list[dict]:
        """An .eml, .mbox, .txt or .md file. Emails keep their sender, date and subject in the text."""
        from ease.importers import parse_file

        p = Path(path)
        out = []
        for m in parse_file(p.read_bytes(), p.name, attribution):
            out.append(self.add_document(m.doc_id, m.text, attribution=m.attribution, source=source or m.sender[:80],
                                         authority=authority))
        return out

    # -- reading the state -------------------------------------------------
    def status(self) -> Status:
        eng = self._engine()
        st = self.runtime.state(self.task_id)
        q = self.runtime.question(self.task_id)
        acts = {}
        for aid, a in st["assessments"].items():
            acts[aid] = ActionStatus(aid, eng._action(aid).description, a["disposition"], float(a["confidence"]),
                                     float(a["p_success"]))
        reqs = {}
        for pid, b in st["beliefs"].items():
            p = b["probabilities"]
            reqs[pid] = RequirementStatus(pid, eng.schema.predicate(pid).text, b["status"], p["SATISFIED"],
                                          p["VIOLATED"], p["UNRESOLVED"] + p["CONFLICT"], list(b["decisive"]))
        question = q["question"]["texts"] if q["question"] else None
        return Status(acts, reqs, question, {k: v for k, v in q["unnecessary"].items() if v})

    def explain(self, action: str) -> str:
        """Which requirements an action rests on, and the records each of those rests on, as text."""
        ex = self.runtime.explain(self.task_id, action)
        a = ex["assessment"]
        lines = [f"{action}: {a['disposition']} ({100 * a['status']['satisfied']:.1f}% likely ready) - {ex['description']}"]
        for p in ex["predicates"]:
            lines.append(f"  {p['status']:<10} {p['text']}")
            if not p["records"]:
                lines.append(f"             no record settles this ({p['records_compared']} compared)")
            for r in p["records"]:
                lines.append(f"             {r['record_id']} (revision {r['revision']}): {r['text']}")
        return "\n".join(lines)

    def records(self) -> dict[str, dict]:
        """Every record with its current revision and whether it is active or withdrawn."""
        eng = self._engine()
        return {rid: {"revision": st.revision, "status": st.status(eng.now).value,
                      "text": st.payloads[0].text if len(st.payloads) == 1 else None}
                for rid, st in sorted(eng.ledger.records().items())}

    # -- people's answers and corrections -----------------------------------
    def confirm(self, requirement: str, by: str, holds: bool = True, note: str = "", authority: int = 0) -> Update:
        """Record a person's answer about one requirement. It enters as an attributed record that is read
        exactly, never interpreted: "Dana at Acme confirmed that this holds: <requirement>". Among records of
        equal authority the later one wins, so a later message can still overturn it."""
        pred = self._engine().schema.predicate(requirement)
        n = 1 + sum(1 for rid in self._engine().ledger.records() if rid.startswith(f"confirmation-{requirement}-"))
        rid = f"confirmation-{requirement}-{n}"
        said = "confirmed that this holds" if holds else "said that this does not hold"
        text = f"{by.strip() or 'Someone'} {said}: {pred.text}" + (f" Note: {note.strip()}" if note.strip() else "")
        up = self.add(rid, text, source=f"confirmation:{by}", authority=authority, about=[requirement])
        fix = self.runtime.correct_reading(self.task_id, rid, requirement, ESTABLISHES["supports" if holds else "refutes"],
                                           source=f"confirmation:{by}")
        return Update(rid, up.outcome, up.changed + [tuple(c) for c in fix["changed_actions"]], up.pairs_read,
                      up.milliseconds)

    def correct(self, record_id: str, requirement: str, establishes: str) -> int:
        """State what a record establishes about a requirement: "supports", "refutes" or "settles_nothing".
        Takes effect at once; returns an id for `undo`."""
        if establishes not in ESTABLISHES:
            raise ValueError(f"establishes must be one of {sorted(ESTABLISHES)}")
        return int(self.runtime.correct_reading(self.task_id, record_id, requirement, ESTABLISHES[establishes])["memory_item"])

    def undo(self, correction_id: int) -> bool:
        """Withdraw a correction; behaviour returns exactly to what it was."""
        return bool(self.runtime.retract_correction(int(correction_id))["removed"])

    # -- feedback on proposals -------------------------------------------------
    def endorse(self, action: str) -> bool:
        """Say that a READY action is being acted on, so that a verdict can follow. False if it is not READY."""
        return bool(self.runtime.endorse(self.task_id, action)["endorsed"])

    def verdict(self, action: str, was_wrong: bool) -> float:
        """Report whether an endorsed action turned out wrong. The readiness bar moves so that the error rate
        among endorsed actions stays near its target (docs/PROOFS.md, T5). Returns the new bar."""
        return float(self.runtime.verdict(self.task_id, action, bool(was_wrong))["tau"])

    @property
    def ready_at(self) -> float:
        """The current readiness bar: an action is READY when at least this likely to be satisfied."""
        return max(0.5, float(self.runtime.tracker.tau))

    def verify(self) -> bool:
        """True when every cached value equals an independent full rebuild."""
        return self.runtime.verify(self.task_id)["not_bitwise_equal"] == 0

    # -- lifetime ------------------------------------------------------------
    def close(self) -> None:
        self.runtime.close()
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self) -> str:
        return f"Tracker({self.task_id!r})\n{self.status()}"
