"""The engine: keeps a task's decisions current as evidence changes.

    engine = Engine(schema, scorer, aggregator)
    report = engine.deliver(Event(...))      # add, correct or withdraw a record
    engine.assessments()                     # what each action's status is now
    engine.explain("send_packet")            # which records the answer rests on
    engine.verify()                          # cached state == independent full rebuild?

Dependency graph (every arrow is a parent -> child dependency):

    w:edge ─┐
    rec:R ──┼─> edge:R|P ─┐
    pred:P ─┘             ├─> belief:P ─┐
    w:agg ────────────────┘             ├─> action:A
    schema ─────────────────────────────┘

A predicate declared with `join=True` is read differently: all records linked to it are
concatenated, highest rank first, and read against the claim in one pass (node `jedge:P`). That
can settle a claim no single record settles, and costs one reading of the whole group whenever any
member changes.

`w:edge`, `w:agg` and `schema` are inputs like any other. Replacing model weights or editing the
schema is therefore handled by the same propagation as a change of evidence, and cannot leave a
stale value behind.

The engine proposes; it never executes. Every assessment of an action that needs approval says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ease.aggregate import Aggregator, EdgeView, STATUS4
from ease.graph import IncrementalGraph, PropagationReport
from ease.ledger import Event, Ledger, Outcome, Receipt, Status
from ease.logic import SATISFIED, VIOLATED, evaluate
from ease.schema import TaskSchema
from ease.scorer import EdgeScorer
from ease.util import text_hash

READY, BLOCKED, NEEDS_INFO = "READY", "BLOCKED", "NEEDS_INFO"

# Chooses which predicates a record is compared with. None means "all of them".
Linker = Callable[[str, str, TaskSchema], Optional[Sequence[str]]]


@dataclass
class EngineConfig:
    ready_threshold: float = 0.5  # P(SATISFIED) required before an action is proposed as READY
    block_threshold: float = 0.5  # P(VIOLATED) required before an action is reported BLOCKED
    honour_declared_links: bool = True  # an event may name the predicates it is about (meta["about"])


@dataclass
class Assessment:
    action_id: str
    disposition: str
    status: np.ndarray  # P(SATISFIED), P(VIOLATED), P(UNRESOLVED) of the requirement
    belief: float  # P(requirement truly holds)
    p_success: float
    expected_utility: float
    needs_approval: bool
    exact: bool

    def as_dict(self) -> dict:
        return {"action_id": self.action_id, "disposition": self.disposition,
                "status": {"satisfied": float(self.status[0]), "violated": float(self.status[1]),
                           "unresolved": float(self.status[2])},
                "belief": self.belief, "p_success": self.p_success, "expected_utility": self.expected_utility,
                "needs_approval": self.needs_approval, "exact": self.exact}


@dataclass
class ChangeReport:
    receipt: Optional[Receipt]
    propagation: dict
    edges_scored: int
    edge_cache_hits: int
    tokens: int
    changed_predicates: list[tuple[str, str, str]]
    changed_actions: list[tuple[str, str, str]]
    seconds: float
    links_added: int = 0
    links_removed: int = 0

    def as_dict(self) -> dict:
        return {
            "ledger": None if self.receipt is None else {
                "outcome": self.receipt.outcome.value, "changed": self.receipt.changed,
                "record_id": self.receipt.record_id, "revision": self.receipt.revision,
                "status": self.receipt.status.value},
            "propagation": self.propagation, "edges_scored": self.edges_scored,
            "edge_cache_hits": self.edge_cache_hits, "tokens": self.tokens,
            "changed_predicates": self.changed_predicates, "changed_actions": self.changed_actions,
            "seconds": self.seconds, "links_added": self.links_added, "links_removed": self.links_removed,
        }


def pair_key(claim: str, evidence: str) -> str:
    """Identity of a (claim, evidence) pair, independent of any model version."""
    return text_hash("pair", claim, evidence)[:24]


def disposition_of(status: np.ndarray, cfg: EngineConfig) -> str:
    if status[SATISFIED] >= cfg.ready_threshold and status[SATISFIED] >= status[VIOLATED]:
        return READY
    if status[VIOLATED] >= cfg.block_threshold:
        return BLOCKED
    return NEEDS_INFO


class Engine:
    def __init__(self, schema: TaskSchema, scorer: EdgeScorer, aggregator: Aggregator,
                 ledger: Optional[Ledger] = None, config: Optional[EngineConfig] = None,
                 linker: Optional[Linker] = None, now: Optional[float] = None, early_cutoff: bool = True):
        self.schema = schema
        self.scorer = scorer
        self.aggregator = aggregator
        self.ledger = ledger or Ledger(task_id=schema.task_id)
        self.cfg = config or EngineConfig()
        self.linker = linker
        self.now = now
        self.links: dict[str, set[str]] = {}  # record_id -> predicate ids it has edges to
        self.declared: dict[str, tuple[str, ...]] = {}
        self.g = IncrementalGraph(early_cutoff=early_cutoff)
        self.g.register("edge", self._edge_fn)
        self.g.register("jedge", self._jedge_fn)
        self.g.register("belief", self._belief_fn)
        self.g.register("action", self._action_fn)
        self.g.add_input("w:edge", scorer.version)
        self.g.add_input("w:agg", aggregator.version)
        self.g.add_input("schema", schema.version())
        for p in schema.predicates:
            self._add_predicate(p.id)
        for a in schema.actions:
            self._add_action(a.id)
        for rid in sorted(self.ledger.records()):
            self._sync_record(rid)
        self.g.propagate()

    # ------------------------------------------------------------------ node functions
    def _edge_fn(self, ids, parent_values):
        out: list = [None] * len(ids)
        todo, pairs = [], []
        for i, (w, rec, pred) in enumerate(parent_values):
            if rec is None or pred is None:
                continue
            todo.append(i)
            pairs.append((pred[0], rec["text"]))
        if pairs:
            for i, (lg, msg) in zip(todo, self.scorer.score(pairs)):
                rec = parent_values[i][1]
                out[i] = {"logits": np.asarray(lg, np.float32), "msg": np.asarray(msg, np.float32),
                          "authority": rec["authority"], "valid_from": rec["valid_from"],
                          "record_id": rec["record_id"], "key": rec["key"],
                          "pair": pair_key(parent_values[i][2][0], rec["text"])}
        return out

    def _jedge_fn(self, ids, parent_values):
        out: list = [None] * len(ids)
        todo, pairs, members = [], [], []
        for i, pv in enumerate(parent_values):
            pred = pv[1]
            recs = sorted((r for r in pv[2:] if r is not None),
                          key=lambda r: (-r["authority"], -r["valid_from"], r["record_id"]))
            if not recs or pred is None:
                continue
            todo.append(i)
            pairs.append((pred[0], " ".join(r["text"] for r in recs)))
            members.append(recs)
        if pairs:
            for i, pair, recs, (lg, msg) in zip(todo, pairs, members, self.scorer.score(pairs)):
                out[i] = {"logits": np.asarray(lg, np.float32), "msg": np.asarray(msg, np.float32),
                          "authority": max(r["authority"] for r in recs),
                          "valid_from": max(r["valid_from"] for r in recs),
                          "record_id": "joint:" + ids[i].split(":", 1)[1],
                          "key": text_hash(*[r["key"] for r in recs])[:24],
                          "pair": pair_key(*pair), "members": tuple(r["record_id"] for r in recs)}
        return out

    def _belief_fn(self, ids, parent_values):
        out = []
        for pv in parent_values:
            pred = pv[1]
            edges = [EdgeView(e["record_id"], e["logits"], e["msg"], e["authority"], e["valid_from"], e["pair"])
                     for e in pv[2:] if e is not None]
            out.append(self.aggregator.aggregate(edges, pred[1]).as_value())
        return out

    def _action_fn(self, ids, parent_values):
        out = []
        for nid, pv in zip(ids, parent_values):
            a = self._action(nid.split(":", 1)[1])
            leaves = self._leaves(a.requires)
            beliefs = dict(zip(leaves, pv[1:]))
            leaf_status = {k: np.array([v["status"][0], v["status"][1], v["status"][2] + v["status"][3]])
                           for k, v in beliefs.items()}
            leaf_belief = {k: v["belief"] for k, v in beliefs.items()}
            ev = evaluate(self.schema, a.requires, leaf_status, leaf_belief)
            p_success = ev.belief * a.reliability + (1.0 - ev.belief) * a.leak
            out.append({
                "status": ev.status, "belief": ev.belief, "p_success": float(p_success),
                "expected_utility": float(p_success * a.value_success - (1.0 - p_success) * a.cost_failure),
                "disposition": disposition_of(ev.status, self.cfg), "exact": ev.exact,
            })
        return out

    # ------------------------------------------------------------------ structure
    def _action(self, aid: str):
        for a in self.schema.actions:
            if a.id == aid:
                return a
        raise KeyError(aid)

    def _leaves(self, node_id: str) -> list[str]:
        return list(dict.fromkeys(self.schema.leaves_of(node_id)))

    def _is_join(self, pid: str) -> bool:
        return bool(self.schema.predicate(pid).join)

    def _add_predicate(self, pid: str) -> None:
        p = self.schema.predicate(pid)
        self.g.add_input(f"pred:{pid}", (p.text, float(p.prior)))
        if p.join:
            self.g.add_node(f"jedge:{pid}", "jedge", ["w:edge", f"pred:{pid}"])
        self.g.add_node(f"belief:{pid}", "belief", self._belief_parents(pid))

    def _remove_predicate(self, pid: str, was_join: bool) -> None:
        self.g.remove_node(f"belief:{pid}")
        if was_join:
            self.g.remove_node(f"jedge:{pid}")
        for rid in sorted(self.links):
            if pid in self.links[rid]:
                self.links[rid].discard(pid)
                if not was_join:
                    self.g.remove_node(f"edge:{rid}|{pid}")
        self.g.remove_node(f"pred:{pid}")

    def _add_action(self, aid: str) -> None:
        a = self._action(aid)
        self.g.add_node(f"action:{aid}", "action", ["schema"] + [f"belief:{l}" for l in self._leaves(a.requires)])

    def _belief_parents(self, pid: str) -> list[str]:
        if self._is_join(pid):
            return ["w:agg", f"pred:{pid}", f"jedge:{pid}"]
        edges = sorted(f"edge:{rid}|{pid}" for rid, pids in self.links.items() if pid in pids)
        return ["w:agg", f"pred:{pid}"] + edges

    def _jedge_parents(self, pid: str) -> list[str]:
        recs = sorted(f"rec:{rid}" for rid, pids in self.links.items() if pid in pids)
        return ["w:edge", f"pred:{pid}"] + recs

    def _rec_value(self, rid: str) -> Optional[dict]:
        st = self.ledger.get(rid)
        if st is None or st.status(self.now) is not Status.ACTIVE:
            return None
        p = st.payloads[0]
        if not p.text:
            return None
        return {"record_id": rid, "key": p.key, "text": p.text, "authority": int(p.authority),
                "valid_from": float(p.valid_from) if p.valid_from is not None else float("-inf"),
                "source_id": p.source_id, "revision": st.revision}

    def _wanted_links(self, rid: str, value: Optional[dict]) -> set[str]:
        all_preds = {p.id for p in self.schema.predicates}
        if self.cfg.honour_declared_links and rid in self.declared:
            return {p for p in self.declared[rid] if p in all_preds}
        if value is None:
            return set(self.links.get(rid, ()))  # nothing to read; keep the structure as it is
        if self.linker is None:
            return all_preds
        chosen = self.linker(rid, value["text"], self.schema)
        return all_preds if chosen is None else {p for p in chosen if p in all_preds}

    def _sync_record(self, rid: str) -> tuple[int, int]:
        """Bring the graph in line with the ledger for one record. Returns (links added, removed)."""
        value = self._rec_value(rid)
        node = f"rec:{rid}"
        if node not in self.g.nodes:
            self.g.add_input(node, value)
        else:
            self.g.set_input(node, value)
        want, have = self._wanted_links(rid, value), self.links.get(rid, set())
        added, removed = want - have, have - want
        if not added and not removed:
            return 0, 0
        self.links[rid] = set(want)
        for pid in sorted(added):
            if not self._is_join(pid):
                self.g.add_node(f"edge:{rid}|{pid}", "edge", ["w:edge", node, f"pred:{pid}"])
        for pid in sorted(added | removed):
            if self._is_join(pid):
                self.g.set_parents(f"jedge:{pid}", self._jedge_parents(pid))
            else:
                self.g.set_parents(f"belief:{pid}", self._belief_parents(pid))
        for pid in sorted(removed):
            if not self._is_join(pid):
                self.g.remove_node(f"edge:{rid}|{pid}")
        return len(added), len(removed)

    # ------------------------------------------------------------------ snapshots
    def _snapshot(self) -> tuple[dict, dict]:
        preds = {p.id: STATUS4[int(np.argmax(self.g.value(f"belief:{p.id}")["status"]))] for p in self.schema.predicates}
        acts = {a.id: self.g.value(f"action:{a.id}")["disposition"] for a in self.schema.actions}
        return preds, acts

    def _finish(self, receipt, before, stats_before, t0, links=(0, 0)) -> ChangeReport:
        rep: PropagationReport = self.g.propagate()
        after = self._snapshot()
        d = self.scorer.stats.since(stats_before)
        return ChangeReport(
            receipt=receipt, propagation=rep.as_dict(), edges_scored=d["pairs_computed"],
            edge_cache_hits=d["cache_hits"], tokens=d["tokens"],
            changed_predicates=[(k, before[0].get(k, "UNRESOLVED"), v) for k, v in after[0].items()
                                if before[0].get(k, "UNRESOLVED") != v],
            changed_actions=[(k, before[1].get(k, NEEDS_INFO), v) for k, v in after[1].items()
                             if before[1].get(k, NEEDS_INFO) != v],
            seconds=time.perf_counter() - t0, links_added=links[0], links_removed=links[1])

    # ------------------------------------------------------------------ changes
    def deliver(self, event: Event) -> ChangeReport:
        t0 = time.perf_counter()
        before, stats_before = self._snapshot(), self.scorer.stats.snapshot()
        about = event.meta.get("about") if event.meta else None
        receipt = self.ledger.deliver(event)
        links = (0, 0)
        if receipt.changed or about is not None:
            if about is not None and receipt.outcome is Outcome.APPLIED:
                self.declared[event.record_id] = tuple(about)
            links = self._sync_record(event.record_id)
        return self._finish(receipt, before, stats_before, t0, links)

    def set_now(self, now: float) -> ChangeReport:
        """Advance the clock. Only records whose validity boundary was crossed change."""
        t0 = time.perf_counter()
        before, stats_before = self._snapshot(), self.scorer.stats.snapshot()
        self.now = now
        added = removed = 0
        for rid in sorted(self.ledger.records()):
            a, r = self._sync_record(rid)
            added, removed = added + a, removed + r
        return self._finish(None, before, stats_before, t0, (added, removed))

    def set_scorer(self, scorer: EdgeScorer) -> ChangeReport:
        t0 = time.perf_counter()
        before = self._snapshot()
        self.scorer = scorer
        stats_before = self.scorer.stats.snapshot()
        self.g.set_input("w:edge", scorer.version)
        return self._finish(None, before, stats_before, t0)

    def set_aggregator(self, aggregator: Aggregator) -> ChangeReport:
        t0 = time.perf_counter()
        before, stats_before = self._snapshot(), self.scorer.stats.snapshot()
        self.aggregator = aggregator
        self.g.set_input("w:agg", aggregator.version)
        return self._finish(None, before, stats_before, t0)

    def set_config(self, config: EngineConfig) -> ChangeReport:
        t0 = time.perf_counter()
        before, stats_before = self._snapshot(), self.scorer.stats.snapshot()
        self.cfg = config
        for a in self.schema.actions:
            self.g.invalidate(f"action:{a.id}")
        return self._finish(None, before, stats_before, t0)

    def update_schema(self, schema: TaskSchema) -> ChangeReport:
        """Replace the schema. Predicates whose text, prior and reading mode are unchanged keep their
        readings."""
        t0 = time.perf_counter()
        before, stats_before = self._snapshot(), self.scorer.stats.snapshot()
        old = {p.id: p for p in self.schema.predicates}
        new = {p.id: p for p in schema.predicates}
        for a in self.schema.actions:
            self.g.remove_node(f"action:{a.id}")
        gone = sorted(pid for pid in old if pid not in new or new[pid].join != old[pid].join)
        self.schema = schema
        for pid in gone:
            self._remove_predicate(pid, old[pid].join)
        for pid in sorted(set(new) & set(old) - set(gone)):
            self.g.set_input(f"pred:{pid}", (new[pid].text, float(new[pid].prior)))
        for pid in sorted(pid for pid in new if pid not in old or pid in gone):
            self._add_predicate(pid)
        added = removed = 0
        for rid in sorted(self.ledger.records()):
            a, r = self._sync_record(rid)
            added, removed = added + a, removed + r
        for a in schema.actions:
            self._add_action(a.id)
        self.g.set_input("schema", schema.version())
        return self._finish(None, before, stats_before, t0, (added, removed))

    # ------------------------------------------------------------------ queries
    def belief(self, pid: str) -> dict:
        v = self.g.value(f"belief:{pid}")
        return {"predicate": pid, "status": STATUS4[int(np.argmax(v["status"]))],
                "probabilities": {n: float(x) for n, x in zip(STATUS4, v["status"])},
                "belief": float(v["belief"]), "decisive": list(v["decisive"])}

    def beliefs(self) -> dict[str, dict]:
        return {p.id: self.belief(p.id) for p in self.schema.predicates}

    def assessment(self, aid: str) -> Assessment:
        v = self.g.value(f"action:{aid}")
        a = self._action(aid)
        return Assessment(aid, v["disposition"], v["status"], v["belief"], v["p_success"], v["expected_utility"],
                          a.needs_approval, v["exact"])

    def assessments(self) -> dict[str, Assessment]:
        return {a.id: self.assessment(a.id) for a in self.schema.actions}

    def leaf_state(self) -> tuple[dict, dict]:
        ls, lb = {}, {}
        for p in self.schema.predicates:
            v = self.g.value(f"belief:{p.id}")
            ls[p.id] = np.array([v["status"][0], v["status"][1], v["status"][2] + v["status"][3]])
            lb[p.id] = float(v["belief"])
        return ls, lb

    def explain(self, aid: str) -> dict:
        """Which predicates an action rests on, and which records each of those rests on."""
        a = self._action(aid)
        items = []
        for pid in self._leaves(a.requires):
            b = self.belief(pid)
            recs = []
            for rid in b["decisive"]:
                if rid.startswith("joint:"):
                    e = self.g.value(f"jedge:{pid}")
                    members = e["members"]
                else:
                    e = self.g.value(f"edge:{rid}|{pid}")
                    members = (rid,)
                for m in members:
                    st = self.ledger.get(m)
                    p = st.payloads[0]
                    recs.append({"record_id": m, "revision": st.revision, "source_id": p.source_id,
                                 "authority": p.authority, "valid_from": p.valid_from, "text": p.text, "span": p.span,
                                 "reading": e["logits"].tolist(), "read_jointly": len(members) > 1})
            items.append({**b, "text": self.schema.predicate(pid).text, "records": recs,
                          "read_jointly": self._is_join(pid),
                          "records_compared": sum(1 for r, ps in self.links.items() if pid in ps)})
        ass = self.assessment(aid)
        return {"action": aid, "description": a.description, "requires": a.requires,
                "assessment": ass.as_dict(), "predicates": items,
                "note": "These are the dependencies the model used. They are not a guarantee that no other "
                        "record bears on the question."}

    def verify(self, bypass_cache: bool = True) -> dict:
        """Compare every cached value with an independent full rebuild.

        With `bypass_cache` the scorer's cache is set aside, so edges are recomputed by the model
        rather than looked up. Bit-for-bit agreement then requires a scorer in canonical mode.
        """
        saved = getattr(self.scorer, "cache", None)
        if bypass_cache and saved is not None:
            self.scorer.cache = {}
        try:
            return self.g.verify()
        finally:
            if bypass_cache and saved is not None:
                self.scorer.cache = saved

    def counts(self) -> dict:
        kinds: dict[str, int] = {}
        for n in self.g.nodes.values():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
        return kinds
