"""Run episodes through the engine once with the real reader and record what it produced.

The expensive part of every experiment is the edge model. `build_table` scores each distinct
(claim, record text) pair of a set of episodes exactly once and stores the result. Everything that
follows (training aggregators, comparing aggregators, sweeping thresholds) replays episodes against
that table and needs no GPU. Token counts are stored with the table so that cost can be accounted
without re-running the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ease.aggregate import RuleAggregator
from ease.engine import Engine
from ease.ledger import Event
from ease.schema import TaskSchema
from ease.scorer import ScorerStats
from ease.util import read_json, atomic_write_json, text_hash


def event_of(step: dict, about: Optional[list[str]] = None) -> Event:
    e = step["event"]
    return Event(e["record_id"], e["revision"], e["text"], source_id=e["source_id"], authority=e["authority"],
                 valid_from=e["valid_from"], meta={"about": about} if about is not None else {})


def pairs_of(episodes: Sequence[dict]) -> list[tuple[str, str]]:
    """Every (claim, record text) pair that linking each record to each predicate can require."""
    seen, out = set(), []
    for ep in episodes:
        claims = [p["text"] for p in ep["schema"]["predicates"]]
        texts = list(dict.fromkeys(s["event"]["text"] for s in ep["steps"] if s["event"]["text"]))
        for c in claims:
            for t in texts:
                k = (c, t)
                if k not in seen:
                    seen.add(k)
                    out.append(k)
    return out


@dataclass
class EdgeTable:
    version: str
    msg_dim: int
    keys: dict[str, int]
    logits: np.ndarray  # [P, 3] float32
    msgs: np.ndarray  # [P, d] float32
    tokens: np.ndarray  # [P] int32, tokens of the encoded pair

    def key(self, claim: str, evidence: str) -> str:
        return text_hash(self.version, claim, evidence)

    def index(self, claim: str, evidence: str) -> int:
        return self.keys[self.key(claim, evidence)]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "table.npz", logits=self.logits, msgs=self.msgs, tokens=self.tokens)
        atomic_write_json(d / "table.json", {"version": self.version, "msg_dim": self.msg_dim,
                                             "keys": list(self.keys)})

    @classmethod
    def load(cls, directory: str | Path) -> "EdgeTable":
        d = Path(directory)
        meta = read_json(d / "table.json")
        z = np.load(d / "table.npz")
        return cls(meta["version"], meta["msg_dim"], {k: i for i, k in enumerate(meta["keys"])},
                   z["logits"], z["msgs"], z["tokens"])


class TableBackedScorer:
    """EdgeScorer over an EdgeTable. Counts pairs and tokens as the real model would incur them."""

    def __init__(self, table: EdgeTable):
        self.table = table
        self.version = table.version
        self.msg_dim = table.msg_dim
        self.stats = ScorerStats()
        self.cache: dict = {}  # pairs already paid for in this task

    def score(self, pairs):
        self.stats.calls += 1
        self.stats.pairs_requested += len(pairs)
        out = []
        for c, e in pairs:
            k = self.table.key(c, e)
            i = self.table.keys[k]
            if k in self.cache:
                self.stats.cache_hits += 1
            else:
                self.cache[k] = True
                self.stats.pairs_computed += 1
                self.stats.tokens += int(self.table.tokens[i])
            out.append((self.table.logits[i], self.table.msgs[i]))
        return out


def build_table(episodes: Sequence[dict], scorer, batch: int = 4096, verbose: bool = True) -> EdgeTable:
    """Score every distinct pair once with the real model."""
    from ease.data.streams import encode_pairs

    pairs = pairs_of(episodes)
    if verbose:
        print(f"  scoring {len(pairs)} distinct pairs from {len(episodes)} episodes", flush=True)
    logits = np.zeros((len(pairs), 3), np.float32)
    msgs = np.zeros((len(pairs), scorer.msg_dim), np.float32)
    tokens = np.zeros(len(pairs), np.int32)
    keys = {}
    for s in range(0, len(pairs), batch):
        chunk = pairs[s : s + batch]
        scorer.cache.clear()
        got = scorer.score(chunk)
        lens = [len(x) for x in encode_pairs(scorer.tok, [c for c, _ in chunk], [e for _, e in chunk], scorer.max_len)]
        for j, ((c, e), (lg, mg)) in enumerate(zip(chunk, got)):
            i = s + j
            logits[i], msgs[i], tokens[i] = lg, mg, lens[j]
            keys[scorer.key(c, e)] = i
        if verbose and (s // batch) % 10 == 0:
            print(f"    {min(s + batch, len(pairs))}/{len(pairs)}", flush=True)
    scorer.cache.clear()
    return EdgeTable(scorer.version, scorer.msg_dim, keys, logits, msgs, tokens)


def trace(episodes: Sequence[dict], table: EdgeTable, scored_only: bool = True) -> dict:
    """Replay episodes and record, for every (step, predicate), the edges the aggregator would see
    and the ground-truth status. Output is index arrays into the table, ready for batching."""
    ex_edges: list[np.ndarray] = []
    ex_auth: list[np.ndarray] = []
    ex_vf: list[np.ndarray] = []
    ex_rid: list[list[str]] = []
    label, prior, ep_ix, step_ix, changed = [], [], [], [], []
    for ei, ep in enumerate(episodes):
        schema = TaskSchema.from_dict(ep["schema"])
        eng = Engine(schema, TableBackedScorer(table), RuleAggregator(msg_dim=table.msg_dim))
        claims = {p.id: p.text for p in schema.predicates}
        for step in ep["steps"]:
            eng.deliver(event_of(step))
            if scored_only and step["setup"]:
                continue
            for p in schema.predicates:
                idx, auth, vf, rids = [], [], [], []
                for rid, pids in eng.links.items():
                    if p.id not in pids:
                        continue
                    v = eng.g.value(f"edge:{rid}|{p.id}")
                    if v is None:
                        continue
                    rec = eng.g.value(f"rec:{rid}")
                    idx.append(table.index(claims[p.id], rec["text"]))
                    auth.append(v["authority"])
                    vf.append(v["valid_from"])
                    rids.append(rid)
                ex_rid.append(rids)
                ex_edges.append(np.asarray(idx, np.int64))
                ex_auth.append(np.asarray(auth, np.int64))
                ex_vf.append(np.asarray(vf, np.float64))
                label.append(step["truth"]["pred"][p.id])
                prior.append(p.prior)
                ep_ix.append(ei)
                step_ix.append(step["t"])
                changed.append(p.id in step["truth"]["changed_preds"])
    return {"edges": ex_edges, "authority": ex_auth, "valid_from": ex_vf, "record_ids": ex_rid,
            "label": np.asarray(label, np.int64), "prior": np.asarray(prior, np.float64),
            "episode": np.asarray(ep_ix, np.int64), "step": np.asarray(step_ix, np.int64),
            "changed": np.asarray(changed, bool)}
