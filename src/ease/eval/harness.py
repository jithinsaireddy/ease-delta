"""Run systems on episodes and score them as the pre-registration specifies.

Every system produces the same three tables, so every metric is computed by the same code:

    actions     one row per (scored step, action)
    predicates  one row per (scored step, predicate)
    events      one row per scored step, with what the step cost

Intervals are percentile bootstraps over whole episodes. Comparisons between two systems reuse the
same resampled episodes for both, so they are paired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ease.aggregate import Aggregator
from ease.data.episodes import PRED_STATUS_NAMES
from ease.engine import BLOCKED, NEEDS_INFO, READY, Engine, EngineConfig, disposition_of
from ease.eval.trace import EdgeTable, TableBackedScorer, event_of
from ease.logic import evaluate
from ease.schema import TaskSchema

DISP = {READY: 0, BLOCKED: 1, NEEDS_INFO: 2}
STATUS_IX = {n: i for i, n in enumerate(PRED_STATUS_NAMES)}


@dataclass
class Results:
    name: str
    n_episodes: int
    actions: dict = field(default_factory=dict)
    predicates: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)


class _Collector:
    def __init__(self):
        self.a = {k: [] for k in ("ep", "t", "true", "pred", "true_changed", "pred_changed", "p_success", "success")}
        self.p = {k: [] for k in ("ep", "t", "true", "pred", "p_true", "true_changed")}
        self.e = {k: [] for k in ("ep", "t", "type", "ledger_changed", "tokens", "encodes", "nodes", "n_active",
                                  "n_pred")}

    def finish(self, name: str, n: int) -> Results:
        conv = lambda d: {k: np.asarray(v) for k, v in d.items()}
        return Results(name, n, conv(self.a), conv(self.p), conv(self.e))


def run_engine(name: str, episodes: Sequence[dict], table: EdgeTable, aggregator: Aggregator,
               config: Optional[EngineConfig] = None, linker=None, declared_links: bool = False) -> Results:
    col = _Collector()
    for ei, ep in enumerate(episodes):
        schema = TaskSchema.from_dict(ep["schema"])
        scorer = TableBackedScorer(table)
        eng = Engine(schema, scorer, aggregator, config=config, linker=linker)
        prev = {a.id: NEEDS_INFO for a in schema.actions}
        for step in ep["steps"]:
            about = ([step["target_pred"]] if step["target_pred"] else []) if declared_links else None
            rep = eng.deliver(event_of(step, about))
            cur = {a.id: eng.assessment(a.id) for a in schema.actions}
            if not step["setup"]:
                for aid, ass in cur.items():
                    tr = step["truth"]["actions"][aid]
                    col.a["ep"].append(ei); col.a["t"].append(step["t"])
                    col.a["true"].append(DISP[tr["disposition"]]); col.a["pred"].append(DISP[ass.disposition])
                    col.a["true_changed"].append(aid in step["truth"]["changed_actions"])
                    col.a["pred_changed"].append(ass.disposition != prev[aid])
                    col.a["p_success"].append(ass.p_success); col.a["success"].append(bool(tr["success"]))
                for p in schema.predicates:
                    v = eng.g.value(f"belief:{p.id}")["status"]
                    y = step["truth"]["pred"][p.id]
                    col.p["ep"].append(ei); col.p["t"].append(step["t"])
                    col.p["true"].append(y); col.p["pred"].append(int(np.argmax(v)))
                    col.p["p_true"].append(float(v[y]))
                    col.p["true_changed"].append(p.id in step["truth"]["changed_preds"])
                col.e["ep"].append(ei); col.e["t"].append(step["t"]); col.e["type"].append(step["type"])
                col.e["ledger_changed"].append(bool(rep.receipt.changed))
                col.e["tokens"].append(rep.tokens); col.e["encodes"].append(rep.edges_scored)
                col.e["nodes"].append(rep.propagation["total_recomputed"])
                col.e["n_active"].append(step["truth"]["n_active"]); col.e["n_pred"].append(len(schema.predicates))
            prev = {aid: ass.disposition for aid, ass in cur.items()}
    return col.finish(name, len(episodes))


def full_rebuild_cost(episodes: Sequence[dict], table: EdgeTable) -> Results:
    """Cost of the same model when every changed event triggers a complete re-reading: every active
    record against every predicate. Decisions are identical to incremental execution (H1), so only
    the events table is filled."""
    from ease.ledger import Ledger

    col = _Collector()
    for ei, ep in enumerate(episodes):
        schema = TaskSchema.from_dict(ep["schema"])
        led = Ledger(task_id=ep["episode_id"])
        for step in ep["steps"]:
            r = led.deliver(event_of(step))
            if step["setup"]:
                continue
            tokens = encodes = 0
            if r.changed:
                for p in led.active().values():
                    if not p.text:
                        continue
                    for pr in schema.predicates:
                        tokens += int(table.tokens[table.index(pr.text, p.text)])
                        encodes += 1
            col.e["ep"].append(ei); col.e["t"].append(step["t"]); col.e["type"].append(step["type"])
            col.e["ledger_changed"].append(bool(r.changed)); col.e["tokens"].append(tokens)
            col.e["encodes"].append(encodes); col.e["nodes"].append(encodes)
            col.e["n_active"].append(step["truth"]["n_active"]); col.e["n_pred"].append(len(schema.predicates))
        led.close()
    return col.finish("E-full", len(episodes))


def run_dense(name: str, episodes: Sequence[dict], rows: Sequence[dict], logits: np.ndarray, tokens: np.ndarray,
              config: Optional[EngineConfig] = None) -> Results:
    """rows/logits/tokens come from ease.eval.dense_data.contexts_of and predict_contexts."""
    cfg = config or EngineConfig()
    z = logits.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    at = {}
    for i, r in enumerate(rows):
        at[(r["episode"], r["t"], r["pid"])] = i
    col = _Collector()
    for ei, ep in enumerate(episodes):
        schema = TaskSchema.from_dict(ep["schema"])
        cur = {p.id: np.array([0.0, 0.0, 1.0, 0.0]) for p in schema.predicates}
        prev = {a.id: NEEDS_INFO for a in schema.actions}
        for step in ep["steps"]:
            tokens_here = encodes = 0
            for p in schema.predicates:
                i = at.get((ei, step["t"], p.id))
                if i is not None:
                    cur[p.id] = probs[i]
                    tokens_here += int(tokens[i]); encodes += 1
            ls = {k: np.array([v[0], v[1], v[2] + v[3]]) for k, v in cur.items()}
            lb = {p.id: float(cur[p.id][0] + (cur[p.id][2] + cur[p.id][3]) * p.prior) for p in schema.predicates}
            disp, psucc = {}, {}
            for a in schema.actions:
                ev = evaluate(schema, a.requires, ls, lb)
                disp[a.id] = disposition_of(ev.status, cfg)
                psucc[a.id] = ev.belief * a.reliability + (1 - ev.belief) * a.leak
            if not step["setup"]:
                for a in schema.actions:
                    tr = step["truth"]["actions"][a.id]
                    col.a["ep"].append(ei); col.a["t"].append(step["t"])
                    col.a["true"].append(DISP[tr["disposition"]]); col.a["pred"].append(DISP[disp[a.id]])
                    col.a["true_changed"].append(a.id in step["truth"]["changed_actions"])
                    col.a["pred_changed"].append(disp[a.id] != prev[a.id])
                    col.a["p_success"].append(psucc[a.id]); col.a["success"].append(bool(tr["success"]))
                for p in schema.predicates:
                    y = step["truth"]["pred"][p.id]
                    col.p["ep"].append(ei); col.p["t"].append(step["t"]); col.p["true"].append(y)
                    col.p["pred"].append(int(np.argmax(cur[p.id]))); col.p["p_true"].append(float(cur[p.id][y]))
                    col.p["true_changed"].append(p.id in step["truth"]["changed_preds"])
                col.e["ep"].append(ei); col.e["t"].append(step["t"]); col.e["type"].append(step["type"])
                col.e["ledger_changed"].append(encodes > 0); col.e["tokens"].append(tokens_here)
                col.e["encodes"].append(encodes); col.e["nodes"].append(encodes)
                col.e["n_active"].append(step["truth"]["n_active"]); col.e["n_pred"].append(len(schema.predicates))
            prev = dict(disp)
    return col.finish(name, len(episodes))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _per_episode(ep: np.ndarray, num: np.ndarray, den: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.bincount(ep, weights=num.astype(np.float64), minlength=n),
            np.bincount(ep, weights=den.astype(np.float64), minlength=n))


def metric_parts(r: Results) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each metric as per-episode (numerator, denominator). The metric is sum(num) / sum(den)."""
    n, out = r.n_episodes, {}
    if len(r.actions.get("ep", ())):
        a = r.actions
        wrong = a["pred"] != a["true"]
        ones = np.ones(len(wrong))
        out["disposition_accuracy"] = _per_episode(a["ep"], ~wrong, ones, n)
        out["stale_decision_rate"] = _per_episode(a["ep"], wrong & a["true_changed"], a["true_changed"], n)
        out["spurious_change_rate"] = _per_episode(a["ep"], a["pred_changed"] & ~a["true_changed"], ~a["true_changed"], n)
        not_ready = a["true"] != DISP[READY]
        out["false_ready_rate"] = _per_episode(a["ep"], (a["pred"] == DISP[READY]) & not_ready, not_ready, n)
        ready = a["true"] == DISP[READY]
        out["missed_ready_rate"] = _per_episode(a["ep"], (a["pred"] != DISP[READY]) & ready, ready, n)
        out["success_brier"] = _per_episode(a["ep"], (a["p_success"] - a["success"].astype(float)) ** 2, ones, n)
    if len(r.predicates.get("ep", ())):
        p = r.predicates
        ones = np.ones(len(p["ep"]))
        out["status_accuracy"] = _per_episode(p["ep"], p["pred"] == p["true"], ones, n)
        out["status_nll"] = _per_episode(p["ep"], -np.log(np.clip(p["p_true"], 1e-12, 1.0)), ones, n)
        ch = p["true_changed"]
        out["status_accuracy_on_change"] = _per_episode(p["ep"], (p["pred"] == p["true"]) & ch, ch, n)
    if len(r.events.get("ep", ())):
        e = r.events
        ch = e["ledger_changed"]
        out["tokens_per_changed_event"] = _per_episode(e["ep"], e["tokens"] * ch, ch, n)
        out["encodes_per_changed_event"] = _per_episode(e["ep"], e["encodes"] * ch, ch, n)
        out["tokens_per_unchanged_event"] = _per_episode(e["ep"], e["tokens"] * ~ch, ~ch, n)
    return out


def _boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def summarise(r: Results, n_boot: int = 2000, seed: int = 0) -> dict:
    idx = _boot_indices(r.n_episodes, n_boot, seed)
    out = {}
    for k, (num, den) in metric_parts(r).items():
        tot = den.sum()
        if tot == 0:
            continue
        b = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
        out[k] = {"value": float(num.sum() / tot), "lo": float(np.quantile(b, 0.025)),
                  "hi": float(np.quantile(b, 0.975)), "n": float(tot)}
    if len(r.actions.get("ep", ())):
        from ease.eval.metrics import binary_ece
        out["success_ece"] = {"value": binary_ece(r.actions["p_success"].astype(float), r.actions["success"].astype(float)),
                              "n": float(len(r.actions["ep"]))}
    return out


def compare(a: Results, b: Results, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired (a - b) for every metric both systems have, and the ratio a / b for cost metrics."""
    assert a.n_episodes == b.n_episodes
    idx = _boot_indices(a.n_episodes, n_boot, seed)
    pa, pb = metric_parts(a), metric_parts(b)
    out = {}
    for k in pa:
        if k not in pb or pa[k][1].sum() == 0 or pb[k][1].sum() == 0:
            continue
        (na, da), (nb, db) = pa[k], pb[k]
        va = na[idx].sum(1) / np.maximum(da[idx].sum(1), 1e-12)
        vb = nb[idx].sum(1) / np.maximum(db[idx].sum(1), 1e-12)
        d = va - vb
        row = {"a": float(na.sum() / da.sum()), "b": float(nb.sum() / db.sum()),
               "difference": float(na.sum() / da.sum() - nb.sum() / db.sum()),
               "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))}
        if k.startswith(("tokens", "encodes")) and nb.sum() > 0:
            ratio = va / np.maximum(vb, 1e-12)
            row["ratio"] = float((na.sum() / da.sum()) / (nb.sum() / db.sum()))
            row["ratio_lo"], row["ratio_hi"] = float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))
        out[k] = row
    return out


def by_event_type(r: Results) -> dict:
    """Disposition accuracy split by the kind of event that preceded it."""
    if not len(r.events.get("ep", ())):
        return {}
    kind = {(int(e), int(t)): ty for e, t, ty in zip(r.events["ep"], r.events["t"], r.events["type"])}
    acc: dict[str, list[int]] = {}
    for e, t, y, p in zip(r.actions["ep"], r.actions["t"], r.actions["true"], r.actions["pred"]):
        acc.setdefault(kind[(int(e), int(t))], []).append(int(y == p))
    return {k: {"accuracy": float(np.mean(v)), "n": len(v)} for k, v in sorted(acc.items())}


def worst_episodes(r: Results, k: int = 5) -> list[dict]:
    num, den = metric_parts(r)["disposition_accuracy"]
    acc = np.where(den > 0, num / np.maximum(den, 1), 1.0)
    order = np.argsort(acc)[:k]
    return [{"episode_index": int(i), "disposition_accuracy": float(acc[i]), "rows": int(den[i])} for i in order]
