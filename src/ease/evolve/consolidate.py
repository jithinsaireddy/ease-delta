"""Gated consolidation: learn from accumulated feedback, adopt the result only if it is safe.

Inputs
    verified readings   (pair, label, the edge model's logits and message vector)
    a champion          the refiner and memory settings currently in use
    a regression suite  fixed readings with known labels that must not get worse

Procedure
    1. Split the verified readings into a fit part and a held-out part. The split is a function
       of the pair's identity, so an item never moves between parts.
    2. Build candidates from the fit part only:
         - memory settings (tau, lam_max) chosen by leave-one-out on the fit part;
         - a refiner fine-tuned on the fit part, penalised for moving away from the champion.
    3. Score champion and candidates on the held-out part and on the regression suite.
    4. Adopt a candidate only if
         a. on every regression set its accuracy is within `max_accuracy_drop` of the champion's
            and its NLL within `max_nll_rise`, and
         b. its held-out NLL is lower than the champion's, with the lower end of a paired
            bootstrap interval of the improvement above `min_gain`.
       Otherwise the champion stays.

Regression sets come in two kinds. *Readings* are single (claim, record) pairs. *Set suites* are
whole predicates from validation episodes: every reading a predicate had at one step, with the
predicate's correct status. The refiner reads an edge in the context of the other edges of its
predicate, so a candidate can be fine on single readings and wrong on sets. That happened in the
experiments: one of three refiners trained by the same procedure lost 6.8 points of disposition
accuracy on tasks larger than those it was trained on (docs/RESULTS.md). Readings alone would not
have caught it; set suites can.

For the same reason fine-tuning does not, by default, update the part of the refiner that reads
the rest of the set (`train_context=False`).

Why (a) checks NLL as well as accuracy: feedback with many wrong labels teaches a model to be
unsure. That can lower held-out NLL (the held-out labels are noisy too) and leave accuracy
untouched, while making probabilities on clean data worse. The regression NLL catches it.

Every adopted version is written to its own directory with its parent recorded, so any adoption
can be undone.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ease.aggregate import EdgeRefiner, RefinerConfig, save_module, load_module, state_digest
from ease.evolve.memory import MemoryConfig, knn_blend, unit
from ease.util import atomic_write_json, read_json


@dataclass
class Readings:
    """Edge readings with known labels. Used for feedback and for regression sets alike."""

    pairs: list[str]
    labels: np.ndarray  # [n]
    logits: np.ndarray  # [n,3] float32
    msgs: np.ndarray  # [n,d] float32

    def __len__(self) -> int:
        return len(self.labels)

    def take(self, ix) -> "Readings":
        ix = np.asarray(ix)
        return Readings([self.pairs[i] for i in ix], self.labels[ix], self.logits[ix], self.msgs[ix])

    @staticmethod
    def concat(parts: Sequence["Readings"]) -> "Readings":
        parts = [p for p in parts if len(p)]
        if not parts:
            return Readings([], np.zeros(0, np.int64), np.zeros((0, 3), np.float32), np.zeros((0, 1), np.float32))
        return Readings(sum((p.pairs for p in parts), []), np.concatenate([p.labels for p in parts]),
                        np.concatenate([p.logits for p in parts]), np.concatenate([p.msgs for p in parts]))


def save_readings(path: str | Path, sets: dict[str, "Readings"]) -> None:
    """Several named sets of readings in one file (used to ship a regression suite with a model)."""
    arrays = {}
    for name, r in sets.items():
        arrays[f"{name}/labels"] = r.labels
        arrays[f"{name}/logits"] = r.logits
        arrays[f"{name}/msgs"] = r.msgs
        arrays[f"{name}/pairs"] = np.asarray(r.pairs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_readings(path: str | Path) -> dict[str, "Readings"]:
    z = np.load(path, allow_pickle=False)
    names = sorted({k.split("/")[0] for k in z.files})
    return {n: Readings([str(x) for x in z[f"{n}/pairs"]], z[f"{n}/labels"].astype(np.int64),
                        z[f"{n}/logits"].astype(np.float32), z[f"{n}/msgs"].astype(np.float32)) for n in names}


@dataclass
class SetSuite:
    """Predicates at steps of validation episodes: all their readings, and their correct status.

    Readings are stored flat; example i owns rows offsets[i]:offsets[i+1], already sorted by rank
    (highest first), with `rank` the dense rank inside the example."""

    logits: np.ndarray  # [M,3] float32
    msgs: np.ndarray  # [M,d] float32
    rank: np.ndarray  # [M] int64
    offsets: np.ndarray  # [n+1] int64
    labels: np.ndarray  # [n] int64, 0..3 (SATISFIED, VIOLATED, UNRESOLVED, CONFLICT)

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def from_trace(cls, tr: dict, table, limit: Optional[int] = None, seed: int = 0) -> "SetSuite":
        """With `limit`, a random sample of that many examples drawn from every episode."""
        from ease.train.stage_b import dense_rank_rows

        total = len(tr["label"])
        if limit is None or limit >= total:
            chosen = np.arange(total)
        else:
            chosen = np.sort(np.random.default_rng(seed).choice(total, size=limit, replace=False))
        lg, mg, rk, off = [], [], [], [0]
        for i in chosen:
            e, a, v, rids = tr["edges"][i], tr["authority"][i], tr["valid_from"][i], tr["record_ids"][i]
            if len(e):
                r = dense_rank_rows(a, v)
                order = sorted(range(len(e)), key=lambda j: (int(r[j]), rids[j]))
                lg.append(table.logits[e[order]]); mg.append(table.msgs[e[order]]); rk.append(r[order])
            off.append(off[-1] + len(e))
        d = table.msgs.shape[1]
        return cls(np.concatenate(lg) if lg else np.zeros((0, 3), np.float32),
                   np.concatenate(mg) if mg else np.zeros((0, d), np.float32),
                   np.concatenate(rk) if rk else np.zeros(0, np.int64),
                   np.asarray(off, np.int64), np.asarray(tr["label"], np.int64)[chosen])


@torch.no_grad()
def set_status_probs(bundle: "Bundle", suite: SetSuite, mem: Optional["Readings"], block: int = 1024) -> np.ndarray:
    """P(SATISFIED, VIOLATED, UNRESOLVED, CONFLICT) for every example of a set suite, computed the
    way the engine computes it: refiner on the padded set, memory adjustment per reading, then the
    exact precedence policy."""
    from ease.aggregate import precedence

    ref = bundle.refiner.double().eval()
    n = len(suite)
    out = np.zeros((n, 4))
    sizes = np.diff(suite.offsets)
    order = np.argsort(sizes, kind="stable")
    use_mem = mem is not None and len(mem) and bundle.memory.lam_max > 0
    mem_unit = unit(mem.msgs) if use_mem else None
    for s in range(0, n, block):
        ix = order[s : s + block]
        N = int(max(1, sizes[ix].max()))
        B = len(ix)
        lg = torch.zeros(B, N, 3, dtype=torch.float64)
        mg = torch.zeros(B, N, suite.msgs.shape[1], dtype=torch.float64)
        rk = torch.zeros(B, N, dtype=torch.long)
        mask = torch.zeros(B, N, dtype=torch.bool)
        for b, i in enumerate(ix):
            a, z = suite.offsets[i], suite.offsets[i + 1]
            k = z - a
            if k:
                lg[b, :k] = torch.from_numpy(suite.logits[a:z].astype(np.float64))
                mg[b, :k] = torch.from_numpy(suite.msgs[a:z].astype(np.float64))
                rk[b, :k] = torch.from_numpy(suite.rank[a:z])
                mask[b, :k] = True
        if N == 0 or not mask.any():
            probs = torch.zeros(B, N, 3, dtype=torch.float64)
        else:
            probs = ref(lg, mg, mask)
        if use_mem and mask.any():
            flat = probs[mask].numpy()
            adj, _ = knn_blend(flat, unit(mg[mask].numpy().astype(np.float32)), mem_unit, mem.labels, bundle.memory)
            probs = probs.clone()
            probs[mask] = torch.from_numpy(adj)
        out[ix] = precedence(probs, rk, mask).numpy()
    return out


def score_sets(p: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0, "accuracy": float("nan"), "nll": float("nan")}
    return {"n": int(len(y)), "accuracy": float((p.argmax(1) == y).mean()),
            "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())}


def save_set_suites(path: str | Path, suites: dict[str, SetSuite]) -> None:
    arrays = {}
    for name, su in suites.items():
        for field_name in ("logits", "msgs", "rank", "offsets", "labels"):
            arrays[f"{name}/{field_name}"] = getattr(su, field_name)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_set_suites(path: str | Path) -> dict[str, SetSuite]:
    z = np.load(path, allow_pickle=False)
    names = sorted({k.split("/")[0] for k in z.files})
    return {n: SetSuite(z[f"{n}/logits"].astype(np.float32), z[f"{n}/msgs"].astype(np.float32),
                        z[f"{n}/rank"].astype(np.int64), z[f"{n}/offsets"].astype(np.int64),
                        z[f"{n}/labels"].astype(np.int64)) for n in names}


def in_holdout(pair: str, frac: float) -> bool:
    h = int(hashlib.sha256(("holdout:" + pair).encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < frac


@torch.no_grad()
def refiner_probs(refiner: EdgeRefiner, r: Readings, block: int = 8192) -> np.ndarray:
    """Per-edge probabilities from the refiner, each edge read on its own (a set of size one)."""
    refiner = refiner.double().eval()
    out = np.zeros((len(r), 3), np.float64)
    for s in range(0, len(r), block):
        lg = torch.as_tensor(r.logits[s : s + block], dtype=torch.float64).unsqueeze(1)
        mg = torch.as_tensor(r.msgs[s : s + block], dtype=torch.float64).unsqueeze(1)
        mask = torch.ones(lg.shape[0], 1, dtype=torch.bool)
        out[s : s + block] = refiner(lg, mg, mask)[:, 0].numpy()
    return out


def nll_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))


def score(p: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0, "accuracy": float("nan"), "nll": float("nan")}
    return {"n": int(len(y)), "accuracy": float((p.argmax(1) == y).mean()), "nll": float(nll_rows(p, y).mean())}


@dataclass
class Bundle:
    """Everything that defines how edges are read after the encoder."""

    refiner: EdgeRefiner
    memory: MemoryConfig
    name: str = "champion"

    def read(self, r: Readings, mem: Optional[Readings], exclude_self: bool = False) -> np.ndarray:
        p = refiner_probs(self.refiner, r)
        if mem is not None and len(mem) and self.memory.lam_max > 0:
            p, _ = knn_blend(p, unit(r.msgs), unit(mem.msgs), mem.labels, self.memory, exclude_self=exclude_self)
        return p


@dataclass
class GateConfig:
    holdout_frac: float = 0.3
    min_feedback: int = 200
    max_accuracy_drop: float = 0.005
    max_negative_flip_rate: Optional[float] = None  # share of a suite the champion got right that a candidate may get wrong; off if None
    max_nll_rise: float = 0.01
    min_gain: float = 0.0
    n_boot: int = 1000
    seed: int = 0
    taus: tuple = (0.6, 0.7, 0.8, 0.9, 0.95)
    lams: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
    finetune_steps: int = 300
    finetune_lr: float = 5e-4
    finetune_batch: int = 256
    anchor: float = 1e-3  # strength of the pull towards the champion's weights
    replay_per_batch: int = 256
    train_context: bool = False  # let fine-tuning change how the refiner reads the rest of a set


def tune_memory(refiner: EdgeRefiner, fit: Readings, base: MemoryConfig, cfg: GateConfig) -> tuple[MemoryConfig, dict]:
    """Leave-one-out on the fit part: would the other verified items have helped read this one?"""
    p0 = refiner_probs(refiner, fit)
    u = unit(fit.msgs)
    best, best_cfg, grid = float(nll_rows(p0, fit.labels).mean()), copy.copy(base), []
    best_cfg.lam_max = 0.0
    for tau in cfg.taus:
        for lam in cfg.lams:
            if lam == 0.0:
                continue
            mc = MemoryConfig(k=base.k, tau=tau, lam_max=lam, exact_confidence=base.exact_confidence,
                              min_votes=base.min_votes)
            p, touched = knn_blend(p0, u, u, fit.labels, mc, exclude_self=True)
            v = float(nll_rows(p, fit.labels).mean())
            grid.append({"tau": tau, "lam_max": lam, "loo_nll": v, "touched": float(touched.mean())})
            if v < best - 1e-9:
                best, best_cfg = v, mc
    return best_cfg, {"no_memory_nll": float(nll_rows(p0, fit.labels).mean()), "best_loo_nll": best, "grid": grid}


def finetune(champion: EdgeRefiner, fit: Readings, replay: Optional[Readings], cfg: GateConfig) -> EdgeRefiner:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = copy.deepcopy(champion).float().train()
    if not cfg.train_context and model.ctx is not None:
        for p in model.ctx.parameters():
            p.requires_grad_(False)
        # `mid` reads [per-edge features, context features]; keep the context half as it was
        ctx_cols = model.cfg.hidden
        mid_mask = torch.ones_like(model.mid.weight)
        mid_mask[:, ctx_cols:] = 0.0
        model.mid.weight.register_hook(lambda g: g * mid_mask)
    anchor = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.finetune_lr, weight_decay=0.0)

    def batch_of(r: Readings, n: int):
        ix = rng.integers(0, len(r), size=min(n, len(r)))
        lg = torch.as_tensor(r.logits[ix], dtype=torch.float32).unsqueeze(1)
        mg = torch.as_tensor(r.msgs[ix], dtype=torch.float32).unsqueeze(1)
        return lg, mg, torch.ones(len(ix), 1, dtype=torch.bool), torch.as_tensor(r.labels[ix])

    for _ in range(cfg.finetune_steps):
        lg, mg, mask, y = batch_of(fit, cfg.finetune_batch)
        loss = F.nll_loss(torch.log(model(lg, mg, mask)[:, 0].clamp_min(1e-12)), y)
        if replay is not None and len(replay):
            lg, mg, mask, y = batch_of(replay, cfg.replay_per_batch)
            loss = loss + F.nll_loss(torch.log(model(lg, mg, mask)[:, 0].clamp_min(1e-12)), y)
        loss = loss + cfg.anchor * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.double().eval()


def paired_gain(nll_champion: np.ndarray, nll_candidate: np.ndarray, n_boot: int, seed: int) -> dict:
    d = nll_champion - nll_candidate  # positive means the candidate is better
    if len(d) == 0:
        return {"gain": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return {"gain": float(d.mean()), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)),
            "n": int(len(d))}


@dataclass
class GateReport:
    adopted: bool
    chosen: str
    reason: str
    feedback: int
    fit: int
    holdout: int
    candidates: dict = field(default_factory=dict)
    memory_search: dict = field(default_factory=dict)
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def consolidate(champion: Bundle, feedback: Readings, regression: dict[str, Readings],
                replay: Optional[Readings] = None, cfg: Optional[GateConfig] = None,
                set_suites: Optional[dict[str, SetSuite]] = None) -> tuple[Bundle, GateReport]:
    cfg = cfg or GateConfig()
    t0 = time.time()
    if len(feedback) < cfg.min_feedback:
        return champion, GateReport(False, "champion", f"only {len(feedback)} verified readings; "
                                    f"{cfg.min_feedback} are required before anything is changed",
                                    len(feedback), 0, 0, seconds=time.time() - t0)
    hold = np.asarray([in_holdout(p, cfg.holdout_frac) for p in feedback.pairs])
    fit, held = feedback.take(np.where(~hold)[0]), feedback.take(np.where(hold)[0])
    if len(held) < 30 or len(fit) < 30:
        return champion, GateReport(False, "champion", "too few readings on one side of the split",
                                    len(feedback), len(fit), len(held), seconds=time.time() - t0)

    mem_cfg, search = tune_memory(champion.refiner, fit, champion.memory, cfg)
    tuned = finetune(champion.refiner, fit, replay, cfg)
    mem_cfg_tuned, search_tuned = tune_memory(tuned, fit, champion.memory, cfg)
    candidates = {
        "champion": champion,
        "memory": Bundle(champion.refiner, mem_cfg, "memory"),
        "refiner": Bundle(tuned, champion.memory, "refiner"),
        "refiner+memory": Bundle(tuned, mem_cfg_tuned, "refiner+memory"),
    }

    def evaluate(b: Bundle) -> dict:
        ph = b.read(held, fit)
        res = {"holdout": score(ph, held.labels), "_nll_rows": nll_rows(ph, held.labels), "regression": {}, "_right": {}}
        for name, r in regression.items():
            pr = b.read(r, fit)
            res["regression"][name] = score(pr, r.labels)
            res["_right"][name] = pr.argmax(1) == r.labels
        for name, su in (set_suites or {}).items():
            ps = set_status_probs(b, su, fit)
            res["regression"][f"sets:{name}"] = score_sets(ps, su.labels)
            res["_right"][f"sets:{name}"] = ps.argmax(1) == su.labels
        return res

    evals = {k: evaluate(b) for k, b in candidates.items()}
    base = evals["champion"]
    report_c, passing = {}, []
    for k, e in evals.items():
        checks = {}
        ok = True
        for name, s in e["regression"].items():
            drop = base["regression"][name]["accuracy"] - s["accuracy"]
            rise = s["nll"] - base["regression"][name]["nll"]
            # items the champion got right and the candidate gets wrong (Xie et al., 2021): a change can
            # keep its accuracy while breaking many readings people relied on
            flips = int((base["_right"][name] & ~e["_right"][name]).sum())
            flip_rate = flips / max(1, len(e["_right"][name]))
            good = drop <= cfg.max_accuracy_drop and rise <= cfg.max_nll_rise
            if cfg.max_negative_flip_rate is not None:
                good = good and flip_rate <= cfg.max_negative_flip_rate
            checks[name] = {"accuracy": s["accuracy"], "nll": s["nll"], "accuracy_drop": drop, "nll_rise": rise,
                            "negative_flips": flips, "negative_flip_rate": flip_rate, "pass": bool(good)}
            ok &= good
        gain = paired_gain(base["_nll_rows"], e["_nll_rows"], cfg.n_boot, cfg.seed)
        improves = gain["lo"] > cfg.min_gain
        report_c[k] = {"holdout": e["holdout"], "gain_over_champion": gain, "regression": checks,
                       "passes_regression": bool(ok), "improves_holdout": bool(improves),
                       "memory": asdict(candidates[k].memory)}
        if k != "champion" and ok and improves:
            passing.append((e["holdout"]["nll"], k))
    if passing:
        passing.sort()
        chosen = passing[0][1]
        rep = GateReport(True, chosen, "passed the regression gate and improved held-out NLL", len(feedback),
                         len(fit), len(held), report_c, {"champion_refiner": search, "tuned_refiner": search_tuned},
                         time.time() - t0)
        return candidates[chosen], rep
    why = []
    for k, c in report_c.items():
        if k == "champion":
            continue
        if not c["passes_regression"]:
            bad = [n for n, v in c["regression"].items() if not v["pass"]]
            why.append(f"{k}: fails regression on {bad}")
        elif not c["improves_holdout"]:
            why.append(f"{k}: no reliable held-out gain (lower bound {c['gain_over_champion']['lo']:.4f})")
    rep = GateReport(False, "champion", "; ".join(why), len(feedback), len(fit), len(held), report_c,
                     {"champion_refiner": search, "tuned_refiner": search_tuned}, time.time() - t0)
    return champion, rep


# ---------------------------------------------------------------------------
# Versioned storage
# ---------------------------------------------------------------------------


class VersionStore:
    """Adopted bundles on disk. `current` names the one in use; each records its parent."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pointer = self.root / "current.json"

    def current(self) -> Optional[str]:
        return read_json(self.pointer)["version"] if self.pointer.exists() else None

    def save(self, bundle: Bundle, report: Optional[GateReport] = None, note: str = "") -> str:
        parent = self.current()
        version = "v-" + state_digest(bundle.refiner) + "-" + hashlib.sha256(
            json.dumps(asdict(bundle.memory), sort_keys=True).encode()).hexdigest()[:8]
        d = self.root / version
        save_module(bundle.refiner, d, {"kind": "refined", "cfg": asdict(bundle.refiner.cfg)})
        atomic_write_json(d / "bundle.json", {"version": version, "parent": parent, "memory": asdict(bundle.memory),
                                              "name": bundle.name, "note": note, "created": time.time(),
                                              "report": report.as_dict() if report else None})
        atomic_write_json(self.pointer, {"version": version})
        return version

    def load(self, version: Optional[str] = None) -> Bundle:
        version = version or self.current()
        if version is None:
            raise FileNotFoundError("no version has been saved")
        d = self.root / version
        meta = read_json(d / "bundle.json")
        agg = read_json(d / "aggregator.json")
        ref = EdgeRefiner(RefinerConfig(**agg["cfg"]))
        load_module(ref, d)
        return Bundle(ref.eval(), MemoryConfig(**meta["memory"]), meta.get("name", version))

    def rollback(self) -> Optional[str]:
        """Return to the parent of the current version. The abandoned version stays on disk."""
        cur = self.current()
        if cur is None:
            return None
        parent = read_json(self.root / cur / "bundle.json").get("parent")
        if parent is None:
            return None
        atomic_write_json(self.pointer, {"version": parent})
        return parent

    def lineage(self) -> list[dict]:
        out, v = [], self.current()
        while v is not None:
            m = read_json(self.root / v / "bundle.json")
            out.append({"version": v, "parent": m.get("parent"), "name": m.get("name"), "created": m.get("created"),
                        "note": m.get("note")})
            v = m.get("parent")
        return out
