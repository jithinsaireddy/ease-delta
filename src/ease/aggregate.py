"""From edge readings to a belief about one predicate.

Several records may bear on one predicate. The declared policy says which prevails:

    Only decisive records count (those that support or refute; a record that settles nothing is
    ignored). Among decisive records the highest rank prevails, rank being (authority, valid_from).
    If decisive records of the top rank disagree, the predicate is in CONFLICT.

The reader's uncertainty is a distribution (p_S, p_R, p_N) per edge. `precedence` computes the
distribution over outcomes that the policy induces from those, exactly, under

    A2  Given the texts, the reader's errors on different edges are independent.

`precedence` is composed of sums and products, so gradients pass through it. That is what lets a
small network be trained to re-read edges (`EdgeRefiner`) while the policy itself is never
approximated. Three aggregators share this file so that they can be compared on equal terms:

    RuleAggregator     calibrated edge probabilities + the exact policy          (no training)
    RefinedAggregator  learned per-edge refinement + the exact policy            (EASE-Delta)
    NeuralAggregator   a recurrent network reads the ranked edges and outputs
                       the status directly; the policy is learned, not executed  (ablation)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ease.labels import NEI, REFUTES, SUPPORTS
from ease.util import text_hash

S, V, U, C = 0, 1, 2, 3  # SATISFIED, VIOLATED, UNRESOLVED, CONFLICT
STATUS4 = ("SATISFIED", "VIOLATED", "UNRESOLVED", "CONFLICT")
EPS = 1e-12


@dataclass(frozen=True)
class EdgeView:
    record_id: str
    logits: np.ndarray
    msg: np.ndarray
    authority: int
    valid_from: float
    pair: str = ""  # identity of the (claim, evidence) pair; lets verified corrections find it


@dataclass
class PredBelief:
    status: np.ndarray  # P(S), P(V), P(U), P(C)
    belief: float  # P(predicate truly holds)
    decisive: tuple[str, ...]  # records most responsible, best first; for explanations

    def as_value(self) -> dict:
        return {"status": self.status, "belief": float(self.belief), "decisive": tuple(self.decisive)}


def dense_rank(edges: Sequence[EdgeView]) -> list[int]:
    """0 for the highest (authority, valid_from), equal keys share a rank."""
    keys = sorted({(e.authority, e.valid_from) for e in edges}, reverse=True)
    pos = {k: i for i, k in enumerate(keys)}
    return [pos[(e.authority, e.valid_from)] for e in edges]


def precedence(probs: torch.Tensor, rank: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Exact outcome distribution of the precedence policy.

    probs [B, N, 3]  per-edge (p_S, p_R, p_N)
    rank  [B, N]     dense rank, 0 = highest; padded positions may hold anything
    mask  [B, N]     1 for real edges
    returns [B, 4]   P(SATISFIED), P(VIOLATED), P(UNRESOLVED), P(CONFLICT)

    For a rank group g:  none_g = prod p_N
                         sup_g  = prod (p_S + p_N) - none_g     (some support, no refutation)
                         ref_g  = prod (p_R + p_N) - none_g
                         con_g  = 1 - none_g - sup_g - ref_g
    The outcome is decided by the first group, in rank order, that is not "none".
    """
    B, N, _ = probs.shape
    if N == 0:
        out = probs.new_zeros(B, 4)
        out[:, U] = 1.0
        return out
    m = mask.to(probs.dtype)
    G = int(rank.max().item()) + 1 if N else 1
    idx = rank.clamp(min=0, max=G - 1)
    log_n = torch.log(probs[..., NEI].clamp_min(EPS)) * m
    log_sn = torch.log((probs[..., SUPPORTS] + probs[..., NEI]).clamp_min(EPS)) * m
    log_rn = torch.log((probs[..., REFUTES] + probs[..., NEI]).clamp_min(EPS)) * m
    zeros = probs.new_zeros(B, G)
    none = torch.exp(zeros.scatter_add(1, idx, log_n))
    sup = (torch.exp(zeros.scatter_add(1, idx, log_sn)) - none).clamp_min(0.0)
    ref = (torch.exp(zeros.scatter_add(1, idx, log_rn)) - none).clamp_min(0.0)
    con = (1.0 - none - sup - ref).clamp_min(0.0)
    log_none = torch.log(none.clamp_min(EPS))
    before = torch.exp(torch.cumsum(log_none, dim=1) - log_none)  # prod of none over higher-ranked groups
    p_s = (before * sup).sum(1)
    p_v = (before * ref).sum(1)
    p_c = (before * con).sum(1)
    p_u = torch.exp(log_none.sum(1))
    out = torch.stack([p_s, p_v, p_u, p_c], dim=1)
    return out / out.sum(1, keepdim=True).clamp_min(EPS)


def belief_from_status(status: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Settled predicates are known; unsettled ones fall back on the declared prior."""
    return status[:, S] + (status[:, U] + status[:, C]) * prior


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    """Temperature and per-class bias applied to edge logits: p = softmax(logits / T + b)."""

    temperature: float = 1.0
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def apply_np(self, logits: np.ndarray) -> np.ndarray:
        z = logits.astype(np.float64) / self.temperature + np.asarray(self.bias, dtype=np.float64)
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        b = torch.tensor(self.bias, dtype=logits.dtype, device=logits.device)
        return F.softmax(logits / self.temperature + b, dim=-1)

    def version(self) -> str:
        return text_hash(repr(float(self.temperature)), repr(tuple(float(x) for x in self.bias)))[:12]


# ---------------------------------------------------------------------------
# Batching helper shared by training and inference
# ---------------------------------------------------------------------------


def pack(edge_sets: Sequence[Sequence[EdgeView]], msg_dim: int, dtype=torch.float64):
    """Pads a list of edge sets, each sorted by rank (highest first)."""
    B = len(edge_sets)
    N = max((len(e) for e in edge_sets), default=0)
    logits = torch.zeros(B, N, 3, dtype=dtype)
    msg = torch.zeros(B, N, msg_dim, dtype=dtype)
    rank = torch.zeros(B, N, dtype=torch.long)
    auth = torch.zeros(B, N, dtype=dtype)
    mask = torch.zeros(B, N, dtype=torch.bool)
    order = []
    for b, edges in enumerate(edge_sets):
        r = dense_rank(edges)
        ix = sorted(range(len(edges)), key=lambda i: (r[i], edges[i].record_id))
        order.append(ix)
        for j, i in enumerate(ix):
            e = edges[i]
            logits[b, j] = torch.as_tensor(np.asarray(e.logits), dtype=dtype)
            msg[b, j] = torch.as_tensor(np.asarray(e.msg), dtype=dtype)
            rank[b, j] = r[i]
            auth[b, j] = float(e.authority)
            mask[b, j] = True
    return logits, msg, rank, auth, mask, order


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


class Aggregator:
    version: str = "abstract"
    msg_dim: int = 0

    def status_batch(self, edge_sets: Sequence[Sequence[EdgeView]]) -> tuple[torch.Tensor, torch.Tensor, list]:
        """Returns (status [B,4], refined edge probs [B,N,3], order) for padded, rank-sorted sets."""
        raise NotImplementedError

    @torch.no_grad()
    def aggregate(self, edges: Sequence[EdgeView], prior: float) -> PredBelief:
        """One predicate at a time. Evaluating sets singly keeps the result independent of what else
        is being evaluated, so cached and recomputed beliefs agree bit for bit."""
        status, probs, order = self.status_batch([list(edges)])
        st = status[0].to(torch.float64).numpy().copy()
        belief = float(st[S] + (st[U] + st[C]) * prior)
        decisive: tuple[str, ...] = ()
        if len(edges):
            p = probs[0].to(torch.float64).numpy()
            weight = 1.0 - p[:, NEI]  # positions are already in rank order, highest first
            decisive = tuple(edges[order[0][j]].record_id for j in range(len(order[0])) if weight[j] >= 0.5)
        return PredBelief(st, belief, decisive)


class RuleAggregator(Aggregator):
    """Calibrated edge probabilities through the exact policy. `hard=True` uses each edge's top label
    only, which is what a system built from a classifier and if-statements would do."""

    def __init__(self, calibration: Optional[Calibration] = None, hard: bool = False, msg_dim: int = 128):
        self.cal = calibration or Calibration()
        self.hard = hard
        self.msg_dim = msg_dim
        self.version = f"rule-{'hard' if hard else 'soft'}-{self.cal.version()}"

    def status_batch(self, edge_sets):
        logits, msg, rank, auth, mask, order = pack(edge_sets, self.msg_dim)
        probs = self.cal.apply(logits)
        if self.hard:
            probs = F.one_hot(probs.argmax(-1), 3).to(probs.dtype)
        return precedence(probs, rank, mask), probs, order


@dataclass
class RefinerConfig:
    msg_dim: int = 128
    hidden: int = 96
    dropout: float = 0.1
    context: bool = True  # let an edge see a summary of the other edges in its set


class EdgeRefiner(nn.Module):
    """Re-reads each edge: logits' = logits / T + b + delta(message, logits, context).

    The last layer starts at zero, so an untrained refiner is exactly the rule aggregator. Whatever
    it learns is therefore a measured improvement over calibrated rules, not a different system.
    """

    def __init__(self, cfg: RefinerConfig):
        super().__init__()
        self.cfg = cfg
        self.log_t = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(3))
        self.inp = nn.Linear(cfg.msg_dim + 3, cfg.hidden)
        self.ctx = nn.Linear(cfg.hidden + 4, cfg.hidden) if cfg.context else None
        self.mid = nn.Linear(cfg.hidden * (2 if cfg.context else 1), cfg.hidden)
        self.out = nn.Linear(cfg.hidden, 3)
        self.drop = nn.Dropout(cfg.dropout)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, logits: torch.Tensor, msg: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        base = logits / torch.exp(self.log_t) + self.bias
        h = F.gelu(self.inp(torch.cat([msg, logits], dim=-1)))
        if self.ctx is not None:
            m = mask.unsqueeze(-1).to(h.dtype)
            n = m.sum(1, keepdim=True).clamp_min(1.0)
            mean = (h * m).sum(1, keepdim=True) / n
            p = F.softmax(base, dim=-1) * m
            # how much support / refutation the rest of the set carries, and how large the set is
            tot = p.sum(1, keepdim=True)
            others = (tot - p)
            size = torch.log1p(n).expand(-1, h.shape[1], -1)
            c = F.gelu(self.ctx(torch.cat([mean.expand_as(h), others, size], dim=-1)))
            h = torch.cat([h, c], dim=-1)
        h = self.drop(F.gelu(self.mid(h)))
        return F.softmax(base + self.out(h), dim=-1)


def rule_refiner(calibration: Calibration, msg_dim: int = 128, hidden: int = 96, seed: int = 0) -> EdgeRefiner:
    """Calibrated rules written as a refiner: temperature and bias set, the learned correction zero.

    It computes exactly what RuleAggregator(calibration) computes, and it can be the starting point
    of gated consolidation (ease/evolve/consolidate.py), which may later learn a correction if held-out
    feedback shows that one helps.

    The hidden layers do not affect the output while the last layer is zero. They are drawn from
    `seed`, not from the global generator, so the same calibration always gives the same weights,
    the same version and the same file."""
    ref = EdgeRefiner(RefinerConfig(msg_dim=msg_dim, hidden=hidden)).double()  # float32 would round T and b
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for lin in (ref.inp, ref.mid):  # the range nn.Linear uses by default
            bound = 1.0 / float(np.sqrt(lin.in_features))
            lin.weight.copy_((torch.rand(lin.weight.shape, generator=g, dtype=torch.float64) * 2 - 1) * bound)
            lin.bias.copy_((torch.rand(lin.bias.shape, generator=g, dtype=torch.float64) * 2 - 1) * bound)
        ref.log_t.fill_(float(np.log(calibration.temperature)))
        ref.bias.copy_(torch.tensor(calibration.bias, dtype=ref.bias.dtype))
        ref.out.weight.zero_()
        ref.out.bias.zero_()
        if ref.ctx is not None:  # rules ignore the rest of the set; so does any correction learned later
            ref.ctx.weight.zero_()
            ref.ctx.bias.zero_()
    return ref.eval()


class RefinedAggregator(Aggregator):
    def __init__(self, refiner: EdgeRefiner, version: Optional[str] = None):
        self.refiner = refiner.double().eval()
        self.msg_dim = refiner.cfg.msg_dim
        self.version = version or "refined-" + state_digest(refiner)

    def status_batch(self, edge_sets):
        logits, msg, rank, auth, mask, order = pack(edge_sets, self.msg_dim)
        if logits.shape[1] == 0:
            return precedence(logits, rank, mask), logits, order
        probs = self.refiner(logits, msg, mask)
        return precedence(probs, rank, mask), probs, order

    def save(self, directory: str | Path, extra: Optional[dict] = None) -> None:
        save_module(self.refiner, directory, {"kind": "refined", "cfg": asdict(self.refiner.cfg), **(extra or {})})

    @classmethod
    def load(cls, directory: str | Path) -> "RefinedAggregator":
        meta = json.loads((Path(directory) / "aggregator.json").read_text())
        ref = EdgeRefiner(RefinerConfig(**meta["cfg"]))
        load_module(ref, directory)
        return cls(ref)


@dataclass
class NeuralConfig:
    msg_dim: int = 128
    hidden: int = 96
    dropout: float = 0.1


class RankedReader(nn.Module):
    """Ablation: a GRU reads edges from lowest to highest rank and emits the status itself.
    Nothing in it executes the policy; it has to learn precedence, ties and conflicts from data."""

    def __init__(self, cfg: NeuralConfig):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(cfg.msg_dim + 3 + 3 + 2, cfg.hidden)
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        self.empty = nn.Parameter(torch.zeros(cfg.hidden))
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.hidden, 4)

    def forward(self, logits, msg, rank, auth, mask):
        B, N, _ = logits.shape
        if N == 0:
            return F.softmax(self.out(self.empty).expand(B, 4), dim=-1)
        lengths = mask.sum(1)
        # reverse each sequence so the highest-ranked edge is read last
        pos = torch.arange(N, device=logits.device).unsqueeze(0).expand(B, N)
        rev = (lengths.unsqueeze(1) - 1 - pos).clamp(min=0)
        tie_prev = torch.zeros(B, N, dtype=logits.dtype, device=logits.device)
        if N > 1:
            tie_prev[:, 1:] = (rank[:, 1:] == rank[:, :-1]).to(logits.dtype)
        x = torch.cat([msg, logits, F.softmax(logits, -1), (auth / 3.0).unsqueeze(-1), tie_prev.unsqueeze(-1)], dim=-1)
        x = F.gelu(self.inp(x)) * mask.unsqueeze(-1).to(logits.dtype)
        x = torch.gather(x, 1, rev.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        out, _ = self.gru(x)
        last = (lengths - 1).clamp(min=0)
        h = out[torch.arange(B), last]
        h = torch.where((lengths > 0).unsqueeze(-1), h, self.empty.expand_as(h))
        return F.softmax(self.out(self.drop(h)), dim=-1)


class NeuralAggregator(Aggregator):
    def __init__(self, reader: RankedReader, version: Optional[str] = None):
        self.reader = reader.double().eval()
        self.msg_dim = reader.cfg.msg_dim
        self.version = version or "neural-" + state_digest(reader)

    def status_batch(self, edge_sets):
        logits, msg, rank, auth, mask, order = pack(edge_sets, self.msg_dim)
        status = self.reader(logits, msg, rank, auth, mask)
        return status, F.softmax(logits, -1), order

    def save(self, directory: str | Path, extra: Optional[dict] = None) -> None:
        save_module(self.reader, directory, {"kind": "neural", "cfg": asdict(self.reader.cfg), **(extra or {})})

    @classmethod
    def load(cls, directory: str | Path) -> "NeuralAggregator":
        meta = json.loads((Path(directory) / "aggregator.json").read_text())
        rd = RankedReader(NeuralConfig(**meta["cfg"]))
        load_module(rd, directory)
        return cls(rd)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def state_digest(module: nn.Module) -> str:
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to(torch.float64).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def save_module(module: nn.Module, directory: str | Path, meta: dict) -> None:
    from safetensors.torch import save_file

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to(torch.float64).cpu().contiguous() for k, v in module.state_dict().items()}
    tmp = d / "aggregator.safetensors.tmp"
    save_file(state, str(tmp))
    tmp.replace(d / "aggregator.safetensors")
    (d / "aggregator.json").write_text(json.dumps({**meta, "digest": state_digest(module)}, indent=2))


def load_module(module: nn.Module, directory: str | Path) -> None:
    from safetensors.torch import load_file

    module.double()
    module.load_state_dict(load_file(str(Path(directory) / "aggregator.safetensors")), strict=True)


def load_aggregator(directory: str | Path) -> Aggregator:
    meta = json.loads((Path(directory) / "aggregator.json").read_text())
    if meta["kind"] == "refined":
        return RefinedAggregator.load(directory)
    if meta["kind"] == "neural":
        return NeuralAggregator.load(directory)
    raise ValueError(f"unknown aggregator kind {meta['kind']!r}")
