"""The aggregator used in deployment: a trained aggregator plus what has been learned since.

    reading of an edge =  memory( base reading )

The base reading is the trained refiner's (or, with no refiner, the calibrated model's)
probabilities. The memory then overrides exactly-matching verified pairs and pulls similar ones.
The exact precedence policy is applied last, unchanged.

Its version string is a hash of everything that can alter an output: base weights, calibration,
memory contents and memory settings. Any change to any of them changes the version, the engine
sees a new `w:agg` input, and belief nodes are recomputed. The encoder is not rerun.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ease.aggregate import (Aggregator, EdgeView, RefinedAggregator, RuleAggregator, pack, precedence)
from ease.evolve.memory import CorrectionMemory
from ease.util import text_hash


class EvolvingAggregator(Aggregator):
    def __init__(self, base: Aggregator, memory: Optional[CorrectionMemory] = None):
        if not isinstance(base, (RefinedAggregator, RuleAggregator)):
            raise TypeError("memory adjusts per-edge readings, so the base must produce them "
                            "(RefinedAggregator or RuleAggregator)")
        self.base = base
        self.memory = memory
        self.msg_dim = base.msg_dim
        self.last_adjusted = {"similar": 0, "exact": 0, "edges": 0}

    @property
    def version(self) -> str:
        mem = self.memory.version if self.memory is not None else "mem-none"
        return "evolving-" + text_hash(self.base.version, mem)[:16]

    @version.setter
    def version(self, _):  # the base class assigns a class attribute; ignore writes
        pass

    def status_batch(self, edge_sets: Sequence[Sequence[EdgeView]]):
        status, probs, order = self.base.status_batch(edge_sets)
        if self.memory is None or len(self.memory) == 0 or probs.shape[1] == 0:
            return status, probs, order
        logits, msg, rank, auth, mask, order2 = pack(edge_sets, self.msg_dim)
        p = probs.detach().to(torch.float64).numpy().copy()
        m = msg.numpy()
        n_sim = n_exact = n_edges = 0
        for b, edges in enumerate(edge_sets):
            n = len(edges)
            if n == 0:
                continue
            pairs = [edges[i].pair for i in order2[b]]
            adj, how = self.memory.adjust(p[b, :n], m[b, :n], pairs)
            p[b, :n] = adj
            n_sim += int((how == 1).sum()); n_exact += int((how == 2).sum()); n_edges += n
        self.last_adjusted = {"similar": n_sim, "exact": n_exact, "edges": n_edges}
        pt = torch.as_tensor(p, dtype=torch.float64)
        return precedence(pt, rank, mask), pt, order2
