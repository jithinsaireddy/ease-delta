"""Incremental evaluation of a computation DAG.

Every quantity a decision depends on is a node: evidence records, claim texts, model weight
versions, the schema version, the clock. A change to any of them is a change to an input node, and
`propagate()` recomputes what that change can reach and nothing else.

Two facts make this exact rather than approximate (proofs in docs/PROOFS.md):

  T1  A node none of whose ancestors changed keeps a valid cached value.
  T2  Recomputing reachable nodes in topological order gives every node the same parent values a
      full rebuild would give it, hence the same output, provided node functions are
      deterministic functions of their parent values.

Early cutoff: if a recomputed node returns a value equal to its cached value, its children are not
scheduled on its account. This is the rule build systems use to stop a rebuild when a recompiled
file is unchanged (Mokhov, Mitchell & Peyton Jones, "Build Systems a la Carte", 2018). It preserves
T2 because a child whose parents all hold their old values would reproduce its old output.

Incremental evaluation of neural graphs is prior art (InkStream, Wu et al. 2023; Sharir &
Anandkumar 2023). What this module adds is only engineering: batched node functions so that many
dirty nodes of one kind become a single GPU call, and cost accounting per kind.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

INPUT = "input"

# A batch function receives the ids and parent-value lists of every node of one kind that must be
# recomputed at one level, and returns one value per node, in order.
BatchFn = Callable[[Sequence[str], Sequence[Sequence[Any]]], Sequence[Any]]


def values_equal(a: Any, b: Any) -> bool:
    """Exact equality. Floats compare bitwise-equal values (NaN equals NaN); no tolerance is applied,
    because a tolerance would turn exact reuse into approximate reuse."""
    if a is b:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b, equal_nan=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    return type(a) is type(b) and a == b


def max_abs_difference(a: Any, b: Any) -> float:
    """Largest absolute numeric difference between two values of the same structure;
    inf if the structures differ."""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return float("inf")
        if a.size == 0:
            return 0.0
        if a.dtype.kind in "fiub":
            return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        return 0.0 if np.array_equal(a, b) else float("inf")
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max([max_abs_difference(a[k], b[k]) for k in a], default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max([max_abs_difference(x, y) for x, y in zip(a, b)], default=0.0)
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if a == b else float("inf")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


class CycleError(ValueError):
    pass


@dataclass
class Node:
    id: str
    kind: str
    parents: tuple[str, ...] = ()
    value: Any = None
    level: int = 0
    computed: bool = False  # a computed node that has never been evaluated has no valid cache


@dataclass
class PropagationReport:
    recomputed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    unchanged_after_recompute: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recomputed_ids: list[str] = field(default_factory=list)
    changed_ids: list[str] = field(default_factory=list)
    batches: int = 0

    @property
    def total_recomputed(self) -> int:
        return sum(self.recomputed.values())

    def as_dict(self) -> dict:
        return {
            "recomputed": dict(self.recomputed),
            "unchanged_after_recompute": dict(self.unchanged_after_recompute),
            "total_recomputed": self.total_recomputed,
            "changed": len(self.changed_ids),
            "batches": self.batches,
        }


class IncrementalGraph:
    def __init__(self, functions: Optional[dict[str, BatchFn]] = None, early_cutoff: bool = True):
        self.functions: dict[str, BatchFn] = dict(functions or {})
        self.early_cutoff = early_cutoff
        self.nodes: dict[str, Node] = {}
        self.children: dict[str, set[str]] = defaultdict(set)
        self._pending: set[str] = set()

    # -- construction ------------------------------------------------------
    def register(self, kind: str, fn: BatchFn) -> None:
        self.functions[kind] = fn

    def add_input(self, node_id: str, value: Any) -> None:
        if node_id in self.nodes:
            raise KeyError(f"node {node_id!r} already exists")
        self.nodes[node_id] = Node(node_id, INPUT, (), value, 0, True)

    def add_node(self, node_id: str, kind: str, parents: Sequence[str]) -> None:
        if node_id in self.nodes:
            raise KeyError(f"node {node_id!r} already exists")
        if kind == INPUT:
            raise ValueError("use add_input for input nodes")
        if kind not in self.functions:
            raise KeyError(f"no function registered for kind {kind!r}")
        for p in parents:
            if p not in self.nodes:
                raise KeyError(f"parent {p!r} of {node_id!r} does not exist")
        self.nodes[node_id] = Node(node_id, kind, tuple(parents), None,
                                   1 + max((self.nodes[p].level for p in parents), default=0), False)
        for p in parents:
            self.children[p].add(node_id)
        self._pending.add(node_id)

    def set_parents(self, node_id: str, parents: Sequence[str]) -> None:
        """Change what a computed node depends on. The node is rescheduled; levels are repaired."""
        node = self.nodes[node_id]
        if node.kind == INPUT:
            raise ValueError("input nodes have no parents")
        parents = tuple(parents)
        if parents == node.parents:
            return
        for p in parents:
            if p not in self.nodes:
                raise KeyError(f"parent {p!r} does not exist")
        if parents and self._reaches(node_id, set(parents)):
            raise CycleError(f"making {parents} parents of {node_id!r} would create a cycle")
        for p in node.parents:
            self.children[p].discard(node_id)
        node.parents = parents
        for p in parents:
            self.children[p].add(node_id)
        self._repair_levels(node_id)
        self._pending.add(node_id)

    def remove_node(self, node_id: str) -> None:
        if self.children.get(node_id):
            raise ValueError(f"cannot remove {node_id!r}: it still has children {sorted(self.children[node_id])[:3]}")
        node = self.nodes.pop(node_id)
        for p in node.parents:
            self.children[p].discard(node_id)
        self.children.pop(node_id, None)
        self._pending.discard(node_id)

    def _reaches(self, start: str, targets: set[str]) -> bool:
        stack, seen = [start], set()
        while stack:
            v = stack.pop()
            if v in targets:
                return True
            if v in seen:
                continue
            seen.add(v)
            stack.extend(self.children.get(v, ()))
        return False

    def _repair_levels(self, start: str) -> None:
        stack = [start]
        while stack:
            v = stack.pop()
            n = self.nodes[v]
            lvl = 1 + max((self.nodes[p].level for p in n.parents), default=0)
            if lvl != n.level or v == start:
                n.level = lvl
                stack.extend(self.children.get(v, ()))

    # -- inputs ------------------------------------------------------------
    def set_input(self, node_id: str, value: Any) -> bool:
        """Returns True if the value differed from the cached one. An identical value schedules nothing."""
        node = self.nodes[node_id]
        if node.kind != INPUT:
            raise ValueError(f"{node_id!r} is computed; only input nodes can be set")
        if values_equal(node.value, value):
            return False
        node.value = value
        self._pending.update(self.children.get(node_id, ()))
        return True

    def invalidate(self, node_id: str) -> None:
        """Force a computed node to be re-evaluated (used when something outside the graph changed)."""
        if self.nodes[node_id].kind != INPUT:
            self._pending.add(node_id)
        else:
            self._pending.update(self.children.get(node_id, ()))

    def value(self, node_id: str) -> Any:
        node = self.nodes[node_id]
        if node.kind != INPUT and (not node.computed or node_id in self._pending):
            raise RuntimeError(f"{node_id!r} is stale; call propagate() first")
        return node.value

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    # -- evaluation --------------------------------------------------------
    def _evaluate_group(self, kind: str, ids: list[str], lookup: Callable[[str], Any]) -> Sequence[Any]:
        parent_values = [[lookup(p) for p in self.nodes[i].parents] for i in ids]
        out = self.functions[kind](ids, parent_values)
        if len(out) != len(ids):
            raise RuntimeError(f"function for kind {kind!r} returned {len(out)} values for {len(ids)} nodes")
        return out

    def propagate(self) -> PropagationReport:
        report = PropagationReport()
        heap: list[tuple[int, str]] = [(self.nodes[i].level, i) for i in self._pending]
        heapq.heapify(heap)
        queued = set(self._pending)
        self._pending = set()
        while heap:
            level = heap[0][0]
            batch_ids = []
            while heap and heap[0][0] == level:
                _, i = heapq.heappop(heap)
                if i in self.nodes:
                    batch_ids.append(i)
            by_kind: dict[str, list[str]] = defaultdict(list)
            for i in sorted(batch_ids):
                by_kind[self.nodes[i].kind].append(i)
            for kind in sorted(by_kind):
                ids = by_kind[kind]
                new_values = self._evaluate_group(kind, ids, lambda p: self.nodes[p].value)
                report.batches += 1
                for i, v in zip(ids, new_values):
                    node = self.nodes[i]
                    same = node.computed and values_equal(node.value, v)
                    node.value, node.computed = v, True
                    report.recomputed[kind] += 1
                    report.recomputed_ids.append(i)
                    if same:
                        report.unchanged_after_recompute[kind] += 1
                        if self.early_cutoff:
                            continue
                    else:
                        report.changed_ids.append(i)
                    for c in self.children.get(i, ()):
                        if c not in queued:
                            queued.add(c)
                            heapq.heappush(heap, (self.nodes[c].level, c))
        return report

    def full_recompute(self) -> dict[str, Any]:
        """Evaluate every computed node from the current inputs into a fresh table.
        Caches are neither read nor written, so this is an independent reference."""
        fresh: dict[str, Any] = {i: n.value for i, n in self.nodes.items() if n.kind == INPUT}
        by_level: dict[int, list[str]] = defaultdict(list)
        for i, n in self.nodes.items():
            if n.kind != INPUT:
                by_level[n.level].append(i)
        for level in sorted(by_level):
            by_kind: dict[str, list[str]] = defaultdict(list)
            for i in sorted(by_level[level]):
                by_kind[self.nodes[i].kind].append(i)
            for kind in sorted(by_kind):
                ids = by_kind[kind]
                for i, v in zip(ids, self._evaluate_group(kind, ids, lambda p: fresh[p])):
                    fresh[i] = v
        return fresh

    def verify(self) -> dict:
        """Compare cached values with an independent full recomputation."""
        if self._pending:
            raise RuntimeError("propagate() before verify()")
        fresh = self.full_recompute()
        worst, worst_id, mismatched = 0.0, None, 0
        for i, n in self.nodes.items():
            if n.kind == INPUT:
                continue
            if not values_equal(n.value, fresh[i]):
                mismatched += 1
            d = max_abs_difference(n.value, fresh[i])
            if d > worst:
                worst, worst_id = d, i
        computed = sum(1 for n in self.nodes.values() if n.kind != INPUT)
        return {"computed_nodes": computed, "not_bitwise_equal": mismatched, "max_abs_difference": worst,
                "worst_node": worst_id}

    # -- introspection -----------------------------------------------------
    def ancestors(self, node_id: str) -> set[str]:
        seen, stack = set(), list(self.nodes[node_id].parents)
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(self.nodes[v].parents)
        return seen

    def descendants(self, node_id: str) -> set[str]:
        seen, stack = set(), list(self.children.get(node_id, ()))
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(self.children.get(v, ()))
        return seen

    def count_computed(self) -> int:
        return sum(1 for n in self.nodes.values() if n.kind != INPUT)
