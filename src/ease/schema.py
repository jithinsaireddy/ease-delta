"""Task schema: what must hold, how requirements combine, and what each action needs.

The schema is declared by a person. It is the part of a task where being exactly right is possible,
so it is executed exactly (ease/logic.py) rather than learned.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from ease.util import text_hash

AND, OR, NOT, ATLEAST = "AND", "OR", "NOT", "ATLEAST"
OPS = (AND, OR, NOT, ATLEAST)


@dataclass(frozen=True)
class Predicate:
    """A statement that evidence can support or refute."""

    id: str
    text: str
    prior: float = 0.5  # probability it holds when no evidence bears on it
    ask_cost: float = 1.0  # cost of asking a person to settle it, in the task's utility units
    join: bool = False  # True if settling it needs several records read together

    def __post_init__(self):
        if not self.id or not self.text.strip():
            raise ValueError("predicate needs an id and non-empty text")
        if not 0.0 <= self.prior <= 1.0:
            raise ValueError("prior must be a probability")
        if self.ask_cost < 0:
            raise ValueError("ask_cost must be non-negative")


@dataclass(frozen=True)
class Gate:
    id: str
    op: str
    children: tuple[str, ...]
    k: int = 0

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"unknown op {self.op!r}")
        if not self.children:
            raise ValueError("a gate needs at least one child")
        if self.op == NOT and len(self.children) != 1:
            raise ValueError("NOT takes exactly one child")
        if self.op == ATLEAST and not 1 <= self.k <= len(self.children):
            raise ValueError("ATLEAST needs 1 <= k <= number of children")
        if len(set(self.children)) != len(self.children):
            raise ValueError("a gate may not list the same child twice")


@dataclass(frozen=True)
class Action:
    id: str
    description: str
    requires: str  # id of a predicate or gate
    value_success: float = 1.0
    cost_failure: float = 1.0
    reliability: float = 1.0  # P(success | requirement truly holds)
    leak: float = 0.0  # P(success | requirement does not hold)
    needs_approval: bool = True  # the engine proposes; it never executes

    def __post_init__(self):
        for name in ("reliability", "leak"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be a probability")


@dataclass(frozen=True)
class Policy:
    """How records about the same question are ranked. Arrival time is deliberately absent."""

    rank_by: tuple[str, ...] = ("authority", "valid_from")


@dataclass
class TaskSchema:
    task_id: str
    predicates: list[Predicate] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    policy: Policy = field(default_factory=Policy)

    def __post_init__(self):
        self.validate()

    # -- structure ---------------------------------------------------------
    def validate(self) -> None:
        ids = [p.id for p in self.predicates] + [g.id for g in self.gates]
        if len(set(ids)) != len(ids):
            raise ValueError("predicate and gate ids must be unique")
        a_ids = [a.id for a in self.actions]
        if len(set(a_ids)) != len(a_ids):
            raise ValueError("action ids must be unique")
        known = set(ids)
        for g in self.gates:
            for c in g.children:
                if c not in known:
                    raise ValueError(f"gate {g.id!r} refers to unknown node {c!r}")
        for a in self.actions:
            if a.requires not in known:
                raise ValueError(f"action {a.id!r} requires unknown node {a.requires!r}")
        self.topological_gates()

    def topological_gates(self) -> list[Gate]:
        by_id = {g.id: g for g in self.gates}
        done: dict[str, int] = {}
        order: list[Gate] = []

        def visit(gid: str, stack: tuple[str, ...]) -> None:
            if done.get(gid) == 2:
                return
            if gid in stack:
                raise ValueError(f"gates form a cycle through {gid!r}")
            for c in by_id[gid].children:
                if c in by_id:
                    visit(c, stack + (gid,))
            done[gid] = 2
            order.append(by_id[gid])

        for g in self.gates:
            visit(g.id, ())
        return order

    def predicate(self, pid: str) -> Predicate:
        for p in self.predicates:
            if p.id == pid:
                return p
        raise KeyError(pid)

    def leaves_of(self, node_id: str) -> list[str]:
        """Predicate ids under a node, with repetition if the formula reuses one."""
        by_id = {g.id: g for g in self.gates}
        if node_id not in by_id:
            return [node_id]
        out: list[str] = []
        for c in by_id[node_id].children:
            out.extend(self.leaves_of(c))
        return out

    def version(self) -> str:
        """Changes whenever anything a decision could depend on changes."""
        return text_hash(json.dumps(self.to_dict(), sort_keys=True))[:16]

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "predicates": [asdict(p) for p in self.predicates],
            "gates": [{**asdict(g), "children": list(g.children)} for g in self.gates],
            "actions": [asdict(a) for a in self.actions],
            "policy": {"rank_by": list(self.policy.rank_by)},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSchema":
        return cls(
            task_id=d["task_id"],
            predicates=[Predicate(**p) for p in d.get("predicates", [])],
            gates=[Gate(g["id"], g["op"], tuple(g["children"]), g.get("k", 0)) for g in d.get("gates", [])],
            actions=[Action(**a) for a in d.get("actions", [])],
            policy=Policy(tuple(d.get("policy", {}).get("rank_by", ("authority", "valid_from")))),
        )
