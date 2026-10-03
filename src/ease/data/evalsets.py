"""Fixed evaluation sets, snapshotted to disk once so every later evaluation reads identical rows.

Split discipline:
    dev   rows used while developing (monitoring, temperature fitting, thresholds)
    test  rows read only by the final evaluation

A corpus with a single held-out split contributes it to `test` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ease.data.sources import get_source
from ease.util import read_jsonl, write_jsonl

# (name, source, split, role, limit). limit=None keeps the whole split.
EVAL_PLAN = [
    ("vitaminc.dev", "vitaminc", "validation", "dev", 6000),
    ("mnli.dev", "mnli", "validation_matched", "dev", None),
    ("snli.dev", "snli", "validation", "dev", 3000),
    ("vitaminc.test", "vitaminc", "test", "test", None),
    ("mnli_mm.test", "mnli", "validation_mismatched", "test", None),
    ("snli.test", "snli", "test", "test", None),
    ("wanli.test", "wanli", "test", "test", None),
    ("anli_r1.test", "anli", "test_r1", "test", None),
    ("anli_r2.test", "anli", "test_r2", "test", None),
    ("anli_r3.test", "anli", "test_r3", "test", None),
]


def build_eval_set(name: str, source: str, split: str, limit: Optional[int], out_dir: Path, seed: int = 0) -> Path:
    from datasets import load_dataset

    path = out_dir / f"{name}.jsonl"
    if path.exists():
        return path
    spec = get_source(source)
    ds = load_dataset(spec.hf_id, spec.config, split=split, streaming=True) if spec.config else load_dataset(
        spec.hf_id, split=split, streaming=True
    )
    rows = []
    keep_whole_groups = source == "vitaminc" and limit is not None
    seen_groups: set[str] = set()
    for raw in ds:
        row = spec.mapper(raw)
        if row is None:
            continue
        if source == "vitaminc":
            row["revision_type"] = raw.get("revision_type")
            row["page"] = raw.get("page")
        if source == "mnli":
            row["genre"] = raw.get("genre")
        if limit is not None and len(rows) >= limit:
            # finish the contrast set in progress, then stop: a truncated set would be unscorable
            if keep_whole_groups and row["group"] in seen_groups:
                rows.append(row)
                continue
            break
        seen_groups.add(row["group"])
        rows.append(row)
    write_jsonl(path, rows)
    return path


def build_unrelated(out_dir: Path, name: str, from_sets: list[str], n: int, seed: int) -> Path:
    """Random cross pairs of claims and evidence from different topics. Correct label: NEI."""
    import random

    from ease.labels import NEI

    path = out_dir / f"{name}.jsonl"
    if path.exists():
        return path
    pool = []
    for s in from_sets:
        for r in read_jsonl(out_dir / f"{s}.jsonl"):
            pool.append((r["claim"], r["evidence"], f"{r['source']}:{r.get('page') or r['group']}"))
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        a, b = rng.choice(pool), rng.choice(pool)
        if a[2] == b[2] or a[0] == b[0] or a[1] == b[1]:
            continue
        rows.append({"claim": a[0], "evidence": b[1], "label": NEI, "source": "unrelated", "group": str(len(rows))})
    write_jsonl(path, rows)
    return path


UNRELATED_PLAN = [
    ("unrelated.dev", ["vitaminc.dev", "mnli.dev"], "dev", 3000, 11),
    ("unrelated.test", ["vitaminc.test", "mnli_mm.test", "wanli.test"], "test", 6000, 12),
]


def build_all(out_dir: str | Path, verbose: bool = True) -> dict[str, dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, source, split, role, limit in EVAL_PLAN:
        p = build_eval_set(name, source, split, limit, out)
        n = sum(1 for _ in open(p, encoding="utf-8"))
        report[name] = {"source": source, "split": split, "role": role, "rows": n, "path": str(p),
                        "licence": get_source(source).license}
        if verbose:
            print(f"  {name:16s} {role:4s} rows={n:6d}  licence={get_source(source).license}", flush=True)
    for name, from_sets, role, n, seed in UNRELATED_PLAN:
        p = build_unrelated(out, name, from_sets, n, seed)
        report[name] = {"source": "unrelated", "split": "+".join(from_sets), "role": role, "rows": n, "path": str(p),
                        "licence": "derived from the listed sets"}
        if verbose:
            print(f"  {name:16s} {role:4s} rows={n:6d}  derived from {from_sets}", flush=True)
    return report


def load_eval_set(out_dir: str | Path, name: str) -> list[dict]:
    return read_jsonl(Path(out_dir) / f"{name}.jsonl")


def names(role: str) -> list[str]:
    return [n for n, _, _, r, _ in EVAL_PLAN if r == role] + [n for n, _, r, _, _ in UNRELATED_PLAN if r == role]
