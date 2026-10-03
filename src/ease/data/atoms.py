"""Atoms: real claim/evidence units from which evolving episodes are composed.

A *revision atom* is one claim with two versions of its evidence, taken from a VitaminC case: the
same sentence before and after a real Wikipedia edit, each with a human-assigned label. Replacing
one version with the other is a revision event whose effect on the claim is known.

A *static atom* is one claim with one evidence passage (MNLI, SNLI, WANLI).

Pools are disjoint from anything the edge model is trained on:

    stage_b   VitaminC validation after the dev rows, MNLI validation_matched[5000:],
              SNLI validation[3000:]. Used to train and tune everything downstream of the edge model.
    test      VitaminC test, MNLI validation_mismatched, SNLI test, WANLI test.
              Read only by the final evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

from ease.data.sources import get_source
from ease.labels import NEI, REFUTES, SUPPORTS
from ease.util import read_jsonl, write_jsonl

DEV_ROWS = {"vitaminc": 6001, "mnli": 5000, "snli": 3000}


def _stream(source: str, split: str, skip: int = 0, limit: Optional[int] = None) -> list[dict]:
    from datasets import load_dataset

    spec = get_source(source)
    ds = load_dataset(spec.hf_id, spec.config, split=split, streaming=True) if spec.config else load_dataset(
        spec.hf_id, split=split, streaming=True)
    rows = []
    for i, raw in enumerate(ds):
        if i < skip:
            continue
        row = spec.mapper(raw)
        if row is None:
            continue
        if source == "vitaminc":
            row["page"] = raw.get("page") or row["group"]
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def vitaminc_atoms(rows: list[dict], tag: str) -> list[dict]:
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["group"]].append(r)
    atoms = []
    for case, rs in by_case.items():
        by_claim = defaultdict(dict)
        for r in rs:
            by_claim[r["claim"]][r["evidence"]] = r["label"]  # identical pairs collapse
        for ci, (claim, ev) in enumerate(sorted(by_claim.items())):
            versions = [{"text": t, "label": l} for t, l in sorted(ev.items())]
            if not versions:
                continue
            atoms.append({
                "atom_id": f"vitc:{tag}:{case}:{ci}",
                "kind": "revision" if len(versions) >= 2 else "static",
                "claim": claim,
                "versions": versions[:2],
                "page": rs[0].get("page") or case,
                "source": "vitaminc",
            })
    return atoms


def static_atoms(rows: list[dict], source: str, tag: str) -> list[dict]:
    return [{
        "atom_id": f"{source}:{tag}:{i}",
        "kind": "static",
        "claim": r["claim"],
        "versions": [{"text": r["evidence"], "label": r["label"]}],
        "page": f"{source}:{tag}:{r['group']}",
        "source": source,
    } for i, r in enumerate(rows)]


def build_pools(out_dir: str | Path, verbose: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    def save(name: str, atoms: list[dict]) -> None:
        write_jsonl(out / f"{name}.jsonl", atoms)
        kinds = defaultdict(int)
        for a in atoms:
            kinds[(a["source"], a["kind"])] += 1
        report[name] = {"atoms": len(atoms), "by_source_kind": {f"{s}/{k}": v for (s, k), v in sorted(kinds.items())}}
        if verbose:
            print(f"  {name}: {len(atoms)} atoms {report[name]['by_source_kind']}", flush=True)

    if not (out / "stage_b.jsonl").exists():
        b = vitaminc_atoms(_stream("vitaminc", "validation", skip=DEV_ROWS["vitaminc"]), "val")
        b += static_atoms(_stream("mnli", "validation_matched", skip=DEV_ROWS["mnli"]), "mnli", "val")
        b += static_atoms(_stream("snli", "validation", skip=DEV_ROWS["snli"]), "snli", "val")
        save("stage_b", b)
    if not (out / "test.jsonl").exists():
        t = vitaminc_atoms(_stream("vitaminc", "test"), "test")
        t += static_atoms(_stream("mnli", "validation_mismatched"), "mnli", "test")
        t += static_atoms(_stream("snli", "test"), "snli", "test")
        t += static_atoms(_stream("wanli", "test"), "wanli", "test")
        save("test", t)
    return report


def load_pool(out_dir: str | Path, name: str) -> list[dict]:
    return read_jsonl(Path(out_dir) / f"{name}.jsonl")


def label_name(l: int) -> str:
    return {SUPPORTS: "SUPPORTS", REFUTES: "REFUTES", NEI: "NEI"}[l]
