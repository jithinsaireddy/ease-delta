"""Leakage audit: how much text is shared between what the edge model trains on and what is used
to train downstream parts or to test?

Exact string matches after whitespace normalisation, for claims, evidence passages and pairs.
Shared *evidence* with a different claim is expected for Wikipedia-derived data and is reported
separately from shared *pairs*, which would be true leakage.
"""
import os, sys
from pathlib import Path
from ease.data.sources import get_source
from ease.util import read_jsonl, atomic_write_json, text_hash

def h(s): return text_hash(s)[:16]

train = {}
from datasets import load_dataset
for key in ["vitaminc", "mnli", "snli", "wanli"]:
    spec = get_source(key)
    ds = load_dataset(spec.hf_id, split=spec.train_split, streaming=True)
    claims, evs, pairs, pages = set(), set(), set(), set()
    n = 0
    for raw in ds:
        r = spec.mapper(raw)
        if r is None: continue
        claims.add(h(r["claim"])); evs.add(h(r["evidence"])); pairs.add(h(r["claim"] + "\x00" + r["evidence"]))
        if key == "vitaminc": pages.add(raw.get("page"))
        n += 1
    train[key] = dict(claims=claims, evs=evs, pairs=pairs, pages=pages, n=n)
    print(f"train {key}: rows={n} claims={len(claims)} evidence={len(evs)} pairs={len(pairs)} pages={len(pages)}", flush=True)

all_claims = set().union(*[t["claims"] for t in train.values()])
all_evs = set().union(*[t["evs"] for t in train.values()])
all_pairs = set().union(*[t["pairs"] for t in train.values()])
vit_pages = train["vitaminc"]["pages"]

report = {"train_rows": {k: v["n"] for k, v in train.items()}, "pools": {}, "eval_sets": {}}
for name in ["stage_b", "test"]:
    atoms = read_jsonl(f"runs/atoms/{name}.jsonl")
    n_pairs = sum(len(a["versions"]) for a in atoms)
    shared_pairs = sum(1 for a in atoms for v in a["versions"] if h(a["claim"] + "\x00" + v["text"]) in all_pairs)
    shared_claims = sum(1 for a in atoms if h(a["claim"]) in all_claims)
    shared_evs = sum(1 for a in atoms for v in a["versions"] if h(v["text"]) in all_evs)
    vit = [a for a in atoms if a["source"] == "vitaminc"]
    shared_pages = sum(1 for a in vit if a["page"] in vit_pages)
    report["pools"][name] = dict(atoms=len(atoms), pairs=n_pairs, pairs_in_train=shared_pairs,
        claims_in_train=shared_claims, evidence_in_train=shared_evs,
        vitaminc_atoms=len(vit), vitaminc_atoms_whose_page_is_in_train=shared_pages)
    print(name, report["pools"][name], flush=True)

sb = read_jsonl("runs/atoms/stage_b.jsonl"); te = read_jsonl("runs/atoms/test.jsonl")
sb_pairs = {h(a["claim"] + "\x00" + v["text"]) for a in sb for v in a["versions"]}
sb_claims = {h(a["claim"]) for a in sb}
te_pairs_shared = sum(1 for a in te for v in a["versions"] if h(a["claim"] + "\x00" + v["text"]) in sb_pairs)
te_claims_shared = sum(1 for a in te if h(a["claim"]) in sb_claims)
report["stage_b_vs_test"] = dict(test_pairs_in_stage_b=te_pairs_shared, test_claims_in_stage_b=te_claims_shared)
print("stage_b vs test", report["stage_b_vs_test"], flush=True)

for p in sorted(Path("runs/eval_cache").glob("*.jsonl")):
    rows = read_jsonl(p)
    report["eval_sets"][p.stem] = dict(rows=len(rows),
        pairs_in_train=sum(1 for r in rows if h(r["claim"] + "\x00" + r["evidence"]) in all_pairs),
        claims_in_train=sum(1 for r in rows if h(r["claim"]) in all_claims),
        evidence_in_train=sum(1 for r in rows if h(r["evidence"]) in all_evs))
    print(p.stem, report["eval_sets"][p.stem], flush=True)
atomic_write_json("runs/leakage_audit.json", report)
print("DONE"); sys.stdout.flush(); os._exit(0)
