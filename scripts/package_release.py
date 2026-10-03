"""Assemble a release directory and its model card from trained artefacts and measured results.

    python scripts/package_release.py --out release

    release/
      edge/                 edge model, tokenizer, training manifest, regression suite
      aggregator/           the refiner (seed 1) trained through the exact policy
      MANIFEST.json         file hashes, versions, provenance
      MODEL_CARD.md         generated from the manifests and docs/RESULTS inputs

Every figure in the model card is read from a results file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=2):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--results", default="runs/results")
    ap.add_argument("--out", default="release")
    a = ap.parse_args()
    out, R = Path(a.out), Path(a.results)
    if out.exists():
        shutil.rmtree(out)
    (out / "edge").mkdir(parents=True)
    for f in Path(a.edge).iterdir():
        if f.is_file():
            shutil.copy2(f, out / "edge" / f.name)
    sb = load(Path(a.stage_b) / "summary.json")
    run = [r for r in sb["runs"] if r["kind"] == "refined"][0]
    shutil.copytree(run["dir"], out / "aggregator")
    shutil.copy2(Path(a.stage_b) / "rule_calibration.json", out / "aggregator" / "rule_calibration.json")
    prop = load(R / "proposer.json")
    if prop and prop["calibration"].get("feasible"):
        c = prop["calibration"]
        (out / "proposer.json").write_text(json.dumps(
            {"embedder": c["embedder"], "threshold": c["threshold"], "alpha": c["alpha"], "min_links": 1}, indent=2))

    import datasets
    import torch
    import transformers

    man = load(Path(a.edge) / "training_manifest.json")
    files = {str(p.relative_to(out)): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {
        "name": "EASE-Delta", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "edge_version": "edge-" + files["edge/model.safetensors"]["sha256"][:16],
        "aggregator": {"kind": "refined", "seed": run["seed"], "version": run["version"], "parameters": run["params"]},
        "backbone": man["config"]["encoder_name"], "parameters_edge": man["parameters"],
        "training_examples": man["examples"], "examples_by_source": man["examples_by_source"],
        "licences_of_training_data": man["licences"],
        "preregistration_sha256": Path("runs/prereg_hash.txt").read_text().split()[0],
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__},
        "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    edge, eps = load(R / "edge.json"), load(R / "episodes.json")
    exact, timing, stale, _evo = load(R / "exactness.json"), load(R / "timing.json"), load(R / "stale.json"), load(R / "evolution.json")
    L = []
    w = L.append
    w("# Model card: EASE-Delta\n")
    w(f"Generated {manifest['created']} by `scripts/package_release.py`. Figures are read from `{a.results}/`. "
      "Full results, including hypotheses that were not supported: `docs/RESULTS.md`.\n")
    w("## What it is\n")
    w("Two learned parts and an exact core.\n")
    w(f"- **Edge model**, {man['parameters']:,} parameters, `{man['config']['encoder_name']}` fine-tuned. Input: a claim "
      "and one evidence passage. Output: probabilities that the passage *supports* the claim, *refutes* it, or "
      f"*settles nothing*, and a {man['config']['msg_dim']}-dimensional message vector.")
    w(f"- **Refiner**, {run['params']:,} parameters. Re-reads each edge in the context of the other edges of the same "
      "predicate. Trained through the exact precedence policy.")
    w("- **Exact core**: versioned ledger, dependency graph, precedence policy, requirement logic, planner. "
      "Not learned.\n")
    w("## Intended use\n")
    w("Tracking whether declared requirements of a task are met as documents and messages arrive, change and are "
      "withdrawn, and proposing what is ready, blocked or worth asking about. English text. Short passages "
      f"(inputs are truncated to {man['config']['max_len']} tokens per pair).\n")
    w("## Not intended for\n")
    w("- Acting without a person's approval. The software has no means of executing an action.\n"
      "- Decisions about people's health, legal status, employment, credit or safety.\n"
      "- Comparing quantities or dates stated in text. The model reads numbers as words.\n"
      "- Requirements that can only be settled by reading several records together.\n"
      "- Languages other than English.\n")
    w("## Training data\n")
    w("| Source | Examples | Licence |")
    w("|---|---:|---|")
    for k, v in sorted(man["examples_by_source"].items()):
        w(f"| {k} | {v:,} | {man['licences'].get(k, 'n/a')} |")
    w("")
    w("Streamed from the Hugging Face Hub. `unrelated` pairs are synthetic: the claim of one example with the "
      "evidence of another from a different topic, labelled as settling nothing. ANLI (non-commercial licence) and "
      "SNLI were not trained on.\n")
    if edge:
        w("## Measured performance of the edge model\n")
        w("| Set | n | Accuracy | Calibration error (ECE) after temperature scaling |")
        w("|---|---:|---:|---:|")
        for name, e in edge["sets"].items():
            if e["role"] == "test":
                w(f"| {name} | {e['n']:,} | {pct(e['raw']['accuracy'])} | {e['calibrated']['ece']:.3f} |")
        w("")
        u = edge["sets"].get("unrelated.test")
        if u:
            w(f"Unrelated passages read as decisive: {pct(u['false_decisive_rate'])}.\n")
    if eps:
        w("## Measured performance of the whole system\n")
        w("On generated tasks with human-written text (see `docs/RESULTS.md`, section 3, for definitions).\n")
        w("| Test set | Disposition accuracy | Stale-decision rate | False-READY rate |")
        w("|---|---:|---:|---:|")
        for s, e in eps["sets"].items():
            m = e["systems"].get("E-seed1")
            if m:
                w(f"| {s} | {pct(m['disposition_accuracy']['value'])} | {pct(m['stale_decision_rate']['value'])} | "
                  f"{pct(m['false_ready_rate']['value'])} |")
        w("")
    if exact:
        r = exact["runs"][0]
        w(f"Exactness: {r['nodes_not_bitwise_equal']} of {r['nodes_checked']:,} cached values differed from a full "
          "rebuild (canonical mode).\n")
    if timing:
        for s, e in timing["sets"].items():
            w(f"- {s}: update after a changed record, median {e['E']['p50_ms']:.0f} ms (95th percentile "
              f"{e['E']['p95_ms']:.0f} ms) on an Apple M4 Max; full re-reading {e['E-full']['p50_ms']:.0f} ms.")
        w("")
    if stale:
        t1, t2 = stale["by_type"]["T1"], stale["by_type"]["T2"]
        w("## Known weakness: implicit conflicts\n")
        w(f"On STALE, at {pct(stale['false_positive_rate_on_controls'],1)} false positives, the edge model recognised "
          f"{pct(t1['detection_rate'],1)} of direct conflicts and {pct(t2['detection_rate'],1)} of conflicts that follow "
          "from a consequence. It should not be relied on to notice that a new fact undermines an old one unless the "
          "two are about the same thing in similar words.\n")
    w("## Limitations\n")
    lim = Path("docs/LIMITATIONS.md")
    w(lim.read_text().strip() + "\n" if lim.exists() else "See docs/LIMITATIONS.md.\n")
    w("## Licence\n")
    w("Backbone: Apache-2.0. Training corpora include share-alike licences (VitaminC CC BY-SA 3.0; parts of MNLI). "
      "Take advice before redistributing these weights commercially.\n")
    (out / "MODEL_CARD.md").write_text("\n".join(L) + "\n")
    shutil.copy2(out / "MODEL_CARD.md", "docs/MODEL_CARD.md")
    total = sum(v["bytes"] for v in files.values())
    print(f"release written to {out}: {len(files)} files, {total/1e6:.1f} MB; edge {manifest['edge_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
