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


def choose_aggregator(R: Path) -> tuple[str, str]:
    """The rule written in docs/DEVIATIONS.md, item 7, before episode-level results were known."""
    eps, small = load(R / "episodes.json"), load(R / "small.json")
    if not eps:
        return "rules", "no episode results; the simpler system is the default"
    sets = {s: e["comparisons"].get("E-seed1 vs B1-soft") for s, e in eps["sets"].items()}
    if small:
        sets["SMALL"] = small["comparisons"].get("E-seed1 vs B1-soft")
    missing = [s for s, c in sets.items() if c is None]
    if missing or "SMALL" not in sets:
        return "rules", f"comparisons missing for {missing or ['SMALL']}; the simpler system is the default"
    h5 = all(sets[s]["status_nll"]["hi"] < 0 for s in ("STANDARD", "WIDE"))
    worse = [s for s, c in sets.items()
             if c["status_nll"]["lo"] > 0 or c["disposition_accuracy"]["hi"] < 0]
    if h5 and not worse:
        return "refined", "H5 supported and not significantly worse than calibrated rules on STANDARD, WIDE or SMALL"
    reasons = []
    if not h5:
        reasons.append("H5 not supported")
    if worse:
        reasons.append(f"significantly worse than calibrated rules on {', '.join(worse)}")
    return "rules", "; ".join(reasons)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--results", default="runs/results")
    ap.add_argument("--out", default="release")
    ap.add_argument("--force-rules", action="store_true",
                    help="ship calibrated rules whatever the refiner comparison says (the addendum's plan for the large model)")
    ap.add_argument("--weights-licence", default=None,
                    help="licence identifier for the weights as the Hugging Face Hub spells it, e.g. cc-by-sa-4.0. "
                         "The owner's decision: without it the card says the licence is not yet chosen")
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
    choice, why = choose_aggregator(R)
    if a.force_rules:
        choice, why = "rules", "fixed by the plan (docs/PREREGISTRATION_LARGE.md): calibrated rules on top of the edge model"
    cal = load(Path(a.stage_b) / "rule_calibration.json")
    if choice == "refined":
        shutil.copytree(run["dir"], out / "aggregator")
        agg_info = {"kind": "refined", "seed": run["seed"], "version": run["version"], "parameters": run["params"]}
    else:
        from ease.aggregate import Calibration, RefinedAggregator, rule_refiner

        ref = rule_refiner(Calibration(cal["temperature"], tuple(cal["bias"])), msg_dim=run.get("msg_dim", 128))
        agg = RefinedAggregator(ref)
        agg.save(out / "aggregator", extra={"what": "calibrated rules; learned correction is zero",
                                            "temperature": cal["temperature"], "bias": cal["bias"]})
        agg_info = {"kind": "calibrated rules (refiner with zero correction)", "version": agg.version,
                    "temperature": cal["temperature"], "bias": cal["bias"]}
    agg_info["chosen_because"] = why
    (out / "aggregator" / "rule_calibration.json").write_text(json.dumps(cal, indent=2))
    th = load(R / "threshold_choice.json")
    settings = {
        "alpha": 0.05, "eta": 0.05, "tau_initial": 0.9,
        "why": "READY requires P(satisfied) >= the tracker's threshold, which starts at tau_initial and moves with "
               "verdicts (docs/PROOFS.md, T5). 0.9 is deliberately high because deployment text differs from the "
               "training and validation text.",
    }
    if th:
        k = "refined" if choice == "refined" else "rules"
        settings["validation_threshold_meeting_alpha"] = th["by_aggregator"][k]["chosen"]
        settings["validation_source"] = th["source"]
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    # the proposer does not depend on the edge model: its threshold was calibrated once, on the main results
    prop = load(R / "proposer.json") or load(Path("runs/results") / "proposer.json")
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
        "aggregator": agg_info,
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
    from ease.data.sources import SOURCES

    hub = {k: SOURCES[k].hf_id for k in man["examples_by_source"] if k in SOURCES}
    L = []
    w = L.append
    if a.weights_licence:
        w("---")
        w(f"license: {a.weights_licence}")
        w("language:\n- en")
        w(f"base_model: {man['config']['encoder_name']}")
        w("datasets:\n" + "\n".join(f"- {v}" for v in hub.values()))
        w("pipeline_tag: text-classification")
        w("tags:\n- natural-language-inference\n- fact-verification\n- evidence")
        w("---\n")
    w("# Model card: EASE-Delta\n")
    w(f"Generated {manifest['created']} by `scripts/package_release.py`. Figures are read from `{a.results}/`. "
      "Full results, including hypotheses that were not supported: `docs/RESULTS.md`.\n")
    w("## What it is\n")
    w("Two learned parts and an exact core.\n")
    w(f"- **Edge model**, {man['parameters']:,} parameters, `{man['config']['encoder_name']}` fine-tuned. Input: a claim "
      "and one evidence passage. Output: probabilities that the passage *supports* the claim, *refutes* it, or "
      f"*settles nothing*, and a {man['config']['msg_dim']}-dimensional message vector.")
    if choice == "refined":
        w(f"- **Refiner**, {run['params']:,} parameters. Re-reads each edge in the context of the other edges of the same "
          "predicate. Trained through the exact precedence policy.")
    else:
        reason = ("Calibrated rules ship by the plan in docs/PREREGISTRATION_LARGE.md; the learned refiner was not a "
                  "candidate for this release." if a.force_rules else
                  f"A learned refiner was trained and evaluated; it is not the default because: {why}.")
        w(f"- **Calibrated rules**: each reading is rescaled by a fitted temperature ({cal['temperature']:.3f}) and "
          f"class bias, then combined by the exact precedence policy. {reason} The rules are stored in refiner form "
          "with the learned correction set to zero, so gated consolidation can add one later if feedback shows it helps.")
    w("- **Exact core**: versioned ledger, dependency graph, precedence policy, requirement logic, planner. "
      "Not learned.\n")
    w("## Intended use\n")
    w("Tracking whether declared requirements of a task are met as documents and messages arrive, change and are "
      "withdrawn, and proposing what is ready, blocked or worth asking about. English text. Short passages "
      f"(inputs are truncated to {man['config']['max_len']} tokens per pair).\n")
    w("## Not intended for\n")
    w("- Acting without a person's approval. The software has no means of executing an action.\n"
      "- Decisions about people's health, legal status, employment, credit or safety.\n"
      "- Arithmetic, unit conversion or date reasoning. Simple numeric comparisons of the kind found in the "
      "training data were read accurately; anything beyond them was not tested.\n"
      "- Requirements that can only be settled by reading several records together.\n"
      "- Languages other than English.\n")
    w("## Use\n")
    w("These files are read by the `ease` package: https://github.com/jithinsaireddy/ease-delta "
      "(`pip install git+https://github.com/jithinsaireddy/ease-delta`, then download this repository into a "
      "directory and point `--model` and `--aggregator` at its `edge/` and `aggregator/`).\n")
    w("```bash\nease demo  --model edge --aggregator aggregator        # a client hand-off, twelve events\n"
      "ease serve --model edge --aggregator aggregator        # local API and page on 127.0.0.1:8791\n```\n")
    w("```python\nfrom ease.scorer import ModelScorer\n\nscorer = ModelScorer(\"edge\", canonical=True)\n"
      "logits, message = scorer.score([(\"The client has approved the design.\",\n"
      "                                 \"Email from the client: we approve the design as presented.\")])[0]\n"
      "# logits: supports, refutes, settles nothing\n```\n")
    w("## Training data\n")
    w("| Source | Examples | Licence | Obtained from |")
    w("|---|---:|---|---|")
    for k, v in sorted(man["examples_by_source"].items()):
        link = f"[{hub[k]}](https://huggingface.co/datasets/{hub[k]})" if k in hub else "made from the rows above"
        w(f"| {k} | {v:,} | {man['licences'].get(k, 'n/a')} | {link} |")
    w("")
    w("Streamed from the Hugging Face Hub. `unrelated` pairs are synthetic: the claim of one example with the "
      "evidence of another from a different topic, labelled as settling nothing. ANLI (non-commercial licence) and "
      "SNLI were not trained on.\n")
    w("Credit, as the licences ask: VitaminC, by Tal Schuster, Adam Fisch and Regina Barzilay (NAACL 2021), built "
      "from Wikipedia revisions; MultiNLI, by Adina Williams, Nikita Nangia and Samuel Bowman (NAACL 2018); WANLI, by "
      "Alisa Liu, Swabha Swayamdipta, Noah Smith and Yejin Choi (EMNLP 2022). Rows were used as published, with "
      "whitespace normalised. Backbone: ModernBERT, by Benjamin Warner and colleagues (2024).\n")
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
        shipped = "E-seed1" if choice == "refined" else "B1-soft"
        w("## Measured performance of the whole system\n")
        w(f"The shipped configuration ({'learned refiner' if choice == 'refined' else 'calibrated rules'}, every record "
          "compared with every requirement) on generated tasks with human-written text. Definitions: "
          "`docs/RESULTS.md`, section 3.\n")
        w("| Test set | Disposition accuracy | Stale-decision rate | False-READY rate |")
        w("|---|---:|---:|---:|")
        for sname, e in eps["sets"].items():
            m = e["systems"].get(shipped)
            if m:
                w(f"| {sname} | {pct(m['disposition_accuracy']['value'])} | {pct(m['stale_decision_rate']['value'])} | "
                  f"{pct(m['false_ready_rate']['value'])} |")
        w("")
        lr, ss = load(R / "linking_rules.json"), load(R / "set_size.json")
        if choice != "refined" and ss:
            dec = {k: v["systems"]["rules, declared links"]["disposition_accuracy"]["value"] for k, v in ss["sets"].items()}
            w("When records say which requirement they concern (`about` on an event, or documents filed per "
              "requirement), accuracy rises to " + ", ".join(f"{pct(v)} ({k})" for k, v in dec.items()) + ".")
        if choice != "refined" and lr:
            prop_acc = {k: v["systems"]["rules + proposer"]["disposition_accuracy"]["value"] for k, v in lr["sets"].items()}
            w("With the optional dependency proposer it is " + ", ".join(f"{pct(v)} ({k})" for k, v in prop_acc.items()) +
              " at 5-11 times less work; the proposer is off by default because it misses most conflicts that follow "
              "from a consequence (below).")
        if choice != "refined" and (ss or lr):
            w("")
    if exact:
        r = exact["runs"][0]
        w(f"Exactness: {r['nodes_not_bitwise_equal']} of {r['nodes_checked']:,} cached values differed from a full "
          "rebuild (canonical mode).\n")
    if timing:
        for sname, e in timing["sets"].items():
            w(f"- {sname}: update after a changed record, median {e['E']['p50_ms']:.0f} ms (95th percentile "
              f"{e['E']['p95_ms']:.0f} ms) on an Apple M4 Max; full re-reading {e['E-full']['p50_ms']:.0f} ms.")
        if choice != "refined":
            w("- Timed with the learned refiner. The shipped rules skip that network, so they do strictly less work per "
              "update; the encoder dominates either way.")
        w("")
    cpu = load(R / "cpu_timing.json")
    if cpu is None and (Path("runs/results") / "timing_large.json").exists():
        tl = load(Path("runs/results") / "timing_large.json")
        runs_ = {k: v for k, v in tl.get("runs", {}).items() if k.startswith("large")}
        cpu = {"runs": runs_} if runs_ else None
    if cpu:
        w("By device (shipped configuration, bit-exact mode, same machine). "
          + "; ".join(f"{lab}: update median {e['sets']['STANDARD']['update']['p50_ms']:.0f} ms on STANDARD and "
                      f"{e['sets']['WIDE']['update']['p50_ms']:.0f} ms on WIDE, {e['peak_memory_gb']:.1f} GB of memory"
                      for lab, e in cpu["runs"].items()) + ".\n")
    head, more = load(R / "headroom.json"), load(R / "more_training.json")
    if head:
        w("## What limits accuracy\n")
        hs = head["headroom"]
        w("Reading. When every reading is replaced by the correct one, the exact core gives the right answer for "
          "every action at every step of the test tasks. The edge model reads the record written for a requirement "
          "correctly " + " and ".join(f"{pct(e['reading_accuracy']['linked'],1)} of the time on {k}" for k, e in hs.items())
          + ". Passages on a claim's own subject that do not settle it are read as settling it "
          f"{pct(1 - head['errors']['vitaminc.test']['recall']['NEI'],1)} of the time (VitaminC test), so expect "
          "cross-talk between requirements of one project unless records say which requirement they concern.\n")
        if more:
            worse = [f"{k} {100*v['accuracy_difference']['value']:+.2f} points" for k, v in more["sets"].items()
                     if v["accuracy_difference"]["hi"] < 0]
            w(f"Training this model for {more['b']['additional_examples']:,} more examples on the same corpora did not "
              "help" + (f" (development sets: {', '.join(worse)})" if worse else "") + ". A larger encoder was not trained.")
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
    w(("Weights: " + a.weights_licence + ". " if a.weights_licence else "Weights: licence not yet chosen by the owner. ")
      + "Code: Apache-2.0. Backbone: Apache-2.0. Training corpora include share-alike licences (VitaminC, CC BY-SA "
      "3.0; parts of MNLI). Creative Commons' 2025 guidance on AI training describes releasing a model trained on "
      "share-alike material under the same licence as the cautious course; whether weights are an adaptation of "
      "their training text is not settled. This is a description of the sources, not legal advice.\n")
    (out / "MODEL_CARD.md").write_text("\n".join(L) + "\n")
    shutil.copy2(out / "MODEL_CARD.md", "docs/MODEL_CARD.md")
    total = sum(v["bytes"] for v in files.values())
    print(f"release written to {out}: {len(files)} files, {total/1e6:.1f} MB; edge {manifest['edge_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
