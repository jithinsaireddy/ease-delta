"""Build docs/RESULTS.md from the JSON files the experiments wrote.

Every number in the report is read from a results file by this script. Verdicts on hypotheses are
computed here from the decision rules in docs/PREREGISTRATION.md, not judged by eye.

    python scripts/make_report.py --results runs/results --out docs/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:.{d}f}"


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def ci(m, d=4, as_pct=False):
    if m is None:
        return "n/a"
    g = (lambda v: pct(v, 2)) if as_pct else (lambda v: f(v, d))
    if "lo" in m:
        return f"{g(m['value'])} [{g(m['lo'])}, {g(m['hi'])}]"
    return g(m["value"])


def diff(c, d=4, as_pct=False):
    if c is None:
        return "n/a"
    g = (lambda v: f"{100 * v:+.2f} pts") if as_pct else (lambda v: f"{v:+.{d}f}")
    return f"{g(c['difference'])} [{g(c['lo'])}, {g(c['hi'])}]"


def verdict(ok):
    return {True: "**supported**", False: "**not supported**", None: "not evaluated"}[ok]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/results")
    ap.add_argument("--out", default="docs/RESULTS.md")
    ap.add_argument("--stage-a", default="runs/stage_a/final")
    ap.add_argument("--dense", default="runs/dense/final")
    ap.add_argument("--deviations", default="docs/DEVIATIONS.md")
    a = ap.parse_args()
    R = Path(a.results)
    edge, b3 = load(R / "edge.json"), load(R / "edge_b3.json")
    eps, exact, timing = load(R / "episodes.json"), load(R / "exactness.json"), load(R / "timing.json")
    timing_c = load(R / "timing_canonical.json")
    prop, stale, evo = load(R / "proposer.json"), load(R / "stale.json"), load(R / "evolution.json")
    joins = load(R / "joins.json")
    numeric = load(R / "numeric.json")
    demo_trace = load(R / "demo_trace.json")
    small = load(R / "small.json")
    set_size = load(R / "set_size.json")
    linking_rules = load(R / "linking_rules.json")
    evo_rules = load(R / "evolution_rules.json")
    suites_check = load(R / "set_suites_check.json")
    man = load(Path(a.stage_a) / "training_manifest.json")
    dman = load(Path(a.dense) / "training_manifest.json")
    probe = load(Path("runs/mechanism_probe.json"))
    audit = load(Path("runs/leakage_audit.json"))
    determinism = load(Path("runs/determinism_probe.json"))
    prereg = Path("runs/prereg_hash.txt").read_text().split() if Path("runs/prereg_hash.txt").exists() else ["?"]

    L: list[str] = []
    w = L.append
    w("# EASE-Delta: results\n")
    w(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/make_report.py` from "
      f"`{a.results}/*.json`. No figure in this document was typed by hand.\n")
    w(f"Pre-registration: `docs/PREREGISTRATION.md`, SHA-256 `{prereg[0]}`, recorded "
      f"{prereg[-1] if len(prereg) > 2 else 'n/a'} before any test split was evaluated.\n")
    rc = load(Path(a.stage_a).parent / "run_config.json")
    machine = (rc or {}).get("machine", {}).get("platform") or platform.platform()
    w(f"Machine: {machine}, Apple M4 Max, 48 GB, 40-core GPU.\n")

    # ------------------------------------------------------------------ verdicts
    V: dict[str, tuple] = {}
    if exact:
        r0 = exact["runs"][0]
        V["H1"] = (r0["nodes_not_bitwise_equal"] == 0,
                   f"{f(r0['nodes_not_bitwise_equal'])} of {f(r0['nodes_checked'])} cached nodes differed from a full "
                   f"rebuild over {f(r0['events_checked'])} events (canonical mode)")
    sets = (eps or {}).get("sets", {})

    def cmp(set_name, key, metric):
        return sets.get(set_name, {}).get("comparisons", {}).get(key, {}).get(metric)

    main_e = "E-seed1"
    if eps:
        tok_ok, parts = True, []
        for s in ("STANDARD", "WIDE"):
            a1, a2 = cmp(s, f"{main_e} vs E-full", "tokens_per_changed_event"), cmp(s, f"{main_e}@B2 vs B2", "tokens_per_changed_event")
            for name, c in (("E-full", a1), ("B2", a2)):
                if c is None:
                    tok_ok = None if tok_ok is not False else False
                    continue
                parts.append(f"{s} tokens vs {name}: ratio {f(c['ratio'],3)} [{f(c['ratio_lo'],3)}, {f(c['ratio_hi'],3)}]")
                if c["ratio_hi"] > 0.5:
                    tok_ok = False
        time_ok = None
        if timing:
            time_ok = True
            for s in ("STANDARD", "WIDE"):
                for k in ("time_ratio_E_over_E-full", "time_ratio_E_over_B2"):
                    r = timing["sets"][s][k]
                    parts.append(f"{s} {k.replace('time_ratio_', 'time ').replace('_over_', ' / ')}: "
                                 f"{f(r['ratio'],3)} [{f(r['lo'],3)}, {f(r['hi'],3)}]")
                    if r["hi"] > 0.5:
                        time_ok = False
        ok = None if tok_ok is None or time_ok is None else bool(tok_ok and time_ok)
        V["H2"] = (ok, "; ".join(parts))
        for h, s in (("H3", "STANDARD"), ("H4", "WIDE")):
            st, ac = cmp(s, f"{main_e}@B2 vs B2", "stale_decision_rate"), cmp(s, f"{main_e}@B2 vs B2", "disposition_accuracy")
            if st is None or ac is None:
                V[h] = (None, "B2 was not evaluated")
                continue
            ok = bool(st["hi"] < 0 and ac["lo"] > -0.01)
            V[h] = (ok, f"stale-decision rate E {pct(st['a'],2)} vs B2 {pct(st['b'],2)}, difference {diff(st, as_pct=True)}; "
                        f"disposition accuracy E {pct(ac['a'],2)} vs B2 {pct(ac['b'],2)}, difference {diff(ac, as_pct=True)}")
        n5 = [cmp(s, f"{main_e} vs B1-soft", "status_nll") for s in ("STANDARD", "WIDE")]
        if all(n5):
            V["H5"] = (bool(all(c["hi"] < 0 for c in n5)),
                       "; ".join(f"{s}: status NLL E {f(c['a'])} vs B1-soft {f(c['b'])}, difference {diff(c)}"
                                 for s, c in zip(("STANDARD", "WIDE"), n5)))
        c6 = cmp("WIDE", f"N-seed1 vs {main_e}", "disposition_accuracy")
        if c6:
            held = c6["hi"] < 0
            V["H6"] = (bool(held), f"WIDE disposition accuracy N {pct(c6['a'],2)} vs E {pct(c6['b'],2)}, "
                                   f"difference (N - E) {diff(c6, as_pct=True)}")
    if evo:
        p = evo["streams"][evo["primary_stream"]]
        V["H7"] = (evo["H7_supported"],
                   f"stream {evo['primary_stream']}, second half: evolving {pct(p['evolving_vs_frozen']['accuracy_a'],2)} vs "
                   f"frozen {pct(p['evolving_vs_frozen']['accuracy_b'],2)}, difference {diff(p['evolving_vs_frozen'], as_pct=True)}; "
                   f"largest regression drop after an adoption {pct(p['H7b_max_regression_drop_evolving'],2)} (truthful), "
                   f"{pct(p['H7c_max_regression_drop_corrupted'],2)} (half the labels wrong)")
    if prop and prop.get("sets"):
        V["H8"] = (prop["H8_supported"],
                   "; ".join(f"{s}: missed links {pct(e['missed_link_rate']['mean'],2)} "
                             f"[{pct(e['missed_link_rate']['lo'],2)}, {pct(e['missed_link_rate']['hi'],2)}], work reduced "
                             f"x{f(e['work_reduction_factor'],2)}, accuracy change {100*e['disposition_accuracy_change']:+.2f} pts"
                             for s, e in prop["sets"].items()))
    if stale:
        t1, t2 = stale["by_type"]["T1"], stale["by_type"]["T2"]
        V["H9"] = (stale["prediction_held"],
                   f"detection at {pct(stale['false_positive_rate_on_controls'],1)} false positives: direct conflicts (T1) "
                   f"{pct(t1['detection_rate'],1)} [{pct(t1['detection_lo'],1)}, {pct(t1['detection_hi'],1)}], propagated (T2) "
                   f"{pct(t2['detection_rate'],1)} [{pct(t2['detection_lo'],1)}, {pct(t2['detection_hi'],1)}]")

    names = {"H1": "Incremental equals full rebuild", "H2": "Update cost at most half",
             "H3": "Revision fidelity vs dense reader (seen structure)",
             "H4": "Revision fidelity vs dense reader (unseen structure)", "H5": "Learned refinement beats calibrated rules",
             "H6": "Executing the policy generalises better than learning it", "H7": "Improves from feedback, safely",
             "H8": "Dependency proposals within the miss bound", "H9": "Propagated conflicts are harder (exploratory)"}
    w("## 1. Verdicts\n")
    w("Decision rules are those written down in advance. \"Not supported\" means the rule was not met; "
      "it is a result, and it is reported as prominently as the others.\n")
    w("| | Hypothesis | Verdict | Evidence |")
    w("|---|---|---|---|")
    for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"):
        ok, ev = V.get(h, (None, "experiment not run"))
        w(f"| {h} | {names[h]} | {verdict(ok)} | {ev} |")
    w("")
    arch = [V.get(h, (None,))[0] for h in ("H3", "H4")]
    if all(x is not None for x in arch):
        if not any(arch) and V.get("H2", (None,))[0] is False:
            w("**Architecture claim (pre-registration, section 6): refuted.** The dense reader decides at least as "
              "well and the cost advantage did not meet the threshold.\n")
        elif not any(arch):
            w("**Architecture claim (pre-registration, section 6): not supported on decision quality.** "
              "The dense reader decides at least as well inside the same wrapper. What remains is the cost advantage.\n")
        elif all(arch):
            w("**Architecture claim (pre-registration, section 6): supported on both test sets**, against a dense "
              "reader with the same backbone, pretraining, training episodes and wrapper.\n")
        else:
            w("**Architecture claim (pre-registration, section 6): supported on one test set and not the other.** "
              "See section 3.\n")

    # ------------------------------------------------------------------ training
    w("## 2. The edge model (Stage A)\n")
    if man:
        w(f"- Parameters: {f(man['parameters'])}. Backbone `{man['config']['encoder_name']}`, fp32, MPS.")
        w(f"- Trained on {f(man['examples'])} examples streamed from the Hugging Face Hub in "
          f"{man['train_seconds']/60:.1f} minutes ({man['examples']/man['train_seconds']:.1f} examples/s). "
          "Fixed budget, final weights, no early stopping.")
        w("- Examples by source: " + ", ".join(f"{k} {f(v)}" for k, v in sorted(man["examples_by_source"].items())) + ".")
        w("- Licences: " + "; ".join(f"{k}: {v}" for k, v in sorted(man["licences"].items())) + ".")
        w(f"- Stream reconnections during training: {man['stream_reopens']}. Passes over each corpus: {man['stream_epochs']}.")
        w("")
    if edge:
        w(f"Temperature {f(edge['temperature'])}, fitted on {', '.join(edge['temperature_fitted_on'])} only.\n")
        w("| Set | Role | n | Accuracy | Macro-F1 | NLL raw -> calibrated | ECE raw -> calibrated | Public NLI model (B3) accuracy |")
        w("|---|---|---:|---:|---:|---:|---:|---:|")
        for name, e in edge["sets"].items():
            other = b3["sets"][name]["raw"]["accuracy"] if b3 and name in b3["sets"] else None
            w(f"| {name} | {e['role']} | {f(e['n'])} | {pct(e['raw']['accuracy'],2)} | {f(e['raw']['macro_f1'],3)} | "
              f"{f(e['raw']['nll'],3)} -> {f(e['calibrated']['nll'],3)} | {f(e['raw']['ece'],3)} -> {f(e['calibrated']['ece'],3)} | "
              f"{pct(other,2)} |")
        w("")
        vt = edge["sets"].get("vitaminc.test", {})
        if "set_accuracy" in vt.get("raw", {}):
            w(f"**Revision sensitivity.** On VitaminC test, {pct(vt['raw']['set_accuracy'],2)} of {f(vt['raw']['n_sets'])} "
              "contrast sets were answered entirely correctly. A contrast set is the same claims judged against the "
              "evidence before and after a real edit; one wrong member fails the set.\n")
        for k in ("unrelated.test",):
            u = edge["sets"].get(k)
            if u:
                w(f"**Unrelated records.** {pct(u['false_decisive_rate'],2)} of {f(u['n'])} unrelated pairs were read as "
                  f"decisive. At that rate the chance that none of 10 unrelated records is misread is "
                  f"{pct(u['p_no_false_decisive_among_10'],1)}, and of 30, {pct(u['p_no_false_decisive_among_30'],1)}."
                  + (f" The public NLI model read {pct(b3['sets'][k]['false_decisive_rate'],2)} as decisive."
                     if b3 and k in b3["sets"] else "") + "\n")
        if numeric and numeric.get("sets"):
            w("**Numbers.** Accuracy by the form of the claim. " + numeric["note"] + "\n")
            w("| Set | Claim | n | Accuracy |")
            w("|---|---|---:|---:|")
            for sname, e in numeric["sets"].items():
                for k, v in e.items():
                    w(f"| {sname} | {k} | {f(v['n'])} | {pct(v['accuracy'],2)} [{pct(v['lo'],2)}, {pct(v['hi'],2)}] |")
            w("")
        w("ANLI is adversarial and was never trained on (its licence is non-commercial). SNLI was excluded from "
          "training because its labels assume both sentences describe one scene. Lower accuracy on both is "
          "expected and is shown rather than omitted.\n")

    # ------------------------------------------------------------------ episodes
    if eps:
        w("## 3. Evolving episodes\n")
        w("All systems share the ledger, schema, exact logic and planner. Intervals are 95% bootstrap intervals "
          "over whole episodes.\n")
        w("| Id | System |")
        w("|---|---|")
        w("| E-seedN | EASE-Delta: edge model, refiner trained through the exact policy (three training seeds) |")
        w("| E-full | the same model re-reading every active record against every predicate at each change |")
        w("| E-declared | E, but each record is compared only with the predicate it was written for. This uses "
          "information a deployment would have only if someone declared it, so it is an upper bound on what linking "
          "can save, not a system on equal terms |")
        w("| B1-soft | edge model, probabilities calibrated through the policy, no refiner |")
        w("| B1-hard | edge model, top label per edge, policy applied to labels |")
        w("| N-seedN | edge model, GRU that outputs the status itself; the policy is learned, not executed |")
        w("| B2 | dense reader: same backbone and initial weights, reads the claim against all active records at once |")
        w("| B3 | public NLI model `tasksource/ModernBERT-base-nli` in place of the edge model, calibrated through "
          "the policy on the same validation episodes as B1-soft |")
        w("| B3-uncalibrated | the same without calibration |")
        w("")
        for s, e in sets.items():
            d = e["episodes"]
            w(f"### {s}\n")
            w(f"{f(d['episodes'])} episodes, {f(d['events'])} scored events, {d['mean_predicates']:.1f} predicates and "
              f"{d['mean_records']:.1f} records per task on average. {f(d['events_changing_some_action'])} events changed "
              f"some action's correct disposition. Ledger outcomes: {d['ledger_outcomes']}.\n")
            w("| System | Disposition accuracy | Stale-decision rate | Spurious-change rate | False-READY rate | "
              "Status accuracy | Status NLL | P(success) Brier | Tokens per changed event |")
            w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            order = [k for k in e["systems"] if "@" not in k] + [k for k in e["systems"] if "@" in k]
            for k in order:
                m = e["systems"][k]
                if "disposition_accuracy" not in m:
                    w(f"| {k} | same as E (H1) | | | | | | | {ci(m.get('tokens_per_changed_event'), 0)} |")
                    continue
                w(f"| {k} | {ci(m['disposition_accuracy'], as_pct=True)} | {ci(m['stale_decision_rate'], as_pct=True)} | "
                  f"{ci(m['spurious_change_rate'], as_pct=True)} | {ci(m['false_ready_rate'], as_pct=True)} | "
                  f"{ci(m['status_accuracy'], as_pct=True)} | {ci(m['status_nll'],3)} | {ci(m['success_brier'],4)} | "
                  f"{ci(m.get('tokens_per_changed_event'), 0)} |")
            w("")
            if "dense_episodes" in e:
                w(f"Rows ending in `@B2` and the row `B2` use the first {e['dense_episodes']} episodes of this set, "
                  "because the dense reader is expensive to evaluate (see Deviations). Comparisons with B2 are paired on "
                  "those episodes.\n")
            if "dense_context_tokens" in e:
                t = e["dense_context_tokens"]
                w(f"Dense reader context length: mean {t['mean']:.0f} tokens, median {t['p50']:.0f}, 95th percentile "
                  f"{t['p95']:.0f}, maximum {t['max']}; {pct(t['truncated_fraction'],2)} truncated.\n")
            w("Paired comparisons (first minus second):\n")
            w("| Comparison | Disposition accuracy | Stale-decision rate | False-READY rate | Status NLL | Tokens ratio |")
            w("|---|---:|---:|---:|---:|---:|")
            for k, c in e["comparisons"].items():
                tr = c.get("tokens_per_changed_event", {})
                ratio = f"{f(tr['ratio'],3)} [{f(tr['ratio_lo'],3)}, {f(tr['ratio_hi'],3)}]" if "ratio" in tr else "n/a"
                w(f"| {k} | {diff(c.get('disposition_accuracy'), as_pct=True)} | {diff(c.get('stale_decision_rate'), as_pct=True)} | "
                  f"{diff(c.get('false_ready_rate'), as_pct=True)} | {diff(c.get('status_nll'))} | {ratio} |")
            w("")
            bt = e.get("by_event_type", {})
            if main_e in bt:
                cols = [c for c in (main_e, "B1-soft", "N-seed1", "B2", "B3") if c in bt]
                w("Disposition accuracy by the kind of event that preceded it:\n")
                w("| Event | n | " + " | ".join(cols) + " |")
                w("|---|---:|" + "---:|" * len(cols))
                for ev in sorted(bt[main_e]):
                    w(f"| {ev} | {f(bt[main_e][ev]['n'])} | " + " | ".join(
                        pct(bt[c][ev]["accuracy"], 1) + (f" (n={bt[c][ev]['n']})" if bt[c][ev]["n"] != bt[main_e][ev]["n"] else "")
                        if ev in bt[c] else "n/a" for c in cols) + " |")
                w("")
            sw = e.get("threshold_sweep")
            if sw:
                w("Confidence required before an action is called READY (exploratory; the pre-registered metrics "
                  "above use 0.5):\n")
                w("| Threshold | False-READY rate | READY missed | Disposition accuracy |")
                w("|---:|---:|---:|---:|")
                for thr, m in sw.items():
                    w(f"| {thr} | {ci(m['false_ready_rate'], as_pct=True)} | {ci(m['missed_ready_rate'], as_pct=True)} | "
                      f"{ci(m['disposition_accuracy'], as_pct=True)} |")
                w("")
            we = e.get("worst_episodes", {}).get(main_e)
            if we:
                w("Worst episodes for E: " + ", ".join(
                    f"#{x['episode_index']} ({pct(x['disposition_accuracy'],0)} of {x['rows']} rows)" for x in we) + ".\n")
        if dman:
            w(f"**The dense reader's training.** Initialised from the Stage A weights ({dman['init']}). "
              f"{f(dman['examples'])} contexts, {f(dman['tokens'])} tokens, {dman['train_seconds']/60:.1f} minutes. "
              f"Validation accuracy {pct(dman['val']['accuracy'],2)} on {f(dman['val']['n'])} held-out contexts.\n")

    # ------------------------------------------------------------------ exactness / timing
    if exact:
        w("## 4. Exactness (H1)\n")
        w("| Mode | Episodes | Events | Cached nodes checked | Not bit-identical | Largest difference | Proposals that differ |")
        w("|---|---:|---:|---:|---:|---:|---:|")
        for r in exact["runs"]:
            w(f"| {r['mode']} | {r['episodes']} | {f(r['events_checked'])} | {f(r['nodes_checked'])} | "
              f"{f(r['nodes_not_bitwise_equal'])} | {r['max_abs_difference']:.3e} | {r['dispositions_that_differ_from_rebuild']} |")
        w("")
        w("The rebuild bypasses the cache, so every edge is recomputed by the model. Canonical mode fixes the shape "
          "of every forward pass; dynamic mode batches freely and is faster.\n")
        if determinism:
            w("Separate probe of 200 pairs regrouped into different batches (`runs/determinism_probe.json`): "
              + "; ".join(f"{k}: {v['different_batch_composition']['pairs_not_bitwise_equal']} differ, max "
                          f"{v['different_batch_composition']['max_abs_diff']:.2e}" for k, v in determinism.items()) + ".\n")
    if timing:
        w("## 5. Wall-clock cost (H2)\n")
        w(f"Device {timing['device']}. {timing['note']}\n")
        for label, t in (("dynamic batching", timing), ("canonical shapes", timing_c)):
            if not t:
                continue
            w(f"**{label}**\n")
            w("| Set | Changed events | E p50 / p95 ms | E-full p50 / p95 ms | B2 p50 / p95 ms | E / E-full | E / B2 | "
              "Duplicate or stale p50 ms |")
            w("|---|---:|---:|---:|---:|---:|---:|---:|")
            for s, e in t["sets"].items():
                r1, r2 = e["time_ratio_E_over_E-full"], e["time_ratio_E_over_B2"]
                u = e["unchanged_events(duplicate or stale)"]
                w(f"| {s} | {e['changed_events']} | {e['E']['p50_ms']:.1f} / {e['E']['p95_ms']:.1f} | "
                  f"{e['E-full']['p50_ms']:.1f} / {e['E-full']['p95_ms']:.1f} | {e['B2']['p50_ms']:.1f} / {e['B2']['p95_ms']:.1f} | "
                  f"{f(r1['ratio'],3)} [{f(r1['lo'],3)}, {f(r1['hi'],3)}] | {f(r2['ratio'],3)} [{f(r2['lo'],3)}, {f(r2['hi'],3)}] | "
                  f"{u.get('p50_ms', float('nan')):.2f} |")
            w("")
        b = timing.get("broad_change")
        if b:
            w(f"**Where the advantage disappears.** Replacing the model weights in a task with {b['edges_in_task']} edges "
              f"recomputed {b['edges_recomputed_on_weights_change']} of them and took {b['weights_change_ms']:.0f} ms, "
              f"against {b['full_rebuild_ms']:.0f} ms for a full rebuild (ratio {f(b['ratio'],2)}). A change that touches "
              "everything costs as much as starting again.\n")

    # ------------------------------------------------------------------ proposer / stale / evolution
    if prop:
        w("## 6. Dependency proposals (H8)\n")
        c = prop["calibration"]
        if c.get("feasible"):
            w(f"Threshold {f(c['threshold'],3)} on cosine similarity (`{c['embedder']}`), chosen by conformal risk control at "
              f"alpha = {c['alpha']} from {c['n']} calibration tasks (corrected risk {f(c['corrected_risk'])}).\n")
            w("| Set | Missed links (mean over tasks) | Tasks with any miss | Pairs compared | Work reduction | "
              "Disposition accuracy with / without | Change |")
            w("|---|---:|---:|---:|---:|---:|---:|")
            for s, e in prop["sets"].items():
                m = e["missed_link_rate"]
                w(f"| {s} | {pct(m['mean'],2)} [{pct(m['lo'],2)}, {pct(m['hi'],2)}] | {pct(m['tasks_with_any_miss'],1)} | "
                  f"{pct(e['fraction_of_pairs_compared'],1)} | x{f(e['work_reduction_factor'],2)} | "
                  f"{pct(e['E+proposer']['disposition_accuracy']['value'],2)} / {pct(e['E']['disposition_accuracy']['value'],2)} | "
                  f"{diff(e['comparison']['disposition_accuracy'], as_pct=True)} |")
            w("")
            w(f"Pre-registered consequence: keep the proposer off by default = **{f(prop['keep_off_by_default'])}**.\n")
        else:
            w(f"No threshold met the bound: {c.get('note')}\n")
    if stale:
        w("## 7. Implicit conflicts from STALE (H9, exploratory)\n")
        w(f"{stale['pairs']['conflict']} conflict pairs and {stale['pairs']['control']} control pairs. "
          f"Threshold on P(REFUTES) = {f(stale['threshold_on_p_refutes'])}, set for "
          f"{pct(stale['false_positive_rate_on_controls'],1)} false positives on controls. "
          f"**Not comparable with {stale['not_comparable_with']}.**\n")
        w("| Type | n | Detected | AUROC vs controls | Read as REFUTES | Read as NEI | Read as SUPPORTS |")
        w("|---|---:|---:|---:|---:|---:|---:|")
        for t, e in stale["by_type"].items():
            w(f"| {t} | {e['n']} | {pct(e['detection_rate'],1)} [{pct(e['detection_lo'],1)}, {pct(e['detection_hi'],1)}] | "
              f"{f(e['auroc_vs_controls'],3)} | {pct(e['top_label']['REFUTES'],1)} | {pct(e['top_label']['NEI'],1)} | "
              f"{pct(e['top_label']['SUPPORTS'],1)} |")
        w("")
        p = stale.get("proposer", {})
        if "links_missed" in p:
            w(f"At the proposer's threshold ({f(p['threshold'],3)}) the old and new statements would not even have been "
              f"compared in {pct(p['links_missed']['T1'],1)} of direct conflicts and {pct(p['links_missed']['T2'],1)} of "
              f"propagated ones. Similarity separates conflict pairs from controls with AUROC "
              f"{f(p['auroc_conflict_vs_control']['T1'],3)} (T1) and {f(p['auroc_conflict_vs_control']['T2'],3)} (T2).\n")
        if stale.get("examples_lowest_and_highest"):
            w("Least and most confidently detected:\n")
            for x in stale["examples_lowest_and_highest"]:
                w(f"- {x['type']}, P(REFUTES) {f(x['p_refutes'],3)}, {'detected' if x['detected'] else 'missed'}: "
                  f"\"{x['old']}\" / \"{x['new']}\"")
            w("")
    if evo:
        w("## 8. Learning after deployment (H7)\n")
        w(f"Verified labels were returned for {pct(evo['feedback_rate'],0)} of items. The encoder was never updated.\n")
        for s, e in evo["streams"].items():
            w(f"### Stream `{s}` ({f(e['items'])} items, {f(e['verified'])} verified)\n")
            w("| Run | Accuracy, whole stream | First half | Second half | Consolidations attempted | Adopted |")
            w("|---|---:|---:|---:|---:|---:|")
            h = e["evolving_vs_frozen"]
            hc = e["corrupted_vs_frozen"]
            rows = (("frozen", h["first_half_accuracy_b"], h["accuracy_b"]), ("evolving", h["first_half_accuracy_a"], h["accuracy_a"]),
                    ("corrupted", hc["first_half_accuracy_a"], hc["accuracy_a"]))
            for k, a1, a2 in rows:
                w(f"| {k} | {pct(e[k]['accuracy'],2)} | {pct(a1,2)} | {pct(a2,2)} | {len(e[k]['consolidations'])} | {e[k]['adopted']} |")
            w("")
            w(f"Second half, evolving minus frozen: {diff(h, as_pct=True)}. "
              f"With half of the returned labels wrong, minus frozen: {diff(hc, as_pct=True)}.\n")
            w(f"Largest fall in regression accuracy after any adoption: {pct(e['H7b_max_regression_drop_evolving'],2)} "
              f"(truthful feedback), {pct(e['H7c_max_regression_drop_corrupted'],2)} (corrupted feedback). Tolerance 0.50%.\n")
            for k in ("evolving", "corrupted"):
                if e[k]["consolidations"]:
                    w(f"Consolidations, {k}:\n")
                    w("| After items | Verified | Adopted | What | Memory (tau, lam) | Held-out gain [interval] | Regression change |")
                    w("|---:|---:|---|---|---|---:|---|")
                    for c in e[k]["consolidations"]:
                        g = c.get("holdout_gain") or {}
                        w(f"| {c['after_items']} | {c['feedback']} | {f(c['adopted'])} | {c['chosen']} | "
                          f"{c['memory']['tau']}, {c['memory']['lam_max']} | "
                          + (f"{g['gain']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]" if g else "n/a") + " | "
                          + ", ".join(f"{n} {100*v:+.2f}" for n, v in c["regression_accuracy_change_vs_trained"].items()) + " |")
                    w("")

    if joins:
        w("## 8b. Requirements that need two records read together (X1, exploratory)\n")
        w(f"Not pre-registered. {joins['n']} cases; wording is {joins['wording']}. Example: claim \"The client has "
          "approved the latest version of the logo\", records \"The client approved version 4 of the logo\" and "
          "\"The latest version of the logo is version 5\". Calling the claim satisfied when the versions differ "
          "is the unsafe outcome.\n")
        w("| Reading | Correct | Unsafe (says satisfied when it is not) | Versions differ: satisfied / violated / open | "
          "Versions match: satisfied / violated / open | Only one record: open |")
        w("|---|---:|---:|---:|---:|---:|")
        for k, t in joins["systems"].items():
            one = (t["only_approval"]["UNRESOLVED"] + t["only_latest"]["UNRESOLVED"]) / 2
            w(f"| {k} | {pct(t['overall_correct'],1)} | {pct(t['unsafe_rate'],1)} | "
              f"{pct(t['differ']['SATISFIED'],0)} / {pct(t['differ']['VIOLATED'],0)} / {pct(t['differ']['UNRESOLVED'],0)} | "
              f"{pct(t['match']['SATISFIED'],0)} / {pct(t['match']['VIOLATED'],0)} / {pct(t['match']['UNRESOLVED'],0)} | "
              f"{pct(one,0)} |")
        w("")

    if small:
        w("## 8c. Small tasks (X2, exploratory)\n")
        d = small["episodes"]
        w(f"Not pre-registered; designed after the demo below, before any episode-level result was known. "
          f"{f(d['episodes'])} episodes from the test pool with {d['mean_predicates']:.1f} predicates and "
          f"{d['mean_records']:.1f} records per task on average.\n")
        w("| System | Disposition accuracy | False-READY rate | Status accuracy | Status NLL |")
        w("|---|---:|---:|---:|---:|")
        for k, m in small["systems"].items():
            w(f"| {k} | {ci(m['disposition_accuracy'], as_pct=True)} | {ci(m['false_ready_rate'], as_pct=True)} | "
              f"{ci(m['status_accuracy'], as_pct=True)} | {ci(m['status_nll'],3)} |")
        w("")
        w("| Comparison | Status NLL | Disposition accuracy |")
        w("|---|---:|---:|")
        for k, c in small["comparisons"].items():
            w(f"| {k} | {diff(c['status_nll'])} | {diff(c['disposition_accuracy'], as_pct=True)} |")
        w("")
    if set_size:
        w("## 8d. Readings per predicate (X3, exploratory)\n")
        w("Not pre-registered; designed after the STANDARD results showed E-declared below E. \"Declared links\" "
          "compares each record only with the predicate it was written for, so each predicate has one to three "
          "readings instead of about eleven.\n")
        w("| Set | System | Disposition accuracy | Status accuracy | Status NLL |")
        w("|---|---|---:|---:|---:|")
        for sname, e in set_size["sets"].items():
            for k, m in e["systems"].items():
                w(f"| {sname} | {k} | {ci(m['disposition_accuracy'], as_pct=True)} | {ci(m['status_accuracy'], as_pct=True)} | "
                  f"{ci(m['status_nll'],3)} |")
        w("")
        w("| Set | Comparison | Disposition accuracy | Status NLL |")
        w("|---|---|---:|---:|")
        for sname, e in set_size["sets"].items():
            for k, c in e["comparisons"].items():
                w(f"| {sname} | {k} | {diff(c['disposition_accuracy'], as_pct=True)} | {diff(c['status_nll'])} |")
        w("")
        w("Status accuracy by the number of readings a predicate has (first 300 episodes of each set):\n")
        w("| Set | Readings | n | E | Calibrated rules |")
        w("|---|---|---:|---:|---:|")
        for sname, e in set_size["sets"].items():
            by = e["status_accuracy_by_readings_per_predicate"]
            for bucket in sorted(by["E"], key=lambda b: int(b.split("-")[0].rstrip("+"))):
                w(f"| {sname} | {bucket} | {f(by['E'][bucket]['n'])} | {pct(by['E'][bucket]['accuracy'],1)} | "
                  f"{pct(by['rules'][bucket]['accuracy'],1)} |")
        w("")
    if linking_rules:
        w("## 8e. The dependency proposer with calibrated rules (X3b, exploratory)\n")
        w(f"Not pre-registered. Same proposer and threshold as H8 ({f(linking_rules['threshold'],3)}); nothing re-tuned.\n")
        w("| Set | System | Disposition accuracy | False-READY rate | Tokens per changed event |")
        w("|---|---|---:|---:|---:|")
        for sname, e in linking_rules["sets"].items():
            for k, m in e["systems"].items():
                w(f"| {sname} | {k} | {ci(m['disposition_accuracy'], as_pct=True)} | {ci(m['false_ready_rate'], as_pct=True)} | "
                  f"{ci(m['tokens_per_changed_event'], 0)} |")
        w("")
        w("| Set | Comparison | Disposition accuracy | Tokens ratio |")
        w("|---|---|---:|---:|")
        for sname, e in linking_rules["sets"].items():
            for k, c in e["comparisons"].items():
                t = c.get("tokens_per_changed_event", {})
                w(f"| {sname} | {k} | {diff(c['disposition_accuracy'], as_pct=True)} | "
                  + (f"{f(t['ratio'],3)} [{f(t['ratio_lo'],3)}, {f(t['ratio_hi'],3)}]" if "ratio" in t else "n/a") + " |")
        w("")
    if suites_check:
        w("## 8f. Would set suites have caught the failing refiner? (non-test data)\n")
        w("Set suites are built from Stage B-pool episodes only (`scripts/build_set_suites.py`). Accuracy and NLL of "
          "each predicate's status:\n")
        names_ = list(next(iter(suites_check.values())).keys())
        w("| Aggregator | " + " | ".join(names_) + " |")
        w("|---|" + "---:|" * len(names_))
        for k, row in suites_check.items():
            w(f"| {k} | " + " | ".join(f"{pct(row[n]['accuracy'],1)} (NLL {f(row[n]['nll'],3)})" for n in names_) + " |")
        w("")
    if evo_rules:
        w("## 8g. Self-evolution from calibrated rules, with set suites (X4, exploratory)\n")
        w("Not pre-registered. The H7 procedure repeated with the changes described in Deviations, item 9: "
          "calibrated rules as the starting point, set suites in the gate, context frozen during fine-tuning.\n")
        w("| Stream | Items | Verified | Frozen accuracy | Evolving, second half | Corrupted, second half | "
          "Consolidations adopted (evolving / corrupted) |")
        w("|---|---:|---:|---:|---:|---:|---:|")
        for sname, e in evo_rules["streams"].items():
            h, hc = e["evolving_vs_frozen"], e["corrupted_vs_frozen"]
            w(f"| {sname} | {f(e['items'])} | {f(e['verified'])} | {pct(e['frozen']['accuracy'],2)} | "
              f"{pct(h['accuracy_a'],2)} ({100*h['difference']:+.2f} pts) | {pct(hc['accuracy_a'],2)} ({100*hc['difference']:+.2f} pts) | "
              f"{e['evolving']['adopted']}/{len(e['evolving']['consolidations'])} / {e['corrupted']['adopted']}/{len(e['corrupted']['consolidations'])} |")
        w("")
        from collections import Counter
        why = Counter()
        for e in evo_rules["streams"].values():
            for c in e["evolving"]["consolidations"]:
                for part in c["reason"].split("; "):
                    if "fails regression on" in part:
                        for n in part.split("[", 1)[1].rstrip("]").replace("'", "").split(", "):
                            why[n] += 1
                    elif "no reliable held-out gain" in part:
                        why["no reliable held-out gain"] += 1
        if why:
            w("Why candidates were rejected (counts over all attempts and candidates): "
              + ", ".join(f"{k} {v}" for k, v in why.most_common()) + ".\n")
    if demo_trace:
        w("## 8h. The hand-off demo, scored (illustration)\n")
        w(demo_trace["note"] + " The scenario's twelve messages are email-like text of a kind absent from "
          "training. Correct statuses were written from the messages' wording before any model was trained.\n")
        w("| System | READY threshold | Predicate statuses correct | Action dispositions correct | False READY |")
        w("|---|---:|---:|---:|---:|")
        for r in demo_trace["runs"]:
            w(f"| {r['system']} | {r['threshold']} | {r['predicate_status_correct']} | "
              f"{r['action_disposition_correct']} | {r['false_ready']} |")
        w("")
        errs = []
        for r in demo_trace["runs"]:
            if r["threshold"] != 0.5:
                continue
            for i, e in enumerate(r["events"], 1):
                for pid, v in e["predicates"].items():
                    if v["model"] != v["truth"]:
                        errs.append(f"- {r['system']}, event {i} ({e['event']}): `{pid}` read as {v['model']}, "
                                    f"should be {v['truth']} (satisfied {v['p']['SATISFIED']:.2f}, violated "
                                    f"{v['p']['VIOLATED']:.2f}, open {v['p']['UNRESOLVED']:.2f})")
        if errs:
            w("Misreadings at threshold 0.5:\n")
            w("\n".join(errs) + "\n")

    rel = load(Path("release") / "MANIFEST.json")
    if rel and "chosen_because" in rel.get("aggregator", {}):
        w("## 8i. What the release ships, and why\n")
        why_ = rel['aggregator']['chosen_because']
        if why_.startswith("fixed by the plan"):
            w(f"Aggregator: **{rel['aggregator']['kind']}**, on the large edge model: {why_}.\n")
        else:
            w(f"Aggregator: **{rel['aggregator']['kind']}**. Rule (Deviations, item 7, fixed before episode-level results): "
              f"{why_}.\n")

    headroom, more, cpu = load(R / "headroom.json"), load(R / "more_training.json"), load(R / "cpu_timing.json")
    if headroom:
        w("## 8j. Where more accuracy would come from (X5, exploratory)\n")
        w("Not pre-registered; run after every result above was known, to decide what to do next. The released "
          "model is unchanged by any of it.\n")
        w("**Replay with reading errors corrected.** The test episodes above, run again with the shipped rules after "
          "replacing some of the edge model's readings by the label the generator recorded. A reading is *wrong* when "
          "its most probable label differs from that label. The *linked* record is the one written for the "
          "requirement; every other record in the task is *unrelated* to it.\n")
        for name, e in headroom["headroom"].items():
            ra, er = e["reading_accuracy"], e["errors"]
            w(f"*{name}*: {f(e['pairs'])} readings, {pct(ra['all'],2)} correct ({pct(ra['linked'],2)} on the "
              f"{f(e['linked_pairs'])} linked, {pct(ra['unrelated'],2)} on the unrelated); {f(er['on_linked_pairs'])} "
              f"errors on linked records, {f(er['on_unrelated_pairs'])} on unrelated ones.\n")
            w("| Readings replaced | Number | Disposition accuracy | Stale decisions | False READY | Missed READY |")
            w("|---|---:|---:|---:|---:|---:|")
            order = ["as measured", "25% of reading errors corrected", "50% of reading errors corrected",
                     "75% of reading errors corrected", "errors on unrelated records corrected",
                     "errors on the linked record corrected", "all reading errors corrected",
                     "every reading correct and confident"]
            for lab in order:
                c = e["conditions"].get(lab)
                if c:
                    w(f"| {lab} | {f(c['readings_corrected'])} | {ci(c['disposition_accuracy'], as_pct=True)} | "
                      f"{pct(c['stale_decision_rate']['value'],2)} | {pct(c['false_ready_rate']['value'],2)} | "
                      f"{pct(c['missed_ready_rate']['value'],2)} |")
            w("")
        w("With every reading correct and confident the exact parts reproduce the generator's answer for every action "
          "at every step, so on these tasks all error comes from reading. The gap between the last two rows is "
          "readings that were right but hesitant. The partial rows correct a random share, chosen once (seed 0).\n")
        if "fit" in headroom:
            w("**Fit.** Accuracy on a systematic sample of each training split against held-out data. The sample "
              "includes rows the model never drew during training.\n")
            w("| Corpus | Training split (sample) | n | Held-out | n | Gap |")
            w("|---|---:|---:|---:|---:|---:|")
            for k, v in headroom["fit"].items():
                w(f"| {k} | {pct(v['train_sample']['accuracy'],2)} | {f(v['train_sample']['n'])} | "
                  f"{pct(v['held_out']['accuracy'],2)} (`{v['held_out']['set']}`) | {f(v['held_out']['n'])} | "
                  f"{100*v['accuracy_gap']:+.2f} pts |")
            w("")
        nei = headroom["errors"].get("vitaminc.test")
        if nei:
            w(f"**Evidence on the same topic that settles nothing.** On `vitaminc.test`, of the passages labelled as "
              f"not settling their claim, {pct(1 - nei['recall']['NEI'],2)} were read as supporting or refuting it. "
              f"These passages are about the claim's subject. For passages on another subject the figure is "
              f"{pct(1 - headroom['errors']['unrelated.test']['recall']['NEI'],2)} (`unrelated.test`).\n")
    if more:
        w("**More training of the same model on the same corpora.** The released edge model (A, "
          f"{f(more['a']['examples'])} examples) was trained for {f(more['b']['additional_examples'])} more examples "
          "on the same mixture (B; `configs/diag_more_training.yaml`) and both were run on the development sets. "
          "No test split was read, and B is not released.\n")
        w("| Development set | n | A | B | B − A | Only A right / only B right | p (McNemar, exact) | Log loss A → B |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k, s in more["sets"].items():
            d = s["accuracy_difference"]
            w(f"| {k} | {f(s['n'])} | {pct(s['accuracy_a'],2)} | {pct(s['accuracy_b'],2)} | "
              f"{100*d['value']:+.2f} pts [{100*d['lo']:+.2f}, {100*d['hi']:+.2f}] | {s['only_a_right']} / {s['only_b_right']} | "
              f"{s['mcnemar_p']:.3f} | {f(s['nll_a'],3)} → {f(s['nll_b'],3)} |")
        w("")
    if cpu:
        w("## 8k. Running without a GPU (exploratory)\n")
        w(cpu["note"] + " The shipped configuration on the first test episodes of each kind.\n")
        w("| Device | Threads | STANDARD update, median (95th) | WIDE update, median (95th) | "
          "First full reading of a task, median (STANDARD / WIDE) | Peak memory |")
        w("|---|---:|---:|---:|---:|---:|")
        for lab, e in cpu["runs"].items():
            s, wd = e["sets"]["STANDARD"], e["sets"]["WIDE"]
            w(f"| {lab} | {e['threads']} | {s['update']['p50_ms']:.0f} ms ({s['update']['p95_ms']:.0f} ms) | "
              f"{wd['update']['p50_ms']:.0f} ms ({wd['update']['p95_ms']:.0f} ms) | "
              f"{s['reading_a_new_task_in_full']['p50_ms']/1e3:.1f} s / {wd['reading_a_new_task_in_full']['p50_ms']/1e3:.1f} s | "
              f"{e['peak_memory_gb']:.1f} GB |")
        w("")

    large = load(R / "large.json")
    if large:
        w("## 8l. A larger encoder (pre-registered addendum)\n")
        pl = Path("runs/prereg_large_hash.txt").read_text().split() if Path("runs/prereg_large_hash.txt").exists() else ["?"]
        w(f"Plan: `docs/PREREGISTRATION_LARGE.md`, SHA-256 `{pl[0]}`, recorded {pl[-1] if len(pl) > 2 else 'n/a'} "
          "before the large model was trained. `answerdotai/ModernBERT-large` by the released recipe, calibrated "
          "rules on top, against the released base model; everything paired.\n")
        vd = large.get("verdicts", {})
        for h, text in (("H10", "reading: the large model is more accurate on `vitaminc.test` and `mnli_mm.test`"),
                        ("H11", "decisions: higher disposition accuracy on STANDARD"),
                        ("H12", "cost: median update on STANDARD at most 3.5× the base model's")):
            if h in vd:
                w(f"- **{h}** ({text}): {verdict(vd[h])}")
        sh = large.get("ships", {})
        if sh.get("large") is not None:
            w(f"- **Which model ships** by the rule in the plan: {'the large model' if sh['large'] else 'the base model stays'}"
              + (f" ({'; '.join(sh['reasons_against'])})" if sh.get("reasons_against") else "") + ".")
        w("")
        if large.get("items"):
            w("| Edge test set | n | Base | Large | Large − Base | Only base right / only large right | p (McNemar) |")
            w("|---|---:|---:|---:|---:|---:|---:|")
            for k, r in large["items"].items():
                w(f"| {k} | {f(r['n'])} | {pct(r['accuracy_base'],2)} | {pct(r['accuracy_large'],2)} | "
                  f"{100*r['difference']:+.2f} pts [{100*r['lo']:+.2f}, {100*r['hi']:+.2f}] | {r['only_base_right']} / {r['only_large_right']} | {r['mcnemar_p']:.3g} |")
            w("")
            u = large["items"].get("unrelated.test")
            if u and "false_decisive_base" in u:
                w(f"Unrelated passages read as decisive: base {pct(u['false_decisive_base'],2)}, large {pct(u['false_decisive_large'],2)}.\n")
            rs = large.get("edge_summaries", {}).get("real_synthetic")
            if rs and rs["base"].get("accuracy_real") is not None:
                w(f"VitaminC real revisions: base {pct(rs['base']['accuracy_real'],2)}, large {pct(rs['large']['accuracy_real'],2)}; "
                  f"synthetic: base {pct(rs['base']['accuracy_synthetic'],2)}, large {pct(rs['large']['accuracy_synthetic'],2)}.\n")
        if large.get("episodes"):
            w("| Test set | Episodes | Disposition accuracy, base | large | Large − Base | Stale decisions, base | large | False READY, base | large |")
            w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for k, e in large["episodes"].items():
                b, l, c = e["base"], e["large"], e["large_minus_base"]
                w(f"| {k} | {f(e['episodes'])} | {pct(b['disposition_accuracy']['value'],2)} | {pct(l['disposition_accuracy']['value'],2)} | "
                  f"{100*c['disposition_accuracy']['difference']:+.2f} pts [{100*c['disposition_accuracy']['lo']:+.2f}, {100*c['disposition_accuracy']['hi']:+.2f}] | "
                  f"{pct(b['stale_decision_rate']['value'],2)} | {pct(l['stale_decision_rate']['value'],2)} | "
                  f"{pct(b['false_ready_rate']['value'],2)} | {pct(l['false_ready_rate']['value'],2)} |")
            w("")
        tm = large.get("timing")
        if tm and tm.get("runs"):
            w("| Model | Device | STANDARD update, median (95th) | WIDE update, median (95th) | Peak memory |")
            w("|---|---|---:|---:|---:|")
            for lab, e in tm["runs"].items():
                s_, wd = e["sets"]["STANDARD"], e["sets"]["WIDE"]
                w(f"| {lab} | {e['device']} | {s_['update']['p50_ms']:.0f} ms ({s_['update']['p95_ms']:.0f} ms) | "
                  f"{wd['update']['p50_ms']:.0f} ms ({wd['update']['p95_ms']:.0f} ms) | {e['peak_memory_gb']:.1f} GB |")
            if "timing_ratio_standard_p50" in large:
                w(f"\nMedian update time, large / base, on STANDARD: {large['timing_ratio_standard_p50']:.2f}.")
            w("")

    # ------------------------------------------------------------------ supporting
    w("## 9. Supporting checks\n")
    if probe:
        for e in probe["experiments"]:
            w(f"- Mechanism probe, {e['topology']}: {e['mean_nodes_recomputed_when_numeric_input_changes']:.2f} of "
              f"{e['computed_nodes_in_full_forward_pass']} nodes recomputed per change "
              f"({pct(e['mean_fraction_recomputed_when_numeric_input_changes'],2)}); "
              f"{e['nodes_not_bitwise_equal_to_full_recompute']} nodes differed from a full rebuild over {e['events']} events.")
    if audit:
        for k, v in audit["pools"].items():
            w(f"- Leakage audit, {k} pool: {v['pairs_in_train']} of {f(v['pairs'])} pairs occur in a training split; "
              f"{v['vitaminc_atoms_whose_page_is_in_train']} of {f(v['vitaminc_atoms'])} VitaminC atoms share a page with training data.")
    w("")
    dev = Path(a.deviations)
    w("## 10. Deviations from the pre-registered plan\n")
    w(dev.read_text().strip() + "\n" if dev.exists() else "None recorded.\n")
    lim = Path("docs/LIMITATIONS.md")
    if lim.exists():
        w("## 11. Limitations\n")
        w(lim.read_text().strip() + "\n")
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"written {a.out} ({len(L)} lines)")
    for h, (ok, ev) in V.items():
        print(h, "supported" if ok else "NOT supported" if ok is False else "not evaluated", "|", ev[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
