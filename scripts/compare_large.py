"""The larger encoder against the released one, paired, by the rules of docs/PREREGISTRATION_LARGE.md.

Reads what the two pipelines wrote (edge logits per test set; edge tables for the test episodes;
each model's calibrated rules) and replays the episodes through the same exact core, so the
comparison is paired by item and by episode. Writes runs/results/large.json, from which
scripts/make_report.py writes section 12. Nothing here evaluates a model; it compares outputs
already produced.

    python scripts/compare_large.py --base runs/results --large runs/results_large \
        --base-stage-b runs/stage_b --large-stage-b runs/stage_b_large --timing runs/results/timing_large.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from ease.aggregate import Calibration, RuleAggregator
from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, WIDE, GenConfig, generate
from ease.eval.harness import compare, run_engine, summarise
from ease.eval.trace import EdgeTable
from ease.labels import NEI
from ease.util import atomic_write_json, read_json


# the same generator settings and seed as scripts/eval_small.py
SMALL = GenConfig(n_pred=(2, 4), n_distractors=(0, 1), n_events=(4, 8), n_actions=(1, 2), max_children=3)


def rules_for(stage_b: Path) -> RuleAggregator:
    cal = read_json(stage_b / "rule_calibration.json")
    return RuleAggregator(Calibration(cal["temperature"], tuple(cal["bias"])))


def paired_items(base_dir: Path, large_dir: Path, name: str, n_boot: int, rng) -> dict:
    a, b = np.load(base_dir / "edge_logits" / f"{name}.npz"), np.load(large_dir / "edge_logits" / f"{name}.npz")
    assert np.array_equal(a["labels"], b["labels"]), f"{name}: the two runs scored different items"
    y = a["labels"]
    ca, cb = (a["logits"].argmax(1) == y).astype(float), (b["logits"].argmax(1) == y).astype(float)
    d = cb - ca
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(1)
    only_a, only_b = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
    out = {"n": int(len(y)), "accuracy_base": float(ca.mean()), "accuracy_large": float(cb.mean()),
           "difference": float(d.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
           "only_base_right": only_a, "only_large_right": only_b,
           "mcnemar_p": float(binomtest(only_b, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0}
    if name.startswith("unrelated"):
        out["false_decisive_base"] = float((a["logits"].argmax(1) != NEI).mean())
        out["false_decisive_large"] = float((b["logits"].argmax(1) != NEI).mean())
    return out


def replay(name, episodes, table, rules, n_boot):
    return run_engine(name, episodes, table, rules)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/results")
    ap.add_argument("--large", default="runs/results_large")
    ap.add_argument("--base-stage-b", default="runs/stage_b")
    ap.add_argument("--large-stage-b", default="runs/stage_b_large")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--timing", default="runs/results/timing_large.json")
    ap.add_argument("--out", default="runs/results/large.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    B, L = Path(a.base), Path(a.large)
    rng = np.random.default_rng(0)
    out = {"plan": "docs/PREREGISTRATION_LARGE.md", "items": {}, "episodes": {}, "timing": None}

    for name in ("vitaminc.test", "mnli_mm.test", "wanli.test", "snli.test", "unrelated.test", "anli_r1.test", "anli_r2.test", "anli_r3.test"):
        if (B / "edge_logits" / f"{name}.npz").exists() and (L / "edge_logits" / f"{name}.npz").exists():
            out["items"][name] = paired_items(B, L, name, a.n_boot, rng)
            r = out["items"][name]
            print(f"{name:16s} base {100*r['accuracy_base']:.2f}  large {100*r['accuracy_large']:.2f}  "
                  f"diff {100*r['difference']:+.2f} [{100*r['lo']:+.2f}, {100*r['hi']:+.2f}]  p={r['mcnemar_p']:.3g}", flush=True)
    eb, el = read_json(B / "edge.json"), read_json(L / "edge.json")
    out["edge_summaries"] = {"base": {k: v["raw"] for k, v in eb["sets"].items()},
                             "large": {k: v["raw"] for k, v in el["sets"].items()}}
    for k in ("vitaminc.test",):
        out["edge_summaries"]["real_synthetic"] = {
            "base": {x: eb["sets"][k].get(x) for x in ("accuracy_real", "accuracy_synthetic")},
            "large": {x: el["sets"][k].get(x) for x in ("accuracy_real", "accuracy_synthetic")}}

    atoms = load_pool(a.atoms, "test")
    rb, rl = rules_for(Path(a.base_stage_b)), rules_for(Path(a.large_stage_b))
    for name, gen, n, seed in (("STANDARD", STANDARD, 1000, 20260929), ("WIDE", WIDE, 300, 20260930), ("SMALL", SMALL, 1000, 20261001)):
        tb, tl = B / "work" / f"table_{name}_edge", L / "work" / f"table_{name}_edge"
        if not (tb.exists() and tl.exists()):
            print(f"{name}: tables missing ({tb.exists()}, {tl.exists()}); skipped", flush=True)
            continue
        episodes = generate(atoms, n, seed, gen, tag=name.lower())
        res_b = replay("base", episodes, EdgeTable.load(tb), rb, a.n_boot)
        res_l = replay("large", episodes, EdgeTable.load(tl), rl, a.n_boot)
        sb, sl = summarise(res_b, a.n_boot), summarise(res_l, a.n_boot)
        c = compare(res_l, res_b, a.n_boot)  # large - base
        out["episodes"][name] = {"episodes": n, "base": sb, "large": sl, "large_minus_base": c}
        d = c["disposition_accuracy"]
        print(f"{name:9s} disposition base {100*sb['disposition_accuracy']['value']:.2f}  large {100*sl['disposition_accuracy']['value']:.2f}  "
              f"diff {100*d['difference']:+.2f} [{100*d['lo']:+.2f}, {100*d['hi']:+.2f}]  "
              f"stale {100*c['stale_decision_rate']['difference']:+.2f}  false READY {100*c['false_ready_rate']['difference']:+.2f}", flush=True)

    if Path(a.timing).exists():
        out["timing"] = read_json(a.timing)

    # ---- verdicts by the pre-registered rules
    v = {}
    it = out["items"]
    if "vitaminc.test" in it and "mnli_mm.test" in it:
        v["H10"] = all(it[k]["lo"] > 0 for k in ("vitaminc.test", "mnli_mm.test"))
    ep = out["episodes"]
    if "STANDARD" in ep:
        v["H11"] = ep["STANDARD"]["large_minus_base"]["disposition_accuracy"]["lo"] > 0
    tm = out["timing"]
    if tm and "base" in tm.get("runs", {}) and "large" in tm["runs"]:
        rb_ms = tm["runs"]["base"]["sets"]["STANDARD"]["update"]["p50_ms"]
        rl_ms = tm["runs"]["large"]["sets"]["STANDARD"]["update"]["p50_ms"]
        v["H12"] = rl_ms <= 3.5 * rb_ms
        out["timing_ratio_standard_p50"] = rl_ms / rb_ms
    ship, why = None, []
    if "H11" in v:
        ship = bool(v["H11"])
        if not v["H11"]:
            why.append("H11 not supported on STANDARD")
        for k in ("WIDE", "SMALL"):
            if k in ep and ep[k]["large_minus_base"]["disposition_accuracy"]["lo"] <= -0.01:
                ship, _ = False, why.append(f"could be more than 1 point worse on {k}")
            if k in ep and ep[k]["large_minus_base"]["false_ready_rate"]["lo"] > 0:
                ship, _ = False, why.append(f"false-READY rate significantly higher on {k}")
        if "STANDARD" in ep and ep["STANDARD"]["large_minus_base"]["false_ready_rate"]["lo"] > 0:
            ship, _ = False, why.append("false-READY rate significantly higher on STANDARD")
        if "unrelated.test" in it and it["unrelated.test"]["false_decisive_large"] > it["unrelated.test"]["false_decisive_base"] + 0.005:
            ship, _ = False, why.append("reads more unrelated passages as decisive")
        if tm and "large" in tm.get("runs", {}):
            if tm["runs"]["large"]["sets"]["STANDARD"]["update"]["p50_ms"] > 100:
                ship, _ = False, why.append("median update above 100 ms on the Apple GPU")
        else:
            ship, _ = False, why.append("timing not measured")
    out["verdicts"] = v
    out["ships"] = {"large": ship, "reasons_against": why}
    atomic_write_json(a.out, out)
    print("verdicts:", v, "| ships large:", ship, why, flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
