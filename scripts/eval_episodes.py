"""Final evaluation on evolving episodes (hypotheses H2-H6 of docs/PREREGISTRATION.md).

    python scripts/eval_episodes.py --edge runs/stage_a/final --stage-b runs/stage_b \
        --dense runs/dense/final --out runs/results/episodes.json

Reads the test atom pool. Run once, after every model is trained.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ease.aggregate import Calibration, RuleAggregator, load_aggregator
from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, WIDE, describe, generate
from ease.eval.dense_data import contexts_of
from ease.eval.harness import (by_event_type, compare, full_rebuild_cost, run_dense, run_engine, summarise,
                               worst_episodes)
from ease.eval.trace import EdgeTable, build_table
from ease.util import atomic_write_json, get_device, read_json

SETS = {"STANDARD": (STANDARD, 1000, 20260929), "WIDE": (WIDE, 300, 20260930)}


def table_for(tag: str, episodes, scorer_factory, out: Path) -> EdgeTable:
    d = out / f"table_{tag}"
    if (d / "table.json").exists():
        return EdgeTable.load(d)
    t0 = time.time()
    table = build_table(episodes, scorer_factory())
    table.save(d)
    print(f"  table {tag}: {len(table.keys)} pairs, {int(table.tokens.sum())} tokens, {(time.time()-t0)/60:.1f} min", flush=True)
    return table


class PublicScorer:
    """Wraps a public NLI classifier as an edge scorer (no message vector is available from it)."""

    def __init__(self, name, device):
        sys.path.insert(0, str(Path(__file__).parent))
        from eval_edge import PublicNLI

        from ease.scorer import ScorerStats
        from ease.util import text_hash

        self.m = PublicNLI(name, device)
        self.tok = self.m.tok
        self.max_len = 256
        self.msg_dim = 128
        self.version = "public-" + text_hash(name)[:12]
        self.cache = {}
        self.stats = ScorerStats()
        self._hash = text_hash

    def key(self, c, e):
        return self._hash(self.version, c, e)

    def score(self, pairs):
        lg = self.m.predict([{"claim": c, "evidence": e} for c, e in pairs])
        z = np.zeros(self.msg_dim, np.float32)
        return [(lg[i], z) for i in range(len(pairs))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--stage-b", default="runs/stage_b")
    ap.add_argument("--dense", default="runs/dense/final")
    ap.add_argument("--public", default="tasksource/ModernBERT-base-nli")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--out", default="runs/results/episodes.json")
    ap.add_argument("--work", default="runs/results/work")
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--limit", type=int, default=None, help="use fewer episodes (for a dry run)")
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument("--skip-public", action="store_true")
    ap.add_argument("--dense-max-len", type=int, default=4096)
    ap.add_argument("--dense-episodes", type=int, nargs=2, default=[400, 100], metavar=("STANDARD", "WIDE"),
                    help="B2 is run on the first K episodes of each set; comparisons with B2 use those episodes")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pool", default="test", help="atom pool; anything but 'test' is a dry run")
    a = ap.parse_args()

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    device = get_device()
    atoms = load_pool(a.atoms, a.pool)
    sb = read_json(Path(a.stage_b) / "summary.json")
    cal = read_json(Path(a.stage_b) / "rule_calibration.json")
    rule_cal = Calibration(cal["temperature"], tuple(cal["bias"]))
    report = {"prereg_sha256": Path("runs/prereg_hash.txt").read_text().split()[0], "edge": a.edge,
              "pool": a.pool, "dry_run": a.pool != "test" or a.limit is not None,
              "stage_b": sb, "sets": {}}

    from ease.scorer import ModelScorer

    # B3 is to be treated "as B1-soft": its probabilities are calibrated through the policy on the
    # same Stage B validation episodes, using its own readings.
    b3_cal = None
    if not a.skip_public:
        from ease.eval.trace import trace
        from ease.train.stage_b import Examples, fit_temperature_through_policy

        sb_eps = generate(load_pool(a.atoms, "stage_b"), sb["episodes"], sb.get("episode_seed", 20260901), STANDARD,
                          tag="stageb")
        n_val = sb.get("val_episodes", max(1, int(round(sb.get("val_frac", 0.1) * len(sb_eps)))))
        val_eps = sb_eps[-n_val:]
        ptab = table_for("stageb_val_public", val_eps, lambda: PublicScorer(a.public, device), work)
        ex = Examples(trace(val_eps, ptab), ptab, torch.device("cpu"))
        b3_cal = fit_temperature_through_policy(ex, np.arange(ex.n))
        report["b3_calibration"] = {"temperature": b3_cal.temperature, "bias": list(b3_cal.bias),
                                    "fitted_on": f"{n_val} Stage B validation episodes, {ex.n} examples"}
        print(f"B3 calibration through the policy: T={b3_cal.temperature:.4f} bias={[round(b, 4) for b in b3_cal.bias]}",
              flush=True)

    for set_name in a.sets:
        gen, n, seed = SETS[set_name]
        n = a.limit or n
        episodes = generate(atoms, n, seed, gen, tag=set_name.lower())
        print(f"== {set_name}: {len(episodes)} episodes", flush=True)
        entry = {"episodes": describe(episodes), "systems": {}, "comparisons": {}, "by_event_type": {},
                 "worst_episodes": {}}
        table = table_for(f"{set_name}_edge", episodes, lambda: ModelScorer(a.edge, batch_size=128), work)

        results = {}
        results["B1-hard"] = run_engine("B1-hard", episodes, table, RuleAggregator(hard=True, msg_dim=table.msg_dim))
        results["B1-soft"] = run_engine("B1-soft", episodes, table, RuleAggregator(rule_cal, msg_dim=table.msg_dim))
        for run in sb["runs"]:
            tag = ("E" if run["kind"] == "refined" else "N") + f"-seed{run['seed']}"
            results[tag] = run_engine(tag, episodes, table, load_aggregator(run["dir"]))
        results["E-full"] = full_rebuild_cost(episodes, table)
        results["E-declared"] = run_engine("E-declared", episodes, table,
                                           load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"]),
                                           declared_links=True)

        if not a.skip_public:
            pt = table_for(f"{set_name}_public", episodes, lambda: PublicScorer(a.public, device), work)
            results["B3"] = run_engine("B3", episodes, pt, RuleAggregator(b3_cal, msg_dim=pt.msg_dim))
            results["B3-uncalibrated"] = run_engine("B3-uncalibrated", episodes, pt, RuleAggregator(msg_dim=pt.msg_dim))

        if not a.skip_dense:
            from transformers import AutoTokenizer

            from ease.model.dense import DenseModel, predict_contexts

            k = min(len(episodes), a.dense_episodes[0 if set_name == "STANDARD" else 1])
            sub = episodes[:k]
            entry["dense_episodes"] = k
            cache = work / f"dense_{set_name}_{k}.npz"
            rows = contexts_of(sub)
            if cache.exists():
                z = np.load(cache)
                logits, tokens = z["logits"], z["tokens"]
            else:
                t0 = time.time()
                dm = DenseModel.load(a.dense, device).eval()
                tok = AutoTokenizer.from_pretrained(a.dense)
                logits, tokens = predict_contexts(dm, tok, [r["claim"] for r in rows], [r["context"] for r in rows],
                                                  device, a.dense_max_len)
                np.savez(cache, logits=logits, tokens=tokens)
                print(f"  dense: {len(rows)} contexts, {int(tokens.sum())} tokens, {(time.time()-t0)/60:.1f} min; "
                      f"truncated={(tokens>=a.dense_max_len).mean():.4f}", flush=True)
                del dm
                if device.type == "mps":
                    torch.mps.empty_cache()
            results["B2"] = run_dense("B2", sub, rows, logits, tokens)
            # the same episodes for the systems B2 is compared with
            for run in sb["runs"]:
                if run["kind"] == "refined":
                    results[f"E-seed{run['seed']}@B2"] = run_engine(f"E-seed{run['seed']}@B2", sub, table,
                                                                    load_aggregator(run["dir"]))
            results["B1-soft@B2"] = run_engine("B1-soft@B2", sub, table, RuleAggregator(rule_cal, msg_dim=table.msg_dim))
            results["E-full@B2"] = full_rebuild_cost(sub, table)
            entry["dense_context_tokens"] = {"mean": float(tokens.mean()), "p50": float(np.median(tokens)),
                                             "p95": float(np.percentile(tokens, 95)), "max": int(tokens.max()),
                                             "truncated_fraction": float((tokens >= a.dense_max_len).mean())}

        # exploratory: what raising the confidence required for READY buys and costs
        from ease.engine import EngineConfig
        sweep = {}
        first_refined = load_aggregator([r for r in sb["runs"] if r["kind"] == "refined"][0]["dir"])
        for thr in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
            m = summarise(run_engine(f"E@{thr}", episodes, table, first_refined,
                                     config=EngineConfig(ready_threshold=thr)), 400)
            sweep[str(thr)] = {k: m[k] for k in ("disposition_accuracy", "false_ready_rate", "missed_ready_rate",
                                                 "stale_decision_rate")}
        entry["threshold_sweep"] = sweep

        for k, r in results.items():
            entry["systems"][k] = summarise(r, a.n_boot)
            if len(r.actions.get("ep", ())):
                entry["by_event_type"][k] = by_event_type(r)
                entry["worst_episodes"][k] = worst_episodes(r)
        main_e = "E-seed1" if "E-seed1" in results else next(
            k for k in results if k.startswith("E-seed") and "@" not in k)
        pairs = [(main_e, "E-full"), (main_e, "B1-soft"), (main_e, "B1-hard"), (main_e, "E-declared")]
        pairs += [(k, "B1-soft") for k in results if k.startswith("E-seed") and k != main_e and "@" not in k]
        pairs += [(k.replace("E-", "N-"), k) for k in results
                  if k.startswith("E-seed") and "@" not in k and k.replace("E-", "N-") in results]
        if "B2" in results:
            pairs += [(k, "B2") for k in results if k.startswith("E-seed") and k.endswith("@B2")]
            pairs += [("B1-soft@B2", "B2"), ("E-full@B2", "B2")]
        if "B3" in results:
            pairs += [(main_e, "B3")]
        for x, y in pairs:
            entry["comparisons"][f"{x} vs {y}"] = compare(results[x], results[y], a.n_boot)

        for k in sorted(entry["systems"]):
            s = entry["systems"][k]
            if "disposition_accuracy" not in s:
                print(f"  {k:12s} tokens/changed event={s['tokens_per_changed_event']['value']:.0f}", flush=True)
                continue
            print(f"  {k:12s} disp={s['disposition_accuracy']['value']:.4f} stale={s['stale_decision_rate']['value']:.4f} "
                  f"spur={s['spurious_change_rate']['value']:.4f} falseREADY={s['false_ready_rate']['value']:.4f} "
                  f"status={s['status_accuracy']['value']:.4f} nll={s['status_nll']['value']:.4f} "
                  f"tok/chg={s['tokens_per_changed_event']['value']:.0f}", flush=True)
        report["sets"][set_name] = entry
        atomic_write_json(a.out, report)
    print(f"written {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
