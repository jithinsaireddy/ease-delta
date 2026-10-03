"""H9 (exploratory): implicit conflicts from the STALE benchmark, read as a decision problem.

STALE (Chao et al., arXiv 2605.06527, CC BY 4.0) pairs an old statement with a later observation
that makes it invalid without negating it. Type T1 conflicts concern the same attribute; type T2
conflicts are propagated through a consequence ("broke my leg" against "I cycle to work").

Question put to the edge model: does the later observation refute the earlier statement?
Controls: the same earlier statement against other user messages from the same scenario's
conversation history, which do not conflict with it.

This is NOT the STALE protocol. The authors evaluate agents answering questions over histories of
up to 150K tokens, scored by an LLM judge. The numbers here are not comparable with theirs.

    python scripts/eval_stale.py --edge runs/stage_a/final --out runs/results/stale.json
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

from ease.eval.metrics import softmax
from ease.labels import NEI, REFUTES, SUPPORTS
from ease.util import atomic_write_json, read_json


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(max_controls: int, seed: int, limit=None):
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("STALEproj/STALE", split="train", streaming=True)
    pos, neg = [], []
    for row in ds:
        old, new, typ = row["M_old"].strip(), row["M_new"].strip(), row["type"]
        pos.append({"uid": row["uid"], "type": typ, "claim": old, "evidence": new})
        cands = []
        for session in row["haystack_session"]:
            for turn in session:
                if turn.get("role") != "user":
                    continue
                t = " ".join(str(turn.get("content", "")).split())
                if 20 <= len(t) <= 400 and t != old and t != new:
                    cands.append(t)
        rng.shuffle(cands)
        for t in cands[:max_controls]:
            neg.append({"uid": row["uid"], "type": typ, "claim": old, "evidence": t})
        if limit is not None and len(pos) >= limit:
            break
    return pos, neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--temperature-from", default="runs/results/edge.json")
    ap.add_argument("--controls-per-scenario", type=int, default=5)
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--proposer-threshold-from", default="runs/results/proposer.json")
    ap.add_argument("--out", default="runs/results/stale.json")
    ap.add_argument("--limit", type=int, default=None, help="dry run on the first scenarios only")
    ap.add_argument("--quiet", action="store_true", help="do not print results (for dry runs)")
    a = ap.parse_args()
    if a.quiet:
        global print
        print = lambda *x, **k: None

    from ease.scorer import ModelScorer

    pos, neg = load(a.controls_per_scenario, seed=0, limit=a.limit)
    print(f"STALE: {len(pos)} conflict pairs ({sum(p['type']=='T1' for p in pos)} T1, "
          f"{sum(p['type']=='T2' for p in pos)} T2), {len(neg)} control pairs", flush=True)
    T = read_json(a.temperature_from)["temperature"] if Path(a.temperature_from).exists() else 1.0
    scorer = ModelScorer(a.edge, batch_size=128)
    lp = np.stack([v[0] for v in scorer.score([(p["claim"], p["evidence"]) for p in pos])])
    ln = np.stack([v[0] for v in scorer.score([(p["claim"], p["evidence"]) for p in neg])])
    pp, pn = softmax(lp, T), softmax(ln, T)
    s_pos, s_neg = pp[:, REFUTES], pn[:, REFUTES]
    thr = float(np.quantile(s_neg, 1.0 - a.fpr))
    types = np.asarray([p["type"] for p in pos])
    out = {"hypothesis": "H9 (exploratory)", "edge_version": scorer.version, "temperature": T,
           "pairs": {"conflict": len(pos), "control": len(neg)},
           "threshold_on_p_refutes": thr, "false_positive_rate_on_controls": float((s_neg > thr).mean()),
           "controls_top_label": {n: float((pn.argmax(1) == k).mean()) for n, k in
                                  (("SUPPORTS", SUPPORTS), ("REFUTES", REFUTES), ("NEI", NEI))},
           "by_type": {},
           "not_comparable_with": "accuracies reported in the STALE paper (different task, different scoring)"}
    rng = np.random.default_rng(0)
    for t in ("T1", "T2", "all"):
        m = np.ones(len(pos), bool) if t == "all" else types == t
        s = s_pos[m]
        det = (s > thr).astype(float)
        boot = det[rng.integers(0, len(det), size=(2000, len(det)))].mean(1)
        out["by_type"][t] = {"n": int(m.sum()), "detection_rate": float(det.mean()),
                             "detection_lo": float(np.quantile(boot, 0.025)), "detection_hi": float(np.quantile(boot, 0.975)),
                             "auroc_vs_controls": auroc(s, s_neg),
                             "top_label": {n: float((pp[m].argmax(1) == k).mean()) for n, k in
                                           (("SUPPORTS", SUPPORTS), ("REFUTES", REFUTES), ("NEI", NEI))},
                             "mean_p_refutes": float(s.mean())}
        print(t, out["by_type"][t], flush=True)
    d = out["by_type"]["T1"]["detection_rate"] - out["by_type"]["T2"]["detection_rate"]
    out["T1_minus_T2_detection"] = d
    out["prediction_made_in_advance"] = "T2 detection substantially lower than T1"
    out["prediction_held"] = bool(d > 0.10)

    # would the similarity proposer even have compared the two statements?
    try:
        from ease.proposer import Embedder

        emb = Embedder()
        sim_pos = np.einsum("ij,ij->i", emb.encode([p["evidence"] for p in pos]), emb.encode([p["claim"] for p in pos]))
        sim_neg = np.einsum("ij,ij->i", emb.encode([p["evidence"] for p in neg]), emb.encode([p["claim"] for p in neg]))
        prop = {"mean_similarity": {"T1": float(sim_pos[types == "T1"].mean()), "T2": float(sim_pos[types == "T2"].mean()),
                                    "controls": float(sim_neg.mean())},
                "auroc_conflict_vs_control": {"T1": auroc(sim_pos[types == "T1"], sim_neg),
                                              "T2": auroc(sim_pos[types == "T2"], sim_neg)}}
        if Path(a.proposer_threshold_from).exists():
            lam = read_json(a.proposer_threshold_from)["calibration"]["threshold"]
            prop["threshold"] = lam
            prop["links_missed"] = {"T1": float((sim_pos[types == "T1"] < lam).mean()),
                                    "T2": float((sim_pos[types == "T2"] < lam).mean())}
            prop["controls_kept"] = float((sim_neg >= lam).mean())
        out["proposer"] = prop
        print("proposer:", prop, flush=True)
    except Exception as e:  # the proposer is optional; its absence must not hide the main result
        out["proposer"] = {"error": f"{type(e).__name__}: {e}"}

    examples = []
    order = np.argsort(s_pos)
    for i in list(order[:4]) + list(order[-4:]):
        examples.append({"type": pos[i]["type"], "p_refutes": float(s_pos[i]), "detected": bool(s_pos[i] > thr),
                         "old": pos[i]["claim"][:200], "new": pos[i]["evidence"][:200]})
    out["examples_lowest_and_highest"] = examples
    atomic_write_json(a.out, out)
    print(f"written {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
