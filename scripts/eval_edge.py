"""Evaluate an edge model on every evaluation set.

    python scripts/eval_edge.py --model runs/stage_a/final --out runs/results/edge.json
    python scripts/eval_edge.py --hf tasksource/ModernBERT-base-nli --out runs/results/edge_b3.json

Temperature is fitted on dev sets only and then applied, unchanged, to the test sets.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ease.data.evalsets import load_eval_set, names
from ease.eval.metrics import fit_temperature, softmax, summarise
from ease.labels import NEI, REFUTES, SUPPORTS
from ease.util import atomic_write_json, get_device


def predict_ours(model_dir, rows, device):
    from transformers import AutoTokenizer

    from ease.model.edge import EdgeModel
    from ease.model.infer import predict_pairs

    if not hasattr(predict_ours, "m"):
        predict_ours.m = EdgeModel.load(model_dir, device).eval()
        predict_ours.t = AutoTokenizer.from_pretrained(model_dir)
    lg, _ = predict_pairs(predict_ours.m, predict_ours.t, [r["claim"] for r in rows], [r["evidence"] for r in rows],
                          device, max_len=predict_ours.m.cfg.max_len, batch_size=128)
    return lg


class PublicNLI:
    """A public three-way NLI classifier, read through its own label names."""

    def __init__(self, name, device, max_len=256):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device, self.max_len = device, max_len
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        want = {SUPPORTS: "entail", REFUTES: "contradict", NEI: "neutral"}
        self.cols = []
        for ours in (SUPPORTS, REFUTES, NEI):
            hit = [i for i, l in id2label.items() if want[ours] in l]
            if len(hit) != 1:
                raise ValueError(f"cannot map labels {id2label} to entailment/contradiction/neutral")
            self.cols.append(hit[0])
        self.id2label = id2label

    @torch.inference_mode()
    def predict(self, rows, batch_size=128):
        prem = [r["evidence"] for r in rows]
        hyp = [r["claim"] for r in rows]
        order = sorted(range(len(rows)), key=lambda i: len(prem[i]) + len(hyp[i]))
        out = np.zeros((len(rows), 3), np.float32)
        for s in range(0, len(rows), batch_size):
            ix = order[s : s + batch_size]
            enc = self.tok([prem[i] for i in ix], [hyp[i] for i in ix], truncation="longest_first",
                           max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
            lg = self.model(**enc).logits.float().cpu().numpy()
            out[ix] = lg[:, self.cols]
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--hf", default=None)
    ap.add_argument("--eval-cache", default="runs/eval_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-logits", default=None, help="directory for per-set logits (npz)")
    a = ap.parse_args()
    device = get_device()
    if bool(a.model) == bool(a.hf):
        raise SystemExit("give exactly one of --model or --hf")
    public = PublicNLI(a.hf, device) if a.hf else None

    def predict(rows):
        return public.predict(rows) if public else predict_ours(a.model, rows, device)

    t0 = time.time()
    dev_logits, dev_labels, results = [], [], {"model": a.model or a.hf, "sets": {}}
    if public:
        results["label_map"] = public.id2label
    cache = {}
    for name in names("dev"):
        rows = load_eval_set(a.eval_cache, name)
        if name == "mnli.dev":
            rows = rows[:5000]  # the rest of this split feeds Stage B and must not tune anything here
        lg = predict(rows)
        cache[name] = (rows, lg)
        if name != "snli.dev":  # SNLI's labelling convention is not the target; it must not set the temperature
            dev_logits.append(lg)
            dev_labels.append(np.asarray([r["label"] for r in rows]))
    T = fit_temperature(np.concatenate(dev_logits), np.concatenate(dev_labels))
    results["temperature"] = T
    results["temperature_fitted_on"] = [n for n in names("dev") if n != "snli.dev"]

    for role in ("dev", "test"):
        for name in names(role):
            rows, lg = cache[name] if name in cache else (None, None)
            if rows is None:
                rows = load_eval_set(a.eval_cache, name)
                lg = predict(rows)
            y = np.asarray([r["label"] for r in rows])
            groups = [r["group"] for r in rows] if name.startswith("vitaminc") else None
            raw = summarise(lg, y, 1.0, groups)
            cal = summarise(lg, y, T, groups)
            entry = {"role": role, "n": len(rows), "raw": raw, "calibrated": cal}
            if name.startswith("unrelated"):
                p = softmax(lg, T)
                entry["false_decisive_rate"] = float((p.argmax(1) != NEI).mean())
                entry["mean_p_nei"] = float(p[:, NEI].mean())
                entry["p_no_false_decisive_among_10"] = float((1 - entry["false_decisive_rate"]) ** 10)
                entry["p_no_false_decisive_among_30"] = float((1 - entry["false_decisive_rate"]) ** 30)
            if name.startswith("vitaminc") and rows and "revision_type" in rows[0]:
                for rt in sorted({r.get("revision_type") for r in rows if r.get("revision_type")}):
                    ix = [i for i, r in enumerate(rows) if r.get("revision_type") == rt]
                    entry[f"accuracy_{rt}"] = float((lg[ix].argmax(1) == y[ix]).mean())
                    entry[f"n_{rt}"] = len(ix)
            results["sets"][name] = entry
            print(f"{name:16s} {role:4s} n={len(rows):6d} acc={raw['accuracy']:.4f} f1={raw['macro_f1']:.4f} "
                  f"nll={raw['nll']:.4f}->{cal['nll']:.4f} ece={raw['ece']:.4f}->{cal['ece']:.4f}"
                  + (f" set_acc={raw['set_accuracy']:.4f}" if "set_accuracy" in raw else "")
                  + (f" false_decisive={entry['false_decisive_rate']:.4f}" if "false_decisive_rate" in entry else ""),
                  flush=True)
            if a.save_logits:
                Path(a.save_logits).mkdir(parents=True, exist_ok=True)
                np.savez(Path(a.save_logits) / f"{name}.npz", logits=lg, labels=y)
    results["seconds"] = round(time.time() - t0, 1)
    atomic_write_json(a.out, results)
    print(f"T={T:.4f}; written {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
