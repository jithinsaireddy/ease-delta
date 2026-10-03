"""Train baseline B2, the dense reader.

    python -m ease.train.dense --edge runs/stage_a/final --out runs/dense

It starts from the Stage A weights and trains on the same Stage B episodes as the aggregators.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, generate
from ease.data.streams import pad_batch
from ease.eval.dense_data import contexts_of
from ease.model.dense import DenseConfig, DenseModel, encode_contexts, predict_contexts
from ease.model.infer import round_up
from ease.train.stage_a import build_optimizer, lr_factor
from ease.util import JsonlLogger, atomic_write_json, clip_grad_norm_fast, device_sync, get_device, seed_everything


def token_batches(ids, labels, token_budget, rng, pad_multiple=32, max_rows=32):
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    batches, s = [], 0
    while s < len(order):
        e = s + 1
        while e < len(order) and (e - s + 1) * round_up(len(ids[order[e]]), pad_multiple) <= token_budget and e - s < max_rows:
            e += 1
        batches.append(order[s:e])
        s = e
    rng.shuffle(batches)
    return batches


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--out", default="runs/dense")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--episode-seed", type=int, default=20260901)
    ap.add_argument("--train-examples", type=int, default=60000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val-examples", type=int, default=4000)
    ap.add_argument("--val-every", type=int, default=400)
    ap.add_argument("--val-probe", type=int, default=800)
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(out / "train_log.jsonl")
    seed_everything(a.seed)
    rng = random.Random(a.seed)
    device = get_device()

    episodes = generate(load_pool(a.atoms, "stage_b"), a.episodes, a.episode_seed, STANDARD, tag="stageb")
    n_val = max(1, int(round(a.val_frac * len(episodes))))
    rows = contexts_of(episodes)
    first_val = len(episodes) - n_val
    train_rows = [r for r in rows if r["episode"] < first_val]
    val_rows = [r for r in rows if r["episode"] >= first_val]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    train_rows, val_rows = train_rows[: a.train_examples], val_rows[: a.val_examples]
    print(f"contexts: train={len(train_rows)} val={len(val_rows)} from {len(episodes)} episodes "
          f"(the same episodes and split as Stage B)", flush=True)

    tok = AutoTokenizer.from_pretrained(a.edge)
    model = DenseModel(DenseConfig(max_len=a.max_len))
    init = model.init_from_edge(a.edge)
    print(f"initialised from Stage A: {init}", flush=True)
    model.to(device).train()

    ids = encode_contexts(tok, [r["claim"] for r in train_rows], [r["context"] for r in train_rows], a.max_len)
    lens = np.asarray([len(x) for x in ids])
    print(f"context tokens: mean={lens.mean():.0f} p50={np.median(lens):.0f} p95={np.percentile(lens,95):.0f} "
          f"max={lens.max()} truncated={(lens>=a.max_len).mean():.4f}", flush=True)
    labels = [r["label"] for r in train_rows]

    class Cfg:  # the optimiser builder of Stage A, with this run's rates
        lr, head_lr, weight_decay = a.lr, a.head_lr, 0.01
    opt = build_optimizer(model, Cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]

    n_epochs = max(1, math.ceil(a.epochs))
    all_batches = []
    for ep in range(n_epochs):
        all_batches += token_batches(ids, labels, a.token_budget, rng)
    total = int(len(all_batches) * a.epochs / n_epochs)
    all_batches = all_batches[:total]
    warmup = int(0.05 * total)
    print(f"{total} steps; token budget {a.token_budget} per batch", flush=True)

    t0 = time.time()
    ema, seen_tokens, seen_ex = None, 0, 0
    for step, ix in enumerate(all_batches):
        f = lr_factor(step, total, warmup)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        L = min(a.max_len, round_up(max(len(ids[i]) for i in ix), 32))
        inp, mask = pad_batch([ids[i] for i in ix], tok.pad_token_id, pad_to=L)
        y = torch.tensor([labels[i] for i in ix], device=device)
        loss = F.cross_entropy(model(inp.to(device), mask.to(device)).float(), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_fast(model.parameters(), 1.0)
        opt.step()
        lv = float(loss.item())
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        seen_tokens += int(mask.sum()); seen_ex += len(ix)
        if (step + 1) % 50 == 0:
            device_sync(device)
            dt = time.time() - t0
            log.log(kind="train", step=step + 1, loss_ema=round(ema, 4), examples=seen_ex, tokens=seen_tokens,
                    ex_per_s=round(seen_ex / dt, 1), tok_per_s=round(seen_tokens / dt))
            if (step + 1) % 200 == 0:
                print(f"step {step+1}/{total} loss={ema:.4f} ex={seen_ex} {seen_ex/dt:.1f} ex/s "
                      f"{seen_tokens/dt:.0f} tok/s eta={(total-step-1)*dt/(step+1)/60:.0f}m", flush=True)
        if device.type == "mps" and (step + 1) % 100 == 0:
            torch.mps.empty_cache()
        if (step + 1) % a.val_every == 0 and step + 1 < total:
            pv = val_rows[: a.val_probe]
            lg, _ = predict_contexts(model, tok, [r["claim"] for r in pv], [r["context"] for r in pv], device, a.max_len)
            model.train()
            acc = float((lg.argmax(1) == np.asarray([r["label"] for r in pv])).mean())
            log.log(kind="val", step=step + 1, examples=seen_ex, accuracy=acc, n=len(pv))
            print(f"  val probe at step {step+1}: accuracy={acc:.4f} (n={len(pv)})", flush=True)

    elapsed = time.time() - t0
    logits, _ = predict_contexts(model, tok, [r["claim"] for r in val_rows], [r["context"] for r in val_rows],
                                 device, a.max_len)
    yv = np.asarray([r["label"] for r in val_rows])
    z = logits.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    val = {"n": len(yv), "accuracy": float((p.argmax(1) == yv).mean()),
           "nll": float(-np.log(np.clip(p[np.arange(len(yv)), yv], 1e-12, 1)).mean())}
    model.save(out / "final", extra={"init": init, "val": val, "examples": seen_ex, "tokens": seen_tokens})
    tok.save_pretrained(out / "final")
    atomic_write_json(out / "final" / "training_manifest.json", {
        "examples": seen_ex, "tokens": seen_tokens, "steps": total, "train_seconds": round(elapsed, 1),
        "val": val, "args": vars(a), "initialised_from": str(a.edge), "init": init})
    print(f"done in {elapsed/60:.1f} min; tokens={seen_tokens}; val={json.dumps(val)}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
