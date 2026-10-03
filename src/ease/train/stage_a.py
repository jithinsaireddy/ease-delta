"""Stage A: train the edge function on a streamed mixture of public corpora.

    python -m ease.train.stage_a --config configs/stage_a.yaml
    python -m ease.train.stage_a --config configs/stage_a.yaml --resume

The run has a fixed example budget and uses the final weights. There is no early stopping and no
checkpoint selection, so no evaluation split influences which weights are kept.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import signal
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from ease.data.evalsets import load_eval_set
from ease.data.streams import BatchStream, MixtureConfig, Prefetcher, StreamMixture
from ease.eval.metrics import summarise
from ease.model.edge import EdgeConfig, EdgeModel, relation_loss
from ease.model.infer import predict_pairs, round_up
from ease.util import JsonlLogger, atomic_write_json, clip_grad_norm_fast, device_sync, get_device, seed_everything


@dataclass
class StageAConfig:
    run_dir: str = "runs/stage_a"
    eval_cache: str = "runs/eval_cache"
    encoder_name: str = "answerdotai/ModernBERT-base"
    msg_dim: int = 128
    pooling: str = "cls"
    dropout: float = 0.1
    max_len: int = 256
    batch_size: int = 32
    total_examples: int = 600_000
    lr: float = 4e-5
    head_lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    seed: int = 1234
    mixture: dict = field(default_factory=lambda: {"vitaminc": 0.40, "mnli": 0.27, "wanli": 0.13, "unrelated": 0.20})
    shuffle_buffer: int = 1_000_000  # spans every corpus used; see ease.data.streams._open
    pool_batches: int = 50
    mix_every: int = 500  # steps between reports of each source's label mix
    mix_tolerance: float = 0.12  # warn when a label's share in a window moves this far from its running share
    pad_multiple: int = 8
    log_every: int = 50
    eval_every: int = 2000
    ckpt_every: int = 1000
    dev_sets: list = field(default_factory=lambda: ["vitaminc.dev", "mnli.dev", "unrelated.dev", "snli.dev"])
    dev_limit: int = 2000
    empty_cache_every: int = 200
    device: str = ""
    init_from: str = ""  # directory of an edge model to continue from, instead of the pretrained encoder
    local_sources: dict = field(default_factory=dict)  # mixture key -> JSONL path (see LocalStream)

    @property
    def total_steps(self) -> int:
        return math.ceil(self.total_examples / self.batch_size)


def load_config(path: str | None, overrides: dict) -> StageAConfig:
    data = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    known = set(StageAConfig.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return StageAConfig(**data)


def build_optimizer(model: EdgeModel, cfg: StageAConfig) -> torch.optim.Optimizer:
    no_decay_markers = ("bias", "norm", "LayerNorm", "layer_norm")
    enc_decay, enc_plain, head_decay, head_plain = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_plain = p.ndim < 2 or any(m in name for m in no_decay_markers)
        if name.startswith("encoder."):
            (enc_plain if is_plain else enc_decay).append(p)
        else:
            (head_plain if is_plain else head_decay).append(p)
    groups = [
        {"params": enc_decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": enc_plain, "lr": cfg.lr, "weight_decay": 0.0},
        {"params": head_decay, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"params": head_plain, "lr": cfg.head_lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.999), eps=1e-6)


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def repad(batch, pad_id: int, multiple: int):
    """Pad to a multiple of `multiple` tokens so the GPU sees a small number of distinct shapes."""
    L = batch.input_ids.shape[1]
    target = round_up(L, multiple)
    if target == L:
        return batch.input_ids, batch.attention_mask
    pad = target - L
    ids = torch.nn.functional.pad(batch.input_ids, (0, pad), value=pad_id)
    mask = torch.nn.functional.pad(batch.attention_mask, (0, pad), value=0)
    return ids, mask


def evaluate(model, tok, cfg: StageAConfig, device, names: list[str], limit: int | None) -> dict:
    out = {}
    for name in names:
        rows = load_eval_set(cfg.eval_cache, name)
        if limit is not None:
            rows = rows[:limit]
        logits, _ = predict_pairs(
            model, tok, [r["claim"] for r in rows], [r["evidence"] for r in rows], device,
            max_len=cfg.max_len, batch_size=128, pad_multiple=cfg.pad_multiple,
        )
        labels = np.array([r["label"] for r in rows])
        groups = [r["group"] for r in rows] if name.startswith("vitaminc") else None
        out[name] = summarise(logits, labels, groups=groups)
    return out


def save_checkpoint(run: Path, model, opt, step: int, examples: int, batch, source_counts: Counter,
                    elapsed: float, keep: int = 2) -> None:
    d = run / "checkpoints" / f"step_{step:08d}"
    tmp = run / "checkpoints" / f".tmp_step_{step:08d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save(tmp)
    torch.save(
        {
            "optimizer": opt.state_dict(),
            "step": step,
            "examples": examples,
            "elapsed": elapsed,
            "source_counts": dict(source_counts),
            "stream": {
                "pool_id": batch.pool_id,
                "state_before_pool": batch.state_before_pool,
                "next_index_in_pool": batch.index_in_pool + 1,
            },
            "torch_rng": torch.get_rng_state(),
        },
        tmp / "trainer_state.pt",
    )
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)
    existing = sorted(p for p in (run / "checkpoints").iterdir() if p.name.startswith("step_"))
    for old in existing[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def latest_checkpoint(run: Path) -> Path | None:
    c = run / "checkpoints"
    if not c.exists():
        return None
    cands = sorted(p for p in c.iterdir() if p.name.startswith("step_") and (p / "trainer_state.pt").exists())
    return cands[-1] if cands else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--total-examples", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--dev-limit", type=int, default=None)
    ap.add_argument("--init-from", default=None, help="continue from this edge model")
    args = ap.parse_args(argv)
    cfg = load_config(
        args.config,
        {"run_dir": args.run_dir, "total_examples": args.total_examples, "eval_every": args.eval_every,
         "ckpt_every": args.ckpt_every, "dev_limit": args.dev_limit, "init_from": args.init_from},
    )

    from transformers import AutoTokenizer
    import datasets as hf_datasets
    import transformers

    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.device or None)
    seed_everything(cfg.seed)
    log = JsonlLogger(run / "train_log.jsonl")

    tok = AutoTokenizer.from_pretrained(cfg.encoder_name)
    edge_cfg = EdgeConfig(cfg.encoder_name, cfg.msg_dim, cfg.dropout, cfg.pooling, cfg.max_len)

    ckpt = latest_checkpoint(run) if args.resume else None
    if ckpt is not None:
        model = EdgeModel.load(ckpt)
        state = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)
    else:
        if args.resume:
            print("no checkpoint found; starting from the pretrained encoder", flush=True)
        if cfg.init_from:
            model = EdgeModel.load(cfg.init_from)
            print(f"continuing from {cfg.init_from}", flush=True)
        else:
            model = EdgeModel(edge_cfg)
        state = None
    model.to(device).train()
    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]

    mixture = StreamMixture(MixtureConfig(dict(cfg.mixture), cfg.seed, cfg.shuffle_buffer,
                                          local=dict(cfg.local_sources)))
    batches = BatchStream(mixture, tok, cfg.batch_size, cfg.max_len, cfg.pool_batches)
    step, examples, elapsed_before = 0, 0, 0.0
    source_counts: Counter = Counter()
    if state is not None:
        opt.load_state_dict(state["optimizer"])
        step, examples = int(state["step"]), int(state["examples"])
        elapsed_before = float(state["elapsed"])
        source_counts.update(state["source_counts"])
        torch.set_rng_state(state["torch_rng"])
        s = state["stream"]
        batches.resume(s["pool_id"], s["state_before_pool"], s["next_index_in_pool"])
        print(f"resumed from {ckpt.name}: step={step} examples={examples}", flush=True)

    total_steps = cfg.total_steps
    warmup = int(cfg.warmup_frac * total_steps)
    n_params = sum(p.numel() for p in model.parameters())
    atomic_write_json(
        run / "run_config.json",
        {
            "config": asdict(cfg), "total_steps": total_steps, "warmup_steps": warmup, "parameters": n_params,
            "device": str(device), "licences": mixture.licences(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "datasets": hf_datasets.__version__, "python": platform.python_version()},
            "machine": {"platform": platform.platform(), "processor": platform.processor()},
        },
    )
    print(f"stage A: {n_params/1e6:.1f}M params on {device}; {total_steps} steps of {cfg.batch_size}; "
          f"mixture={mixture.cfg.normalised()}", flush=True)

    stop = {"flag": False}

    def _on_signal(signum, _frame):
        stop["flag"] = True
        print(f"signal {signum} received: will checkpoint and exit after this step", flush=True)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    pre = Prefetcher(batches)
    t_start = time.time()
    ema = None
    # label mix per source: over the current window, and over the whole run so far
    mix_window: dict = {}
    mix_total: dict = {}
    drift_warnings = 0
    win_ex, win_tok, win_t = 0, 0, time.time()
    last_batch = None
    try:
        while step < total_steps and not stop["flag"]:
            batch = pre.get()
            last_batch = batch
            f = lr_factor(step, total_steps, warmup)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * f
            ids, mask = repad(batch, tok.pad_token_id, cfg.pad_multiple)
            logits, _ = model(ids.to(device), mask.to(device))
            loss = relation_loss(logits, batch.labels.to(device), cfg.label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = clip_grad_norm_fast(model.parameters(), cfg.grad_clip)
            opt.step()
            step += 1
            examples += len(batch.labels)
            source_counts.update(batch.sources)
            for src, lab in zip(batch.sources, batch.labels.tolist()):
                for table in (mix_window, mix_total):
                    table.setdefault(src, [0, 0, 0, 0])[min(lab, 3)] += 1
            if step % cfg.mix_every == 0:
                shares, drift = {}, []
                for src, counts in sorted(mix_window.items()):
                    n_w, n_t = sum(counts), sum(mix_total[src])
                    if n_w < 200:
                        continue
                    shares[src] = [round(c / n_w, 3) for c in counts[:3]]
                    for k in range(3):
                        gap = abs(counts[k] / n_w - mix_total[src][k] / n_t)
                        if gap > cfg.mix_tolerance and n_t > 4 * n_w:
                            drift.append((src, k, round(gap, 3)))
                log.log(kind="mix", step=step, label_share_by_source=shares, drift=drift)
                if drift:
                    drift_warnings += 1
                    print(f"  WARNING step {step}: label mix of a source moved away from its running mix: {drift}. "
                          "The stream may be ordered.", flush=True)
                mix_window = {}
            win_ex += len(batch.labels)
            win_tok += batch.n_tokens
            lv = float(loss.item())
            ema = lv if ema is None else 0.98 * ema + 0.02 * lv

            if step % cfg.log_every == 0:
                device_sync(device)
                dt = time.time() - win_t
                rate = win_ex / dt
                eta = (total_steps - step) * cfg.batch_size / max(rate, 1e-6)
                rec = dict(kind="train", step=step, examples=examples, loss=round(lv, 4), loss_ema=round(ema, 4),
                           lr=round(base_lrs[0] * f, 8), grad_norm=round(float(gnorm), 3),
                           ex_per_s=round(rate, 1), tok_per_s=round(win_tok / dt, 0), eta_min=round(eta / 60, 1),
                           queue=pre.q.qsize(),
                           mps_gb=round(torch.mps.driver_allocated_memory() / 2**30, 2) if device.type == "mps" else None)
                log.log(**rec)
                if step % (cfg.log_every * 4) == 0:
                    print(f"step {step:6d}/{total_steps} ex={examples:7d} loss={ema:.4f} lr={rec['lr']:.2e} "
                          f"{rate:6.1f} ex/s eta={rec['eta_min']:.0f}m mem={rec['mps_gb']}GB", flush=True)
                win_ex, win_tok, win_t = 0, 0, time.time()

            if device.type == "mps" and step % cfg.empty_cache_every == 0:
                torch.mps.empty_cache()

            if step % cfg.eval_every == 0 or step == total_steps:
                ev = evaluate(model, tok, cfg, device, cfg.dev_sets, cfg.dev_limit)
                model.train()
                log.log(kind="eval", step=step, examples=examples, metrics=ev)
                print("  dev: " + "  ".join(f"{k} acc={v['accuracy']:.4f} ece={v['ece']:.3f}" for k, v in ev.items()),
                      flush=True)
                win_t = time.time(); win_ex = 0; win_tok = 0

            if step % cfg.ckpt_every == 0 and step < total_steps:
                save_checkpoint(run, model, opt, step, examples, batch, source_counts,
                                elapsed_before + time.time() - t_start)
    finally:
        pre.close()

    elapsed = elapsed_before + time.time() - t_start
    if stop["flag"] and step < total_steps:
        if last_batch is not None:
            save_checkpoint(run, model, opt, step, examples, last_batch, source_counts, elapsed)
        print(f"stopped at step {step}; resume with --resume", flush=True)
        return 130

    final = run / "final"
    model.save(final, extra={"stage": "A", "steps": step, "examples": examples})
    tok.save_pretrained(final)
    dev = evaluate(model, tok, cfg, device, cfg.dev_sets, None)
    manifest = {
        "stage": "A",
        "steps": step,
        "examples": examples,
        "examples_by_source": dict(source_counts),
        "licences": mixture.licences(),
        "stream_reopens": {k: s.reopen_count for k, s in mixture.streams.items()},
        "stream_epochs": {k: s.epoch for k, s in mixture.streams.items()},
        "shuffle_buffer": cfg.shuffle_buffer,
        "label_counts_by_source(S,R,N,not-S)": mix_total,
        "label_mix_drift_warnings": drift_warnings,
        "train_seconds": round(elapsed, 1),
        "dev_metrics_full": dev,
        "config": asdict(cfg),
        "parameters": n_params,
    }
    atomic_write_json(final / "training_manifest.json", manifest)
    log.log(kind="final", step=step, examples=examples, metrics=dev, seconds=round(elapsed, 1))
    print(f"done: {examples} examples in {elapsed/60:.1f} min -> {final}", flush=True)
    print(json.dumps({k: {m: round(x, 4) if isinstance(x, float) else x for m, x in v.items()} for k, v in dev.items()}, indent=1))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Streaming readers keep network threads alive; a normal interpreter shutdown can block on them.
    os._exit(code)
