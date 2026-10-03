"""Stage B: train the parts downstream of the edge model, on evolving episodes.

    python -m ease.train.stage_b --edge runs/stage_a/final --out runs/stage_b

Two models are trained on identical examples so they can be compared on equal terms:

    refined   EdgeRefiner: re-reads each edge; the precedence policy is executed exactly and
              gradients flow through it
    neural    RankedReader: a GRU over ranked edges that outputs the status itself

Examples come only from the Stage B atom pool. Model selection uses a validation split of those
episodes. No test episode is read here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ease.aggregate import (Calibration, EdgeRefiner, NeuralAggregator, NeuralConfig, RankedReader,
                            RefinedAggregator, RefinerConfig, precedence)
from ease.data.atoms import load_pool
from ease.data.episodes import STANDARD, describe, generate
from ease.eval.trace import EdgeTable, build_table, trace
from ease.util import JsonlLogger, atomic_write_json, seed_everything


def dense_rank_rows(auth: np.ndarray, vf: np.ndarray) -> np.ndarray:
    keys = sorted({(int(a), float(v)) for a, v in zip(auth, vf)}, reverse=True)
    pos = {k: i for i, k in enumerate(keys)}
    return np.asarray([pos[(int(a), float(v))] for a, v in zip(auth, vf)], np.int64)


class Examples:
    """Traced examples, batched by padding. Edges inside an example are sorted by rank, highest first,
    exactly as ease.aggregate.pack sorts them at inference."""

    def __init__(self, tr: dict, table: EdgeTable, device: torch.device, dtype=torch.float32):
        self.n = len(tr["label"])
        self.label = torch.as_tensor(tr["label"], device=device)
        self.prior = torch.as_tensor(tr["prior"], dtype=dtype, device=device)
        self.episode = tr["episode"]
        self.changed = tr["changed"]
        self.logits_t = torch.as_tensor(table.logits, dtype=dtype, device=device)
        self.msgs_t = torch.as_tensor(table.msgs, dtype=dtype, device=device)
        self.device, self.dtype = device, dtype
        self.idx, self.rank, self.auth = [], [], []
        for e, a, v, rids in zip(tr["edges"], tr["authority"], tr["valid_from"], tr["record_ids"]):
            if len(e) == 0:
                self.idx.append(e); self.rank.append(e); self.auth.append(e)
                continue
            r = dense_rank_rows(a, v)
            # by rank, ties by record id: the same order ease.aggregate.pack uses at inference
            order = np.asarray(sorted(range(len(e)), key=lambda i: (int(r[i]), rids[i])), np.int64)
            self.idx.append(e[order]); self.rank.append(r[order]); self.auth.append(a[order])
        self.size = np.asarray([len(e) for e in self.idx])

    def batch(self, ix: np.ndarray):
        B, N = len(ix), int(max(1, self.size[ix].max()))
        idx = torch.zeros(B, N, dtype=torch.long)
        rank = torch.zeros(B, N, dtype=torch.long)
        auth = torch.zeros(B, N, dtype=self.dtype)
        mask = torch.zeros(B, N, dtype=torch.bool)
        for b, i in enumerate(ix):
            n = self.size[i]
            if n:
                idx[b, :n] = torch.from_numpy(self.idx[i])
                rank[b, :n] = torch.from_numpy(self.rank[i])
                auth[b, :n] = torch.from_numpy(self.auth[i]).to(self.dtype)
                mask[b, :n] = True
        idx, rank, auth, mask = idx.to(self.device), rank.to(self.device), auth.to(self.device), mask.to(self.device)
        m = mask.unsqueeze(-1).to(self.dtype)
        return self.logits_t[idx] * m, self.msgs_t[idx] * m, rank, auth, mask, self.label[torch.as_tensor(ix, device=self.device)]


def forward(kind: str, model, batch) -> torch.Tensor:
    logits, msg, rank, auth, mask, _ = batch
    if kind == "refined":
        return precedence(model(logits, msg, mask), rank, mask)
    return model(logits, msg, rank, auth, mask)


@torch.no_grad()
def evaluate(kind: str, model, ex: Examples, ix: np.ndarray, bs: int = 2048) -> dict:
    model.eval()
    nll, correct, n = 0.0, 0, 0
    nll_c, correct_c, n_c = 0.0, 0, 0
    order = ix[np.argsort(ex.size[ix], kind="stable")]
    for s in range(0, len(order), bs):
        b = order[s : s + bs]
        batch = ex.batch(b)
        p = forward(kind, model, batch)
        y = batch[-1]
        ll = -torch.log(p.gather(1, y.unsqueeze(1)).squeeze(1).clamp_min(1e-12))
        hit = (p.argmax(1) == y)
        nll += float(ll.sum()); correct += int(hit.sum()); n += len(b)
        ch = torch.as_tensor(ex.changed[b], device=y.device)
        if ch.any():
            nll_c += float(ll[ch].sum()); correct_c += int(hit[ch].sum()); n_c += int(ch.sum())
    return {"n": n, "nll": nll / max(1, n), "accuracy": correct / max(1, n),
            "n_changed": n_c, "nll_changed": nll_c / max(1, n_c), "accuracy_changed": correct_c / max(1, n_c)}


def fit_temperature_through_policy(ex: Examples, ix: np.ndarray) -> Calibration:
    """Temperature and class bias for the rule aggregator, fitted on the same validation examples by
    minimising status NLL through the exact policy. This is the strongest version of 'calibrated
    rules': it is given the same supervision as the learned models."""
    log_t = torch.zeros((), dtype=ex.dtype, device=ex.device, requires_grad=True)
    bias = torch.zeros(3, dtype=ex.dtype, device=ex.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t, bias], lr=0.5, max_iter=60, line_search_fn="strong_wolfe")
    order = ix[np.argsort(ex.size[ix], kind="stable")]
    batches = [ex.batch(order[s : s + 4096]) for s in range(0, len(order), 4096)]

    def closure():
        opt.zero_grad()
        tot = 0.0
        for logits, msg, rank, auth, mask, y in batches:
            p = precedence(F.softmax(logits / torch.exp(log_t) + bias, -1), rank, mask)
            tot = tot + (-torch.log(p.gather(1, y.unsqueeze(1)).squeeze(1).clamp_min(1e-12))).sum()
        tot = tot / len(order)
        tot.backward()
        return tot

    opt.step(closure)
    b = (bias - bias.mean()).detach().cpu().tolist()
    return Calibration(float(torch.exp(log_t).item()), tuple(float(x) for x in b))


def train_one(kind: str, ex: Examples, train_ix, val_ix, msg_dim: int, seed: int, epochs: int, lr: float,
              batch_size: int, log: JsonlLogger, hidden: int):
    seed_everything(seed)
    if kind == "refined":
        model = EdgeRefiner(RefinerConfig(msg_dim=msg_dim, hidden=hidden))
    else:
        model = RankedReader(NeuralConfig(msg_dim=msg_dim, hidden=hidden))
    model.to(ex.device).to(ex.dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_total = epochs * int(np.ceil(len(train_ix) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps_total, pct_start=0.1)
    rng = np.random.default_rng(seed)
    best, best_state, step = None, None, 0
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train_ix)
        # batches of similar set size keep padding small; batch order is shuffled
        chunks = [perm[s : s + batch_size * 50] for s in range(0, len(perm), batch_size * 50)]
        running, seen = 0.0, 0
        for ch in chunks:
            ch = ch[np.argsort(ex.size[ch], kind="stable")]
            bs = [ch[s : s + batch_size] for s in range(0, len(ch), batch_size)]
            for bi in rng.permutation(len(bs)):
                batch = ex.batch(bs[bi])
                p = forward(kind, model, batch)
                loss = -torch.log(p.gather(1, batch[-1].unsqueeze(1)).squeeze(1).clamp_min(1e-12)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); step += 1
                running += float(loss) * len(bs[bi]); seen += len(bs[bi])
        val = evaluate(kind, model, ex, val_ix)
        log.log(kind=kind, seed=seed, epoch=epoch, train_nll=running / max(1, seen), **{f"val_{k}": v for k, v in val.items()})
        print(f"    {kind} seed={seed} epoch={epoch} train_nll={running/max(1,seen):.4f} "
              f"val_nll={val['nll']:.4f} val_acc={val['accuracy']:.4f} val_acc_changed={val['accuracy_changed']:.4f}",
              flush=True)
        if best is None or val["nll"] < best["nll"]:
            best = dict(val, epoch=epoch)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge", default="runs/stage_a/final")
    ap.add_argument("--atoms", default="runs/atoms")
    ap.add_argument("--out", default="runs/stage_b")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--episode-seed", type=int, default=20260901)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(out / "train_log.jsonl")
    atoms = load_pool(a.atoms, "stage_b")
    t0 = time.time()
    episodes = generate(atoms, a.episodes, a.episode_seed, STANDARD, tag="stageb")
    print(f"generated {len(episodes)} episodes in {time.time()-t0:.1f}s", flush=True)
    atomic_write_json(out / "episodes_summary.json", describe(episodes))

    table_dir = out / "table"
    if (table_dir / "table.json").exists():
        table = EdgeTable.load(table_dir)
        print(f"loaded edge table: {len(table.keys)} pairs, version {table.version}", flush=True)
    else:
        from ease.scorer import ModelScorer

        scorer = ModelScorer(a.edge, batch_size=128)
        t0 = time.time()
        table = build_table(episodes, scorer)
        table.save(table_dir)
        print(f"edge table built in {(time.time()-t0)/60:.1f} min; tokens={int(table.tokens.sum())}", flush=True)
        del scorer

    t0 = time.time()
    tr = trace(episodes, table)
    print(f"traced {len(tr['label'])} (step, predicate) examples in {time.time()-t0:.1f}s; "
          f"label counts={np.bincount(tr['label'], minlength=4).tolist()} "
          f"mean edges={np.mean([len(e) for e in tr['edges']]):.1f}", flush=True)

    device = torch.device(a.device)
    ex = Examples(tr, table, device)
    n_val_eps = max(1, int(round(a.val_frac * len(episodes))))
    val_mask = ex.episode >= (len(episodes) - n_val_eps)
    train_ix, val_ix = np.where(~val_mask)[0], np.where(val_mask)[0]
    print(f"train examples={len(train_ix)} val examples={len(val_ix)} (split by episode)", flush=True)

    summary = {"edge_version": table.version, "episodes": len(episodes), "episode_seed": a.episode_seed,
               "val_frac": a.val_frac, "val_episodes": int(n_val_eps), "train_examples": int(len(train_ix)),
               "val_examples": int(len(val_ix)), "pairs": len(table.keys), "runs": []}

    cal = fit_temperature_through_policy(ex, val_ix)
    atomic_write_json(out / "rule_calibration.json", {"temperature": cal.temperature, "bias": list(cal.bias)})
    print(f"rule calibration through the policy: T={cal.temperature:.4f} bias={[round(b,4) for b in cal.bias]}", flush=True)

    for kind in ("refined", "neural"):
        for seed in a.seeds:
            t0 = time.time()
            model, best = train_one(kind, ex, train_ix, val_ix, table.msg_dim, seed, a.epochs, a.lr, a.batch_size,
                                    log, a.hidden)
            d = out / f"{kind}_seed{seed}"
            model = model.cpu()
            agg = RefinedAggregator(model) if kind == "refined" else NeuralAggregator(model)
            agg.save(d, extra={"edge_version": table.version, "seed": seed, "val": best})
            n_params = sum(p.numel() for p in model.parameters())
            summary["runs"].append({"kind": kind, "seed": seed, "dir": str(d), "params": n_params, "val": best,
                                    "seconds": round(time.time() - t0, 1), "version": agg.version})
            print(f"  saved {d} params={n_params} best={json.dumps({k: round(v,4) if isinstance(v,float) else v for k,v in best.items()})}", flush=True)
    atomic_write_json(out / "summary.json", summary)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
