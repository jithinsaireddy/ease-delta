"""Batched inference for the edge model."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ease.data.streams import encode_pairs, pad_batch


def round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


@torch.inference_mode()
def predict_pairs(
    model,
    tokenizer,
    claims: Sequence[str],
    evidences: Sequence[str],
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 128,
    pad_multiple: int = 8,
    fixed_len: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits [n,3], messages [n,d]) as float32 numpy arrays, in input order.

    Pairs are processed shortest-first to limit padding. With `fixed_len` every batch is padded to
    the same length; that removes padding-dependent numerical variation and is what the engine
    uses when it needs cached and recomputed values to agree bit for bit.
    """
    n = len(claims)
    if n == 0:
        d = model.cfg.msg_dim
        return np.zeros((0, 3), np.float32), np.zeros((0, d), np.float32)
    was_training = model.training
    model.eval()
    ids = encode_pairs(tokenizer, list(claims), list(evidences), max_len)
    order = sorted(range(n), key=lambda i: len(ids[i]))
    logits_out: list[Optional[np.ndarray]] = [None] * n
    msg_out: list[Optional[np.ndarray]] = [None] * n
    for s in range(0, n, batch_size):
        ix = order[s : s + batch_size]
        chunk = [ids[i] for i in ix]
        if fixed_len is not None:
            L = fixed_len
        else:
            L = min(max_len, round_up(max(len(c) for c in chunk), pad_multiple))
        input_ids, mask = pad_batch(chunk, tokenizer.pad_token_id, pad_to=L)
        lg, msg = model(input_ids.to(device), mask.to(device))
        lg = lg.float().cpu().numpy()
        msg = msg.float().cpu().numpy()
        for j, i in enumerate(ix):
            logits_out[i] = lg[j]
            msg_out[i] = msg[j]
    if was_training:
        model.train()
    return np.stack(logits_out).astype(np.float32), np.stack(msg_out).astype(np.float32)
