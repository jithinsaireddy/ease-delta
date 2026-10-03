"""Small shared utilities: seeding, hashing, atomic writes, device selection, JSONL logging."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def text_hash(*parts: str) -> str:
    """Stable content hash. Parts are length-prefixed so ("ab","c") != ("a","bc")."""
    h = hashlib.sha256()
    for p in parts:
        b = p.encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def get_device(prefer: str | None = None) -> torch.device:
    """Explicit argument first, then the EASE_DEVICE environment variable, then the best available."""
    prefer = prefer or os.environ.get("EASE_DEVICE")
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_sync(device: torch.device) -> None:
    """Block until queued GPU work finishes, so wall-clock timings are honest."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


class JsonlLogger:
    """Append-only metrics log; every line is flushed so a crash loses nothing."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **fields: Any) -> None:
        fields.setdefault("time", round(time.time(), 3))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields, default=str) + "\n")
            f.flush()


def clip_grad_norm_fast(parameters, max_norm: float) -> float:
    """Global-norm gradient clipping.

    Equivalent to torch.nn.utils.clip_grad_norm_, written as one reduction per tensor followed by a
    single stack. On Apple MPS (torch 2.14) the stock utility took 214 ms for 149M parameters and
    this takes 5 ms, with norms agreeing to 7 significant digits (scratch/profile_clip.py).
    Returns the norm before clipping.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total = float(torch.stack([g.detach().pow(2).sum() for g in grads]).sum().sqrt())
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g.mul_(scale)
    return total
