"""Baseline B2: a dense reader.

It answers the same question as the edge model plus aggregator, in one pass: given a claim and
*every* active record of the task, what is the claim's status? Records are written into the
context in rank order with their authority and effective time, so everything the precedence
policy needs is visible to the model. The policy itself has to be learned.

This is how a conventional decision model would be used behind the same ledger: state in, decision
out, recomputed when the state changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DenseConfig:
    encoder_name: str = "answerdotai/ModernBERT-base"
    hidden: int = 128
    dropout: float = 0.1
    max_len: int = 1024


def render_context(records: Sequence[dict]) -> str:
    """records: dicts with text, authority, valid_from; any order. Rendered highest rank first."""
    ordered = sorted(records, key=lambda r: (-r["authority"], -r["valid_from"], r["record_id"]))
    if not ordered:
        return "(no records)"
    return " ".join(f"[record {i + 1} | authority {r['authority']} | time {r['valid_from']:g}] {r['text']}"
                    for i, r in enumerate(ordered))


class DenseModel(nn.Module):
    def __init__(self, cfg: DenseConfig, load_pretrained: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.cfg = cfg
        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(cfg.encoder_name))
        h = self.encoder.config.hidden_size
        self.proj = nn.Linear(h, cfg.hidden)
        self.norm = nn.LayerNorm(cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden, 4)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        z = self.norm(F.gelu(self.proj(hidden[:, 0])))
        return self.classifier(self.drop(z))

    def init_from_edge(self, edge_dir: str | Path) -> dict:
        """Start from the Stage A weights: same encoder, same projection. Only the 4-way head is new."""
        from safetensors.torch import load_file

        state = load_file(str(Path(edge_dir) / "model.safetensors"))
        own = self.state_dict()
        taken = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape and not k.startswith("classifier.")}
        self.load_state_dict(taken, strict=False)
        return {"copied_tensors": len(taken), "total_tensors": len(own)}

    def save(self, directory: str | Path, extra: Optional[dict] = None) -> None:
        from safetensors.torch import save_file

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().to("cpu").contiguous() for k, v in self.state_dict().items()}
        tmp = d / "model.safetensors.tmp"
        save_file(state, str(tmp))
        tmp.replace(d / "model.safetensors")
        (d / "dense_config.json").write_text(json.dumps({"dense": asdict(self.cfg), "extra": extra or {}}, indent=2))

    @classmethod
    def load(cls, directory: str | Path, device: Optional[torch.device] = None) -> "DenseModel":
        from safetensors.torch import load_file

        d = Path(directory)
        meta = json.loads((d / "dense_config.json").read_text())
        m = cls(DenseConfig(**meta["dense"]), load_pretrained=False)
        m.load_state_dict(load_file(str(d / "model.safetensors")), strict=True)
        if device is not None:
            m.to(device)
        return m


def encode_contexts(tok, claims: Sequence[str], contexts: Sequence[str], max_len: int) -> list[list[int]]:
    enc = tok(list(claims), list(contexts), truncation="longest_first", max_length=max_len, padding=False,
              return_attention_mask=False, return_token_type_ids=False)
    return enc["input_ids"]


@torch.inference_mode()
def predict_contexts(model: DenseModel, tok, claims, contexts, device, max_len: int, token_budget: int = 16384,
                     pad_multiple: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits [n,4], token count per input). Batches are sized by a token budget because
    context lengths vary by an order of magnitude."""
    from ease.data.streams import pad_batch
    from ease.model.infer import round_up

    model.eval()
    ids = encode_contexts(tok, claims, contexts, max_len)
    n = len(ids)
    out = np.zeros((n, 4), np.float32)
    lens = np.asarray([len(x) for x in ids], np.int32)
    order = sorted(range(n), key=lambda i: len(ids[i]))
    s = 0
    while s < n:
        L = round_up(len(ids[order[s]]), pad_multiple)
        e = s + 1
        while e < n and (e - s + 1) * round_up(len(ids[order[e]]), pad_multiple) <= token_budget and e - s < 128:
            e += 1
        ix = order[s:e]
        L = min(max_len, round_up(max(len(ids[i]) for i in ix), pad_multiple))
        inp, mask = pad_batch([ids[i] for i in ix], tok.pad_token_id, pad_to=L)
        lg = model(inp.to(device), mask.to(device)).float().cpu().numpy()
        for j, i in enumerate(ix):
            out[i] = lg[j]
        s = e
    return out, lens
