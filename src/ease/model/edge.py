"""The edge function: what does one evidence passage establish about one claim?

    f(claim, evidence) -> (relation logits over SUPPORTS / REFUTES / NEI, message vector)

It is a cross-encoder: claim and evidence are read jointly, because whether a passage supports a
claim usually depends on words in both ("more than 540,000" against "545,500"). The message vector
is the representation immediately before the classifier. Downstream components consume it, and the
self-evolution memory uses it as a retrieval key.

An edge value depends only on (weights version, claim text, evidence text). That is what makes it
cacheable, and it is why a change to one evidence record invalidates only the edges touching it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ease.labels import NEI, NOT_SUPPORTS, NUM_RELATIONS, REFUTES


@dataclass
class EdgeConfig:
    encoder_name: str = "answerdotai/ModernBERT-base"
    msg_dim: int = 128
    dropout: float = 0.1
    pooling: str = "cls"  # "cls" or "mean"
    max_len: int = 256


class EdgeModel(nn.Module):
    def __init__(self, cfg: EdgeConfig, load_pretrained: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.cfg = cfg
        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(cfg.encoder_name))
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden, cfg.msg_dim)
        self.norm = nn.LayerNorm(cfg.msg_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.msg_dim, NUM_RELATIONS)

    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "cls":
            return hidden[:, 0]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        message = self.norm(F.gelu(self.proj(self.pool(hidden, attention_mask))))
        logits = self.classifier(self.drop(message))
        return logits, message

    # -- persistence -------------------------------------------------------
    def save(self, directory: str | Path, extra: Optional[dict] = None) -> None:
        from safetensors.torch import save_file

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().to("cpu").contiguous() for k, v in self.state_dict().items()}
        tmp = d / "model.safetensors.tmp"
        save_file(state, str(tmp))
        tmp.replace(d / "model.safetensors")
        (d / "edge_config.json").write_text(json.dumps({"edge": asdict(self.cfg), "extra": extra or {}}, indent=2))

    @classmethod
    def load(cls, directory: str | Path, device: Optional[torch.device] = None) -> "EdgeModel":
        from safetensors.torch import load_file

        d = Path(directory)
        meta = json.loads((d / "edge_config.json").read_text())
        model = cls(EdgeConfig(**meta["edge"]), load_pretrained=False)
        state = load_file(str(d / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=True)
        if device is not None:
            model.to(device)
        return model


def relation_loss(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy for three-way labels.

    Rows labelled NOT_SUPPORTS come from corpora that only say "not entailed". For those the loss is
    -log(p(REFUTES) + p(NEI)): the model is told the claim is not supported without being told which
    of the two alternatives holds.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    loss = logits.new_zeros(labels.shape[0], dtype=torch.float32)
    three = labels < NOT_SUPPORTS
    if three.any():
        loss[three] = F.cross_entropy(
            logits[three].float(), labels[three], reduction="none", label_smoothing=label_smoothing
        )
    ns = labels == NOT_SUPPORTS
    if ns.any():
        loss[ns] = -torch.logsumexp(logp[ns][:, [REFUTES, NEI]], dim=-1)
    return loss.mean()
