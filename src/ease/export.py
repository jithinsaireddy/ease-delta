"""The edge model as a plain `transformers` sequence classifier.

The edge model reads `[CLS] claim [SEP] evidence [SEP]` and classifies the [CLS] vector with

    Linear(H -> m) -> GELU -> LayerNorm(m) -> Dropout -> Linear(m -> 3)          (m = 128)

`ModernBertForSequenceClassification` classifies it with

    Linear(H -> H) -> activation -> LayerNorm(H, no bias) -> Dropout -> Linear(H -> labels)

The first is written exactly as the second, so the classifier loads with `AutoModelForSequenceClassification`,
`pipeline("text-classification")` and sentence-transformers' `CrossEncoder`, with no custom code. Because H is
a multiple of m (768 = 6 x 128, 1024 = 8 x 128), the dense layer can write k = H / m identical copies of the
m projected values. A vector made of k copies of u has the same mean and variance as u, so the LayerNorm over
H normalises every copy exactly as the LayerNorm over m normalises u. The first copy carries the trained
LayerNorm weight; the classifier reads only that copy; the trained LayerNorm bias, which ModernBERT's head
has no slot for, is folded into the classifier's bias (W (z + b_norm) + b = W z + (b + W b_norm)).

What is dropped: the 128-dimensional message vector, which only the learned refiner reads. The released
system uses calibrated rules, which read the logits alone.
"""

from __future__ import annotations

import copy

import torch

LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")


def to_sequence_classifier(edge, max_len: int | None = None):
    """A `ModernBertForSequenceClassification` computing the edge model's logits."""
    from transformers import ModernBertForSequenceClassification

    enc_cfg = edge.encoder.config
    H, m = enc_cfg.hidden_size, edge.proj.out_features
    if H % m:
        raise ValueError(f"hidden size {H} is not a multiple of the message size {m}")
    if edge.cfg.pooling != "cls":
        raise ValueError("only [CLS] pooling maps onto ModernBERT's classifier as written")
    k = H // m
    cfg = copy.deepcopy(enc_cfg)
    cfg.num_labels = len(LABELS)
    cfg.id2label = dict(enumerate(LABELS))
    cfg.label2id = {v: i for i, v in enumerate(LABELS)}
    cfg.classifier_pooling = "cls"
    cfg.classifier_bias = True
    cfg.classifier_activation = "gelu"  # ACT2FN["gelu"] is the exact (erf) GELU, as F.gelu is
    cfg.classifier_dropout = 0.0
    if abs(cfg.norm_eps - edge.norm.eps) > 0:
        raise ValueError(f"LayerNorm eps differs: encoder config {cfg.norm_eps}, edge head {edge.norm.eps}")
    cfg.architectures = ["ModernBertForSequenceClassification"]
    if max_len:
        cfg.ease_max_length = int(max_len)

    model = ModernBertForSequenceClassification(cfg).eval()
    model.model.load_state_dict(edge.encoder.state_dict(), strict=True)
    with torch.no_grad():
        head = model.head
        head.dense.weight.copy_(edge.proj.weight.repeat(k, 1))
        head.dense.bias.copy_(edge.proj.bias.repeat(k))
        head.norm.weight.zero_()
        head.norm.weight[:m].copy_(edge.norm.weight)
        if head.norm.bias is not None:
            head.norm.bias.zero_()
        model.classifier.weight.zero_()
        model.classifier.weight[:, :m].copy_(edge.classifier.weight)
        model.classifier.bias.copy_(edge.classifier.bias + edge.classifier.weight @ edge.norm.bias)
    return model
