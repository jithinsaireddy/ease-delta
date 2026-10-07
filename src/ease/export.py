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


def compatible_tokenizer(encoder_name: str, out_dir, model_max_length: int, check_against=None) -> None:
    """Write the backbone's own tokenizer files into `out_dir`, with `model_max_length` set.

    transformers 5 saves `"tokenizer_class": "TokenizersBackend"`, a name that transformers 4 and Transformers.js
    do not know, so a tokenizer saved by it fails to load there. The backbone publishes the same vocabulary with
    `"tokenizer_class": "PreTrainedTokenizerFast"`, which every version loads; fine-tuning does not change the
    vocabulary. If `check_against` names a directory with a working tokenizer, the two must give identical ids
    on a sample of texts, or nothing is written.
    """
    import json
    import shutil
    import tempfile
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    out = Path(out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            try:
                shutil.copy2(hf_hub_download(encoder_name, name), stage / name)
            except Exception:
                if name != "special_tokens_map.json":
                    raise
        cfg = json.loads((stage / "tokenizer_config.json").read_text())
        cfg["model_max_length"] = int(model_max_length)
        (stage / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))
        if check_against is not None:
            from transformers import AutoTokenizer

            a, b = AutoTokenizer.from_pretrained(str(check_against)), AutoTokenizer.from_pretrained(str(stage))
            samples = [("The client has approved the final design.", "Email from the client: we approve it, go ahead!"),
                       ("Ünïcode, numbers 545,500 and $1.2m", "  spaces\tand\nnew lines; emoji 🙂 and 中文"),
                       ("short", "x" * 3000)]
            for c, e in samples:
                ia = a(c, e, truncation=True, max_length=model_max_length)["input_ids"]
                ib = b(c, e, truncation=True, max_length=model_max_length)["input_ids"]
                if ia != ib:
                    raise ValueError(f"the backbone's tokenizer gives different ids for {c!r}")
        for f in stage.iterdir():
            shutil.copy2(f, out / f.name)
