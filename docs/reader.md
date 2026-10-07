# The reader on its own

The reader answers one question: **what does this passage establish about this claim?** `SUPPORTS`, `REFUTES` or
`NOT_ENOUGH_INFO`. It is a ModernBERT cross-encoder fine-tuned on VitaminC (claims against Wikipedia revisions),
MultiNLI and WANLI, with 20% of training pairs made from a claim and a passage about something else, labelled
`NOT_ENOUGH_INFO`. That last part is what it is for: in the same evaluation, passages from unrelated documents were
read as decisive 0.58% of the time by the large reader and 45% of the time by
[`tasksource/ModernBERT-base-nli`](https://huggingface.co/tasksource/ModernBERT-base-nli).

| Checkpoint | Parameters | Use with |
|---|---|---|
| [`jithinpothireddy21/ease-delta-reader`](https://huggingface.co/jithinpothireddy21/ease-delta-reader) | 395M | `transformers`, sentence-transformers |
| [`jithinpothireddy21/ease-delta-reader-base`](https://huggingface.co/jithinpothireddy21/ease-delta-reader-base) | 149M | the same, plus ONNX (fp32 and int8) for onnxruntime and Transformers.js |

Both are exact conversions of the readers inside the full system (largest logit difference about 1e-6 over 4,000
development pairs; `ease/export.py` explains the conversion).

## Rules of use

- **Claim first, evidence second.** (NLI models usually take the premise first; this one does not.)
- **Short passages.** It was trained on pairs of at most 256 tokens; `truncation=True` cuts there. Split documents
  into paragraphs and read each.
- **Say who is speaking, in the passage.** It reads only text: "Email from the client: we approve" can establish
  that the client approved; "we approve" cannot.
- **Calibration.** Probabilities are best calibrated after dividing the logits by 1.203 (large) or 1.227 (base)
  before the softmax.

## transformers

```python
from transformers import pipeline

reader = pipeline("text-classification", model="jithinpothireddy21/ease-delta-reader", top_k=None)
reader({"text": "The film grossed more than $550 million worldwide.",
        "text_pair": "The film went on to gross $545.5 million worldwide."}, truncation=True)
```

Or by hand, with calibrated probabilities:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "jithinpothireddy21/ease-delta-reader"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()
claims = ["The client has approved the final design."] * 2
passages = ["Email from the client: we approve the final design.", "The office is closed on Monday."]
with torch.no_grad():
    logits = model(**tok(claims, passages, truncation=True, padding=True, return_tensors="pt")).logits
probs = torch.softmax(logits / 1.203, dim=-1)       # columns: SUPPORTS, REFUTES, NOT_ENOUGH_INFO
```

Loads with transformers 4.x and 5.x.

## sentence-transformers

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("jithinpothireddy21/ease-delta-reader")
probs = model.predict([(claim, passage) for passage in passages], apply_softmax=True)
```

## ONNX (base reader)

```python
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

repo = "jithinpothireddy21/ease-delta-reader-base"
tok = AutoTokenizer.from_pretrained(repo)
session = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx"))     # or onnx/model_quantized.onnx
enc = tok("The deposit invoice has been paid.", "Office notice: the kitchen will be closed on Friday.",
          truncation=True, max_length=256, return_tensors="np")
logits = session.run(None, {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask")})[0]
```

The fp32 ONNX model matches PyTorch to 4e-5 in the logits with no label differing on 2,000 development pairs. The int8
model is a quarter of the size (150 MB) and gave the same label as the full model on 98.1% of 3,000 development pairs
(accuracy 88.5% against 89.2%).

## In the browser (Transformers.js)

```js
import { AutoTokenizer, AutoModelForSequenceClassification } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

const repo = "jithinpothireddy21/ease-delta-reader-base";
const tokenizer = await AutoTokenizer.from_pretrained(repo);
const model = await AutoModelForSequenceClassification.from_pretrained(repo, { dtype: "q8" });
const inputs = await tokenizer("The deposit invoice has been paid.",
                               { text_pair: "Office notice: the kitchen will be closed on Friday.", truncation: true, max_length: 256 });
const { logits } = await model(inputs);   // SUPPORTS, REFUTES, NOT_ENOUGH_INFO
```

This is what the [demo page](https://jithinsaireddy.github.io/ease-delta/) runs; the text never leaves the browser.

## Checking an answer against several passages

A claim checked against several retrieved passages needs a rule for combining them: which passage wins when they
disagree, what an irrelevant passage counts for, what happens when one is withdrawn. That is what the full system
does exactly, with ranks for authority and time ([concepts](concepts.md#records)). For one claim:

```python
from ease import EvidenceReader, Tracker

reader = EvidenceReader.from_pretrained()
check = Tracker.define({"claim": "The Eiffel Tower is 330 metres tall."}, {"accept": "claim"}, reader)
for i, passage in enumerate(retrieved_passages):
    check.add(f"p{i}", passage, valid_from=0)     # equal rank: passages that disagree show as CONFLICT
print(check.status().requirements["claim"])      # SATISFIED / VIOLATED / UNRESOLVED / CONFLICT, with probabilities
```

Without `valid_from=0` each passage would rank by when it was added, and the latest decisive one would decide.

## Results

Test sets, accuracy, from [RESULTS.md](RESULTS.md) (sections 2 and 8l):

| | Reader (large) | Reader (base) | `tasksource/ModernBERT-base-nli` |
|---|---:|---:|---:|
| Unrelated passages read as decisive (lower is better) | 0.58% | 0.82% | 45.07% |
| VitaminC | 91.53% | 90.20% | 68.30% |
| MultiNLI, mismatched | 90.21% | 88.56% | 89.84% |
| WANLI | 77.42% | 74.74% | 66.34% |
| SNLI (not trained on) | 84.39% | 80.33% | 89.15% |
| ANLI r1 / r2 / r3 (not trained on) | 53.7 / 38.1 / 36.0 | 46.4 / 33.3 / 34.8 | 63.4 / 48.2 / 42.3 |

The public model's training includes SNLI and ANLI; for adversarial NLI of the ANLI kind it is the better choice.

## Limits

English only. Passages on the claim's own subject that do not settle it are still read as decisive about one time in
five. Conflicts that follow only from a consequence are mostly missed. Numbers and dates are read as text. It reports
what a passage says, not whether the passage is true.
