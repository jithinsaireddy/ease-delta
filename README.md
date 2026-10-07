<div align="center">

# EASE-Delta

**Keep decisions current as the facts change.**

[![tests](https://github.com/jithinsaireddy/ease-delta/actions/workflows/tests.yml/badge.svg)](https://github.com/jithinsaireddy/ease-delta/actions/workflows/tests.yml)
[![demo](https://img.shields.io/badge/demo-in_your_browser-3b4fd8)](https://jithinsaireddy.github.io/ease-delta/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jithinsaireddy/ease-delta/blob/main/examples/try_ease_delta.ipynb)
[![models](https://img.shields.io/badge/%F0%9F%A4%97%20models-ease--delta-ffcc4d)](https://huggingface.co/jithinpothireddy21/ease-delta)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/jithinsaireddy/ease-delta/blob/main/pyproject.toml)
[![code licence](https://img.shields.io/badge/code-Apache--2.0-blue)](https://github.com/jithinsaireddy/ease-delta/blob/main/LICENSE)
[![weights licence](https://img.shields.io/badge/weights-CC_BY--SA_4.0-lightgrey)](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/licenses/CC-BY-SA-4.0.txt)

</div>

You say what must be true before something can be done: *the client approved the design*, *the
delivery date is confirmed*, *the NDA is signed*. Messages arrive, get corrected, get withdrawn.
EASE-Delta reads each message against each requirement once, re-reads only what a change touches,
and after every change tells you **what is ready, what is blocked and why, and which single
question is worth asking**. It proposes; it never sends, pays or executes anything.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jithinsaireddy/ease-delta/main/docs/assets/story-dark.png">
  <img src="https://raw.githubusercontent.com/jithinsaireddy/ease-delta/main/docs/assets/story-light.png" alt="A client hand-off, step 7 of 12: the client withdraws the approval, and send_packet goes from needs info to blocked, resting on that email" width="860">
</picture>
</p>

## Try it

| Where | What you get | Install |
|---|---|---|
| **[In your browser](https://jithinsaireddy.github.io/ease-delta/)** (also on [Hugging Face](https://huggingface.co/spaces/jithinpothireddy21/ease-delta-demo)) | the reader, running on your own machine; a recorded run of a full task | nothing |
| **[In Colab](https://colab.research.google.com/github/jithinsaireddy/ease-delta/blob/main/examples/try_ease_delta.ipynb)** | the whole system, step by step, and the interactive demo | nothing |
| **On your computer** | everything, with your own messages | `pip install "ease-delta @ git+https://github.com/jithinsaireddy/ease-delta"` |

## Ten lines

```python
from ease import EvidenceReader, Tracker

reader = EvidenceReader.from_pretrained()          # downloads jithinpothireddy21/ease-delta once
task = Tracker.define(
    requirements={"approved": "Acme has approved the final design.",
                  "date": "Acme has confirmed the delivery date.",
                  "nda": "The non-disclosure agreement has been signed by both parties."},
    actions={"send_packet": "approved and date and nda"},
    reader=reader)
task.add("mail-1", "Email from Acme: we approve the final design, please go ahead.", about=["approved"])
task.add("mail-2", "Email from Acme: we confirm delivery on 12 March.", about=["date"])
task.add("nda", "The NDA has been signed by Acme and countersigned by us.", about=["nda"])
print(task.status())
```

```
Actions
  send_packet   NEEDS INFO   86.3% likely ready
Requirements
  approved      SATISFIED   satisfied 0.95  violated 0.00  open 0.04  rests on mail-1
  date          SATISFIED   satisfied 0.96  violated 0.00  open 0.04  rests on mail-2
  nda           SATISFIED   satisfied 0.94  violated 0.01  open 0.05  rests on nda
Worth asking: The non-disclosure agreement has been signed by both parties.
```

Three likely requirements are not a sure thing, so it does not call the packet ready (the default
bar is 90%); it names the one answer that would settle it. Then people answer, and change their minds:

```python
task.confirm("nda", by="Priya in Legal")           # send_packet: NEEDS_INFO -> READY
task.add("mail-3", "Email from Acme: we withdraw our approval; the colours are wrong.", about=["approved"])
                                                   # send_packet: READY -> BLOCKED
print(task.explain("send_packet"))                 # ...resting on mail-3, revision 1
```

That output is real: [`examples/quickstart.py`](https://github.com/jithinsaireddy/ease-delta/blob/main/examples/quickstart.py) runs it. Start from a
template instead of writing requirements: `Tracker.from_template("client-onboarding", reader=reader, client="Acme")`.

## Workflows it fits

| Workflow | Requirements it tracks | It tells you |
|---|---|---|
| Client onboarding | brief received, files supplied, access granted, kickoff agreed | "Work can start once website access is confirmed." |
| Campaign launch | copy and artwork approved, materials in, compliance cleared | "The revised artwork still needs approval." |
| Event coordination | venue, equipment, schedule, at least *k* speakers confirmed | "A speaker cancelled; you still need one more." |
| Support follow-up | fix or replacement delivered, customer confirmed | "Replacement arrived; the customer has not confirmed." |
| Software release | test report passed, notes complete, sign-offs, rollback plan | "The new build needs a fresh test report." |
| Group trips, shared projects | everyone confirmed, everyone paid, bookings made | "Everyone confirmed except one." |

Each is a ready-made template (`ease.templates`). Any workflow that can be written as *what must be
true before what* works the same way; how well its messages are read is measured per kind of
text, not assumed (see [limitations](#what-it-does-not-do-yet)).

## Only the reader

The reader is also published as a standard `transformers` model, for checking a claim against a
passage anywhere: RAG answer verification, fact-checking, triage. It answers `SUPPORTS`,
`REFUTES` or `NOT_ENOUGH_INFO`, and it was trained to say `NOT_ENOUGH_INFO` when a passage is about
something else.

```python
from transformers import pipeline

reader = pipeline("text-classification", model="jithinpothireddy21/ease-delta-reader", top_k=None)
reader({"text": "The deposit invoice has been paid.",                       # claim first
        "text_pair": "Office notice: the kitchen will be closed on Friday."},  # evidence second
       truncation=True)
# [{'label': 'NOT_ENOUGH_INFO', 'score': 0.999}, ...]
```

It also loads in sentence-transformers' `CrossEncoder`, and the base model has ONNX weights for
onnxruntime and Transformers.js. See [the reader guide](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/reader.md).

## Why it is built this way

- **It re-reads only what changed.** Readings are cached per (requirement, message); a change
  re-reads the pairs it touches, about 2 to 7% of the tokens of re-reading the task. A duplicate or an
  old copy costs nothing.
- **Its cached state is exact.** It equals a full rebuild bit for bit (0 of 195,434 values
  differed), so the shortcut never drifts. A person's correction takes effect at once and can be
  undone exactly.
- **It knows when a message is irrelevant.** Passages from unrelated documents are read as
  evidence 0.6% of the time; a widely used public NLI model does so 45% of the time.
- **It explains, asks, and waits.** Every proposal shows the records it rests on, it asks only
  questions worth their cost, and a READY action is a proposal for a person to approve.

Learned parts read language; exact code handles versions, precedence, logic and planning:

```
message ──> ledger (exact)        revisions, withdrawals, duplicates, stale copies
              │
              ▼
          reader (learned)        one (requirement, message) pair -> supports / refutes / not enough info
              │   cached per pair
              ▼
          precedence + logic      which record counts; all of / any of / at least k of / not
              │   (exact)
              ▼
          planner (exact)         ready / blocked / needs info; the question worth asking
```

More in [concepts](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/concepts.md) and [design](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/DESIGN.md).

## Results

Measured on an Apple M4 Max against plans written and hashed before any test data was read
([`PREREGISTRATION.md`](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/PREREGISTRATION.md), [`PREREGISTRATION_LARGE.md`](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/PREREGISTRATION_LARGE.md)).
Tasks are generated; their text is human-written (Wikipedia revisions and NLI corpora). Hypotheses
that failed are reported as failed.

| | Shipped system |
|---|---|
| Correct ready / blocked / needs-info decisions | 91.6% on standard tasks, 86.8% on larger ones |
| Against a dense reader of the whole task (same backbone and training) | +2.2 points [+0.6, +3.9] and fewer stale decisions on standard tasks; +5.5 [+2.3, +8.7] on larger ones (base reader) |
| Larger reader against the base one | +2.0 points [+1.1, +2.9] and +2.2 [+0.7, +3.6], a little above the gain predicted before training |
| Update after a change | 59 ms on the Apple GPU, 0.2 to 0.6 s on its CPU (395M reader) |
| Reader on VitaminC test (claims against Wikipedia revisions) | 91.5% |
| Cached state against a full rebuild | 0 of 195,434 values differed |

Every number, with intervals: [results at a glance](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/results-summary.md) and the generated
[full report](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/RESULTS.md).

## What it does not do yet

- **It has not been measured with people.** Whether it saves anyone time is the next question
  ([study design](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/NEXT_STEPS.md)), not a claim.
- **English only**, short passages (256 tokens; documents are cut into attributed passages).
- **Messages on the same subject that settle nothing** are still read as decisive about one time
  in five. When several requirements are of one kind (one per speaker, say), tell it which one a
  message concerns with `about`.
- **"The latest version"** needs two records read together and is read unsafely; name the version.
- **Conflicts that follow from a consequence** ("broke a leg" against "cycles to work") are mostly missed.
- It reports what messages say, not whether they are true.

The full list: [LIMITATIONS.md](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/LIMITATIONS.md).

## Models

| Hugging Face | Size | For |
|---|---|---|
| [`jithinpothireddy21/ease-delta`](https://huggingface.co/jithinpothireddy21/ease-delta) | 395M, 1.6 GB | the system: `EvidenceReader.from_pretrained()` (default) |
| [`jithinpothireddy21/ease-delta-base`](https://huggingface.co/jithinpothireddy21/ease-delta-base) | 149M, 0.6 GB | the system, 2.7x faster, about 2 points less accurate |
| [`jithinpothireddy21/ease-delta-reader`](https://huggingface.co/jithinpothireddy21/ease-delta-reader) | 395M | the reader alone, plain `transformers` |
| [`jithinpothireddy21/ease-delta-reader-base`](https://huggingface.co/jithinpothireddy21/ease-delta-reader-base) | 149M | the reader alone; ONNX and in-browser |

## Documentation

| | |
|---|---|
| [Getting started](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/getting-started.md) | install, first task, templates, emails, confirmations, keeping state |
| [Concepts](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/concepts.md) | requirements, records, readings, statuses, questions, corrections |
| [Python API](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/python-api.md) | `EvidenceReader`, `Tracker`, requirement expressions, templates |
| [HTTP API](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/http-api.md) | every endpoint, with examples |
| [Reader guide](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/reader.md) | the reader with transformers, CrossEncoder, ONNX, the browser |
| [Deployment](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/DEPLOYMENT.md) | one person, or many teams behind keys; Docker; backups |
| [FAQ](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/faq.md) | accuracy, privacy, licences, GPUs, languages, how it learns |
| [Research](https://github.com/jithinsaireddy/ease-delta/blob/main/docs/README.md#research) | results, pre-registrations, deviations, proofs, design, sources |

## Contributing and citing

Issues and pull requests are welcome: see [CONTRIBUTING.md](https://github.com/jithinsaireddy/ease-delta/blob/main/CONTRIBUTING.md). The most useful
contribution right now is real (anonymised) correspondence where it reads wrongly.

```bibtex
@software{pothireddy2026easedelta,
  author = {Pothireddy, Jithin},
  title  = {EASE-Delta: revision-aware decision computation},
  year   = {2026},
  url    = {https://github.com/jithinsaireddy/ease-delta}
}
```

Code: Apache-2.0. Weights: CC BY-SA 4.0, because one training corpus (VitaminC) is share-alike.
No hosted model is called anywhere in this project, and no comparison with any commercial product
is claimed.
