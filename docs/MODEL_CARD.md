---
license: cc-by-sa-4.0
language:
- en
base_model: answerdotai/ModernBERT-base
datasets:
- nyu-mll/multi_nli
- tals/vitaminc
- alisawuffles/WANLI
pipeline_tag: text-classification
tags:
- natural-language-inference
- fact-verification
- evidence
---

# Model card: EASE-Delta

Generated 2026-10-03T09:44:00+00:00 by `scripts/package_release.py`. Figures are read from `runs/results/`. Full results, including hypotheses that were not supported: `docs/RESULTS.md`.

## What it is

Two learned parts and an exact core.

- **Edge model**, 149,113,347 parameters, `answerdotai/ModernBERT-base` fine-tuned. Input: a claim and one evidence passage. Output: probabilities that the passage *supports* the claim, *refutes* it, or *settles nothing*, and a 128-dimensional message vector.
- **Calibrated rules**: each reading is rescaled by a fitted temperature (0.998) and class bias, then combined by the exact precedence policy. A learned refiner was trained and evaluated; it is not the default because: significantly worse than calibrated rules on SMALL. The rules are stored in refiner form with the learned correction set to zero, so gated consolidation can add one later if feedback shows it helps.
- **Exact core**: versioned ledger, dependency graph, precedence policy, requirement logic, planner. Not learned.

## Intended use

Tracking whether declared requirements of a task are met as documents and messages arrive, change and are withdrawn, and proposing what is ready, blocked or worth asking about. English text. Short passages (inputs are truncated to 256 tokens per pair).

## Not intended for

- Acting without a person's approval. The software has no means of executing an action.
- Decisions about people's health, legal status, employment, credit or safety.
- Arithmetic, unit conversion or date reasoning. Simple numeric comparisons of the kind found in the training data were read accurately; anything beyond them was not tested.
- Requirements that can only be settled by reading several records together.
- Languages other than English.

## Use

These files are read by the `ease` package: https://github.com/jithinsaireddy/ease-delta (`pip install git+https://github.com/jithinsaireddy/ease-delta`, then download this repository into a directory and point `--model` and `--aggregator` at its `edge/` and `aggregator/`).

```bash
ease demo  --model edge --aggregator aggregator        # a client hand-off, twelve events
ease serve --model edge --aggregator aggregator        # local API and page on 127.0.0.1:8791
```

```python
from ease.scorer import ModelScorer

scorer = ModelScorer("edge", canonical=True)
logits, message = scorer.score([("The client has approved the design.",
                                 "Email from the client: we approve the design as presented.")])[0]
# logits: supports, refutes, settles nothing
```

## Training data

| Source | Examples | Licence | Obtained from |
|---|---:|---|---|
| mnli | 243,178 | cc-by-3.0 / cc-by-sa-3.0 / mit / other (per genre) | [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) |
| unrelated | 179,460 | derived: texts drawn from the real sources above | made from the rows above |
| vitaminc | 359,978 | cc-by-sa-3.0 | [tals/vitaminc](https://huggingface.co/datasets/tals/vitaminc) |
| wanli | 117,384 | cc-by-4.0 | [alisawuffles/WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) |

Streamed from the Hugging Face Hub. `unrelated` pairs are synthetic: the claim of one example with the evidence of another from a different topic, labelled as settling nothing. ANLI (non-commercial licence) and SNLI were not trained on.

Credit, as the licences ask: VitaminC, by Tal Schuster, Adam Fisch and Regina Barzilay (NAACL 2021), built from Wikipedia revisions; MultiNLI, by Adina Williams, Nikita Nangia and Samuel Bowman (NAACL 2018); WANLI, by Alisa Liu, Swabha Swayamdipta, Noah Smith and Yejin Choi (EMNLP 2022). Rows were used as published, with whitespace normalised. Backbone: ModernBERT, by Benjamin Warner and colleagues (2024).

## Measured performance of the edge model

| Set | n | Accuracy | Calibration error (ECE) after temperature scaling |
|---|---:|---:|---:|
| anli_r1.test | 1,000 | 46.40% | 0.241 |
| anli_r2.test | 1,000 | 33.30% | 0.372 |
| anli_r3.test | 1,200 | 34.75% | 0.359 |
| mnli_mm.test | 9,832 | 88.56% | 0.027 |
| snli.test | 9,824 | 80.33% | 0.012 |
| unrelated.test | 6,000 | 99.18% | 0.008 |
| vitaminc.test | 55,197 | 90.20% | 0.024 |
| wanli.test | 5,000 | 74.74% | 0.030 |

Unrelated passages read as decisive: 0.82%.

## Measured performance of the whole system

The shipped configuration (calibrated rules, every record compared with every requirement) on generated tasks with human-written text. Definitions: `docs/RESULTS.md`, section 3.

| Test set | Disposition accuracy | Stale-decision rate | False-READY rate |
|---|---:|---:|---:|
| STANDARD | 89.52% | 13.05% | 2.97% |
| WIDE | 84.63% | 16.26% | 4.20% |

When records say which requirement they concern (`about` on an event, or documents filed per requirement), accuracy rises to 90.74% (STANDARD), 88.70% (WIDE).
With the optional dependency proposer it is 89.71% (STANDARD), 86.62% (WIDE) at 5-11 times less work; the proposer is off by default because it misses most conflicts that follow from a consequence (below).

Exactness: 0 of 195,434 cached values differed from a full rebuild (canonical mode).

- STANDARD: update after a changed record, median 16 ms (95th percentile 38 ms) on an Apple M4 Max; full re-reading 167 ms.
- WIDE: update after a changed record, median 36 ms (95th percentile 79 ms) on an Apple M4 Max; full re-reading 960 ms.
- Timed with the learned refiner. The shipped rules skip that network, so they do strictly less work per update; the encoder dominates either way.

By device (shipped configuration, bit-exact mode, same machine). Apple GPU: update median 23 ms on STANDARD and 48 ms on WIDE, 2.6 GB of memory; CPU, 4 threads: update median 75 ms on STANDARD and 223 ms on WIDE, 2.5 GB of memory; CPU, all cores: update median 108 ms on STANDARD and 295 ms on WIDE, 2.6 GB of memory.

## What limits accuracy

Reading. When every reading is replaced by the correct one, the exact core gives the right answer for every action at every step of the test tasks. The edge model reads the record written for a requirement correctly 88.8% of the time on STANDARD and 88.7% of the time on WIDE. Passages on a claim's own subject that do not settle it are read as settling it 22.3% of the time (VitaminC test), so expect cross-talk between requirements of one project unless records say which requirement they concern.

Training this model for 200,000 more examples on the same corpora did not help (development sets: mnli.dev -0.39 points, vitaminc.dev -0.48 points). A larger encoder was not trained.

## Known weakness: implicit conflicts

On STALE, at 5.0% false positives, the edge model recognised 77.0% of direct conflicts and 34.5% of conflicts that follow from a consequence. It should not be relied on to notice that a new fact undermines an old one unless the two are about the same thing in similar words.

## Limitations

These were written before the results were known and are not excuses for them. Items marked *(updated)* were revised afterwards to replace an expectation with what was measured.

**What was not measured**

* *People.* The blueprint's primary measure was human correction and completion time per verified
  task. No person took part in any experiment here. Whether this saves anyone time is unknown.
* *Other products.* No hosted model was called. Nothing here says how this compares with Jev, Kev,
  Nimble or any assistant.
* *Energy.* Not measured.
* *Languages.* English only. The backbone was pretrained on English and code.

**Where the evaluation is easier than real use**

* *Structure is generated.* Text in the episodes is human-written, but which record concerns which
  predicate, and how requirements combine, were produced by a generator. Real tasks are messier.
* *(updated)* *Distractors are off-topic.* Records unrelated to a predicate come from different Wikipedia pages
  or different NLI prompts. In a real project the distractors are other messages about the same
  project. The only same-topic hard negatives are VitaminC evidence versions labelled NEI, and
  on those the reader is much weaker: passages about a claim's subject that do not settle it were
  read as settling it 22% of the time, against 0.8% for passages on another subject (results,
  section 8j). Expect cross-talk between the requirements of one project. In a hand check of the
  service, an email withdrawing approval of a design visibly lowered the belief that the
  delivery date had been confirmed. Saying which requirement a record concerns (`about`) removes
  this; a person's correction of a single reading takes effect at once.
* *(updated)* *One record per question, and references such as "the latest version".* In the test sets each
  predicate is settled by single records. A requirement that needs two records read together is
  read **unsafely**, as measured in X1 (results, section 8b): for "the client has approved the
  latest version" with records "approved version 4" and "the latest is version 5", reading each
  record separately answered *satisfied* 82% of the time. Reading the two together
  (`join=True`) lowered that to 26%; the dense reader answered satisfied 54% of the time. X1 uses
  templated wording on one pattern, so the numbers are indicative, but the direction is clear.
  **Write requirements that name the specific thing** ("the client has approved version 5"), and
  let exact code resolve references such as "latest" or "current" before a requirement is read.
* *Domain.* Training text is Wikipedia sentences and NLI corpora. Email and contract language is
  out of distribution. The demo shows behaviour on such text; it is one scenario, not a test.
* *Attribution.* The reader sees a record's text and nothing else. A passage that does not say who
  is speaking cannot establish who approved what. Ingestion keeps an attribution in each passage
  only if it is given one.
* *Passages.* Documents are cut at paragraph and sentence boundaries. A statement and the
  condition that qualifies it can land in different passages.

**Assumptions that real data can break**

* A1 (predicates independent given the evidence) and A2 (reading errors independent across
  edges), `docs/PROOFS.md`. Two predicates settled by one sentence violate A1. Two records quoting
  the same sentence violate A2.
* The ledger assumes each source numbers its revisions consistently and that record identity is
  stable. It cannot detect that two differently named records are the same document.
* The endorsement bound (T5) assumes verdicts are truthful.
* The missed-link bound (T6) assumes new tasks resemble calibration tasks, and covers only links
  of the kind present in the calibration data.

**What the model is weak at by construction**

* *(updated)* Numbers and dates are read as text by a neural encoder, not compared by code. Measured on
  VitaminC test (results, section 2), claims containing a numeric comparison were read at least
  as accurately as claims without numbers, so this is not the weakness it was expected to be on
  that corpus. VitaminC's comparisons follow a few patterns ("more than 540,000" against
  "545,500"); other formats, units, arithmetic and date reasoning were not tested and should be
  handled by exact code when they matter.
* Consequences that need world knowledge ("broke my leg" against "cycles to work") are not taught
  by any training corpus used here.

**Found by the experiments** *(added after measurement)*

* *The learned refiner is fragile outside the conditions it was trained in.* One of three refiners
  trained by the same procedure lost 6.8 points on larger tasks; all three lose to calibrated rules
  when a predicate has few readings. The release therefore ships calibrated rules. The refiner is
  kept as a research component.
* *Self-evolution learned nothing it could verify as safe from calibrated rules.* The exact
  corrections and the threshold tracker work as specified; gated consolidation adopted no update
  in X4. Its value in real use is unmeasured.
* *Propagated conflicts are mostly missed.* The edge model recognised 35% of conflicts that follow
  from a consequence, against 77% of direct ones (STALE, at 5% false positives).
* *All measured error is reading error, and more of the same training does not reduce it.* With
  every reading correct and confident, the exact parts gave the right answer for every action at
  every step of every test task (section 8j). The edge model reads the record written for a
  requirement correctly 89% of the time (base) or 90.5% (large), and that figure limits the
  system. Training the base model for 200,000 more examples on the same corpora made it slightly
  worse on development data. A larger encoder, trained under a second plan, gained 2.0 and 2.2
  points on the two task sets (section 8l), close to the estimate made beforehand; it is what
  ships.
* *It is cautious on text unlike its training data.* In the hand-off demo, on email-like text, no
  action reached the default endorsement threshold of 0.9 (section 8h). That is the safe
  direction. It also means a person is still asked to confirm.

**What one run cannot show**

* Each edge model (149M and 395M parameters) and the dense baseline were trained once, with one
  seed, because each takes hours on this machine. Their run-to-run variation is unknown. The
  small aggregators were trained with three seeds.
* All timings are from one laptop with other applications running.

**Deployment**

* The HTTP API has no authentication and is meant for the local machine.
* Bit-exact reuse was measured on the Apple GPU (H1). On the CPU the shipped model's reading of a
  pair depended on the other pairs in its batch, by up to 6e-6, so the CPU uses batches of one
  (Deviations, item 15); with that, the full-rebuild check reports no differences on the CPU
  either, at the cost given in section 8k.
* Run on macOS with Apple silicon (Python 3.12) and, through the Dockerfile, on Linux (aarch64,
  CPU) where the test suite passes and the report regenerates. Windows, x86-64 and other Python
  versions are untested. Without a GPU, an update took about 0.1 seconds with the base reader
  and 0.2 to 0.6 seconds with the shipped larger reader on this machine's processor, with 2.5 and
  4.8 GB of memory (sections 8k and 8l); reading a new task of 13 requirements and 34 records in
  full takes about 22 seconds there. Slower processors were not measured.
* With `--multi`, workspaces are separated by key and by files; the model and its reading cache
  are shared. One process serves requests that need the model one at a time. Accounts,
  passwords and billing do not exist. None of this has been load-tested or audited.
* "Production-ready" here means: persistent, crash-safe storage, input validation, versioned and
  reversible model updates, an audit log, and tests. It does not mean audited for security or
  validated with users.

## Licence

Weights: cc-by-sa-4.0. Code: Apache-2.0. Backbone: Apache-2.0. Training corpora include share-alike licences (VitaminC, CC BY-SA 3.0; parts of MNLI). Creative Commons' 2025 guidance on AI training describes releasing a model trained on share-alike material under the same licence as the cautious course; whether weights are an adaptation of their training text is not settled. This is a description of the sources, not legal advice.

