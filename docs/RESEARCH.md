# Research record

Compiled 2026-09-29. Every source below was opened on that date and its existence, title, authors
and date confirmed at the source. Where a figure is quoted it is the source's own figure, not a
replication.

Pages were first read through an automatic summariser. It was wrong twice, and both errors were
caught only by reading the raw page:

* it reported that arXiv 2609.18935 "does not exist" (the server returned HTTP 200 and the title);
* it reported the Jev launch post as dated 28 September 2026, which is the site's build timestamp.
  The post's own date field is 2026-09-15, as the blueprint said.

Identifiers, titles and dates below were therefore checked against raw responses.

## 1. What the blueprint claimed, checked

The project started from a blueprint written by another assistant. Its 22 citations all exist, and
the facts checked against them hold (including the Jev launch date, 15 September 2026). Two of
its statements were open and are now settled:

| Blueprint statement | Now |
|---|---|
| "The user has an M4 Max; installed memory was not verified" | Verified here: Apple M4 Max, 48 GB, 40-core GPU. |
| The mechanism probe code "is available" | Only its results file was supplied. The probe was re-implemented from its written description (`scripts/mechanism_probe.py`). Result: sparse circuits recomputed 21.5 of 208 nodes per change (10.35%), dense circuits 208 of 208, and 0 nodes differed from a full rebuild. The blueprint reported 18.2 of 208 (8.76%); the difference is a different random wiring. |

## 2. The reference systems

| Source | Established | Not established |
|---|---|---|
| TypeSafe, Jev documentation | Typed probabilistic decisions from text; "ingests the state once and evaluates every question against it in parallel"; 64k tokens per request; trained with a method the vendor calls RLCD; $0.042 per million input tokens | Architecture, parameter count and training data are not disclosed. Nothing here infers them. |
| TypeSafe, Jev 1.13 limitations (17 Sep 2026) | The vendor lists: literal reading; unreliable counting and arithmetic; dates read "as text, not as ordered quantities"; accuracy that "falls as the state grows with content unrelated to the decision"; adversarial text that "can move the answer"; probabilities of related questions that may not be consistent | These are the vendor's own cautions, not measured failure rates. |
| Pydantic AI integration notes | A tool "is therefore not offered again once its result is in the turn"; a tool needing an unstated argument "is one Jev will propose" | Observations from one integration. |
| ASSAY-001 (15-18 Sep 2026) | 8,576 responses, 0 type errors; Banking77 accuracy 0.7977, ECE 0.0936; CLINC150 accuracy 0.8812, ECE 0.0204 | The audit states it "is not a comparison to any other model". |
| Kev (Apache-2.0) | Qwen backbone, rank-16 LoRA (11.3M trainable), pointer head. Kev-0.8B: 149 ms for new text and 28 ms for repeated text on an M5 with MLX. Trails Jev on its development suites (0.842 vs 0.868; 0.594 vs 0.777; 0.602 vs 0.713) | An open model of this kind is trainable. It has not matched Jev. |
| Nimble | Shared-prefix prefill, batched per-field suffixes, projection onto permitted tokens; 2.2x to 7.3x faster than independent scoring on an M5 Pro | Shared processing and parallel decisions are already public. They are not a contribution of this project. |

**Consequence for this project.** No hosted model is called and no comparison with Jev is made or
implied. Two of Jev's documented weaknesses are directly relevant and are tested here on our own
system: degradation with unrelated content (the WIDE test set and the unrelated-pair evaluation)
and inconsistency between related questions (removed by construction, since related quantities
are computed from shared leaves by exact logic, `docs/PROOFS.md` T3).

## 3. The problem: acting on what is no longer true

| Source | Finding (authors' figures) |
|---|---|
| STALE, Chao et al., arXiv 2605.06527 (7 May 2026) | 400 expert-validated scenarios, 1,200 queries. Best evaluated model 55.2%; the authors' CUPMem 68.0%. Defines *implicit conflict*: a later observation invalidates a memory without negating it. Type I concerns the same attribute; Type II is propagated through a consequence. Scored by an LLM judge (95.8% agreement with people). Data CC BY 4.0. |
| MemStrata, Yadav, arXiv 2606.26511 (25 Jun 2026) | Cosine similarity separates a contradicted fact from a duplicate with AUROC 0.59. A deterministic supersession rule over (subject, relation, object) in a bi-temporal ledger drives stale-fact errors to about 0%, where retrieval serves superseded values 15-40% of the time. |
| StateAuditor, Sun & He, arXiv 2608.01619 (3 Aug 2026) | "Deterministic code pins each quotation to a single entry, checks that the new evidence really is newer, and lets only these verified transitions trigger repair." +5.0 points on STALE (95% CI +2.9 to +7.2). |
| StateMemBench, Fan et al., arXiv 2608.19652 (20 Aug 2026) | 234 multi-session scenarios; existing memory systems, retrieval and long-context baselines all struggle to report current rather than superseded state. |
| Nakayashiki, arXiv 2608.25553 (26 Aug 2026) | With a verification budget, sixteen language models "rarely re-verified a constraint that read as settled" and produced stale-consistent decisions in about 75% of episodes once the constraint had been superseded. Directing one verification slot to the critical path removed most of them. |
| Xu, arXiv 2609.18935 (16 Sep 2026) | Composing independently cached blocks can weaken query-conditioned memory selection; where an update is placed affects results. |

The last three were published after the blueprint's main sources and were not in it. They point
the same way: models do not reliably notice on their own that something they hold has been
superseded, and systems improve when exact code, not the model, decides what is current and what
must be re-examined. That is the division of labour adopted here.

**What this does not show.** These are benchmark results. None is a study of people using an
assistant. Demand for this project's application has not been measured.

## 4. Prior art for each mechanism

Nothing in the following list is new in this project.

| Mechanism | Prior art | Relation to this project |
|---|---|---|
| Recording why each belief is held and retracting dependents when a premise goes | Doyle, *A Truth Maintenance System*, Artificial Intelligence 12, 1979; de Kleer, *An Assumption-based TMS*, Artificial Intelligence 28, 1986 | **Missing from the blueprint, and the closest ancestor.** The engine is a truth-maintenance system whose justifications are computed by a neural reader and carry probabilities. de Kleer's "environments under which a belief holds" is what `certify_stable` enumerates. |
| Recomputing only what a change reaches; stopping when a recomputed value is unchanged | Acar, *Self-Adjusting Computation*, 2005; Mokhov, Mitchell & Peyton Jones, *Build Systems a la Carte*, ICFP 2018 | **Missing from the blueprint.** Early cutoff in `ease/graph.py` is the build-system rule. |
| Incremental inference in neural graphs | InkStream, Wu, Li & Mitra, arXiv 2309.11071 | Same guarantee (incremental result equals full result) for graph neural networks over a given graph. |
| Transformers that reuse computation under edits | Sharir & Anandkumar, arXiv 2307.14988: vector quantisation of intermediate values, 12.1x fewer operations (median) at comparable accuracy; Li et al., *Incremental Transformer*, COLING 2025: 6.2x faster, 2.2 points lower than RoBERTa-large on SQuAD 1.1 | **Missing from the blueprint.** These make the encoder itself incremental and pay for it in accuracy. This project takes the other route: the encoder is unchanged and reuse happens between whole (claim, record) readings, which costs no accuracy per reading but cannot reuse work inside a reading. |
| Temporal validity in memory | Zep, arXiv 2501.13956; MemStrata | The ledger's supersession rule is of this kind. |
| Fact verification sensitive to revisions | VitaminC, Schuster, Fisch & Barzilay, NAACL 2021: claims paired with evidence before and after real Wikipedia edits | Training and evaluation data. |
| Bounds on a model's output over a region of inputs | CROWN, arXiv 1811.00866 | Not used. Requirements here are evaluated by exact logic, whose extremes over a box are at its corners, so enumeration gives exact bounds without relaxation. |
| Value of information | Howard, *Information Value Theory*, 1966 | `ease/planner.py`, with joint evaluation of small sets to avoid the known myopia of single questions. |
| Error control without distributional assumptions | Gibbs & Candes, arXiv 2106.00170; Angelopoulos, Candes & Tibshirani, arXiv 2307.16895 | `ease/evolve/threshold.py` applies the quantile-tracking update to endorsements. The bound with delayed verdicts (T5) follows the same argument. |
| Guaranteed false-negative rate for a selection rule | Angelopoulos, Bates, Fisch, Lei & Schuster, *Conformal Risk Control*, ICLR 2024 | `ease/proposer.py`. |
| Probability calibration | Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 | Temperature scaling, fitted on dev data. |
| Separating what happens from how it is valued | Barreto et al., successor features, arXiv 1606.05312 | Predicted outcomes and declared utilities are kept apart; changing a utility recomputes no belief. |
| Optimising plans for cost and quality | Palimpzest, arXiv 2405.14696 | Related aim, different object. |

## 5. What is left that could be a contribution

After removing everything above, the candidate contribution is the combination and its
measurement:

1. a reader trained so that an unrelated record is read as establishing nothing (see 6), feeding
2. a precedence policy that is executed exactly yet differentiable, so that a small network can be
   trained *through* it, inside
3. a dependency graph in which model weights, calibration, memory and schema are inputs like any
   other, so that learning after deployment is handled by the same exact propagation as a change
   of evidence, with
4. every adaptive step either retractable (memory), bounded (threshold) or gated (consolidation).

Whether that combination is better than a conventional model behind the same ledger is an
empirical question. It is answered in `docs/RESULTS.md` against the pre-registered plan, including
the cases where it is not.

No search for patents or for unpublished systems was made. Global novelty is not claimed.

## 6. Findings made during development that changed the design

| Finding | Evidence | Change |
|---|---|---|
| A reader trained on NLI corpora reads unrelated text as decisive | Pilot model: 11.8% of 3,000 unrelated pairs read as SUPPORTS or REFUTES, mostly REFUTES at high confidence. With 30 records the chance that none is misread is 2%. | 20% of training pairs are synthetic unrelated pairs labelled NEI, half of them mined for word overlap with the claim. After 32k examples the rate was 0.5%. |
| **A streamed corpus can be ordered, and a shuffle buffer smaller than the file does not mix it** | VitaminC's train file holds 248,953 real-revision rows, then 121,700 synthetic rows with no NEI label. The first full training run used a 10,000-row buffer. Its dev accuracy on VitaminC rose to 78.1% at step 18,000 and fell to 72.2% by the end, while training loss and MNLI accuracy kept improving and calibration error doubled. The turn came where the stream crossed from real to synthetic rows (predicted step 19,450). MNLI, WANLI and SNLI were checked and are mixed. | The shuffle buffer now spans every corpus (35 s and 1.4 GB to fill for VitaminC). The trainer logs each source's label mix every 500 steps and warns when it drifts. A test reproduces the failure with a small buffer. The first run is kept in `runs/stage_a_v1_ordered_stream` and was never evaluated on a test set. |
| SNLI's labels assume both sentences describe one scene | Same probe: misreadings concentrated on SNLI-style claims | SNLI removed from training and from episode ground truth; kept as an evaluation set on which lower accuracy is expected. |
| `torch.nn.utils.clip_grad_norm_` is slow on MPS | 214 ms per step for 149M parameters, more than half the step | Replaced by an equivalent 5 ms implementation; norms agree to 7 significant digits and clipped gradients to 1.5e-8 (`ease/util.py`). |
| Mixed precision does not help on this GPU | fp16 autocast was 4-15% slower than fp32 | fp32 throughout. |
| Edge values depend on batch composition | Up to 4.8e-6 on MPS, 6.8e-5 on CPU with dynamic batching; exactly 0 with fixed shapes | `ModelScorer(canonical=True)`; H1 is tested in that mode. |
| A licence check excluded two corpora | ANLI is CC BY-NC 4.0; the FEVER-NLI repack declares no licence | ANLI is evaluation-only. FEVER-NLI is unused. |
| The dense baseline is expensive to evaluate | 5.0M tokens for 12 wide episodes | B2 is evaluated on a subset of test episodes; declared as a deviation. |

## 7. Sources

Reference systems: docs.typesafe.ai/models; docs.typesafe.ai/model-jaggedness/jev-1.13;
typesafe.ai/blog/introducing-system-one-models-and-jev; pydantic.dev/docs/ai/models/typesafe;
donttrustme.ai/assay-001.html; github.com/jaredpalmer/kev;
github.com/bespokelabsai/nimble/blob/main/docs/PARALLEL_SCORING.md.

arXiv: 1606.05312, 1704.05426, 1811.00866, 1910.14599, 2103.08541, 2106.00170, 2201.05955,
2208.02814, 2307.14988, 2307.16895, 2309.11071, 2405.14696, 2412.13663, 2501.13956, 2604.00842,
2605.06527, 2606.26511, 2608.01619, 2608.19652, 2608.25553, 2609.18935.

Other: aclanthology.org/2025.coling-main.590; huggingface.co/answerdotai/ModernBERT-base;
huggingface.co/datasets/STALEproj/STALE; huggingface.co/datasets/tals/vitaminc.

Cited from the standard literature without being re-fetched: Doyle 1979; de Kleer 1986; Howard
1966; Acar 2005; Mokhov, Mitchell & Peyton Jones 2018; Guo et al. 2017.

## 8. Verified survey, 2 October 2026: encoders, learning after deployment, data for the weaknesses

Done to decide what to train next and how far the self-learning design can be pushed. A search
agent found candidates; every figure below was then read from the primary page (arXiv abstract,
HTML or PDF; raw model or dataset cards on the Hub; repository licence files) by `curl`, and the
figures quoted here were re-checked by hand from the saved pages. The search engine's own
summaries were wrong at least once (it gave ModernBERT-large an MNLI score that is in fact a
CodeSearchNet number), so nothing rests on them.

### 8a. Encoders for the reader

MNLI dev accuracy from each model's own paper; scores from different papers differ by up to a
point for the same model because fine-tuning protocols differ, so small gaps mean little.

| Encoder | Parameters | Licence | Context | MNLI | Source |
|---|---:|---|---:|---:|---|
| `answerdotai/ModernBERT-base` (in use) | 149M | Apache-2.0 | 8,192 | 89.1 | arXiv 2412.13663, Table 5 |
| `answerdotai/ModernBERT-large` (addendum) | 395M | Apache-2.0 | 8,192 | 90.8 | same |
| `jhu-clsp/ettin-encoder-150m` | 149M | MIT | 8K | 89.2 | arXiv 2507.11412, Table 3 |
| `jhu-clsp/ettin-encoder-400m` | 395M | MIT | 8K | 91.3 | same |
| `jhu-clsp/ettin-encoder-1b` | 1.03B | MIT | 8K | 91.8 | same |
| `chandar-lab/NeoBERT` | 250M | MIT | 4,096 | 89.0 | arXiv 2502.19587, Table 3; pins an `xformers` build with Linux x86-64 wheels only |
| `microsoft/deberta-v3-base` | 184M | MIT | 512 | 90.6 (matched) | arXiv 2111.09543, Table 5 |
| `microsoft/deberta-v3-large` | 435M | MIT | 512 | 91.8 | same, Table 3 |
| `jhu-clsp/mmBERT-base` (multilingual) | 307M | MIT | 8,192 | 87.7 | arXiv 2509.06888, Table 2 |
| `EuroBERT/EuroBERT-210m` | 210M | Apache-2.0 | 8,192 | no MNLI reported; XNLI-en 83.5 | arXiv 2503.05500, Table 10 |
| `LiquidAI/LFM2.5-Encoder-350M` (2026) | 355M | LFM Open License v1.0, commercial use limited by revenue | 8,192 | 89.0 (card, mean of 5 seeds) | model card; no paper |
| `avey-ai/avey-b1-large-exp` (2026) | 391M | Apache-2.0 | 2,048 | 85.7 | arXiv 2602.15814, Table 2 |

What it says for this project: Ettin encoders share ModernBERT's architecture and tokenizer
(`model_type: modernbert`), so `ettin-encoder-400m` is a drop-in with a permissive licence and a
higher published MNLI than ModernBERT-large; DeBERTa-v3-base beats ModernBERT-base on MNLI
(90.6 against 89.1) at about half the throughput (ModernBERT paper, Table 2) and a 512-token
limit that this project's 256-token pairs never reach. None of these was trained here; the
addendum tests ModernBERT-large only. Published base-to-large gains on MNLI: ModernBERT +1.7,
Ettin +2.1, DeBERTa-v3 +1.2.

### 8b. Learning after deployment with guarantees

| Work | What it establishes | Bearing on the design |
|---|---|---|
| Conformal PID control, Angelopoulos, Candès and Tibshirani, arXiv 2307.16895, §2.1 | Online gradient descent on the quantile loss, which the paper names *quantile tracking*, "achieves long-run coverage under no assumptions except boundedness of the scores"; unlike ACI it "does not return infinite sets after a sequence of miscoverage events" | The endorsement threshold here is this method; T5 in `docs/PROOFS.md` is the same kind of bound. The paper's error-integration term is the next step if drift turns out systematic. |
| Adaptive conformal inference, Gibbs and Candès, arXiv 2106.00170 | The original online update with a finite-horizon bound on the realised error rate for any data sequence | The form of T5's bound. |
| Conformal risk control, arXiv 2208.02814; Learn then Test, arXiv 2110.01052 | Thresholds and model choices with finite-sample risk bounds on exchangeable held-out data | The proposer's miss bound is CRC; Learn-then-Test is how a consolidation gate can give a candidate an error-rate guarantee instead of a comparison. |
| SERAC, Mitchell et al., arXiv 2206.06520 | Memory-based editing: a scope classifier routes in-scope inputs to a counterfactual model and leaves the base model untouched; its fact-checking setting is built on VitaminC | The correction memory here is this with a scope of "exact match only". SERAC's VitaminC protocol is a ready benchmark for extending corrections to paraphrases. |
| GRACE, Hartvigsen et al., arXiv 2211.11031 | A codebook of cached activations with a deferral radius per entry; thousands of sequential edits without changing weights | A codebook keyed on the pair representation, with the radius under the gate, is the way to generalise corrections without touching the encoder. |
| Pitfalls of test-time adaptation, Zhao et al., arXiv 2306.03536 | Across ten methods, selection is hard because of online batch dependence and no method handles all shift types | Supports keeping the encoder frozen in use. Test-time adaptation (Ye et al., arXiv 2302.04618, Tent, arXiv 2006.10726) is the alternative not taken. |
| Meta lifelong learning with limited memory, Wang et al., arXiv 2010.02500 | Replay from a 1% episodic memory reduces forgetting and negative transfer | Consolidation already replays earlier feedback; the budget is a parameter to measure. |
| Regression bugs in model updates, Xie et al., arXiv 2105.03048 | Negative flips (right before, wrong after) are common across GLUE updates; distillation from the current model reduces them | The gate should report negative flips, not only accuracy; distillation from the current refiner is a documented remedy. |

Weight-editing methods (MEND arXiv 2110.11309, ROME arXiv 2202.05262, KnowledgeEditor arXiv
2104.08164) change weights and so conflict with the rule that nothing in use updates the
encoder; per-task adapters (AdapterFusion arXiv 2005.00247, O-LoRA arXiv 2310.14152) are the
route if that rule is ever relaxed per workspace.

### 8c. Data for the measured weaknesses

| Dataset | Licence (card or repository) | Why it fits | Commercial use |
|---|---|---|---|
| FEVEROUS, `fever/feverous` | CC BY-SA 3.0 | 4,242 not-enough-information claims *with* evidence; a claim counts as refuted when the evidence would mislead, which covers consequence-level conflicts (arXiv 2106.05707, §3.1.2) | yes (share-alike) |
| VitaminC (in use) | CC BY-SA 3.0 | NEI evidence is the same edited sentence (arXiv 2103.08541, §3.2) | yes |
| WANLI (in use) | CC BY 4.0 | neutral and contradiction pairs are lexically close (arXiv 2201.05955, §4.3) | yes |
| δ-NLI, defeasible inference | MIT (repository) | updates that weaken or strengthen an inference without negating it; labels are not 3-way | yes; source-text terms unchecked |
| WikiContradict, `ibm-research/Wikipedia_contradict_benchmark` | MIT | 253 explicit or implicit contradictions, for evaluation | yes |
| WiCE | annotations ODC-BY; text CC BY-SA | supported / partially supported / not supported for real Wikipedia claims | yes |
| ANLI (evaluation only here) | CC BY-NC 4.0 | about 58% of items need outside knowledge (arXiv 1910.14599, Table 7) | no |
| ConTRoL | CC BY-NC-SA 4.0 | passage-level logical reasoning | no |
| SciFact, FOLIO | licences conflict between the Hub card and the repository | | unresolved |
| DialFact, BoolQ contrast sets, Climate-FEVER | no data licence stated | | not cleared |

The FEVER and FEVEROUS Hub repositories hold loading scripts only, which `datasets` 4.0 and later
no longer run; they would have to be fetched from their own site.

What could not be verified is listed with the saved pages: the SciTail licence at its source, the
DialFact data licence, what the GPL tag on the FEVER card covers, the memory needed to fine-tune
the 1B and 2.1B encoders here, and whether any of the models above has problems on Apple's GPU
(no primary source says, and none was tried).
