# What to do next

Ordered by how much each step would change what we know.

## 1. Measure it with people

Everything in `docs/RESULTS.md` is about the system. The goal is about people: less time spent
re-explaining, re-checking and repairing work after something changes. That has not been measured.
Until it is, "this makes people's lives easier" is a hypothesis.

### A study small enough to run

**Question.** When the facts of a task change, do people finish correctly in less time, with
fewer stale decisions, using EASE-Delta than using what they use now?

**Participants.** 16 to 24 people who prepare client deliverables as part of their work
(freelancers, account managers, small-team leads). Recruit outside the project's circle. Pay them.

**Design.** Within-subject, two conditions, counterbalanced order:

* *A, baseline*: the participant's usual tools (mail client, documents, a checklist).
* *B, EASE-Delta*: the same materials loaded into the local page.

Each participant does two matched task packets, one per condition. Which packet goes with which
condition is also counterbalanced, so neither packet nor order is confounded with condition.

**Task.** A hand-off packet with 6 to 8 requirements and about 25 messages and documents. After
the participant has reached a first answer, five changes arrive, one at a time:

1. a correction that changes a requirement's status (an approval is withdrawn);
2. a correction that changes nothing relevant (a typo fixed in an unrelated message);
3. an old version of a document delivered again;
4. a change that affects a requirement indirectly (a new design version after approval of the old);
5. a withdrawal that leaves a requirement unknown.

After each change the participant states which actions are ready, blocked or unknown.

**Measures.**

| Measure | How |
|---|---|
| time to a correct state after each change (primary) | screen recording, timestamps |
| stale decisions | statements that still reflect the state before the change |
| unnecessary re-checking | documents reopened that the change could not have affected |
| questions the participant had to answer | count, and how many were needed |
| corrections the participant made to the system | count, and time spent |
| trust calibration | after each change: "how sure are you?" against whether they were right |

**Analysis, fixed in advance.** Paired difference in time per change, per participant, with a 95%
interval. A change counts only if the final state was correct; incorrect final states are
reported separately and never dropped. The blueprint's threshold was 30% less correction time at
no worse task success; keep it, and decide before looking.

**What would count against the system.** People take as long or longer; people accept a wrong
READY because the page said so (over-trust); time saved on changes is spent correcting misreadings.

**Ethics.** Use fabricated materials, not participants' own mail. Record consent. Let people stop
at any time. Store recordings locally and delete them after coding.

### Before the study

* Run three pilot sessions and fix what confuses people. Expect the page to need work.
* Item 4 above is a join. Experiment X1 shows what the system does with joins; if it is unsafe,
  declare such requirements `join=True` or exclude them, and say which.

## 2. Close the gaps the experiments found

Read the verdict table in `docs/RESULTS.md` first. Then, for each hypothesis that was not supported,
decide between fixing and removing. Candidates known in advance:

| Gap | Direction |
|---|---|
| numbers and dates read as words | extract values with a span head, compare with exact code; the dependency graph already supports a new node kind |
| implicit, propagated conflicts | the STALE result measures the size of the gap; training data with consequence-level conflicts does not exist in the corpora used here |
| same-topic distractors (measured: read as decisive 22% of the time, against 0.8% off-topic) | build an evaluation set from one project's real correspondence, labelled by the people who wrote it; meanwhile have records say which requirement they concern |
| one seed for the large models | train the edge model and the dense baseline twice more and report the spread |
| English only | `jhu-clsp/mmBERT-base` (MIT) is a drop-in backbone; training data in other languages is the constraint |

### More training, or a larger model?

Measured in results, section 8j, after everything else was known:

* With every reading correct, the system was right at every step of every test task. All of its
  error is the edge model's reading, and task accuracy rises almost in a straight line as reading
  errors are removed: about 2.4 points for each quarter of them on standard tasks.
* More of the same training does not remove them. 200,000 further examples on the same corpora
  left the released model slightly worse on development data (VitaminC -0.48 points, MNLI -0.39).
* The model is at the level published for its corpus. The VitaminC authors' largest model,
  trained on VitaminC and MNLI, reported 88.69% on real revisions and 93.92% on synthetic ones
  (Schuster et al., 2021, Table 4). This model: 88.65% and 92.78%.

So, in order of gain for effort:

| Step | Gain | Cost | Status |
|---|---|---|---|
| say which requirement a record concerns | +1.2 points on standard tasks, +4.1 on larger ones | none: a field on the event, or one folder per requirement | measured (results, section 8d: rules, declared links) |
| training text from real use: one project's messages, labelled by the people who wrote them | unknown; it targets the two weaknesses seen on real text, same-topic messages and low confidence on email | a pilot (section 1); `Runtime.export_feedback` and `configs/stage_a_continue.yaml` exist for this | not done |
| a larger encoder, `answerdotai/ModernBERT-large` (395M parameters) | estimated beforehand at about +1.5 points on standard tasks and +1.5 to +2 on larger ones; **measured** +2.0 [+1.1, +2.9] and +2.2 [+0.7, +3.6] (results, section 8l) | 5 hours of training here; 2.7 times the update time; 4.8 GB | done; it ships |
| more passes over the same corpora | none | | measured: no |

That run was made under `docs/PREREGISTRATION_LARGE.md`, with the shipping rule written first;
all three of its hypotheses were supported and the larger model ships (results, section 8l).

Other encoders, from the verified survey in `docs/RESEARCH.md`, section 8, none trained here:

| Encoder | Parameters | Licence | MNLI (its own paper) | Note |
|---|---|---|---|---|
| `jhu-clsp/ettin-encoder-400m` | 395M | MIT | 91.3 | same architecture and tokenizer as ModernBERT: `encoder_name` is the only change |
| `jhu-clsp/ettin-encoder-1b` | 1.03B | MIT | 91.8 | memory for fine-tuning on this machine not measured |
| `microsoft/deberta-v3-base` | 184M | MIT | 90.6 (matched) | higher than ModernBERT-base on MNLI; about twice as slow per token (ModernBERT paper, Table 2); 512 tokens |
| `microsoft/deberta-v3-large` | 435M | MIT | 91.8 | as above |

Training data for the measured weaknesses (same survey): FEVEROUS (CC BY-SA 3.0) labels
not-enough-information claims *with* their evidence and counts a claim as refuted when a
reader would be misled, which is the consequence-level case; WANLI's neutral and contradiction
pairs are lexically close. Both are commercial-use compatible. ANLI and ConTRoL are not. It does not address the weaknesses that matter most in use
(joins, consequences, same-topic messages), which need different data, not more capacity.
ANLI would help adversarial reasoning and is licensed for non-commercial use only; leave it out
unless the release is to be non-commercial.

## 3. Connect it to where documents live

The engine takes passages. Something has to supply them. Done so far: files a person supplies
(`.eml`, `.mbox`, `.txt`, `.md`) are cut into attributed passages with stable identity (the
message id) and the sender, date and subject in the text; a workspace can be told by webhook
when a proposal changes; a reviewer can be sent a one-question link. Not done: connectors that
watch a mailbox, a document store or a project folder on their own. Each must provide stable
record identity, revision numbers and attribution in the text. Read access only. Nothing here
should ever send.

## 4. Keep the discipline

* New claims go in `docs/PREREGISTRATION.md` before the experiment, with a decision rule.
* `docs/RESULTS.md` is generated. If a number is wanted in it, add it to a results file.
* A deeper model update (`configs/stage_a_continue.yaml`) goes through the whole pipeline before
  release, and ships with its own regression suite.
* No comparison with a hosted model unless its terms allow the use and the comparison uses the
  same ledger, schema and planner on both sides.
