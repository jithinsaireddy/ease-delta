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
