# FAQ

**What is it, in one sentence?**
A system that keeps a task's "ready / blocked / needs info" decisions current as the messages they depend on
arrive, change and are withdrawn, re-reading only what a change touches.

**Is it a large language model? Does it call one?**
No. It is a 395M-parameter reader (a fine-tuned ModernBERT encoder; a 149M version exists) plus exact code for
versions, precedence, logic and planning. It runs on your machine and calls no hosted model.

**Does it send emails, pay invoices or do anything on its own?**
No. There is no code path that acts. A READY action is a proposal; a person approves it.

**How accurate is it?**
On generated tasks with human-written text, 91.6% of its ready / blocked / needs-info decisions were correct on
standard tasks and 86.8% on larger ones. The reader scores 91.5% on VitaminC test. On your own messages accuracy
will differ: email and contract language is unlike its training text. The full measurements, including what failed,
are in [RESULTS.md](RESULTS.md).

**Has anyone used it for real work?**
Not yet. It has not been measured with people; whether it saves time is the question the next study answers
([NEXT_STEPS.md](NEXT_STEPS.md)). If you try it on real work, an issue describing what it got wrong is the most
valuable thing you can send.

**Why does it say NEEDS INFO when every requirement shows SATISFIED?**
Because likely is not certain. Three requirements that are each 95% likely are together about 86% likely, and the
bar for READY is 90%. It then names the question whose answer would settle it.

**What does it need to run?**
A laptop. On an Apple M4 Max processor an update takes 0.2 to 0.6 s with the large reader (4.8 GB of memory) and
about 0.1 to 0.3 s with the base reader (2.5 GB). A GPU makes it faster (59 ms on the Apple GPU) but is not needed.

**Is my data sent anywhere?**
No. Models are downloaded once from the Hugging Face Hub; after that everything runs locally. The demo page runs the
reader inside your browser. The service binds to 127.0.0.1 unless you tell it otherwise.

**Which languages?**
English only, so far.

**How do I make it read my messages better?**
Say who is speaking in each message ("Email from the client: ..."); write requirements that name the specific thing
("approved version 5"); cut long documents into passages (`add_document` does this); say which requirement a message
concerns (`about`) when there are several of the same kind; and correct it when it is wrong: a correction takes
effect at once.

**Does it learn from me?**
In small, reversible ways. Corrections are remembered exactly. The readiness bar moves with your verdicts on what it
proposed, under a proven error bound. A consolidation step can learn a small adjustment from accumulated corrections,
adopted only if held-out feedback improves and regression suites do not get worse, and rolled back on request. The
reader itself is never updated while in use; a deeper update is a new training run that goes through the full
evaluation ([concepts](concepts.md#learning-after-deployment)).

**Can I use it commercially?**
The code is Apache-2.0. The weights are CC BY-SA 4.0, because one training corpus (VitaminC) is share-alike:
commercial use is allowed, with attribution, and models you adapt from these weights are shared under the same
licence. Nothing non-commercial (such as ANLI) was used in training. This is a description, not legal advice.

**How does it compare with Jev, or with asking a chatbot?**
No comparison has been measured, so none is claimed. It is designed for a narrower job: keeping many small decisions
consistent with a changing record, cheaply, exactly and with an audit trail. The comparisons that were measured are
against a dense reader of the whole task with the same backbone and training, and against a public NLI model.

**Can I train it on my own data?**
Yes: corrections can be exported (`Runtime.export_feedback`) and folded into a continued training run
(`configs/stage_a_continue.yaml`), which should go through the full evaluation before use. Training runs on one
Apple-silicon laptop; the large reader took five hours.

**How do I cite it?**
See [CITATION.cff](../CITATION.cff), or the BibTeX in the [README](../README.md#contributing-and-citing).
