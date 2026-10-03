Each entry is something done differently from `docs/PREREGISTRATION.md`, with the reason and with
what it could affect.

1. **The dense reader (B2) was evaluated on a subset of the test episodes**: the first 400 of 1,000
   STANDARD episodes and the first 100 of 300 WIDE episodes. Reason: a dry run on non-test data
   showed B2 needs about 5.0 million tokens of inference for 12 WIDE episodes, which puts the full
   sets at several hours. Episodes are independent draws from the generator, so the first K are a
   random sample. Effect: intervals for comparisons involving B2 are wider than planned. The
   comparisons are paired on exactly those episodes (rows marked `@B2`). All other systems were
   evaluated on the full sets. Decided before any test episode was evaluated.

2. **A dry run printed test-set accuracy of a discarded model.** While checking that
   `scripts/eval_edge.py` ran, it was pointed at the first 150 rows of every evaluation set,
   including test sets, using the pilot model that had already been discarded for its training
   mixture. Those figures were printed. Nothing was chosen or changed on the basis of them, and the
   final model had not finished training. Later dry runs used non-test data or suppressed output.

3. **H2 was given two tables instead of one.** Wall-clock time was measured both with dynamic
   batching (the faster mode) and with canonical shapes (the mode in which H1 holds). The
   pre-registered decision rule is applied to dynamic batching, which was the mode intended when
   the plan was written; canonical timings are reported beside it.

4. **H6 was given a decision rule.** The plan said only "reported either way". The report counts
   the prediction as held if the 95% interval of (N minus E) disposition accuracy on WIDE lies
   below zero.

5. **The edge model was trained twice; the first run was discarded on development data.** The first
   full run finished with 78.8% on the VitaminC development set, below a pilot trained on
   one fourteenth of the data (82.2%). The cause was an ordered training file combined with a
   shuffle buffer smaller than the file (`docs/RESEARCH.md`, section 6). The experiment pipeline
   had started automatically when that run finished. It was stopped within a minute, during its
   first step. It had evaluated the four development sets and no test set; its output was set
   aside unread (`runs/aborted_pipeline_1`). The decision to retrain, and the fix, were made from
   development data and from the training file's own metadata. Nothing about the evaluation plan
   was changed.

6. **Additions made after the pipeline was frozen.** The experiments ran from a frozen copy of the
   code (`runs/code_snapshot`, hash in `runs/results_waiter.log`). Afterwards, and before any
   episode-level result was known, three exploratory additions were made, each labelled as such in
   the results:
   * *the demo, scored* (`scripts/demo_trace.py`): the hand-off scenario was scored against
     statuses written from its messages. The learned refiner misread three predicate states that
     calibrated rules read correctly.
   * *X2, small tasks* (`scripts/eval_small.py`): designed because of that observation, to test
     whether it holds beyond one scenario.
   * *report sections* for those two. The code that computes verdicts in `scripts/make_report.py`
     is unchanged from the frozen copy (checked by `diff`).
   By then the edge model's test-set results (`runs/results/edge.json`) and the public model's
   were known. They did not bear on these additions.

7. **How the released default is chosen, fixed before episode-level results.** The release ships
   the learned refiner (seed 1) only if (a) H5 is supported, and (b) on each of STANDARD, WIDE and
   the small tasks of X2, the refiner is not significantly worse than calibrated rules: the 95%
   interval of its status-NLL difference does not lie entirely above zero, and the interval of its
   disposition-accuracy difference does not lie entirely below zero. Otherwise the release ships
   calibrated rules, written as a refiner whose learned correction is zero
   (`ease.aggregate.rule_refiner`). That form computes exactly what calibrated rules compute and
   can still be improved later by gated consolidation.

8. **X3, readings per predicate, was designed after the STANDARD episode results were known.** Those
   results showed E-declared (each record compared only with the predicate it was written for)
   scoring 3.5 points below E, although it has more information. X3 tests one explanation: that
   the learned refiner depends on how many readings a predicate has. It is exploratory and is
   reported as such; no pre-registered verdict depends on it.

9. **Changes made after the experiments exposed a weakness, and one more exploratory experiment.**
   The WIDE results showed one of three refiners (seed 3) losing 6.8 points against calibrated
   rules, and X3 showed the refiner degrading when a predicate has few readings. The consolidation
   gate of H7 checked candidates only on single readings, so it could not have caught either.
   After those results, and before any conclusion was drawn from H7:
   * the gate gained *set suites*: whole predicates from validation episodes of the Stage B pool,
     in three sizes (`scripts/build_set_suites.py`). On them, seed 3 scores 77.3% against 88.9%
     for seed 1 on the large-task suite, while the standard suite cannot tell them apart
     (`runs/results/set_suites_check.json`). No test data is involved;
   * fine-tuning during consolidation no longer changes how the refiner reads the rest of a set,
     unless explicitly allowed;
   * calibrated rules can be written as a refiner with a zero correction and no context
     (`ease.aggregate.rule_refiner`), so they can be the starting point of self-evolution.
   H7 was run by the frozen code, without any of these. **X4** (exploratory) repeats the H7
   procedure with them, starting from calibrated rules, on the same streams plus MNLI-mismatched,
   whose genres are absent from training and whose labels follow the same conventions as training.


10. **Diagnostics run after all results were known, to decide what to do next (X5).** None was
    pre-registered and none changes a verdict or the released model.
    * *Replay with corrected readings* (`scripts/diagnose_headroom.py`): the STANDARD and WIDE test
      episodes, already evaluated, run again with some of the edge model's readings replaced by
      the generator's labels. It uses test labels, so it is analysis of the reported runs, not a
      new test, and nothing was tuned on it.
    * *Fit check*: the released model read a sample of its training splits.
    * *More training* (`configs/diag_more_training.yaml`, `scripts/compare_models.py`): the
      released model trained for 200,000 more examples and compared with the released model on
      the development sets only. The continued model was never run on a test split and is not
      released.
    * *Running without a GPU* (`scripts/eval_cpu.py`): update time of the shipped configuration
      on the CPU. Timing only.
    Results: section 8j and 8k.

11. **A fault found by using the service, and fixed.** The planner proposed questions only about
    requirements the model considered open (at least 0.2 probability of being unresolved). A
    requirement read as satisfied at about 0.86 is not open, so an action could sit below its
    threshold with no question proposed. Candidates are now all requirements the model is not
    sure of, and the value-of-information calculation decides among them
    (`ease.planner.askable_predicates`, two tests). No reported measurement used the planner's
    choice of question, so no figure in this report changes.

12. **The released aggregator file was rebuilt so that packaging is repeatable.** Calibrated rules
    are stored as a refiner whose last layer is zero. Its hidden layers, which do not affect the
    output, were drawn from the global random generator, so packaging the same release twice gave
    two different files and version identifiers. They are now drawn from a fixed seed
    (`ease.aggregate.rule_refiner`, one test). The output is unchanged: exactly the calibrated
    rules. X4 started from a refiner built the old way; its results are reported as measured.

13. **A second planner fault found by using the page, and fixed.** "Unnecessary" questions were
    certified one predicate at a time: a requirement whose answer changes nothing *on its own*
    (an AND with another open requirement) was reported as unable to change the proposal, while
    the same requirement appeared in the question worth asking. A predicate is now unnecessary
    for an action only if its answer cannot change the proposal under every settlement of the
    other open predicates (`ease.planner.unnecessary_questions`, one test). No reported
    measurement used this function.

14. **The larger-encoder addendum, run as planned.** `docs/PREREGISTRATION_LARGE.md` was hashed
    before training; its hash is unchanged. The pilot found batch 32 fits, so the main setting
    was used. Three things were done that the plan did not spell out, none of which touches a
    verdict: the Stage B script fitted one refiner seed as a by-product of fitting the rules'
    calibration (the refiner is not used; calibrated rules ship, and `package_release.py
    --force-rules` makes that choice explicit); the update-time comparison of H12 used
    `scripts/eval_cpu.py` with each model's calibrated rules, both models in one run, rather than
    the H2 script, which needs the dense baseline; and the fit and headroom diagnostics of X5
    were repeated for the large model (exploratory). The base model is published alongside as
    `ease-delta-base`, for machines where 2.7× the update time matters.

15. **Bit-exact reuse on the CPU needed a change, found after publication of the code.** Running
    the shipped (395M) model on the CPU from a fresh clone, the full-rebuild check reported 19 of
    26 cached values differing by up to 6e-6. The cause was measured, not guessed: on the CPU a
    pair's reading depended on which other pairs shared its batch of eight (the base model did
    not show this; on the Apple GPU neither model does, which is what H1 measured). Canonical mode
    now uses a batch of one pair on the CPU, which makes a reading depend on the pair alone there
    too; the GPU keeps batches of eight. The cost on the CPU and the re-check are in section 8k.
    H1's figure (0 of 195,434, Apple GPU) is unchanged.
