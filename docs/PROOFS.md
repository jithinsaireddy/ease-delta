# What is proved, under which assumptions, and where it is tested

Each statement below is a property of the software as specified. None of them says that the model
reads language correctly. A system can satisfy every theorem here and still be wrong about what an
email means; that is measured in `docs/RESULTS.md`, not proved.

| Id | Statement | Assumptions | Checked by |
|---|---|---|---|
| L0 | A compressed summary cannot always be corrected | none | `test_ledger.py::test_why_the_ledger_keeps_records...` |
| P1-P4 | Ledger merge is idempotent, commutative, never resurrects, never resolves a conflict by arrival order | stable record identity and revision numbers | `test_ledger.py` (random event sets) |
| T1-T2 | Incremental execution equals full recomputation | deterministic node functions; every changing quantity is an input node | `test_graph.py`, `test_engine.py`, H1 |
| T3 | Requirement evaluation is exact | A1 | `test_logic.py` (enumeration of all worlds) |
| T4 | Precedence aggregation is exact | A2 | `test_aggregate.py` (enumeration of all readings) |
| T5 | Endorsement error rate is bounded | truthful verdicts | `test_evolve.py` (adversarial sequences) |
| T6 | Expected missed-link rate is at most alpha | exchangeable tasks | `test_proposer.py` (repeated draws), H8 |
| T7 | A stability certificate is sound | the model's own beliefs | `test_planner.py` (random settlements) |

---

## L0. Information that was discarded cannot always be corrected

Let `h` map histories to stored states and let `u(h(H), c)` be any updater that sees only the
stored state and a correction `c`. If two histories satisfy `h(H1) = h(H2)` while the correct
results after `c` differ, then `u` receives identical arguments in both cases and returns one value,
so it is wrong for at least one history.

Instance: `H1 = {r1: 1, r2: 2}`, `H2 = {r1: 2, r2: 1}`, `h` = sum, `c` = delete `r1`. Both states
are 3; the correct results are 2 and 1.

This is a counting argument, not a new theorem, and it does not say recurrent models always fail.
It says exact retraction needs the stored state to distinguish the histories that the correction
distinguishes. The ledger satisfies that by keeping addressable records.

## P1-P4. The ledger

State of a record: `(r, S)` with `r` the highest revision seen and `S` the set of distinct
payloads seen at revision `r`. Merging an event `(q, p)`:

```
q < r :  (r, S)
q = r :  (r, S ∪ {p})
q > r :  (q, {p})
```

Write the state as the pair `(r, S)` ordered lexicographically: `(r, S) ⊑ (r', S')` iff `r < r'`,
or `r = r'` and `S ⊆ S'`. Merging an event `e = (q, p)` computes the least upper bound of the state
and `(q, {p})`:

* if `q < r` the bound is `(r, S)`;
* if `q = r` it is `(r, S ∪ {p})`;
* if `q > r` it is `(q, {p})`.

A least upper bound is associative, commutative and idempotent. Hence the state after a multiset
of events is the least upper bound of the events, which does not depend on order (P2) or on
repetition (P1). An event with `q < r` leaves the state unchanged (P3). Two payloads at the same
revision both enter `S`, whichever came first, and `|S| > 1` is reported as CONFLICTED (P4).

Assumption that matters in practice: the source assigns revision numbers consistently. If two
systems number revisions independently, their events must be given different record identities.

## T1-T2. Incremental execution

Let the computation be a finite DAG. Each computed node `v` holds `f_v(parents' values)`, with
`f_v` deterministic. Inputs change; caches hold the last computed values.

**T1.** If no ancestor of `v` changed value, the cached value of `v` is what a full recomputation
would produce. *Proof.* Induction on depth. A node all of whose parents are inputs receives
unchanged arguments. For deeper nodes, each parent has no changed ancestor either (its ancestors
are ancestors of `v`), so by induction each parent's cached value is correct and unchanged; `f_v`
is deterministic, so its cached output is correct. ∎

**T2.** `propagate()` processes scheduled nodes in order of level, where a node's level exceeds
that of every parent, and schedules the children of every node whose value changed. After it
returns, every cached value equals the full recomputation. *Proof.* Induction on level. Suppose
all nodes below level `k` hold correct values. A node `v` at level `k` is either scheduled or not.
If scheduled, it is evaluated on its parents' current values, which are correct, so its value is
correct. If not scheduled, none of its parents changed value during this propagation (a changed
parent schedules its children) and `v` was not newly created or rewired (those are scheduled
directly), so its arguments equal those of its last evaluation and its cache is correct. ∎

**Early cutoff** is the rule "a node whose recomputed value equals its cached value does not
schedule its children". It is already contained in T2, which schedules children only on change.

**What the assumptions exclude.**

* *Hidden inputs.* Model weights, schema and clock would be hidden inputs if they were read from
  outside the graph. They are input nodes (`w:edge`, `w:agg`, `schema`) or are applied by
  rewriting input nodes (`set_now`), so a change to any of them propagates like any other change.
* *Non-deterministic functions.* On the development machine a pair's edge value depended on the
  batch it was computed in, by up to 4.8e-6, when batches were formed freely. With fixed shapes
  (`ModelScorer(canonical=True)`) 0 of 200 pairs differed under regrouping, on MPS and on CPU
  (`runs/determinism_probe.json`). H1 is tested in canonical mode. In the default mode the
  guarantee is equality up to that numerical variation, and the measured maximum is reported.
* *Approximate equality.* `values_equal` is exact. A tolerance would turn T2 into an approximation.

## T3. Requirements

A1: given the evidence, distinct predicates are independent.

For a formula in which every predicate occurs once, the children of each gate depend on disjoint
sets of predicates and are independent by A1. The status of `ATLEAST-k` over `n` children is
SATISFIED iff at least `k` are satisfied and VIOLATED iff at least `n - k + 1` are violated (then
`k` successes are impossible); these events are disjoint. The joint distribution of
(number satisfied, number violated) is computed by dynamic programming over the children, which
is exact for independent children. AND is `k = n`, OR is `k = 1`, NOT swaps the two outcomes.

If a predicate occurs more than once, its occurrences are not independent. `evaluate` conditions
on each repeated predicate, evaluates the formula with that predicate fixed, and sums with the
predicate's probabilities. Conditional on the repeated predicates the remaining leaves are
distinct, so the inner evaluation is exact, and the law of total probability gives the result.

A1 can fail: two predicates may be settled by the same sentence. Then the computed probabilities
of gates are approximations. The generator used in the experiments attaches a separate record to
each predicate, so A1 holds there by construction; it is an assumption about real tasks.

## T4. Precedence

A2: given the texts, the reader's errors on different edges are independent.

Group edges by rank. For a group, the events "every edge reads NEI", "some edge reads SUPPORTS
and none reads REFUTES", "some reads REFUTES and none SUPPORTS" and "both occur" partition the
possibilities, with probabilities

```
none = Π p_N        sup = Π (p_S + p_N) − none        ref = Π (p_R + p_N) − none
con  = 1 − none − sup − ref
```

The policy's outcome is that of the first group, in rank order, that is not "none". Groups are
disjoint sets of edges and therefore independent by A2, so

```
P(SATISFIED) = Σ_g (Π_{h<g} none_h) · sup_g      and likewise for VIOLATED and CONFLICT,
P(UNRESOLVED) = Π_g none_g .
```

The expression is built from sums and products of the edge probabilities, so it is differentiable
and a network that produces those probabilities can be trained through it.

A2 is the weaker of the two assumptions. Two records quoting the same sentence will be misread
together. The refiner's context input exists to let training compensate; whether it does is an
empirical question (H5).

## T5. Endorsement error rate

Update: `τ ← τ + η (e − α)` after each verdict `e ∈ {0, 1}` on an endorsed action. An action is
endorsed only if its confidence is at least the current `τ`. Let `D` be the largest number of
endorsements ever awaiting a verdict at once.

Summing the update over the first `K` verdicts: `η Σ_k (e_k − α) = τ_final − τ_1`.

Consider the last endorsement made. Confidence is at most 1, so `τ ≤ 1` at that moment. Afterwards
no endorsement is made, so only verdicts already awaited can arrive, at most `D` of them, each
raising `τ` by at most `η(1 − α)`. Hence `τ_final ≤ 1 + D η (1 − α)` and

```
(1/K) Σ_k e_k  ≤  α + (1 + D η (1 − α) − τ_1) / (η K).
```

No assumption is made about the sequence of cases. The bound is over endorsed actions that
received a verdict. It can be satisfied by endorsing nothing; `stalled` reports when `τ > 1`.

The update is the *quantile tracking* of Angelopoulos, Candès and Tibshirani (Conformal PID
Control, arXiv 2307.16895, section 2.1): online gradient descent on the quantile loss, which
they show gives long-run coverage assuming only that scores are bounded (here confidences lie
in [0, 1]) and which, unlike adaptive conformal inference (Gibbs and Candès, arXiv 2106.00170),
does not run off to an infinite threshold after a run of errors. The finite-horizon form of the
bound above follows their and ACI's argument; the `D` term accounts for verdicts that arrive
after the last endorsement.

## T6. Missed links

Let `L_i(λ) ∈ [0, 1]` be the fraction of true links of task `i` missed at threshold `λ`,
non-decreasing in `λ`. With `n` calibration tasks choose the largest `λ̂` with
`(n/(n+1)) · mean_i L_i(λ̂) + 1/(n+1) ≤ α`. If the `n` calibration tasks and the new task are
exchangeable, `E[L_new(λ̂)] ≤ α` (Angelopoulos et al., *Conformal Risk Control*, ICLR 2024,
Theorem 1). The expectation is over the draw of calibration tasks and new task together.

Exchangeability fails when deployment tasks differ in kind from calibration tasks, and the
guarantee then says nothing. "True links" are the links known to the calibration data.

## T7. Stability certificates

`certify_stable(action, F)` evaluates the action's disposition under every assignment of
{satisfied, violated, open} to the predicates in `F` and reports *stable* iff all `3^|F|`
dispositions equal the current one. By construction no settlement of `F` changes the disposition
the model proposes. The certificate concerns the model's proposal. It is void if a record outside
`F` changes, and it says nothing about whether the proposal is right.

The belief of a requirement is multilinear in the beliefs of its predicates (T3: each predicate
enters through conditioning, which is linear). A multilinear function attains its extremes over a
box at the corners, which is why `belief_interval` may enumerate corners and return exact bounds.
