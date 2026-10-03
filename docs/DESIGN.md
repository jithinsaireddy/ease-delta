# Design

## The rule the design follows

> Learned models interpret language. Exact code handles identity, versions, precedence, logic,
> arithmetic and permissions.

Every component below is on one side of that line, and nothing is on both.

| Component | Side | Why |
|---|---|---|
| Ledger | exact | which text is current is a matter of revision numbers, not judgement |
| Edge model | learned | what a passage says about a claim is a matter of language |
| Refiner | learned | how far to trust a reading, given the other readings, is learned from outcomes |
| Precedence policy | exact | which source prevails is declared by the person, so it is executed as declared |
| Requirement logic | exact | AND, OR, NOT and k-of-n have one correct answer |
| Planner | exact | expected utility is arithmetic over the model's beliefs |
| Correction memory | exact store, learned retrieval key | what a person said is stored verbatim; which new cases resemble it is judged by the model |

## Data flow

```
                    ┌───────────────────────────── inputs ─────────────────────────────┐
                    │  rec:R      one per record; value is the current payload or None │
                    │  pred:P     one per predicate; (claim text, prior)               │
                    │  w:edge     version of the edge model's weights                  │
                    │  w:agg      version of refiner + calibration + memory            │
                    │  schema     version of the task's structure                      │
                    └──────────────────────────────────────────────────────────────────┘
   level 1   edge:R|P    = f_edge(w:edge, rec:R, pred:P)          logits, message, rank metadata
             jedge:P     = f_edge(w:edge, pred:P, rec:* ...)      only for predicates declared join=True
   level 2   belief:P    = f_agg(w:agg, pred:P, edge:*|P ...)     P(satisfied, violated, open, conflict), belief
   level 3   action:A    = f_logic(schema, belief:* ...)          status, belief, P(success), disposition
```

A change is always "set an input node, then propagate". What differs is which input:

| What happened | Input that changes | What is recomputed |
|---|---|---|
| a record is added, corrected or withdrawn | `rec:R` | edges of R, beliefs they feed, actions above those |
| the same revision arrives again; an old revision arrives late | none (the ledger's state is unchanged) | nothing |
| a validity window opens or closes | `rec:R` for the records concerned | as for a changed record |
| a person corrects a reading | `w:agg` | every belief, every action; **no edge** |
| calibration or refiner is updated | `w:agg` | every belief, every action; no edge |
| the edge model is replaced | `w:edge` | every edge, and everything above |
| a predicate's wording changes | `pred:P` | edges of P, its belief, actions above it |
| a predicate is added | new `pred:P` | edges of every record against P only |

The last column is not a design intention that might drift. It is what `IncrementalGraph.propagate`
does for any graph, and `tests/test_engine.py` asserts the counts.

## Why one pair at a time

The edge model reads one claim against one record. Three consequences:

1. **Reuse is exact and per record.** A reading depends on the weights and two texts. Changing a
   record invalidates the readings that contain its text. A model that reads the whole state at
   once has no part of its computation that survives a change to any record.
2. **Cost of an update is the number of predicates**, not predicates times records.
3. **It cannot join.** A claim that needs two records read together is out of reach of separate
   readings. This is the price of 1 and 2. Experiment X1 measures what separate reading does when
   it meets such a claim. A predicate can be declared `join=True`; its linked records are then
   concatenated in rank order and read in one pass (node `jedge:P`), at the cost of re-reading the
   group whenever a member changes. The edge model was not trained on concatenated records, so how
   well that works is also measured in X1 rather than assumed.

The alternative designs were considered and are present as baselines, not dismissed:
the dense reader (B2) reads everything jointly, and the neural aggregator (N) learns the policy
instead of executing it.

## Why the policy is differentiable

`precedence()` computes, from per-edge probabilities and ranks, the exact distribution over
{satisfied, violated, open, conflict} that the declared policy induces. It is a composition of sums
and products, so gradients pass through it. The refiner is trained with the loss applied *after*
the policy. It therefore learns whatever per-edge adjustment makes the policy's output right, for
example to distrust a marginal "refutes" from a record when many other records are present,
without the policy itself ever being approximated.

An untrained refiner has a zero last layer and is exactly the rule aggregator. Anything it learns
is a measured change from calibrated rules (H5).

## Canonical computation

On a GPU, the value of a pair can depend on what else was in the batch. To make "cached equals
recomputed" true bit for bit, `ModelScorer(canonical=True)` gives every forward pass a shape that
depends on the pair alone: a fixed batch size (short batches are filled with copies of their first
row) and the smallest length bucket that fits. This costs padding. `ease serve` uses it by
default; `--fast` turns it off.

Downstream of the edge model everything runs on the CPU in float64, one predicate at a time, and
is deterministic.

## Learning after deployment

```
person corrects a reading ──> memory (SQLite) ──> w:agg changes ──> beliefs recomputed
person judges an endorsed action ──> threshold tracker ──> READY needs more (or less) confidence
enough feedback accumulated ──> consolidate():
        fit / held-out split by pair identity
        candidates: memory settings, fine-tuned refiner, both
        gate: regression suite must not get worse AND held-out must improve
        adopted ──> new version directory, parent recorded ──> w:agg changes
        rejected ──> nothing changes
```

Defaults are conservative. Similar-case generalisation is off (`lam_max = 0`) until a gate has
enabled it. Without a regression suite, consolidation refuses to change anything.

The encoder is not trained in deployment. A deeper update is possible by continuing Stage A from
the released weights with feedback added to the stream; it changes `w:edge`, invalidates every
cached reading, and should go through the full evaluation before release.

## Persistence and failure

| State | Where | If the process dies mid-write |
|---|---|---|
| evidence | SQLite, append-only, `synchronous=FULL`, one transaction per event | the event is either in the log or not; the merged state is rebuilt from the log and cannot disagree with it, because merge is order-independent |
| edge readings | SQLite, keyed by weights version and texts | a missing row is recomputed |
| corrections | SQLite | as for evidence |
| model versions | one directory per version, pointer file written atomically | the pointer names the old or the new version, never half of each |
| threshold | JSON, written atomically | the last saved value |

## Extension points

| To add | Implement | Register |
|---|---|---|
| a different reader | `EdgeScorer` (`score(pairs) -> [(logits, message)]`, `version`) | pass to `Engine` |
| a different combination rule | `Aggregator.status_batch` | pass to `Engine` |
| sparse linking | a `Linker` callable | `Engine(linker=...)` |
| value extraction (dates, amounts) | a new node kind reading `rec:R`, and exact comparison nodes above it | `IncrementalGraph.register` |
| documents longer than a passage | `ease/ingest.py` splits them and keeps attribution in each passage | `POST /tasks/{id}/documents` |

## What is deliberately absent

* No endpoint or function executes an action.
* No user fact is written into model weights.
* No tolerance in equality checks.
* No tuning on test data: the scripts that read the test pool are run once, by the pipeline.

## The service layer

`ease.runtime.Runtime` is one person's deployment: tasks, corrections, adopted versions, the
endorsement threshold and an audit log under one directory, all going through `Engine`.
`ease.service.workspaces.Service` holds many of them: one Runtime per workspace, each in its own
directory, so nothing learned or set in one workspace reaches another. Only the model and its
reading cache are shared; the cache holds model outputs keyed by the weights version and hashes
of the texts, never the texts. A workspace is addressed by a bearer key stored as a hash. Quotas
are plain numbers in the workspace's file; usage is counted per UTC day. One lock serialises
everything that touches the model. `ease.service.confirmations` gives one requirement to one
person through a single-use link; the answer enters the ledger as an attributed record and an
exact reading, so it is never reinterpreted. `ease.templates` turns a few form fields into a
task definition; `ease.importers` turns mail and text files into attributed passages. In local
mode the single workspace's files sit directly under the data directory, as before.
