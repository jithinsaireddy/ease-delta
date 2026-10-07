# Concepts

## Tasks, requirements and actions

A **task** is something you are trying to get done that depends on facts other people control.

A **requirement** is one of those facts, written as a plain sentence that can be true or false: *"The client has
approved the final design."* Write them so that a message could settle them, and name the specific thing:
*"approved version 5"*, not *"approved the latest version"* (see [limitations](LIMITATIONS.md)).

An **action** is something you might do, and the requirements it needs, combined with `and`, `or`, `not` and
`at least k of (...)`. An action also carries what it is worth when its requirements hold and what it costs when
they do not; only the planner uses those.

## Records

Every message, document passage or confirmation is a **record** with an id and a **revision**.

| What arrives | What happens |
|---|---|
| a new id | it is read against the requirements it may concern |
| an existing id with a higher revision | it replaces the earlier one, which is no longer counted |
| the same text at the same revision | a **duplicate**: nothing changes and nothing is read |
| a lower revision | a **stale** copy: nothing changes and nothing is read |
| a withdrawal | the record stops counting for or against anything |
| two different texts at the same revision | the record is **conflicted** until a higher revision arrives |

These rules are exact and order-independent: delivering the same set of records in any order gives the same state
(`PROOFS.md`, P1 to P4).

When several records bear on one requirement, they are ranked by **authority** (0 to 9, higher first) and then by
**time** (later first). The highest-ranked records that say something decisive decide; a later record that settles
nothing does not override an earlier one that did. If decisive records of the same rank disagree, the requirement is
in **conflict**.

`about` on a record limits which requirements it is read against. Without it, every record is compared with every
requirement.

## Readings

A **reading** is the reader's answer for one (requirement, record) pair: the probability that the record
**supports** the requirement, **refutes** it, or gives **not enough info**. The reader sees only the text, never who
sent it, so the text must say who is speaking.

Readings are cached by the model version and the two texts. A change re-reads only the pairs it touches, and a
duplicate or stale copy reads nothing. In **exact** mode a reading never depends on what else was read with it, and
the cached state is bit-for-bit what a full rebuild would compute (`RESULTS.md`, section 4).

## Requirement status

From the ranked readings the exact core computes, for each requirement, the probability of four outcomes:

| Status | Meaning |
|---|---|
| SATISFIED | the deciding records support it |
| VIOLATED | the deciding records refute it |
| UNRESOLVED | nothing decisive has arrived (or everything that did was withdrawn) |
| CONFLICT | records of the same rank disagree |

The status shown is the most likely of the four; the probabilities are shown with it.

## Ready, blocked, needs info

For each action the core computes how likely its requirements are to hold together, through its `and` / `or` /
`not` / `at least` formula.

| Disposition | When |
|---|---|
| READY | at least as likely as the readiness bar (90% to begin with), and more likely satisfied than violated |
| BLOCKED | its requirements are at least 50% likely to be violated |
| NEEDS INFO | anything else |

READY is a **proposal**. Nothing in EASE-Delta can send, pay, submit or execute; a person approves.

The readiness bar is not fixed. When people report whether READY proposals they acted on were right, the bar moves
so that the error rate among them stays near a target (5% by default), with a bound that holds for any sequence of
cases (`PROOFS.md`, T5).

## Questions

When an action needs information, the **planner** looks for the question whose answer is worth the most, net of the
cost of asking, considering sets of up to three requirements at once (asking one alone is often worthless when an
action needs two). It also lists requirements whose answer cannot change a proposal, whatever else is answered, so
you do not ask them. This is value of information over the model's own beliefs; it is exact, not sampled.

## Corrections and confirmations

- A **correction** says what a record establishes about a requirement. It takes effect at once, is stored, and can be
  undone, which returns every number exactly to what it was.
- A **confirmation** is a person's answer about a requirement. It enters as a record attributed to them and is read
  exactly. Among records of equal authority the later one wins, so a later message can still overturn it.

Over the service, a confirmation can be requested through a single-use link that shows the person only that one
requirement ([HTTP API](http-api.md#confirmation-links)).

## Learning after deployment

The encoder is never updated in use. What changes is small, versioned and reversible:

| Part | What it learns from | Safeguard |
|---|---|---|
| correction memory | corrections | exact; each item can be deleted; nothing is generalised to similar cases unless a gate has shown it helps |
| readiness bar | verdicts on READY proposals | the error bound above |
| consolidation | accumulated corrections | adopted only if held-out feedback improves and regression suites (single readings and whole tasks) do not get worse; every version keeps its parent and can be rolled back |

A deeper update of the reader itself is a new training run that goes through the full evaluation before release
(`configs/stage_a_continue.yaml`).
