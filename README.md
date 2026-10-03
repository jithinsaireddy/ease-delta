# EASE-Delta

Keeps a task's decisions current as its evidence changes.

You declare what must be true before something can be done ("the client approved the design",
"the delivery date is confirmed"). Messages and documents arrive, get corrected, get withdrawn.
EASE-Delta reads each one against each requirement, remembers the reading, and after every change
tells you which actions are ready, which are blocked and why, and which single question is worth
asking. When one record changes it re-reads that record and nothing else.

It proposes. It never sends, pays, submits or executes anything.

**Read `docs/RESULTS.md` before relying on it.** That file reports every measurement, including
the hypotheses that were not supported. `docs/LIMITATIONS.md` lists what was never measured,
starting with the most important: no person has used this, so whether it saves anyone time is
unknown.

## Results in brief

Measured on an Apple M4 Max against a plan written before any test data was read
(`docs/PREREGISTRATION.md`). Tasks are generated; their text is human-written (Wikipedia
revisions and NLI corpora). Brackets are 95% intervals over episodes. The release ships the
larger reader (ModernBERT-large), chosen by a second plan also written before it was trained
(`docs/PREREGISTRATION_LARGE.md`); rows that say "base reader" were measured with the 149M model
and are kept as reported.

| | Result |
|---|---|
| Exactness | 0 of 195,434 cached values differed from an independent full rebuild, with either reader |
| Cost of an update | base reader: 16 ms median on tasks of 5 requirements and 11 records (full re-reading: 167 ms); 36 ms on tasks of 13 requirements and 34 records (960 ms). Shipped larger reader: 59 ms and 118 ms on the Apple GPU, 210 ms and 433 ms on its CPU |
| Larger reader (shipped) | ModernBERT-large in place of base, same recipe: +2.0 accuracy points [+1.1, +2.9] on standard tasks and +2.2 [+0.7, +3.6] on larger ones, not worse on small ones; stale decisions 13.1% → 11.1% on standard tasks; it reads the record written for a requirement correctly 90.5% of the time (base 88.9%) at 2.7× the update time |
| Against a dense reader with the same backbone, training and wrapper (base reader) | with the learned refiner: +3.6 accuracy points [+2.1, +5.1] and 3.9 points fewer stale decisions [1.4, 6.6] on standard tasks, at 6.5% of the tokens. On larger tasks +5.4 points [+2.0, +8.7], but the stale-decision difference was not significant, so that hypothesis is **not supported**. The shipped calibrated rules: +2.2 points [+0.6, +3.9] and 3.6 fewer stale decisions [1.0, 6.2] on standard tasks; +5.5 points [+2.3, +8.7] on larger ones |
| Against a public NLI model in the same wrapper (base reader) | about +19 to +20 points; it reads 45% of unrelated documents as evidence, this model 0.8% |
| Learned refiner | small gains on average, but one of three training seeds failed on larger tasks and all lose to calibrated rules on small tasks. **The release ships calibrated rules** |
| Learning a policy instead of executing it | a recurrent network did as well as the executed policy: hypothesis **not supported** |
| Self-evolution | exact corrections and a threshold with a proven error bound work as specified; gated consolidation adopted no change it could verify as safe: hypothesis **not supported** |
| Dependency proposer | within its miss bound on the tasks it was calibrated for, just outside it on larger ones: **not supported**; off by default |
| Conflicts that follow from a consequence | recognised 35% of the time by the base reader and 45% by the larger one, against 77% and 88.5% for direct conflicts |
| "The latest version" | read **unsafely**: name the specific version in requirements |
| Where the error comes from | with every reading correct, the exact parts were right at every step of every test task, with either reader. All error is the reader's: 89% (base) and 90.5% (large) correct on the record written for a requirement. Messages on the same subject that settle nothing are read as settling it 22% (base) and 20% (large) of the time |
| More training | 200,000 more examples on the same corpora did not help (slightly worse on development data). A larger encoder was not trained |
| Without a GPU | base reader: 0.09 s per update on standard tasks, 0.19 s on larger ones, 2.5 GB of memory; shipped larger reader: 0.21 s, 0.43 s, 4.8 GB, on this laptop's processor |

## How it works

```
evidence event ──> ledger (exact)            versions, withdrawals, duplicates, conflicts
                     │
                     ▼
      edge model (learned, 395M)             one (claim, record) pair -> supports / refutes / settles nothing
                     │   cached per pair
                     ▼
      calibration (2 numbers) ──> precedence policy (exact, differentiable)
                     │
                     ▼
      requirement logic (exact)              AND / OR / NOT / k-of-n over three-valued statuses
                     │
                     ▼
      planner (exact)                        what is ready, what to ask, what need not be asked
```

Learned parts interpret language. Exact parts handle identity, versions, precedence, logic and
arithmetic. A learned refiner (41K parameters) can take the calibration step's place; it was
trained and evaluated, and it is not the default (see Results in brief). Every quantity a decision depends on, including the model weights and the schema, is a
node in one dependency graph, so a change to any of them recomputes what it reaches and nothing
else (`docs/PROOFS.md`).

## Quick start

```bash
git clone https://github.com/jithinsaireddy/ease-delta && cd ease-delta
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11 or later; developed on 3.12
pip install -e ".[dev]"
pytest                      # the test suite; needs no model and no network
ruff check src scripts tests
```

The trained models are on the Hugging Face Hub: `jithinpothireddy21/ease-delta` (the shipped
395M reader) and `jithinpothireddy21/ease-delta-base` (the 149M reader: 2.7 times faster, half
the memory, about 2 points less accurate on tasks).

```bash
hf download jithinpothireddy21/ease-delta --local-dir release
```

Developed and run on macOS with Apple silicon only. `requirements-lock.txt` lists the exact
versions used.

After training (below), or with a trained model in `release/`:

```bash
ease demo  --model release/edge --aggregator release/aggregator
ease serve --model release/edge --aggregator release/aggregator --data ~/.ease
```

The demo plays a client hand-off through twelve events: approvals, a withdrawal, a duplicate, a
late copy of an old message, a changed date, a bounced payment. It prints what the model actually
concluded at each step.

`ease serve` also serves a page at `http://127.0.0.1:8791/` for doing the same by hand: start a
task from a template (client hand-off, onboarding, campaign launch, event, support follow-up,
software release, group trip), paste messages or import an email file, see what is ready and
why, correct a reading the model got wrong, and send someone a one-question link to confirm a
requirement. The page loads nothing from the internet.

For several teams on one server, `ease serve --multi` gives each team a private workspace behind
its own key, with quotas, signed webhooks and backups: see `docs/DEPLOYMENT.md`. A Dockerfile
builds a CPU-only image.

**Write passages that say who is speaking.** The reader sees only a record's text, never its
metadata. "We approve the design" does not establish that *the client* approved. "Email from the
client: we approve the design" does. `POST /tasks/{id}/documents` takes an `attribution` and puts
it in front of every passage it cuts from a document.

### API

```bash
curl -s localhost:8791/tasks -H 'content-type: application/json' -d '{
  "task_id": "handoff",
  "predicates": [{"id": "approved", "text": "The client has approved the design.", "prior": 0.3, "ask_cost": 0.2}],
  "actions": [{"id": "send", "description": "Send the packet", "requires": "approved", "value_success": 10, "cost_failure": 25}]
}'

curl -s localhost:8791/tasks/handoff/events -H 'content-type: application/json' -d '{
  "record_id": "email-14", "revision": 1, "source_id": "client", "authority": 2, "valid_from": 1,
  "text": "We approve the design as presented, please go ahead."
}'

curl -s localhost:8791/tasks/handoff/explain/send      # which records the answer rests on
curl -s localhost:8791/tasks/handoff/question          # what is worth asking, and what is not
curl -s localhost:8791/tasks/handoff/verify            # cached state == full rebuild?
```

| Endpoint | Purpose |
|---|---|
| `POST /tasks` | declare a task |
| `POST /tasks/{id}/events` | add, correct or withdraw a record (`"text": null` withdraws) |
| `POST /tasks/{id}/documents` | split a document into attributed passages and deliver them; a revision withdraws passages that are gone |
| `GET /tasks/{id}` | statuses, beliefs, records, endorsement threshold |
| `GET /tasks/{id}/explain/{action}` | the predicates and records an assessment rests on |
| `GET /tasks/{id}/question` | the question with the highest value, and questions that cannot matter |
| `POST /tasks/{id}/certificate` | can any answer to these questions change this proposal? |
| `POST /tasks/{id}/readings` | a person states what a record establishes; takes effect at once |
| `DELETE /readings/{item}` | withdraw that statement; behaviour returns exactly to what it was |
| `POST /tasks/{id}/endorsements/{action}`, `POST /tasks/{id}/verdicts` | report whether an endorsed action was right |
| `POST /evolve/consolidate`, `POST /evolve/rollback` | learn from accumulated feedback behind a gate; undo |
| `PUT /tasks/{id}/schema`, `POST /tasks/{id}/clock` | change the requirements; advance time |
| `GET /templates`, `POST /tasks/from-template` | start a task from a workflow template |
| `POST /tasks/{id}/import` | an `.eml`, `.mbox`, `.txt` or `.md` file as attributed passages |
| `POST /tasks/{id}/confirmations`, `GET/POST /confirm/{workspace}/{token}` | a one-question link for someone to confirm a requirement; the answer is read exactly |
| `GET /workspace`, `PUT /workspace/webhook` | quota and usage; a signed webhook when a proposal changes |
| `/admin/...` | workspaces, keys, quotas and backups (`--multi` with an admin key) |

### Python

```python
from ease.engine import Engine
from ease.ledger import Event
from ease.schema import Action, Predicate, TaskSchema
from ease.scorer import ModelScorer
from ease.aggregate import load_aggregator

schema = TaskSchema("handoff",
    [Predicate("approved", "The client has approved the design.", prior=0.3)],
    [], [Action("send", "Send the packet", "approved")])
engine = Engine(schema, ModelScorer("release/edge", canonical=True), load_aggregator("release/aggregator"))

report = engine.deliver(Event("email-14", 1, "We approve the design as presented.",
                              source_id="client", authority=2, valid_from=1.0))
print(report.changed_actions, engine.assessment("send").disposition)
```

## Learning after deployment

The encoder (395M parameters; a 149M version is published too) is never updated in use. What changes is small, versioned and
reversible.

| Mechanism | Takes effect | Safeguard |
|---|---|---|
| Correction memory | next query | an item can be deleted, restoring earlier behaviour exactly; nothing is generalised to similar cases until a gate has shown on held-out feedback that it helps |
| Endorsement threshold | after each verdict | realised error rate among endorsed actions is bounded for every sequence of cases (`docs/PROOFS.md`, T5) |
| Consolidation | on request | adopted only if single readings *and whole validation tasks* (set suites) do not get worse and held-out feedback improves; the report counts negative flips (readings the previous version got right), and a ceiling on them can be set; every version keeps its parent and can be rolled back. In the experiments it adopted nothing when starting from calibrated rules |

## Training

Everything runs on one Apple-silicon laptop. Data is streamed from the Hugging Face Hub; nothing
is downloaded in full.

```bash
python -m ease.train.stage_a --config configs/stage_a.yaml      # base edge model, about 2.5 h
bash scripts/run_experiments.sh                                 # pre-registered experiments, about 4.5 h
python -m ease.train.stage_a --config configs/stage_a_large.yaml  # large edge model, about 5 h
bash scripts/run_experiments_large.sh                           # the addendum's comparison, about 2.5 h
python scripts/build_set_suites.py                              # set suites for the consolidation gate
python scripts/package_release.py --out release                 # release directory and model card
```

| Corpus | Licence | Use |
|---|---|---|
| VitaminC | CC BY-SA 3.0 | train, test |
| MNLI | CC BY 3.0 / CC BY-SA 3.0 / MIT / other, by genre | train, test |
| WANLI | CC BY 4.0 | train, test |
| SNLI | CC BY-SA 4.0 | evaluation only (its labelling convention is wrong for evidence) |
| ANLI | CC BY-NC 4.0 | evaluation only (non-commercial) |
| STALE | CC BY 4.0 | evaluation only |

Training resumes exactly after an interruption (`--resume`): the data stream's state is part of
every checkpoint.

## Layout

```
src/ease/
  ledger.py        versioned evidence, order-independent merge
  graph.py         incremental evaluation with early cutoff
  schema.py        predicates, gates, actions
  logic.py         exact three-valued and probabilistic evaluation
  scorer.py        edge model inference and caching; canonical shapes
  aggregate.py     precedence policy; rule, refined and neural aggregators
  engine.py        ties the above together
  planner.py       value of information; stability certificates
  proposer.py      dependency proposals with a conformal miss bound (off by default)
  evolve/          memory, threshold tracker, gated consolidation, versions
  ingest.py        documents into self-contained passages
  runtime.py       persistence, limits, audit log
  service/api.py   HTTP API (local, or many workspaces behind keys)
  service/workspaces.py  workspaces, keys, quotas, webhooks, backups
  service/confirmations.py  one-question confirmation links
  service/ui.html  the page; no external resources
  templates.py     workflow templates
  importers.py     .eml / .mbox / text files into attributed passages
  data/            streaming corpora, atoms, episode generator
  train/           stage_a (edge), stage_b (aggregators), dense (baseline)
  eval/            metrics and the evaluation harness
scripts/           one script per pre-registered hypothesis, and the report generator
docs/
  PREREGISTRATION.md   hypotheses and decision rules, written before any test evaluation
  RESULTS.md           generated from the measurements
  DEVIATIONS.md        what was done differently from the plan
  LIMITATIONS.md       what was not measured and what can break
  PROOFS.md            what is proved, under which assumptions, and which test checks it
  RESEARCH.md          sources, checked; prior art; findings that changed the design
  MODEL_CARD.md
tests/             property-based and unit tests
```

## Licence

Code: Apache-2.0. Model weights derive from ModernBERT-base (Apache-2.0) and from corpora under
the licences above; VitaminC and part of MNLI are share-alike. Creative Commons' 2025 guidance on
AI training describes releasing a model trained on share-alike material under the same licence
as the cautious course; whether weights are an adaptation of their training text is not settled.
The weights' licence is the owner's decision and is stated in the model card once made.
