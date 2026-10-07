# Python API

```python
from ease import EvidenceReader, Tracker, define, Reading
```

Two classes cover ordinary use. Both are thin layers over `ease.runtime.Runtime` and `ease.engine.Engine`, which
remain available for finer control.

## `EvidenceReader`

The trained reader of a release, with the calibration the system applies to it.

### `EvidenceReader.from_pretrained(name_or_path=None, device=None, exact=True, revision=None)`

| Argument | |
|---|---|
| `name_or_path` | a Hugging Face model id (`"jithinpothireddy21/ease-delta"`, `"jithinpothireddy21/ease-delta-base"`) or a release directory. Default: the `EASE_MODEL` environment variable, else `jithinpothireddy21/ease-delta` |
| `device` | `"cpu"`, `"mps"`, `"cuda"`; default: the best available, or `EASE_DEVICE` |
| `exact` | `True` fixes the shape of every computation, so a reading never depends on what else was read with it and cached state equals a full rebuild bit for bit; `False` batches freely and is faster (values vary by about 1e-5) |
| `revision` | a Hub revision (branch, tag or commit) |

A Hub model is downloaded once and cached. The plain `transformers` checkpoints (`ease-delta-reader`) are not
release directories; load those with `transformers` ([reader guide](reader.md)).

### `reader.read(claim, passage) -> Reading`

### `reader.read_many([(claim, passage), ...]) -> list[Reading]`

`Reading` has `supports`, `refutes`, `not_enough_info` (calibrated probabilities that sum to 1) and `label`
(`"SUPPORTS"`, `"REFUTES"` or `"NOT_ENOUGH_INFO"`). The claim comes first.

## `Tracker`

One task, kept current as its records are added, changed and withdrawn.

### Creating one

```python
Tracker.define(requirements, actions, reader, task_id="task", data_dir=None, ready_at=None)
Tracker.from_template(template, reader, task_id=None, data_dir=None, ready_at=None, **params)
Tracker(definition, reader, data_dir=None, ready_at=None)
```

- `requirements`: `{"approved": "Acme has approved the final design.", ...}`. A value can also be a dict with
  `text`, `prior` (probability it holds with no evidence, default 0.5) and `ask_cost` (cost of asking, default 1.0).
- `actions`: `{"send": "approved and date"}`, or `{"send": {"requires": "...", "description": "...", "value": 10,
  "cost": 25}}`. `value` is what the action is worth when its requirements hold, `cost` what it costs when they do
  not (defaults 10 and 25).
- `template`: one of `ease.templates.TEMPLATES` ([list](getting-started.md#5-start-from-a-template)), with its
  parameters as keywords: `Tracker.from_template("client-onboarding", reader=reader, client="Acme")`.
- `definition`: a full task definition, as a dict or `ease.schema.TaskSchema` (the format `POST /tasks` takes).
- `data_dir`: where to keep everything; reopening the same `task_id` there restores it. Default: a temporary
  directory, removed by `close()`.
- `ready_at`: the readiness bar to start from (default 0.9, from the release's settings).

#### Requirement expressions

```
expr     := term ("or" term)*
term     := factor ("and" factor)*
factor   := "not" factor | "(" expr ")" | "at least" N "of" "(" expr ("," expr)* ")" | requirement-id
```

Keywords are case-insensitive; ids are matched exactly. `a and b and c` becomes one gate. Malformed expressions and
unknown ids raise `ValueError` naming the problem.

### Evidence

| Method | |
|---|---|
| `add(record_id, text, *, source="", authority=0, about=None, valid_from=None) -> Update` | add a record, or replace it with a new revision if the id exists |
| `update(...)` | the same as `add`, for readability |
| `withdraw(record_id) -> Update` | the record stops counting |
| `add_document(doc_id, text, *, attribution=None, source="", authority=0) -> dict` | cut a document into passages, each starting with `attribution` ("Email from the client"); adding the same `doc_id` again replaces it and withdraws passages that are gone |
| `add_file(path, *, attribution=None, source="", authority=0) -> list[dict]` | an `.eml`, `.mbox`, `.txt` or `.md` file; emails keep sender, date and subject in the text |

`about` is a list of requirement ids the record concerns; without it the record is read against all of them.
`authority` (0 to 9) ranks records that disagree; among equal authority the later one wins. `valid_from` is a Unix
time; the default is now.

`Update` has `record_id`, `outcome` (`"applied"`, `"duplicate"`, `"stale"`, ...), `changed` (a list of
`(action, before, after)`), `pairs_read` and `milliseconds`, and prints as one line.

### State

| Method | |
|---|---|
| `status() -> Status` | everything below, at once |
| `explain(action) -> str` | the requirements the action rests on, and the records each of those rests on |
| `records() -> dict` | each record's revision, whether it is active, and its text |
| `ready_at` | the current readiness bar |
| `verify() -> bool` | `True` when every cached value equals an independent full rebuild |

`Status` has `actions` (id → `ActionStatus`: `disposition`, `confidence`, `p_success`, `description`),
`requirements` (id → `RequirementStatus`: `status`, `satisfied`, `violated`, `open`, `rests_on`, `text`),
`question` (the requirement texts worth asking about, or `None`), `unnecessary` (per action, requirements whose
answer cannot change it), and the shortcuts `ready`, `blocked` and `needs_info` (lists of action ids). It prints as
a table.

### People

| Method | |
|---|---|
| `confirm(requirement, by, holds=True, note="", authority=0) -> Update` | a person's answer, entered as an attributed record and read exactly |
| `correct(record_id, requirement, establishes) -> int` | state that a record `"supports"`, `"refutes"` or `"settles_nothing"` about a requirement; returns an id |
| `undo(correction_id) -> bool` | withdraw a correction; every number returns exactly to what it was |
| `endorse(action) -> bool` | record that a READY action is being acted on |
| `verdict(action, was_wrong) -> float` | report how it turned out; returns the new readiness bar |

### Lifetime

`close()` releases files and removes a temporary directory. A tracker is a context manager:

```python
with Tracker.from_template("software-release", reader=reader, version="v2.3") as release:
    release.add("ci", "CI report for v2.3: all 412 required tests passed.")
    print(release.status())
```

## Lower levels

| Module | For |
|---|---|
| `ease.runtime.Runtime` | many tasks under one directory, consolidation, rollback, feedback export |
| `ease.engine.Engine` | one task in memory, with full control of events, linking and configuration |
| `ease.service.workspaces.Service` | many runtimes behind keys, as the multi-workspace server uses |
| `ease.export.to_sequence_classifier` | the reader as a plain `ModernBertForSequenceClassification` |

The guarantees in [PROOFS.md](PROOFS.md) are stated for the engine; everything above inherits them.
