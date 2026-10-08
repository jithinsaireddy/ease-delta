# Changelog

## Unreleased

- **Misreading reports become evaluation data**: `scripts/misreadings.py sync` rebuilds `evals/misreadings.jsonl`
  from the issues labelled `misreading`, keeping only reports whose author consented to publication (CC BY 4.0);
  `scripts/misreadings.py score` measures both released readers on it.
- `scripts/usage_stats.py` also reports the two readers and the demo Space.
- **The local page needs no record ids**: it opens on starting a task from a template when there are none, adds a
  message with who it is from, and edits or withdraws any message from the list, numbering ids and revisions itself;
  an edit keeps the message's original time, so a correction never jumps ahead of later messages. The full record
  form is under Advanced. The page shows each requirement and action in words.
- API: `GET /tasks/{id}/records` returns what each record says and who said it; `GET /tasks/{id}` adds `text` to each
  requirement and `description` to each action.
- The demo page, the online app and the local page offer a way to report a wrong reading.

## 0.2.0 (2026-10-07)

- **Short API**: `ease.EvidenceReader` (read a claim against a passage; models load from the Hugging Face Hub by name)
  and `ease.Tracker` (keep a task current: `define`, `from_template`, `add`, `update`, `withdraw`, `add_document`,
  `add_file`, `status`, `explain`, `confirm`, `correct`, `undo`, `endorse`, `verdict`). Requirements are written as
  expressions: `approved and (paid or waived)`, `at least 2 of (a, b, c)`.
- **The reader as a plain `transformers` model**: `jithinpothireddy21/ease-delta-reader` and `-reader-base`, exact
  conversions loadable with `pipeline`, `AutoModelForSequenceClassification` and sentence-transformers' `CrossEncoder`;
  ONNX weights (fp32 and int8) for onnxruntime and Transformers.js.
- Tokenizer files now load in transformers 4.x as well as 5.x (identical token ids on 93,045 evaluation pairs).
- A demo page that runs the reader in the browser and replays a recorded task; a Colab notebook; a Gradio demo app.
- Documentation: getting started, concepts, Python and HTTP references, reader guide, FAQ, results at a glance.
- Fixes: "unnecessary" questions are now certified against every combination of the other open requirements.

## 0.1.0 (2026-10-03)

- The system: exact versioned ledger, incremental dependency graph, precedence policy, three-valued logic, planner,
  correction memory, readiness bar with an error bound, gated consolidation, versions and rollback.
- Pre-registered evaluation of the base reader (H1-H9) and the larger-reader addendum (H10-H12); the larger reader
  ships.
- Service: local page and HTTP API; workspaces behind keys with quotas, templates, email import, confirmation links,
  signed webhooks and backups; Dockerfile.
