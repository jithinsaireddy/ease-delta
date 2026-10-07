# HTTP API

```bash
ease serve --model release/edge --aggregator release/aggregator                 # one person: 127.0.0.1:8791, no key
ease serve --model release/edge --aggregator release/aggregator --multi \
           --data /srv/ease --host 0.0.0.0                                      # workspaces: every request carries a key
```

With `--multi`, send `Authorization: Bearer <workspace key>` on every request below except the confirmation pages.
Create workspaces with `ease workspace create` ([deployment](DEPLOYMENT.md)). Interactive documentation is served at
`/docs` while the server runs.

Errors are JSON `{"detail": "..."}`: 401 without a valid key, 404 for an unknown task, record or action, 409 for a
task that exists, 422 for invalid input, 429 over a quota or rate limit.

## Tasks

```bash
# from a template
curl -s localhost:8791/tasks/from-template -H 'content-type: application/json' \
  -d '{"template": "client-onboarding", "task_id": "acme", "params": {"client": "Acme"}}'

# or declared in full
curl -s localhost:8791/tasks -H 'content-type: application/json' -d '{
  "task_id": "handoff",
  "predicates": [{"id": "approved", "text": "The client has approved the design.", "prior": 0.3, "ask_cost": 0.2},
                 {"id": "date", "text": "The client has confirmed the delivery date."}],
  "gates": [{"id": "ready", "op": "AND", "children": ["approved", "date"]}],
  "actions": [{"id": "send", "description": "Send the packet", "requires": "ready",
               "value_success": 10, "cost_failure": 25}]
}'
```

| Endpoint | |
|---|---|
| `GET /templates` | the templates and their parameters |
| `POST /tasks/from-template` | `{template, task_id, params}` |
| `POST /tasks` | a full definition: `predicates`, `gates` (`op` is `AND`, `OR`, `NOT` or `ATLEAST` with `k`), `actions` |
| `GET /tasks` | task ids |
| `GET /tasks/{id}` | `assessments` (per action: `disposition`, `confidence`, `p_success`, ...), `beliefs` (per requirement: `status`, `probabilities`, `decisive` records), `records`, `threshold` |
| `PUT /tasks/{id}/schema` | change the requirements or actions; only what changed is recomputed |

## Evidence

```bash
curl -s localhost:8791/tasks/handoff/events -H 'content-type: application/json' -d '{
  "record_id": "email-14", "revision": 1, "source_id": "client", "authority": 2,
  "text": "Email from the client: we approve the design as presented, please go ahead."}'
```

| Endpoint | |
|---|---|
| `POST /tasks/{id}/events` | `{record_id, revision, text, source_id, authority, valid_from, valid_until, span, about}`; `"text": null` withdraws |
| `POST /tasks/{id}/documents` | `{doc_id, revision, text, attribution, source_id, authority}`: cut into attributed passages; a new revision withdraws passages that are gone |
| `POST /tasks/{id}/import` | `{filename, content_base64}` or `{filename, text}`: `.eml`, `.mbox`, `.txt`, `.md` |
| `POST /tasks/{id}/clock` | `{now}`: advance time, for records with `valid_until` |

Each returns the change (`ledger` outcome, `edges_scored`, `changed_actions` as `[action, before, after]`,
`seconds`) and the new assessments.

## Reading the state

| Endpoint | |
|---|---|
| `GET /tasks/{id}/explain/{action}` | the requirements an action rests on and the records each rests on, with their text |
| `GET /tasks/{id}/question` | `question` (the set worth asking, its value and cost) and `unnecessary` (per action, what cannot change it) |
| `POST /tasks/{id}/certificate` | `{action_id, free}`: can any answer to these requirements change this proposal? |
| `GET /tasks/{id}/verify` | cached state against a full rebuild: `not_bitwise_equal` should be 0 |

## Corrections and feedback

| Endpoint | |
|---|---|
| `POST /tasks/{id}/readings` | `{record_id, predicate_id, establishes}` with `supports`, `refutes` or `settles_nothing`; returns `memory_item` |
| `DELETE /readings/{item}` | withdraw that correction |
| `POST /tasks/{id}/endorsements/{action}` | a READY action is being acted on |
| `POST /tasks/{id}/verdicts` | `{action_id, was_wrong}`: how it turned out; moves the readiness bar |
| `POST /evolve/consolidate`, `POST /evolve/rollback` | learn from accumulated corrections behind the gate; undo the last adoption |

## Confirmation links

Ask someone outside the team to confirm one requirement. The page they open shows only that requirement's text.

```bash
curl -s localhost:8791/tasks/acme/confirmations -H 'content-type: application/json' \
  -d '{"predicate_id": "access_confirmed", "to_name": "Dana at Acme"}'
# {"url": "http://localhost:8791/confirm/local/7Hq...", "asks": "Acme has given access to ...", "expires": ...}
```

| Endpoint | |
|---|---|
| `POST /tasks/{id}/confirmations` | `{predicate_id, to_name, authority, ttl_hours}`: a single-use link (default 72 hours) |
| `GET /tasks/{id}/confirmations` | the links of a task and their answers |
| `GET /confirm/{workspace}/{token}` | the page the person opens; no key needed |
| `POST /confirm/{workspace}/{token}` | `{answer: "yes" \| "no" \| "unsure", note}`; yes and no enter the ledger as an exact reading |

## Workspaces (`--multi`)

| Endpoint | |
|---|---|
| `GET /workspace` | name, quota and today's usage |
| `PUT /workspace/webhook` | `{url}`: a public https address that receives `POST`s signed with `X-EASE-Signature: sha256=HMAC(secret, body)` when a proposal changes; returns the secret |
| `GET /info` | model versions, corrections, readiness bar |
| `GET /admin/workspaces`, `POST /admin/workspaces`, `POST /admin/workspaces/{id}/rotate-key`, `PUT /admin/workspaces/{id}/quota`, `POST /admin/backup` | administration, with the admin key (`EASE_ADMIN_KEY`) |

`GET /health` answers without a key.
