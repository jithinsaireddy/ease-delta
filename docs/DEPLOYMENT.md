# Running EASE-Delta for other people

Two modes of the same service.

| | `ease serve` | `ease serve --multi` |
|---|---|---|
| Who | one person on their own machine | several teams behind one server |
| Keys | none; binds to 127.0.0.1 | every request carries `Authorization: Bearer <workspace key>` |
| Files | `--data` directory as before | `--data/workspaces/<id>/` per workspace; the model's reading cache is shared |
| Separation | — | each workspace has its own tasks, corrections, endorsement threshold, adopted versions and audit log |

Nothing in either mode sends, pays, submits or executes. The service reads evidence and reports.

## Workspaces

```bash
ease workspace create --data /srv/ease --id acme --name "Acme agency"   # prints the key once; only its hash is stored
ease workspace list   --data /srv/ease
ease workspace rotate-key --data /srv/ease --id acme                    # the old key stops working at once
```

Each workspace has a quota in its `workspace.json` (tasks, records per task, text size, events per
day, imported characters per day, open confirmation links, requests per minute). The service
answers 429 when a quota is reached and `GET /workspace` shows today's usage.

If `EASE_ADMIN_KEY` is set when the service starts, the same operations are available over HTTP
under `/admin/...` with that key. Without it, administration happens on the server only.

## What a team can do with its key

- Create a task from a template (`GET /templates`, `POST /tasks/from-template`): client hand-off,
  client onboarding, campaign launch, event coordination, support follow-up, software release,
  group trip. Or declare a task in full (`POST /tasks`).
- Add evidence: single records (`POST /tasks/{id}/events`), documents cut into attributed passages
  (`POST /tasks/{id}/documents`), or files (`POST /tasks/{id}/import`): `.eml`, `.mbox`, `.txt`,
  `.md`. Email import keeps the sender, date and subject in the text the reader sees and drops
  quoted earlier messages.
- Ask a person outside the team to confirm one requirement (`POST /tasks/{id}/confirmations`):
  the link shows only that requirement, works once, expires, and the answer enters the ledger
  under the person's name as an exact reading.
- Be told when a proposal changes (`PUT /workspace/webhook`): a signed POST
  (`X-EASE-Signature: sha256=HMAC(secret, body)`) to a public https address, retried three times.
- Everything else the local API offers: explanations, questions, certificates, corrections,
  endorsements and verdicts, consolidation and rollback.

## Container

```bash
docker build -t ease-delta .
docker run --rm -p 127.0.0.1:8791:8791 -e EASE_MULTI=1 -e EASE_ADMIN_KEY=change-me \
  -v $PWD/release:/models:ro -v ease-data:/data ease-delta
```

or `EASE_ADMIN_KEY=change-me docker compose up`. The image runs on the CPU; on a laptop-class
processor an update takes about 0.1 to 0.2 s and the process uses about 2.5 GB
(`docs/RESULTS.md`, section 8k). Put a TLS-terminating reverse proxy in front of it; the service
itself speaks plain HTTP.

## Backups

```bash
ease backup --data /srv/ease --out /srv/ease-backups      # a dated directory with every database and setting
```

SQLite's online backup is used, so a copy taken while the service runs is a valid database. To
restore, stop the service and copy a backup directory over `--data`.

## Limits of this setup

- One process, one model, one lock: requests that need the model are served one at a time. That
  is enough for a pilot with a handful of teams; it is not a design for thousands of concurrent
  users.
- Keys are bearer tokens: anyone holding a key is the workspace. Rotate a key that may have leaked.
- Tested on macOS with Apple silicon and in the Linux container built from the Dockerfile (CPU).
- Accounts, passwords, invitations and billing do not exist. A workspace key is the whole of
  access control.
