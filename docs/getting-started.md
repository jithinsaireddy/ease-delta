# Getting started

Ten minutes from nothing to a task that keeps itself current.

## 1. Install

Python 3.11 or later. A GPU is not needed; on a laptop processor an update takes a fraction of a second.

```bash
pip install "ease-delta @ git+https://github.com/jithinsaireddy/ease-delta"
```

The first time a model is used it is downloaded from the Hugging Face Hub and cached:
`jithinpothireddy21/ease-delta` (1.6 GB, the default) or `jithinpothireddy21/ease-delta-base` (0.6 GB, about 2.7
times faster and about 2 points less accurate). To use a copy you already have, pass its directory, or set
`EASE_MODEL=/path/to/release`.

## 2. Read one passage

```python
from ease import EvidenceReader

reader = EvidenceReader.from_pretrained()
reader.read("The client has approved the final design.",
            "Email from the client: we approve the final design, please go ahead.")
# Reading(SUPPORTS: supports 0.93, refutes 0.00, not enough info 0.07)
```

The claim comes first and the passage second. The answer is one of *supports*, *refutes* or *not enough info*,
with calibrated probabilities.

## 3. Track a task

A task is a set of **requirements** (plain sentences that can be true or false) and **actions** (what you want to
do, and which requirements it needs).

```python
from ease import Tracker

task = Tracker.define(
    requirements={
        "brief": "Acme has sent the project brief.",
        "files": "Acme has supplied the files that were requested.",
        "access": "Acme has given access to the website we are rebuilding.",
    },
    actions={"start_work": "brief and files and access"},
    reader=reader,
)
print(task.add("mail-1", "Email from Acme: the brief is attached, and the brand files are in the shared folder."))
print(task.status())
```

`requires` is an expression over requirement ids: `and`, `or`, `not`, parentheses, and
`at least 2 of (a, b, c)`. Each action can also carry a description and what it is worth:

```python
actions={"start_work": {"requires": "brief and files and access", "description": "Start the rebuild",
                        "value": 10, "cost": 25}}
```

`value` is what doing the action is worth when its requirements hold and `cost` what it costs when they do not.
They decide which questions are worth asking (step 6).

## 4. Add, change and withdraw messages

```python
task.add("mail-2", "Email from Acme: website access for your team is set up; logins sent separately.")
task.update("mail-2", "Email from Acme: sorry, the access we set up was for the wrong site.")   # a new revision
task.withdraw("mail-1")                                                                     # it no longer counts
```

- **Say who is speaking, in the text.** The reader sees only the text of a message. "We approve the design" does
  not establish that the client approved; "Email from the client: we approve the design" does.
- **Adding an id that exists replaces it** with a new revision. Duplicates and older copies change nothing and
  cost nothing.
- **Say what a message is about when you know**: `task.add("mail-3", text, about=["access"])`. Without `about` a
  message is read against every requirement, which is what you want for imported mail. When several requirements
  are of one kind (one per speaker, one per participant), `about` keeps a message about one person from being read
  as answering for another.
- **`authority`** (0 to 9) decides between messages that disagree: a higher authority wins, and among equal
  authority the later message wins.

## 5. Start from a template

```python
from ease.templates import TEMPLATES
print({k: v.name for k, v in TEMPLATES.items()})

task = Tracker.from_template("client-onboarding", reader=reader, client="Acme")
event = Tracker.from_template("event-coordination", reader=reader, speakers=["Ann Lee", "Bo Chen"], min_speakers=1)
```

| Template | Parameters |
|---|---|
| `client-handoff` | `client` |
| `client-onboarding` | `client` |
| `campaign-launch` | `client` |
| `event-coordination` | `speakers` (list), `min_speakers` (0 means all) |
| `support-followup` | `customer` |
| `software-release` | `version` |
| `group-trip` | `participants` (list), `min_participants` (0 means all) |

## 6. Ask, and let people answer

```python
st = task.status()
st.ready, st.blocked, st.needs_info     # lists of action ids
st.question                             # the requirement texts worth asking about, or None
task.confirm("access", by="Dana at Acme")              # a person's answer, read exactly
task.confirm("access", by="Dana at Acme", holds=False)  # or a denial
```

An action is READY only when its requirements are at least 90% likely to hold (the bar moves with feedback;
see [concepts](concepts.md#ready-blocked-needs-info)). Below that, the tracker names the single question whose
answer is worth its cost. A confirmation enters as a message attributed to the person and is read exactly, not
interpreted.

## 7. Correct a misreading

```python
print(task.explain("start_work"))                         # the requirements and the messages they rest on
cid = task.correct("mail-2", "access", "refutes")         # "this message refutes that requirement"
task.undo(cid)                                            # back exactly to where it was
```

## 8. Import email

```python
task.add_file("thread.mbox")       # also .eml, .txt, .md
```

Each email becomes one document, cut into passages that keep the sender, date and subject in their text
("Email from Dana Ortiz on 2026-03-02, subject: Design v5: ..."). Quoted earlier messages are dropped. Importing
the same message again replaces it.

## 9. Keep the state

```python
task = Tracker.define(..., reader=reader, task_id="acme-onboarding", data_dir="~/ease-data")
```

With `data_dir`, everything (messages, revisions, corrections, readings, the readiness bar, an audit log) is kept in
SQLite files there and comes back when you open the same `task_id` again. Without it, the tracker lives in a
temporary directory.

## 10. The page and the service

```bash
hf download jithinpothireddy21/ease-delta --local-dir release      # or any release directory
ease demo  --model release/edge --aggregator release/aggregator    # a client hand-off, twelve events, in the terminal
ease serve --model release/edge --aggregator release/aggregator    # http://127.0.0.1:8791
```

`ease serve` gives you a page for doing all of the above without writing code. Start a task from a template (the
page opens there when there are none), paste each message as it arrives and say who it is from, and edit or withdraw
any message from the list; the page numbers messages and their versions for you, and the full record form is under
*Advanced*. It also makes one-question confirmation links you can send to someone outside your team, and serves an
HTTP API ([reference](http-api.md)). For several teams on one server, see [deployment](DEPLOYMENT.md).

## Faster, or exact

`EvidenceReader.from_pretrained(exact=True)` (the default) fixes the shape of every computation, so a reading never
depends on what else was read with it and the cached state equals a full rebuild bit for bit. `exact=False` batches
freely and is faster; values then vary by about 1e-5.
