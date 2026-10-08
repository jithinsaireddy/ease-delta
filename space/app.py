"""EASE-Delta, online: keep a task's decisions current as messages change, and read evidence.

Runs on a CPU. Everything shown is computed by the released model when you click; nothing is scripted
to look right, so a misreading shows as a misreading.
"""

from __future__ import annotations

import html
import os
import threading
import time
from urllib.parse import quote

import gradio as gr
import torch

torch.set_num_threads(max(1, os.cpu_count() or 2))

from ease import EvidenceReader, Tracker  # noqa: E402
from ease.demo import packet_schema, scenario  # noqa: E402
from ease.templates import TEMPLATES, render  # noqa: E402

MODEL = os.environ.get("EASE_MODEL", "jithinpothireddy21/ease-delta")
PUBLIC_NLI = "tasksource/ModernBERT-base-nli"
GITHUB = "https://github.com/jithinsaireddy/ease-delta"
RESULTS = GITHUB + "/blob/main/docs/RESULTS.md"
REPORT = GITHUB + "/issues/new?template=misreading.yml"  # the Misreading form; its text fields fill from the URL

READER = EvidenceReader.from_pretrained(MODEL, exact=False)
LOCK = threading.Lock()  # one model, read by one request at a time


def _load_public():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PUBLIC_NLI)
    model = AutoModelForSequenceClassification.from_pretrained(PUBLIC_NLI).eval()
    id2 = {int(k): v.lower() for k, v in model.config.id2label.items()}
    cols = [next(i for i, lab in id2.items() if w in lab) for w in ("entail", "contradict", "neutral")]
    return tok, model, cols


PUB_TOK, PUB_MODEL, PUB_COLS = _load_public()


# ---------------------------------------------------------------------------
# Reading one passage
# ---------------------------------------------------------------------------

def read_passage(claim: str, passage: str):
    if not claim.strip() or not passage.strip():
        raise gr.Error("Enter a claim and a passage.")
    with LOCK, torch.no_grad():
        r = READER.read(claim, passage)
        enc = PUB_TOK(passage, claim, truncation=True, max_length=256, return_tensors="pt")  # NLI order: premise first
        p = torch.softmax(PUB_MODEL(**enc).logits[0], -1)[PUB_COLS].tolist()
    ours = {"supports": r.supports, "refutes": r.refutes, "not enough info": r.not_enough_info}
    theirs = {"supports (entailment)": p[0], "refutes (contradiction)": p[1], "not enough info (neutral)": p[2]}
    said = max(ours, key=ours.get)
    fields = {"title": "Misreading: " + claim.strip()[:80], "requirement": claim.strip(), "message": passage.strip(),
              "model": f"{MODEL} (online demo app)"}
    url = REPORT + "".join(f"&{k}={quote(v, safe='')}" for k, v in fields.items())
    report = (f"Is the EASE-Delta answer wrong? [Report it]({url}). It opens a public GitHub issue with this claim and "
              f"passage filled in, so remove anything private first. On the form, choose what it said (**{said}**) and "
              "what it should have said.")
    return ours, theirs, report


READ_EXAMPLES = [
    ["The client has approved the final design.", "Email from the client: we approve the final design, please go ahead."],
    ["The client has approved the final design.", "Email from the client: we cannot approve the design yet; the colours are wrong."],
    ["The deposit invoice has been paid.", "Office notice: the kitchen will be closed on Friday for maintenance."],
    ["The film grossed more than $550 million worldwide.", "The film went on to gross $545.5 million worldwide."],
    ["Ann Lee will speak at the conference.", "Ann Lee has had to cancel her trip and will not be at the conference."],
    ["The release passed all required tests.", "Test report for build 2.3: 412 passed, 3 failed (payments module)."],
]


# ---------------------------------------------------------------------------
# Keeping a task current
# ---------------------------------------------------------------------------

BADGE = {"READY": "ready", "BLOCKED": "blocked", "NEEDS_INFO": "open", "SATISFIED": "ready", "VIOLATED": "blocked",
         "UNRESOLVED": "open", "CONFLICT": "open"}


def status_html(t: Tracker | None) -> str:
    if t is None:
        return "<div class='ease-empty'>Start a task, or press <b>Play the hand-off story</b>.</div>"
    with LOCK:
        st = t.status()
    e = html.escape
    tau = max(0.5, float(t.runtime.tracker.tau))
    out = ["<div class='ease'>", f"<div class='ease-h'>What can be done &middot; ready at {100 * tau:.0f}% or more</div>"]
    for a in st.actions.values():
        out.append(f"<div class='ease-item'><div class='ease-row'><span class='ease-id'>{e(a.id)}</span>"
                   f"<span class='ease-badge {BADGE[a.disposition]}'>{a.disposition.replace('_', ' ')}</span>"
                   f"<span class='ease-muted'>{100 * a.confidence:.1f}% likely ready</span></div>"
                   f"<div class='ease-muted'>{e(a.description)}</div>"
                   + ("<div class='ease-note'>Ready to propose. Nothing happens until a person approves it.</div>"
                      if a.disposition == "READY" else "") + "</div>")
    if st.question:
        out.append("<div class='ease-ask'><b>Worth asking:</b> " + " &middot; ".join(e(q) for q in st.question) + "</div>")
    out.append("<div class='ease-h'>What must be true</div>")
    for r in st.requirements.values():
        bar = (f"<div class='ease-bar'><span class='s' style='width:{100 * r.satisfied:.1f}%'></span>"
               f"<span class='v' style='width:{100 * r.violated:.1f}%'></span>"
               f"<span class='u' style='width:{100 * r.open:.1f}%'></span></div>")
        rests = f" &middot; rests on {e(', '.join(r.rests_on))}" if r.rests_on else ""
        out.append(f"<div class='ease-item'><div class='ease-row'><span class='ease-id'>{e(r.id)}</span>"
                   f"<span class='ease-badge {BADGE[r.status]}'>{r.status}</span></div>"
                   f"<div>{e(r.text)}</div>{bar}<div class='ease-muted'>satisfied {r.satisfied:.2f} &middot; "
                   f"violated {r.violated:.2f} &middot; open {r.open:.2f}{rests}</div></div>")
    out.append("</div>")
    return "".join(out)


def log_md(log: list[str]) -> str:
    return "\n".join(f"- {line}" for line in reversed(log[-14:])) if log else "_Nothing has happened yet._"


def explain_md(t: Tracker | None, action: str | None) -> str:
    if t is None or not action:
        return ""
    with LOCK:
        ex = t.runtime.explain(t.task_id, action)
    a = ex["assessment"]
    lines = [f"**{action}: {a['disposition'].replace('_', ' ')}** ({100 * a['status']['satisfied']:.1f}% likely ready). "
             f"{ex['description']}", ""]
    for p in ex["predicates"]:
        lines.append(f"- **{p['status']}**: {p['text']}")
        if not p["records"]:
            lines.append(f"  - no record settles this ({p['records_compared']} compared)")
        for r in p["records"]:
            lines.append(f"  - `{r['record_id']}` (revision {r['revision']}): {r['text']}")
    return "\n".join(lines)


def requirement_choices(t: Tracker | None):
    if t is None:
        return []
    return [(f"{p.id}: {p.text}", p.id) for p in t._engine().schema.predicates]


def record_choices(t: Tracker | None):
    if t is None:
        return []
    return [rid for rid, r in t.records().items() if r["status"] == "active"]


def action_choices(t: Tracker | None):
    return [] if t is None else [a.id for a in t._engine().schema.actions]


def outputs(state):
    t = state["tracker"] if state else None
    acts = action_choices(t)
    return (state, status_html(t), log_md(state["log"] if state else []),
            gr.Dropdown(choices=requirement_choices(t), value=[]),
            gr.Dropdown(choices=record_choices(t), value=None),
            gr.Dropdown(choices=requirement_choices(t), value=None),
            gr.Dropdown(choices=acts, value=acts[0] if acts else None),
            explain_md(t, acts[0] if acts else None))


def _close(state):
    if state and state.get("tracker") is not None:
        state["tracker"].close()


def describe(u, withdrawn: bool = False) -> str:
    changed = "; ".join(f"**{a}**: {b.replace('_', ' ')} &rarr; {c.replace('_', ' ')}" for a, b, c in u.changed)
    cost = f"{u.pairs_read} pair{'s' if u.pairs_read != 1 else ''} read, {u.milliseconds:.0f} ms"
    outcome = "withdrawn" if withdrawn and u.outcome == "applied" else u.outcome
    return f"`{u.record_id}` {outcome} ({cost})" + (f": {changed}" if changed else ": no proposal changed")


def start_task(state, template_name, name, people):
    _close(state)
    tid = next(k for k, v in TEMPLATES.items() if v.name == template_name)
    params = {}
    for key, spec in TEMPLATES[tid].params.items():
        if spec["type"] == "text" and name.strip():
            params[key] = name.strip()
        elif spec["type"] == "list":
            params[key] = [p.strip() for p in people.split(",") if p.strip()]
    with LOCK:
        t = Tracker(render(tid, tid, params), READER)
    state = {"tracker": t, "log": [f"Started **{template_name}**. Add messages below."], "n": 0}
    return outputs(state)


def add_message(state, text, about):
    if not state or state.get("tracker") is None:
        raise gr.Error("Start a task first.")
    if not text.strip():
        raise gr.Error("Write a message first.")
    state["n"] += 1
    with LOCK:
        u = state["tracker"].add(f"msg-{state['n']}", text.strip(), about=about or None)
    state["log"].append(describe(u) + f" &mdash; _{html.escape(text.strip()[:90])}_")
    return outputs(state)


def withdraw(state, record):
    if not state or state.get("tracker") is None or not record:
        raise gr.Error("Choose a record to withdraw.")
    with LOCK:
        u = state["tracker"].withdraw(record)
    state["log"].append(describe(u, withdrawn=True))
    return outputs(state)


def confirm(state, requirement, who, holds):
    if not state or state.get("tracker") is None or not requirement:
        raise gr.Error("Choose a requirement to confirm.")
    with LOCK:
        u = state["tracker"].confirm(requirement, by=who.strip() or "A colleague", holds=holds)
    state["log"].append(describe(u) + (" &mdash; confirmed by a person (read exactly)" if holds else
                                       " &mdash; denied by a person (read exactly)"))
    return outputs(state)


def play_story(state):
    _close(state)
    with LOCK:
        t = Tracker(packet_schema(), READER)
    state = {"tracker": t, "log": ["Playing the client hand-off: twelve events, read as they arrive."], "n": 0}
    yield outputs(state)
    for title, ev in scenario():
        time.sleep(0.6)
        with LOCK:
            rep = t.runtime.deliver(t.task_id, ev)
        u = t._update(rep, ev.record_id)
        state["log"].append(f"**{title}** " + describe(u, withdrawn=ev.text is None))
        yield outputs(state)


CSS = """
.ease { font-size: 15px; }
.ease-h { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; opacity: .7; margin: 14px 0 6px; font-weight: 600; }
.ease-item { border-top: 1px solid var(--border-color-primary); padding: 8px 0; }
.ease-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.ease-id { font-weight: 600; font-family: var(--font-mono); }
.ease-muted { opacity: .7; font-size: 13px; }
.ease-badge { padding: 1px 9px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: .03em; }
.ease-badge.ready { background: rgba(31,122,77,.18); color: #1f9a5f; }
.ease-badge.blocked { background: rgba(200,60,40,.18); color: #d4513b; }
.ease-badge.open { background: rgba(200,150,0,.18); color: #b8860b; }
.ease-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--border-color-primary); margin: 6px 0 3px; }
.ease-bar span { display: block; height: 100%; }
.ease-bar .s { background: #2fa36b; } .ease-bar .v { background: #d4513b; } .ease-bar .u { background: #d1a21c; }
.ease-ask, .ease-note { border-left: 3px solid var(--color-accent); padding: 6px 10px; margin: 8px 0; background: var(--block-background-fill); }
.ease-note { font-size: 13px; }
.ease-empty { opacity: .7; padding: 24px 0; }
"""

INTRO = f"""
# EASE-Delta
**Keep a task's decisions current as its messages change.** You say what must be true before something can be done;
messages arrive, get corrected, get withdrawn. EASE-Delta reads each message against each requirement once, and after
every change tells you what is ready, what is blocked and why, and which single question is worth asking.
It proposes; it never acts. [Code and docs]({GITHUB}) &middot; [Measured results]({RESULTS}) &middot;
[Model](https://huggingface.co/jithinpothireddy21/ease-delta)

This app runs on a server, so what you type is processed there; it is not saved to any account, but please do not
paste anything private. Found a wrong reading? [Report it]({REPORT}): if you allow it on the form, it becomes a test case for the next model.
"""

with gr.Blocks(title="EASE-Delta: keep decisions current") as demo:
    gr.Markdown(INTRO)
    state = gr.State(None)
    with gr.Tab("Keep a task current"):
        with gr.Row():
            with gr.Column(scale=5):
                play = gr.Button("Play the hand-off story", variant="primary")
                gr.Markdown("Or start your own task:")
                with gr.Row():
                    template = gr.Dropdown([v.name for v in TEMPLATES.values()], value="Client onboarding",
                                           label="Workflow", scale=2)
                    name = gr.Textbox(label="Client, customer or release name", value="Acme", scale=2)
                people = gr.Textbox(label="People, for events and trips (comma-separated)", placeholder="Ann Lee, Bo Chen",
                                    max_lines=1)
                start = gr.Button("Start this task")
                gr.Markdown("**Add a message.** Say who is speaking: the reader sees only the text.")
                text = gr.Textbox(lines=3, label="Message", placeholder="Email from Acme: the brief is attached; files follow tomorrow.")
                about = gr.Dropdown([], multiselect=True, label="It concerns (optional: leave empty to read it against every requirement)")
                add = gr.Button("Add message", variant="primary")
                with gr.Accordion("Withdraw a message, or confirm a requirement yourself", open=False):
                    with gr.Row():
                        record = gr.Dropdown([], label="Message to withdraw", scale=3)
                        wd = gr.Button("Withdraw", scale=1)
                    with gr.Row():
                        req = gr.Dropdown([], label="Requirement", scale=3)
                        who = gr.Textbox(label="Who says so", value="A colleague", scale=2)
                    with gr.Row():
                        yes = gr.Button("It holds")
                        no = gr.Button("It does not hold")
                gr.Markdown("**What changed**")
                log = gr.Markdown(log_md([]))
            with gr.Column(scale=6):
                status = gr.HTML(status_html(None))
                why_action = gr.Dropdown([], label="Why? Choose an action")
                why = gr.Markdown()
                gr.Markdown(f"Did it read a message wrongly? [Report it]({REPORT}) with the requirement and the "
                            "message, after removing anything private.")
    with gr.Tab("Read one passage"):
        gr.Markdown("**Does this passage support the claim, refute it, or not settle it at all?** "
                    "The EASE-Delta reader next to a widely used public NLI model, on the same pair.")
        with gr.Row():
            claim = gr.Textbox(label="Claim", lines=2)
            passage = gr.Textbox(label="Passage", lines=4)
        read_btn = gr.Button("Read", variant="primary")
        with gr.Row():
            ours = gr.Label(label="EASE-Delta reader")
            theirs = gr.Label(label=f"{PUBLIC_NLI} (for comparison)")
        report = gr.Markdown()
        gr.Examples(READ_EXAMPLES, inputs=[claim, passage], outputs=[ours, theirs, report], fn=read_passage,
                    cache_examples=False, run_on_click=True)
        gr.Markdown("On passages taken from unrelated documents, the EASE-Delta reader gave a decisive answer 0.58% "
                    f"of the time and the public model 45%, in the same evaluation ([results]({RESULTS})). The public "
                    "model is trained on more NLI data and does better on adversarial NLI (ANLI).")
    with gr.Tab("How it works"):
        gr.Markdown(f"""
Each message is a **record** with a revision. A trained **reader** (ModernBERT-large, 395M parameters) reads every
(requirement, record) pair once and caches the reading. **Exact code** does the rest: which revision is current,
which record outranks which, how requirements combine (all of, any of, at least k of, not), and which question is
worth its cost. When a record changes, only the pairs that touch it are read again; when nothing changed (a
duplicate, an old copy), nothing is read at all.

| Measured (pre-registered) | |
|---|---|
| Correct ready / blocked / needs-info decisions on generated tasks | 91.6% (standard), 86.8% (larger tasks) |
| Cached state equal to a full rebuild | 0 of 195,434 values differed |
| Work per change compared with re-reading the task | about 2-7% of the tokens |
| Unrelated passages read as evidence | 0.6% (a public NLI model: 45%) |

What it does not do yet: it has not been measured with people, it reads English only, and messages on the same
subject that settle nothing are still misread about one time in five. Everything, including the hypotheses that
failed, is in the [results report]({RESULTS}).
""")

    outs = [state, status, log, about, record, req, why_action, why]
    play.click(play_story, [state], outs)
    start.click(start_task, [state, template, name, people], outs)
    add.click(add_message, [state, text, about], outs).then(lambda: "", None, text)
    wd.click(withdraw, [state, record], outs)
    yes.click(lambda s, r, w: confirm(s, r, w, True), [state, req, who], outs)
    no.click(lambda s, r, w: confirm(s, r, w, False), [state, req, who], outs)
    why_action.change(lambda s, a: explain_md(s["tracker"] if s else None, a), [state, why_action], why)
    read_btn.click(read_passage, [claim, passage], [ours, theirs, report])

if __name__ == "__main__":
    # EASE_SHARE=1 prints a temporary public link (for running the demo from a notebook, e.g. on Colab)
    demo.queue(default_concurrency_limit=4).launch(css=CSS, theme=gr.themes.Soft(),
                                                   share=os.environ.get("EASE_SHARE") == "1")
