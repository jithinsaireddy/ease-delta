"""HTTP API.

    ease serve --model release/edge --aggregator release/aggregator --data ~/.ease           # one person, no keys
    ease serve --model release/edge --aggregator release/aggregator --data /srv/ease --multi  # workspaces with keys

It reads evidence and reports assessments; it has no endpoint that sends, pays, submits or
executes anything. With `--multi`, every request carries `Authorization: Bearer <workspace key>`
and sees only its own workspace. Administration (creating workspaces, backups) is done on the
server with `ease workspace ...` and `ease backup`, or over HTTP with the admin key if one is set.
"""

from __future__ import annotations

import base64
import html
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ease import __version__
from ease.importers import parse_file
from ease.ledger import Event
from ease.runtime import Runtime
from ease.schema import TaskSchema
from ease.service.workspaces import AuthError, QuotaError, Service, Workspace
from ease.templates import TEMPLATES, render

LABELS = {"supports": 0, "refutes": 1, "settles_nothing": 2}
SECURITY_HEADERS = {"Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                               "style-src 'self' 'unsafe-inline'; connect-src 'self'",
                    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}


class PredicateIn(BaseModel):
    id: str
    text: str = Field(min_length=1, max_length=2000)
    prior: float = Field(0.5, ge=0.0, le=1.0)
    ask_cost: float = Field(1.0, ge=0.0)
    join: bool = False


class GateIn(BaseModel):
    id: str
    op: str
    children: list[str]
    k: int = 0


class ActionIn(BaseModel):
    id: str
    description: str = Field(max_length=2000)
    requires: str
    value_success: float = 1.0
    cost_failure: float = 1.0
    reliability: float = Field(1.0, ge=0.0, le=1.0)
    leak: float = Field(0.0, ge=0.0, le=1.0)
    needs_approval: bool = True


class SchemaIn(BaseModel):
    task_id: str
    predicates: list[PredicateIn]
    gates: list[GateIn] = []
    actions: list[ActionIn] = []

    def build(self) -> TaskSchema:
        return TaskSchema.from_dict({"task_id": self.task_id,
                                     "predicates": [p.model_dump() for p in self.predicates],
                                     "gates": [g.model_dump() for g in self.gates],
                                     "actions": [a.model_dump() for a in self.actions]})


class FromTemplateIn(BaseModel):
    template: str
    task_id: str
    params: dict = {}


class EventIn(BaseModel):
    record_id: str = Field(min_length=1, max_length=200)
    revision: int = Field(ge=0)
    text: Optional[str] = Field(None, description="omit or null to withdraw the record")
    source_id: str = ""
    authority: int = Field(0, ge=0, le=9)
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    span: Optional[str] = Field(None, max_length=500)
    about: Optional[list[str]] = Field(None, description="predicates this record is declared to concern")


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=150, pattern=r"^[^#]+$")
    revision: int = Field(ge=0)
    text: str = Field(min_length=1)
    source_id: str = ""
    authority: int = Field(0, ge=0, le=9)
    attribution: Optional[str] = Field(None, max_length=200,
                                       description='who is speaking, e.g. "Email from the client"; kept in every passage')
    valid_from: Optional[float] = None


class ImportIn(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    text: Optional[str] = Field(None, description="the file's content as text")
    content_base64: Optional[str] = Field(None, description="or its bytes, base64-encoded")
    source_id: str = ""
    authority: int = Field(0, ge=0, le=9)
    attribution: Optional[str] = Field(None, max_length=200)


class ReadingIn(BaseModel):
    record_id: str
    predicate_id: str
    establishes: str = Field(description="supports | refutes | settles_nothing")


class VerdictIn(BaseModel):
    action_id: str
    was_wrong: bool


class ClockIn(BaseModel):
    now: float


class CertificateIn(BaseModel):
    action_id: str
    free: list[str]


class ConfirmationIn(BaseModel):
    predicate_id: str
    to_name: str = Field(min_length=1, max_length=120)
    authority: int = Field(3, ge=0, le=9)
    ttl_hours: float = Field(72.0, gt=0, le=24 * 30)


class AnswerIn(BaseModel):
    answer: str = Field(description="yes | no | unsure")
    note: str = Field("", max_length=1000)


class WebhookIn(BaseModel):
    url: Optional[str] = Field(None, max_length=500)


class WorkspaceIn(BaseModel):
    id: str = Field(min_length=3, max_length=32, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field("", max_length=120)


class QuotaIn(BaseModel):
    changes: dict[str, int]


class BackupIn(BaseModel):
    out_dir: Optional[str] = None


def create_app(service: Service | Runtime) -> FastAPI:
    if isinstance(service, Runtime):  # one runtime, local mode: the shape tests and the demo use
        service = Service(service.dir, service.scorer, None, local=True, runtime=service)
    svc: Service = service
    app = FastAPI(title="EASE-Delta", version=__version__,
                  description="Keeps a task's decisions current as evidence changes. Proposes; never executes.")

    def guard(fn):
        try:
            return fn()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\""))
        except FileExistsError as e:
            raise HTTPException(409, str(e))
        except QuotaError as e:
            raise HTTPException(429, str(e))
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e))

    def bearer(request: Request) -> Optional[str]:
        h = request.headers.get("authorization", "")
        return h[7:].strip() if h.lower().startswith("bearer ") else None

    def workspace(request: Request) -> Workspace:
        try:
            ws = svc.authenticate(bearer(request))
        except AuthError as e:
            raise HTTPException(401, str(e), headers={"WWW-Authenticate": "Bearer"})
        if not svc.rate_limit(ws):
            raise HTTPException(429, "too many requests for this workspace; try again in a minute")
        return ws

    def admin(request: Request) -> None:
        if not svc.is_admin(bearer(request)):
            raise HTTPException(401, "admin key required (none is configured, or it is wrong)", headers={"WWW-Authenticate": "Bearer"})

    def rt(ws: Workspace) -> Runtime:
        return svc.runtime(ws)

    def changed(ws: Workspace, task_id: str, change: dict, what: str) -> None:
        if change.get("changed_actions"):
            svc.notify(ws, {"event": what, "workspace": ws.id, "task": task_id, "changed_actions": change["changed_actions"],
                            "at": time.time()})

    # ------------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui():
        # no external scripts, fonts or styles: the page works offline and sends nothing elsewhere
        return HTMLResponse((Path(__file__).parent / "ui.html").read_text(encoding="utf-8"), headers=SECURITY_HEADERS)

    @app.get("/health")
    def health():
        return {"ok": True, "version": __version__, "mode": "local" if svc.local else "workspaces"}

    # ------------------------------------------------------------------ workspace
    @app.get("/workspace")
    def workspace_info(ws: Workspace = Depends(workspace)):
        with svc.lock:
            return {"id": ws.id, "name": ws.name, "quota": ws.quota.__dict__, "usage": svc.usage(ws),
                    "webhook": bool(ws.webhook_url), "open_confirmations": svc.confirmations(ws).open_count()}

    @app.put("/workspace/webhook")
    def webhook(body: WebhookIn, ws: Workspace = Depends(workspace)):
        secret = guard(lambda: svc.set_webhook(ws.id, body.url))
        return {"url": body.url, "secret": secret, "note": "requests carry X-EASE-Signature: sha256=HMAC(secret, body)"}

    @app.get("/info")
    def info(ws: Workspace = Depends(workspace)):
        with svc.lock:
            return rt(ws).info()

    @app.get("/templates")
    def templates():
        return {"templates": [t.describe() for t in TEMPLATES.values()]}

    # ------------------------------------------------------------------ tasks
    @app.get("/tasks")
    def tasks(ws: Workspace = Depends(workspace)):
        with svc.lock:
            return {"tasks": rt(ws).list_tasks()}

    @app.post("/tasks", status_code=201)
    def create_task(body: SchemaIn, ws: Workspace = Depends(workspace)):
        with svc.lock:
            guard(lambda: rt(ws).create_task(body.build()))
            return rt(ws).state(body.task_id)

    @app.post("/tasks/from-template", status_code=201)
    def create_from_template(body: FromTemplateIn, ws: Workspace = Depends(workspace)):
        with svc.lock:
            d = guard(lambda: render(body.template, body.task_id, body.params))
            guard(lambda: rt(ws).create_task(TaskSchema.from_dict(d)))
            return {"definition": d, "state": rt(ws).state(body.task_id)}

    @app.get("/tasks/{task_id}")
    def state(task_id: str, ws: Workspace = Depends(workspace)):
        with svc.lock:
            return guard(lambda: rt(ws).state(task_id))

    @app.put("/tasks/{task_id}/schema")
    def update_schema(task_id: str, body: SchemaIn, ws: Workspace = Depends(workspace)):
        if body.task_id != task_id:
            raise HTTPException(422, "task_id in the body does not match the path")
        with svc.lock:
            return guard(lambda: rt(ws).update_schema(body.build()))

    @app.post("/tasks/{task_id}/events")
    def deliver(task_id: str, body: EventIn, ws: Workspace = Depends(workspace)):
        ev = guard(lambda: Event(body.record_id, body.revision, body.text, source_id=body.source_id,
                                 authority=body.authority, valid_from=body.valid_from, valid_until=body.valid_until,
                                 span=body.span, meta={"about": body.about} if body.about is not None else {}))
        with svc.lock:
            guard(lambda: svc.charge(ws, events=1))
            change = guard(lambda: rt(ws).deliver(task_id, ev))
            changed(ws, task_id, change, "event")
            return {"change": change, "assessments": rt(ws).state(task_id)["assessments"]}

    @app.post("/tasks/{task_id}/documents")
    def document(task_id: str, body: DocumentIn, ws: Workspace = Depends(workspace)):
        with svc.lock:
            guard(lambda: svc.charge(ws, events=1, document_chars=len(body.text)))
            rep = guard(lambda: rt(ws).ingest_document(task_id, body.doc_id, body.revision, body.text, body.source_id,
                                                       body.authority, body.attribution, body.valid_from))
            changed(ws, task_id, rep, "document")
            return {"ingest": rep, "assessments": rt(ws).state(task_id)["assessments"]}

    @app.post("/tasks/{task_id}/import")
    def import_file(task_id: str, body: ImportIn, ws: Workspace = Depends(workspace)):
        if (body.text is None) == (body.content_base64 is None):
            raise HTTPException(422, "give exactly one of text or content_base64")
        try:
            raw = body.text.encode("utf-8") if body.text is not None else base64.b64decode(body.content_base64, validate=True)
        except (ValueError, UnicodeEncodeError) as e:
            raise HTTPException(422, f"could not decode the content: {e}")
        if len(raw) > 20_000_000:
            raise HTTPException(422, "file larger than 20 MB")
        messages = guard(lambda: parse_file(raw, body.filename, body.attribution))
        with svc.lock:
            guard(lambda: svc.charge(ws, events=len(messages), document_chars=sum(len(m.text) for m in messages)))
            out = []

            def ingest_one(m):
                return rt(ws).ingest_document(task_id, m.doc_id, 1, m.text, body.source_id or m.sender[:80],
                                              body.authority, m.attribution, m.valid_from)

            for m in messages:
                rep = guard(lambda m=m: ingest_one(m))
                changed(ws, task_id, rep, "import")
                out.append({"doc_id": m.doc_id, "attribution": m.attribution, "passages": rep["passages"],
                            "applied": rep["applied"], "changed_actions": rep["changed_actions"]})
            return {"messages": out, "assessments": rt(ws).state(task_id)["assessments"] if messages else {}}

    @app.post("/tasks/{task_id}/clock")
    def clock(task_id: str, body: ClockIn, ws: Workspace = Depends(workspace)):
        with svc.lock:
            return guard(lambda: rt(ws).set_now(task_id, body.now))

    @app.get("/tasks/{task_id}/explain/{action_id}")
    def explain(task_id: str, action_id: str, ws: Workspace = Depends(workspace)):
        with svc.lock:
            return guard(lambda: rt(ws).explain(task_id, action_id))

    @app.get("/tasks/{task_id}/question")
    def question(task_id: str, max_set: int = 3, ws: Workspace = Depends(workspace)):
        if not 1 <= max_set <= 4:
            raise HTTPException(422, "max_set must be between 1 and 4")
        with svc.lock:
            return guard(lambda: rt(ws).question(task_id, max_set))

    @app.post("/tasks/{task_id}/certificate")
    def certificate(task_id: str, body: CertificateIn, ws: Workspace = Depends(workspace)):
        with svc.lock:
            return guard(lambda: rt(ws).certificate(task_id, body.action_id, body.free))

    @app.get("/tasks/{task_id}/verify")
    def verify(task_id: str, ws: Workspace = Depends(workspace)):
        with svc.lock:
            return guard(lambda: rt(ws).verify(task_id))

    @app.post("/tasks/{task_id}/readings")
    def reading(task_id: str, body: ReadingIn, ws: Workspace = Depends(workspace)):
        if body.establishes not in LABELS:
            raise HTTPException(422, f"establishes must be one of {sorted(LABELS)}")
        with svc.lock:
            return guard(lambda: rt(ws).correct_reading(task_id, body.record_id, body.predicate_id, LABELS[body.establishes]))

    @app.delete("/readings/{item_id}")
    def retract(item_id: int, ws: Workspace = Depends(workspace)):
        with svc.lock:
            return rt(ws).retract_correction(item_id)

    @app.post("/tasks/{task_id}/endorsements/{action_id}")
    def endorse(task_id: str, action_id: str, ws: Workspace = Depends(workspace)):
        with svc.lock:
            return guard(lambda: rt(ws).endorse(task_id, action_id))

    @app.post("/tasks/{task_id}/verdicts")
    def verdict(task_id: str, body: VerdictIn, ws: Workspace = Depends(workspace)):
        with svc.lock:
            return guard(lambda: rt(ws).verdict(task_id, body.action_id, body.was_wrong))

    @app.post("/evolve/consolidate")
    def consolidate_now(ws: Workspace = Depends(workspace)):
        with svc.lock:
            return rt(ws).consolidate_now()

    @app.post("/evolve/rollback")
    def rollback(ws: Workspace = Depends(workspace)):
        with svc.lock:
            return rt(ws).rollback()

    # ------------------------------------------------------------------ confirmations
    @app.post("/tasks/{task_id}/confirmations", status_code=201)
    def create_confirmation(task_id: str, body: ConfirmationIn, request: Request, ws: Workspace = Depends(workspace)):
        with svc.lock:
            eng = guard(lambda: rt(ws).engine(task_id))
            pred = guard(lambda: eng.schema.predicate(body.predicate_id))
            store = svc.confirmations(ws)
            if store.open_count() >= ws.quota.max_open_confirmations:
                raise HTTPException(429, "too many open confirmation links; wait for answers or let some expire")
            c = store.create(task_id, body.predicate_id, pred.text, body.to_name, body.authority, body.ttl_hours)
            rt(ws).audit.log(kind="confirmation_requested", task=task_id, predicate=body.predicate_id,
                             to=c["to_name"], token_prefix=c["token"][:6])
            path = f"/confirm/{ws.id}/{c['token']}"
            return {"token": c["token"], "path": path, "url": str(request.base_url).rstrip("/") + path,
                    "expires": c["expires"], "to_name": c["to_name"], "asks": pred.text}

    @app.get("/tasks/{task_id}/confirmations")
    def list_confirmations(task_id: str, ws: Workspace = Depends(workspace)):
        with svc.lock:
            guard(lambda: rt(ws).engine(task_id))
            return {"confirmations": svc.confirmations(ws).for_task(task_id)}

    def _confirmation(ws_id: str, token: str):
        try:
            ws = svc.get(ws_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "unknown link")
        c = svc.confirmations(ws).get(token)
        if c is None:
            raise HTTPException(404, "this link is unknown or has expired")
        return ws, c

    @app.get("/confirm/{ws_id}/{token}", response_class=HTMLResponse, include_in_schema=False)
    def confirm_page(ws_id: str, token: str):
        with svc.lock:
            _, c = _confirmation(ws_id, token)
        page = (Path(__file__).parent / "confirm.html").read_text(encoding="utf-8")
        page = (page.replace("{{NAME}}", html.escape(c["to_name"])).replace("{{TEXT}}", html.escape(c["predicate_text"]))
                    .replace("{{DONE}}", "true" if c["answer"] else "false"))
        return HTMLResponse(page, headers=SECURITY_HEADERS)

    @app.post("/confirm/{ws_id}/{token}")
    def confirm_answer(ws_id: str, token: str, body: AnswerIn):
        with svc.lock:
            ws, c = _confirmation(ws_id, token)
            c = guard(lambda: svc.confirmations(ws).answer(token, body.answer, body.note))
            r = rt(ws)
            applied = None
            if body.answer in ("yes", "no"):
                try:
                    eng = r.engine(c["task_id"])
                    pred = eng.schema.predicate(c["predicate_id"])
                except KeyError:
                    pred = None
                if pred is not None and pred.text == c["predicate_text"]:
                    said = "confirmed that this holds" if body.answer == "yes" else "said that this does not hold"
                    text = f"{c['to_name']} {said}: {pred.text}" + (f" Note: {c['note']}" if c["note"] else "")
                    rid = f"confirmation-{token[:8]}"
                    ev = Event(rid, 1, text, source_id=f"confirmation:{c['to_name']}", authority=c["authority"],
                               valid_from=time.time(), meta={"about": [c["predicate_id"]]})
                    change = guard(lambda: r.deliver(c["task_id"], ev))
                    fix = r.correct_reading(c["task_id"], rid, c["predicate_id"], 0 if body.answer == "yes" else 1,
                                            source=f"confirmation:{c['to_name']}")
                    applied = {"record_id": rid, "changed_actions": change["changed_actions"] or fix["changed_actions"]}
                    changed(ws, c["task_id"], applied, "confirmation")
            r.audit.log(kind="confirmation_answered", task=c["task_id"], predicate=c["predicate_id"],
                        answer=body.answer, applied=applied is not None)
            return {"thanks": True, "answer": body.answer, "applied": applied}

    # ------------------------------------------------------------------ administration
    @app.get("/admin/workspaces", dependencies=[Depends(admin)])
    def admin_list():
        with svc.lock:
            return {"workspaces": svc.list_workspaces()}

    @app.post("/admin/workspaces", status_code=201, dependencies=[Depends(admin)])
    def admin_create(body: WorkspaceIn):
        with svc.lock:
            ws, key = guard(lambda: svc.create_workspace(body.id, body.name))
            return {"id": ws.id, "name": ws.name, "key": key, "note": "the key is shown once; only its hash is stored"}

    @app.post("/admin/workspaces/{ws_id}/rotate-key", dependencies=[Depends(admin)])
    def admin_rotate(ws_id: str):
        with svc.lock:
            return {"id": ws_id, "key": guard(lambda: svc.rotate_key(ws_id))}

    @app.put("/admin/workspaces/{ws_id}/quota", dependencies=[Depends(admin)])
    def admin_quota(ws_id: str, body: QuotaIn):
        with svc.lock:
            return guard(lambda: svc.set_quota(ws_id, **body.changes)).__dict__

    @app.post("/admin/backup", dependencies=[Depends(admin)])
    def admin_backup(body: BackupIn):
        with svc.lock:
            return svc.backup(body.out_dir or (svc.dir / "backups"))

    return app
