"""Local HTTP API.

    ease serve --model runs/stage_a/final --aggregator runs/stage_b/refined_seed1 --data ~/.ease

Binds to 127.0.0.1 unless told otherwise. It reads evidence and reports assessments; it has no
endpoint that sends, pays, submits or executes anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ease import __version__
from ease.ledger import Event
from ease.runtime import Runtime
from ease.schema import TaskSchema

LABELS = {"supports": 0, "refutes": 1, "settles_nothing": 2}


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


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="EASE-Delta", version=__version__,
                  description="Keeps a task's decisions current as evidence changes. Proposes; never executes.")

    def guard(fn):
        try:
            return fn()
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\""))
        except FileExistsError as e:
            raise HTTPException(409, str(e))
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui():
        # no external scripts, fonts or styles: the page works offline and sends nothing elsewhere
        return HTMLResponse((Path(__file__).parent / "ui.html").read_text(encoding="utf-8"),
                            headers={"Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                                                "style-src 'self' 'unsafe-inline'; connect-src 'self'",
                                     "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})

    @app.get("/health")
    def health():
        return {"ok": True, "version": __version__}

    @app.get("/info")
    def info():
        return runtime.info()

    @app.get("/tasks")
    def tasks():
        return {"tasks": runtime.list_tasks()}

    @app.post("/tasks", status_code=201)
    def create_task(body: SchemaIn):
        guard(lambda: runtime.create_task(body.build()))
        return runtime.state(body.task_id)

    @app.get("/tasks/{task_id}")
    def state(task_id: str):
        return guard(lambda: runtime.state(task_id))

    @app.put("/tasks/{task_id}/schema")
    def update_schema(task_id: str, body: SchemaIn):
        if body.task_id != task_id:
            raise HTTPException(422, "task_id in the body does not match the path")
        return guard(lambda: runtime.update_schema(body.build()))

    @app.post("/tasks/{task_id}/events")
    def deliver(task_id: str, body: EventIn):
        ev = guard(lambda: Event(body.record_id, body.revision, body.text, source_id=body.source_id,
                                 authority=body.authority, valid_from=body.valid_from, valid_until=body.valid_until,
                                 span=body.span, meta={"about": body.about} if body.about is not None else {}))
        change = guard(lambda: runtime.deliver(task_id, ev))
        return {"change": change, "assessments": runtime.state(task_id)["assessments"]}

    @app.post("/tasks/{task_id}/documents")
    def document(task_id: str, body: DocumentIn):
        rep = guard(lambda: runtime.ingest_document(task_id, body.doc_id, body.revision, body.text, body.source_id,
                                                    body.authority, body.attribution, body.valid_from))
        return {"ingest": rep, "assessments": runtime.state(task_id)["assessments"]}

    @app.post("/tasks/{task_id}/clock")
    def clock(task_id: str, body: ClockIn):
        return guard(lambda: runtime.set_now(task_id, body.now))

    @app.get("/tasks/{task_id}/explain/{action_id}")
    def explain(task_id: str, action_id: str):
        return guard(lambda: runtime.explain(task_id, action_id))

    @app.get("/tasks/{task_id}/question")
    def question(task_id: str, max_set: int = 3):
        if not 1 <= max_set <= 4:
            raise HTTPException(422, "max_set must be between 1 and 4")
        return guard(lambda: runtime.question(task_id, max_set))

    @app.post("/tasks/{task_id}/certificate")
    def certificate(task_id: str, body: CertificateIn):
        return guard(lambda: runtime.certificate(task_id, body.action_id, body.free))

    @app.get("/tasks/{task_id}/verify")
    def verify(task_id: str):
        return guard(lambda: runtime.verify(task_id))

    @app.post("/tasks/{task_id}/readings")
    def reading(task_id: str, body: ReadingIn):
        if body.establishes not in LABELS:
            raise HTTPException(422, f"establishes must be one of {sorted(LABELS)}")
        return guard(lambda: runtime.correct_reading(task_id, body.record_id, body.predicate_id, LABELS[body.establishes]))

    @app.delete("/readings/{item_id}")
    def retract(item_id: int):
        return runtime.retract_correction(item_id)

    @app.post("/tasks/{task_id}/endorsements/{action_id}")
    def endorse(task_id: str, action_id: str):
        return guard(lambda: runtime.endorse(task_id, action_id))

    @app.post("/tasks/{task_id}/verdicts")
    def verdict(task_id: str, body: VerdictIn):
        return guard(lambda: runtime.verdict(task_id, body.action_id, body.was_wrong))

    @app.post("/evolve/consolidate")
    def consolidate_now():
        return runtime.consolidate_now()

    @app.post("/evolve/rollback")
    def rollback():
        return runtime.rollback()

    return app
