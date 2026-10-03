"""Turn a document into ledger records.

The reader judges one short passage at a time and sees only its text. So a document is split into
passages, and each passage carries its attribution in its own words ("Email from the client: ...").

Record identity is `<doc_id>#<index>`. When a document is revised and a paragraph is inserted near
the top, later passages shift to new indices and their records change. That costs ledger writes
but not model time: readings are cached by text, so a passage that merely moved is looked up, not
re-read. Passages that no longer exist are withdrawn at the new revision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ease.ledger import Event, Status

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_BLANK = re.compile(r"\n\s*\n")


def split_passages(text: str, max_chars: int = 600, min_chars: int = 40) -> list[str]:
    """Paragraphs first; a paragraph that is too long is cut at sentence boundaries. Very short
    paragraphs (a greeting, a signature line) are attached to their neighbour so that no passage
    is a fragment. Deterministic."""
    paras = [" ".join(p.split()) for p in _BLANK.split(text.replace("\r\n", "\n")) if p.strip()]
    pieces: list[str] = []
    for p in paras:
        if len(p) <= max_chars:
            pieces.append(p)
            continue
        cur = ""
        for sent in _SENTENCE.split(p):
            while len(sent) > max_chars:  # a single sentence longer than a passage: cut at a space
                cut = sent.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                if cur:
                    pieces.append(cur)
                    cur = ""
                pieces.append(sent[:cut])
                sent = sent[cut:].strip()
            if cur and len(cur) + 1 + len(sent) > max_chars:
                pieces.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            pieces.append(cur)
    merged: list[str] = []
    for piece in pieces:
        if merged and (len(piece) < min_chars or len(merged[-1]) < min_chars) and len(merged[-1]) + 1 + len(piece) <= max_chars:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


@dataclass
class IngestReport:
    doc_id: str
    revision: int
    passages: int
    applied: int
    unchanged: int
    withdrawn: int
    edges_scored: int
    changed_actions: list

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def ingest(deliver, existing_records, doc_id: str, revision: int, text: str, source_id: str = "",
           authority: int = 0, attribution: Optional[str] = None, valid_from: Optional[float] = None,
           max_chars: int = 600) -> IngestReport:
    """`deliver(event) -> dict` is Runtime.deliver bound to a task, or anything with the same shape.
    `existing_records` maps record ids to RecordState (Ledger.records())."""
    if "#" in doc_id:
        raise ValueError("doc_id may not contain '#'")
    prefix = f"{attribution.strip().rstrip(':')}: " if attribution and attribution.strip() else ""
    passages = split_passages(text, max_chars=max_chars)
    applied = unchanged = withdrawn = scored = 0
    changes: list = []
    for i, passage in enumerate(passages):
        rep = deliver(Event(f"{doc_id}#{i}", revision, prefix + passage, source_id=source_id, authority=authority,
                            valid_from=valid_from, span=f"{doc_id}, passage {i + 1} of {len(passages)}"))
        if rep["ledger"]["changed"]:
            applied += 1
        else:
            unchanged += 1
        scored += rep["edges_scored"]
        changes.extend(rep["changed_actions"])
    for rid, st in sorted(existing_records.items()):
        if not rid.startswith(f"{doc_id}#"):
            continue
        try:
            idx = int(rid.rsplit("#", 1)[1])
        except ValueError:
            continue
        if idx >= len(passages) and st.revision < revision and st.stored_status is not Status.WITHDRAWN:
            rep = deliver(Event(rid, revision, None, source_id=source_id, authority=authority))
            withdrawn += 1
            changes.extend(rep["changed_actions"])
    return IngestReport(doc_id, revision, len(passages), applied, unchanged, withdrawn, scored, changes)
