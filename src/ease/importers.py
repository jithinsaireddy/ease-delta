"""Messages and documents from files, as attributed passages.

Email files (.eml, .mbox) become one document per message, attributed in the text the reader
sees: "Email from Dana Ortiz on 2026-03-02, subject: Design v5". Quoted earlier messages are
dropped, because they would be read again as if they were new. Plain text and Markdown become
one document each.

Nothing here connects to a mail server. Files are supplied by the person; this only parses them.
"""

from __future__ import annotations

import email
import email.policy
import email.utils
import hashlib
import mailbox
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_QUOTE_START = re.compile(r"^(On .{5,200} wrote:|-{2,}\s*Original Message\s*-{2,}|From: .+)$", re.M)
_SIGNATURE = re.compile(r"^-- \s*$", re.M)


@dataclass
class ImportedMessage:
    doc_id: str
    attribution: str
    text: str
    valid_from: Optional[float]
    subject: str = ""
    sender: str = ""


def _clean_body(body: str) -> str:
    body = body.replace("\r\n", "\n")
    m = _QUOTE_START.search(body)
    if m and m.start() > 0:
        body = body[: m.start()]
    m = _SIGNATURE.search(body)
    if m:
        body = body[: m.start()]
    lines = [ln for ln in body.split("\n") if not ln.lstrip().startswith(">")]
    return "\n".join(lines).strip()


def _doc_id(msg, fallback: bytes) -> str:
    mid = (msg.get("Message-ID") or "").strip().strip("<>")
    mid = re.sub(r"[^A-Za-z0-9._@-]", "", mid)[:80]
    if not mid:
        mid = "msg-" + hashlib.sha256(fallback).hexdigest()[:16]
    return mid.replace("#", "")


def message_to_import(msg, raw: bytes) -> Optional[ImportedMessage]:
    """The text/plain content of one parsed message, attributed. None if there is no usable text."""
    body = None
    if msg.is_multipart():
        part = msg.get_body(preferencelist=("plain",))
        if part is not None:
            body = part.get_content()
    else:
        try:
            body = msg.get_content() if msg.get_content_type() == "text/plain" else None
        except (KeyError, LookupError):
            body = None
    if not body or not isinstance(body, str):
        return None
    text = _clean_body(body)
    if not text:
        return None
    name, addr = email.utils.parseaddr(msg.get("From", ""))
    sender = name or addr or "unknown sender"
    subject = " ".join((msg.get("Subject") or "").split())
    valid_from = None
    date_text = ""
    if msg.get("Date"):
        try:
            dt = email.utils.parsedate_to_datetime(msg["Date"])
            valid_from = dt.timestamp()
            date_text = dt.strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            pass
    attribution = f"Email from {sender}" + (f" on {date_text}" if date_text else "") + (f", subject: {subject}" if subject else "")
    return ImportedMessage(_doc_id(msg, raw), attribution, text, valid_from, subject, sender)


def parse_eml(raw: bytes) -> list[ImportedMessage]:
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    m = message_to_import(msg, raw)
    return [m] if m else []


def parse_mbox(raw: bytes) -> list[ImportedMessage]:
    out = []
    with tempfile.NamedTemporaryFile(suffix=".mbox", delete=True) as f:
        f.write(raw)
        f.flush()
        box = mailbox.mbox(f.name, factory=lambda fp: email.message_from_binary_file(fp, policy=email.policy.default))
        try:
            for key in box.keys():
                msg = box[key]
                m = message_to_import(msg, box.get_bytes(key))
                if m:
                    out.append(m)
        finally:
            box.close()
    return out


def parse_text(raw: bytes, filename: str, attribution: Optional[str] = None) -> list[ImportedMessage]:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    stem = re.sub(r"[^A-Za-z0-9._-]", "-", Path(filename).stem)[:80] or "document"
    return [ImportedMessage(stem, attribution or f"Document {Path(filename).name}", text, None, Path(filename).name, "")]


def parse_file(raw: bytes, filename: str, attribution: Optional[str] = None) -> list[ImportedMessage]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".eml":
        return parse_eml(raw)
    if suffix == ".mbox":
        return parse_mbox(raw)
    if suffix in (".txt", ".md", ".markdown", ".text", ""):
        return parse_text(raw, filename, attribution)
    raise ValueError(f"unsupported file type {suffix!r}: use .eml, .mbox, .txt or .md")
