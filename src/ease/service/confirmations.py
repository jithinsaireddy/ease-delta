"""Confirmation requests: one requirement put to one person through a link.

The person who opens the link sees the requirement's text and nothing else about the task. Their
answer becomes a record in the ledger, attributed to them, compared only with that requirement,
and read exactly (it is entered as a verified reading, not interpreted by the model). A link is
single-use and expires.
"""

from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path
from typing import Optional

from ease.util import atomic_write_json, read_json

ANSWERS = ("yes", "no", "unsure")


class Confirmations:
    def __init__(self, path: str | Path, default_ttl_hours: float = 72.0):
        self.path = Path(path)
        self.ttl = default_ttl_hours * 3600
        self._lock = threading.RLock()
        self._items: dict[str, dict] = read_json(self.path) if self.path.exists() else {}

    def _save(self) -> None:
        atomic_write_json(self.path, self._items)

    def open_count(self) -> int:
        now = time.time()
        return sum(1 for c in self._items.values() if c["answer"] is None and c["expires"] > now)

    def create(self, task_id: str, predicate_id: str, predicate_text: str, to_name: str, authority: int = 3,
               ttl_hours: Optional[float] = None) -> dict:
        with self._lock:
            token = secrets.token_urlsafe(24)
            item = {"token": token, "task_id": task_id, "predicate_id": predicate_id, "predicate_text": predicate_text,
                    "to_name": to_name.strip()[:120] or "a reviewer", "authority": int(authority),
                    "created": time.time(), "expires": time.time() + (ttl_hours * 3600 if ttl_hours else self.ttl),
                    "answer": None, "note": "", "answered": None}
            self._items[token] = item
            self._save()
            return dict(item)

    def get(self, token: str) -> Optional[dict]:
        with self._lock:
            c = self._items.get(token)
            if c is None or c["expires"] < time.time():
                return None
            return dict(c)

    def answer(self, token: str, answer: str, note: str = "") -> dict:
        if answer not in ANSWERS:
            raise ValueError(f"answer must be one of {ANSWERS}")
        with self._lock:
            c = self._items.get(token)
            if c is None or c["expires"] < time.time():
                raise KeyError("this link is unknown or has expired")
            if c["answer"] is not None:
                raise ValueError("this link was already used")
            c["answer"], c["note"], c["answered"] = answer, note.strip()[:1000], time.time()
            self._save()
            return dict(c)

    def for_task(self, task_id: str) -> list[dict]:
        with self._lock:
            return [{k: v for k, v in c.items() if k != "token"} | {"token_prefix": c["token"][:6]}
                    for c in self._items.values() if c["task_id"] == task_id]
