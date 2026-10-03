"""Many private workspaces behind one model.

    data_dir/
      service.json                     when the service was created; the admin key's hash, if one is set
      edge_cache.sqlite                readings shared by every workspace. They are model outputs keyed by
                                       the weights version and hashes of the texts; no document text is stored
      workspaces/<id>/workspace.json   name, key hash, quota, webhook
      workspaces/<id>/usage.json       counters for the current UTC day
      workspaces/<id>/confirmations.json
      workspaces/<id>/...              the workspace's own Runtime: tasks, corrections, versions, threshold, audit
      backups/<timestamp>/             copies made by `backup`

A request identifies its workspace by a key. Keys are stored as hashes; a lost key is replaced,
never recovered. Each workspace has its own Runtime, so one workspace's corrections, endorsement
threshold and adopted versions cannot affect another's. Only the model and its reading cache are
shared.

In local mode (`ease serve` without `--multi`) there is one workspace, "local", and no key. Its
files live directly under data_dir, as they did before workspaces existed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import shutil
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ease.runtime import Limits, PersistentEdgeCache, Runtime
from ease.util import atomic_write_json, read_json

WS_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")
LOCAL = "local"


class AuthError(Exception):
    pass


class QuotaError(Exception):
    pass


@dataclass
class Quota:
    """Ceilings per workspace. Each is a plain number a person can read in workspace.json."""

    max_tasks: int = 50
    max_predicates: int = 60
    max_records_per_task: int = 2_000
    max_text_chars: int = 20_000
    max_events_per_day: int = 5_000
    max_document_chars_per_day: int = 2_000_000
    max_open_confirmations: int = 200
    max_requests_per_minute: int = 300

    def limits(self) -> Limits:
        return Limits(max_text_chars=self.max_text_chars, max_records_per_task=self.max_records_per_task,
                      max_predicates=self.max_predicates, max_tasks=self.max_tasks)


@dataclass
class Workspace:
    id: str
    name: str
    key_sha256: str
    created: float
    quota: Quota = field(default_factory=Quota)
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Workspace":
        q = d.get("quota") or {}
        return cls(d["id"], d["name"], d["key_sha256"], float(d["created"]),
                   Quota(**{k: v for k, v in q.items() if k in Quota.__dataclass_fields__}),
                   d.get("webhook_url"), d.get("webhook_secret"))


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class _Bucket:
    """Token bucket: `rate` requests per minute, burst of the same size."""

    def __init__(self, rate: int):
        self.rate = max(1, rate)
        self.tokens = float(self.rate)
        self.last = time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate / 60.0)
        self.last = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class Service:
    def __init__(self, data_dir: str | Path, scorer, base_aggregator=None, settings: Optional[dict] = None,
                 regression_suite: Optional[str | Path] = None, local: bool = False, admin_key: Optional[str] = None,
                 runtime: Optional[Runtime] = None):
        self.dir = Path(data_dir)
        (self.dir / ("workspaces" if not local else "")).mkdir(parents=True, exist_ok=True)
        self.scorer = scorer
        self.base_aggregator = base_aggregator
        self.settings = settings or {}
        self.regression_suite = regression_suite
        self.local = local
        # one lock for everything that touches the model: the encoder is not safe to call from two
        # threads at once, and requests are short
        self.lock = threading.RLock()
        self._runtimes: dict[str, Runtime] = {}
        self._buckets: dict[str, _Bucket] = {}
        # the shared reading cache; an injected runtime keeps whatever cache it was built with
        if runtime is None and hasattr(scorer, "cache") and not isinstance(scorer.cache, PersistentEdgeCache):
            scorer.cache = PersistentEdgeCache(str(self.dir / "edge_cache.sqlite"))
        meta_path = self.dir / "service.json"
        meta = read_json(meta_path) if meta_path.exists() else {"created": time.time()}
        if admin_key:
            meta["admin_key_sha256"] = _sha(admin_key)
        self.admin_key_sha256 = meta.get("admin_key_sha256")
        atomic_write_json(meta_path, meta)
        if local and not (self._ws_dir(LOCAL) / "workspace.json").exists():
            self._write(Workspace(LOCAL, "local", "", time.time()))
        if runtime is not None:  # an existing local runtime (tests, the demo)
            self._runtimes[LOCAL] = runtime

    # ------------------------------------------------------------------ workspaces
    def _ws_dir(self, ws_id: str) -> Path:
        if not WS_ID.match(ws_id):
            raise ValueError("workspace id must be 3-32 characters: lower-case letters, digits, '-'")
        if self.local and ws_id == LOCAL:
            return self.dir
        return self.dir / "workspaces" / ws_id

    def _write(self, ws: Workspace) -> None:
        d = self._ws_dir(ws.id)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_json(d / "workspace.json", ws.to_dict())

    def get(self, ws_id: str) -> Workspace:
        p = self._ws_dir(ws_id) / "workspace.json"
        if not p.exists():
            raise KeyError(f"no workspace {ws_id!r}")
        return Workspace.from_dict(read_json(p))

    def list_workspaces(self) -> list[dict]:
        out = []
        root = self.dir if self.local else self.dir / "workspaces"
        for p in sorted([self.dir] if self.local else root.iterdir()):
            if (p / "workspace.json").exists():
                ws = Workspace.from_dict(read_json(p / "workspace.json"))
                out.append({"id": ws.id, "name": ws.name, "created": ws.created, "tasks": len(self.runtime(ws).list_tasks())})
        return out

    @staticmethod
    def _new_key(ws_id: str) -> str:
        return f"ease_{ws_id}_{secrets.token_hex(24)}"

    def create_workspace(self, ws_id: str, name: str, quota: Optional[Quota] = None) -> tuple[Workspace, str]:
        """Returns the workspace and its key. The key is shown once; only its hash is kept."""
        with self.lock:
            if self.local:
                raise ValueError("this service runs in local mode with a single workspace")
            d = self._ws_dir(ws_id)
            if (d / "workspace.json").exists():
                raise FileExistsError(f"workspace {ws_id!r} already exists")
            key = self._new_key(ws_id)
            ws = Workspace(ws_id, name.strip() or ws_id, _sha(key), time.time(), quota or Quota())
            self._write(ws)
            return ws, key

    def rotate_key(self, ws_id: str) -> str:
        with self.lock:
            ws = self.get(ws_id)
            key = self._new_key(ws_id)
            ws.key_sha256 = _sha(key)
            self._write(ws)
            return key

    def set_webhook(self, ws_id: str, url: Optional[str]) -> Optional[str]:
        """Sets or clears the webhook; returns the signing secret for a new URL."""
        with self.lock:
            ws = self.get(ws_id)
            if url:
                if not re.match(r"^https?://", url) or re.match(r"^https?://(localhost|127\.|0\.0\.0\.0|10\.|192\.168\.|\[::1\])", url):
                    raise ValueError("webhook must be a public http(s) address")
                ws.webhook_url, ws.webhook_secret = url, secrets.token_hex(16)
            else:
                ws.webhook_url = ws.webhook_secret = None
            self._write(ws)
            return ws.webhook_secret

    def set_quota(self, ws_id: str, **changes) -> Quota:
        with self.lock:
            ws = self.get(ws_id)
            for k, v in changes.items():
                if k not in Quota.__dataclass_fields__:
                    raise ValueError(f"unknown quota field {k!r}")
                setattr(ws.quota, k, int(v))
            self._write(ws)
            rt = self._runtimes.get(ws.id)
            if rt is not None:
                rt.limits = ws.quota.limits()
            return ws.quota

    # ------------------------------------------------------------------ authentication
    def authenticate(self, key: Optional[str]) -> Workspace:
        if self.local:
            return self.get(LOCAL)
        if not key or not key.startswith("ease_"):
            raise AuthError("a workspace key is required: Authorization: Bearer ease_...")
        parts = key.split("_", 2)
        if len(parts) != 3:
            raise AuthError("malformed key")
        try:
            ws = self.get(parts[1])
        except (KeyError, ValueError):
            raise AuthError("unknown workspace") from None
        if not ws.key_sha256 or not hmac.compare_digest(ws.key_sha256, _sha(key)):
            raise AuthError("wrong key")
        return ws

    def is_admin(self, key: Optional[str]) -> bool:
        return bool(self.admin_key_sha256 and key and hmac.compare_digest(self.admin_key_sha256, _sha(key)))

    def rate_limit(self, ws: Workspace) -> bool:
        with self.lock:
            b = self._buckets.get(ws.id)
            if b is None or b.rate != ws.quota.max_requests_per_minute:
                b = self._buckets[ws.id] = _Bucket(ws.quota.max_requests_per_minute)
            return b.take()

    # ------------------------------------------------------------------ runtimes
    def runtime(self, ws: Workspace) -> Runtime:
        with self.lock:
            rt = self._runtimes.get(ws.id)
            if rt is None:
                suite = Path(self.regression_suite) if self.regression_suite else None
                rt = Runtime(self._ws_dir(ws.id), self.scorer, self.base_aggregator, limits=ws.quota.limits(),
                             alpha=self.settings.get("alpha", 0.05), eta=self.settings.get("eta", 0.05),
                             tau=self.settings.get("tau_initial", 0.9), regression_suite=suite)
                self._runtimes[ws.id] = rt
            return rt

    def confirmations(self, ws: Workspace):
        from ease.service.confirmations import Confirmations

        with self.lock:
            c = getattr(self, "_confirmations", None)
            if c is None:
                c = self._confirmations = {}
            if ws.id not in c:
                c[ws.id] = Confirmations(self._ws_dir(ws.id) / "confirmations.json")
            return c[ws.id]

    # ------------------------------------------------------------------ usage
    def _usage_path(self, ws: Workspace) -> Path:
        return self._ws_dir(ws.id) / "usage.json"

    def usage(self, ws: Workspace) -> dict:
        p = self._usage_path(ws)
        u = read_json(p) if p.exists() else {}
        if u.get("day") != _today():
            u = {"day": _today(), "events": 0, "document_chars": 0}
        return u

    def charge(self, ws: Workspace, events: int = 0, document_chars: int = 0) -> dict:
        """Counts usage against the day's quota before the work is done; raises if it would exceed it."""
        with self.lock:
            u = self.usage(ws)
            if u["events"] + events > ws.quota.max_events_per_day:
                raise QuotaError(f"daily limit of {ws.quota.max_events_per_day} events reached")
            if u["document_chars"] + document_chars > ws.quota.max_document_chars_per_day:
                raise QuotaError(f"daily limit of {ws.quota.max_document_chars_per_day} imported characters reached")
            u["events"] += events
            u["document_chars"] += document_chars
            atomic_write_json(self._usage_path(ws), u)
            return u

    # ------------------------------------------------------------------ webhooks
    def notify(self, ws: Workspace, payload: dict) -> None:
        """Posts a signed JSON message to the workspace's webhook, if one is set. Never blocks a request."""
        if not ws.webhook_url:
            return
        body = json.dumps(payload, default=str).encode()
        sig = "sha256=" + hmac.new((ws.webhook_secret or "").encode(), body, hashlib.sha256).hexdigest()
        url = ws.webhook_url

        def send():
            import urllib.request

            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, data=body, method="POST",
                                                 headers={"content-type": "application/json", "X-EASE-Signature": sig})
                    with urllib.request.urlopen(req, timeout=10):
                        return
                except Exception:
                    time.sleep(2 ** attempt)
            self.runtime(ws).audit.log(kind="webhook_failed", url=url)

        threading.Thread(target=send, daemon=True).start()

    # ------------------------------------------------------------------ backups
    def backup(self, out_dir: str | Path) -> dict:
        """Consistent copies of every database and setting. SQLite's online backup is used, so a
        copy taken while the service runs is still a valid database."""
        out = Path(out_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        files = 0
        with self.lock:
            for p in sorted(self.dir.rglob("*")):
                if not p.is_file() or "backups" in p.parts[len(self.dir.parts):] or p.suffix in (".sqlite-wal", ".sqlite-shm"):
                    continue
                rel = p.relative_to(self.dir)
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if p.suffix == ".sqlite":
                    src = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
                    try:
                        dest = sqlite3.connect(str(dst))
                        with dest:
                            src.backup(dest)
                        dest.close()
                    finally:
                        src.close()
                else:
                    shutil.copy2(p, dst)
                files += 1
        return {"directory": str(out), "files": files}

    def close(self) -> None:
        with self.lock:
            for rt in self._runtimes.values():
                rt.close()
            self._runtimes.clear()
