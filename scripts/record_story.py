"""Record the client hand-off story as the released system reads it, for the static web page.

    python scripts/record_story.py --model release --out site/story.json

Every state in the file is what the model produced on this machine, in exact mode. The page replays it;
it does not compute it. Nothing is edited by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from ease import EvidenceReader, Tracker
from ease.demo import packet_schema, scenario


def snapshot(t: Tracker) -> dict:
    st = t.status()
    return {
        "actions": [{"id": a.id, "description": a.description, "disposition": a.disposition,
                     "confidence": round(a.confidence, 4)} for a in st.actions.values()],
        "requirements": [{"id": r.id, "text": r.text, "status": r.status, "satisfied": round(r.satisfied, 4),
                          "violated": round(r.violated, 4), "open": round(r.open, 4), "rests_on": r.rests_on}
                         for r in st.requirements.values()],
        "question": st.question,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="release")
    ap.add_argument("--out", default="site/story.json")
    a = ap.parse_args()
    reader = EvidenceReader.from_pretrained(a.model, exact=True)
    t = Tracker(packet_schema(), reader)
    steps = [{"title": "The task is declared. Nothing has arrived yet.", "event": None, "update": None, "state": snapshot(t)}]
    for title, ev in scenario():
        t0 = time.perf_counter()
        rep = t.runtime.deliver(t.task_id, ev)
        u = t._update(rep, ev.record_id)
        steps.append({
            "title": title,
            "event": {"record": ev.record_id, "revision": ev.revision, "text": ev.text, "source": ev.source_id},
            "update": {"outcome": "withdrawn" if ev.text is None and u.outcome == "applied" else u.outcome,
                       "pairs_read": u.pairs_read, "milliseconds": round(1000 * (time.perf_counter() - t0), 1),
                       "changed": [list(c) for c in u.changed]},
            "state": snapshot(t),
        })
    out = {"model": a.model, "edge_version": reader.scorer.version, "exact": True,
           "ready_at": float(t.runtime.tracker.tau), "machine": platform.platform(),
           "recorded": time.strftime("%Y-%m-%d", time.gmtime()), "verified_against_full_rebuild": t.verify(),
           "steps": steps}
    t.close()
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{len(steps)} steps -> {a.out}; matches a full rebuild: {out['verified_against_full_rebuild']}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
