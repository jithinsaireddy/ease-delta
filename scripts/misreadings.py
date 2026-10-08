"""Turn misreading reports into a scored evaluation set.

    python scripts/misreadings.py sync              # issues labelled "misreading" -> evals/misreadings.jsonl
    python scripts/misreadings.py sync --dry-run    # say what would change; write nothing
    python scripts/misreadings.py score             # both released readers on that file
    python scripts/misreadings.py score --save      # ... and write runs/results/misreadings.json

A report is an issue opened with the "Misreading" form (.github/ISSUE_TEMPLATE/misreading.yml). Only reports
whose author answered "Yes" to the form's consent question are written to the public file, under CC BY 4.0,
with the author and the issue as attribution. A report without consent never enters the repository: it is
counted, and nothing from it is stored. A maintainer keeps a report out by labelling the issue `invalid` (the
answer it expects is wrong) or `duplicate`.

The file is rebuilt from the issues as they stand on every sync, so re-running never duplicates a row, an
edited report replaces its old row, and a withdrawn consent removes it. Cases from elsewhere, such as a pilot,
go in through the same form, so every example has its consent on record.

`score` reads each released reader from the Hub with plain transformers, claim first and passage second, at
the length the reader was trained for, and divides the logits by the temperature recorded for that reader in
the results files the model cards were written from. The readers are exact conversions of the readers inside
the full systems (`ease/export.py`), so their readings are the systems' readings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "jithinsaireddy/ease-delta"
CASES = ROOT / "evals" / "misreadings.jsonl"
SUMMARY = ROOT / "runs" / "results" / "misreadings.json"
LICENCE = "CC-BY-4.0"

# The form's field ids and the headings GitHub writes for them into the issue body. A test checks these
# against the form itself, so the two cannot drift apart.
HEADINGS = {
    "requirement": "Requirement (claim)",
    "message": "Message or passage",
    "got": "What it said",
    "should": "What it should have said",
    "model": "Model",
    "licence": "May this example be added to a public evaluation set (CC BY 4.0)?",
}
NO_RESPONSE = "_No response_"  # what GitHub writes for an optional field left empty
LABELS = {"supports": "SUPPORTS", "refutes": "REFUTES", "not enough info": "NOT_ENOUGH_INFO"}
CONSENT = "yes"
KEEP_OUT = {"invalid", "duplicate"}

# The released readers, and the results file each one's temperature was fitted in.
READERS = {
    "large": ("jithinpothireddy21/ease-delta-reader", "runs/results_large/edge.json"),
    "base": ("jithinpothireddy21/ease-delta-reader-base", "runs/results/edge.json"),
}

REASONS = (
    ("public", "public, written to the file"),
    ("no_consent", "without consent, counted only"),
    ("kept_out", "kept out by an invalid or duplicate label"),
    ("unreadable", "missing a field the form requires"),
    ("same_label", "the same answer to both questions"),
    ("repeat", "the same case reported again"),
    ("conflict", "conflicting answers for one case"),
)


def sections(body: str) -> dict[str, str]:
    """The form's answers in an issue body, by heading. Only the form's own headings start a section, so a
    line such as "### notes" pasted inside a message stays part of the message."""
    known = set(HEADINGS.values())
    out: dict[str, list[str]] = {}
    current = None
    for line in (body or "").replace("\r\n", "\n").split("\n"):
        heading = line[4:].strip() if line.startswith("### ") else None
        if heading in known and heading not in out:
            current = heading
            out[current] = []
        elif current is not None:
            out[current].append(line)
    answers = {k: "\n".join(v).strip() for k, v in out.items()}
    return {k: ("" if v == NO_RESPONSE else v) for k, v in answers.items()}


def to_label(answer: str) -> str | None:
    return LABELS.get(" ".join(answer.lower().split()))


def consented(answer: str) -> bool:
    return answer.strip().lower() == CONSENT


def norm(text: str) -> str:
    return " ".join(text.split())


def read_issue(issue: dict) -> tuple[dict | None, str]:
    """One issue as a row of the public file, or None and the reason it stays out."""
    if {lab["name"].lower() for lab in issue.get("labels") or []} & KEEP_OUT:
        return None, "kept_out"
    answers = sections(issue.get("body") or "")
    if not consented(answers.get(HEADINGS["licence"], "")):
        return None, "no_consent"
    claim, passage = answers.get(HEADINGS["requirement"], ""), answers.get(HEADINGS["message"], "")
    got, should = to_label(answers.get(HEADINGS["got"], "")), to_label(answers.get(HEADINGS["should"], ""))
    if not claim or not passage or got is None or should is None:
        return None, "unreadable"
    if got == should:
        return None, "same_label"
    n = int(issue["number"])
    return {
        "id": f"gh-{n}",
        "issue": n,
        "url": issue.get("url", f"https://github.com/{REPO}/issues/{n}"),
        "author": ((issue.get("author") or {}).get("login") or ""),
        "created": (issue.get("createdAt") or "")[:10],
        "claim": claim,
        "passage": passage,
        "reported_label": got,
        "gold_label": should,
        "model": answers.get(HEADINGS["model"], ""),
        "licence": LICENCE,
    }, "public"


def build(issues: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """The public rows, sorted by issue number, and how many issues ended where."""
    counts = {key: 0 for key, _ in REASONS}
    by_case: dict[tuple[str, str], list[dict]] = {}
    for issue in sorted(issues, key=lambda i: int(i["number"])):
        row, why = read_issue(issue)
        if row is None:
            counts[why] += 1
        else:
            by_case.setdefault((norm(row["claim"]), norm(row["passage"])), []).append(row)
    rows = []
    for group in by_case.values():
        if len({r["gold_label"] for r in group}) > 1:
            counts["conflict"] += len(group)
            continue
        rows.append(group[0])
        counts["repeat"] += len(group) - 1
    rows.sort(key=lambda r: r["issue"])
    counts["public"] = len(rows)
    return rows, counts


def render(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def fetch(repo: str, limit: int) -> list[dict]:
    fields = "number,url,author,createdAt,labels,body"
    cmd = ["gh", "issue", "list", "--repo", repo, "--label", "misreading", "--state", "all",
           "--limit", str(limit), "--json", fields]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("gh is not installed; install the GitHub CLI or pass --issues-json")
    except subprocess.CalledProcessError as e:
        sys.exit(f"gh failed: {e.stderr.strip()}")
    issues = json.loads(out or "[]")
    if len(issues) >= limit:
        sys.exit(f"{len(issues)} reports reached --limit {limit}; raise it so none is left out")
    return issues


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sync(a) -> int:
    issues = json.loads(Path(a.issues_json).read_text()) if a.issues_json else fetch(a.repo, a.limit)
    rows, counts = build(issues)
    out = Path(a.out)
    print(f"misreading reports in {a.issues_json or a.repo}: {len(issues)}")
    for key, words in REASONS:
        print(f"  {words}: {counts[key]}")
    text = render(rows)
    old = out.read_text(encoding="utf-8") if out.exists() else None
    if not rows and old is None:
        print(f"{rel(out)}: no cases, nothing written")
    elif text == old:
        print(f"{rel(out)}: unchanged, {len(rows)} cases")
    elif a.dry_run:
        print(f"{rel(out)}: would write {len(rows)} cases (dry run, nothing written)")
    else:
        from ease.util import atomic_write_text

        atomic_write_text(out, text)
        print(f"{rel(out)}: wrote {len(rows)} cases")
    return 0


def predict(model, tokenizer, rows: list[dict], temperature: float, max_length: int, device,
            batch_size: int = 16) -> list[dict]:
    """The reader's answer for each case: claim first, passage second, calibrated by the temperature."""
    import torch

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = tokenizer([r["claim"] for r in chunk], [r["passage"] for r in chunk], truncation=True,
                        max_length=max_length, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float().cpu()
        probs = torch.softmax(logits / temperature, dim=-1)
        for row, p in zip(chunk, probs):
            out.append({"id": row["id"], "predicted": id2label[int(p.argmax())],
                        "probabilities": {id2label[j]: round(float(p[j]), 4) for j in sorted(id2label)}})
    return out


def summarize(rows: list[dict], preds: list[dict]) -> dict:
    gold = {r["id"]: r["gold_label"] for r in rows}
    reported = {r["id"]: r["reported_label"] for r in rows}
    correct = sum(p["predicted"] == gold[p["id"]] for p in preds)
    repeats = sum(p["predicted"] == reported[p["id"]] for p in preds)
    return {"cases": len(preds), "correct": correct, "accuracy": correct / len(preds) if preds else None,
            "repeats_reported": repeats}


def score(a) -> int:
    from ease.util import read_json, read_jsonl

    cases = Path(a.cases)
    rows = read_jsonl(cases) if cases.exists() else []
    if not rows:
        print(f"{rel(cases)}: no cases to score")
        return 0
    digest = hashlib.sha256(cases.read_bytes()).hexdigest()
    print(f"{rel(cases)}: {len(rows)} cases, sha256 {digest[:12]}")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ease.util import get_device

    device = get_device(a.device)
    results = []
    for key in (("large", "base") if a.reader == "both" else (a.reader,)):
        name, source = READERS[key]
        temperature = float(read_json(ROOT / source)["temperature"])
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        max_length = int(getattr(model.config, "ease_max_length", None) or tokenizer.model_max_length)
        preds = predict(model, tokenizer, rows, temperature, max_length, device)
        res = {"reader": name, "revision": getattr(model.config, "_commit_hash", None),
               "temperature": temperature, "temperature_from": source, "max_length": max_length,
               **summarize(rows, preds), "predictions": preds}
        results.append(res)
        print(f"{name} (revision {str(res['revision'])[:7]}, temperature {temperature:.3f}): "
              f"{res['correct']} of {res['cases']} correct ({100 * res['accuracy']:.1f}%); "
              f"repeats the reported reading on {res['repeats_reported']}")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    names = [r["reader"].split("/")[-1] for r in results]
    print("\n" + "  ".join(["case".ljust(10), "gold".ljust(16), "reported".ljust(16)] + [n.ljust(30) for n in names]))
    by_reader = [{p["id"]: p for p in r["predictions"]} for r in results]
    for row in rows:
        cells = []
        for preds in by_reader:
            p = preds[row["id"]]
            cells.append(f"{p['predicted']} {p['probabilities'][p['predicted']]:.2f}".ljust(30))
        print("  ".join([row["id"].ljust(10), row["gold_label"].ljust(16), row["reported_label"].ljust(16)] + cells))

    if a.save:
        from ease.util import atomic_write_json

        atomic_write_json(a.save_to, {"cases_file": rel(cases), "sha256": digest, "cases": len(rows),
                                      "device": str(device),
                                      "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      "readers": results})
        print(f"\nwrote {rel(Path(a.save_to))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync", help="rebuild the public file from the misreading issues")
    s.add_argument("--repo", default=REPO)
    s.add_argument("--out", default=str(CASES))
    s.add_argument("--limit", type=int, default=1000, help="most issues to read; reaching it is an error")
    s.add_argument("--issues-json", help="read issues from this file (the JSON gh prints) instead of GitHub")
    s.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("score", help="measure the released readers on the public file")
    c.add_argument("--cases", default=str(CASES))
    c.add_argument("--reader", choices=("both", "large", "base"), default="both")
    c.add_argument("--device", default=None, help="default: EASE_DEVICE, else the best available")
    c.add_argument("--save", action="store_true", help=f"also write {rel(SUMMARY)}")
    c.add_argument("--save-to", default=str(SUMMARY))
    a = ap.parse_args(argv)
    return sync(a) if a.cmd == "sync" else score(a)


if __name__ == "__main__":
    sys.exit(main())
