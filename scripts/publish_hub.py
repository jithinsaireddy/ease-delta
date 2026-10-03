"""Upload the release directory to a Hugging Face model repository.

    python scripts/publish_hub.py --repo jithinpothireddy21/ease-delta --release release [--private] [--dry-run]

Uploads every file in the release directory (weights, tokenizer, aggregator, regression suites,
settings, MANIFEST.json, MODEL_CARD.md as README.md) and the weights' licence text. Uses the
token the `hf` CLI has stored; the token is never printed. Nothing is deleted on the Hub.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--release", default="release")
    ap.add_argument("--licence-text", default="docs/licenses/CC-BY-SA-4.0.txt")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--message", default="Release")
    ap.add_argument("--note", default=None, help="a paragraph placed under the card's title, e.g. to mark a variant")
    a = ap.parse_args()
    rel = Path(a.release)
    if not (rel / "MANIFEST.json").exists() or not (rel / "MODEL_CARD.md").exists():
        print("no packaged release: run scripts/package_release.py first", file=sys.stderr)
        return 2
    card = (rel / "MODEL_CARD.md").read_text()
    if not card.startswith("---\nlicense:"):
        print("the model card has no licence metadata: package with --weights-licence", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "repo"
        shutil.copytree(rel, stage)
        if a.note:
            head, _, rest = card.partition("\n# Model card: EASE-Delta\n")
            card = head + "\n# Model card: EASE-Delta\n\n" + a.note.strip() + "\n" + rest
        (stage / "README.md").write_text(card)
        (stage / "MODEL_CARD.md").unlink()
        if Path(a.licence_text).exists():
            shutil.copy2(a.licence_text, stage / "LICENSE")
        files = sorted(p.relative_to(stage) for p in stage.rglob("*") if p.is_file())
        total = sum((stage / f).stat().st_size for f in files)
        print(f"{len(files)} files, {total/1e6:.1f} MB ->", a.repo, "(private)" if a.private else "(public)")
        for f in files:
            print("  ", f)
        if a.dry_run:
            return 0
        from huggingface_hub import HfApi

        api = HfApi()
        me = api.whoami()["name"]
        print("as", me)
        api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
        info = api.upload_folder(folder_path=str(stage), repo_id=a.repo, repo_type="model", commit_message=a.message)
        print("uploaded:", getattr(info, "commit_url", info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
