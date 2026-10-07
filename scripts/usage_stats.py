"""Who is using it? The numbers the Hub and GitHub keep, in one place.

    python scripts/usage_stats.py

Hub: downloads (last 30 days and all time), likes and discussions of the two systems and the two
readers, and likes of the demo Space. GitHub: stars, forks, watchers, issues, pull requests, and
the owner-only traffic figures (views and clones over the last 14 days, referrers). The GitHub
traffic calls need `gh` logged in as the repository owner. GitHub Pages keeps no visitor counts
for the demo page, and the Hub API reports none for the Space.

What the numbers mean: a Hub download is counted when the repository's root config.json is
fetched, so one person running `hf download` counts once; the counter lags by up to a day.
GitHub views count page visits by anyone; clones count git clones, not Actions checkouts.
"""

from __future__ import annotations

import json
import subprocess
import sys

MODELS = ("jithinpothireddy21/ease-delta", "jithinpothireddy21/ease-delta-base",
          "jithinpothireddy21/ease-delta-reader", "jithinpothireddy21/ease-delta-reader-base")
SPACE = "jithinpothireddy21/ease-delta-demo"
REPO = "jithinsaireddy/ease-delta"


def hub() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for r in MODELS:
        i = api.model_info(r, expand=["downloads", "downloadsAllTime", "likes", "private", "lastModified"])
        try:
            n_disc = len(list(api.get_repo_discussions(r, repo_type="model")))
        except Exception:
            n_disc = "?"
        print(f"{r}: {'private' if i.private else 'public'}; downloads last 30 days {i.downloads}, "
              f"all time {getattr(i, 'downloads_all_time', '?')}; likes {i.likes}; discussions {n_disc}; "
              f"last modified {i.last_modified:%Y-%m-%d %H:%M}")
    try:
        sp = api.space_info(SPACE)
        print(f"{SPACE} (Space): likes {sp.likes}; the Hub API reports no visit counts")
    except Exception as e:  # noqa: BLE001
        print(f"{SPACE} (Space): could not read ({type(e).__name__})")
    print("pages: " + "  ".join(f"https://huggingface.co/{r}" for r in MODELS))


def gh(args: list[str]) -> dict | list | None:
    try:
        out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout
        return json.loads(out) if out.strip() else None
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None


def github() -> None:
    v = gh(["repo", "view", REPO, "--json", "visibility,stargazerCount,forkCount,watchers,issues,pullRequests"])
    if v is None:
        print(f"GitHub: could not read {REPO} (is gh logged in?)")
        return
    print(f"{REPO}: {v['visibility'].lower()}; stars {v['stargazerCount']}, forks {v['forkCount']}, "
          f"watchers {v['watchers']['totalCount']}, open issues {v['issues']['totalCount']}, open pull requests "
          f"{v['pullRequests']['totalCount']}")
    views, clones = gh(["api", f"repos/{REPO}/traffic/views"]), gh(["api", f"repos/{REPO}/traffic/clones"])
    refs = gh(["api", f"repos/{REPO}/traffic/popular/referrers"]) or []
    if views is not None and clones is not None:
        print(f"last 14 days: {views['count']} views by {views['uniques']} visitors; {clones['count']} clones by "
              f"{clones['uniques']} cloners; referrers: " + (", ".join(f"{x['referrer']} ({x['count']})" for x in refs) or "none"))
    print(f"insights: https://github.com/{REPO}/graphs/traffic")


if __name__ == "__main__":
    hub()
    github()
    sys.exit(0)
