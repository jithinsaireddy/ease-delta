#!/bin/bash
# Waits for the edge model to finish training, freezes the code, then runs every experiment
# from the frozen copy. Later edits to the working tree cannot affect a run in progress.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
until [ -f runs/stage_a/final/training_manifest.json ]; do
  if ! pgrep -f "ease.train.stage_a" > /dev/null; then
    sleep 20
    [ -f runs/stage_a/final/training_manifest.json ] && break
    echo "$(date) training process is gone and no final model exists" >> runs/results_waiter.log
    exit 1
  fi
  sleep 30
done
# Development-set gate: test sets are not touched by a model whose development metrics look broken.
# Thresholds come from development data only (a pilot reached 82% on VitaminC dev after 64k examples).
GATE=$("$ROOT/.venv/bin/python" - <<'PY'
import json
m = json.load(open("runs/stage_a/final/training_manifest.json"))
d = m["dev_metrics_full"]
need = {"vitaminc.dev": 0.83, "mnli.dev": 0.86, "unrelated.dev": 0.99}
bad = [f"{k} {d[k]['accuracy']:.4f} < {v}" for k, v in need.items() if d[k]["accuracy"] < v]
if m.get("label_mix_drift_warnings", 0) > 0:
    bad.append(f"{m['label_mix_drift_warnings']} label-mix drift warnings during training")
print("PASS" if not bad else "FAIL: " + "; ".join(bad))
PY
)
echo "$(date) development gate: $GATE" >> runs/results_waiter.log
case "$GATE" in PASS*) ;; *) echo "$(date) experiments NOT started" >> runs/results_waiter.log; exit 2 ;; esac
SNAP="$ROOT/runs/code_snapshot"
rm -rf "$SNAP" && mkdir -p "$SNAP"
cp -R src scripts configs docs/PREREGISTRATION.md pyproject.toml "$SNAP"/
find "$SNAP" -name "__pycache__" -type d -prune -exec rm -rf {} +
( cd "$SNAP" && find . -type f ! -name CODE_SHA256 | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | cut -d" " -f1 > CODE_SHA256 )
echo "$(date) training finished; code frozen at $SNAP ($(cat "$SNAP/CODE_SHA256")); starting experiments" >> runs/results_waiter.log
export EASE_ROOT="$ROOT" EASE_SCRIPTS="$SNAP/scripts" PYTHONPATH="$SNAP/src"
exec caffeinate -i -s bash "$SNAP/scripts/run_experiments.sh"
