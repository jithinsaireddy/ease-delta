#!/bin/bash
# Waits for the large edge model, applies the development gate of the main pipeline, then runs the
# addendum's experiments (docs/PREREGISTRATION_LARGE.md). Test splits are not touched by a model
# that fails the gate.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG=runs/results_large_waiter.log
until [ -f runs/stage_a_large/final/training_manifest.json ]; do
  if ! pgrep -f "configs/stage_a_large.yaml" > /dev/null; then
    sleep 20
    [ -f runs/stage_a_large/final/training_manifest.json ] && break
    echo "$(date) training process is gone and no final model exists" >> "$LOG"
    exit 1
  fi
  sleep 60
done
GATE=$("$ROOT/.venv/bin/python" - <<'PY'
import json
m = json.load(open("runs/stage_a_large/final/training_manifest.json"))
d = m["dev_metrics_full"]
need = {"vitaminc.dev": 0.83, "mnli.dev": 0.86, "unrelated.dev": 0.99}
bad = [f"{k} {d[k]['accuracy']:.4f} < {v}" for k, v in need.items() if d[k]["accuracy"] < v]
if m.get("label_mix_drift_warnings", 0) > 0:
    bad.append(f"{m['label_mix_drift_warnings']} label-mix drift warnings during training")
print("PASS" if not bad else "FAIL: " + "; ".join(bad))
PY
)
echo "$(date) development gate (large): $GATE" >> "$LOG"
case "$GATE" in PASS*) ;; *) echo "$(date) experiments NOT started" >> "$LOG"; exit 2 ;; esac
echo "$(date) starting the addendum's experiments" >> "$LOG"
exec caffeinate -i -s bash scripts/run_experiments_large.sh
