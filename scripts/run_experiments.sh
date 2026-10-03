#!/bin/bash
# Runs every experiment of docs/PREREGISTRATION.md, in order, once the edge model is trained.
# Each step writes JSON under runs/results/. A failed step is logged and the rest still run.
#
#   bash scripts/run_experiments.sh            # everything
#   bash scripts/run_experiments.sh timing     # one step
#
# EASE_ROOT     repository root (default: the parent of this script's directory)
# EASE_SCRIPTS  where the experiment scripts are read from (default: scripts)
# Set both, with PYTHONPATH, to run from a frozen snapshot of the code.
set -uo pipefail
ROOT="${EASE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
S="${EASE_SCRIPTS:-scripts}"
PY="$ROOT/.venv/bin/python"
EDGE=runs/stage_a/final
OUT=runs/results
LOG=$OUT/pipeline.log
mkdir -p "$OUT"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ONLY="${1:-all}"

step() {
  local name="$1"; shift
  if [ "$ONLY" != "all" ] && [ "$ONLY" != "$name" ]; then return 0; fi
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') START $name" | tee -a "$LOG"
  local t0=$(date +%s)
  "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') END   $name exit=$rc seconds=$(( $(date +%s) - t0 ))" | tee -a "$LOG"
  return 0
}

if [ ! -f "$EDGE/model.safetensors" ]; then
  echo "no trained edge model at $EDGE" | tee -a "$LOG"; exit 1
fi

step edge        $PY $S/eval_edge.py --model $EDGE --out $OUT/edge.json --save-logits $OUT/edge_logits
step numeric     $PY $S/analyze_numeric.py --logits $OUT/edge_logits --out $OUT/numeric.json
step edge_b3     $PY $S/eval_edge.py --hf tasksource/ModernBERT-base-nli --out $OUT/edge_b3.json
step stage_b     $PY -m ease.train.stage_b --edge $EDGE --out runs/stage_b
step suite       $PY $S/build_regression_suite.py --edge $EDGE
step dense       $PY -m ease.train.dense --edge $EDGE --out runs/dense
step episodes    $PY $S/eval_episodes.py --edge $EDGE --stage-b runs/stage_b --dense runs/dense/final --out $OUT/episodes.json --work $OUT/work
step proposer    $PY $S/eval_proposer.py --stage-b runs/stage_b --work $OUT/work --out $OUT/proposer.json
step stale       $PY $S/eval_stale.py --edge $EDGE --temperature-from $OUT/edge.json --proposer-threshold-from $OUT/proposer.json --out $OUT/stale.json
step evolution   $PY $S/eval_evolution.py --edge $EDGE --stage-b runs/stage_b --out $OUT/evolution.json --work $OUT/work/evolution
step joins       $PY $S/eval_joins.py --edge $EDGE --stage-b runs/stage_b --dense runs/dense/final --out $OUT/joins.json
step exactness   $PY $S/eval_exactness.py --edge $EDGE --stage-b runs/stage_b --out $OUT/exactness.json
# timing last: nothing else may be using the GPU while it runs
step timing      $PY $S/eval_timing.py --edge $EDGE --stage-b runs/stage_b --dense runs/dense/final --standard 80 --wide 25 --out $OUT/timing.json
step timing_canonical $PY $S/eval_timing.py --edge $EDGE --stage-b runs/stage_b --dense runs/dense/final --canonical --standard 40 --wide 12 --out $OUT/timing_canonical.json
step probe       $PY $S/mechanism_probe.py --out runs/mechanism_probe.json
step report      $PY $S/make_report.py --results $OUT --out docs/RESULTS.md
step release     $PY $S/package_release.py --out release
echo "=== $(date '+%Y-%m-%d %H:%M:%S') PIPELINE DONE" | tee -a "$LOG"
