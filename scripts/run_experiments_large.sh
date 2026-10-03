#!/bin/bash
# The pre-registered addendum (docs/PREREGISTRATION_LARGE.md): evaluates the large edge model with the
# same scripts as the main pipeline, into runs/results_large, then compares it with the released model.
#
#   bash scripts/run_experiments_large.sh            # everything
#   bash scripts/run_experiments_large.sh episodes   # one step
#
# The dense baseline and the public model are not re-run: they were measured once and do not change.
set -uo pipefail
ROOT="${EASE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
S="${EASE_SCRIPTS:-scripts}"
PY="$ROOT/.venv/bin/python"
EDGE=runs/stage_a_large/final
SB=runs/stage_b_large
OUT=runs/results_large
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
  echo "no trained large edge model at $EDGE" | tee -a "$LOG"; exit 1
fi

step edge        $PY $S/eval_edge.py --model $EDGE --out $OUT/edge.json --save-logits $OUT/edge_logits
step stage_b     $PY -m ease.train.stage_b --edge $EDGE --out $SB --seeds 1
step suite       $PY $S/build_regression_suite.py --edge $EDGE
step episodes    $PY $S/eval_episodes.py --edge $EDGE --stage-b $SB --out $OUT/episodes.json --work $OUT/work --skip-dense --skip-public
step small       $PY $S/eval_small.py --edge $EDGE --stage-b $SB --work $OUT/work --out $OUT/small.json
step set_size    $PY $S/eval_set_size.py --stage-b $SB --work $OUT/work --out $OUT/set_size.json
step exactness   $PY $S/eval_exactness.py --edge $EDGE --stage-b $SB --out $OUT/exactness.json
step stale       $PY $S/eval_stale.py --edge $EDGE --temperature-from $OUT/edge.json --proposer-threshold-from runs/results/proposer.json --out $OUT/stale.json
step headroom    $PY $S/diagnose_headroom.py --edge $EDGE --stage-b $SB --results $OUT --out $OUT/headroom.json
# timing last, both models in one run on an idle GPU
step timing_base  $PY $S/eval_cpu.py --device mps --label base  --model runs/stage_a/final --rules-from runs/stage_b --out runs/results/timing_large.json
step timing_large $PY $S/eval_cpu.py --device mps --label large --model $EDGE --rules-from $SB --out runs/results/timing_large.json
step timing_large_cpu $PY $S/eval_cpu.py --device cpu --label "large, CPU" --model $EDGE --rules-from $SB --out runs/results/timing_large.json
step compare     $PY $S/compare_large.py --base runs/results --large $OUT --base-stage-b runs/stage_b --large-stage-b $SB --timing runs/results/timing_large.json --out runs/results/large.json
step report      $PY $S/make_report.py --results runs/results --out docs/RESULTS.md
echo "=== $(date '+%Y-%m-%d %H:%M:%S') LARGE PIPELINE DONE" | tee -a "$LOG"
