#!/usr/bin/env bash
set -euo pipefail

PLAN_JSON="${1:-/workspace/scaling/joint/joint_plan.json}"
SCRIPT_PATH="${2:-/workspace/horn_nanochat_benchmark.py}"
OUT_ROOT="${3:-/workspace/scaling/joint}"
DATASET="${DATASET:-fineweb}"
DATA_PATH="${DATA_PATH:-/workspace/data/fineweb_sample.txt}"
FINEWEB_TARGET_CHARS="${FINEWEB_TARGET_CHARS:-8000000}"
FINEWEB_MAX_DOCS="${FINEWEB_MAX_DOCS:-40000}"

if [[ ! -f "$PLAN_JSON" ]]; then
  echo "missing joint plan json: $PLAN_JSON" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "missing benchmark script: $SCRIPT_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

if command -v jq >/dev/null 2>&1; then
  NLAYER="$(jq -r '.run_config.n_layer' "$PLAN_JSON")"
  NHEAD="$(jq -r '.run_config.n_head' "$PLAN_JSON")"
  NEMBD="$(jq -r '.run_config.n_embd' "$PLAN_JSON")"
  BLOCK="$(jq -r '.run_config.block_size' "$PLAN_JSON")"
  BATCH="$(jq -r '.run_config.batch_size' "$PLAN_JSON")"
  STEPS="$(jq -r '.run_config.steps' "$PLAN_JSON")"
  EVAL_INTERVAL="$(jq -r '.run_config.eval_interval' "$PLAN_JSON")"
  EVAL_ITERS="$(jq -r '.run_config.eval_iters' "$PLAN_JSON")"
  M_INIT="$(jq -r '.run_config.horn_m_init' "$PLAN_JSON")"
  mapfile -t SEEDS < <(jq -r '.run_config.seeds[]' "$PLAN_JSON")
else
  # Fallback parser using python if jq is not available.
  read -r NLAYER NHEAD NEMBD BLOCK BATCH STEPS EVAL_INTERVAL EVAL_ITERS M_INIT SEED1 SEED2 < <(
    python3 - <<'PY' "$PLAN_JSON"
import json,sys
p=json.load(open(sys.argv[1]))
c=p["run_config"]
seeds=c["seeds"]
print(c["n_layer"], c["n_head"], c["n_embd"], c["block_size"], c["batch_size"], c["steps"], c["eval_interval"], c["eval_iters"], c["horn_m_init"], seeds[0], seeds[1])
PY
  )
  SEEDS=("$SEED1" "$SEED2")
fi

SEEDS_STR="${SEEDS[*]}"
OUTDIR="$OUT_ROOT/joint_scaled_from_branch_winners"
LOGFILE="$OUT_ROOT/joint_scaled_from_branch_winners.log"

echo "Running joint scaling config:"
echo "  n_layer=$NLAYER n_head=$NHEAD n_embd=$NEMBD block_size=$BLOCK batch_size=$BATCH"
echo "  steps=$STEPS eval_interval=$EVAL_INTERVAL eval_iters=$EVAL_ITERS seeds=$SEEDS_STR"

python3 "$SCRIPT_PATH" \
  --dataset "$DATASET" \
  --data-path "$DATA_PATH" \
  --fineweb-target-chars "$FINEWEB_TARGET_CHARS" \
  --fineweb-max-docs "$FINEWEB_MAX_DOCS" \
  --variants baseline horn \
  --seeds ${SEEDS_STR} \
  --steps "$STEPS" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-iters "$EVAL_ITERS" \
  --batch-size "$BATCH" \
  --block-size "$BLOCK" \
  --n-layer "$NLAYER" \
  --n-head "$NHEAD" \
  --n-embd "$NEMBD" \
  --horn-m-init "$M_INIT" \
  --outdir "$OUTDIR" | tee "$LOGFILE"

echo "Joint scaling run complete."
echo "  summary: $OUTDIR/benchmark_summary.json"
echo "  log:     $LOGFILE"
