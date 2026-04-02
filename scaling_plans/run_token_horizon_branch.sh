#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_JSON="${PLAN_JSON:-${SCRIPT_DIR}/token_horizon_branch.json}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-$(cd "${SCRIPT_DIR}/.." && pwd)/horn_nanochat_benchmark.py}"
ROOT_OUTDIR="${ROOT_OUTDIR:-/workspace/scaling/token_horizon}"
DATASET="${DATASET:-fineweb}"
DATA_PATH="${DATA_PATH:-/workspace/data/fineweb_sample.txt}"
FINEWEB_TARGET_CHARS="${FINEWEB_TARGET_CHARS:-8000000}"
FINEWEB_MAX_DOCS="${FINEWEB_MAX_DOCS:-40000}"

if [[ ! -f "${BENCHMARK_SCRIPT}" ]]; then
  echo "Missing benchmark script: ${BENCHMARK_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${PLAN_JSON}" ]]; then
  echo "Missing plan JSON: ${PLAN_JSON}" >&2
  exit 1
fi

run_branch() {
  local name="$1"
  local block_size="$2"
  local batch_size="$3"
  local steps="$4"
  local eval_interval="$5"
  local eval_iters="$6"
  local horn_m_init="$7"
  local outdir="${ROOT_OUTDIR}/${name}"

  mkdir -p "${outdir}"
  echo "=== Running ${name} ==="
  echo "outdir=${outdir}"
  echo "block_size=${block_size} batch_size=${batch_size} steps=${steps}"
  python3 "${BENCHMARK_SCRIPT}" \
    --dataset "${DATASET}" \
    --data-path "${DATA_PATH}" \
    --fineweb-target-chars "${FINEWEB_TARGET_CHARS}" \
    --fineweb-max-docs "${FINEWEB_MAX_DOCS}" \
    --variants baseline horn \
    --seeds 1337 2027 \
    --steps "${steps}" \
    --eval-interval "${eval_interval}" \
    --eval-iters "${eval_iters}" \
    --batch-size "${batch_size}" \
    --block-size "${block_size}" \
    --n-layer 4 \
    --n-head 4 \
    --n-embd 128 \
    --dropout 0.0 \
    --horn-m-init "${horn_m_init}" \
    --outdir "${outdir}" \
    2>&1 | tee "${outdir}/run.log"
}

if command -v jq >/dev/null 2>&1; then
  mapfile -t branch_lines < <(jq -r '.branches[] | "\(.name) \(.block_size) \(.batch_size) \(.tokens_per_step) \(.outdir_name)"' "${PLAN_JSON}")
  steps="$(jq -r '.shared_training.steps' "${PLAN_JSON}")"
  eval_interval="$(jq -r '.shared_training.eval_interval' "${PLAN_JSON}")"
  eval_iters="$(jq -r '.shared_training.eval_iters' "${PLAN_JSON}")"
  horn_m_init="$(jq -r '.shared_training.horn_m_init' "${PLAN_JSON}")"
  for line in "${branch_lines[@]}"; do
    read -r name block_size batch_size tokens_per_step outdir_name <<<"${line}"
    echo "planned_tokens_per_step=${tokens_per_step}"
    run_branch "${outdir_name}" "${block_size}" "${batch_size}" "${steps}" "${eval_interval}" "${eval_iters}" "${horn_m_init}"
  done
else
  echo "jq not found; using hardcoded branch definitions from the plan."
  run_branch "horizon_128" 128 32 500 100 20 0.5
  run_branch "horizon_256" 256 16 500 100 20 0.5
  run_branch "horizon_384" 384 11 500 100 20 0.5
fi
