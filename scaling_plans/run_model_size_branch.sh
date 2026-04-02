#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLAN_JSON="${PLAN_JSON:-${SCRIPT_DIR}/model_size_branch.json}"
SCRIPT_PATH="${SCRIPT_PATH:-${ROOT_DIR}/horn_nanochat_benchmark.py}"
OUT_ROOT="${OUT_ROOT:-/workspace/scaling/model_size}"
DATASET="${DATASET:-fineweb}"
DATA_PATH="${DATA_PATH:-/workspace/data/fineweb_sample.txt}"
FINEWEB_TARGET_CHARS="${FINEWEB_TARGET_CHARS:-8000000}"
FINEWEB_MAX_DOCS="${FINEWEB_MAX_DOCS:-40000}"

mkdir -p "${OUT_ROOT}"

log() {
  printf '[model-size-branch] %s\n' "$*"
}

run_one() {
  local exp_name="$1"
  local variant="$2"
  local n_layer="$3"
  local n_head="$4"
  local n_embd="$5"
  local batch_size="$6"
  local block_size="$7"
  local steps="$8"
  local eval_interval="$9"
  local eval_iters="${10}"
  local horn_m_init="${11}"
  local seed="${12}"

  local outdir="${OUT_ROOT}/${exp_name}/${variant}/seed_${seed}"
  mkdir -p "${outdir}"
  log "run exp=${exp_name} variant=${variant} seed=${seed} layers=${n_layer} heads=${n_head} embd=${n_embd} batch=${batch_size} outdir=${outdir}"

  python3 "${SCRIPT_PATH}" \
    --dataset "${DATASET}" \
    --data-path "${DATA_PATH}" \
    --fineweb-target-chars "${FINEWEB_TARGET_CHARS}" \
    --fineweb-max-docs "${FINEWEB_MAX_DOCS}" \
    --variants "${variant}" \
    --seeds "${seed}" \
    --steps "${steps}" \
    --eval-interval "${eval_interval}" \
    --eval-iters "${eval_iters}" \
    --batch-size "${batch_size}" \
    --block-size "${block_size}" \
    --n-layer "${n_layer}" \
    --n-head "${n_head}" \
    --n-embd "${n_embd}" \
    --horn-m-init "${horn_m_init}" \
    --outdir "${outdir}" \
    | tee "${outdir}/run.log"
}

run_from_json_with_jq() {
  local branch_count
  branch_count="$(jq '.branches | length' "${PLAN_JSON}")"
  local block_size steps eval_interval eval_iters horn_m_init
  block_size="$(jq -r '.fixed_hyperparams.block_size' "${PLAN_JSON}")"
  steps="$(jq -r '.fixed_hyperparams.steps' "${PLAN_JSON}")"
  eval_interval="$(jq -r '.fixed_hyperparams.eval_interval' "${PLAN_JSON}")"
  eval_iters="$(jq -r '.fixed_hyperparams.eval_iters' "${PLAN_JSON}")"
  horn_m_init="$(jq -r '.fixed_hyperparams.horn_m_init' "${PLAN_JSON}")"
  mapfile -t seeds < <(jq -r '.fixed_hyperparams.seeds[]' "${PLAN_JSON}")

  for ((i = 0; i < branch_count; i++)); do
    local size exp_name n_layer n_head n_embd batch_size
    IFS=$'\t' read -r size exp_name n_layer n_head n_embd batch_size < <(
      jq -r ".branches[${i}] | [.size, .exp_name, .n_layer, .n_head, .n_embd, .batch_size] | @tsv" "${PLAN_JSON}"
    )

    log "branch=${size} exp_name=${exp_name}"
    mapfile -t variants < <(jq -r ".branches[${i}].variants[]" "${PLAN_JSON}")

    for variant in "${variants[@]}"; do
      for seed in "${seeds[@]}"; do
        run_one "${exp_name}" "${variant}" "${n_layer}" "${n_head}" "${n_embd}" "${batch_size}" "${block_size}" "${steps}" "${eval_interval}" "${eval_iters}" "${horn_m_init}" "${seed}"
      done
    done
  done
}

run_from_hardcoded_defaults() {
  local block_size=128
  local steps=500
  local eval_interval=100
  local eval_iters=20
  local horn_m_init=0.5
  local seeds=(1337 2027)

  local branch_specs=(
    "small_fixed_horizon:2:2:64:32"
    "medium_fixed_horizon:4:4:128:24"
    "large_fixed_horizon:6:8:192:12"
  )
  local variants=(baseline horn)

  for spec in "${branch_specs[@]}"; do
    IFS=: read -r exp_name n_layer n_head n_embd batch_size <<< "${spec}"
    log "branch=${exp_name}"
    for variant in "${variants[@]}"; do
      for seed in "${seeds[@]}"; do
        run_one "${exp_name}" "${variant}" "${n_layer}" "${n_head}" "${n_embd}" "${batch_size}" "${block_size}" "${steps}" "${eval_interval}" "${eval_iters}" "${horn_m_init}" "${seed}"
      done
    done
  done
}

main() {
  if [[ ! -f "${PLAN_JSON}" ]]; then
    log "missing plan json: ${PLAN_JSON}"
    exit 1
  fi
  if [[ ! -f "${SCRIPT_PATH}" ]]; then
    log "missing benchmark script: ${SCRIPT_PATH}"
    exit 1
  fi

  log "plan=${PLAN_JSON}"
  log "benchmark=${SCRIPT_PATH}"
  log "out_root=${OUT_ROOT}"
  log "dataset=${DATASET} data_path=${DATA_PATH}"

  if command -v jq >/dev/null 2>&1; then
    log "using jq-driven plan expansion"
    run_from_json_with_jq
  else
    log "jq not found; using hardcoded fallback plan"
    run_from_hardcoded_defaults
  fi
}

main "$@"
