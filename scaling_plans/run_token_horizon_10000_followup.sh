#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PLAN_JSON="${PLAN_JSON:-${SCRIPT_DIR}/token_horizon_10000_steps_512_2048.json}"
ROOT_OUTDIR="${ROOT_OUTDIR:-${REPO_ROOT}/scaling_runs/horizon_10000_steps_512_2048}"
PLOTS_OUTDIR="${PLOTS_OUTDIR:-${REPO_ROOT}/scaling_runs/plots}"
SUMMARY_JSON="${SUMMARY_JSON:-${ROOT_OUTDIR}/horizon_branch_summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${ROOT_OUTDIR}/horizon_branch_report.md}"
PLOT_PNG="${PLOT_PNG:-${PLOTS_OUTDIR}/horizon_10000_steps_512_2048_bars.png}"

RUNNER="${SCRIPT_DIR}/run_token_horizon_branch.sh"
SUMMARIZER="${SCRIPT_DIR}/summarize_horizon_branch.py"
PLOTTER="${REPO_ROOT}/scaling_runs/plots/make_extended_horizon_plot.py"

if [[ ! -f "${PLAN_JSON}" ]]; then
  echo "missing plan: ${PLAN_JSON}" >&2
  exit 1
fi
if [[ ! -x "${RUNNER}" ]]; then
  echo "missing runner: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${SUMMARIZER}" ]]; then
  echo "missing summarizer: ${SUMMARIZER}" >&2
  exit 1
fi
if [[ ! -f "${PLOTTER}" ]]; then
  echo "missing plotter: ${PLOTTER}" >&2
  exit 1
fi

mkdir -p "${ROOT_OUTDIR}" "${PLOTS_OUTDIR}"
cp "${PLAN_JSON}" "${ROOT_OUTDIR}/plan.json"

echo "[10000-followup] plan=${PLAN_JSON}"
echo "[10000-followup] root_outdir=${ROOT_OUTDIR}"
echo "[10000-followup] summary_json=${SUMMARY_JSON}"
echo "[10000-followup] plot_png=${PLOT_PNG}"

PLAN_JSON="${PLAN_JSON}" ROOT_OUTDIR="${ROOT_OUTDIR}" "${RUNNER}"

python3 "${SUMMARIZER}" \
  --root-outdir "${ROOT_OUTDIR}" \
  --plan-json "${PLAN_JSON}" \
  --summary-json "${SUMMARY_JSON}" \
  --summary-md "${SUMMARY_MD}"

python3 "${PLOTTER}" \
  --summary-json "${SUMMARY_JSON}" \
  --out-png "${PLOT_PNG}"

echo "[10000-followup] complete"
