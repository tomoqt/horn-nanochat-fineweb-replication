# HORN NanoChat-Style FineWeb Replication

This repository contains a scaled-down pretraining replication of higher-order residual (HORN-style) updates against a baseline residual transformer, with experiments tracked in a Flywheel graph.

## Main Entrypoints
- Benchmark runner: `horn_nanochat_benchmark.py`
- Model-size branch plan/runner: `scaling_plans/model_size_branch.json`, `scaling_plans/run_model_size_branch.sh`
- Token-horizon branch plan/runner (initial): `scaling_plans/token_horizon_branch.json`, `scaling_plans/run_token_horizon_branch.sh`
- Token-horizon branch plan (extended): `scaling_plans/token_horizon_extended_branch.json`
- Token-horizon long-step follow-up (1200 steps): `scaling_plans/token_horizon_long_steps_512_2048.json`
- Horizon summary script: `scaling_plans/summarize_horizon_branch.py`
- Joint-plan selector: `scaling_plans/prepare_joint_plan.py`
- Joint runner: `scaling_plans/run_joint_scaling.sh`
- Plotting: `scaling_runs/plots/make_loss_plots.py`

## Reproducibility
- Repro guide: `scaling_runs/repro/REPRODUCE.md`
- Repro manifest: `scaling_runs/repro/replication_manifest.json`
- Node bundle map: `scaling_runs/repro/node_bundle_manifest.json`
- Node-index mapping (graph -> repo paths): `docs/GRAPH_NODE_INDEX.md`

## Latest Long-Step Follow-Up
- Output root: `scaling_runs/horizon_long_steps_512_2048/`
- Summary: `scaling_runs/horizon_long_steps_512_2048/horizon_branch_summary.json`
- 600-vs-1200 comparison: `scaling_runs/horizon_long_steps_512_2048/horizon_600_vs_1200_comparison.md`

## Dataset
Experiments are configured for FineWeb (`--dataset fineweb`) and set to fail if FineWeb loading fails, rather than silently falling back to TinyShakespeare.
