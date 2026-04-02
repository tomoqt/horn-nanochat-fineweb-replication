# Flywheel Graph -> Repo Index

This document maps each Flywheel node to the exact code/config/results paths in this repository.

## Graph Roots
- Original HORN root: `f3c1b9cf-5904-57e5-abc7-ad8b91789f89`
- Scaling-study root: `14c6d442-4846-5f3f-9f26-bf878ff563bf`

## Existing Nodes
- `72f74c83-265e-5bd3-bea7-d10b5ed72d12` (model-size branch empirical)
  - `scaling_plans/model_size_branch.json`
  - `scaling_plans/run_model_size_branch.sh`
  - `scaling_runs/model_branch/`
- `f8f939dd-096f-57f1-8a2c-bd507ecb1b08` (token-horizon branch empirical, 128/256/384)
  - `scaling_plans/token_horizon_branch.json`
  - `scaling_plans/run_token_horizon_branch.sh`
  - `scaling_runs/horizon_branch/`
- `502fe30b-fefc-5ea0-a6b5-ba448941cabf` (branch synthesis insight)
  - `scaling_plans/prepare_joint_plan.py`
  - `scaling_runs/joint_branch/joint_plan.json`
  - `scaling_runs/joint_branch/joint_plan.md`
- `797f2040-c59f-518c-b492-7c133f3c7a36` (joint branch empirical)
  - `scaling_plans/run_joint_scaling.sh`
  - `scaling_runs/joint_branch/joint_scaled_from_branch_winners/benchmark_summary.json`
  - `scaling_runs/joint_branch/joint_branch_summary.json`
- `8a3d53c5-c68c-5c4b-8a41-58a4c52f6a0c` (final insight)
  - `scaling_runs/plots/model_size_val_curves.png`
  - `scaling_runs/plots/horizon_final_loss_bars.png`
  - `scaling_runs/plots/joint_val_curves.png`
  - `scaling_runs/plots/loss_plot_metrics.json`

## New Extended-Horizon Work (Beyond 384)
- Plan/config:
  - `scaling_plans/token_horizon_extended_branch.json`
  - `scaling_plans/summarize_horizon_branch.py`
- Result paths (after run):
  - `scaling_runs/horizon_branch_extended/`
  - `scaling_runs/horizon_branch_extended/horizon_branch_summary.json`
  - `scaling_runs/horizon_branch_extended/horizon_branch_report.md`

If a new Flywheel node is created for this extended sweep, add its node ID here and include these paths in that node's artifacts/summary.
