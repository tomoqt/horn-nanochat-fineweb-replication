# HORN Scaling Study Reproducibility Guide (FineWeb)

This guide reproduces the full scaling study represented in the Flywheel graph.

## Graph Roots and Node Mapping
- Original HORN root graph node: `f3c1b9cf-5904-57e5-abc7-ad8b91789f89`
- Scaling study root node: `14c6d442-4846-5f3f-9f26-bf878ff563bf`
- Model-size branch empirical node: `72f74c83-265e-5bd3-bea7-d10b5ed72d12`
- Token-horizon branch empirical node: `f8f939dd-096f-57f1-8a2c-bd507ecb1b08`
- Branch synthesis insight node: `502fe30b-fefc-5ea0-a6b5-ba448941cabf`
- Joint branch empirical node: `797f2040-c59f-518c-b492-7c133f3c7a36`
- Final insight node: `8a3d53c5-c68c-5c4b-8a41-58a4c52f6a0c`

Topology intent:
1. Run model-size and token-horizon branches independently.
2. Use branch results to select a combined config.
3. Run the joint experiment.
4. Produce final comparison and plots.

## Environment
- Python: 3.10+
- Required packages: `torch`, `numpy`, `matplotlib`, `datasets`
- GPU recommended (script supports CPU fallback)

Install:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch numpy matplotlib datasets
```

## Dataset
All branch scripts are configured for FineWeb by default:
- `DATASET=fineweb`
- `DATA_PATH=/workspace/data/fineweb_sample.txt`
- `FINEWEB_TARGET_CHARS=8000000`
- `FINEWEB_MAX_DOCS=40000`

The benchmark script is configured to fail if FineWeb loading fails (no tinyshakespeare fallback).

## Reproduction Steps
Assume repo/workspace root at `/workspace` containing these files.

1. Model-size branch
```bash
cd /workspace
bash scaling_plans/run_model_size_branch.sh
```
Expected summary:
- `/workspace/scaling/model_size/model_branch_summary.json` (or your selected `OUT_ROOT` mirror)

2. Token-horizon branch
```bash
cd /workspace
bash scaling_plans/run_token_horizon_branch.sh
```
Expected summary:
- `/workspace/scaling/token_horizon/horizon_branch_summary.json` (or your selected `ROOT_OUTDIR` mirror)

3. Build joint plan from branch outputs
```bash
cd /workspace
python3 scaling_plans/prepare_joint_plan.py \
  --model-branch-dir /workspace/scaling/model_size \
  --horizon-branch-dir /workspace/scaling/token_horizon \
  --out-json /workspace/scaling/joint/joint_plan.json \
  --out-md /workspace/scaling/joint/joint_plan.md \
  --steps 600 \
  --eval-interval 100 \
  --eval-iters 30
```

4. Joint run
```bash
cd /workspace
bash scaling_plans/run_joint_scaling.sh \
  /workspace/scaling/joint/joint_plan.json \
  /workspace/horn_nanochat_benchmark.py \
  /workspace/scaling/joint
```
Expected summary:
- `/workspace/scaling/joint/joint_branch_summary.json`

5. Plot generation
```bash
cd /workspace
python3 scaling_runs/plots/make_loss_plots.py
```
Expected files:
- `scaling_runs/plots/model_size_val_curves.png`
- `scaling_runs/plots/horizon_final_loss_bars.png`
- `scaling_runs/plots/joint_val_curves.png`
- `scaling_runs/plots/loss_plot_metrics.json`

## Verification Gates
- Every branch summary must report `dataset=fineweb`.
- No summary may report `tinyshakespeare` dataset source.
- Final metric for joint run is `summary.delta_final_vs_baseline` from `joint_branch_summary.json`.

## Packaged Repro Artifacts
See:
- `scaling_runs/repro/replication_manifest.json`
- `scaling_runs/repro/checksums.sha256`
- `scaling_runs/repro/component_bundles.sha256`
- `scaling_runs/repro/bundles/*.tar.gz`
