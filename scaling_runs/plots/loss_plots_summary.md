# FineWeb Loss Plot Summary

Generated: 2026-04-01

## Files
- model_size_val_curves.png
- horizon_final_loss_bars.png
- joint_val_curves.png
- loss_plot_metrics.json

## Key outcomes
- Model-size branch (delta = horn - baseline):
  - small_fixed_horizon: -0.002589
  - medium_fixed_horizon: +0.025055
  - large_fixed_horizon: -0.014882
- Token-horizon branch:
  - horizon_128: +0.000575
  - horizon_256: +0.019618
  - horizon_384: -0.000759
- Joint branch (selected config):
  - baseline mean final val loss: 2.166862
  - horn mean final val loss: 2.178075
  - delta: +0.011213
  - winner: baseline
