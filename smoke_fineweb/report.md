# HORN vs Baseline NanoChat Benchmark

- Device: `mps`
- Variants: `baseline, horn, horn_no_momentum`
- Seeds: `[1337, 2027]`
- Steps per run: `2`

## Final Validation Loss (mean +/- std)
- `baseline`: 4.5558 +/- 0.0195 (delta vs baseline: +0.0000)
- `horn`: 4.5558 +/- 0.0195 (delta vs baseline: -0.0000)
- `horn_no_momentum`: 4.5558 +/- 0.0195 (delta vs baseline: -0.0000)

## Files
- `benchmark_summary.json`
- `vocab.json`
- `val_loss_curves.png`
- `final_val_bar.png`
- `horn_momentum_evolution.png`
