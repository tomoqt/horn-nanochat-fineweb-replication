# HORN vs Baseline NanoChat Benchmark

- Device: `mps`
- Variants: `baseline, horn, horn_no_momentum`
- Seeds: `[1337, 2027]`
- Steps per run: `20`

## Final Validation Loss (mean +/- std)
- `baseline`: 2.7289 +/- 0.0057 (delta vs baseline: +0.0000)
- `horn`: 2.7289 +/- 0.0057 (delta vs baseline: +0.0001)
- `horn_no_momentum`: 2.7289 +/- 0.0057 (delta vs baseline: +0.0000)

## Files
- `benchmark_summary.json`
- `vocab.json`
- `val_loss_curves.png`
- `final_val_bar.png`
- `horn_momentum_evolution.png`
