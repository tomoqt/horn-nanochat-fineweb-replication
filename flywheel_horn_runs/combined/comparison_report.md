# HORN vs Baseline Comparison (Flywheel V100)

- Model family: scaled nanochat-like causal decoder (4 layers, 4 heads, 128 dim)
- Dataset: chat-like reformatted Tiny Shakespeare (character-level)
- Train steps per run: 600
- Seeds: 1337, 2027, 3141
- Device: Tesla V100 32GB via Flywheel compute

## Mean Final Validation Loss (lower is better)
- baseline: 1.148522 +/- 0.009565 (delta vs baseline +0.000000, improvement +0.000%)
- horn: 1.134167 +/- 0.008557 (delta vs baseline -0.014355, improvement +1.250%)
- horn_no_momentum: 1.145220 +/- 0.007236 (delta vs baseline -0.003302, improvement +0.288%)

## Key Pairwise Deltas
- horn - baseline: -0.014355
- horn_no_momentum - baseline: -0.003302
- horn - horn_no_momentum: -0.011053

Interpretation: HORN with non-zero initial momentum (`m_init=0.5`) achieved the best final validation loss among tested variants; fixing momentum at 0 recovers only a smaller gain over baseline.
