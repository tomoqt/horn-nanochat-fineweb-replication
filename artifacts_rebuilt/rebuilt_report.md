# Attention Sinks Rebuilt Artifact Report

- Model: `EleutherAI/pythia-70m-deduped`
- Sequence length: `128`
- Batch size: `8` across `6` batches
- Seed: `1337`

## Core Metrics
- `baseline`: loss=4.4066, sink_first4=0.1656, sink_last3=0.0043, delta_loss=+0.0000
- `shuffled`: loss=8.0509, sink_first4=0.1624, sink_last3=0.0044, delta_loss=+3.6443
- `reverse`: loss=7.1672, sink_first4=0.1621, sink_last3=0.0044, delta_loss=+2.7605
- `shift32`: loss=4.4923, sink_first4=0.1685, sink_last3=0.0044, delta_loss=+0.0857
- `mask_first4`: loss=32.8553, sink_first4=0.1102, sink_last3=0.0043, delta_loss=+28.4487
- `mask_random4`: loss=42.4244, sink_first4=0.1599, sink_last3=0.0044, delta_loss=+38.0178
- `swap_pos0`: loss=4.5387, sink_first4=0.1606, sink_last3=0.0043, delta_loss=+0.1321

## Files
- `attention_sinks_core_metrics.json`
- `hypothesis_contiguous_vs_shuffled.json`
- `rope_position_controls.json`
- `bias_reverse_shift_deltas.json`
- `masking_ablation_deltas.json`
- `source_position_profiles.png`
- `sink_first4_by_condition.png`
- `masking_delta_loss.png`
