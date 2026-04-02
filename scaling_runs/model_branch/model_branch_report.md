# Branch A FineWeb Rerun

Dataset: `fineweb`
Dataset source: `cache:/workspace/data/fineweb_sample.txt`
FineWeb target chars: `8000000`
FineWeb max docs: `40000`

## Best HORN Config
- Experiment: `medium_fixed_horizon`
- Mean final validation loss: `2.3079`
- Seeds: `1337, 2027`

## Aggregate Table
| experiment | baseline final val loss | horn final val loss | delta horn - baseline |
|---|---:|---:|---:|
| small_fixed_horizon | 2.4779 | 2.4753 | -0.0026 |
| medium_fixed_horizon | 2.2828 | 2.3079 | +0.0251 |
| large_fixed_horizon | 2.4067 | 2.3918 | -0.0149 |

## Per-Run Final Validation Loss
| experiment | variant | seed | final val loss |
|---|---|---:|---:|
| large_fixed_horizon | baseline | 1337 | 2.4458 |
| large_fixed_horizon | baseline | 2027 | 2.3676 |
| large_fixed_horizon | horn | 1337 | 2.4197 |
| large_fixed_horizon | horn | 2027 | 2.3639 |
| medium_fixed_horizon | baseline | 1337 | 2.2807 |
| medium_fixed_horizon | baseline | 2027 | 2.2849 |
| medium_fixed_horizon | horn | 1337 | 2.3222 |
| medium_fixed_horizon | horn | 2027 | 2.2935 |
| small_fixed_horizon | baseline | 1337 | 2.4672 |
| small_fixed_horizon | baseline | 2027 | 2.4886 |
| small_fixed_horizon | horn | 1337 | 2.4626 |
| small_fixed_horizon | horn | 2027 | 2.4880 |

## Result
The FineWeb rerun completed successfully and replaced the earlier Tiny Shakespeare fallback in the Branch A outputs.
