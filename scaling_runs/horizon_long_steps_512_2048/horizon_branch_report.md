# token_horizon_long_steps_512_2048 Summary

- Dataset verification passed: `True`
- Best horizon by horn absolute loss: `horizon_512`
- Best horizon by horn-vs-baseline delta: `horizon_2048`

| horizon | block | batch | baseline | horn | delta (horn-baseline) | rel_impr_pct |
|---|---:|---:|---:|---:|---:|---:|
| horizon_512 | 512 | 8 | 2.3421 | 2.3771 | +0.0350 | -1.494% |
| horizon_2048 | 2048 | 2 | 2.6045 | 2.6029 | -0.0015 | +0.059% |

## Verification Evidence
- horizon_512: dataset=fineweb dataset_source=cache:/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt data_path=/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt
- horizon_2048: dataset=fineweb dataset_source=cache:/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt data_path=/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt
