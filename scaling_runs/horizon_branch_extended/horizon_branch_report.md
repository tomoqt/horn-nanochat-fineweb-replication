# token_horizon_scaling_horn_nanochat_extended Summary

- Dataset verification passed: `True`
- Best horizon by horn absolute loss: `horizon_512`
- Best horizon by horn-vs-baseline delta: `horizon_512`

| horizon | block | batch | baseline | horn | delta (horn-baseline) | rel_impr_pct |
|---|---:|---:|---:|---:|---:|---:|
| horizon_512 | 512 | 8 | 2.6081 | 2.6030 | -0.0051 | +0.197% |
| horizon_768 | 768 | 5 | 2.6231 | 2.6211 | -0.0020 | +0.075% |
| horizon_1024 | 1024 | 4 | 2.6053 | 2.6054 | +0.0001 | -0.005% |
| horizon_1536 | 1536 | 2 | 2.6337 | 2.6333 | -0.0004 | +0.015% |
| horizon_2048 | 2048 | 2 | 2.6087 | 2.6091 | +0.0004 | -0.015% |

## Verification Evidence
- horizon_512: dataset=fineweb dataset_source=hf:HuggingFaceFW/fineweb/sample-10BT data_path=/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt
- horizon_768: dataset=fineweb dataset_source=cache:/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt data_path=/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt
- horizon_1024: dataset=fineweb dataset_source=cache:/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt data_path=/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt
- horizon_1536: dataset=fineweb dataset_source=cache:/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt data_path=/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt
- horizon_2048: dataset=fineweb dataset_source=cache:/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt data_path=/Users/tensorqt/Downloads/paper replication/data/fineweb_sample_8m.txt
