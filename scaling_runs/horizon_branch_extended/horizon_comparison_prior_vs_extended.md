# Token-Horizon Comparison: Prior vs Extended

| sweep | horizon | block | baseline | horn | delta (horn-baseline) | rel_impr_pct |
|---|---|---:|---:|---:|---:|---:|
| prior_128_384 | horizon_128 | 128 | 2.2257 | 2.2262 | +0.0006 | -0.026% |
| prior_128_384 | horizon_256 | 256 | 2.4497 | 2.4693 | +0.0196 | -0.801% |
| prior_128_384 | horizon_384 | 384 | 2.5961 | 2.5953 | -0.0008 | +0.029% |
| extended_512_2048 | horizon_512 | 512 | 2.6081 | 2.6030 | -0.0051 | +0.197% |
| extended_512_2048 | horizon_768 | 768 | 2.6231 | 2.6211 | -0.0020 | +0.075% |
| extended_512_2048 | horizon_1024 | 1024 | 2.6053 | 2.6054 | +0.0001 | -0.005% |
| extended_512_2048 | horizon_1536 | 1536 | 2.6337 | 2.6333 | -0.0004 | +0.015% |
| extended_512_2048 | horizon_2048 | 2048 | 2.6087 | 2.6091 | +0.0004 | -0.015% |

- Prior best delta horizon: `horizon_384`
- Extended best delta horizon: `horizon_512`
