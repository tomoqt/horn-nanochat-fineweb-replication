# Horizon Branch Report

Dataset: FineWeb
- dataset: `fineweb`
- data_path: `/workspace/data/fineweb_sample.txt`
- verification: `passed`
- CUDA: `torch.cuda.is_available()=True`
- lease_id: `36342377-9b2d-4892-a414-0fc4aaad0706`

## Results

| Horizon | Block | Batch | Baseline final val | HORN final val | Delta vs baseline | Relative improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| horizon_128 | 128 | 32 | 2.225658 | 2.226233 | +0.000575 | -0.026% |
| horizon_256 | 256 | 16 | 2.449693 | 2.469311 | +0.019618 | -0.801% |
| horizon_384 | 384 | 11 | 2.596056 | 2.595297 | -0.000759 | +0.029% |

## Notes

- `horizon_128` loaded directly from `hf:HuggingFaceFW/fineweb/sample-10BT`.
- `horizon_256` and `horizon_384` loaded from the local FineWeb cache at `/workspace/data/fineweb_sample.txt`.
- Best HORN horizon by mean final validation loss: `horizon_128` (`block_size=128`).
