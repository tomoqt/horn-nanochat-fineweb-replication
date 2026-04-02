# Joint Branch C Report

Dataset: `fineweb`
Dataset source: `hf:HuggingFaceFW/fineweb/sample-10BT`
Data path: `/workspace/data/fineweb_sample.txt`
Verification: passed (`dataset == fineweb`, source not tinyshakespeare)

## Joint Config

- `n_layer=4`
- `n_head=4`
- `n_embd=128`
- `batch_size=32`
- `block_size=128`
- `steps=600`
- `horn_m_init=0.5`

## Results

| variant | final val loss mean | std | delta vs baseline |
|---|---:|---:|---:|
| baseline | 2.166862 | 0.007537 | 0.000000 |
| horn | 2.178075 | 0.006267 | +0.011213 |

The joint Horn configuration did not beat the baseline on FineWeb at this scale. The observed gap was +0.011213 final validation loss.

## Per-seed Final Losses

- baseline seed 1337: 2.159325
- baseline seed 2027: 2.174399
- horn seed 1337: 2.184342
- horn seed 2027: 2.171808

## Interpretation

The branch-level model-size and token-horizon winners were good enough to justify a joint test, but the combined setting did not compound into a win. That points to an interaction effect rather than a missing hyperparameter coincidence: the Horn update may be more sensitive to the joint configuration than to either scale dimension alone.
