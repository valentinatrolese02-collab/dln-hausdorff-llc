my plan for now is: apply the Hausdorff dimension procedure (https://github.com/umutsimsekli/Hausdorff-Dimension-and-Generalization) to a DLN, then integrate the llc estimation (https://github.com/timaeus-research/devinterp), and I want to analyze both the trajectory of omega over training and the computation of LLC

# LLC, tail index and generalisation

Two experiments share `src/estimators.py` and `src/models.py`:

- Deep linear network: `src/train.py`.
- VGG on CIFAR-10: `src/cnn/` and `scripts/`, described below.

The VGG pipeline fixes one small VGG, varies one training regulariser at a time, and for every run measures the generalisation gap, the local learning coefficient (LLC) and the tail index beta. The hypothesis under test, not assumed, is that stronger regularisation gives higher test accuracy, lower LLC and lower beta.

Papers:

- HD: Simsekli et al. 2020, Hausdorff Dimension, Heavy Tails, and Generalization in Neural Networks ([code](https://github.com/umutsimsekli/Hausdorff-Dimension-and-Generalization), MIT).
- AW: Lau et al., The Local Learning Coefficient: A Singularity-Aware Complexity Measure.

SGD trains the models. SGLD is only used to estimate the LLC of the trained weights, and the tail index does not use SGLD.

## Running it

Python 3.12, managed by uv. A CUDA GPU is expected for anything beyond the smoke run. `--device` defaults to `cuda` when available, otherwise `cpu`.

```bash
uv run pytest                                  # includes the smoke run once data/ exists
uv run python scripts/run_sweep.py --smoke     # 2 epochs on 2000 images, tiny SGLD chain, writes results/smoke/

# 1. Reference run. This directory name is the sweep's baseline with seed 0, so the sweep reuses it.
uv run python src/cnn/train_vgg.py --seed 0 --out-dir results/runs/VGG4_lr0.01_bs128_m0_l20_seed0

# 2. Calibrate SGLD. Writes configs/sgld.yaml and results/calibration/loss_traces.png. Check the traces.
uv run python src/cnn/calibrate_sgld.py --run-dir results/runs/VGG4_lr0.01_bs128_m0_l20_seed0

# 3. Sweep: 10 configs x 3 seeds = 30 runs. Stop and rerun at any time; finished stages are skipped.
uv run python scripts/run_sweep.py

# 4. Figures and results/report.md
uv run python scripts/analyse.py
```

Compute, in passes over the training set: the default calibration is 12 settings x 61 = 737, and the sweep is 1500 training epochs + 1843 SGLD epochs, using the 3000-step chain from the calibration defaults. Both scripts print this before starting.

`run_sweep.py` refuses to start without `configs/sgld.yaml`, and refuses to resume when finished runs used other SGLD settings, because LLC values are only comparable under one sampler.

Outputs in `results/`, none of them committed:

- `summary.csv`: one row per run, rebuilt from the per-run JSONs so that a resume cannot duplicate a row.
- `runs/<config>_seed<seed>/`: `train.json`, `state_dict.pt` for re-estimating the LLC without retraining, `projections.npz`, `sgld_loss_trace.npy` and `run.json`.
- `report.md` and `figures/`.

Spearman correlations in the report, all two-sided:

| n | What is reported |
|---|---|
| below 8 | rho only, marked "too few runs", no p-value or verdict |
| 8 to 20 | exact permutation test over all n! pairings, counted by dynamic programming because 20! cannot be listed |
| above 20 | t-approximation |

A p-value of 0 is never printed: it appears as `p < 1/n!`, or as `p <` the smallest positive double above n = 20, where the t tail underflows.

## How the tail index is computed

The reference is `estimator_vector_projected` in HD's `alpha_estimator_vgg.py`, and `tests/test_cnn.py` checks our pipeline against a port of it. For each weight tensor (biases skipped, as in HD):

1. Project onto 10 unit Gaussian directions.
2. Apply `alpha_estimator` (Mohammadi et al. 2014, Cor. 2.4) to each projected series and take the median over block sizes `ms`.
3. Take the median over projections to get alpha_l. The maximum over projections, which HD also returns, is kept per layer in `run.json`.

Then beta is the maximum over layers (HD Sec. 4). The median over layers is also reported, because the maximum of noisy estimates is biased upward.

Two variants are always reported:

- **Iterates**, as in HD: centre over all recorded iterates of the final epoch, then keep the last N.
- **Increments**: the centred first differences of the last N+1 iterates.

A run is flagged when the mean |lag-1 autocorrelation| of the increments exceeds 0.1, or the fraction of exactly-zero weight updates exceeds 0.05. Strongly negative lag-1 inflates alpha, and the iterate variant reads about 1 on slowly moving paths, so neither variant should be read without these diagnostics. HD report estimates above 2 and treat them as an estimator artifact.

## Deviations from HD

| Topic | HD | Here | Why |
|---|---|---|---|
| Increment variant | iterates only | iterates and increments | HD's model makes the increments i.i.d. stable, not the iterates |
| Window | fixed table by iterate count, `KeyError` otherwise | N = largest multiple of 36 with N+1 <= recorded iterates | works for any count and uses more iterates |
| Block sizes | vary with the count, e.g. `[2,3,6,7,9,18,36]` at 782, `[2,4,7,14,28]` at 196 | always `(2,3,6,9,18,36)` | same `ms` for every run |
| Directions | fresh for every estimate, from `np.random`, float32 | fixed per run, from a dedicated `torch.Generator`, float64 | never touches the training RNG, and float32 would round the increments away |
| Storage | one full `state_dict` per step | 10 scalars per layer per step | projection is linear, so both variants follow from the scalars |
| Batches | no `drop_last`: 391 steps at batch 128 | `drop_last=True`: 390 steps | every step has the same batch size |
| Epochs | code 81, paper 100 | 50 | compute |
| Momentum | 0.9 in `train_vgg.py` | 0 at baseline, swept | one factor at a time |
| Depths | VGG4 to VGG19 | VGG4 by default, VGG6 allowed | scope; VGG4 has only 2 weight layers, so beta is a maximum over 2 values |

Window lengths under this rule, against HD's table:

| Batch size | Steps per epoch | N here | N in HD's `alpha_estimator_vgg.py` |
|---|---|---|---|
| 32 | 1562 | 1548 | 756 |
| 64 | 781 | 756 | 756 |
| 128 | 390 | 360 | 324 |
| 256 | 195 | 180 | 196 |

N varies with batch size in both, so beta across batch sizes also compares sample sizes. `load_tail_index` takes `N` and `ms`, so the stored projections can be re-analysed with a common window, such as N = 180, without retraining.

Matched exactly: `with_bias=True`, the normalisation constants of `train_vgg.py`, no augmentation, batchnorm or dropout, cross-entropy, torch's default initialisation.

## Deviations from AW

| Topic | AW | Here |
|---|---|---|
| Model | ResNet18 | HD's VGG4 |
| Seeds | 5 | 3 |
| Learning rate grid | 0.005 to 0.2 | 0.001 to 0.01 |
| SGLD minibatch | 2048 | 512 |
| Calibration | once per set of runs | once for the whole sweep, a stricter version of the same rule |
| When | along training, at checkpoints | final weights only |
| Negative LLC | removed by hand | kept and counted in the report |
| MALA acceptance check | recommended for tuning the step size | not used; selection follows the rule below |

The same as AW: L2 is `l2 * sum of squared parameters` in the training loss only, SGLD samples the plain cross-entropy, nbeta = n / ln n, burn-in is 90% of the chain, and L(w*) is computed on the full training set.

## Using devinterp 1.3.2

Checked against the installed source; each point differs from a plain reading of the docs:

- `estimate_learning_coeff` returns only the mean, so `src/cnn/llc.py` calls `estimate_learning_coeff_with_summary`, which it wraps, for the per-chain values and traces.
- The default L(w*) averages 2 random minibatches, a noisy offset once multiplied by nbeta (about 4621). We pass `init_loss` computed on all images, as in AW's Algorithm 1.
- The default nbeta is `batch_size / ln(batch_size)`, which is not AW's value. We pass n / ln n.
- The sampler has no generator argument and calls `torch.manual_seed` on the global RNG for each chain. We give each chain a seed inside `torch.random.fork_rng`, so SGLD has its own reproducible stream and the caller's RNG is restored.
- The sampler always calls `model.train()`, even after our `model.eval()`. This has no effect here because VGG has no batchnorm or dropout.
- Its `SGLD` class emits a `DeprecationWarning` pointing to `SGMCMC.sgld`, but stays the default sampler. It is used as is, and devinterp's warnings are silenced inside the call.
- Draws exist only after burn-in, so calibration samples with no burn-in and discards the same 90% itself. A test checks that this gives the same LLC as the sweep's call.

## Choosing the SGLD setting

`calibrate_sgld.py` scans step size and localisation on one reference run. A setting passes when, for all chains:

- nothing diverged;
- the LLC is positive;
- the mean loss stays below ln 10, the loss of a uniform guess;
- between the last two windows of 10% of the chain, the mean loss moves by less than 5% of its rise above L(w*).

Among those that pass it takes the smallest localisation (AW H.3: large values drown the likelihood), then the smallest step. If none passes it writes nothing and exits with an error. The 5% threshold and the tie-break are choices made here, so check them against `loss_traces.png`.

LLC sanity check: binary logistic regression with p = 5 parameters, n = 2000, at the optimum. Over 6 sampler seeds, with 4000 steps and 2 chains, the mean was 2.50 against the theoretical p/2 = 2.5, and the largest single error was 0.43. The test uses a tolerance of 0.5.
