my plan for now is: apply the Hausdorff dimension procedure (https://github.com/umutsimsekli/Hausdorff-Dimension-and-Generalization) to a DLN, then integrate the llc estimation (https://github.com/timaeus-research/devinterp), and I want to analyze both the trajectory of omega over training and the computation of LLC

# LLC, tail index and generalisation

Two experiments live here:

- Deep linear network: `src/train.py`, `src/estimators.py` and `tests/test_estimators.py`. Separate from everything below.
- VGG on CIFAR-10 over training: `src/cnn/` and `scripts/`, described below.

The VGG pipeline makes a figure in the style of Figure 1 of AW for a VGG instead of ResNet18, with the tail index added. Rows are one-factor sweeps of SGD (learning rate, batch size, momentum) and columns are LLC, tail index, test accuracy and training loss, all against the SGD step. The observation under test, not assumed, is that stronger implicit regularisation goes with lower LLC and higher test accuracy even when the training loss reaches about 0.

Papers:

- HD: Simsekli et al. 2020, Hausdorff Dimension, Heavy Tails, and Generalization in Neural Networks ([code](https://github.com/umutsimsekli/Hausdorff-Dimension-and-Generalization), MIT). The original scripts are kept for reference in `src/cnn/hd_reference/`; nothing imports them.
- AW: Lau et al., The Local Learning Coefficient: A Singularity-Aware Complexity Measure.

SGD trains the models. SGLD is only used to estimate the LLC at saved weights, and the tail index does not use SGLD.

## Running it

Python 3.12, managed by uv (`pyproject.toml` requires `>=3.12`, so 3.11 is not supported by this lock). `--device` takes `cuda`, `mps` or `cpu` and defaults to the first one available. All values of the experiment are in `configs/sweep.yaml`.

```bash
uv run pytest                                  # add -m "not slow" to skip the smoke run
uv run python scripts/run_sweep.py --smoke     # 2000 images, 300 steps, tiny SGLD chain, writes results/smoke/

# 1. See the cost. Runs, SGD steps, SGLD epochs and nothing else.
uv run python scripts/run_sweep.py --dry-run

# 2. Train the baseline, without LLC, because the SGLD setting does not exist yet.
uv run python scripts/run_one.py --factor lr --value 0.01 --seed 0 --train-only

# 3. Calibrate SGLD on its final weights. Writes configs/sgld.yaml and results/calibration/loss_traces.png. Check the traces.
uv run python src/cnn/calibrate_sgld.py --run-dir results/runs/VGG4_lr0.01_bs128_m0_seed0

# 4. See the cost again with the chosen chain, then run everything. Stop and rerun at any time.
uv run python scripts/run_sweep.py --dry-run
uv run python scripts/run_sweep.py

# 5. Figures and report
uv run python scripts/plot_training_curves.py   # --alpha-variant to change the alpha column
uv run python scripts/analyse.py
```

Each run can also be launched alone. `--list` prints one line of arguments per distinct run, with the shared baseline once. After step 3, for a job array of 27:

```bash
uv run python scripts/run_one.py $(uv run python scripts/run_sweep.py --list | sed -n "${SLURM_ARRAY_TASK_ID}p") --device cuda
```

Use the lines from `--list` rather than every factor and value: naming the baseline twice in parallel would train it twice into one folder.

Each run has two stages:

1. **Training** writes the weights at each checkpoint and the tail index computed there, then `train.json` last. A finished training stage is skipped; an unfinished one starts again from step 0.
2. **LLC** estimates each checkpoint from the saved weights and rewrites the CSV after each one, so a crash loses at most one estimate. This is what lets the baseline be calibrated on without retraining it, and it means SGLD cannot touch the training RNG.

A resume is refused when the measurement settings or the SGLD setting differ from what the run folder recorded, because the values would not be comparable.

Outputs in `results/`, none of them committed:

- `timeseries/<run_id>.csv`: one row per logged step. NaN where not measured.
- `runs/<run_id>/`: `weights/step_*.pt`, `state_dict.pt`, `train.json`, `tail_details.json` (per-layer alphas, lag-1 and zero fractions, and the mean |lag-1|), `settings.json` and `sgld.json`. About 70 MB per VGG4 run.
- `figures/fig1_style.png`, `figures/alpha_variants.png` and `report.md`.
- `sweep_resolved.yaml`: the settings the figures and the report read.

CSV columns that need a definition:

| Column | Meaning |
|---|---|
| `train_loss` | mean of the minibatch losses since the last logged row |
| `llc_flag` | `ok`, `negative`, `left_basin` (a chain's mean loss reached ln 10) or `diverged` (no value) |
| `beta_*_max`, `beta_*_median` | max and median over layers of alpha_l. VGG4 has 2 weight tensors, so the median is their mean |
| `lag1` | lag-1 autocorrelation of the increments, signed mean over layers and projections. The flag uses the mean of the absolute values, which is in `tail_details.json` |
| `alpha_flag` | 1 if mean \|lag-1\| > 0.1 or zero fraction > 0.05, otherwise 0 |

How unstable points are treated: unstable LLC values (`negative`, `left_basin`) stay in the CSV, are drawn hollow for each seed, and stay out of the mean line and band. `diverged` has no value to draw. Flagged alpha values are diagnostics, not failures, so they stay in the mean and are also drawn hollow. Panels are scaled to the means and bands, so a far-off hollow point can fall outside its panel. The figure's title area and the report give the counts.

In both figures, darker means stronger implicit regularisation, which for batch size means smaller.

The report calls two settings "apart" only when their 95% t intervals do not overlap, with the same rule for every factor and metric, and gives no p-values. "Training loss about 0" is taken as below 0.01, a threshold set before any data were seen.

## How the tail index is computed

The reference is `estimator_vector_projected` in HD's `alpha_estimator_vgg.py`, and `tests/test_cnn.py` checks the rolling buffer against a port of it on full stored iterates. For each weight tensor (biases skipped, as in HD), at each checkpoint:

1. Take the last N+1 = 361 values on each of 10 fixed unit Gaussian directions from the rolling buffer.
2. Apply `alpha_estimator` (Mohammadi et al. 2014, Cor. 2.4, from `src/utils.py`) to each projected series and take the median over block sizes `ms`.
3. Take the median over projections to get alpha_l.

Then report the maximum over layers (HD Sec. 4) and the median over layers, because the maximum of noisy estimates is biased upward.

Two variants are always reported:

- **Iterates**, as in HD: centre over the 361 values in the window, then keep the last 360.
- **Increments**: the centred first differences of the 361 values.

Strongly negative lag-1 pushes alpha above 2, and at the extreme, as with i.i.d. positions around a fixed point where block sums telescope, past infinity to negative values. All of these stay in the data, and the report counts values outside (0, 2]. The iterate variant reads about 1 on slowly moving paths. Momentum makes increments positively autocorrelated by construction, which also trips the flag: in the smoke run, momentum 0.5 gave lag-1 of about +0.35. The momentum row should be read with `alpha_variants.png`.

## Deviations from HD

| Topic | HD | Here | Why |
|---|---|---|---|
| Increment variant | iterates only | iterates and increments | HD's model makes the increments i.i.d. stable, not the iterates |
| When | final epoch of a run | rolling window at each of 12 checkpoints | to follow alpha over training |
| Window | table by iterate count, `KeyError` otherwise | N = 360 for every batch size and checkpoint (180 in smoke mode) | same sample size everywhere |
| Block sizes | vary with the count | always `(2,3,6,9,18,36)` | same `ms` for every run |
| Directions | fresh for every estimate, from `np.random`, float32 | fixed per run, from a dedicated `torch.Generator`, float64 on the CPU | never touches the training RNG, float32 would round the increments away, and MPS has no float64 |
| Storage | one full `state_dict` per step | 10 scalars per layer per step, for the last 361 steps | projection is linear, so both variants follow from the scalars |
| Batches | no `drop_last` | `drop_last=True` | every step has the same batch size |
| Length | 81 epochs in code, 100 in the paper | 15000 steps | as AW; batch-size runs see different numbers of epochs |
| Momentum | 0.9 in `train_vgg.py` | 0 at baseline, swept | one factor at a time |
| Depths | VGG4 to VGG19 | VGG4 by default, VGG6 with `--arch` | scope; VGG4 has only 2 weight layers, so beta is a maximum over 2 values |

Matched exactly: `with_bias=True`, the normalisation constants of `train_vgg.py`, no augmentation, batchnorm or dropout, cross-entropy, torch's default initialisation.

## Deviations from AW

| Topic | AW | Here |
|---|---|---|
| Model | ResNet18 | HD's VGG4 |
| Seeds | 5 | 3 |
| Learning rate grid | 0.005 to 0.2 | 0.001 to 0.01 |
| SGLD minibatch | 2048 | 512 by default, set by calibration |
| Calibration | once per set of runs | once for the whole experiment, on the baseline's final weights, a stricter version of the same rule. Early checkpoints are far from a minimum, so unstable estimates there are expected |
| Negative LLC | removed by hand | kept, drawn hollow and counted |
| MALA acceptance check | recommended for tuning the step size | not used; selection follows the rule below |
| Added | | tail index column |

The same as AW: SGD steps on the x axis, SGLD samples the plain cross-entropy, nbeta = n / ln n, burn-in is 90% of the chain, and L(w*) is computed on the full training set.

## Using devinterp 1.3.2

Checked against the installed source; each point differs from a plain reading of the docs:

- The default L(w*) averages 2 random minibatches, a noisy offset once multiplied by nbeta (about 4621). We pass `init_loss` computed on all images, as in AW's Algorithm 1.
- The default nbeta is `batch_size / ln(batch_size)`, which is not AW's value. We pass n / ln n.
- The sampler has no generator argument and calls `torch.manual_seed` on the global RNG for each chain. We pass one seed per chain, derived from the run's seed and the step, inside `torch.random.fork_rng`.
- The sampler always calls `model.train()`, even after our `model.eval()`. This has no effect here because VGG has no batchnorm or dropout.
- Draws exist only after burn-in, so calibration samples with no burn-in and discards the same 90% itself.

On GPUs, seeds fix the run up to the nondeterminism of the backend; each run is only trained once, so nothing depends on repeating it bit for bit.

## Choosing the SGLD setting

`calibrate_sgld.py` scans step size and localisation on the baseline's final weights. A setting passes when, for all chains:

- nothing diverged;
- the LLC is positive;
- the mean loss stays below ln 10, the loss of a uniform guess;
- between the last two windows of 10% of the chain, the mean loss moves by less than 5% of its rise above L(w*).

It takes the first that passes, going through localisations from small to large (AW H.3: large values drown the likelihood), then steps from small to large. If none passes it writes nothing and exits with an error. The 5% threshold and the tie-break are choices made here, so check them against `loss_traces.png`. `run_sweep.py` and `run_one.py` refuse to start without `configs/sgld.yaml`, except for `--train-only`.

The LLC is sanity-checked in `tests/test_cnn.py`: binary logistic regression with p = 5 parameters at the optimum must give p/2 = 2.5, within 0.5.

## Overnight run

`configs/overnight.yaml` is the cut-down version for one night on an M1 Pro, about 6 to 7 hours. It departs from the full design in four ways: 1 seed instead of 3, so there are no CI bands; 6 checkpoints instead of 12; no batch size 512; and an SGLD minibatch of 256 instead of 512. Pass `--config configs/overnight.yaml` to `run_sweep.py`. Figures and the report can be made from whatever has finished.

## Smoke mode

2000 images, 300 steps, 3 checkpoints, a chain of 20 SGLD steps, 2 values per factor and 1 seed. 300 steps cannot fill the window of 361, so smoke mode uses 181 (N = 180). The first checkpoint, at step 100, has fewer than 181 iterates, so its alpha is NaN. The smoke chain is not calibrated, so its LLC values mean nothing; it only exercises the code.
