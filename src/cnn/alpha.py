"""Tail index estimates from the weight projections stored by train_vgg.py.

The reference is estimator_vector_projected in HD's alpha_estimator_vgg.py. Its table of iterate
counts is replaced by alpha_window and ALPHA_MS (see the README).
"""

import numpy as np
import torch

from estimators import ALPHA_MS, alpha_window, lag1_autocorrelation, projected_alpha

MAX_MEAN_ABS_LAG1 = 0.1
MAX_ZERO_FRACTION = 0.05


def iterate_series(series, N):
    """As HD: centre over all recorded iterates, then keep the last N rows."""
    return (series - series.mean(dim=0, keepdim=True))[-N:]


def increment_series(series, N):
    """Centred first differences of the last N+1 iterates."""
    increments = series[-(N + 1) :].diff(dim=0)
    return increments - increments.mean(dim=0, keepdim=True)


def layer_summary(series, zero_fraction, N=None, ms=ALPHA_MS):
    """series is the T by K projections of one layer, zero_fraction its T per-step values."""
    series = torch.as_tensor(series)
    N = N or alpha_window(len(series))
    increments = increment_series(series, N)
    alpha_iter = projected_alpha(iterate_series(series, N), ms)
    alpha_incr = projected_alpha(increments, ms)
    lag1 = lag1_autocorrelation(increments)
    return {
        "alpha_iter": float(np.median(alpha_iter)),
        "alpha_iter_proj_max": float(np.max(alpha_iter)),
        "alpha_incr": float(np.median(alpha_incr)),
        "alpha_incr_proj_max": float(np.max(alpha_incr)),
        "lag1": float(np.mean(lag1)),
        "abs_lag1": float(np.mean(np.abs(lag1))),
        "zero_fraction": float(np.mean(zero_fraction[-N:])),
    }


def _finite(values):
    values = np.array(values, dtype=float)
    return values[np.isfinite(values)]


def tail_index_summary(layers, zero_fraction, N=None, ms=ALPHA_MS):
    """layers maps name to T by K projections, zero_fraction is T by L in the same order."""
    per_layer = {name: layer_summary(series, zero_fraction[:, i], N, ms) for i, (name, series) in enumerate(layers.items())}
    summary = {"window": N or alpha_window(len(next(iter(layers.values())))), "per_layer": per_layer}
    for variant in ("iter", "incr"):
        # alpha is infinite on a layer that never moves; zero_fraction flags that run
        alphas = _finite([layer[f"alpha_{variant}"] for layer in per_layer.values()])
        summary[f"beta_{variant}_max"] = float(alphas.max()) if alphas.size else float("nan")
        summary[f"beta_{variant}_median"] = float(np.median(alphas)) if alphas.size else float("nan")
    summary["mean_lag1"] = float(np.mean([layer["lag1"] for layer in per_layer.values()]))
    summary["mean_abs_lag1"] = float(np.mean([layer["abs_lag1"] for layer in per_layer.values()]))
    summary["zero_fraction"] = float(np.mean([layer["zero_fraction"] for layer in per_layer.values()]))
    # Written so that a NaN lag-1, from a series that never moves, also flags the run
    summary["flagged"] = not (summary["mean_abs_lag1"] <= MAX_MEAN_ABS_LAG1 and summary["zero_fraction"] <= MAX_ZERO_FRACTION)
    return summary


def load_tail_index(path, N=None, ms=ALPHA_MS):
    """tail_index_summary of a projections.npz written by train_vgg.py."""
    with np.load(path) as arrays:
        layers = {key.removeprefix("layer/"): arrays[key] for key in arrays.files if key.startswith("layer/")}
        return tail_index_summary(layers, arrays["zero_fraction"], N, ms)
