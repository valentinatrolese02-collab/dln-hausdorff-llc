"""Tail index over training: a rolling buffer of weight projections, and the estimates computed from it.

The reference is estimator_vector_projected in HD's alpha_estimator_vgg.py, kept in hd_reference/alpha.py.
Deviations from it are listed in the README.
"""

from collections import deque

import numpy as np
import torch

from utils import alpha_estimator

ALPHA_MS = (2, 3, 6, 9, 18, 36)
NUM_PROJECTIONS = 10
MAX_MEAN_ABS_LAG1 = 0.1
MAX_ZERO_FRACTION = 0.05


def random_directions(dim, num_projections, generator):
    """num_projections by dim matrix whose rows are Gaussian directions of unit norm, in float64."""
    directions = torch.stack([torch.randn(dim, generator=generator, dtype=torch.float64) for _ in range(num_projections)])
    return directions / directions.norm(dim=1, keepdim=True)


class ProjectionBuffer:
    """The projections of the last `window` iterates of each weight tensor (ndim >= 2, biases skipped as in HD).

    window = N + 1 iterates give N updates, so N zero fractions are kept. Projections run on the CPU in
    float64 because increments are far smaller than the iterates, a float32 dot product over a million
    weights would round them away, and MPS has no float64.
    """

    def __init__(self, model, generator, window, num_projections=NUM_PROJECTIONS):
        self.weights = {name: p for name, p in model.named_parameters() if p.ndim >= 2}
        self.directions = {name: random_directions(p.numel(), num_projections, generator) for name, p in self.weights.items()}
        self.window = window
        self.iterates = {name: deque(maxlen=window) for name in self.weights}
        self.zero_fraction = deque(maxlen=window - 1)
        self.previous = self._flat_weights()

    def _flat_weights(self):
        return {name: p.detach().flatten().cpu().clone() for name, p in self.weights.items()}

    def push(self):
        """Call after every optimizer step."""
        current = self._flat_weights()
        for name, w in current.items():
            self.iterates[name].append(self.directions[name] @ w.double())
        self.zero_fraction.append([(current[name] == self.previous[name]).double().mean().item() for name in current])
        self.previous = current

    def is_full(self):
        return len(self.zero_fraction) == self.window - 1 and len(next(iter(self.iterates.values()))) == self.window

    def series(self):
        """Layer name to window by K projections, and the N by L zero fractions."""
        return {name: torch.stack(list(rows)) for name, rows in self.iterates.items()}, np.array(self.zero_fraction)


def iterate_series(series):
    """As HD: centre over all iterates in the window, then keep the last N rows."""
    return (series - series.mean(dim=0, keepdim=True))[1:]


def increment_series(series):
    """Centred first differences of the N + 1 iterates."""
    increments = series.diff(dim=0)
    return increments - increments.mean(dim=0, keepdim=True)


def projected_alpha(series, ms=ALPHA_MS):
    """series is N by K with one column per projection. Returns the K medians over block sizes."""
    return np.array([np.median([alpha_estimator(m, column.unsqueeze(1)).item() for m in ms]) for column in series.T])


def lag1_autocorrelation(series):
    """Lag-1 autocorrelation of each column of series (T by K)."""
    return np.array([torch.corrcoef(torch.stack([column[1:], column[:-1]]))[0, 1].item() for column in series.T])


def layer_summary(series, zero_fraction, ms=ALPHA_MS):
    """series is the window by K projections of one layer, zero_fraction its N per-step values."""
    increments = increment_series(series)
    lag1 = lag1_autocorrelation(increments)
    return {
        "alpha_iter": float(np.median(projected_alpha(iterate_series(series), ms))),
        "alpha_incr": float(np.median(projected_alpha(increments, ms))),
        "lag1": float(np.mean(lag1)),
        "abs_lag1": float(np.mean(np.abs(lag1))),
        "zero_fraction": float(np.mean(zero_fraction)),
    }


def _finite(values):
    values = np.array(values, dtype=float)
    return values[np.isfinite(values)]


def tail_index(layers, zero_fraction, ms=ALPHA_MS):
    """layers maps name to window by K projections, zero_fraction is N by L in the same order."""
    per_layer = {name: layer_summary(series, zero_fraction[:, i], ms) for i, (name, series) in enumerate(layers.items())}
    summary = {"per_layer": per_layer}
    for variant in ("iter", "incr"):
        # alpha is infinite on a layer that never moves; zero_fraction flags that checkpoint
        alphas = _finite([layer[f"alpha_{variant}"] for layer in per_layer.values()])
        summary[f"beta_{variant}_max"] = float(alphas.max()) if alphas.size else float("nan")
        summary[f"beta_{variant}_median"] = float(np.median(alphas)) if alphas.size else float("nan")
    summary["lag1"] = float(np.mean([layer["lag1"] for layer in per_layer.values()]))
    summary["mean_abs_lag1"] = float(np.mean([layer["abs_lag1"] for layer in per_layer.values()]))
    summary["zero_fraction"] = float(np.mean([layer["zero_fraction"] for layer in per_layer.values()]))
    # Written so that a NaN lag-1, from a series that never moves, also flags the checkpoint
    summary["alpha_flag"] = not (summary["mean_abs_lag1"] <= MAX_MEAN_ABS_LAG1 and summary["zero_fraction"] <= MAX_ZERO_FRACTION)
    return summary
