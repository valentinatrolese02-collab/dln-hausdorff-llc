import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

# N iterates and block sizes m from HD's `resolutions` table. Every m must divide N.
ALPHA_WINDOW = 324
ALPHA_MS = (2, 3, 6, 9, 18, 36)


def estimate_llc(model, x, epsilon=1e-6, gamma=1.0, num_steps=1000, burn_in=0.9, batch_size=128, generator=None):
    """SGLD estimate of the local learning coefficient at the model's current weights.

    The random generator lives on the CPU whatever the device of the model, so results do not
    depend on the device up to floating point error.

    Port of LLC's run_sgld, with lambdahat = n * beta * (mean sampled loss - loss at w*).
    Targets are the model's own outputs, as in expt_dln_saddle_dynamics.py, so the loss
    at w* is zero and the model is realisable at the point being measured. The first
    burn_in fraction of samples is discarded (Lau et al., Appendix H.2 and J.1).
    """
    n = len(x)
    beta = 1 / math.log(n)
    sampler = copy.deepcopy(model)
    w_star = [p.detach().clone() for p in sampler.parameters()]
    with torch.no_grad():
        y = model(x)

    loss_trace = []
    while len(loss_trace) < num_steps:
        perm = torch.randperm(n, generator=generator).to(x.device)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            loss = F.mse_loss(sampler(x[idx]), y[idx])
            loss_trace.append(loss.item())

            sampler.zero_grad()
            (n * beta * loss).backward()
            with torch.no_grad():
                for p, p_star in zip(sampler.parameters(), w_star):
                    drift = p.grad + gamma * (p - p_star)
                    noise = torch.randn(p.shape, generator=generator).to(p.device)
                    p += -epsilon / 2 * drift + math.sqrt(epsilon) * noise

            if len(loss_trace) >= num_steps:
                break

    return n * beta * float(np.mean(loss_trace[int(burn_in * num_steps) :]))


def true_dln_learning_coefficient(rank, widths):
    """Theoretical learning coefficient of a DLN (Aoyagi 2023), ported from LLC's dln.py.

    widths is the full [input, hidden..., output] list used by DLN.
    """
    M_list = np.array(widths) - rank
    M_subset = _search_subset(M_list)
    M_sum = np.sum(M_subset)
    ell = len(M_subset) - 1
    M = np.ceil(M_sum / ell)
    a = M_sum - (M - 1) * ell

    term1 = (-(rank**2) + rank * (widths[0] + widths[-1])) / 2
    term2 = a * (ell - a) / (4 * ell)
    term3 = -ell * (ell - 1) / 4 * (M_sum / ell) ** 2
    term4 = sum(M_subset[i] * M_subset[j] for i in range(ell + 1) for j in range(i + 1, ell + 1)) / 2
    return float(term1 + term2 + term3 + term4)


def _search_subset(M_list):
    """The unique prefix of the sorted M_list satisfying Aoyagi's conditions."""
    M_sorted = np.sort(M_list)
    for size in range(1, len(M_sorted) + 1):
        subset, complement = M_sorted[:size], M_sorted[size:]
        ell = size - 1
        if len(complement) and not subset.max() < complement.min():
            continue
        if not subset.sum() >= ell * subset.max():
            continue
        if len(complement) and not subset.sum() < ell * complement.min():
            continue
        return subset
    raise RuntimeError("No subset found")


def alpha_estimator(m, X):
    """Tail index estimator of Mohammadi et al. 2014, Corollary 2.4. X is N by d, ported from HD."""
    N = len(X)
    n = N // m  # m must divide N
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff


def alpha_window(num_iterates, block=max(ALPHA_MS)):
    """Largest multiple N of block with N+1 <= num_iterates, so that every m in ALPHA_MS divides N."""
    N = (num_iterates - 1) // block * block
    if N == 0:
        raise ValueError(f"need at least {block + 1} iterates, got {num_iterates}")
    return N


def random_directions(dim, num_projections=10, generator=None, dtype=None):
    """num_projections by dim matrix whose rows are Gaussian directions of unit norm."""
    directions = torch.stack([torch.randn(dim, generator=generator, dtype=dtype) for _ in range(num_projections)])
    return directions / directions.norm(dim=1, keepdim=True)


def projected_alpha(series, ms=ALPHA_MS):
    """One alpha per projection, as in the loop of HD's estimator_vector_projected.

    series is N by K, one column per projection, and every m must divide N.
    Returns the K medians over ms.
    """
    return np.array([np.median([alpha_estimator(m, column.unsqueeze(1)).item() for m in ms]) for column in series.T])


def lag1_autocorrelation(series):
    """Lag-1 autocorrelation of each column of series (T by K)."""
    return np.array([torch.corrcoef(torch.stack([column[1:], column[:-1]]))[0, 1].item() for column in series.T])


def estimate_alpha(iterates, ms=ALPHA_MS, num_projections=10, generator=None):
    """HD's estimator_vector_projected applied to increments. iterates is N+1 by d, returns (median, max).

    HD's code passes the iterates themselves, but the estimator assumes i.i.d. samples and
    reads about 1 on any slowly moving path. Under HD's frozen-coefficient Levy model (Sec. 4)
    it is the increments of the trajectory that are i.i.d. stable.
    """
    increments = iterates[1:] - iterates[:-1]
    increments = increments - increments.mean(dim=0, keepdim=True)
    directions = random_directions(increments.shape[1], num_projections, generator).to(increments.device)
    alphas = projected_alpha(increments @ directions.T, ms)
    return float(np.median(alphas)), float(np.max(alphas))


def increment_diagnostics(iterates, generator=None):
    """Checks that the increments suit estimate_alpha, returns (lag-1 autocorrelation, zero fraction).

    Strongly negative autocorrelation means sums of increments cancel, which inflates alpha. A high
    zero fraction means the updates are below floating point resolution, so alpha measures rounding.
    """
    increments = iterates[1:] - iterates[:-1]
    direction = torch.randn(increments.shape[1], generator=generator).to(increments.device)
    lag1 = lag1_autocorrelation((increments @ direction).unsqueeze(1))[0].item()
    return lag1, (increments == 0).float().mean().item()
