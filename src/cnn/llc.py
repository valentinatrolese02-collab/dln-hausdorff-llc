"""LLC of trained weights: a thin wrapper around devinterp 1.3.2."""
import math
import warnings

import torch
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce
from torch.utils.data import DataLoader, TensorDataset

from cnn.data import evaluate

# calibrate_sgld.py's chain, also what --dry-run assumes before configs/sgld.yaml exists
SGLD_CHAIN_DEFAULTS = {"num_steps": 1000, "burn_in": 0.9, "num_chains": 2, "batch_size": 512}


def equivalent_epochs(sgld, n):
    """Compute cost of one LLC estimate over all chains, in passes over the training set."""
    return sgld["num_chains"] * sgld["num_steps"] * sgld["batch_size"] / n


def llc_flag(res):
    """ok, or why this estimate is unstable: diverged (no number), negative, or left_basin (a chain's mean loss reached ln 10, a uniform guess)."""
    if res.get("diverged") or not math.isfinite(res["llc/mean"]):
        return "diverged"
    if res["llc/mean"] < 0:
        return "negative"
    return "left_basin" if res["loss/trace"].mean(axis=1).max() >= math.log(10) else "ok"


def estimate_llc(model, x, y, sgld, seed, device):
    """sgld: lr, localization, num_steps, burn_in (fraction), num_chains, batch_size. seed: an int, or one per chain."""
    n = len(x)
    nbeta = n / math.log(n)  # AW: beta = 1/log n, n = dataset size (devinterp's default is B/log B)
    model = model.to(device).eval()
    init_loss = evaluate(model, x, y, device)[1]  # full-data L(w*), not devinterp's two-minibatch default
    loader = DataLoader(TensorDataset(x, y), batch_size=sgld["batch_size"], shuffle=True, drop_last=True)
    burnin = int(sgld["burn_in"] * sgld["num_steps"])
    with torch.random.fork_rng(), warnings.catch_warnings():  # the sampler reseeds the global RNG, and warns on every call
        warnings.simplefilter("ignore")
        try:
            res = estimate_learning_coeff_with_summary(
                model, loader, evaluate=evaluate_ce,
                optimizer_kwargs=dict(lr=sgld["lr"], localization=sgld["localization"], nbeta=nbeta),
                num_draws=sgld["num_steps"] - burnin, num_burnin_steps=burnin,
                num_chains=sgld["num_chains"], seed=seed, device=device, init_loss=init_loss, verbose=False,
            )
        except RuntimeError as error:  # devinterp raises when a chain reaches a NaN loss
            if "NaN detected" not in str(error):
                raise
            res = {"llc/mean": float("nan"), "llc/std": float("nan"), "diverged": True}
    return res | {"init_loss": init_loss, "nbeta": nbeta}