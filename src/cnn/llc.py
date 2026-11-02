"""LLC of trained weights with the devinterp 1.3.2 SGLD sampler."""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce
from torch.utils.data import DataLoader, TensorDataset

from cnn.data import evaluate
from models import VGG


def load_sgld_config(path):
    return yaml.safe_load(Path(path).read_text())


def load_trained_model(run_dir):
    run_dir = Path(run_dir)
    arch = json.loads((run_dir / "train.json").read_text())["arch"]
    model = VGG(arch, with_bias=True)
    model.load_state_dict(torch.load(run_dir / "state_dict.pt", map_location="cpu"))
    return model


def equivalent_epochs(sgld, n):
    """Compute cost of one LLC estimate over all chains, in passes over the training set."""
    return sgld["num_chains"] * sgld["num_steps"] * sgld["batch_size"] / n


def estimate_llc(model, x, y, sgld, seed, device, keep_burnin=False, verbose=False):
    """SGLD estimate of the LLC at the weights of model, on CPU tensors x, y with plain cross-entropy.

    sgld has lr, localization, num_steps, burn_in (a fraction), num_chains and batch_size. keep_burnin
    samples the same chain but returns the whole loss trace, for calibration; the LLC does not change.
    """
    n = len(x)
    nbeta = n / math.log(n)
    device = torch.device(device)
    model = model.to(device).eval()
    # devinterp's default L(w*) averages two random minibatches, a noisy offset once multiplied by nbeta.
    # AW's Algorithm 1 uses the full data.
    init_loss = evaluate(model, x, y, device)[1]

    num_kept = sgld["num_steps"] - int(sgld["burn_in"] * sgld["num_steps"])
    num_burnin = 0 if keep_burnin else sgld["num_steps"] - num_kept
    num_draws = sgld["num_steps"] - num_burnin
    loader = DataLoader(TensorDataset(x, y), batch_size=sgld["batch_size"], shuffle=True, drop_last=True)
    chain_seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(sgld["num_chains"])]

    # The sampler has no generator argument: it reseeds the global RNG per chain and draws both the
    # noise and the minibatches from it. fork_rng confines that, which gives SGLD its own stream.
    with torch.random.fork_rng(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            results = estimate_learning_coeff_with_summary(
                model,
                loader,
                evaluate=evaluate_ce,
                optimizer_kwargs={"lr": sgld["lr"], "localization": sgld["localization"], "nbeta": nbeta},
                num_draws=num_draws,
                num_chains=sgld["num_chains"],
                num_burnin_steps=num_burnin,
                num_steps_bw_draws=1,
                init_loss=init_loss,
                seed=chain_seeds,
                device=device,
                verbose=verbose,
            )
        except RuntimeError as error:  # LLCEstimator raises when a chain reaches a NaN loss
            if "NaN detected" not in str(error):
                raise
            return _diverged(sgld, nbeta, init_loss, num_kept, num_draws)

    trace = results["loss/trace"].astype(np.float64)
    llc_chains = nbeta * (trace[:, -num_kept:].mean(axis=1) - init_loss)
    if not np.all(np.isfinite(llc_chains)):
        return _diverged(sgld, nbeta, init_loss, num_kept, num_draws, trace)
    return {
        "llc_mean": float(llc_chains.mean()),
        "llc_std": float(llc_chains.std(ddof=1)) if len(llc_chains) > 1 else 0.0,
        "llc_chains": llc_chains.tolist(),
        "init_loss": init_loss,
        "nbeta": nbeta,
        "num_kept": num_kept,
        "loss_trace": trace,
        "diverged": False,
    }


def _diverged(sgld, nbeta, init_loss, num_kept, num_draws, trace=None):
    return {
        "llc_mean": float("nan"),
        "llc_std": float("nan"),
        "llc_chains": [float("nan")] * sgld["num_chains"],
        "init_loss": init_loss,
        "nbeta": nbeta,
        "num_kept": num_kept,
        "loss_trace": np.full((sgld["num_chains"], num_draws), np.nan) if trace is None else trace,
        "diverged": True,
    }
