"""Scan SGLD step size and localisation on one trained run and write the first stable setting to configs/sgld.yaml.

Run with: uv run python src/cnn/calibrate_sgld.py --run-dir results/runs/<baseline run>
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, when run as a script

from cnn.data import default_device, load_cifar10
from cnn.llc import SGLD_CHAIN_DEFAULTS, equivalent_epochs, estimate_llc
from models import VGG


def assess(res, num_steps, burn_in, max_drift=0.05):
    """Returns (stable, reason, llc per chain) for a run made with burn_in=0, so res["loss/trace"] is the whole chain."""
    if res.get("diverged"):
        return False, "diverged", None
    k = num_steps - int(burn_in * num_steps)  # the draws the sweep will keep
    trace = res["loss/trace"]
    last, previous = trace[:, -k:].mean(axis=1), trace[:, -2 * k : -k].mean(axis=1)
    llc = res["nbeta"] * (last - res["init_loss"])
    if llc.min() <= 0:
        return False, "negative LLC", llc
    if last.max() >= math.log(10):  # the loss of a uniform guess over 10 classes
        return False, "left the basin", llc
    drift = np.abs(last - previous) / (last - res["init_loss"])
    if drift.max() >= max_drift:
        return False, f"no plateau (drift {drift.max():.2f})", llc
    return True, "stable", llc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True, help="directory with state_dict.pt and train.json")
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-7, 1e-6])
    p.add_argument("--localizations", type=float, nargs="+", default=[1.0, 10.0])
    p.add_argument("--num-steps", type=int, default=SGLD_CHAIN_DEFAULTS["num_steps"])
    p.add_argument("--burn-in", type=float, default=SGLD_CHAIN_DEFAULTS["burn_in"])
    p.add_argument("--num-chains", type=int, default=SGLD_CHAIN_DEFAULTS["num_chains"])
    p.add_argument("--batch-size", type=int, default=SGLD_CHAIN_DEFAULTS["batch_size"])
    p.add_argument("--device", default=default_device())
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("configs/sgld.yaml"))
    args = p.parse_args()

    x, y, _, _ = load_cifar10("data", args.subset)
    model = VGG(json.loads((args.run_dir / "train.json").read_text())["arch"], with_bias=True)
    model.load_state_dict(torch.load(args.run_dir / "state_dict.pt", map_location="cpu"))
    base = {"num_steps": args.num_steps, "burn_in": args.burn_in, "num_chains": args.num_chains, "batch_size": args.batch_size}
    cost = equivalent_epochs(base, len(x))
    print(f"{len(args.lrs) * len(args.localizations)} settings x {cost:.0f} equivalent training epochs on {args.device}")

    fig, axes = plt.subplots(len(args.localizations), len(args.lrs), figsize=(4 * len(args.lrs), 3 * len(args.localizations)), squeeze=False)
    chosen = None
    for i, localization in enumerate(args.localizations):  # smallest localisation first (AW H.3), then smallest step
        for j, lr in enumerate(args.lrs):
            res = estimate_llc(model, x, y, base | {"lr": lr, "localization": localization, "burn_in": 0}, 0, args.device)
            stable, reason, llc = assess(res, args.num_steps, args.burn_in)
            print(f"lr={lr:g} localization={localization:g}: {reason}" + ("" if llc is None else f", LLC {llc.mean():.1f}"), flush=True)
            if stable and chosen is None:
                chosen = (lr, localization)
            ax = axes[i][j]
            if llc is not None:
                ax.plot(res["loss/trace"].T, lw=1)
                ax.axhline(res["init_loss"], color="gray", ls="--")
            ax.set_title(f"lr={lr:g}, loc={localization:g}: {reason}", fontsize=8)
    out_dir = Path("results/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_traces.png", dpi=150)

    if chosen is None:
        sys.exit(f"No setting passed, so {args.out} was not written. Inspect {out_dir / 'loss_traces.png'} and widen the scan.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(base | {"lr": chosen[0], "localization": chosen[1]}, sort_keys=True))
    print(f"chose lr={chosen[0]:g}, localization={chosen[1]:g}, wrote {args.out}")


if __name__ == "__main__":
    main()