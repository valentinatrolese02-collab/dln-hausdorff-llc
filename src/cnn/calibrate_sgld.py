"""Scan SGLD step size and localisation on one trained run, and pick one setting for the whole sweep.

Run with: uv run python src/cnn/calibrate_sgld.py --run-dir results/runs/<baseline run>
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, when run as a script

from cnn.data import default_device, load_cifar10
from cnn.llc import equivalent_epochs, estimate_llc, load_trained_model

UNIFORM_GUESS_LOSS = math.log(10)
MAX_PLATEAU_DRIFT = 0.05
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def assess(result, max_drift=MAX_PLATEAU_DRIFT):
    """Applies the selection rule to a result with the whole trace, returns (stable, reason).

    Plateau: between the last two windows of num_kept draws, the mean loss moves by less than
    max_drift of its total rise above L(w*).
    """
    if result["diverged"]:
        return False, "diverged"
    if min(result["llc_chains"]) <= 0:
        return False, "negative LLC"
    trace, keep = result["loss_trace"], result["num_kept"]
    last, previous = trace[:, -keep:].mean(axis=1), trace[:, -2 * keep : -keep].mean(axis=1)
    if last.max() >= UNIFORM_GUESS_LOSS:
        return False, "left the basin"
    drift = np.abs(last - previous) / (last - result["init_loss"])
    if drift.max() >= max_drift:
        return False, f"no plateau (drift {drift.max():.2f})"
    return True, "stable"


def pick(entries):
    """Among stable settings, the smallest localisation (AW H.3: large values drown the likelihood), then the smallest step."""
    stable = [e for e in entries if e["stable"]]
    return min(stable, key=lambda e: (e["localization"], e["lr"])) if stable else None


def plot(entries, lrs, localizations, path):
    fig, axes = plt.subplots(
        len(localizations), len(lrs), figsize=(max(3 * len(lrs), 7), max(2.2 * len(localizations), 3)), squeeze=False, facecolor=SURFACE
    )
    for entry in entries:
        ax = axes[localizations.index(entry["localization"])][lrs.index(entry["lr"])]
        for trace, color, label in zip(entry["trace"], (BLUE, ORANGE), ("chain 1", "chain 2")):
            ax.plot(trace, color=color, lw=1, label=label)
        ax.axhline(entry["init_loss"], color=MUTED, lw=1, ls="--", label=r"$L(w^*)$")
        ax.set_title(
            rf"$\epsilon$={entry['lr']:g}, $\gamma$={entry['localization']:g}: {entry['reason']}"
            f"\nLLC {entry['llc_mean']:.0f}",
            fontsize=8,
            loc="left",
            color=INK,
        )
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, length=0, labelsize=7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, labelcolor=INK, fontsize=8)
    fig.suptitle("SGLD minibatch loss by step", x=0.02, ha="left", color=INK)
    fig.supxlabel("SGLD step", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="directory with state_dict.pt and train.json")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-7, 3e-7, 1e-6, 3e-6], help="AW used 5e-8 to 2e-7 on ResNet18")
    parser.add_argument("--localizations", type=float, nargs="+", default=[1.0, 10.0, 100.0], help="AW used 1 and recommend 1 to 10")
    parser.add_argument("--num-steps", type=int, default=3000, help="AW used 2000 to 3000")
    parser.add_argument("--burn-in", type=float, default=0.9)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("configs/sgld.yaml"))
    parser.add_argument("--plot-dir", type=Path, default=Path("results/calibration"))
    return parser.parse_args()


def main():
    args = parse_args()
    x, y, _, _ = load_cifar10(args.data_root, args.subset)
    model = load_trained_model(args.run_dir)
    base = {"num_steps": args.num_steps, "burn_in": args.burn_in, "num_chains": args.num_chains, "batch_size": args.batch_size}

    num_settings = len(args.lrs) * len(args.localizations)
    per_setting = equivalent_epochs(base, len(x))
    print(f"{num_settings} settings x {per_setting:.0f} equivalent training epochs = {num_settings * per_setting:.0f} epochs on {args.device}")

    entries = []
    for localization in args.localizations:
        for lr in args.lrs:
            result = estimate_llc(model, x, y, base | {"lr": lr, "localization": localization}, args.seed, args.device, keep_burnin=True)
            stable, reason = assess(result)
            entries.append(
                {
                    "lr": lr,
                    "localization": localization,
                    "stable": stable,
                    "reason": reason,
                    "llc_mean": result["llc_mean"],
                    "llc_std": result["llc_std"],
                    "init_loss": result["init_loss"],
                    "nbeta": result["nbeta"],
                    "trace": result["loss_trace"],
                }
            )
            print(f"lr={lr:g} localization={localization:g}: LLC {result['llc_mean']:.1f} +- {result['llc_std']:.1f}, {reason}", flush=True)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot(entries, args.lrs, args.localizations, args.plot_dir / "loss_traces.png")
    table = [{k: v for k, v in e.items() if k != "trace"} for e in entries]
    (args.plot_dir / "calibration.json").write_text(json.dumps(table, indent=2))
    print(f"traces in {args.plot_dir / 'loss_traces.png'}")

    chosen = pick(entries)
    if chosen is None:
        sys.exit(f"No setting passed, so {args.out} was not written. Inspect the traces and widen the scan.")
    config = base | {"lr": chosen["lr"], "localization": chosen["localization"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=True))
    print(f"chose lr={chosen['lr']:g}, localization={chosen['localization']:g}, wrote {args.out}")


if __name__ == "__main__":
    main()
