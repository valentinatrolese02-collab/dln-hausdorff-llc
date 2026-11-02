"""Train a DLN on a teacher matrix, tracking the LLC along training and estimating alpha at the end.

Model and SGD defaults follow LLC's expt_dln_saddle_dynamics.py. Run with: uv run python src/train.py
"""

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import levy_stable

from estimators import (
    ALPHA_WINDOW,
    estimate_alpha,
    estimate_llc,
    increment_diagnostics,
    true_dln_learning_coefficient,
)
from models import DLN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-out-dim", type=int, default=5)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--init-exponent", type=float, default=1.5)
    parser.add_argument("--num-data", type=int, default=10000)
    parser.add_argument("--label-noise", type=float, default=1.0, help="scale of the label noise")
    parser.add_argument("--noise-alpha", type=float, default=2.0, help="tail index of the label noise, 2 is Gaussian")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sgld-epsilon", type=float, default=1e-6)
    parser.add_argument("--sgld-gamma", type=float, default=1.0)
    parser.add_argument("--sgld-steps", type=int, default=1000)
    parser.add_argument("--sgld-batch-size", type=int, default=128)
    parser.add_argument("--rank-tol", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", help="cpu, mps or cuda; the GPU was slower on every model tried here")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def make_teacher_data(in_out_dim, num_data, label_noise, noise_alpha, seed):
    # Without label noise SGD has no gradient noise left at the minimum, and alpha has nothing to measure.
    # Symmetric alpha-stable noise makes the gradient noise heavy-tailed with a known index.
    teacher = 10.0 * torch.diag(torch.arange(1, in_out_dim + 1).float())
    x = torch.randn(num_data, in_out_dim)
    if noise_alpha == 2.0:
        noise = torch.randn(num_data, in_out_dim)
    else:
        noise = torch.from_numpy(levy_stable.rvs(noise_alpha, 0.0, size=(num_data, in_out_dim), random_state=seed)).float()
    return x, x @ teacher + label_noise * noise

# train use SGD
def train(args):
    torch.manual_seed(args.seed)
    # The estimators draw from their own stream so that measuring does not change the training run
    estimator_rng = torch.Generator().manual_seed(args.seed + 1)

    x, y = make_teacher_data(args.in_out_dim, args.num_data, args.label_noise, args.noise_alpha, args.seed)
    widths = [args.in_out_dim] + [args.width] * args.num_hidden_layers + [args.in_out_dim]
    model = DLN(widths, sigma=args.width ** (-args.init_exponent / 2))
    device = torch.device(args.device)
    print(f"device: {device}")
    model, x, y = model.to(device), x.to(device), y.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    iterates = deque(maxlen=ALPHA_WINDOW + 1)
    steps_per_epoch = args.num_data // args.batch_size
    records = []
    step = 0
    while step < args.num_steps:
        perm = torch.randperm(args.num_data)
        for start in range(0, args.num_data - args.batch_size + 1, args.batch_size):
            idx = perm[start : start + args.batch_size].to(device)
            loss = F.mse_loss(model(x[idx]), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            iterates.append([p.detach().flatten().clone() for p in model.parameters()])

            if step % args.log_every == 0:
                records.append(measure_llc(model, x, loss.item(), step, steps_per_epoch, widths, args, estimator_rng))
                print({k: v for k, v in records[-1].items() if k in ("step", "train_loss", "lambdahat", "rank")})
            if step >= args.num_steps:
                break

    with torch.no_grad():
        final_loss = F.mse_loss(model(x), y).item()
    return records, measure_alpha(iterates, estimator_rng) | {"final_loss": final_loss}


def measure_llc(model, x, train_loss, step, steps_per_epoch, widths, args, estimator_rng):
    lambdahat = estimate_llc(
        model,
        x,
        epsilon=args.sgld_epsilon,
        gamma=args.sgld_gamma,
        num_steps=args.sgld_steps,
        batch_size=args.sgld_batch_size,
        generator=estimator_rng,
    )
    # Thresholded rank: matrix_rank's numerical tolerance counts the unlearned modes as nonzero
    singular_values = torch.linalg.svdvals(model.end_to_end().detach())
    rank = (singular_values > args.rank_tol).sum().item()

    return {
        "step": step,
        "epoch": step / steps_per_epoch,
        "train_loss": train_loss,
        "lambdahat": lambdahat,
        "true_lambda": true_dln_learning_coefficient(rank, widths),
        "rank": rank,
        "singular_values": singular_values.tolist(),
    }


def measure_alpha(iterates, estimator_rng):
    """HD estimates once, at the end of training: one alpha per layer, and beta_S = max over layers (Sec. 4)."""
    layers = [torch.stack(layer_iterates) for layer_iterates in zip(*iterates)]
    alpha_per_layer = [estimate_alpha(layer, generator=estimator_rng)[0] for layer in layers]
    diagnostics = [increment_diagnostics(layer, generator=estimator_rng) for layer in layers]
    return {
        "beta": max(alpha_per_layer),
        "alpha_per_layer": alpha_per_layer,
        "increment_lag1": float(np.mean([lag1 for lag1, _ in diagnostics])),
        "increment_zero_fraction": float(np.mean([zero_fraction for _, zero_fraction in diagnostics])),
    }


def plot(records, final, path):
    ink, muted, grid = "#52514e", "#898781", "#e1e0d9"
    blue, orange = "#2a78d6", "#eb6834"
    epoch = [r["epoch"] for r in records]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, facecolor="#fcfcfb")
    fig.suptitle(rf"End of training: $\beta_S$ = {final['beta']:.2f}", x=0.02, ha="left", color=ink)

    axes[0].plot(epoch, [r["train_loss"] for r in records], color=blue, lw=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Training loss", loc="left", color=ink)

    axes[1].plot(epoch, [r["lambdahat"] for r in records], color=blue, lw=2, label=r"estimated $\hat\lambda$")
    axes[1].plot(epoch, [r["true_lambda"] for r in records], color=orange, lw=2, label=r"theoretical $\lambda$ at current rank")
    axes[1].legend(frameon=False, labelcolor=ink)
    axes[1].set_title("Local learning coefficient", loc="left", color=ink)
    axes[1].set_xlabel("Epoch", color=ink)

    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def append_summary(path, args, records, final):
    # lambdahat is noisy from one measurement to the next, so average the last few
    lambdahats = [r["lambdahat"] for r in records[-5:]]
    row = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "label_noise": args.label_noise,
        "noise_alpha": args.noise_alpha,
        "seed": args.seed,
        "final_loss": f"{final['final_loss']:.4g}",
        "rank": records[-1]["rank"],
        "lambdahat": f"{np.mean(lambdahats):.2f}",
        "true_lambda": records[-1]["true_lambda"],
        "beta": f"{final['beta']:.2f}",
        "lag1": f"{final['increment_lag1']:.2f}",
        "zero_fraction": f"{final['increment_zero_fraction']:.2f}",
    }
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(row)


def main():
    args = parse_args()
    if args.num_steps <= ALPHA_WINDOW:
        raise ValueError(f"--num-steps must exceed the alpha window of {ALPHA_WINDOW} steps")
    records, final = train(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"lr{args.lr:g}_bs{args.batch_size}_noise{args.label_noise:g}_na{args.noise_alpha:g}_seed{args.seed}"
    (args.out_dir / f"{run_name}.json").write_text(json.dumps({"records": records, "final": final}))
    plot(records, final, args.out_dir / f"{run_name}.png")
    append_summary(args.out_dir / "summary.csv", args, records, final)


if __name__ == "__main__":
    main()
