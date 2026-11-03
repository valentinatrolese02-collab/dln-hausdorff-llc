"""Train an HD VGG on CIFAR-10 with plain SGD for a fixed number of steps.

Logs cheap metrics, computes the tail index from a rolling buffer at each checkpoint, and saves the
weights there for the LLC stage in scripts/run_one.py. HD's original script is in hd_reference/train_vgg.py.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cnn.alpha import ALPHA_MS, NUM_PROJECTIONS, ProjectionBuffer, tail_index
from cnn.data import evaluate, load_cifar10
from models import VGG

TIMESERIES_COLUMNS = [
    "step", "train_loss", "test_loss", "test_acc",
    "llc_mean", "llc_std", "llc_flag",
    "beta_iter_max", "beta_iter_median", "beta_incr_max", "beta_incr_median", "lag1", "zero_fraction", "alpha_flag",
]  # fmt: skip
DIRECTIONS_SEED_OFFSET = 100_000  # so the directions of one seed never share a stream with the training RNG of another


def checkpoint_steps(steps, num_checkpoints):
    """Evenly spaced over the run, the last one at the final step."""
    return [round(steps * (i + 1) / num_checkpoints) for i in range(num_checkpoints)]


def logged_steps(steps, log_every, checkpoints):
    return sorted(set(range(log_every, steps + 1, log_every)) | set(checkpoints))


def minibatch_indices(n, batch_size, generator):
    """Endless minibatches: reshuffled at each epoch, with the incomplete last batch dropped."""
    while True:
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n - batch_size + 1, batch_size):
            yield perm[start : start + batch_size]


def write_csv(df, path):
    """Atomic, so a crash or a parallel reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    df.to_csv(tmp, index=False, na_rep="NaN")
    os.replace(tmp, path)


def train(
    arch, lr, batch_size, momentum, steps, seed, run_dir, csv_path, device,
    log_every=250, num_checkpoints=12, window=361, ms=ALPHA_MS, num_projections=NUM_PROJECTIONS, data_root="data", subset=None,
):  # fmt: skip
    """One run: writes weights/step_*.pt, state_dict.pt, tail_details.json, csv_path, and last train.json."""
    if any((window - 1) % m for m in ms):
        raise ValueError(f"every block size in {ms} must divide N = window - 1 = {window - 1}")
    run_dir = Path(run_dir)
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_cifar10(data_root, subset)

    torch.manual_seed(seed)
    model = VGG(arch, with_bias=True).to(device)
    # Own generators, so that measuring never changes the training run
    batches = minibatch_indices(len(x_train), batch_size, torch.Generator().manual_seed(seed))
    buffer = ProjectionBuffer(model, torch.Generator().manual_seed(seed + DIRECTIONS_SEED_OFFSET), window, num_projections)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    x, y = x_train.to(device), y_train.to(device)

    checkpoints = checkpoint_steps(steps, num_checkpoints)
    to_log = set(logged_steps(steps, log_every, checkpoints))
    rows, details = [], {}
    running_loss, running_steps = 0.0, 0
    diverged_at = None
    model.train()
    for step in range(1, steps + 1):
        idx = next(batches).to(device)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        if not torch.isfinite(loss):
            diverged_at = step
            print(f"training diverged at step {step}", flush=True)
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        buffer.push()
        running_loss += loss.item()
        running_steps += 1

        if step not in to_log:
            continue
        test_acc, test_loss = evaluate(model, x_test, y_test, device)
        model.train()
        row = {"step": step, "train_loss": running_loss / running_steps, "test_loss": test_loss, "test_acc": test_acc}
        running_loss, running_steps = 0.0, 0
        if step in checkpoints:
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "weights" / f"step_{step}.pt")
            if buffer.is_full():  # the first checkpoints of a smoke run come before the window is full
                summary = tail_index(*buffer.series(), ms)
                details[step] = summary["per_layer"] | {"mean_abs_lag1": summary["mean_abs_lag1"]}
                row |= {k: v for k, v in summary.items() if k in TIMESERIES_COLUMNS}
        rows.append(row)
        print(f"step {step}: train loss {row['train_loss']:.4f}, test acc {test_acc:.4f}", flush=True)

    df = pd.DataFrame(rows, columns=TIMESERIES_COLUMNS)
    df["llc_flag"] = df["llc_flag"].astype("str").where(df["llc_flag"].notna())
    df["alpha_flag"] = df["alpha_flag"].astype("float")
    write_csv(df, csv_path)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "state_dict.pt")
    (run_dir / "tail_details.json").write_text(json.dumps(details, indent=2))
    metrics = {
        "arch": arch, "lr": lr, "batch_size": batch_size, "momentum": momentum, "steps": steps, "seed": seed,
        "log_every": log_every, "window": window, "subset": subset, "num_train": len(x_train),
        "epochs": steps * batch_size / (len(x_train) - len(x_train) % batch_size),
        "checkpoints": [int(s) for s in np.intersect1d(checkpoints, df["step"])],
        "diverged_at": diverged_at,
    }  # fmt: skip
    # Last, so that an existing train.json means the training stage is complete
    (run_dir / "train.json").write_text(json.dumps(metrics, indent=2))
    return metrics
