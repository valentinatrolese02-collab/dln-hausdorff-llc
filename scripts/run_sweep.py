"""Run the one-factor-at-a-time sweep: train, tail index and LLC for every config and seed. Safe to resume.

Run with: uv run python scripts/run_sweep.py
Smoke test: uv run python scripts/run_sweep.py --smoke
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analyse
from cnn.alpha import load_tail_index
from cnn.data import default_device, load_cifar10
from cnn.llc import equivalent_epochs, estimate_llc, load_sgld_config, load_trained_model
from cnn.train_vgg import train

# Batch 32 is the largest whose final epoch on 2000 images (62 steps) still fits a window of 36
SMOKE_SWEEP = {
    "arch": "VGG4",
    "epochs": 2,
    "seeds": [0],
    "baseline": {"lr": 0.01, "batch_size": 32, "momentum": 0.0, "l2": 0.0},
    "factors": {"lr": [0.003, 0.01], "momentum": [0.0, 0.5]},
}
SMOKE_SUBSET = 2000
SMOKE_SGLD = {"lr": 1e-6, "localization": 1.0, "num_steps": 20, "burn_in": 0.9, "num_chains": 2, "batch_size": 64}


def expand_configs(sweep):
    """Distinct hyperparameter settings, with the shared baseline counted once."""
    configs = []
    for factor, values in sweep["factors"].items():
        for value in values:
            config = sweep["baseline"] | {factor: value}
            if config not in configs:
                configs.append(config)
    return configs


def config_name(arch, config):
    return f"{arch}_lr{config['lr']:g}_bs{config['batch_size']}_m{config['momentum']:g}_l2{config['l2']:g}"


def summary_row(name, metrics, tail, llc):
    return {
        "config": name,
        "arch": metrics["arch"],
        "lr": metrics["lr"],
        "batch_size": metrics["batch_size"],
        "momentum": metrics["momentum"],
        "l2": metrics["l2"],
        "epochs": metrics["epochs"],
        "seed": metrics["seed"],
        "train_acc": metrics["train_acc"],
        "test_acc": metrics["test_acc"],
        "train_loss": metrics["train_loss"],
        "test_loss": metrics["test_loss"],
        "gap_acc": metrics["gap_acc"],
        "gap_loss": metrics["gap_loss"],
        "llc_mean": llc["llc_mean"],
        "llc_std": llc["llc_std"],
        "beta_iter_max": tail["beta_iter_max"],
        "beta_iter_median": tail["beta_iter_median"],
        "beta_incr_max": tail["beta_incr_max"],
        "beta_incr_median": tail["beta_incr_median"],
        "alpha_per_layer": json.dumps(tail["per_layer"]),
        "window": tail["window"],
        "mean_lag1": tail["mean_lag1"],
        "mean_abs_lag1": tail["mean_abs_lag1"],
        "zero_fraction": tail["zero_fraction"],
        "flagged": tail["flagged"],
        "train_diverged": metrics["diverged"],
        "sgld_diverged": llc["diverged"],
    }


NO_TAIL = dict.fromkeys(
    ["beta_iter_max", "beta_iter_median", "beta_incr_max", "beta_incr_median", "mean_lag1", "mean_abs_lag1", "zero_fraction", "window"],
    float("nan"),
) | {"per_layer": {}, "flagged": True}
NO_LLC = {"llc_mean": float("nan"), "llc_std": float("nan"), "llc_chains": [], "diverged": False}


def run_one(sweep, config, seed, sgld, out_dir, device, data_root, subset):
    """One config and seed. train.json and run.json mark finished stages, so a crash loses at most one stage."""
    name = config_name(sweep["arch"], config)
    run_dir = out_dir / "runs" / f"{name}_seed{seed}"
    if (run_dir / "run.json").exists():
        finished = json.loads((run_dir / "run.json").read_text())
        if finished["sgld"] != sgld:
            sys.exit(f"{run_dir} used SGLD settings {finished['sgld']}, not {sgld}. LLC values would not be comparable.")
        print(f"skip {run_dir.name}")
        return
    print(f"run {run_dir.name}", flush=True)
    if (run_dir / "train.json").exists():
        metrics = json.loads((run_dir / "train.json").read_text())
    else:
        metrics = train(sweep["arch"], **config, epochs=sweep["epochs"], seed=seed, out_dir=run_dir, device=device, data_root=data_root, subset=subset)

    if metrics["diverged"]:
        tail, llc = NO_TAIL, NO_LLC
    else:
        tail = load_tail_index(run_dir / "projections.npz")
        x, y, _, _ = load_cifar10(data_root, subset)
        # Offset from the training seeds so the SGLD stream never coincides with the training stream
        llc = estimate_llc(load_trained_model(run_dir), x, y, sgld, seed + 1000, device)
        np.save(run_dir / "sgld_loss_trace.npy", llc.pop("loss_trace"))

    record = {"row": summary_row(name, metrics, tail, llc), "train": metrics, "tail_index": tail, "llc": llc, "sgld": sgld}
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))


def write_summary(out_dir):
    """Rebuilt from the per-run JSONs, so that a crash or a resume cannot duplicate or lose a row."""
    rows = [json.loads(path.read_text())["row"] for path in sorted(out_dir.glob("runs/*/run.json"))]
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sweep.yaml"))
    parser.add_argument("--sgld", type=Path, default=Path("configs/sgld.yaml"))
    parser.add_argument("--out-dir", type=Path, default=None, help="default results, or results/smoke with --smoke")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 2000 images, tiny SGLD chain, end to end")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        sweep, sgld, subset = SMOKE_SWEEP, SMOKE_SGLD, SMOKE_SUBSET
        out_dir = args.out_dir or Path("results/smoke")
    else:
        if not args.sgld.exists():
            sys.exit(f"{args.sgld} not found. Run src/cnn/calibrate_sgld.py first: every run must use one calibrated sampler.")
        sweep, sgld, subset = yaml.safe_load(args.config.read_text()), load_sgld_config(args.sgld), None
        out_dir = args.out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = expand_configs(sweep)
    num_runs = len(configs) * len(sweep["seeds"])
    n = subset or 50000
    print(
        f"{len(configs)} configs x {len(sweep['seeds'])} seeds = {num_runs} runs on {args.device}: "
        f"{num_runs * sweep['epochs']} training epochs + {num_runs * equivalent_epochs(sgld, n):.0f} equivalent SGLD epochs"
    )
    (out_dir / "sweep_resolved.yaml").write_text(yaml.safe_dump({"sweep": sweep, "sgld": sgld}))
    for config in configs:
        for seed in sweep["seeds"]:
            run_one(sweep, config, seed, sgld, out_dir, args.device, args.data_root, subset)
            write_summary(out_dir)
    if args.smoke:
        analyse.analyse(out_dir)


if __name__ == "__main__":
    main()
