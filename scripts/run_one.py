"""One run: train, then the LLC at each saved checkpoint. Each run can be launched on its own, e.g. from a job array.

Run with: uv run python scripts/run_one.py --factor lr --value 0.01 --seed 0

Safe to resume: a finished training stage is skipped, an unfinished one starts again, and the LLC continues
from the first checkpoint without an estimate.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnn.data import default_device, load_cifar10
from cnn.llc import estimate_llc, llc_flag
from cnn.sweep import RESOLVED, load_sweep, run_config, run_id
from cnn.train_vgg import train, write_csv
from models import VGG


def resolve_settings(args):
    """The sweep and the SGLD setting to use, and the output directory. SGLD is None when only training."""
    sweep = load_sweep(args.config, args.smoke)
    if args.arch:
        sweep["arch"] = args.arch
    out_dir = args.out_dir or Path("results/smoke" if args.smoke else "results")
    if args.smoke:
        sgld = sweep["sgld"]
    elif args.sgld.exists():
        sgld = yaml.safe_load(args.sgld.read_text())
    elif getattr(args, "train_only", False):
        sgld = None
    else:
        sys.exit(f"{args.sgld} not found. Train the baseline with --train-only and run src/cnn/calibrate_sgld.py first: every run must use one sampler.")
    return sweep, sgld, out_dir


def write_resolved(out_dir, sweep, sgld):
    """What the plots and the report read. Atomic, because parallel jobs all write the same content."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{RESOLVED}.{os.getpid()}.tmp"
    tmp.write_text(yaml.safe_dump({"sweep": sweep, "sgld": sgld}, sort_keys=False))  # the key order is the row order of the figures
    os.replace(tmp, out_dir / RESOLVED)


def check_same_settings(path, settings, what):
    """Values from one experiment are only comparable under one setting, so a resume must not mix them."""
    if path.exists():
        found = json.loads(path.read_text())
        if found != settings:
            sys.exit(f"{path} has {what} {found}, not {settings}. Values would not be comparable: use another --out-dir.")
    else:
        path.write_text(json.dumps(settings, indent=2))


def llc_chain_seeds(seed, step, num_chains):
    """Separate from the training streams, and different for each seed, checkpoint and chain."""
    return [int(s) for s in np.random.SeedSequence([seed, step]).generate_state(num_chains)]


def estimate_pending_llc(run_dir, csv_path, arch, seed, sgld, device, data_root, subset):
    """Writes csv_path after each checkpoint, so a crash loses at most one estimate."""
    df = pd.read_csv(csv_path, dtype={"llc_flag": "str"})
    checkpoints = json.loads((run_dir / "train.json").read_text())["checkpoints"]
    pending = [step for step in checkpoints if pd.isna(df.loc[df["step"] == step, "llc_flag"]).all()]
    if not pending:
        return
    x, y, _, _ = load_cifar10(data_root, subset)
    model = VGG(arch, with_bias=True)
    for step in pending:
        model.load_state_dict(torch.load(run_dir / "weights" / f"step_{step}.pt", map_location="cpu"))
        res = estimate_llc(model, x, y, sgld, llc_chain_seeds(seed, step, sgld["num_chains"]), device)
        df.loc[df["step"] == step, ["llc_mean", "llc_std", "llc_flag"]] = [res["llc/mean"], res["llc/std"], llc_flag(res)]
        write_csv(df, csv_path)
        print(f"step {step}: LLC {res['llc/mean']:.1f} +- {res['llc/std']:.1f} ({llc_flag(res)})", flush=True)


def run_one(sweep, factor, value, seed, sgld, out_dir, device, data_root="data", train_only=False):
    config = run_config(sweep, factor, value)
    rid = run_id(sweep, config, seed)
    run_dir, csv_path = out_dir / "runs" / rid, out_dir / "timeseries" / f"{rid}.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    measurement = {k: sweep[k] for k in ("steps", "log_every", "num_checkpoints", "window", "ms", "num_projections", "subset")}
    check_same_settings(run_dir / "settings.json", measurement, "settings")

    if (run_dir / "train.json").exists():
        print(f"training finished, skip {rid}", flush=True)
    else:
        print(f"train {rid}", flush=True)
        train(sweep["arch"], **config, seed=seed, run_dir=run_dir, csv_path=csv_path, device=device, data_root=data_root, **measurement)
    if not train_only:
        check_same_settings(run_dir / "sgld.json", sgld, "SGLD settings")
        estimate_pending_llc(run_dir, csv_path, sweep["arch"], seed, sgld, device, data_root, sweep["subset"])


def add_common_args(parser):
    parser.add_argument("--config", type=Path, default=Path("configs/sweep.yaml"))
    parser.add_argument("--sgld", type=Path, default=Path("configs/sgld.yaml"))
    parser.add_argument("--out-dir", type=Path, default=None, help="default results, or results/smoke with --smoke")
    parser.add_argument("--arch", choices=["VGG4", "VGG6"], default=None, help="overrides the config")
    parser.add_argument("--device", default=default_device(), help="cuda, mps or cpu")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke", action="store_true", help="the small experiment in the smoke block of the config")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", required=True, choices=["lr", "batch_size", "momentum"])
    parser.add_argument("--value", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-only", action="store_true", help="skip the LLC stage, for the baseline that calibration needs")
    add_common_args(parser)
    args = parser.parse_args()
    sweep, sgld, out_dir = resolve_settings(args)
    write_resolved(out_dir, sweep, sgld)
    run_one(sweep, args.factor, args.value, args.seed, sgld, out_dir, args.device, args.data_root, args.train_only)


if __name__ == "__main__":
    main()
