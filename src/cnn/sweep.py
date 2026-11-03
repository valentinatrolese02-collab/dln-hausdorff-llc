"""The experiment as configured in configs/sweep.yaml: which runs exist, their ids, and their stored results."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RESOLVED = "sweep_resolved.yaml"


def load_sweep(path, smoke=False):
    """The smoke block of the file replaces the top-level values with the same name."""
    sweep = yaml.safe_load(Path(path).read_text())
    overrides = sweep.pop("smoke", {})
    return sweep | overrides if smoke else sweep


def run_config(sweep, factor, value):
    """lr, batch_size and momentum of one point of a sweep. Refuses values the config does not list."""
    if factor not in sweep["factors"]:
        raise ValueError(f"unknown factor {factor}, the config has {list(sweep['factors'])}")
    matches = [v for v in sweep["factors"][factor] if np.isclose(v, value)]
    if not matches:
        raise ValueError(f"{factor}={value} is not in the config: {sweep['factors'][factor]}")
    return sweep["baseline"] | {factor: matches[0]}


def run_id(sweep, config, seed):
    return f"{sweep['arch']}_lr{config['lr']:g}_bs{config['batch_size']}_m{config['momentum']:g}_seed{seed}"


def distinct_runs(sweep):
    """(factor, value, seed) of every run to train. The baseline is shared between the sweeps, so it appears once."""
    runs, seen = [], set()
    for factor, values in sweep["factors"].items():
        for value in values:
            for seed in sweep["seeds"]:
                rid = run_id(sweep, run_config(sweep, factor, value), seed)
                if rid not in seen:
                    seen.add(rid)
                    runs.append((factor, value, seed))
    return runs


def strength_order(factor, values):
    """Values from weakest to strongest implicit regularisation: lower batch size is stronger, higher lr and momentum are."""
    return sorted(values, reverse=factor == "batch_size")


def load_resolved(results_dir):
    resolved = yaml.safe_load((Path(results_dir) / RESOLVED).read_text())
    return resolved["sweep"], resolved.get("sgld")


def load_timeseries(results_dir):
    """Long table with one row per factor, value, seed and logged step: a shared baseline appears once per factor.

    Also returns the run ids the config expects that have no CSV yet.
    """
    results_dir = Path(results_dir)
    sweep, _ = load_resolved(results_dir)
    frames, missing = [], []
    for factor, values in sweep["factors"].items():
        for value in values:
            for seed in sweep["seeds"]:
                rid = run_id(sweep, run_config(sweep, factor, value), seed)
                path = results_dir / "timeseries" / f"{rid}.csv"
                if path.exists():
                    frames.append(pd.read_csv(path, dtype={"llc_flag": "str"}).assign(factor=factor, value=value, seed=seed, run_id=rid))
                elif rid not in missing:
                    missing.append(rid)
    if not frames:
        raise FileNotFoundError(f"no timeseries in {results_dir / 'timeseries'}")
    return pd.concat(frames, ignore_index=True), missing
