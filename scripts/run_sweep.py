"""Build the list of runs from configs/sweep.yaml and run them one after the other. Safe to resume.

See the cost first:  uv run python scripts/run_sweep.py --dry-run
Run everything:      uv run python scripts/run_sweep.py
Job array lines:     uv run python scripts/run_sweep.py --list
Smoke test:          uv run python scripts/run_sweep.py --smoke
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analyse
import plot_training_curves
from cnn.llc import SGLD_CHAIN_DEFAULTS, equivalent_epochs
from cnn.sweep import distinct_runs, load_sweep
from run_one import add_common_args, resolve_settings, run_one, write_resolved


def dry_run_lines(sweep, sgld, sgld_source):
    runs = distinct_runs(sweep)
    per_run = sweep["num_checkpoints"] * equivalent_epochs(sgld, sweep["subset"] or 50000)
    return [
        f"runs: {len(runs)}",
        f"SGD steps: {len(runs) * sweep['steps']}",
        f"SGLD: {len(runs) * per_run:.0f} equivalent training epochs ({sgld_source})",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print the number of runs, SGD steps and SGLD cost, and nothing else")
    parser.add_argument("--list", action="store_true", help="print one line of run_one.py arguments per run")
    add_common_args(parser)
    args = parser.parse_args()

    if args.dry_run or args.list:
        sweep = load_sweep(args.config, args.smoke)
        if args.list:
            for factor, value, seed in distinct_runs(sweep):
                print(f"--factor {factor} --value {value:g} --seed {seed}")
        elif args.smoke:
            print("\n".join(dry_run_lines(sweep, sweep["sgld"], "smoke chain")))
        elif args.sgld.exists():
            print("\n".join(dry_run_lines(sweep, yaml.safe_load(args.sgld.read_text()), str(args.sgld))))
        else:
            print("\n".join(dry_run_lines(sweep, SGLD_CHAIN_DEFAULTS, f"calibration default chain, no {args.sgld} yet")))
        return

    sweep, sgld, out_dir = resolve_settings(args)
    write_resolved(out_dir, sweep, sgld)
    for factor, value, seed in distinct_runs(sweep):
        run_one(sweep, factor, value, seed, sgld, out_dir, args.device, args.data_root)
    if args.smoke:
        plot_training_curves.plot_all(out_dir)
        analyse.analyse(out_dir)


if __name__ == "__main__":
    main()
