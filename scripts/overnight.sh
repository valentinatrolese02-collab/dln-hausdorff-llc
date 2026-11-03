#!/bin/sh
# Unattended night on one Mac: calibrate, falling back to a wider grid, then everything else, then figures and report.
# If no SGLD setting passes, there can be no LLC, so the remaining runs are still trained for the other three columns.
CONFIG=configs/overnight.yaml
CALIBRATE="src/cnn/calibrate_sgld.py --run-dir results/runs/VGG4_lr0.01_bs128_m0_seed0 --batch-size 256 --device mps"

if [ ! -f configs/sgld.yaml ]; then
    uv run python $CALIBRATE || uv run python $CALIBRATE --lrs 1e-7 3e-7 1e-6 --localizations 1 10 100
fi
if [ -f configs/sgld.yaml ]; then
    uv run python scripts/run_sweep.py --config $CONFIG --device mps
else
    echo "NO SGLD SETTING PASSED: training without LLC"
    uv run python scripts/run_sweep.py --config $CONFIG --list | while read -r run; do
        uv run python scripts/run_one.py --config $CONFIG --train-only --device mps $run < /dev/null
    done
fi
uv run python scripts/plot_training_curves.py
uv run python scripts/analyse.py
