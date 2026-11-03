"""results/report.md from results/timeseries. The text of the report is computed from the numbers.

Run with: uv run python scripts/analyse.py [--results-dir results]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnn.sweep import load_resolved, load_timeseries, strength_order
from plot_training_curves import epochs_note, mean_ci

COLUMNS = {
    "llc_mean": "LLC",
    "beta_iter_max": "alpha iter max",
    "beta_iter_median": "alpha iter median",
    "beta_incr_max": "alpha incr max",
    "beta_incr_median": "alpha incr median",
    "test_acc": "test acc",
    "train_loss": "train loss",
    "lag1": "lag-1",
    "zero_fraction": "zero fraction",
}
BETAS = ["beta_iter_max", "beta_iter_median", "beta_incr_max", "beta_incr_median"]
NEAR_ZERO_LOSS = 0.01  # what "training loss reaches about 0" is taken to mean here, fixed before seeing any data


def fmt(value, digits=3):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def fmt_ci(mean, half_width):
    """The mean alone when there is no interval, as with one seed."""
    return fmt(mean) if not np.isfinite(half_width) else f"{fmt(mean)} ± {fmt(half_width, 2)}"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def final_rows(at_value):
    """Each seed's last row with an LLC or tail measurement. Unstable LLC values do not enter the LLC column."""
    rows = at_value[at_value["llc_flag"].notna() | at_value["alpha_flag"].notna()]
    return rows.loc[rows.groupby("seed")["step"].idxmax()]


def final_value(rows, metric):
    values = rows[metric].where(rows["llc_flag"] == "ok") if metric == "llc_mean" else rows[metric]
    return mean_ci(values)


def compare_extremes(weak, strong, predicted=None):
    """Direction from the weakest to the strongest setting, and whether the 95% CIs of the two are apart."""
    (mean_w, half_w, n_w), (mean_s, half_s, n_s) = weak, strong
    if n_w == 0 or n_s == 0:
        return "not computable (no value at one extreme)"
    direction = "higher" if mean_s > mean_w else "lower"
    if n_w < 2 or n_s < 2:
        return f"{direction} at stronger regularisation, but no CI with one seed"
    apart = mean_w + half_w < mean_s - half_s or mean_s + half_s < mean_w - half_w
    text = f"{direction} at stronger regularisation, CIs {'apart' if apart else 'overlap'}"
    if predicted is None:
        return text
    if not apart:
        return text + ": not separated at these seeds"
    return text + (": as in the paper" if direction == predicted else ": opposite to the paper")


def describe_factor(factor, part, values):
    """Plain sentences from the final checkpoint. The rule is the same for every factor and every metric."""
    ordered = strength_order(factor, values)
    weakest, strongest = ordered[0], ordered[-1]
    finals = {v: final_rows(part[part["value"] == v]) for v in ordered if (part["value"] == v).any()}
    if weakest not in finals or strongest not in finals:
        return f"- No results yet at {factor} = {weakest:g} or {strongest:g}, so the extremes cannot be compared."
    lines = [f"- Weakest regularisation: {factor} = {weakest:g}. Strongest: {factor} = {strongest:g}."]
    for metric, label, predicted in [
        ("llc_mean", "LLC", "lower"),
        ("test_acc", "Test accuracy", "higher"),
        ("train_loss", "Training loss", None),
        ("beta_incr_max", "Alpha (increments, max)", None),
        ("beta_iter_max", "Alpha (iterates, max)", None),
    ]:
        weak, strong = final_value(finals[weakest], metric), final_value(finals[strongest], metric)
        lines.append(f"- {label}: {fmt_ci(*weak[:2])} at {weakest:g}, {fmt_ci(*strong[:2])} at {strongest:g}; {compare_extremes(weak, strong, predicted)}.")
        means = [final_value(finals[v], metric)[0] for v in ordered if v in finals]
        if len(means) > 2 and np.all(np.isfinite(means)) and not (np.all(np.diff(means) > 0) or np.all(np.diff(means) < 0)):
            lines[-1] += " Not monotonic across the values."
    not_near_zero = [f"{v:g}" for v in ordered if v in finals and not final_value(finals[v], "train_loss")[0] < NEAR_ZERO_LOSS]
    if not_near_zero:
        lines.append(f"- Final training loss is not below {NEAR_ZERO_LOSS} at {factor} = {', '.join(not_near_zero)}, so the comparison above is not one at training loss about 0.")
    else:
        lines.append(f"- Final training loss is below {NEAR_ZERO_LOSS} at every value.")
    for v in ordered:
        if v not in finals:
            continue
        at_value = part[part["value"] == v]
        by_step = at_value.groupby("step")["test_acc"].mean().dropna()
        llc = at_value[at_value["llc_flag"] == "ok"].groupby("step")["llc_mean"].mean()
        llc_text = f"mean LLC goes from {fmt(llc.iloc[0])} at step {llc.index[0]} to {fmt(llc.iloc[-1])} at step {llc.index[-1]}" if len(llc) else "no stable LLC value"
        lines.append(f"- Over training at {factor} = {v:g}: {llc_text}; mean test accuracy peaks at {fmt(by_step.max())} at step {by_step.idxmax()} and ends at {fmt(by_step.iloc[-1])}.")
    return "\n".join(lines)


def factor_section(df, sweep):
    parts = []
    for factor, values in sweep["factors"].items():
        part = df[df["factor"] == factor]
        rows = []
        for value in strength_order(factor, values):
            at_value = part[part["value"] == value]
            if at_value.empty:
                rows.append([f"{value:g}", 0, "n/a", 0] + ["n/a"] * len(COLUMNS))
                continue
            finals = final_rows(at_value)
            status = ", ".join(f"{name} {count}" for name, count in finals["llc_flag"].value_counts().items()) or "none yet"
            rows.append([f"{value:g}", len(finals), status, int((finals["alpha_flag"] == 1).sum()), *(fmt_ci(*final_value(finals, metric)[:2]) for metric in COLUMNS)])
        table = md_table([factor, "seeds", "LLC status", "alpha flagged", *COLUMNS.values()], rows)
        parts.append(f"### {factor}\n\nRows go from weakest to strongest implicit regularisation.\n\n{table}\n\n{describe_factor(factor, part, values)}")
    return "\n\n".join(parts)


def counts_section(df, sweep, missing):
    points = df.drop_duplicates(["run_id", "step"])
    runs = df["run_id"].nunique()
    llc = points["llc_flag"].value_counts()
    measured_llc = points["llc_flag"].notna().sum()
    measured_alpha = points["alpha_flag"].notna().sum()
    expected = sweep["num_checkpoints"] * (runs + len(missing))
    outside = {b: int((points[b].notna() & ((points[b] <= 0) | (points[b] > 2))).sum()) for b in BETAS}
    last_step = df.groupby("run_id")["step"].max()
    incomplete = int((last_step < sweep["steps"]).sum())
    lines = [
        f"- Runs with results: {runs}. Expected but missing: {len(missing)}{' (' + ', '.join(missing) + ')' if missing else ''}.",
        f"- Runs that stopped before step {sweep['steps']}, because training diverged: {incomplete}.",
        f"- LLC estimates: {measured_llc} of {expected} checkpoints. Stable {llc.get('ok', 0)}, negative {llc.get('negative', 0)}, left the basin {llc.get('left_basin', 0)}, diverged {llc.get('diverged', 0)}. All are kept in the CSVs; only stable values enter the LLC means.",
        f"- Tail index: {measured_alpha} of {expected} checkpoints measured, {int((points['alpha_flag'] == 1).sum())} flagged (mean |lag-1| > 0.1 or zero fraction > 0.05). Flagged values stay in the means.",
        f"- Alpha outside the theoretical range (0, 2]: {', '.join(f'{b} {count}' for b, count in outside.items())}.",
    ]
    return "\n".join(lines)


def analyse(results_dir):
    results_dir = Path(results_dir)
    sweep, sgld = load_resolved(results_dir)
    df, missing = load_timeseries(results_dir)
    report = f"""# VGG over training: LLC, tail index and test accuracy

Observation of Lau et al. tested here, not assumed: stronger implicit regularisation (higher learning rate, lower batch size, higher momentum) goes with lower LLC and higher test accuracy, even when the training loss reaches about 0. All numbers and sentences below are computed from `timeseries/`.

## What was run

- {sweep['arch']}, plain SGD on cross-entropy with constant learning rate, {sweep['steps']} steps, seeds {sweep['seeds']}.
- Baseline {sweep['baseline']}; one factor at a time: {sweep['factors']}.
- Tail index: window of {sweep['window']} iterates, {sweep['num_projections']} projections per weight tensor, block sizes {sweep['ms']}, at {sweep['num_checkpoints']} checkpoints.
- SGLD, one setting for every estimate: `{sgld}`.
- {epochs_note(sweep)}

![fig1](figures/fig1_style.png)

![alpha variants](figures/alpha_variants.png)

## Counts

{counts_section(df, sweep, missing)}

## Final checkpoint, by factor

Mean ± half-width of the 95% t interval over seeds, at each seed's last checkpoint. The LLC column uses stable estimates only, and "LLC status" counts all of them. A value without ± has no interval, as with one seed; "n/a" is a value that does not exist. Alpha columns include flagged checkpoints.

Each comparison below uses one rule: the two extreme settings are called "apart" when their 95% CIs do not overlap. No p-values are given. With a number of seeds as small as {len(sweep['seeds'])} an overlap is not evidence that there is no effect.

{factor_section(df, sweep)}
"""
    (results_dir / "report.md").write_text(report)
    print(f"wrote {results_dir / 'report.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    analyse(parser.parse_args().results_dir)


if __name__ == "__main__":
    main()
