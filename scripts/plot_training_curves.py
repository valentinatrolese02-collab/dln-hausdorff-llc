"""Figure 1 of Lau et al. in time, for a VGG and with the tail index: results/figures/fig1_style.png and alpha_variants.png.

Run with: uv run python scripts/plot_training_curves.py [--results-dir results] [--alpha-variant beta_incr_max]
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnn.sweep import load_resolved, load_timeseries, strength_order

# One blue ramp: darker always means stronger implicit regularisation, in every row
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK, MUTED, GRID, AXIS, SURFACE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ALPHA_LABELS = {
    "beta_iter_max": "iterates, max over layers",
    "beta_iter_median": "iterates, median over layers",
    "beta_incr_max": "increments, max over layers",
    "beta_incr_median": "increments, median over layers",
}
FACTOR_LABELS = {"lr": "learning rate", "batch_size": "batch size", "momentum": "momentum"}
UNSTABLE_LLC = ("negative", "left_basin")


def mean_ci(values):
    """Mean and half-width of the 95% t interval over seeds, NaN when it is undefined."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), 0
    if len(values) == 1:
        return float(values[0]), float("nan"), 1
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    return float(values.mean()), float(half_width), len(values)


def ramp_colours(factor, values):
    """Value to colour, from weakest (light) to strongest (dark) regularisation."""
    ordered = strength_order(factor, values)
    picks = np.linspace(0, len(RAMP) - 1, len(ordered)).round().astype(int) if len(ordered) > 1 else [len(RAMP) - 1]
    return {value: RAMP[i] for value, i in zip(ordered, picks)}


def style_axes(ax, title=None, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK)


def draw_curve(ax, rows, metric, colour, markers=False, log=False):
    """Mean over seeds with a 95% CI band, where more than one seed has a value. Returns what was drawn, for fit_limits."""
    summary = [(step, *mean_ci(part[metric])) for step, part in rows.groupby("step")]
    summary = [row for row in summary if row[3] > 0]
    if not summary:
        return np.array([])
    steps, means, half_widths = (np.array(column, dtype=float) for column in list(zip(*summary))[:3])
    lower, upper = means - half_widths, means + half_widths
    if log:  # a t interval can reach below zero, which a log axis cannot show
        lower = np.where(lower > 0, lower, means)
    ax.fill_between(steps, lower, upper, color=colour, alpha=0.18, lw=0)
    ax.plot(steps, means, color=colour, lw=2, marker="o" if markers else None, ms=4)
    return np.concatenate([means, lower, upper])


def draw_hollow(ax, rows, metric, colour):
    ax.scatter(rows["step"], rows[metric], s=30, facecolors=SURFACE, edgecolors=colour, linewidths=1.2, zorder=3)


def fit_limits(ax, drawn, include=()):
    """Scale to the means and bands, so that one far-off hollow point cannot flatten a panel."""
    data = np.concatenate([*drawn, np.array(include, dtype=float)]) if drawn else np.array(include, dtype=float)
    data = data[np.isfinite(data)]
    if data.size:
        low, high = data.min(), data.max()
        margin = 0.1 * (high - low) if high > low else 0.5
        ax.set_ylim(low - margin, high + margin)


def draw_panels(axes, rows, colours, alpha_variant):
    """One row of fig1_style.png: LLC, tail index, test accuracy, training loss."""
    drawn_llc, drawn_alpha = [], []
    for value, colour in colours.items():
        at_value = rows[rows["value"] == value]
        checkpoints = at_value[at_value["llc_flag"].notna()]
        # Unstable LLC values stay out of the mean and the band; each seed's value is drawn hollow
        drawn_llc.append(draw_curve(axes[0], checkpoints[checkpoints["llc_flag"] == "ok"], "llc_mean", colour, markers=True))
        draw_hollow(axes[0], checkpoints[checkpoints["llc_flag"].isin(UNSTABLE_LLC)], "llc_mean", colour)
        # Flagged alpha values are diagnostics, not failures: they stay in the mean and are also drawn hollow
        measured = at_value[at_value[alpha_variant].notna()]
        drawn_alpha.append(draw_curve(axes[1], measured, alpha_variant, colour, markers=True))
        draw_hollow(axes[1], measured[measured["alpha_flag"] == 1], alpha_variant, colour)
        draw_curve(axes[2], at_value, "test_acc", colour)
        draw_curve(axes[3], at_value, "train_loss", colour, log=True)
    fit_limits(axes[0], drawn_llc)
    fit_limits(axes[1], drawn_alpha, include=[2.0])
    axes[1].axhline(2, color=INK, lw=1, ls="--")
    axes[3].set_yscale("log")


def add_row_legend(ax, factor, colours):
    ordered = strength_order(factor, colours)
    handles = [Line2D([], [], color=colours[v], lw=2, label=f"{v:g}") for v in ordered]
    ax.legend(
        handles=handles, title=FACTOR_LABELS[factor], loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=9, title_fontsize=9, labelcolor=INK,
    )  # fmt: skip


def epochs_note(sweep):
    """Steps are fixed, so the batch-size runs see different numbers of epochs."""
    n = sweep["subset"] or 50000
    cells = [f"{b}: {sweep['steps'] * b / (n - n % b):.1f}" for b in sweep["factors"]["batch_size"]]
    return "Same number of SGD steps for every run, so batch-size runs see different numbers of epochs (" + ", ".join(cells) + ")."


def counts_note(df):
    """Every point is counted once, across the sweeps that share the baseline."""
    points = df.drop_duplicates(["run_id", "step"])
    llc = points["llc_flag"].value_counts()
    measured = points["alpha_flag"].notna().sum()
    return (
        f"Hollow markers: LLC negative ({llc.get('negative', 0)}) or left the basin ({llc.get('left_basin', 0)}), out of the mean; "
        f"LLC diverged, no value to draw ({llc.get('diverged', 0)}); "
        f"alpha flagged ({int((points['alpha_flag'] == 1).sum())} of {measured}), in the mean."
    )


def caption(sweep, sgld, alpha_variant, df):
    sgld_text = "SGLD not calibrated" if not sgld else (
        f"SGLD lr {sgld['lr']:g}, localisation {sgld['localization']:g}, {sgld['num_steps']} steps, {sgld['burn_in']:.0%} burn-in, "
        f"{sgld['num_chains']} chains, batch {sgld['batch_size']}"
    )  # fmt: skip
    return "\n".join(
        [
            f"{sweep['arch']}, {sweep['steps']} steps, seeds: {len(sweep['seeds'])}; lines are means, bands 95% CI; darker = stronger implicit regularisation",
            f"Alpha plotted: {ALPHA_LABELS[alpha_variant]} (dashed line at 2). {sgld_text}.",
            epochs_note(sweep),
            counts_note(df),
        ]
    )


def plot_fig1(df, sweep, sgld, alpha_variant, path):
    factors = list(sweep["factors"])
    fig, axes = plt.subplots(len(factors), 4, figsize=(14.5, 2.9 * len(factors) + 1.4), sharex=True, squeeze=False, facecolor=SURFACE)
    for row, factor in enumerate(factors):
        colours = ramp_colours(factor, sweep["factors"][factor])
        draw_panels(axes[row], df[df["factor"] == factor], colours, alpha_variant)
        add_row_legend(axes[row][-1], factor, colours)
        titles = ["LLC", "Tail index alpha", "Test accuracy", "Training loss (log)"] if row == 0 else [None] * 4
        for col, ax in enumerate(axes[row]):
            style_axes(ax, titles[col], FACTOR_LABELS[factor] if col == 0 else None, "SGD step" if row == len(factors) - 1 else None)
    fig.suptitle(caption(sweep, sgld, alpha_variant, df), x=0.01, y=0.995, ha="left", va="top", fontsize=9.5, color=INK, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_alpha_variants(df, sweep, path):
    """Every alpha variant and lag-1 by step, so that the alpha panel of fig1_style.png can be judged."""
    factors, variants = list(sweep["factors"]), list(ALPHA_LABELS)
    fig, axes = plt.subplots(len(factors), len(variants) + 1, figsize=(17.5, 2.7 * len(factors) + 1), sharex=True, squeeze=False, facecolor=SURFACE)
    for row, factor in enumerate(factors):
        rows = df[df["factor"] == factor]
        colours = ramp_colours(factor, sweep["factors"][factor])
        for col, variant in enumerate(variants):
            ax = axes[row][col]
            drawn = []
            for value, colour in colours.items():
                measured = rows[(rows["value"] == value) & rows[variant].notna()]
                drawn.append(draw_curve(ax, measured, variant, colour, markers=True))
                draw_hollow(ax, measured[measured["alpha_flag"] == 1], variant, colour)
            fit_limits(ax, drawn, include=[2.0])
            ax.axhline(2, color=INK, lw=1, ls="--")
        lag_ax = axes[row][-1]
        for value, colour in colours.items():
            measured = rows[(rows["value"] == value) & rows["lag1"].notna()]
            draw_curve(lag_ax, measured, "lag1", colour, markers=True)
            draw_hollow(lag_ax, measured[measured["alpha_flag"] == 1], "lag1", colour)
        for threshold in (-0.1, 0.1):
            lag_ax.axhline(threshold, color=INK, lw=1, ls=":")
        add_row_legend(lag_ax, factor, colours)
        titles = [ALPHA_LABELS[v].capitalize() for v in variants] + ["Lag-1 of increments (signed mean)"] if row == 0 else [None] * 5
        for col, ax in enumerate(axes[row]):
            style_axes(ax, titles[col], FACTOR_LABELS[factor] if col == 0 else None, "SGD step" if row == len(factors) - 1 else None)
    fig.suptitle(
        "Alpha variants and lag-1 by step; dashed line at 2 (top of the theoretical range), dotted at +-0.1; hollow = flagged\n"
        "A checkpoint is flagged when mean |lag-1| > 0.1 or the zero fraction > 0.05. Strongly negative lag-1 inflates alpha; iterates read about 1 on slowly moving paths.",
        x=0.01, y=0.995, ha="left", va="top", fontsize=9.5, color=INK, linespacing=1.4,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_all(results_dir, alpha_variant="beta_incr_max"):
    results_dir = Path(results_dir)
    sweep, sgld = load_resolved(results_dir)
    df, missing = load_timeseries(results_dir)
    if missing:
        print(f"{len(missing)} runs have no results yet and are not drawn: {missing}")
    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_fig1(df, sweep, sgld, alpha_variant, figures / "fig1_style.png")
    plot_alpha_variants(df, sweep, figures / "alpha_variants.png")
    print(f"wrote {figures / 'fig1_style.png'} and {figures / 'alpha_variants.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--alpha-variant", choices=sorted(ALPHA_LABELS), default="beta_incr_max", help="the variant in the alpha column")
    args = parser.parse_args()
    plot_all(args.results_dir, args.alpha_variant)


if __name__ == "__main__":
    main()
