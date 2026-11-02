"""Plots and results/report.md from results/summary.csv. The text of the report is computed from the numbers.

Run with: uv run python scripts/analyse.py
"""

import argparse
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import yaml
from scipy import stats

# +1 when a larger value means stronger regularisation, as in the hypothesis
STRONGER = {"lr": 1, "batch_size": -1, "momentum": 1, "l2": 1}
METRICS = {
    "llc_mean": "LLC",
    "beta_iter_max": "beta, iterates, max over layers",
    "beta_iter_median": "beta, iterates, median over layers",
    "beta_incr_max": "beta, increments, max over layers",
    "beta_incr_median": "beta, increments, median over layers",
    "test_acc": "Test accuracy",
    "gap_acc": "Accuracy gap",
    "gap_loss": "Loss gap",
}
# Direction the hypothesis predicts as regularisation gets stronger; the gaps have no stated prediction
PREDICTED = {"llc_mean": -1, "beta_iter_max": -1, "beta_iter_median": -1, "beta_incr_max": -1, "beta_incr_median": -1, "test_acc": 1}
BETAS = ["beta_iter_max", "beta_iter_median", "beta_incr_max", "beta_incr_median"]
PAIRS = [("llc_mean", "gap_acc")] + [(b, "gap_acc") for b in BETAS] + [("llc_mean", b) for b in BETAS]

MIN_VERDICT_N = 8
MAX_EXACT_N = 20  # 20! is the largest factorial that fits in the int64 counts of pairing_counts
TOO_FEW = "too few runs"

BLUE = "#2a78d6"
FACTOR_STYLE = {"lr": ("#2a78d6", "o"), "batch_size": ("#eb6834", "s"), "momentum": ("#1baf7a", "^"), "l2": ("#eda100", "D"), "baseline": ("#52514e", "P")}
INK, MUTED, GRID, AXIS, SURFACE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def factor_rows(df, baseline, factor):
    """Runs where only this factor differs from the baseline, which includes the baseline itself."""
    mask = pd.Series(True, index=df.index)
    for name, value in baseline.items():
        if name != factor:
            mask &= np.isclose(df[name], value)
    return df[mask]


def run_factor(row, baseline):
    for factor, value in baseline.items():
        if not np.isclose(row[factor], value):
            return factor
    return "baseline"


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


def integer_ranks(values):
    """Midranks scaled to coprime integers, so that sums of their products are exact."""
    ranks = (2 * stats.rankdata(values)).astype(np.int64)
    return ranks // np.gcd.reduce(ranks)


@lru_cache(maxsize=None)
def pairing_counts(a, b):
    """Exact distribution of sum(a[i] * b[perm[i]]) over the n! permutations, as (lowest sum, counts).

    Listing 20! permutations is out of reach, but the sum so far depends only on which values of b
    are used up, so one distribution per such set is enough: 2^n sets without ties, fewer with them.
    """
    values, multiplicity = np.unique(b, return_counts=True)
    values, radices = values.tolist(), (multiplicity + 1).tolist()
    weights = [math.prod(radices[:j]) for j in range(len(radices))]
    layer = {sum(w * (r - 1) for w, r in zip(weights, radices)): np.ones(1, dtype=np.int64)}
    # Centring a and taking small |a_i| first keeps the partial sums narrow, which at n = 20 decides the memory
    centre = a[len(a) // 2]
    lowest = centre * sum(b)
    for a_i in sorted((v - centre for v in a), key=abs):
        anchor = values[0] if a_i >= 0 else values[-1]
        width = len(next(iter(layer.values()))) + abs(a_i) * (values[-1] - values[0])
        following = defaultdict(lambda: np.zeros(width, dtype=np.int64))
        while layer:
            state, counts = layer.popitem()
            for value, weight, radix in zip(values, weights, radices):
                left = state // weight % radix
                if left:
                    shift = a_i * (value - anchor)
                    following[state - weight][shift : shift + len(counts)] += left * counts
        layer = following
        lowest += a_i * anchor
    return lowest, layer[0]


def exact_spearman_p(x, y):
    """Two-sided p: the share of the n! pairings of x and y with |rho| at least the observed one."""
    a, b = integer_ranks(x), integer_ranks(y)
    if len(np.unique(a)) < len(np.unique(b)):
        a, b = b, a  # ties in b mean fewer states
    n = len(a)
    lowest, counts = pairing_counts(tuple(sorted(a.tolist())), tuple(sorted(b.tolist())))
    # n * (sum - its mean) is proportional to rho and is an integer, so the comparison has no rounding
    centred = n * (lowest + np.arange(len(counts))) - int(a.sum()) * int(b.sum())
    extreme = np.abs(centred) >= abs(n * int(a @ b) - int(a.sum()) * int(b.sum()))
    return int(counts[extreme].sum()) / math.factorial(n)


def spearman(x, y):
    """Returns (rho, p, n) on the pairs where both values are finite. p is NaN when n < MIN_VERDICT_N."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan"), n
    result = stats.spearmanr(x, y)
    if n < MIN_VERDICT_N:
        p = float("nan")
    elif n <= MAX_EXACT_N:
        p = exact_spearman_p(x, y)
    else:
        p = float(result.pvalue)
    return float(result.statistic), p, n


def fmt(value, digits=3):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def fmt_p(p, n):
    """Never prints 0. Above MAX_EXACT_N there are no permutations, and the t tail underflows to 0 at |rho| = 1."""
    if n < MIN_VERDICT_N:
        return TOO_FEW
    if p == 0:
        return f"< {fmt(1 / math.factorial(n) if n <= MAX_EXACT_N else sys.float_info.min)}"
    return fmt(p)


def rho_text(rho, n):
    return f"rho {fmt(rho, 2)}, n {n}" + (f", {TOO_FEW}" if n < MIN_VERDICT_N else "")


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def draw_levels(ax, rows, factor, metric):
    """Seeds as hollow dots, their mean as a filled dot, and its 95% CI as a bar, at each level of factor."""
    levels = sorted(rows[factor].unique())
    for i, level in enumerate(levels):
        values = rows.loc[rows[factor] == level, metric].dropna().to_numpy(dtype=float)
        mean, half_width, _ = mean_ci(values)
        ax.scatter(np.full(len(values), i), values, s=18, facecolors="none", edgecolors=MUTED, linewidths=0.8, zorder=2)
        ax.errorbar(i, mean, yerr=0 if np.isnan(half_width) else half_width, color=BLUE, marker="o", ms=7, capsize=4, lw=1.5, zorder=3)
    ax.set_xticks(range(len(levels)), [f"{level:g}\n(n={int(rows.loc[rows[factor] == level, metric].notna().sum())})" for level in levels])
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    style_axes(ax)


def plot_factor(rows, factor, path):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), facecolor=SURFACE)
    for ax, (metric, label) in zip(axes.flat, METRICS.items()):
        draw_levels(ax, rows, factor, metric)
        ax.set_title(label, loc="left", fontsize=9, color=INK)
    fig.suptitle(f"Effect of {factor}: mean and 95% CI over seeds (hollow dots = single seeds)", x=0.01, ha="left", color=INK)
    fig.supxlabel(factor, color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


GRID_METRICS = {"llc_mean": "LLC", "test_acc": "Test accuracy", "train_loss": "Train loss", "beta_incr_max": "alpha (beta, increments, max)"}


def plot_grid(df, baseline, factors, path):
    """Rows are factors, columns are LLC, test accuracy, train loss and alpha. Each row varies one factor only."""
    fig, axes = plt.subplots(len(factors), len(GRID_METRICS), figsize=(13, 2.6 * len(factors) + 0.8), squeeze=False, facecolor=SURFACE)
    for row, factor in enumerate(factors):
        part = factor_rows(df, baseline, factor)
        for col, (metric, label) in enumerate(GRID_METRICS.items()):
            ax = axes[row][col]
            draw_levels(ax, part, factor, metric)
            if row == 0:
                ax.set_title(label, loc="left", fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(factor, color=INK, fontsize=9)
    fig.suptitle("One factor at a time: mean and 95% CI over seeds (hollow dots = single seeds)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatters(df, baseline, path):
    labels = df.apply(run_factor, axis=1, baseline=baseline)
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.5), facecolor=SURFACE)
    for ax, (x_col, y_col) in zip(axes.flat, PAIRS):
        for factor, (color, marker) in FACTOR_STYLE.items():
            for flagged in (False, True):
                part = df[(labels == factor) & (df["flagged"] == flagged)]
                ax.scatter(
                    part[x_col],
                    part[y_col],
                    s=36,
                    marker=marker,
                    facecolors="none" if flagged else color,
                    edgecolors=color if flagged else SURFACE,
                    linewidths=1.2 if flagged else 0.8,
                )
        unflagged = df[~df["flagged"]]
        rho, _, n = spearman(df[x_col], df[y_col])
        rho_u, _, n_u = spearman(unflagged[x_col], unflagged[y_col])
        ax.set_title(f"all: {rho_text(rho, n)}\nunflagged: {rho_text(rho_u, n_u)}", loc="left", fontsize=8, color=INK)
        ax.set_xlabel(METRICS[x_col], fontsize=8, color=INK)
        ax.set_ylabel(METRICS[y_col], fontsize=8, color=INK)
        style_axes(ax)
    handles = [plt.Line2D([], [], ls="", marker=marker, color=color, label=factor) for factor, (color, marker) in FACTOR_STYLE.items()]
    handles.append(plt.Line2D([], [], ls="", marker="o", markerfacecolor="none", color=MUTED, label="flagged (hollow)"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, labelcolor=INK, fontsize=8)
    fig.suptitle("Across all runs", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def describe(rho, p, n):
    """Plain description of one correlation. The threshold is the conventional one, nothing is tuned to the data."""
    if n < MIN_VERDICT_N:
        return f"{TOO_FEW}, rho = {fmt(rho, 2)}, n = {n}"
    if not np.isfinite(rho):
        return f"not computable (n = {n})"
    p_text = fmt_p(p, n)
    p_text = p_text if p_text.startswith("<") else f"= {p_text}"
    finding = "monotonic relationship" if p < 0.05 else "no detectable monotonic relationship"
    return f"{finding}, rho = {rho:.2f}, p {p_text}, n = {n}"


def trend_section(df, baseline, factors):
    rows, verdicts = [], []
    for factor in factors:
        part = factor_rows(df, baseline, factor)
        strength = STRONGER[factor] * part[factor]
        for metric, predicted in PREDICTED.items():
            rho, p, n = spearman(strength, part[metric])
            if n < MIN_VERDICT_N:
                verdict = TOO_FEW
            elif not np.isfinite(rho):
                verdict = "not computable"
            elif p >= 0.05:
                verdict = "no clear trend"
            elif np.sign(rho) == predicted:
                verdict = "as predicted"
            else:
                verdict = "opposite to prediction"
            verdicts.append(verdict)
            rows.append([factor, METRICS[metric], fmt(rho, 2), fmt_p(p, n), n, verdict])
    table = md_table(["factor", "metric", "rho with reg. strength", "p", "n", "verdict"], rows)
    counts = ", ".join(f"{verdicts.count(v)} {v}" for v in dict.fromkeys(verdicts))
    return f"Spearman rho between regularisation strength and each metric, over runs. Of {len(verdicts)} tests: {counts}.\n\n{table}"


def correlation_section(df, baseline, factors):
    scopes = {"overall": df} | {factor: factor_rows(df, baseline, factor) for factor in factors}
    rows = []
    for (x_col, y_col) in PAIRS:
        for scope, part in scopes.items():
            rho, p, n = spearman(part[x_col], part[y_col])
            unflagged = part[~part["flagged"]]
            rho_u, p_u, n_u = spearman(unflagged[x_col], unflagged[y_col])
            rows.append([f"{x_col} vs {y_col}", scope, fmt(rho, 2), fmt_p(p, n), n, fmt(rho_u, 2), fmt_p(p_u, n_u), n_u])
    table = md_table(["pair", "scope", "rho", "p", "n", "rho unflagged", "p unflagged", "n unflagged"], rows)
    lines = [f"- {x_col} vs {y_col}, all runs: {describe(*spearman(df[x_col], df[y_col]))}." for x_col, y_col in PAIRS]
    return "\n".join(lines) + "\n\n" + table


def factor_section(df, baseline, factors):
    parts = []
    for factor in factors:
        part = factor_rows(df, baseline, factor)
        rows = []
        for level in sorted(part[factor].unique()):
            at_level = part[part[factor] == level]
            cells = [f"{level:g}", len(at_level), int(at_level["flagged"].sum())]
            for metric in METRICS:
                mean, half_width, _ = mean_ci(at_level[metric])
                cells.append(f"{fmt(mean)} ± {fmt(half_width, 2)}")
            rows.append(cells)
        table = md_table([factor, "runs", "flagged", *METRICS], rows)
        parts.append(f"### {factor}\n\n![{factor}](figures/factor_{factor}.png)\n\n{table}")
    return "\n\n".join(parts)


def analyse(results_dir):
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "summary.csv")
    resolved = yaml.safe_load((results_dir / "sweep_resolved.yaml").read_text())
    baseline, factors = resolved["sweep"]["baseline"], list(resolved["sweep"]["factors"])
    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)

    for factor in factors:
        plot_factor(factor_rows(df, baseline, factor), factor, figures / f"factor_{factor}.png")
    plot_scatters(df, baseline, figures / "scatter.png")
    plot_grid(df, baseline, factors, figures / "grid.png")

    outside = {b: int(((df[b] <= 0) | (df[b] > 2)).sum()) for b in BETAS}
    report = f"""# VGG sweep: LLC, tail index and generalisation

Hypothesis tested, not assumed: stronger regularisation gives higher test accuracy, lower LLC and lower beta.
All text and numbers below are computed from `summary.csv`. p-values are two-sided and unadjusted for the number of tests; every test that was run is listed.
Below n = {MIN_VERDICT_N} only rho is given, marked "{TOO_FEW}", with no p-value or verdict. For n = {MIN_VERDICT_N} to {MAX_EXACT_N}, p is an exact permutation test over all n! pairings. Above that it is the t-approximation.

## Runs

- Runs: {len(df)}, over {df['config'].nunique()} configs.
- Training diverged: {int(df['train_diverged'].sum())}. SGLD diverged: {int(df['sgld_diverged'].sum())}.
- Flagged by the increment diagnostics (mean |lag-1| > 0.1 or zero fraction > 0.05, or undefined): {int(df['flagged'].sum())}.
- Negative LLC: {int((df['llc_mean'] < 0).sum())}. These are kept; AW removed theirs by hand.
- Beta outside the theoretical range (0, 2]: {', '.join(f'{b} {count}' for b, count in outside.items())}. HD also report values above 2 and treat them as an estimator artifact.
- SGLD settings, identical for every run: `{resolved['sgld']}`.

## Does the hypothesis hold?

{trend_section(df, baseline, factors)}

## Each factor

{factor_section(df, baseline, factors)}

## Correlations across runs

![grid](figures/grid.png)

![scatter](figures/scatter.png)

{correlation_section(df, baseline, factors)}

## Limits of this analysis

- Window length N depends on batch size ({', '.join(f'batch {int(b)}: N = {int(w)}' for b, w in df.dropna(subset=['window']).groupby('batch_size')['window'].first().items())}), so beta across batch sizes also compares different sample sizes.
- The number of seeds per config is small ({df.groupby('config').size().min()} to {df.groupby('config').size().max()}): t intervals are wide, and within-factor correlations have little power, so "no clear trend" is not evidence of no effect.
- Runs at the same factor level share a value, so rho within a factor has many ties.
- LLC values are comparable only within this one sampler setting, and have no absolute calibration on this model.
- Under L2 the weights are not near a critical point of the plain cross-entropy that SGLD samples, which can shift the LLC of those runs.
- The max over layers of noisy estimates is biased upward, and with VGG4 it is taken over only 2 layers. Both the max and the median are reported for this reason.
"""
    (results_dir / "report.md").write_text(report)
    print(f"wrote {results_dir / 'report.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    analyse(parser.parse_args().results_dir)


if __name__ == "__main__":
    main()
