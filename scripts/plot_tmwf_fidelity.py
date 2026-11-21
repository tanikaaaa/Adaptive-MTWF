"""Render the fidelity audit figures as publication-quality vector PDFs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_mtwf.data.fidelity import CANDIDATE_LENGTHS

REAL_COLOR = "#20272d"
SYNTH_COLOR = "#007e80"
STYLES = {
    "ratio_0.10": ("Ratio 10%", "#1263ad", "-"),
    "ratio_0.30": ("Ratio 30%", SYNTH_COLOR, "--"),
    "ratio_0.50": ("Ratio 50%", "#9e415c", "-."),
    "delay": ("Delay", "#b66b16", ":"),
}


def _ecdf(axis, rows, metric, *, scale, color, label, linestyle="-"):
    values = np.array([row.scalars[metric] for row in rows])
    values = np.sort(values[np.isfinite(values)]) * scale
    cdf = np.arange(1, len(values) + 1) / len(values)
    axis.step(values, cdf, where="post", color=color, label=label, linestyle=linestyle, linewidth=1.6)


def _distributions(path, cohorts, samples):
    features = (
        ("cells", "Trace length (thousand cells)", 0.001),
        ("duration_s", "Duration (s)", 1),
        ("positive_iat_p95_s", "Positive IAT p95 (ms)", 1000),
        ("burst_mean_cells", "Mean burst length (cells)", 1),
    )
    figure, axes = plt.subplots(4, 3, figsize=(10.8, 9.3), constrained_layout=True, sharey=True)
    for row, (metric, label, scale) in enumerate(features):
        for col, tabs in enumerate((2, 3, 4)):
            axis = axes[row, col]
            _ecdf(axis, cohorts[f"{tabs}tabs"], metric, scale=scale, color=REAL_COLOR, label="Real TBB")
            for name, (condition, color, linestyle) in STYLES.items():
                _ecdf(axis, samples[tabs][name], metric, scale=scale, color=color, label=condition, linestyle=linestyle)
            if row == 0:
                axis.set_title(f"{tabs} tabs", fontweight="bold")
            axis.set_xlabel(label)
            axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
            if col == 0:
                axis.set_ylabel("Empirical CDF")
            if row >= 2:
                axis.set_xscale("log")
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=7.5)
    figure.suptitle("Trace-level distributions in matched-label cohorts", fontsize=15, fontweight="bold")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _coverage(path, data):
    figure, axes = plt.subplots(2, 2, figsize=(9, 5.9), constrained_layout=True, sharex=True, sharey=True)
    x = np.array(CANDIDATE_LENGTHS) / 1000
    for tabs, axis in enumerate(axes.ravel(), start=2):
        synthetic = data["synthetic"][str(tabs)]["ratio_0.10"]["candidates"]
        axis.plot(x, [item["fully_retained_pct"] for item in synthetic.values()], "o--", color=SYNTH_COLOR,
                  label="Synthetic, all schedules", linewidth=1.65, markersize=4)
        if tabs < 5:
            real = data["real"][f"{tabs}tabs"]["candidates"]
            axis.plot(x, [item["fully_retained_pct"] for item in real.values()], "o-", color=REAL_COLOR,
                      label="Real TBB", linewidth=1.65, markersize=4)
        if tabs == 2:
            extra = data["real"]["2tabs-extra"]["candidates"]
            axis.plot(x, [item["fully_retained_pct"] for item in extra.values()], "^:", color="#755499",
                      label="Real TBB extra", linewidth=1.65, markersize=4)
            axis.legend(frameon=False, loc="lower right", fontsize=8)
        if tabs == 5:
            axis.text(0.03, 0.92, "Real TBB unavailable", transform=axis.transAxes, fontsize=8, va="top")
        axis.set_title(f"{tabs} tabs", fontweight="bold")
        axis.set_ylim(-2, 104)
        axis.set_xticks(x)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.grid(axis="y", alpha=0.2)
    figure.supxlabel("Candidate window (thousand cells)")
    figure.supylabel("Fully retained traces (%)")
    figure.suptitle("Sequence-length coverage", fontsize=15, fontweight="bold")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _heatmap(path, conditions):
    metrics = (
        ("cells", "Cells"), ("duration_s", "Duration"), ("cells_per_s", "Cells per second"),
        ("outgoing_fraction", "Outgoing fraction"), ("direction_change_rate", "Direction changes"),
        ("zero_iat_fraction", "Zero IAT fraction"), ("positive_iat_median_s", "Positive IAT median"),
        ("iat_p95_s", "IAT p95"), ("burst_mean_cells", "Mean burst length"), ("burst_p95_cells", "Burst length p95"),
    )
    names = ["ratio_0.10", "ratio_0.20", "ratio_0.30", "ratio_0.40", "ratio_0.50", "delay"]
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 4.8), constrained_layout=True, sharey=True)
    for index, (axis, tabs) in enumerate(zip(axes, (2, 3, 4))):
        values = np.array([[conditions[str(tabs)][name]["comparisons"][f"{tabs}tabs"]["scalars"][metric]["ks"]
                            for name in names] for metric, _ in metrics])
        mesh = axis.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for (row, col), value in np.ndenumerate(values):
            axis.text(col, row, f"{value:.2f}", va="center", ha="center", fontsize=7,
                      color="white" if value < 0.48 else "#121212")
        axis.set_title(f"{tabs} tabs", fontweight="bold")
        axis.set_xticks(range(len(names)), ["10%", "20%", "30%", "40%", "50%", "Delay"], rotation=35, ha="right")
        if index == 0:
            axis.set_yticks(range(len(metrics)), [label for _, label in metrics])
        axis.tick_params(length=0)
    figure.colorbar(mesh, ax=axes, location="right", shrink=0.84, label="KS statistic")
    figure.suptitle("Synthetic-to-real distribution distances", fontsize=15, fontweight="bold")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def make_figures(output, cohorts, samples, conditions, coverage):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        _distributions(output / "tmwf_fidelity_distributions.pdf", cohorts, samples)
        _coverage(output / "tmwf_sequence_length_coverage.pdf", coverage)
        _heatmap(output / "tmwf_fidelity_distances.pdf", conditions)
