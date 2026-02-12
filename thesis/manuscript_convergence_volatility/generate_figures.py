"""Generate publication-quality figures for the convergence-volatility manuscript.

Figures:
1. Convergence time series with volatility overlay
2. Baseline-shift signature (per-bin intercept vs volatility)
3. Era comparison scatter
4. Robustness panel (LOO + controls)
5. Conceptual schematic
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Publication style.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT_DIR = Path("thesis/manuscript_convergence_volatility/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths.
MERGED = "thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/merged.csv"
LOO = "thesis/synthesis/output_robustness_battery/leave_one_out.csv"
TIMEBIN = "thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv"


def load_bins() -> pd.DataFrame:
    df = pd.read_csv(MERGED)
    df = df.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    df = df.sort_values("time_bin", ascending=False).reset_index(drop=True)
    return df


def _era_color(tb: float) -> str:
    if tb > 252:
        return "#2ca02c"  # Paleozoic green
    if tb > 66:
        return "#d62728"  # Mesozoic red
    return "#1f77b4"  # Cenozoic blue


def _era(tb: float) -> str:
    if tb > 252:
        return "Paleozoic"
    if tb > 66:
        return "Mesozoic"
    return "Cenozoic"


def figure_1_timeseries(bins: pd.DataFrame) -> None:
    """Convergence time series with volatility overlay + scatter."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), gridspec_kw={"height_ratios": [2, 1.3]})

    ax = axes[0]
    tb = bins["time_bin"]
    y = bins["functional_excess_similarity_js"]
    v = bins["delta_from_prev_T_field_meanabs"]

    ax.fill_between(tb, 0, y, where=y >= 0, alpha=0.25, color="#2ca02c", linewidth=0)
    ax.fill_between(tb, 0, y, where=y < 0, alpha=0.25, color="#d62728", linewidth=0)
    ax.plot(tb, y, "o-", color="#2ca02c", markersize=4, linewidth=1.2, label="Functional excess similarity", zorder=3)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="-")

    ax2 = ax.twinx()
    ax2.plot(tb, v, "s-", color="#d62728", markersize=3.5, linewidth=1, alpha=0.7, label="Climate volatility")
    ax2.set_ylabel("Climate volatility\n(|$\\Delta T$| field mean, °C)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax.set_ylabel("Functional excess similarity\n(JS residual)")
    ax.set_xlabel("")
    ax.set_title("(A) Functional convergence and climate volatility across the Phanerozoic")
    ax.invert_xaxis()

    # Era shading.
    ax.axvspan(540, 252, alpha=0.04, color="#2ca02c")
    ax.axvspan(252, 66, alpha=0.04, color="#d62728")
    ax.axvspan(66, 0, alpha=0.04, color="#1f77b4")
    ax.text(400, ax.get_ylim()[1] * 0.88, "Paleozoic", fontsize=8, ha="center", color="#2ca02c", alpha=0.8)
    ax.text(159, ax.get_ylim()[1] * 0.88, "Mesozoic", fontsize=8, ha="center", color="#d62728", alpha=0.8)
    ax.text(33, ax.get_ylim()[1] * 0.88, "Cz", fontsize=8, ha="center", color="#1f77b4", alpha=0.8)

    # Scatter.
    ax3 = axes[1]
    colors = [_era_color(t) for t in bins["time_bin"]]
    ax3.scatter(v, y, c=colors, s=50, alpha=0.85, edgecolors="none", zorder=3)

    # Fit line.
    mask = np.isfinite(v.to_numpy()) & np.isfinite(y.to_numpy())
    xx = v.to_numpy()[mask]
    yy = y.to_numpy()[mask]
    A = np.vstack([xx, np.ones(len(xx))]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    xline = np.linspace(float(xx.min()), float(xx.max()), 50)
    ax3.plot(xline, coef[0] * xline + coef[1], color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax3.axhline(0, color="grey", linewidth=0.5)

    ax3.set_xlabel("Climate volatility (|$\\Delta T$| field mean, °C)")
    ax3.set_ylabel("Functional excess\nsimilarity (JS)")
    ax3.set_title("(B) Bin-level scatter")

    # Legend patches.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="#2ca02c", markersize=6, linestyle="none", label="Paleozoic"),
        Line2D([0], [0], marker="o", color="#d62728", markersize=6, linestyle="none", label="Mesozoic"),
        Line2D([0], [0], marker="o", color="#1f77b4", markersize=6, linestyle="none", label="Cenozoic"),
    ]
    ax3.legend(handles=handles, loc="upper left", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_timeseries.png")
    fig.savefig(OUT_DIR / "fig1_timeseries.pdf")
    plt.close(fig)
    print("Wrote Figure 1")


def figure_2_baseline_shift(bins: pd.DataFrame) -> None:
    """Per-bin intercept of functional~taxonomic plotted against volatility."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    if "model_js_intercept" not in bins.columns:
        # Fall back to timebin metrics.
        tb = pd.read_csv(TIMEBIN)
        bins = bins.merge(tb[["time_bin", "model_js_intercept"]], on="time_bin", how="left", suffixes=("", "_tb"))
        if "model_js_intercept_tb" in bins.columns:
            bins["model_js_intercept"] = bins["model_js_intercept_tb"]

    if "model_js_intercept" not in bins.columns:
        print("Skipping Figure 2: no intercept data")
        return

    v = bins["delta_from_prev_T_field_meanabs"]
    intercept = bins["model_js_intercept"]
    colors = [_era_color(t) for t in bins["time_bin"]]

    ax.scatter(v, intercept, c=colors, s=55, alpha=0.85, edgecolors="none", zorder=3)

    mask = np.isfinite(v.to_numpy()) & np.isfinite(intercept.to_numpy())
    xx = v.to_numpy()[mask]
    yy = intercept.to_numpy()[mask]
    A = np.vstack([xx, np.ones(len(xx))]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    xline = np.linspace(float(xx.min()), float(xx.max()), 50)
    ax.plot(xline, coef[0] * xline + coef[1], color="black", linewidth=1.2, linestyle="--", alpha=0.7)

    r = float(np.corrcoef(xx, yy)[0, 1])
    ax.text(0.95, 0.05, f"r = {r:.2f}", transform=ax.transAxes, ha="right", fontsize=10)

    ax.set_xlabel("Climate volatility (|$\\Delta T$| field mean, °C)")
    ax.set_ylabel("Per-bin intercept of functional ~ taxonomic\nsimilarity regression (JS)")
    ax.set_title("The baseline-shift signature")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="#2ca02c", markersize=6, linestyle="none", label="Paleozoic"),
        Line2D([0], [0], marker="o", color="#d62728", markersize=6, linestyle="none", label="Mesozoic"),
        Line2D([0], [0], marker="o", color="#1f77b4", markersize=6, linestyle="none", label="Cenozoic"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_baseline_shift.png")
    fig.savefig(OUT_DIR / "fig2_baseline_shift.pdf")
    plt.close(fig)
    print("Wrote Figure 2")


def figure_3_era_comparison(bins: pd.DataFrame) -> None:
    """Era-split volatility-convergence scatter with per-era fits."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    era_names = ["Paleozoic", "Mesozoic", "Cenozoic"]
    era_colors = {"Paleozoic": "#2ca02c", "Mesozoic": "#d62728", "Cenozoic": "#1f77b4"}
    era_ranges = {"Paleozoic": (252, 540), "Mesozoic": (66, 252), "Cenozoic": (0, 66)}

    for i, era in enumerate(era_names):
        ax = axes[i]
        lo, hi = era_ranges[era]
        mask = (bins["time_bin"] > lo) & (bins["time_bin"] <= hi)
        sub = bins[mask]

        v = sub["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
        y = sub["functional_excess_similarity_js"].to_numpy(dtype=float)

        ax.scatter(v, y, c=era_colors[era], s=55, alpha=0.85, edgecolors="none", zorder=3)
        ax.axhline(0, color="grey", linewidth=0.5)

        valid = np.isfinite(v) & np.isfinite(y)
        if valid.sum() >= 4:
            A = np.vstack([v[valid], np.ones(valid.sum())]).T
            coef, *_ = np.linalg.lstsq(A, y[valid], rcond=None)
            xl = np.linspace(float(v[valid].min()), float(v[valid].max()), 30)
            ax.plot(xl, coef[0] * xl + coef[1], color=era_colors[era], linewidth=1.5, linestyle="--", alpha=0.8)
            r = float(np.corrcoef(v[valid], y[valid])[0, 1])
            ax.text(0.95, 0.05, f"r = {r:.2f}\nn = {valid.sum()}", transform=ax.transAxes, ha="right", fontsize=9)

        ax.set_xlabel("Climate volatility")
        if i == 0:
            ax.set_ylabel("Functional excess similarity (JS)")
        ax.set_title(era)

    fig.suptitle("Volatility-convergence relationship by era", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_era_comparison.png")
    fig.savefig(OUT_DIR / "fig3_era_comparison.pdf")
    plt.close(fig)
    print("Wrote Figure 3")


def figure_4_robustness(bins: pd.DataFrame) -> None:
    """Robustness panel: LOO + block bootstrap summary."""
    loo = pd.read_csv(LOO)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # LOO.
    ax = axes[0]
    colors = [_era_color(tb) for tb in loo["dropped_bin"]]
    ax.bar(loo["dropped_bin"], loo["partial_corr"], width=7, color=colors, alpha=0.8)
    ax.axhline(0.380, color="red", linewidth=1.2, linestyle="--", label="Full sample r = 0.38")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Dropped bin (Ma)")
    ax.set_ylabel("Partial correlation")
    ax.set_title("(A) Leave-one-out stability")
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    # Summary of inference methods.
    ax = axes[1]
    methods = ["Circular\nshift", "Block\nboot\n(b=2)", "Block\nboot\n(b=3)", "Block\nboot\n(b=5)",
               "OLS +\nHAC", "SARIMAX\nAR(0)"]
    pvals = [0.050, 0.020, 0.021, 0.029, 0.037, 0.079]
    betas = [0.380, None, None, None, 0.013, 0.012]  # corr or beta as appropriate
    colors_bar = ["#1f77b4"] * 6
    bars = ax.bar(range(len(methods)), pvals, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0.05, color="red", linewidth=1, linestyle="--", label="p = 0.05")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("p-value")
    ax.set_title("(B) Inference methods comparison")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.12)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_robustness.png")
    fig.savefig(OUT_DIR / "fig4_robustness.pdf")
    plt.close(fig)
    print("Wrote Figure 4")


def figure_5_schematic() -> None:
    """Conceptual schematic of the synchronising-filter hypothesis."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Stable climate panel.
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Stable climate", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Two "provinces" with distinct role distributions.
    # Province A.
    roles_a = {"Filter\nfeeder": 4, "Predator": 3, "Grazer": 2, "Burrower": 1, "Scavenger": 1.5, "Chemosymbiont": 1}
    roles_b = {"Filter\nfeeder": 1, "Predator": 2, "Grazer": 4, "Burrower": 3, "Parasite": 2, "Browser": 1.5}

    # Bar representation.
    y_start = 1.5
    bw = 0.6
    for i, (role, val) in enumerate(roles_a.items()):
        ax.barh(y_start + i * 0.9, val, height=0.7, left=0.5, color="#2ca02c", alpha=0.6, edgecolor="black", linewidth=0.4)
        ax.text(0.3, y_start + i * 0.9, role, ha="right", va="center", fontsize=6.5)
    ax.text(3, 7.5, "Province A", fontsize=10, ha="center", fontweight="bold", color="#2ca02c")

    for i, (role, val) in enumerate(roles_b.items()):
        ax.barh(y_start + i * 0.9, val, height=0.7, left=5.5, color="#1f77b4", alpha=0.6, edgecolor="black", linewidth=0.4)
        ax.text(5.3, y_start + i * 0.9, role, ha="right", va="center", fontsize=6.5)
    ax.text(7.5, 7.5, "Province B", fontsize=10, ha="center", fontweight="bold", color="#1f77b4")

    ax.text(5, 9, "Different roles, different abundances", ha="center", fontsize=9, style="italic")
    ax.text(5, 0.3, "Low functional similarity", ha="center", fontsize=10, fontweight="bold", color="#666666")

    # Volatile climate panel.
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Volatile climate", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Both converge on similar role distribution.
    roles_conv = {"Filter\nfeeder": 4, "Predator": 2, "Grazer": 2.5}
    for i, (role, val) in enumerate(roles_conv.items()):
        ax.barh(y_start + 1.5 + i * 1.2, val, height=0.8, left=0.5, color="#2ca02c", alpha=0.6, edgecolor="black", linewidth=0.4)
        ax.barh(y_start + 1.5 + i * 1.2 - 0.1, val * 0.95, height=0.8, left=5.5, color="#1f77b4", alpha=0.6, edgecolor="black", linewidth=0.4)
        ax.text(0.3, y_start + 1.5 + i * 1.2, role, ha="right", va="center", fontsize=7)
        ax.text(5.3, y_start + 1.5 + i * 1.2, role, ha="right", va="center", fontsize=7)

    ax.text(3, 7.5, "Province A", fontsize=10, ha="center", fontweight="bold", color="#2ca02c")
    ax.text(7.5, 7.5, "Province B", fontsize=10, ha="center", fontweight="bold", color="#1f77b4")

    ax.text(5, 9, "Same roles, similar abundances", ha="center", fontsize=9, style="italic")
    ax.text(5, 0.3, "High functional similarity\n(despite different species)", ha="center", fontsize=10, fontweight="bold", color="#d62728")

    # Arrow showing filtering.
    ax.annotate("Environmental\nfiltering", xy=(5, 8), xytext=(5, 8.7),
                fontsize=8, ha="center", color="#d62728",
                arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.5})

    fig.suptitle("The synchronising-filter hypothesis", fontsize=13, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_schematic.png")
    fig.savefig(OUT_DIR / "fig5_schematic.pdf")
    plt.close(fig)
    print("Wrote Figure 5")


def main() -> None:
    bins = load_bins()
    figure_1_timeseries(bins)
    figure_2_baseline_shift(bins)
    figure_3_era_comparison(bins)
    figure_4_robustness(bins)
    figure_5_schematic()
    print(f"\nAll figures written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
