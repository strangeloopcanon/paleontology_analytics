"""Generate blog-ready charts from the convergence analysis outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "blog_assets"
OUT.mkdir(exist_ok=True)

# Style: clean, blog-friendly, readable at small sizes.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#444",
    "xtick.color": "#444",
    "ytick.color": "#444",
    "axes.labelcolor": "#222",
    "axes.titlecolor": "#111",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

# Era colors (consistent across all charts).
COLOR_PALEO = "#2E7D5B"   # green
COLOR_MESO = "#C0392B"    # red
COLOR_CENO = "#1F5F8B"    # blue


def era_color(time_bin: float) -> str:
    if time_bin > 252:
        return COLOR_PALEO
    if time_bin > 66:
        return COLOR_MESO
    return COLOR_CENO


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

bins = pd.read_csv(
    ROOT / "thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/merged.csv"
)
bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)
bins["color"] = bins["time_bin"].apply(era_color)


# ---------------------------------------------------------------------------
# 1. HERO: Functional convergence across 540 million years
# ---------------------------------------------------------------------------

def make_hero():
    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Plot the time series as connected scatter
    ax.plot(bins["time_bin"], bins["functional_excess_similarity_js"],
            color="#888", linewidth=1.0, alpha=0.5, zorder=1)

    # Era backgrounds (subtle)
    ax.axvspan(540, 252, alpha=0.05, color=COLOR_PALEO, zorder=0)
    ax.axvspan(252, 66, alpha=0.05, color=COLOR_MESO, zorder=0)
    ax.axvspan(66, 0, alpha=0.05, color=COLOR_CENO, zorder=0)

    # Era labels (placed near top, not in PT annotation zone)
    ax.text(395, -0.14, "PALEOZOIC", color=COLOR_PALEO, fontsize=11, fontweight="bold",
            ha="center", va="bottom", alpha=0.9)
    ax.text(159, -0.14, "MESOZOIC", color=COLOR_MESO, fontsize=11, fontweight="bold",
            ha="center", va="bottom", alpha=0.9)
    ax.text(33, -0.14, "CENOZOIC", color=COLOR_CENO, fontsize=11, fontweight="bold",
            ha="center", va="bottom", alpha=0.9)

    # Scatter with era coloring, sized by volatility
    sizes = 30 + 60 * (bins["delta_from_prev_T_field_meanabs"] / bins["delta_from_prev_T_field_meanabs"].max())
    ax.scatter(bins["time_bin"], bins["functional_excess_similarity_js"],
               c=bins["color"], s=sizes, alpha=0.85, edgecolors="white", linewidths=1.2, zorder=3)

    # Highlight Permian-Triassic boundary
    pt_row = bins[bins["time_bin"] == 250].iloc[0]
    ax.annotate("Permian–Triassic\nextinction\n(96% of marine\nspecies died)",
                xy=(250, pt_row["functional_excess_similarity_js"]),
                xytext=(180, 0.155),
                fontsize=9, color="#222",
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))

    # Zero line
    ax.axhline(0, color="#999", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)

    # Era boundary lines
    ax.axvline(252, color="#333", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(66, color="#333", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.invert_xaxis()
    ax.set_xlabel("Million years ago", fontsize=11)
    ax.set_ylabel("Functional excess similarity\n(higher = ecosystems more alike)", fontsize=11)
    ax.set_title("540 million years of marine ecosystem convergence",
                 fontsize=15, pad=15, loc="left")

    ax.set_xlim(550, -10)
    ax.set_ylim(-0.16, 0.18)

    # Subtitle line
    fig.text(0.04, 0.005,
             "Each dot = one 10-million-year time bin. Marker size = climate volatility (|ΔT| between snapshots). "
             "Higher values mean distant ocean regions had more similar ecological 'job portfolios' than expected from shared species.",
             fontsize=8.5, color="#555", style="italic", wrap=True)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "01_hero_540myr_convergence.png", dpi=220)
    plt.close(fig)
    print("Wrote 01_hero_540myr_convergence.png")


# ---------------------------------------------------------------------------
# 2. The headline correlation: volatility vs convergence, by era
# ---------------------------------------------------------------------------

def make_volatility_scatter():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for era_name, color in [("Paleozoic", COLOR_PALEO),
                             ("Mesozoic", COLOR_MESO),
                             ("Cenozoic", COLOR_CENO)]:
        sub = bins[bins["color"] == color]
        ax.scatter(sub["delta_from_prev_T_field_meanabs"],
                   sub["functional_excess_similarity_js"],
                   c=color, s=80, alpha=0.85, edgecolors="white", linewidths=1.2,
                   label=era_name, zorder=3)

    # Highlight Permian-Triassic
    pt_row = bins[bins["time_bin"] == 250].iloc[0]
    ax.scatter([pt_row["delta_from_prev_T_field_meanabs"]],
               [pt_row["functional_excess_similarity_js"]],
               s=240, facecolors="none", edgecolors="#000", linewidths=2, zorder=4)
    ax.annotate("Permian–Triassic\nboundary",
                xy=(pt_row["delta_from_prev_T_field_meanabs"], pt_row["functional_excess_similarity_js"]),
                xytext=(4.4, 0.025),
                fontsize=9, color="#222",
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))

    # Trend line for Mesozoic (where the signal lives)
    meso = bins[bins["color"] == COLOR_MESO]
    z = np.polyfit(meso["delta_from_prev_T_field_meanabs"], meso["functional_excess_similarity_js"], 1)
    xs = np.linspace(meso["delta_from_prev_T_field_meanabs"].min(),
                     meso["delta_from_prev_T_field_meanabs"].max(), 50)
    ax.plot(xs, z[0] * xs + z[1], color=COLOR_MESO, linestyle="--", linewidth=1.5, alpha=0.7,
            label=f"Mesozoic trend (r = 0.53)")

    ax.axhline(0, color="#999", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Climate volatility (|ΔT| between 10 Myr snapshots, °C)", fontsize=11)
    ax.set_ylabel("Functional excess similarity", fontsize=11)
    ax.set_title("Volatility drives convergence — but only in the Mesozoic",
                 fontsize=14, pad=12, loc="left")

    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "02_volatility_vs_convergence_by_era.png", dpi=220)
    plt.close(fig)
    print("Wrote 02_volatility_vs_convergence_by_era.png")


# ---------------------------------------------------------------------------
# 3. The functional fingerprint: which roles expand vs contract
# ---------------------------------------------------------------------------

def make_fingerprint():
    with open(ROOT / "thesis/synthesis/output_functional_fingerprint/analysis_results.json") as f:
        fp = json.load(f)

    # The JSON has top_expanding sorted descending (largest +ve first).
    # top_contracting is sorted ascending of the bottom 10 (so most-negative is LAST).
    expanding = list(fp["top_expanding_roles"].items())[:7]  # top 7 expanding
    contracting_all = list(fp["top_contracting_roles"].items())
    contracting = contracting_all[-7:]  # last 7 = most-contracting
    contracting = list(reversed(contracting))  # most-contracting first in display

    # Combine: expanding at top, then contracting (most negative at bottom)
    items = expanding + contracting
    labels = [k.replace("|", " · ") for k, _ in items]
    diffs = [v["diff"] for _, v in items]
    colors = [COLOR_MESO if d > 0 else COLOR_CENO for d in diffs]

    fig, ax = plt.subplots(figsize=(11, 7.5))
    y_pos = np.arange(len(items))
    ax.barh(y_pos, diffs, color=colors, alpha=0.85, edgecolor="white", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", linewidth=1)

    # Add value labels
    for i, d in enumerate(diffs):
        offset = 0.0015 if d > 0 else -0.0015
        ha = "left" if d > 0 else "right"
        ax.text(d + offset, i, f"{d:+.3f}", va="center", ha=ha, fontsize=8.5, color="#444")

    ax.set_xlabel("Change in representation (high-volatility bins minus low-volatility bins)", fontsize=10)
    ax.set_title("Under volatile climates: filter feeders expand, mobile predators contract",
                 fontsize=14, pad=14, loc="left")

    # Set x limits with padding
    ax.set_xlim(min(diffs) - 0.015, max(diffs) + 0.015)

    # Legend
    legend_elements = [
        Patch(facecolor=COLOR_MESO, alpha=0.85, label="Expand under volatility"),
        Patch(facecolor=COLOR_CENO, alpha=0.85, label="Contract under volatility"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95, fontsize=10)

    fig.text(0.04, 0.005,
             "All top expanding roles are suspension feeders. The largest contracting role is fast-moving carnivores. "
             "Each row is one PBDB ecological role: diet · motility · life habit.",
             fontsize=8.5, color="#555", style="italic")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "03_functional_fingerprint.png", dpi=220)
    plt.close(fig)
    print("Wrote 03_functional_fingerprint.png")


# ---------------------------------------------------------------------------
# 4. Climate vs paleogeography: it's the climate, not the continents
# ---------------------------------------------------------------------------

def make_exposure_portfolio():
    with open(ROOT / "thesis/synthesis/output_exposure_portfolio/analysis_results.json") as f:
        ep = json.load(f)

    # Order: climate exposures first, paleogeography last
    order = [
        ("Field-mean |ΔT|", "field_mean_dT", "climate"),
        ("Global |ΔT|", "global_dT", "climate"),
        ("Global |ΔP|", "global_dP", "climate"),
        ("Land-area change", "land_area_change", "geography"),
        ("Coastline change", "coastline_change", "geography"),
    ]

    labels = [o[0] for o in order]
    rs = [ep[o[1]]["corr"] for o in order]
    ps = [ep[o[1]]["p_shift"] for o in order]
    colors = [COLOR_MESO if o[2] == "climate" else "#888" for o in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(order))
    bars = ax.barh(y_pos, rs, color=colors, alpha=0.85, edgecolor="white", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", linewidth=1)

    # Add r and p labels on each bar
    for i, (r, p) in enumerate(zip(rs, ps)):
        ax.text(r + 0.01, i, f" r = {r:.2f}, p = {p:.2g}",
                va="center", fontsize=9.5, color="#222")

    ax.set_xlabel("Partial correlation with functional excess similarity", fontsize=10)
    ax.set_xlim(-0.05, 0.55)
    ax.set_title("It's the climate, not the continents",
                 fontsize=14, pad=12, loc="left")

    legend_elements = [
        Patch(facecolor=COLOR_MESO, alpha=0.85, label="Climate variables"),
        Patch(facecolor="#888", alpha=0.85, label="Paleogeography (continental rearrangement)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95, fontsize=10)

    ax.text(0.01, -0.18,
            "All controls applied: time + sampling + provinciality. "
            "Continental rearrangement does not drive the convergence signal.",
            transform=ax.transAxes, fontsize=8.5, color="#555", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "04_climate_not_continents.png", dpi=220)
    plt.close(fig)
    print("Wrote 04_climate_not_continents.png")


# ---------------------------------------------------------------------------
# 5. Modern analog: where current warming sits in the Phanerozoic distribution
# ---------------------------------------------------------------------------

def make_modern_analog():
    vol = bins["delta_from_prev_T_field_meanabs"].values
    modern_rate = 4.0  # from modern_analog default

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    # Histogram of Phanerozoic volatility
    n, _, _ = ax.hist(vol, bins=20, color=COLOR_CENO, alpha=0.65,
                      edgecolor="white", linewidth=1.2)

    # Compute percentile
    pct = (vol < modern_rate).mean() * 100

    # Modern reference line
    ax.axvline(modern_rate, color=COLOR_MESO, linewidth=2.5,
               label=f"Anthropogenic warming reference\n(~{modern_rate}°C |ΔT| per 10 Myr equivalent)\n{pct:.0f}th percentile of Phanerozoic")

    # Set y limit with headroom for legend
    ax.set_ylim(0, max(n) * 1.35)

    ax.set_xlabel("Climate volatility (|ΔT| between 10 Myr snapshots, °C)", fontsize=10)
    ax.set_ylabel("Number of 10-Myr bins", fontsize=10)
    ax.set_title("Current warming sits in the top 10% of Phanerozoic climate volatility",
                 fontsize=13, pad=12, loc="left")

    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    fig.text(0.04, 0.005,
             "If the convergence theory holds, marine ecosystems today should be losing regional distinctiveness.",
             fontsize=9, color="#555", style="italic")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "05_modern_analog.png", dpi=220)
    plt.close(fig)
    print("Wrote 05_modern_analog.png")


# ---------------------------------------------------------------------------
# 6. The baseline shift mechanism (conceptual + data)
# ---------------------------------------------------------------------------

def make_baseline_shift():
    """Conceptual diagram showing the baseline shift mechanism."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Panel A: schematic
    ax = axes[0]
    taxsim = np.linspace(0, 1, 100)

    # Low volatility line: low intercept, normal slope
    func_low = 0.1 + 0.7 * taxsim
    # High volatility line: HIGHER intercept (raised floor), same slope
    func_high = 0.3 + 0.7 * taxsim

    ax.plot(taxsim, func_low, color=COLOR_CENO, linewidth=2.5, label="Stable climate")
    ax.plot(taxsim, func_high, color=COLOR_MESO, linewidth=2.5, label="Volatile climate")

    # Shade the floor lift
    ax.fill_between([0, 0.15], 0.1, 0.3, color=COLOR_MESO, alpha=0.2)
    ax.annotate("Volatility raises\nthe FLOOR\n(not the slope)",
                xy=(0.075, 0.2), xytext=(0.35, 0.18),
                fontsize=10, ha="left", color=COLOR_MESO, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR_MESO, lw=1.5))

    ax.set_xlabel("Taxonomic similarity\n(fraction of shared genera)", fontsize=10)
    ax.set_ylabel("Functional similarity\n(job-portfolio overlap)", fontsize=10)
    ax.set_title("(A) The mechanism (schematic)", fontsize=12, loc="left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.95)

    # Panel B: era comparison
    ax = axes[1]

    era_means = []
    era_names = []
    era_colors = []
    era_stds = []
    for era_name, color in [("Paleozoic", COLOR_PALEO),
                             ("Mesozoic", COLOR_MESO),
                             ("Cenozoic", COLOR_CENO)]:
        sub = bins[bins["color"] == color]
        era_means.append(sub["functional_excess_similarity_js"].mean())
        era_stds.append(sub["functional_excess_similarity_js"].std())
        era_names.append(f"{era_name}\n(n={len(sub)})")
        era_colors.append(color)

    x = np.arange(len(era_names))
    ax.bar(x, era_means, yerr=era_stds, color=era_colors, alpha=0.85,
           edgecolor="white", linewidth=1.5, capsize=6,
           error_kw={"ecolor": "#444", "lw": 1.2})
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(era_names, fontsize=10)
    ax.set_ylabel("Functional excess similarity (mean per bin)", fontsize=10)
    ax.set_title("(B) Ceiling, transition, floor", fontsize=12, loc="left")

    # Set y limits with room for annotations
    ax.set_ylim(-0.2, 0.22)

    # Annotate above each bar (well-separated)
    ax.text(0, 0.18, "Ceiling\n(saturated)", ha="center", fontsize=9.5,
            color=COLOR_PALEO, fontweight="bold")
    ax.text(1, 0.18, "Sweet spot\n(transition)", ha="center", fontsize=9.5,
            color=COLOR_MESO, fontweight="bold")
    ax.text(2, 0.18, "Floor\n(entrenched)", ha="center", fontsize=9.5,
            color=COLOR_CENO, fontweight="bold")

    fig.suptitle("Why the volatility–convergence link only shows up in the Mesozoic",
                 fontsize=14, fontweight="bold", y=1.02, x=0.04, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "06_baseline_shift_mechanism.png", dpi=220)
    plt.close(fig)
    print("Wrote 06_baseline_shift_mechanism.png")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def make_social_card():
    """Square-ish image for social sharing / blog header thumbnail."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Era backgrounds
    ax.axvspan(540, 252, alpha=0.06, color=COLOR_PALEO, zorder=0)
    ax.axvspan(252, 66, alpha=0.06, color=COLOR_MESO, zorder=0)
    ax.axvspan(66, 0, alpha=0.06, color=COLOR_CENO, zorder=0)

    ax.plot(bins["time_bin"], bins["functional_excess_similarity_js"],
            color="#888", linewidth=1.0, alpha=0.5, zorder=1)

    sizes = 60 + 100 * (bins["delta_from_prev_T_field_meanabs"] / bins["delta_from_prev_T_field_meanabs"].max())
    ax.scatter(bins["time_bin"], bins["functional_excess_similarity_js"],
               c=bins["color"], s=sizes, alpha=0.85, edgecolors="white", linewidths=1.5, zorder=3)

    ax.axhline(0, color="#999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(252, color="#333", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(66, color="#333", linestyle=":", linewidth=0.8, alpha=0.5)

    # Era labels
    ax.text(395, -0.155, "PALEOZOIC", color=COLOR_PALEO, fontsize=11, fontweight="bold",
            ha="center", va="bottom", alpha=0.9)
    ax.text(159, -0.155, "MESOZOIC", color=COLOR_MESO, fontsize=11, fontweight="bold",
            ha="center", va="bottom", alpha=0.9)
    ax.text(33, -0.155, "CENOZOIC", color=COLOR_CENO, fontsize=11, fontweight="bold",
            ha="center", va="bottom", alpha=0.9)

    ax.invert_xaxis()
    ax.set_xlim(550, -10)
    ax.set_ylim(-0.18, 0.22)
    ax.set_xlabel("Million years ago", fontsize=12)
    ax.set_ylabel("Functional convergence between distant ocean regions", fontsize=12)

    ax.set_title("540 million years of\nmarine ecosystem convergence",
                 fontsize=18, pad=20, loc="left", fontweight="bold")

    fig.text(0.04, 0.02,
             "From the Paleozoic ceiling, through the Mesozoic transition,\nto the Cenozoic floor. Marker size = climate volatility.",
             fontsize=10, color="#555", style="italic")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "00_social_card.png", dpi=220)
    plt.close(fig)
    print("Wrote 00_social_card.png")


if __name__ == "__main__":
    make_hero()
    make_social_card()
    make_volatility_scatter()
    make_fingerprint()
    make_exposure_portfolio()
    make_modern_analog()
    make_baseline_shift()
    print(f"\nAll charts written to {OUT}/")
