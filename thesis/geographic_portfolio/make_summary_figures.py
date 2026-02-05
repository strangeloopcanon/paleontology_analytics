from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_summary(path: Path, *, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run_label"] = label
    return df


def _plot_connectedness(
    df: pd.DataFrame,
    *,
    target: str,
    title: str,
    out_path: Path,
) -> None:
    d = df[df["target"] == target].copy()
    if d.empty:
        return

    # Stable ordering.
    event_order = ["end_ordovician", "late_devonian", "end_permian", "end_triassic"]
    d["event"] = pd.Categorical(d["event"], categories=event_order, ordered=True)
    d = d.sort_values(["event", "coords_mode", "run_label"])

    # Build y positions with small offsets for coords_mode.
    events = [e for e in event_order if e in set(d["event"].astype(str))]
    y_base = {e: i for i, e in enumerate(events)}
    offsets = {"paleo": -0.12, "modern": 0.12}

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = {"paleo": "#1f77b4", "modern": "#ff7f0e"}

    for (event, coords_mode, run_label), sub in d.groupby(
        ["event", "coords_mode", "run_label"], sort=False, observed=True
    ):
        if sub.empty:
            continue
        row = sub.iloc[0]
        y = y_base[str(event)] + offsets.get(str(coords_mode), 0.0)
        x = float(row["or_largest_component_frac"])
        lo = float(row["or_largest_component_frac_p2_5"])
        hi = float(row["or_largest_component_frac_p97_5"])

        ax.errorbar(
            x,
            y,
            xerr=[[x - lo], [hi - x]],
            fmt="o",
            color=colors.get(str(coords_mode), "black"),
            alpha=0.85,
            capsize=3,
        )
        ax.text(
            hi + 0.02,
            y,
            str(run_label),
            fontsize=8,
            va="center",
            ha="left",
            color=colors.get(str(coords_mode), "black"),
            alpha=0.85,
        )

    ax.axvline(1.0, color="black", linewidth=1, alpha=0.6)
    ax.set_yticks(list(y_base.values()))
    ax.set_yticklabels(events)
    ax.set_xlabel("Odds ratio per 1 SD (connectedness = largest_component_frac)")
    ax.set_title(title)
    ax.set_ylim(-0.6, len(events) - 0.4)
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="thesis/geographic_portfolio/figures")
    p.add_argument(
        "--summaries",
        nargs="+",
        default=[
            "thesis/geographic_portfolio/output_with_phylum/summary.csv",
            "thesis/geographic_portfolio/output_grid10_with_phylum/summary.csv",
        ],
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    parts = []
    for s in args.summaries:
        path = Path(s)
        label = path.parent.name.replace("output_", "")
        parts.append(_load_summary(path, label=label))
    df = pd.concat(parts, ignore_index=True)

    _plot_connectedness(
        df,
        target="survived_any",
        title="Connectedness vs survivorship (by event)",
        out_path=out_dir / "connectedness_or_survived_any.png",
    )
    _plot_connectedness(
        df,
        target="survived_10myr",
        title="Connectedness vs 0–10 Myr survivorship (by event)",
        out_path=out_dir / "connectedness_or_survived_10myr.png",
    )

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
