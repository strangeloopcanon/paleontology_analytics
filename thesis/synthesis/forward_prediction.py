"""Forward prediction test: descriptive → predictive.

Train the convergence~volatility model on the first half of the Phanerozoic
(540–270 Ma); predict the second half (270–0 Ma) without refitting. Report
out-of-sample R² and calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from thesis._lib import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_sampling_autocorr/merged.csv",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_forward_prediction")
    ap.add_argument("--split-ma", type=float, default=270.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)

    t = bins["time_bin"].to_numpy(dtype=float)
    v = bins["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    prov = bins["provinciality"].to_numpy(dtype=float) if "provinciality" in bins.columns else np.zeros(len(bins))
    y = bins["functional_excess_similarity_js"].to_numpy(dtype=float)

    # Build log sampling features (PCA fitted on train set only to avoid leakage).
    feat_names = ["n_localities", "marine_n_collections", "marine_n_occurrences"]
    feat_names = [c for c in feat_names if c in bins.columns]
    log_feats = np.column_stack([np.log1p(bins[c].to_numpy(dtype=float)) for c in feat_names]) if feat_names else None

    good = np.isfinite(v) & np.isfinite(y) & np.isfinite(t) & np.isfinite(prov)
    if log_feats is not None:
        good &= np.all(np.isfinite(log_feats), axis=1)

    train = good & (t >= args.split_ma)
    test = good & (t < args.split_ma)

    if train.sum() < 8 or test.sum() < 5:
        results = {"error": f"too few bins in split: train={train.sum()}, test={test.sum()}"}
        (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")
        return

    # Fit PCA on training data only, then project test data through the same transform.
    if log_feats is not None:
        train_feats = log_feats[train]
        mu = np.mean(train_feats, axis=0)
        sd = np.std(train_feats, axis=0, ddof=0)
        sd = np.where(sd == 0, 1.0, sd)
        U_tr, S_tr, Vt_tr = np.linalg.svd((train_feats - mu) / sd, full_matrices=False)
        k = min(2, Vt_tr.shape[0])
        pcs_train = U_tr[:, :k] * S_tr[:k]
        pcs_test = ((log_feats[test] - mu) / sd) @ Vt_tr[:k].T
    else:
        pcs_train = np.zeros((int(train.sum()), 2))
        pcs_test = np.zeros((int(test.sum()), 2))

    X_train = np.column_stack([np.ones(int(train.sum())), v[train], t[train], pcs_train[:, 0], pcs_train[:, 1], prov[train]])
    X_test = np.column_stack([np.ones(int(test.sum())), v[test], t[test], pcs_test[:, 0], pcs_test[:, 1], prov[test]])
    y_train = y[train]
    y_test = y[test]

    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    y_pred_train = X_train.dot(beta)
    y_pred_test = X_test.dot(beta)

    # In-sample R².
    ss_res_train = float(np.sum((y_train - y_pred_train) ** 2))
    ss_tot_train = float(np.sum((y_train - np.mean(y_train)) ** 2))
    r2_train = 1 - ss_res_train / ss_tot_train if ss_tot_train > 0 else float("nan")

    # Out-of-sample R² (relative to test mean).
    ss_res_test = float(np.sum((y_test - y_pred_test) ** 2))
    ss_tot_test = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2_oos = 1 - ss_res_test / ss_tot_test if ss_tot_test > 0 else float("nan")

    # Correlation of predicted vs actual on test set.
    corr_oos = float(np.corrcoef(y_test, y_pred_test)[0, 1]) if len(y_test) >= 3 else float("nan")

    results = {
        "split_ma": args.split_ma,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "r2_in_sample": r2_train,
        "r2_out_of_sample": r2_oos,
        "corr_out_of_sample": corr_oos,
        "vol_beta": float(beta[1]),
        "rmse_train": float(np.sqrt(ss_res_train / train.sum())),
        "rmse_test": float(np.sqrt(ss_res_test / test.sum())),
    }

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Figure: predicted vs actual on test set.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(t[train], y_train, c="#2ca02c", s=30, alpha=0.7, label="Train (observed)")
    ax.scatter(t[train], y_pred_train, c="#2ca02c", s=30, alpha=0.3, marker="x", label="Train (predicted)")
    ax.scatter(t[test], y_test, c="#d62728", s=30, alpha=0.7, label="Test (observed)")
    ax.scatter(t[test], y_pred_test, c="#d62728", s=30, alpha=0.3, marker="x", label="Test (predicted)")
    ax.axvline(args.split_ma, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Time (Ma)")
    ax.set_ylabel("Functional excess similarity")
    ax.set_title("Forward prediction split")
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(y_pred_test, y_test, c="#d62728", s=40, alpha=0.7)
    lims = [min(y_pred_test.min(), y_test.min()), max(y_pred_test.max(), y_test.max())]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(f"Out-of-sample calibration\nR²={r2_oos:.3f}, r={corr_oos:.3f}")

    fig.tight_layout()
    fig.savefig(fig_dir / "forward_prediction.png", dpi=220)
    plt.close(fig)

    lines = [
        "# Forward prediction test",
        "",
        f"- Split: train >= {args.split_ma} Ma (n={results['n_train']}), test < {args.split_ma} Ma (n={results['n_test']})",
        f"- In-sample R²: {r2_train:.3f}",
        f"- Out-of-sample R²: {r2_oos:.3f}",
        f"- Out-of-sample correlation: {corr_oos:.3f}",
        f"- Volatility beta (trained on ancient half): {results['vol_beta']:.4f}",
        f"- RMSE train: {results['rmse_train']:.4f}, test: {results['rmse_test']:.4f}",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
