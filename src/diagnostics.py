"""Descriptive Meeting 2 checks for an already fitted, fixed baseline.

These summaries do not select models or remove observations. Errors are reported
in percentage points (pp). Repeated companies and overlapping windows mean that
the rows should not be treated as independent experiments for significance tests.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def grouped_performance(predictions, group_columns):
    """Score held-out rows only, keeping each target and period separate."""
    heldout = predictions.loc[predictions["split"].isin(["validation", "test"])].copy()
    keys = list(dict.fromkeys(["target", "split", *group_columns]))
    rows = []
    for values, sample in heldout.groupby(keys, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        group = dict(zip(keys, values))
        actual = sample["observed"].to_numpy()
        forecasts = {
            "OLS": sample["ols_prediction"].to_numpy(),
            "Training mean": sample["training_mean_prediction"].to_numpy(),
            "Zero": np.zeros(len(sample)),
        }
        for model, forecast in forecasts.items():
            error = actual - forecast
            rows.append({**group, "model": model, "n": len(sample),
                "mae_pp": 100 * np.mean(np.abs(error)),
                "rmse_pp": 100 * np.sqrt(np.mean(error ** 2)),
                "mean_error_pp": 100 * np.mean(error),
                "direction_accuracy": np.mean((actual > 0) == (forecast > 0))})
    return pd.DataFrame(rows)


def export_meeting2_diagnostics(data, predictions, influence, metrics, output_dir, figure_dir):
    """Save group errors, residual tails and training influence for discussion."""
    joined = predictions.merge(data[["ticker", "reported_date", "volatility_20d"]],
        on=["ticker", "reported_date"], how="left", validate="many_to_one")
    assert joined["volatility_20d"].notna().all()
    joined["year"] = joined["reaction_day"].dt.year
    train = joined.loc[joined["split"].eq("train") & joined["target"].eq("future_5d_return")]
    volatility_cutoff = float(train["volatility_20d"].median())
    joined["prior_volatility_group"] = np.where(joined["volatility_20d"] > volatility_cutoff,
        "Above training median", "At or below training median")
    grouped_performance(joined, ["year"]).to_csv(output_dir / "performance_by_year.csv", index=False)
    grouped_performance(joined, ["ticker"]).to_csv(output_dir / "performance_by_ticker.csv", index=False)
    grouped_performance(joined, ["prior_volatility_group"]).to_csv(output_dir / "performance_by_volatility.csv", index=False)
    heldout = joined.loc[joined["split"].isin(["validation", "test"])].copy()
    # Drop a whole calendar quarter at a time, preserving company co-movement.
    # This is a deterministic stability check, not a confidence interval or test.
    heldout["calendar_quarter"] = heldout["reaction_day"].dt.to_period("Q").astype(str)
    quarter_rows = []
    for (target, split), sample in heldout.groupby(["target", "split"]):
        quarters = sorted(sample["calendar_quarter"].unique())
        for quarter in quarters:
            remaining = sample.loc[~sample["calendar_quarter"].eq(quarter)]
            actual = remaining["observed"].to_numpy()
            ols_rmse = 100 * np.sqrt(np.mean((actual - remaining["ols_prediction"]) ** 2))
            mean_rmse = 100 * np.sqrt(np.mean((actual - remaining["training_mean_prediction"]) ** 2))
            quarter_rows.append({"target": target, "split": split, "quarter_omitted": quarter,
                "n_evaluated": len(remaining), "total_quarters": len(quarters),
                "ols_rmse_pp": ols_rmse, "training_mean_rmse_pp": mean_rmse,
                "rmse_gain_vs_training_mean_pp": mean_rmse - ols_rmse})
    pd.DataFrame(quarter_rows).to_csv(output_dir / "leave_one_quarter_out.csv", index=False)
    rows = []
    for (target, split), sample in heldout.groupby(["target", "split"]):
        error = sample["residual"].to_numpy() * 100
        rows.append({"target": target, "split": split, "n": len(sample),
            "mean_error_pp": np.mean(error), "median_error_pp": np.median(error),
            "residual_p05_pp": np.quantile(error, .05), "residual_p95_pp": np.quantile(error, .95),
            "max_absolute_error_pp": np.max(np.abs(error)),
            "absolute_error_over_5pp_count": int((np.abs(error) > 5).sum()),
            "absolute_error_over_5pp_share": np.mean(np.abs(error) > 5),
            "predicted_min_pct": sample["ols_prediction"].min() * 100,
            "predicted_max_pct": sample["ols_prediction"].max() * 100,
            "observed_min_pct": sample["observed"].min() * 100,
            "observed_max_pct": sample["observed"].max() * 100})
    residuals = pd.DataFrame(rows)
    residuals.to_csv(output_dir / "holdout_residual_summary.csv", index=False)
    # Keep the largest errors visible rather than dropping inconvenient outcomes.
    worst = heldout.assign(absolute_error_pp=heldout["residual"].abs() * 100)
    worst = worst.sort_values("absolute_error_pp", ascending=False).groupby("target", sort=False).head(5)
    worst[["target", "split", "ticker", "reported_date", "observed", "ols_prediction", "residual", "absolute_error_pp"]].to_csv(
        output_dir / "largest_holdout_errors.csv", index=False)
    top_influence = influence.sort_values("cooks_distance", ascending=False).groupby("target", sort=False).head(5)
    top_influence.to_csv(output_dir / "largest_training_influence.csv", index=False)
    (output_dir / "meeting2_diagnostics.json").write_text(json.dumps({
        "volatility_cutoff_from_training_only": volatility_cutoff,
        "n_heldout_per_target": {str(t): int(len(g)) for t, g in heldout.groupby("target")},
        "raw_return_training_mean": float(train["training_mean_prediction"].iloc[0]),
        "interpretation": "Descriptive checks of the fixed baseline; no tuning, exclusions, or independence-based p-values.",
    }, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    model_names = ["OLS", "Training mean", "Zero"]
    palette = ["#286581", "#ae4a31", "#8a969c"]
    x = np.arange(2)
    width = .24
    for ax, target, title in zip(axes, ["future_5d_return", "market_adjusted_5d_return"], ["Raw signed return", "SPY-adjusted return"]):
        for j, (model, color) in enumerate(zip(model_names, palette)):
            selected = metrics.loc[metrics["target"].eq(target) & metrics["model"].eq(model)]
            values = [float(selected.loc[selected["split"].eq(split), "rmse_pp"].iloc[0]) for split in ["validation", "test"]]
            bars = ax.bar(x + (j - 1) * width, values, width, color=color, label=model)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set_xticks(x, ["2023-24\nn = 80", "2025\nn = 40"])
        ax.set_title(title)
        ax.set_ylim(0, 5.5)
        ax.set_xlabel("Evaluation period")
    axes[0].set_ylabel("RMSE (percentage points; lower is better)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / "04_baseline_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return residuals
