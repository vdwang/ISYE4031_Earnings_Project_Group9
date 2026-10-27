"""Training-only outlier sensitivity; every held-out event remains scored.

The original full-data OLS stays the primary baseline. These are diagnostic
variants, not a model-selection search or permission to delete difficult tests.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SURPRISE_COLUMNS = ["earnings_surprise_pct", "previous_earnings_surprise_pct"]


def fit_variant(train, sample, features, target, variant, influence=None):
    """Return predictions plus the training-only selection/transformation record."""
    selected = train.copy()
    x_train = selected[features].copy()
    x_sample = sample[features].copy()
    details = {"variant": variant, "n_train_original": len(train), "clip_limits": {}}
    if variant == "Exclude surprise over 100 pct":
        keep = train["earnings_surprise_pct"].abs().le(100)
        selected = train.loc[keep].copy()
        x_train = selected[features].copy()
    elif variant == "Exclude influential training events":
        assert influence is not None
        flagged = influence.loc[influence["cooks_distance"] > 4 / len(train)]
        flagged_keys = set(zip(flagged["ticker"], pd.to_datetime(flagged["reported_date"])))
        keep = [key not in flagged_keys for key in zip(train["ticker"], train["reported_date"])]
        selected = train.loc[keep].copy()
        x_train = selected[features].copy()
    elif variant == "Clip surprise to training 1st and 99th pct":
        for column in SURPRISE_COLUMNS:
            low, high = train[column].quantile([.01, .99]).to_numpy()
            details["clip_limits"][column] = [float(low), float(high)]
            x_train[column] = x_train[column].clip(low, high)
            x_sample[column] = x_sample[column].clip(low, high)
    elif variant != "Original OLS":
        raise ValueError("Unknown sensitivity variant: " + variant)
    assert len(selected) > len(features) + 1
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(x_train, selected[target])
    details["n_train_used"] = len(selected)
    details["n_train_excluded"] = len(train) - len(selected)
    return model.predict(x_sample), details


def export_outlier_sensitivity(data, splits, features, targets, influence, output_dir, figure_dir):
    variants = ["Original OLS", "Exclude surprise over 100 pct",
        "Exclude influential training events", "Clip surprise to training 1st and 99th pct"]
    rows, records = [], []
    train = splits["train"]
    for target in targets:
        target_influence = influence.loc[influence["target"].eq(target)]
        mean = float(train[target].mean())
        for split in ["validation", "test"]:
            sample = splits[split]
            actual = sample[target].to_numpy()
            reference_rmse = 100 * np.sqrt(np.mean((actual - mean) ** 2))
            for variant in variants:
                prediction, details = fit_variant(train, sample, features, target, variant, target_influence)
                error = actual - prediction
                rmse = 100 * np.sqrt(np.mean(error ** 2))
                rows.append({"target": target, "split": split, "variant": variant,
                    "n_train_used": details["n_train_used"], "n_train_excluded": details["n_train_excluded"],
                    "n_evaluated": len(sample), "mae_pp": 100 * np.mean(np.abs(error)),
                    "rmse_pp": rmse, "training_mean_rmse_pp": reference_rmse,
                    "rmse_gain_vs_training_mean_pp": reference_rmse - rmse,
                    "direction_accuracy": np.mean((actual > 0) == (prediction > 0))})
                if split == "validation":
                    records.append({"target": target, **details})
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "outlier_sensitivity.csv", index=False)
    (output_dir / "outlier_sensitivity_rules.json").write_text(json.dumps({
        "large_surprise_rule": "abs(current earnings surprise percentage) > 100; training rows only",
        "influence_rule": "Cook distance > 4/n in the original training fit; target-specific",
        "clip_rule": "Current and prior surprise clipped to training 1st/99th percentiles; no outcome clipping",
        "evaluation_rule": "All 80 validation and all 40 test events retained for every variant",
        "interpretation": "Descriptive sensitivity; the previously inspected 2025 sample is not a fresh test",
        "fits": records}, indent=2), encoding="utf-8")
    large = data.loc[data["earnings_surprise_pct"].abs().gt(100), ["ticker", "reported_date", "estimated_eps",
        "reported_eps", "earnings_surprise_pct", "future_5d_return"]].copy()
    large["split"] = np.select([large["reported_date"] < "2023-01-01", large["reported_date"] < "2025-01-01"],
        ["train", "validation"], default="test")
    large.to_csv(output_dir / "large_surprise_events.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    labels = ["Original", "Surprise\nexclusion", "Influence\nexclusion", "Surprise\nclipping"]
    for ax, target, title in zip(axes, targets, ["Raw signed return", "SPY-adjusted return"]):
        for split, color, marker in [("validation", "#286581", "o"), ("test", "#ae4a31", "s")]:
            selected = results.loc[results["target"].eq(target) & results["split"].eq(split)].set_index("variant")
            ax.plot(np.arange(4), selected.loc[variants, "rmse_pp"], marker=marker, color=color,
                label="2023-24" if split == "validation" else "2025")
            ref = float(selected["training_mean_rmse_pp"].iloc[0])
            ax.axhline(ref, color=color, linestyle="--", alpha=.65, linewidth=1)
        ax.set_xticks(np.arange(4), labels, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Training treatment; all evaluation events retained")
    axes[0].set_ylabel("RMSE (percentage points; lower is better)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "05_outlier_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return results


def export_history_audit(raw_earnings, output_dir):
    """Count available records separately from usable extended event observations."""
    raw = raw_earnings.copy().drop_duplicates()
    raw["reportedDate"] = pd.to_datetime(raw["reportedDate"], errors="coerce")
    numeric_columns = ["reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    rows = []
    for ticker, sample in raw.groupby("ticker"):
        pre = sample.loc[sample["reportedDate"] < "2016-01-01"]
        extended = sample.loc[sample["reportedDate"].between("2010-01-01", "2025-12-31")]
        study = sample.loc[sample["reportedDate"].between("2016-01-01", "2025-12-31")]
        rows.append({"ticker": ticker, "cached_earnings_records": len(sample),
            "earliest_reported_date": sample["reportedDate"].min(), "latest_reported_date": sample["reportedDate"].max(),
            "pre2016_earnings_records": len(pre), "events_2016_2025": len(study),
            "potential_events_2010_2025": len(extended),
            "extended_events_missing_any_earnings_value": int(extended[numeric_columns].isna().any(axis=1).sum()),
            "status": "Counts only; older price lookbacks/targets, lagged surprises and release timing not yet validated"})
    pd.DataFrame(rows).to_csv(output_dir / "history_coverage_audit.csv", index=False)
    raw["year"] = raw["reportedDate"].dt.year
    raw.groupby(["ticker", "year"]).size().rename("cached_earnings_records").reset_index().to_csv(
        output_dir / "earnings_coverage_by_year.csv", index=False)
