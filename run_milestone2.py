"""Reproduce Milestone 2 tables, figures and diagnostics from the processed data.

Run run_pipeline.py first. No downloads, imputation, tuning or random splitting
occur here. Financial calculations use decimal returns; outputs label units.
"""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, ROOT, TICKERS
from src.diagnostics import export_meeting2_diagnostics
from src.sensitivity import export_outlier_sensitivity, export_history_audit

OUT = ROOT / "outputs" / "milestone2"
FIG = ROOT / "figures" / "milestone2"
# reported_eps = estimated_eps + earnings_surprise, exactly in this snapshot.
# Keep the estimate and reported EPS, omit the redundant dollar surprise.
MODEL_FEATURES = [x for x in FEATURE_COLUMNS if x != "earnings_surprise"]
TARGETS = ["future_5d_return", "market_adjusted_5d_return"]
DATE_COLUMNS = ["reported_date", "reaction_day", "prediction_start_date", "prediction_end_date"]


def chronological_splits(data):
    """Split by prediction time; purge labels crossing the next period boundary."""
    reaction = data["reaction_day"]
    end = data["prediction_end_date"]
    masks = {
        "train": (reaction < "2023-01-01") & (end < "2023-01-01"),
        "validation": reaction.between("2023-01-01", "2024-12-31") & (end < "2025-01-01"),
        "test": reaction.between("2025-01-01", "2025-12-31") & (end < "2026-01-01"),
    }
    return {name: data.loc[mask].copy() for name, mask in masks.items()}


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def data_audit(data, all_events):
    raw = pd.read_csv(ROOT / "data/raw/earnings/quarterly_earnings.csv")
    raw_dates = pd.to_datetime(raw.reportedDate, errors="coerce")
    pairs = 0
    same_ticker_pairs = 0
    shared_dates = 0
    for i in range(len(data)):
        later = data.iloc[i + 1:]
        overlap = ((later.prediction_start_date <= data.iloc[i].prediction_end_date)
                   & (later.prediction_end_date >= data.iloc[i].prediction_start_date))
        pairs += int(overlap.sum())
        same_ticker_pairs += int((overlap & later.ticker.eq(data.iloc[i].ticker)).sum())
        shared_dates += int(later.reaction_day.eq(data.iloc[i].reaction_day).sum())
    quality = {
        "raw_records": len(raw), "exact_raw_duplicates": int(raw.duplicated().sum()),
        "invalid_reported_dates": int(raw_dates.isna().sum()),
        "outside_event_window": int((~raw_dates.between("2016-01-01", "2025-12-31") & raw_dates.notna()).sum()),
        "in_window_events": len(all_events), "modeling_events": len(data),
        "duplicate_event_keys": int(data.duplicated(["ticker", "reported_date", "fiscal_date_ending"]).sum()),
        "missing_required_predictor_cells": int(all_events[FEATURE_COLUMNS].isna().sum().sum()),
        "nonfinite_modeling_values": int((~np.isfinite(data[FEATURE_COLUMNS + TARGETS])).sum().sum()),
        "invalid_timing_rows": int((~((data.reported_date < data.reaction_day)
            & (data.reaction_day < data.prediction_start_date)
            & (data.prediction_start_date <= data.prediction_end_date))).sum()),
        "overlapping_target_window_pairs": pairs, "same_ticker_overlap_pairs": same_ticker_pairs,
        "same_reaction_date_pairs": shared_dates,
        "events_with_shared_reaction_date": int(data.reaction_day.duplicated(keep=False).sum()),
        "eps_identity_max_error": float((data.reported_eps - data.estimated_eps - data.earnings_surprise).abs().max()),
        "nonpositive_estimated_eps_events": int(data.estimated_eps.le(0).sum()),
        "abs_surprise_over_100_pct_events": int(data.earnings_surprise_pct.abs().gt(100).sum()),
    }
    prices_summary = []
    for ticker in [*TICKERS, "SPY"]:
        p = pd.read_csv(ROOT / "data/raw/prices" / f"{ticker}.csv")
        numeric = p[["close", "adjusted_close", "volume"]]
        prices_summary.append({"ticker": ticker, "rows": len(p),
            "duplicate_dates": int(p.date.duplicated().sum()),
            "invalid_dates": int(pd.to_datetime(p.date, errors="coerce").isna().sum()),
            "missing_close": int(p.close.isna().sum()),
            "missing_adjusted_close": int(p.adjusted_close.isna().sum()),
            "missing_volume": int(p.volume.isna().sum()),
            "nonpositive_close": int(p.close.le(0).sum()),
            "negative_volume": int(p.volume.lt(0).sum()),
            "nonfinite_values": int((~np.isfinite(numeric)).sum().sum()),
            "first_date": p.date.min(), "last_date": p.date.max()})
    pd.DataFrame(prices_summary).to_csv(OUT / "price_quality.csv", index=False)
    pd.DataFrame({"metric": quality.keys(), "value": quality.values()}).to_csv(OUT / "data_quality.csv", index=False)
    important = [*DATE_COLUMNS, *FEATURE_COLUMNS, *TARGETS, "spy_future_5d_return"]
    missing = pd.DataFrame({"variable": important,
        "missing_count": [int(all_events[x].isna().sum()) for x in important]})
    missing["missing_percent"] = 100 * missing.missing_count / len(all_events)
    missing.to_csv(OUT / "missingness.csv", index=False)
    data[FEATURE_COLUMNS + TARGETS].describe().T.to_csv(OUT / "descriptive_statistics.csv")
    return quality


def independent_formula_check(data):
    """Recalculate every predictor and target directly from cached observations."""
    raw = pd.read_csv(ROOT / "data/raw/earnings/quarterly_earnings.csv")
    raw["reportedDate"] = pd.to_datetime(raw.reportedDate)
    raw = raw.drop_duplicates().sort_values(["ticker", "reportedDate", "fiscalDateEnding"], kind="stable")
    raw["previous"] = raw.groupby("ticker").surprisePercentage.shift(1)
    prices = {}
    for ticker in [*TICKERS, "SPY"]:
        p = pd.read_csv(ROOT / "data/raw/prices" / f"{ticker}.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        p["price"] = p.adjusted_close if p.adjusted_close.notna().any() else p.close
        prices[ticker] = p
    spy = prices["SPY"]
    errors = {x: [] for x in [*FEATURE_COLUMNS, *TARGETS, "spy_future_5d_return", "positive_return", "positive_return_amount"]}
    date_checks = 0
    for row in data.itertuples():
        stock = prices[row.ticker]
        i = np.flatnonzero(stock.date.gt(row.reported_date))[0]
        j = np.flatnonzero(spy.date.eq(row.reaction_day))[0]
        assert stock.date.iloc[i] == row.reaction_day
        assert stock.date.iloc[i + 1] == row.prediction_start_date
        assert stock.date.iloc[i + 5] == row.prediction_end_date
        assert (row.reaction_day - row.reported_date).days <= 7
        date_checks += 1
        p, s = stock.price, spy.price
        end_j = np.flatnonzero(spy.date.eq(row.prediction_end_date))[0]
        source = raw.loc[raw.ticker.eq(row.ticker) & raw.reportedDate.eq(row.reported_date)]
        assert len(source) == 1
        source = source.iloc[0]
        y = p.iloc[i + 5] / p.iloc[i] - 1
        market = s.iloc[end_j] / s.iloc[j] - 1
        expected = {
            "reported_eps": source.reportedEPS, "estimated_eps": source.estimatedEPS,
            "earnings_surprise": source.surprise, "earnings_surprise_pct": source.surprisePercentage,
            "previous_earnings_surprise_pct": source.previous,
            "reaction_day_return": p.iloc[i] / p.iloc[i - 1] - 1,
            "return_5d_before": p.iloc[i - 1] / p.iloc[i - 6] - 1,
            "return_20d_before": p.iloc[i - 1] / p.iloc[i - 21] - 1,
            "volatility_20d": p.iloc[i - 21:i].pct_change(fill_method=None).iloc[1:].std(ddof=1),
            "abnormal_volume": stock.volume.iloc[i] / stock.volume.iloc[i - 20:i].mean(),
            "spy_return_5d_before": s.iloc[j - 1] / s.iloc[j - 6] - 1,
            "future_5d_return": y, "spy_future_5d_return": market,
            "market_adjusted_5d_return": y - market,
            "positive_return": int(y > 0), "positive_return_amount": max(y, 0),
        }
        for column, value in expected.items():
            error = abs(getattr(row, column) - value)
            assert np.isfinite(error) and error < 1e-10, (row.ticker, row.reported_date, column, error)
            errors[column].append(error)
    table = pd.DataFrame({"variable": errors.keys(), "max_absolute_error": [max(v) for v in errors.values()], "events_checked": date_checks})
    table.to_csv(OUT / "formula_checks.csv", index=False)
    return float(table.max_absolute_error.max())


def fit_baselines(splits):
    train = splits["train"]
    rows, predictions, diagnostics, coefficients, models = [], [], [], [], {}
    for target in TARGETS:
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(train[MODEL_FEATURES], train[target])
        models[target] = model
        mean = float(train[target].mean())
        for split, sample in splits.items():
            actual = sample[target].to_numpy()
            pred = model.predict(sample[MODEL_FEATURES])
            reference = np.full(len(sample), mean)
            reference_mse = mean_squared_error(actual, reference)
            for name, forecast in [("OLS", pred), ("Training mean", reference), ("Zero", np.zeros(len(sample)))]:
                mse = mean_squared_error(actual, forecast)
                rows.append({"target": target, "split": split, "model": name, "n": len(sample),
                    "mae_pp": 100 * mean_absolute_error(actual, forecast),
                    "rmse_pp": 100 * np.sqrt(mse), "r2": r2_score(actual, forecast),
                    "mse_skill_vs_training_mean": 1 - mse / reference_mse,
                    "mean_error_pp": 100 * np.mean(actual - forecast),
                    "direction_accuracy": np.mean((actual > 0) == (forecast > 0))})
            prediction = sample[["ticker", *DATE_COLUMNS]].copy()
            prediction["target"] = target
            prediction["split"] = split
            prediction["observed"] = actual
            prediction["ols_prediction"] = pred
            prediction["training_mean_prediction"] = mean
            prediction["residual"] = actual - pred
            predictions.append(prediction)
        # OLS influence uses the rank of the design rather than assuming full rank.
        z = model.named_steps["standardscaler"].transform(train[MODEL_FEATURES])
        design = np.column_stack([np.ones(len(train)), z])
        u, singular, _ = np.linalg.svd(design, full_matrices=False)
        rank = int(np.linalg.matrix_rank(design))
        leverage = (u[:, :rank] ** 2).sum(axis=1)
        residual = train[target].to_numpy() - model.predict(train[MODEL_FEATURES])
        mse = np.sum(residual ** 2) / (len(train) - rank)
        cooks = residual ** 2 / (rank * mse) * leverage / (1 - leverage) ** 2
        standardized_residual = residual / np.sqrt(mse * (1 - leverage))
        diagnostic = train[["ticker", "reported_date"]].copy()
        diagnostic["target"] = target
        diagnostic["leverage"] = leverage
        diagnostic["standardized_residual"] = standardized_residual
        diagnostic["cooks_distance"] = cooks
        diagnostics.append(diagnostic)
        models[target + "_diagnostics"] = {"design_rank": rank, "design_columns": design.shape[1],
            "scaled_design_condition_number": float(singular[0] / singular[-1]),
            "high_leverage_count": int((leverage > 2 * rank / len(train)).sum()),
            "cooks_above_4_over_n": int((cooks > 4 / len(train)).sum()),
            "largest_cooks_ticker": str(train.iloc[np.argmax(cooks)].ticker),
            "largest_cooks_date": str(train.iloc[np.argmax(cooks)].reported_date.date()),
            "largest_cooks_distance": float(cooks.max()), "training_mean": mean}
        for feature, coefficient in zip(MODEL_FEATURES, model.named_steps["linearregression"].coef_):
            coefficients.append({"target": target, "feature": feature, "coefficient_pp_per_training_sd": 100 * coefficient})
    metrics = pd.DataFrame(rows)
    prediction = pd.concat(predictions, ignore_index=True)
    diagnostic = pd.concat(diagnostics, ignore_index=True)
    metrics.to_csv(OUT / "baseline_metrics.csv", index=False)
    prediction.to_csv(OUT / "baseline_predictions.csv", index=False)
    diagnostic.to_csv(OUT / "training_influence.csv", index=False)
    pd.DataFrame(coefficients).to_csv(OUT / "coefficients.csv", index=False)
    return metrics, prediction, diagnostic, models


def figures(data, prediction):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.3, 2.7))
    ax.hist(100 * data.future_5d_return, bins=30, color="#286581", edgecolor="white")
    ax.axvline(0, color="#ae4a31", lw=1.4)
    ax.set(xlabel="Signed five-session return (%)", ylabel="Earnings events", title="Return distribution after reaction-day close (n = 400)")
    fig.tight_layout()
    savefig(fig, "01_return_distribution.png")
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.7))
    for ax, column, label in zip(axes, ["earnings_surprise_pct", "reaction_day_return"], ["Earnings surprise (%)", "Reaction-day return (%)"]):
        x = data[column].to_numpy() * (100 if column == "reaction_day_return" else 1)
        y = data.future_5d_return.to_numpy() * 100
        ax.scatter(x, y, s=12, alpha=.45, color="#286581")
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, color="#ae4a31", lw=1.4)
        ax.axhline(0, color="gray", lw=.5)
        ax.set(xlabel=label, ylabel="Future five-session return (%)", title=f"Pearson r = {np.corrcoef(x, y)[0, 1]:.3f}; n = {len(data)}")
    fig.tight_layout(w_pad=1)
    savefig(fig, "02_predictor_relationships.png")
    holdout = prediction.loc[prediction.target.eq("future_5d_return") & prediction.split.ne("train")]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.7))
    for split, color in [("validation", "#286581"), ("test", "#ae4a31")]:
        p = holdout.loc[holdout.split.eq(split)]
        axes[0].scatter(100 * p.ols_prediction, 100 * p.residual, s=17, alpha=.65, color=color, label=f"{split} (n = {len(p)})")
        axes[1].scatter(p.reaction_day, 100 * p.residual, s=17, alpha=.65, color=color)
    for ax in axes:
        ax.axhline(0, color="gray", lw=1)
        ax.set_ylabel("Observed - predicted (pp)")
    axes[0].set_xlabel("OLS predicted return (%)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel("Reaction date")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout(w_pad=1)
    savefig(fig, "03_holdout_residuals.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data/processed/modeling_dataset.csv", parse_dates=DATE_COLUMNS)
    all_events = pd.read_csv(ROOT / "data/processed/earnings_events_all.csv", parse_dates=DATE_COLUMNS)
    quality = data_audit(data, all_events)
    formula_error = independent_formula_check(data)
    splits = chronological_splits(data)
    split_rows = []
    for name, sample in splits.items():
        assert len(sample) > len(MODEL_FEATURES) + 1
        split_rows.append({"split": name, "n": len(sample), "first_reaction": sample.reaction_day.min(),
            "last_reaction": sample.reaction_day.max(), "last_target_end": sample.prediction_end_date.max()})
    pd.DataFrame(split_rows).to_csv(OUT / "split_summary.csv", index=False)
    metrics, predictions, diagnostics, models = fit_baselines(splits)
    figures(data, predictions)
    export_meeting2_diagnostics(data, predictions, diagnostics, metrics, OUT, FIG)
    export_outlier_sensitivity(data, splits, MODEL_FEATURES, TARGETS, diagnostics, OUT, FIG)
    export_history_audit(pd.read_csv(ROOT / "data/raw/earnings/quarterly_earnings.csv"), OUT)
    yearly = data.assign(year=data.reaction_day.dt.year).groupby("year").agg(n=("ticker", "size"),
        mean_return=("future_5d_return", "mean"), sd_return=("future_5d_return", "std"), positive_share=("positive_return", "mean"))
    yearly.to_csv(OUT / "year_summary.csv")
    by_ticker = data.groupby("ticker").agg(n=("ticker", "size"), mean_return=("future_5d_return", "mean"),
        sd_return=("future_5d_return", "std"), positive_share=("positive_return", "mean"))
    by_ticker.to_csv(OUT / "ticker_summary.csv")
    correlations = data[FEATURE_COLUMNS + TARGETS].corr()
    correlations.to_csv(OUT / "correlations.csv")
    # Actual source-file timestamps are not asserted to be retrieval timestamps.
    # Candidate-source audits are separate from the sources used in this model.
    files = sorted(p for folder in ["earnings", "prices"]
                   for p in (ROOT / "data/raw" / folder).iterdir() if p.suffix in {".json", ".csv"})
    manifest = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in files]
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    packages = ["pandas", "numpy", "requests", "yfinance", "matplotlib", "scikit-learn", "python-dotenv", "notebook", "ipykernel"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    summary = {
        "python": platform.python_version(), "packages": versions,
        "quality": quality, "formula_max_error": formula_error,
        "model_features": MODEL_FEATURES, "omitted_redundant_feature": "earnings_surprise",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "purged_or_outside_split_events": len(data) - sum(len(v) for v in splits.values()),
        "raw_return_summary": {"mean": float(data.future_5d_return.mean()), "median": float(data.future_5d_return.median()),
            "sd": float(data.future_5d_return.std()), "min": float(data.future_5d_return.min()), "max": float(data.future_5d_return.max()),
            "positive_n": int(data.positive_return.sum())},
        "diagnostics": {t: models[t + "_diagnostics"] for t in TARGETS},
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.loc[metrics.split.ne("train"), ["target", "split", "model", "n", "mae_pp", "rmse_pp", "mse_skill_vs_training_mean"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
