"""Evaluate magnitude and direction models on the committed S&P 500 dataset.

Run from the project directory. No downloads or data rebuilding occur. All
hyperparameter selection uses rolling folds within 2010-2022, and the same
pre-2023 fitted models predict both later periods. The 2025 analysis is exploratory.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.expanded_features import BASE_FEATURES, MODEL_FEATURES
from research.modeling import TrainingQuantileClipper, rolling_folds

DATA = ROOT / "research/datasets/sp500_modeling_dataset.csv"
OUT = ROOT / "outputs/sp500_models"
TARGETS = ["future_5d_return", "market_adjusted_5d_return"]
NAMES = ["Training mean", "Zero", "OLS core 8", "OLS expanded 25", "Ridge 25", "Clipped Ridge 25", "Gradient boosting 25"]
ALPHAS = [.1, 1., 10., 100., 1000.]
CS = [.01, .1, 1., 10.]


def ridge_model(alpha, clipped=False):
    steps = [TrainingQuantileClipper()] if clipped else []
    return make_pipeline(*steps, StandardScaler(), Ridge(alpha=alpha))


def classifier_model(c):
    return make_pipeline(TrainingQuantileClipper(), StandardScaler(),
        LogisticRegression(C=c, max_iter=3000, tol=1e-6, random_state=4031))


def tune_ridge(train, target, clipped):
    records = []
    for alpha in ALPHAS:
        for fold, past, later in rolling_folds(train):
            model = ridge_model(alpha, clipped)
            model.fit(past[MODEL_FEATURES], past[target])
            prediction = model.predict(later[MODEL_FEATURES])
            mse = mean_squared_error(later[target], prediction)
            records.append({"target": target, "family": "Clipped Ridge 25" if clipped else "Ridge 25",
                "parameter": alpha, "fold": fold, "n_train": len(past), "n_eval": len(later),
                "mse_fraction": mse, "rmse_pp": 100 * np.sqrt(mse),
                "mae_pp": 100 * mean_absolute_error(later[target], prediction)})
    table = pd.DataFrame(records)
    weighted = table.assign(weighted_loss=table.mse_fraction * table.n_eval).groupby("parameter").agg(total=("weighted_loss", "sum"), n=("n_eval", "sum"))
    selected = float((weighted.total / weighted.n).idxmin())
    return selected, table


def regression_metrics(actual, prediction, training_mean):
    mse = mean_squared_error(actual, prediction)
    reference_mse = np.mean((np.asarray(actual) - training_mean) ** 2)
    return {"mae_pp": 100 * mean_absolute_error(actual, prediction), "rmse_pp": 100 * np.sqrt(mse),
        "r2": r2_score(actual, prediction) if len(actual) > 1 else np.nan, "mse_skill_vs_training_mean": 1 - mse / reference_mse,
        "direction_accuracy": np.mean((np.asarray(actual) > 0) == (np.asarray(prediction) > 0)),
        "mean_error_pp": 100 * np.mean(np.asarray(actual) - np.asarray(prediction))}


def fit_regressions(splits):
    train = splits["train"]
    metrics, forecasts, cv, parameters, fitted = [], [], [], {}, {}
    for target in TARGETS:
        print("Tuning and fitting", target, flush=True)
        alpha, raw_cv = tune_ridge(train, target, False)
        clipped_alpha, clipped_cv = tune_ridge(train, target, True)
        cv.extend([raw_cv, clipped_cv])
        parameters[target] = {"ridge_alpha": alpha, "clipped_ridge_alpha": clipped_alpha,
            "training_mean_fraction": float(train[target].mean())}
        candidates = {"OLS core 8": (BASE_FEATURES, make_pipeline(StandardScaler(), LinearRegression())),
            "OLS expanded 25": (MODEL_FEATURES, make_pipeline(StandardScaler(), LinearRegression())),
            "Ridge 25": (MODEL_FEATURES, ridge_model(alpha)),
            "Clipped Ridge 25": (MODEL_FEATURES, ridge_model(clipped_alpha, True)),
            "Gradient boosting 25": (MODEL_FEATURES, HistGradientBoostingRegressor(max_iter=150, learning_rate=.05,
                max_leaf_nodes=7, max_depth=3, min_samples_leaf=100, l2_regularization=10.,
                early_stopping=False, random_state=4031))}
        for name, (features, model) in candidates.items():
            model.fit(train[features], train[target])
            fitted[(target, name)] = (features, model)
        mean = float(train[target].mean())
        for split, sample in splits.items():
            prediction = pd.DataFrame({"Training mean": np.full(len(sample), mean), "Zero": np.zeros(len(sample))}, index=sample.index)
            for name, (features, model) in candidates.items():
                prediction[name] = model.predict(sample[features])
            for name in NAMES:
                metrics.append({"target": target, "split": split, "model": name, "n": len(sample),
                    **regression_metrics(sample[target], prediction[name], mean)})
            if split != "train":
                rows = sample[["event_id", "ticker", "cik", "sector", "earnings_source", "reported_date", "reaction_day", "prediction_end_date", "large_surprise_flag"]].copy()
                rows["target"] = target
                rows["split"] = split
                rows["observed"] = sample[target]
                rows = pd.concat([rows, prediction], axis=1)
                forecasts.append(rows)
    return pd.DataFrame(metrics), pd.concat(forecasts, ignore_index=True), pd.concat(cv, ignore_index=True), parameters, fitted


def classification(splits):
    train = splits["train"]
    records = []
    for c in CS:
        for fold, past, later in rolling_folds(train):
            model = classifier_model(c)
            model.fit(past[MODEL_FEATURES], past.future_5d_return.gt(0).astype(int))
            p = model.predict_proba(later[MODEL_FEATURES])[:, 1]
            records.append({"C": c, "fold": fold, "n_train": len(past), "n_eval": len(later),
                "log_loss": log_loss(later.future_5d_return.gt(0).astype(int), p)})
    cv = pd.DataFrame(records)
    weighted = cv.assign(weighted=cv.log_loss * cv.n_eval).groupby("C").agg(loss=("weighted", "sum"), n=("n_eval", "sum"))
    selected = float((weighted.loss / weighted.n).idxmin())
    model = classifier_model(selected)
    y_train = train.future_5d_return.gt(0).astype(int)
    model.fit(train[MODEL_FEATURES], y_train)
    prior = float(y_train.mean())
    metrics, forecasts = [], []
    for split, sample in splits.items():
        actual = sample.future_5d_return.gt(0).astype(int)
        probability = model.predict_proba(sample[MODEL_FEATURES])[:, 1]
        for name, p in [("Training prevalence", np.full(len(sample), prior)), ("Logistic regression 25", probability)]:
            predicted_class = np.asarray(p) >= .5
            metrics.append({"split": split, "model": name, "n": len(sample), "positive_fraction": actual.mean(),
                "accuracy": accuracy_score(actual, predicted_class), "balanced_accuracy": balanced_accuracy_score(actual, predicted_class),
                "roc_auc": roc_auc_score(actual, p), "log_loss": log_loss(actual, p), "brier_score": brier_score_loss(actual, p)})
        if split != "train":
            rows = sample[["event_id", "ticker", "reaction_day"]].copy()
            rows["split"] = split
            rows["positive_observed"] = actual
            rows["logistic_probability"] = probability
            rows["training_prevalence"] = prior
            forecasts.append(rows)
    return pd.DataFrame(metrics), pd.concat(forecasts, ignore_index=True), cv, {"C": selected, "training_positive_fraction": prior}


def diagnostics(predictions):
    records = []
    for group in ["year", "sector", "earnings_source", "ticker", "large_surprise_flag"]:
        work = predictions.copy()
        work["year"] = work.reaction_day.dt.year
        for (target, split, level), sample in work.groupby(["target", "split", group]):
            for model in NAMES:
                mean = float(sample["Training mean"].iloc[0])
                records.append({"group": group, "level": str(level), "target": target, "split": split, "model": model,
                    "n": len(sample), **regression_metrics(sample.observed, sample[model], mean)})
    return pd.DataFrame(records)


def recent_source_sensitivity(splits, params):
    """Fixed already-selected Ridge settings; this comparison is not retuned."""
    rows = []
    recent = {name: sample.loc[sample.reported_date.ge("2016-01-01") & sample.earnings_source.ne("hf_archive")].copy()
        for name, sample in splits.items()}
    for target in TARGETS:
        train = recent["train"]
        mean = float(train[target].mean())
        model = ridge_model(params[target]["clipped_ridge_alpha"], True)
        model.fit(train[MODEL_FEATURES], train[target])
        core = make_pipeline(StandardScaler(), LinearRegression()).fit(train[BASE_FEATURES], train[target])
        for split in ["validation", "exploratory_test"]:
            sample = recent[split]
            for name, prediction in [("Training mean", np.full(len(sample), mean)), ("Zero", np.zeros(len(sample))),
                ("OLS core 8", core.predict(sample[BASE_FEATURES])),
                ("Clipped Ridge 25", model.predict(sample[MODEL_FEATURES]))]:
                rows.append({"target": target, "split": split, "model": name, "n_train": len(train), "n_eval": len(sample),
                    **regression_metrics(sample[target], prediction, mean)})
    return pd.DataFrame(rows)


def figures(metrics, predictions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    labels = ["Mean", "Zero", "OLS 8", "OLS 25", "Ridge", "Clipped\nRidge", "Boosting"]
    for ax, target, title in zip(axes, TARGETS, ["Signed stock return", "Stock return minus SPY"]):
        for split, offset, color, label in [("validation", -.19, "#29677f", "2023–2024"),
            ("exploratory_test", .19, "#bc5c3c", "2025 exploratory")]:
            sample = metrics.loc[metrics.target.eq(target) & metrics.split.eq(split)].set_index("model").loc[NAMES]
            ax.bar(np.arange(len(NAMES)) + offset, sample.rmse_pp, .38, label=label, color=color)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(NAMES)), labels, fontsize=9)
        ax.set_ylabel("RMSE in percentage points  |  lower is better")
        largest = metrics.loc[metrics.target.eq(target) & metrics.split.ne("train"), "rmse_pp"].max()
        ax.set_ylim(0, largest * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("S&P 500 model comparison  |  fitted through 2022", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target, title in zip(axes, TARGETS, ["Signed stock return", "Stock return minus SPY"]):
        sample = predictions.loc[predictions.target.eq(target)].copy()
        sample["year"] = sample.reaction_day.dt.year
        for name, color in [("Training mean", "#777777"), ("OLS core 8", "#29677f"),
            ("Ridge 25", "#669b4d"), ("Gradient boosting 25", "#bc5c3c")]:
            values = 100 * sample.assign(squared_error=(sample.observed - sample[name]) ** 2).groupby("year").squared_error.mean().pow(.5)
            ax.plot(values.index, values.values, marker="o", label=name, color=color)
        ax.set(title=title, xlabel="Evaluation year", ylabel="RMSE in percentage points")
        ax.set_xticks([2023, 2024, 2025])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Performance stability across later years")
    fig.tight_layout()
    fig.savefig(OUT / "yearly_performance.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    expected = json.loads((ROOT / "outputs/dataset_research/sp500/dataset_summary.json").read_text())["dataset_sha256"]
    if data_hash != expected:
        raise ValueError("Dataset changed from its verified snapshot; audit the new data first")
    data = pd.read_csv(DATA, dtype={"cik": str}, parse_dates=["reported_date", "reaction_day", "prediction_end_date"])
    if not np.isfinite(data[MODEL_FEATURES + TARGETS]).all().all():
        raise ValueError("Unexpected nonfinite modeling values")
    splits = {name: data.loc[data.analysis_split.eq(name)].copy() for name in ["train", "validation", "exploratory_test"]}
    for label, sample in splits.items():
        end = {"train": "2023-01-01", "validation": "2025-01-01", "exploratory_test": "2026-01-01"}[label]
        if sample.prediction_end_date.ge(end).any():
            raise ValueError("Boundary-crossing target in " + label)
    specification = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_sha256": data_hash,
        "split_counts": {k: len(v) for k, v in splits.items()}, "purged_rows": int(data.analysis_split.eq("purged_boundary").sum()),
        "features": MODEL_FEATURES, "core_features": BASE_FEATURES, "regression_models": NAMES,
        "ridge_alpha_grid": ALPHAS, "logistic_C_grid": CS,
        "tuning": "Pooled squared error for Ridge; log loss for logistic; purged rolling folds 2017-18, 2019-20, 2021-22 within training only",
        "clipping": "Predictors clipped to fit-sample 1st/99th percentiles; outcomes and held-out rows never clipped or removed",
        "boosting": {"max_iter": 150, "learning_rate": .05, "max_leaf_nodes": 7, "max_depth": 3, "min_samples_leaf": 100,
            "l2_regularization": 10., "early_stopping": False, "random_state": 4031},
        "fit_policy": "Fit through 2022 only; no validation/test refit; 2025 remains exploratory",
        "package_versions": {x: importlib.metadata.version(x) for x in ["pandas", "numpy", "scikit-learn"]}}
    # Save the candidate/selection rules before computing any evaluation results.
    (OUT / "run_specification.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    print("Starting models; split counts", specification["split_counts"], flush=True)
    with threadpool_limits(limits=2):
        metrics, predictions, cv, params, fitted = fit_regressions(splits)
        print("Fitting direction classifier", flush=True)
        class_metrics, class_predictions, class_cv, class_params = classification(splits)
        recent = recent_source_sensitivity(splits, params)
    metrics.to_csv(OUT / "regression_metrics.csv", index=False)
    predictions.to_csv(OUT / "regression_predictions.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d")
    cv.to_csv(OUT / "ridge_rolling_cv.csv", index=False)
    class_metrics.to_csv(OUT / "classification_metrics.csv", index=False)
    class_predictions.to_csv(OUT / "classification_predictions.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d")
    class_cv.to_csv(OUT / "classification_rolling_cv.csv", index=False)
    diagnostics(predictions).to_csv(OUT / "grouped_performance.csv", index=False)
    recent.to_csv(OUT / "recent_source_sensitivity.csv", index=False)
    (OUT / "selected_parameters.json").write_text(json.dumps({"regression": params, "classification": class_params}, indent=2), encoding="utf-8")
    coefficients = []
    for (target, name), (features, model) in fitted.items():
        last = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(last, "coef_"):
            for feature, value in zip(features, last.coef_):
                coefficients.append({"target": target, "model": name, "feature": feature, "coefficient_pp_per_training_sd": 100 * value})
    pd.DataFrame(coefficients).to_csv(OUT / "linear_coefficients.csv", index=False)
    figures(metrics, predictions)
    print(metrics.loc[metrics.split.ne("train"), ["target", "split", "model", "mae_pp", "rmse_pp", "direction_accuracy"]].to_string(index=False))
    print(class_metrics.loc[class_metrics.split.ne("train")].to_string(index=False))


if __name__ == "__main__":
    main()
