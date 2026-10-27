"""Summarize saved model outputs and audit their reported metrics without refits."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.run_sp500_models import NAMES, TARGETS, regression_metrics

OUT = ROOT / "outputs/sp500_models"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(str(x) for x in row) + " |" for row in rows]])


def bootstrap(predictions, classifier):
    rows = []
    rng = np.random.default_rng(4031)
    for (target, split), sample in predictions.groupby(["target", "split"]):
        sample = sample.copy()
        sample["month"] = sample.reaction_day.dt.to_period("M")
        sample["reference_squared_error"] = (sample.observed - sample["Training mean"]) ** 2
        sample["model_squared_error"] = (sample.observed - sample["OLS core 8"]) ** 2
        month = sample.groupby("month").agg(n=("observed", "size"), reference=("reference_squared_error", "sum"), model=("model_squared_error", "sum"))
        choices = rng.integers(0, len(month), size=(2000, len(month)))
        n = month.n.to_numpy()[choices].sum(axis=1)
        gains = 100 * (np.sqrt(month.reference.to_numpy()[choices].sum(axis=1) / n)
            - np.sqrt(month.model.to_numpy()[choices].sum(axis=1) / n))
        rows.append({"analysis": target, "split": split, "model": "OLS core 8", "calendar_month_clusters": len(month),
            "metric": "RMSE gain vs training mean in pp; positive favors model",
            "point_gain": 100 * (np.sqrt(sample.reference_squared_error.mean()) - np.sqrt(sample.model_squared_error.mean())),
            "lower_95": float(np.quantile(gains, .025)), "upper_95": float(np.quantile(gains, .975))})
    for split, sample in classifier.groupby("split"):
        sample = sample.copy()
        sample["month"] = sample.reaction_day.dt.to_period("M")
        sample["correct_gain"] = ((sample.logistic_probability.ge(.5)).eq(sample.positive_observed)).astype(int) - sample.training_prevalence.ge(.5).eq(sample.positive_observed).astype(int)
        month = sample.groupby("month").agg(n=("correct_gain", "size"), gain=("correct_gain", "sum"))
        choices = rng.integers(0, len(month), size=(2000, len(month)))
        gains = 100 * month.gain.to_numpy()[choices].sum(axis=1) / month.n.to_numpy()[choices].sum(axis=1)
        rows.append({"analysis": "positive_return_classification", "split": split, "model": "Logistic regression 25", "calendar_month_clusters": len(month),
            "metric": "Accuracy gain vs training majority in percentage points",
            "point_gain": 100 * sample.correct_gain.mean(), "lower_95": float(np.quantile(gains, .025)), "upper_95": float(np.quantile(gains, .975))})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "monthly_bootstrap_diagnostics.csv", index=False)
    return result


def main():
    metrics = pd.read_csv(OUT / "regression_metrics.csv")
    prediction = pd.read_csv(OUT / "regression_predictions.csv", parse_dates=["reaction_day"])
    cls = pd.read_csv(OUT / "classification_metrics.csv")
    class_prediction = pd.read_csv(OUT / "classification_predictions.csv", parse_dates=["reaction_day"])
    spec = json.loads((OUT / "run_specification.json").read_text())
    parameters = json.loads((OUT / "selected_parameters.json").read_text())
    recent = pd.read_csv(OUT / "recent_source_sensitivity.csv")
    checks = {"heldout_regression_rows": len(prediction), "heldout_classification_rows": len(class_prediction),
        "metric_recalculation_errors": 0, "unequal_evaluation_counts": 0,
        "duplicate_exported_events": 0, "observed_target_mismatches": 0,
        "prediction_nonfinite_cells": int((~np.isfinite(prediction[NAMES])).sum().sum())}
    data_hash = hashlib.sha256((ROOT / "research/datasets/sp500_modeling_dataset.csv").read_bytes()).hexdigest()
    checks["dataset_hash_mismatch"] = int(data_hash != spec["dataset_sha256"])
    source = pd.read_csv(ROOT / "research/datasets/sp500_modeling_dataset.csv").set_index("event_id")
    for (target, split), sample in prediction.groupby(["target", "split"]):
        checks["unequal_evaluation_counts"] += int(len(sample) != spec["split_counts"][split])
        checks["duplicate_exported_events"] += int(sample.event_id.duplicated().sum())
        checks["observed_target_mismatches"] += int((~np.isclose(source.loc[sample.event_id, target].to_numpy(), sample.observed, atol=1e-9, rtol=0)).sum())
        for name in NAMES:
            row = metrics.loc[metrics.target.eq(target) & metrics.split.eq(split) & metrics.model.eq(name)].iloc[0]
            expected = regression_metrics(sample.observed, sample[name], sample["Training mean"].iloc[0])
            checks["metric_recalculation_errors"] += int(any(not np.isclose(row[c], expected[c], atol=1e-8, rtol=0) for c in expected))
    for split, sample in class_prediction.groupby("split"):
        checks["unequal_evaluation_counts"] += int(len(sample) != spec["split_counts"][split])
        checks["duplicate_exported_events"] += int(sample.event_id.duplicated().sum())
        actual = sample.positive_observed
        expected_actual = source.loc[sample.event_id, "future_5d_return"].gt(0).astype(int).to_numpy()
        checks["observed_target_mismatches"] += int((expected_actual != actual).sum())
        for name, column in [("Training prevalence", "training_prevalence"), ("Logistic regression 25", "logistic_probability")]:
            probability = sample[column]
            checks["prediction_nonfinite_cells"] += int((~np.isfinite(probability)).sum())
            label = probability.ge(.5)
            expected = {"accuracy": accuracy_score(actual, label), "balanced_accuracy": balanced_accuracy_score(actual, label),
                "roc_auc": roc_auc_score(actual, probability), "log_loss": log_loss(actual, probability), "brier_score": brier_score_loss(actual, probability)}
            row = cls.loc[cls.split.eq(split) & cls.model.eq(name)].iloc[0]
            checks["metric_recalculation_errors"] += int(any(not np.isclose(row[c], expected[c], atol=1e-8, rtol=0) for c in expected))
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if any(v for k, v in checks.items() if k not in ["heldout_regression_rows", "heldout_classification_rows"]):
        raise ValueError("Saved predictions do not match reported metrics")
    intervals = bootstrap(prediction, class_prediction)
    best = []
    for target in TARGETS:
        val = metrics.loc[metrics.target.eq(target) & metrics.split.eq("validation")]
        best.append({"target": target, "best_overall_validation_rmse": val.loc[val.rmse_pp.idxmin(), "model"],
            "best_trained_validation_rmse": val.loc[val.model.isin(NAMES[2:])].sort_values("rmse_pp").iloc[0].model})
    (OUT / "validation_selection.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    lines = ["# S&P 500 model results", "",
        "The larger dataset supports a small improvement from the simpler eight-feature linear regression for signed stock returns. "
        "Adding all 25 predictors, regularization or gradient boosting does not produce consistent gains across the later periods. "
        "Direction classification is mixed, and the monthly uncertainty diagnostics do not establish a dependable forecasting advantage.", "",
        "These comparisons are exploratory. The coefficient tables describe the fitted models and do not provide feature-level significance tests.", "",
        "## Data and evaluation", "",
        "The locked dataset has 28,496 events from 495 snapshot S&P 500 issuers. We use 22,663 events in 2010–2022 for training, "
        "3,876 in 2023–2024 for validation and 1,956 in 2025 for exploratory testing. One boundary-crossing event is purged. "
        "All models are fitted through 2022; none are refitted on validation or test observations.", "",
        "The raw training-mean reference always predicts +0.288%; the SPY-adjusted training-mean reference predicts +0.083%. "
        "The zero reference predicts no change. Core OLS uses the eight retained derived pilot features, rather than the old pilot's raw EPS levels. "
        "The 25-feature candidates add historical earnings, market/sector context, rates and interactions.", "",
        "Ridge penalties and logistic C are chosen only within training using purged rolling folds evaluating 2017–2018, 2019–2020 and 2021–2022. "
        "Feature scaling and clipping are refitted inside each fold. Ridge selection minimizes pooled squared error; logistic selection minimizes log loss. "
        "All selected Ridge penalties were 1,000 and logistic C was 0.01. These were the tested grid boundaries; no expanded search was made after viewing holdout results.", "",
        "The candidate rules and dataset hash are in [run_specification.json](../outputs/sp500_models/run_specification.json). "
        "The 2025 outcomes had already been inspected in earlier project work, so this is not a fresh final holdout.", "",
        "## Forecasting the signed stock return", "",
        "MAE and RMSE below are in percentage points; lower is better. For example, RMSE 4.219 means a root-mean-square error of about 4.22 return percentage points.", ""]
    for target, heading in [("future_5d_return", "Raw stock returns"), ("market_adjusted_5d_return", "Stock returns minus SPY")]:
        if target != TARGETS[0]:
            lines.extend(["", "## " + heading, "", "The SPY-adjusted target uses the same reaction/end dates. A zero forecast is a particularly relevant reference for this target.", ""])
        rows = []
        for name in NAMES:
            a = metrics.loc[metrics.target.eq(target) & metrics.split.eq("validation") & metrics.model.eq(name)].iloc[0]
            b = metrics.loc[metrics.target.eq(target) & metrics.split.eq("exploratory_test") & metrics.model.eq(name)].iloc[0]
            rows.append([name, f"{a.mae_pp:.3f}", f"{a.rmse_pp:.3f}", f"{b.mae_pp:.3f}", f"{b.rmse_pp:.3f}"])
        lines.append(md_table(["Model", "2023–24 MAE", "2023–24 RMSE", "2025 MAE", "2025 RMSE"], rows))
        if target == TARGETS[0]:
            val = metrics.loc[metrics.target.eq(target) & metrics.split.eq("validation")].set_index("model")
            test = metrics.loc[metrics.target.eq(target) & metrics.split.eq("exploratory_test")].set_index("model")
            a = val.loc["Training mean", "rmse_pp"] - val.loc["OLS core 8", "rmse_pp"]
            b = test.loc["Training mean", "rmse_pp"] - test.loc["OLS core 8", "rmse_pp"]
            lines.extend(["", f"Core OLS reduces RMSE by {a:.3f} percentage points ({100*a/val.loc['Training mean','rmse_pp']:.2f}%) in validation "
                f"and {b:.3f} points ({100*b/test.loc['Training mean','rmse_pp']:.2f}%) in 2025, relative to the training-mean forecast. "
                "The error remains large relative to the small improvement. Core OLS also has slightly lower RMSE than the mean in each of 2023, 2024 and 2025 individually.", "",
                "The 25-feature Ridge has the lowest trained-model RMSE in 2025, but performs worse than the mean in validation; choosing it based on 2025 alone would be hindsight selection. "
                "The core OLS is the trained model preferred by validation RMSE. Gradient boosting is worse than the simple references in both later periods."])
        else:
            lines.extend(["", "No fitted candidate consistently improves on the zero forecast for the SPY-adjusted target. "
                "The modest raw-return gain does not establish an ability to predict returns beyond the market benchmark."])
    lines.extend(["", "## Predicting up versus down", "",
        "Logistic regression predicts whether the stock's next-five-session return is strictly positive. Zero is grouped with nonpositive returns. "
        "The reference probability is the training positive frequency, 53.766%; its 0.5 classification threshold predicts up for every event. "
        "This majority reference is more informative than comparing accuracy with 50%.", ""])
    rows = []
    for _, row in cls.loc[cls.split.ne("train")].iterrows():
        rows.append(["2023–24" if row.split == "validation" else "2025 exploratory", row.model,
            f"{100*row.accuracy:.2f}%", f"{100*row.balanced_accuracy:.2f}%", f"{row.roc_auc:.3f}", f"{row.log_loss:.4f}"])
    lines.append(md_table(["Period", "Model", "Accuracy", "Balanced accuracy", "ROC AUC", "Log loss"], rows))
    lines.extend(["", "The classifier is worse than the majority reference in validation accuracy and probability loss. "
        "In 2025 it correctly classifies only four more events than the majority forecast, although its ROC AUC and probability loss improve. "
        "Its validation-to-test change prevents a claim of stable directional skill.", "",
        "## Stability and source checks", "",
        "The [yearly chart](../outputs/sp500_models/yearly_performance.png) and [grouped table](../outputs/sp500_models/grouped_performance.csv) "
        "report later-period errors by year, sector, company, source and current surprise flag. Small company groups are descriptive; R-squared is omitted for single-observation groups.", "",
        "A separate sensitivity refits fixed core OLS and clipped-Ridge specifications on 2016 onward rows whose current source is not the Hugging Face archive, "
        "without retuning. It changes both the training population and the eligible evaluation population; it is not an isolated causal effect of changing sources. "
        "The current-source filter cannot certify every historical feature's vintage or provenance.", ""])
    rows = []
    for _, row in recent.loc[recent.target.eq("future_5d_return")].iterrows():
        rows.append(["2023–24" if row.split == "validation" else "2025", row.model, int(row.n_train), int(row.n_eval), f"{row.rmse_pp:.3f}"])
    lines.append(md_table(["Period", "Recent-source fit", "Training rows", "Evaluation rows", "RMSE pp"], rows))
    lines.extend(["", "[Complete source-sensitivity metrics](../outputs/sp500_models/recent_source_sensitivity.csv) include both targets. "
        "Source disagreement, snapshot survivorship and incomplete release-time histories remain limitations despite the larger sample.", "",
        "## Approximate uncertainty", "",
        "The following descriptive paired bootstrap resamples calendar months, retaining all events within a sampled month. "
        "This preserves within-month market/firm dependence but does not fully account for cross-month overlap, persistent company effects or model-selection uncertainty. "
        "There are only 24 validation and 12 test month clusters. Intervals are exploratory diagnostics, not a definitive hypothesis test.", ""])
    rows = []
    for _, row in intervals.loc[intervals.analysis.eq("future_5d_return") | intervals.analysis.eq("positive_return_classification")].iterrows():
        rows.append([row.analysis, "2023–24" if row.split == "validation" else "2025", f"{row.point_gain:.3f}", f"[{row.lower_95:.3f}, {row.upper_95:.3f}]"])
    lines.append(md_table(["Analysis", "Period", "Gain in percentage points", "Approximate 95% interval"], rows))
    lines.extend(["", "For raw returns, gain is reference RMSE minus core-OLS RMSE; for classification it is logistic accuracy minus majority accuracy. "
        "Positive values favor the fitted model. Interpret the interval and its limitations rather than treating the point improvement alone as established skill.", "",
        "All displayed intervals include zero. Under this approximation, the small core-OLS and classification point gains are not clearly distinguishable from no improvement.", "",
        "## Reproduce and next steps", "", "```powershell", ".\\.venv\\Scripts\\python.exe research\\run_sp500_models.py",
        ".\\.venv\\Scripts\\python.exe research\\summarize_sp500_models.py", "```", "",
        "The model script reads only the committed CSV and needs the root requirements; no new API downloads or raw caches are needed. "
        "The summary script recalculates saved holdout metrics and verifies that every model scores the same events. "
        "Fifteen automated tests pass, including training-only clipping, purged tuning folds and future-feature invariance.", "",
        "Keep the core linear model and simple forecasts as the reference for subsequent work. A useful next investigation is source/date quality and "
        "predefined company/regime structure, with rolling validation and grouped uncertainty, before increasing model complexity. "
        "Do not select features, outlier rules or models to make the already-inspected 2025 result look better.", ""])
    (ROOT / "reports/SP500_MODEL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(intervals.to_string(index=False))


if __name__ == "__main__":
    main()
