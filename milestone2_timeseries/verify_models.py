"""Recalculate exported scores and forecast counts from committed model inputs."""

import json

import numpy as np
import pandas as pd

from milestone2_timeseries.build_data import DAILY, EVENT, OUT


def _score(y, prediction):
    error = np.asarray(y) - np.asarray(prediction)
    return [100 * np.abs(error).mean(), 100 * np.sqrt(np.mean(error ** 2)), np.mean((np.asarray(y) > 0) == (np.asarray(prediction) > 0))]


def _matches(row, y, prediction):
    values = _score(y, prediction)
    return int(len(y) != row.n or not np.allclose(values, [row.mae_pp, row.rmse_pp, row.direction_accuracy], atol=1e-7, rtol=0))


def verify_models(panel=None):
    if panel is None:
        panel = pd.concat([pd.read_parquet(p) for p in sorted(DAILY.glob("*.parquet"))], ignore_index=True)
    checks = {"daily_heldout_rows_checked": 0, "daily_metric_errors": 0, "event_metric_errors": 0,
        "aapl_metric_errors": 0, "exported_observation_errors": 0}
    metrics = pd.read_csv(OUT / "daily_metrics.csv")
    mean = float(panel.loc[panel.analysis_split_1d.eq("train"), "future_return_1d"].mean())
    for split in ["validation", "exploratory_test"]:
        sample = panel.loc[panel.analysis_split_1d.eq(split)]
        checks["daily_heldout_rows_checked"] += len(sample)
        predictions = {"Training mean": np.full(len(sample), mean), "Zero": np.zeros(len(sample))}
        for filename, name in [("daily_ar", "Pooled AR lags 1 2 5"), ("daily_arx", "Pooled ARX with quarter year")]:
            spec = json.loads((OUT / (filename + "_parameters.json")).read_text())
            x = sample[spec["features"]].to_numpy(dtype=np.float32).copy()
            x -= np.asarray(spec["training_means"])
            x /= np.asarray(spec["training_scales"])
            design = np.column_stack([np.ones(len(sample)), x.astype(float)])
            predictions[name] = design @ np.asarray(spec["coefficients"])
        for name, prediction in predictions.items():
            row = metrics.loc[metrics.split.eq(split) & metrics.model.eq(name)].iloc[0]
            checks["daily_metric_errors"] += _matches(row, sample.future_return_1d, prediction)
    event = pd.read_csv(EVENT).set_index("event_id")
    predictions = pd.read_csv(OUT / "event_predictions.csv")
    actual = event.loc[predictions.event_id, "future_5d_return"].to_numpy()
    checks["exported_observation_errors"] += int((~np.isclose(actual, predictions.future_5d_return, atol=1e-9, rtol=0)).sum())
    metrics = pd.read_csv(OUT / "event_metrics.csv")
    names = ["Training mean", "Zero", "Event autoregression", "Event ARX quarter year"]
    for split, sample in predictions.groupby("split"):
        for name in names:
            row = metrics.loc[metrics.split.eq(split) & metrics.model.eq(name)].iloc[0]
            checks["event_metric_errors"] += _matches(row, sample.future_5d_return, sample[name])
    predictions = pd.read_csv(OUT / "aapl_autoreg_predictions.csv", parse_dates=["date"])
    aapl = panel.loc[panel.ticker.eq("AAPL")].set_index("date")
    expected = aapl.loc[predictions.date, "future_return_1d"].to_numpy()
    checks["exported_observation_errors"] += int((~np.isclose(expected, predictions.observed, atol=1e-9, rtol=0)).sum())
    mean = float(aapl.loc[aapl.analysis_split_1d.eq("train"), "future_return_1d"].mean())
    metrics = pd.read_csv(OUT / "aapl_metrics.csv")
    for split, sample in predictions.groupby("split"):
        for name, values in [("Training mean", np.full(len(sample), mean)), ("Zero", np.zeros(len(sample))), ("AAPL AutoReg ARX", sample.prediction)]:
            row = metrics.loc[metrics.split.eq(split) & metrics.model.eq(name)].iloc[0]
            checks["aapl_metric_errors"] += _matches(row, sample.observed, values)
    (OUT / "model_verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if any(value for name, value in checks.items() if name != "daily_heldout_rows_checked"):
        raise ValueError("Exported model verification failed")
    print(json.dumps(checks, indent=2))
    return checks
