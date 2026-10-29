"""Check every stored daily target, calendar lag and completed-event lag."""

import json

import numpy as np
import pandas as pd

from milestone2_timeseries.build_data import ROOT, DAILY, EVENT, OUT, CACHE
from milestone2_timeseries.features import DAILY_FEATURES, EVENT_FEATURES
from research.build_sp500 import choose_earnings


def verify():
    panel = pd.concat([pd.read_parquet(p) for p in sorted(DAILY.glob("*.parquet"))], ignore_index=True)
    checks = {"daily_rows_checked": len(panel), "daily_duplicate_keys": int(panel.duplicated(["ticker", "date"]).sum()),
        "daily_nonfinite_cells": int((~np.isfinite(panel[DAILY_FEATURES + ["future_return_1d", "future_return_5d"]])).sum().sum()),
        "daily_target_errors": 0, "daily_date_errors": 0, "daily_lag_errors": 0,
        "calendar_value_errors": 0, "calendar_availability_errors": 0, "split_errors": 0,
        "event_target_errors": 0, "event_lag_errors": 0, "event_availability_errors": 0,
        "event_rows_checked": 0, "event_row_coverage_errors": 0}
    precision = {"stored_predictors": "float32", "target_storage": "float64",
        "predictor_absolute_tolerance": 1e-7, "predictor_relative_tolerance": 1e-7,
        "max_calendar_rounding_error_fraction": 0.0}
    market = pd.read_csv(CACHE / "SPY.csv", usecols=["date", "adjusted_close"], parse_dates=["date"]).set_index("date").adjusted_close
    earnings = choose_earnings(start_date="2008-01-01")
    event = pd.read_csv(EVENT, parse_dates=["reported_date", "reaction_day", "prediction_end_date", "event_lag1_source_date", "event_lag4_source_date"])
    original = pd.read_csv(ROOT / "research/datasets/sp500_modeling_dataset.csv").set_index("event_id")
    checks["event_target_errors"] = int((~np.isclose(event.future_5d_return, original.loc[event.event_id, "future_5d_return"].to_numpy(), atol=1e-9, rtol=0)).sum())
    for ticker, sample in panel.groupby("ticker", observed=True):
        stock = pd.read_csv(CACHE / (str(ticker) + ".csv"), usecols=["date", "adjusted_close"], parse_dates=["date"]).set_index("date").sort_index()
        dates, price = stock.index, stock.adjusted_close.to_numpy()
        i = dates.get_indexer(sample.date)
        for horizon in [1, 5]:
            expected = price[i + horizon] / price[i] - 1
            checks["daily_target_errors"] += int((~np.isclose(expected, sample[f"future_return_{horizon}d"], atol=1e-9, rtol=0)).sum())
            checks["daily_date_errors"] += int((dates[i + horizon] != sample[f"target_end_date_{horizon}d"]).sum())
        for lag in [1, 2, 5]:
            expected = price[i - lag + 1] / price[i - lag] - 1
            checks["daily_lag_errors"] += int((~np.isclose(expected, sample[f"return_lag{lag}"], atol=1e-7, rtol=0)).sum())
        known = market.pct_change(fill_method=None).reindex(sample.date).to_numpy()
        checks["daily_lag_errors"] += int((~np.isclose(known, sample.spy_return_known, atol=1e-7, rtol=0)).sum())
        for feature, frequency, lag in [("quarter_return_lag1", "Q", 1), ("quarter_return_lag4", "Q", 4), ("year_return_lag1", "Y", 1)]:
            source = pd.to_datetime(sample[feature + "_source_date"])
            expected_period = sample.date.dt.to_period(frequency) - lag
            checks["calendar_availability_errors"] += int((~(source.dt.to_period(frequency).eq(expected_period) & source.lt(sample.date))).sum())
            j = dates.get_indexer(source)
            denominator = dates.searchsorted(source.dt.to_period(frequency).dt.start_time, side="left") - 1
            expected = price[j] / price[denominator] - 1
            precision["max_calendar_rounding_error_fraction"] = max(precision["max_calendar_rounding_error_fraction"], float(np.max(np.abs(expected - sample[feature]))))
            checks["calendar_value_errors"] += int((~np.isclose(expected, sample[feature], atol=1e-7, rtol=1e-7)).sum())
        current = event.loc[event.ticker.eq(ticker) & event.model_complete].copy()
        checks["event_rows_checked"] += len(current)
        history = earnings.loc[earnings.ticker.eq(ticker)].sort_values("reported_date")
        positions = dates.searchsorted(history.reported_date, side="right")
        valid = (positions + 5 < len(dates))
        positions, history = positions[valid], history.loc[valid]
        gap = (dates[positions] - history.reported_date).days if isinstance(history.reported_date, pd.DatetimeIndex) else (pd.Series(dates[positions], index=history.index) - history.reported_date).dt.days.to_numpy()
        positions = positions[np.asarray(gap) <= 7]
        end = dates[positions + 5]
        target = price[positions + 5] / price[positions] - 1
        order = np.argsort(end)
        end, target = end[order], target[order]
        available = end.searchsorted(current.reaction_day, side="right") - 1
        for lag in [1, 4]:
            selected = available - lag + 1
            checks["event_lag_errors"] += int((~np.isclose(current[f"completed_event_return_lag{lag}"], target[selected], atol=1e-9, rtol=0)).sum())
            source = current[f"event_lag{lag}_source_date"]
            checks["event_availability_errors"] += int((~(source.le(current.reaction_day) & source.eq(pd.Series(end[selected], index=current.index)))).sum())
    for horizon in [1, 5]:
        for split, start, end in [("train", "2010-01-01", "2023-01-01"), ("validation", "2023-01-01", "2025-01-01"), ("exploratory_test", "2025-01-01", "2026-01-01")]:
            sample = panel.loc[panel[f"analysis_split_{horizon}d"].eq(split)]
            good = sample.date.ge(start) & sample.date.lt(end) & sample[f"target_end_date_{horizon}d"].lt(end)
            checks["split_errors"] += int((~good).sum())
    checks["event_row_coverage_errors"] = int(checks["event_rows_checked"] != int(event.model_complete.sum()))
    (OUT / "storage_precision.json").write_text(json.dumps(precision, indent=2), encoding="utf-8")
    (OUT / "data_verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if any(value for key, value in checks.items() if key not in ["daily_rows_checked", "event_rows_checked"]):
        raise ValueError("Time-series data check failed; inspect data_verification.json")
    print(json.dumps(checks, indent=2))
    return checks
