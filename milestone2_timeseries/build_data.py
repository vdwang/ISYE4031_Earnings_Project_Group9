"""Build public derived daily shards and a separate earnings-event lag table."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from milestone2_timeseries.features import DAILY_FEATURES, EVENT_FEATURES, company_daily, split_labels, completed_event_lags
from research.build_sp500 import choose_earnings

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/raw/research/sp500/prices"
DAILY = ROOT / "research/datasets/timeseries_daily"
EVENT = ROOT / "research/datasets/timeseries_events.csv"
OUT = ROOT / "outputs/milestone2_timeseries"


def build():
    DAILY.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(ROOT / "research/sp500_universe.csv", dtype={"cik": str})
    events = pd.read_csv(ROOT / "research/datasets/sp500_modeling_dataset.csv", dtype={"cik": str},
        parse_dates=["reported_date", "reaction_day", "prediction_end_date"])
    earnings = choose_earnings(start_date="2008-01-01")
    spy = pd.read_csv(CACHE / "SPY.csv")
    frames, histories, event_calendar, coverage = [], [], [], []
    for number, company in enumerate(universe.itertuples(), 1):
        path = CACHE / (company.ticker + ".csv")
        if not path.exists():
            coverage.append({"ticker": company.ticker, "candidate_stock_days": 0, "complete_stock_days": 0, "status": "no_prices"})
            continue
        raw = pd.read_csv(path)
        daily = company_daily(raw, spy)
        candidate = daily.date.between("2010-01-01", "2025-12-31")
        needed = [*DAILY_FEATURES, "future_return_1d", "future_return_5d"]
        good = candidate & np.isfinite(daily[needed]).all(axis=1)
        sample = daily.loc[good].copy()
        sample["ticker"] = company.ticker
        sample["cik"] = company.cik
        sample["analysis_split_1d"] = split_labels(sample.date, sample.target_end_date_1d)
        sample["analysis_split_5d"] = split_labels(sample.date, sample.target_end_date_5d)
        reaction_dates = events.loc[events.ticker.eq(company.ticker), "reaction_day"]
        sample["earnings_reaction_day"] = sample.date.isin(reaction_dates)
        # Float32 predictors reduce storage; targets retain float64 precision.
        sample[DAILY_FEATURES] = sample[DAILY_FEATURES].astype("float32")
        frames.append(sample)
        coverage.append({"ticker": company.ticker, "candidate_stock_days": int(candidate.sum()),
            "complete_stock_days": len(sample), "status": "complete_rows" if len(sample) else "insufficient_calendar_history"})
        current = events.loc[events.ticker.eq(company.ticker)].copy()
        calendar_columns = ["date", "quarter_return_lag1", "quarter_return_lag4", "year_return_lag1",
            "quarter_return_lag1_source_date", "quarter_return_lag4_source_date", "year_return_lag1_source_date"]
        event_calendar.append(current.merge(daily[calendar_columns], left_on="reaction_day", right_on="date", how="left", validate="many_to_one").drop(columns="date"))
        source = earnings.loc[earnings.ticker.eq(company.ticker), ["ticker", "reported_date"]].copy()
        available = daily[["date", "future_return_5d", "target_end_date_5d"]].rename(columns={"date": "reaction_day", "future_return_5d": "future_5d_return", "target_end_date_5d": "prediction_end_date"})
        history = pd.merge_asof(source.sort_values("reported_date"), available.sort_values("reaction_day"),
            left_on="reported_date", right_on="reaction_day", direction="forward", allow_exact_matches=False)
        history = history.loc[(history.reaction_day - history.reported_date).dt.days.le(7)]
        histories.append(history)
        if number % 50 == 0:
            print("Prepared company", number, "/", len(universe), flush=True)
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    for column in ["ticker", "cik", "analysis_split_1d", "analysis_split_5d"]:
        panel[column] = panel[column].astype("category")
    panel.groupby(panel.date.dt.year).head(10).to_csv(ROOT / "research/datasets/timeseries_daily_preview.csv",
        index=False, float_format="%.9g", date_format="%Y-%m-%d")
    files = []
    for year, sample in panel.groupby(panel.date.dt.year):
        path = DAILY / (str(year) + ".parquet")
        sample.to_parquet(path, index=False, compression="zstd", compression_level=9)
        files.append({"year": int(year), "rows": len(sample), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(files).to_csv(OUT / "daily_file_manifest.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "daily_company_coverage.csv", index=False)
    panel.groupby("analysis_split_1d", observed=True).size().rename("rows").to_csv(OUT / "daily_split_counts.csv")
    event = completed_event_lags(pd.concat(event_calendar, ignore_index=True), pd.concat(histories, ignore_index=True))
    event["model_complete"] = np.isfinite(event[EVENT_FEATURES + ["future_5d_return"]]).all(axis=1)
    metadata = ["event_id", "ticker", "cik", "sector", "reported_date", "reaction_day", "prediction_end_date", "analysis_split", "earnings_source"]
    dates = ["quarter_return_lag1_source_date", "quarter_return_lag4_source_date", "year_return_lag1_source_date", "event_lag1_source_date", "event_lag4_source_date"]
    event[metadata + EVENT_FEATURES + dates + ["future_5d_return", "model_complete"]].to_csv(EVENT,
        index=False, float_format="%.10g", date_format="%Y-%m-%d", lineterminator="\n")
    summary = {"daily_candidate_stock_days": int(sum(x["candidate_stock_days"] for x in coverage)),
        "daily_complete_stock_days": len(panel), "daily_companies": int(panel.ticker.nunique()),
        "daily_first_date": str(panel.date.min().date()), "daily_last_date": str(panel.date.max().date()),
        "daily_storage_bytes": int(sum(x["bytes"] for x in files)), "daily_predictors": DAILY_FEATURES,
        "event_rows_preserved": len(event), "event_complete_time_series_rows": int(event.model_complete.sum()),
        "event_dataset_sha256": hashlib.sha256(EVENT.read_bytes()).hexdigest(),
        "event_predictors": EVENT_FEATURES, "warmup": "2008-2009 used only for historical features; no pre-2010 training rows",
        "grain": "Daily stock-days and earnings announcements are separate observation types",
        "models_fitted_during_data_build": False}
    (OUT / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
