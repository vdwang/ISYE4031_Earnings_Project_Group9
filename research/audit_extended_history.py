"""Test longer free histories using the existing earnings source and separate prices.

The primary data and model outputs are not overwritten. No models are selected
or refitted in this dataset-quality audit.
"""

import json
from pathlib import Path
import sys
import time

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.build_dataset import build_events
from src.config import FEATURE_COLUMNS, PRICE_COLUMNS, PRICE_TICKERS

CACHE = ROOT / "data/raw/research/extended_prices"
PROCESSED = ROOT / "data/processed/research"
OUT = ROOT / "outputs/dataset_research"


def main():
    for folder in [CACHE, PROCESSED, OUT]:
        folder.mkdir(parents=True, exist_ok=True)
    prices = {}
    price_quality = []
    for ticker in PRICE_TICKERS:
        path = CACHE / f"{ticker}.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            history = yf.Ticker(ticker).history(start="1995-11-01", end="2026-02-01",
                interval="1d", auto_adjust=False, actions=False, raise_errors=True)
            if history.empty:
                raise RuntimeError("No extended history returned for " + ticker)
            frame = history.reset_index()
            frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
            frame = frame.rename(columns={"adj_close": "adjusted_close"})
            frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
            frame["ticker"] = ticker
            frame = frame.reindex(columns=PRICE_COLUMNS)
            frame.to_csv(path, index=False)
            time.sleep(1)
        prices[ticker] = frame
        price_quality.append({"ticker": ticker, "rows": len(frame), "first_date": frame.date.min(),
            "last_date": frame.date.max(), "missing_adjusted_close": int(frame.adjusted_close.isna().sum()),
            "duplicate_dates": int(frame.date.duplicated().sum())})
        print("Extended-price cache:", ticker, len(frame), flush=True)
    pd.DataFrame(price_quality).to_csv(OUT / "extended_price_quality.csv", index=False)
    raw = pd.read_csv(ROOT / "data/raw/earnings/quarterly_earnings.csv")
    summaries = []
    for start in ["1996-01-01", "2010-01-01", "2016-01-01"]:
        events, counts = build_events(raw, prices, event_start=start, event_end="2025-12-31")
        required = [*FEATURE_COLUMNS, "future_5d_return", "spy_future_5d_return"]
        complete = events[required].notna().all(axis=1)
        events["complete_case"] = complete
        label = start[:4] + "_2025"
        events.to_csv(PROCESSED / ("events_" + label + ".csv"), index=False)
        events.loc[complete].to_csv(PROCESSED / ("modeling_" + label + ".csv"), index=False)
        for ticker, sample in events.groupby("ticker"):
            summaries.append({"window": label, "ticker": ticker, "candidate_events": len(sample),
                "matched_events": int(sample.reaction_day.notna().sum()), "complete_events": int(sample.complete_case.sum()),
                "excluded_events": int((~sample.complete_case).sum()), "first_report": sample.reported_date.min(),
                "last_report": sample.reported_date.max()})
        missing = pd.DataFrame({"variable": required, "missing_count": [int(events[c].isna().sum()) for c in required]})
        missing.to_csv(OUT / ("extended_missingness_" + label + ".csv"), index=False)
        print(label, "candidate:", len(events), "complete:", int(complete.sum()), flush=True)
    table = pd.DataFrame(summaries)
    table.to_csv(OUT / "extended_event_coverage.csv", index=False)
    totals = table.groupby("window")[["candidate_events", "matched_events", "complete_events", "excluded_events"]].sum()
    totals.to_csv(OUT / "extended_event_totals.csv")
    (OUT / "extended_history_status.json").write_text(json.dumps({
        "audit_date": "2026-10-27", "source": "Existing Alpha Vantage caches plus separate new Yahoo/yfinance price downloads",
        "primary_data_unchanged": True, "models_refitted": False,
        "limitations": ["Current-company selection remains", "Exact earnings release times remain unverified",
            "Vendor values are retrospective", "Complete cases do not guarantee error-free financial observations",
            "Expanded 2016-2025 targets may differ after fresh vendor adjustments; compare rather than silently replace"]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
