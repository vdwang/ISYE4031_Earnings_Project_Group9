"""Attempt public Yahoo earnings only for issuers absent from the bulk archives."""

from pathlib import Path
import sys
import time

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.build_sp500 import choose_earnings

CACHE = ROOT / "data/raw/research/sp500"
OUT = ROOT / "outputs/dataset_research/sp500"


def main():
    universe = pd.read_csv(ROOT / "research/sp500_universe.csv")
    earnings = choose_earnings()
    missing = universe.loc[~universe.ticker.isin(earnings.ticker), "ticker"]
    records = []
    audit = []
    destination = CACHE / "gap_earnings.csv"
    if destination.exists():
        records = pd.read_csv(destination).to_dict("records")
    for ticker in missing:
        try:
            raw = yf.Ticker(ticker).get_earnings_dates(limit=80)
            if raw is None or raw.empty:
                audit.append({"ticker": ticker, "status": "no_earnings_returned", "usable_rows_2010_2025": 0})
                continue
            raw = raw.reset_index()
            date_column = next(c for c in raw if "date" in c.lower())
            dates = pd.to_datetime(raw[date_column])
            if dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)
            frame = pd.DataFrame({"ticker": ticker, "source_symbol": ticker, "reported_date": dates.dt.normalize(),
                "estimated_eps": pd.to_numeric(raw["EPS Estimate"], errors="coerce"),
                "reported_eps": pd.to_numeric(raw["Reported EPS"], errors="coerce"), "earnings_source": "direct_yahoo_gap_fill"})
            frame = frame.loc[frame.reported_date.between("2010-01-01", "2025-12-31")].dropna(subset=["reported_eps", "estimated_eps"])
            records.extend(frame.to_dict("records"))
            audit.append({"ticker": ticker, "status": "downloaded", "usable_rows_2010_2025": len(frame)})
            print(ticker, "usable earnings", len(frame), flush=True)
        except Exception as exc:
            audit.append({"ticker": ticker, "status": "failed_" + type(exc).__name__, "usable_rows_2010_2025": 0})
            print(ticker, type(exc).__name__, flush=True)
            if "ratelimit" in type(exc).__name__.lower():
                break
        finally:
            time.sleep(1)
    pd.DataFrame(records, columns=["ticker", "source_symbol", "reported_date", "estimated_eps", "reported_eps", "earnings_source"]).drop_duplicates(["ticker", "reported_date"]).to_csv(destination, index=False)
    pd.DataFrame(audit).to_csv(OUT / "earnings_gap_attempts.csv", index=False)


if __name__ == "__main__":
    main()
