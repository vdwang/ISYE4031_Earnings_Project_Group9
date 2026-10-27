"""Cache expanded public research inputs without overwriting the original pilot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import PRICE_COLUMNS
from src.utils import valid_price_frame

CACHE = ROOT / "data/raw/research/expanded"
OUT = ROOT / "outputs/dataset_research/expanded"
UNIVERSE = ROOT / "research/universe.csv"
END = "2026-02-01"
START = "1995-11-01"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_earnings(universe, max_requests):
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    folder = CACHE / "earnings"
    folder.mkdir(parents=True, exist_ok=True)
    ledger_path = CACHE / "alpha_request_ledger.json"
    today = datetime.now(timezone.utc).date().isoformat()
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    already = ledger.get(today, 0)
    calls = 0
    blocked = False
    manifest = []
    for ticker in universe.ticker:
        source = ROOT / "data/raw/earnings" / (ticker + ".json")
        cache = folder / (ticker + ".json")
        payload = None
        status = "unavailable"
        if source.exists():
            payload = json.loads(source.read_text())
            path = source
            status = "original_cache"
        elif cache.exists():
            payload = json.loads(cache.read_text())
            path = cache
            status = "research_cache"
        elif blocked or calls >= max_requests or already + calls >= 25:
            status = "not_requested_budget_or_provider_limit"
        elif not api_key or api_key == "YOUR_KEY_HERE":
            status = "missing_api_key"
        else:
            calls += 1
            ledger[today] = already + calls
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
            try:
                response = requests.get("https://www.alphavantage.co/query", params={
                    "function": "EARNINGS", "symbol": ticker, "apikey": api_key}, timeout=45)
                response.raise_for_status()
                candidate = response.json()
                if isinstance(candidate.get("quarterlyEarnings"), list):
                    payload = candidate
                    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    path = cache
                    status = "downloaded"
                else:
                    # Stop on provider information/rate-limit responses; do not work around limits.
                    status = "provider_declined_no_quarterly_records"
                    blocked = True
            except (requests.RequestException, ValueError) as exc:
                status = "request_failed_" + type(exc).__name__
                blocked = True
            time.sleep(1.3)
        records = payload.get("quarterlyEarnings", []) if isinstance(payload, dict) else []
        row = {"ticker": ticker, "status": status, "quarterly_records": len(records),
            "sha256": digest(path) if payload is not None else "",
            "source": "https://www.alphavantage.co/documentation/#earnings"}
        manifest.append(row)
        print("Earnings", ticker, status, len(records), flush=True)
    pd.DataFrame(manifest).to_csv(OUT / "earnings_download_manifest.csv", index=False)


def collect_prices(universe):
    folder = CACHE / "prices"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    # XLRE/XLC have shorter histories. IYR/VOX proxies are explicit for early periods.
    tickers = list(dict.fromkeys([*universe.ticker, "SPY", *universe.sector_etf, "IYR", "VOX"]))
    for ticker in tickers:
        path = folder / (ticker + ".csv")
        original = ROOT / "data/raw/research/extended_prices" / (ticker + ".csv")
        status = "cached"
        try:
            if path.exists():
                frame = pd.read_csv(path)
            elif original.exists():
                frame = pd.read_csv(original)
                frame.to_csv(path, index=False)
                status = "reused_extended_cache"
            else:
                frame = yf.Ticker(ticker).history(start=START, end=END, interval="1d",
                    auto_adjust=False, actions=False)
                if frame.empty:
                    raise ValueError("Empty price history")
                frame = frame.reset_index()
                frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame]
                frame = frame.rename(columns={"adj_close": "adjusted_close"})
                frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
                frame["ticker"] = ticker
                frame = frame.reindex(columns=PRICE_COLUMNS)
                if not valid_price_frame(frame, ticker):
                    raise ValueError("Invalid prices")
                frame.to_csv(path, index=False)
                status = "downloaded"
                time.sleep(1)
            if not valid_price_frame(frame, ticker):
                raise ValueError("Invalid cached prices")
            rows.append({"ticker": ticker, "status": status, "rows": len(frame),
                "first_date": frame.date.min(), "last_date": frame.date.max(),
                "missing_adjusted_close": int(frame.adjusted_close.isna().sum()),
                "duplicate_dates": int(frame.date.duplicated().sum()), "sha256": digest(path)})
            print("Prices", ticker, len(frame), status, flush=True)
        except Exception as exc:
            rows.append({"ticker": ticker, "status": "failed_" + type(exc).__name__, "rows": 0})
            print("Prices", ticker, "unavailable", type(exc).__name__, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "price_download_manifest.csv", index=False)


def collect_context():
    folder = CACHE / "context"
    folder.mkdir(parents=True, exist_ok=True)
    sources = {
        "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
        "VVIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
        "DGS10": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=1995-01-01&coed=2025-12-31",
        "DGS2": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2&cosd=1995-01-01&coed=2025-12-31",
    }
    rows = []
    for name, url in sources.items():
        path = folder / (name + ".csv")
        try:
            if not path.exists():
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                if len(response.content) > 5_000_000:
                    raise ValueError("Unexpected context file size")
                path.write_bytes(response.content)
            frame = pd.read_csv(path)
            rows.append({"series": name, "status": "cached_or_downloaded", "rows": len(frame),
                "columns": ",".join(frame.columns), "sha256": digest(path), "source_url": url})
            print("Context", name, len(frame), flush=True)
        except (requests.RequestException, ValueError) as exc:
            rows.append({"series": name, "status": "failed_" + type(exc).__name__, "source_url": url})
            print("Context", name, "unavailable", type(exc).__name__, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "context_download_manifest.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-earnings-requests", type=int, default=20)
    parser.add_argument("--only", choices=["earnings", "prices", "context", "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(UNIVERSE)
    if args.only in ["earnings", "all"]:
        collect_earnings(universe, max(0, min(args.max_earnings_requests, 25)))
    if args.only in ["prices", "all"]:
        collect_prices(universe)
    if args.only in ["context", "all"]:
        collect_context()


if __name__ == "__main__":
    main()
