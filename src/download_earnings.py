"""Download and cache Alpha Vantage quarterly EPS records."""

import json
import os
import time
import warnings

import pandas as pd
import requests
from dotenv import load_dotenv

from .config import EARNINGS_COLUMNS, EARNINGS_DIR, ROOT, TICKERS
from .utils import numeric

API_URL = "https://www.alphavantage.co/query"


def _valid_payload(payload: object) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("quarterlyEarnings"), list)


def _safe_error(message: object, api_key: str) -> str:
    return str(message).replace(api_key, "[redacted]") if api_key else str(message)


def download_earnings(tickers: list[str] = TICKERS, refresh: bool = False) -> pd.DataFrame:
    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    records: list[dict] = []
    failures: list[str] = []
    last_request_at = 0.0

    for ticker in tickers:
        cache = EARNINGS_DIR / f"{ticker}.json"
        payload = None
        if cache.exists() and not refresh:
            try:
                candidate = json.loads(cache.read_text(encoding="utf-8"))
                if _valid_payload(candidate):
                    payload = candidate
                else:
                    warnings.warn(f"Ignoring malformed earnings cache for {ticker}.")
            except (OSError, json.JSONDecodeError) as exc:
                warnings.warn(f"Cannot read earnings cache for {ticker}: {exc}")

        if payload is None:
            if not api_key or api_key == "YOUR_KEY_HERE":
                failures.append(f"{ticker}: missing ALPHA_VANTAGE_API_KEY in .env")
                continue
            try:
                # Alpha Vantage's free tier asks clients to stay below one request/second.
                delay = 1.2 - (time.monotonic() - last_request_at)
                if delay > 0:
                    time.sleep(delay)
                last_request_at = time.monotonic()
                response = requests.get(
                    API_URL,
                    params={"function": "EARNINGS", "symbol": ticker, "apikey": api_key},
                    timeout=30,
                )
                response.raise_for_status()
                candidate = response.json()
                if not _valid_payload(candidate):
                    reason = next((str(candidate[k]) for k in ("Note", "Information", "Error Message") if isinstance(candidate, dict) and k in candidate), "missing quarterlyEarnings")
                    failures.append(f"{ticker}: {_safe_error(reason, api_key)}")
                    continue
                payload = candidate
                cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"Downloaded earnings: {ticker}")
            except (requests.RequestException, ValueError, OSError) as exc:
                failures.append(f"{ticker}: {_safe_error(exc, api_key)}")
                continue
        else:
            print(f"Using cached earnings: {ticker}")

        for item in payload["quarterlyEarnings"]:
            if isinstance(item, dict):
                records.append({key: ticker if key == "ticker" else item.get(key) for key in EARNINGS_COLUMNS})

    for failure in failures:
        warnings.warn(f"Earnings unavailable - {failure}")
    if not records:
        raise RuntimeError("No quarterly earnings were available. Add a valid key to .env or valid cached JSON files.")

    frame = pd.DataFrame.from_records(records, columns=EARNINGS_COLUMNS)
    for column in ("reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"):
        frame[column] = numeric(frame[column])
    frame.to_csv(EARNINGS_DIR / "quarterly_earnings.csv", index=False)
    print(f"Saved {len(frame)} raw quarterly earnings records.")
    return frame
