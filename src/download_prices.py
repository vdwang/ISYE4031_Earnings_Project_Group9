"""Download daily Yahoo Finance prices and keep a local cache per ticker."""

import json
import warnings

import pandas as pd
import yfinance as yf

from .config import PRICE_COLUMNS, PRICE_END_EXCLUSIVE, PRICE_START, PRICE_TICKERS, PRICES_DIR
from .utils import numeric, valid_price_frame


def _cached_prices(ticker: str) -> pd.DataFrame | None:
    csv_path = PRICES_DIR / f"{ticker}.csv"
    metadata_path = PRICES_DIR / f"{ticker}.json"
    if not csv_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(csv_path)
        if (
            metadata.get("ticker") == ticker
            and metadata.get("start") == PRICE_START
            and metadata.get("end_exclusive") == PRICE_END_EXCLUSIVE
            and valid_price_frame(frame, ticker)
        ):
            return frame
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.warn(f"Invalid price cache for {ticker}: {exc}")
    return None


def download_prices(tickers: list[str] = PRICE_TICKERS, refresh: bool = False) -> dict[str, pd.DataFrame]:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    output: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        cached = None if refresh else _cached_prices(ticker)
        if cached is not None:
            print(f"Using cached prices: {ticker}")
            output[ticker] = cached
            continue

        try:
            history = yf.Ticker(ticker).history(
                start=PRICE_START,
                end=PRICE_END_EXCLUSIVE,
                interval="1d",
                auto_adjust=False,
                actions=False,
                raise_errors=True,
            )
            if history.empty:
                warnings.warn(f"No daily prices returned for {ticker}.")
                continue
            history = history.reset_index()
            history.columns = [str(column).strip().lower().replace(" ", "_") for column in history.columns]
            history = history.rename(columns={"adj_close": "adjusted_close"})
            if "date" not in history:
                raise ValueError("Yahoo response has no date column")
            history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            history["ticker"] = ticker
            frame = history.reindex(columns=PRICE_COLUMNS)
            for column in ("open", "high", "low", "close", "adjusted_close", "volume"):
                frame[column] = numeric(frame[column])
            frame = frame.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
            if not valid_price_frame(frame, ticker):
                raise ValueError("Yahoo response failed price validation")

            frame.to_csv(PRICES_DIR / f"{ticker}.csv", index=False)
            (PRICES_DIR / f"{ticker}.json").write_text(
                json.dumps({"ticker": ticker, "start": PRICE_START, "end_exclusive": PRICE_END_EXCLUSIVE}, indent=2),
                encoding="utf-8",
            )
            output[ticker] = frame
            print(f"Downloaded {len(frame)} daily prices: {ticker}")
        except Exception as exc:  # One failed ticker should not destroy successful downloads.
            warnings.warn(f"Prices unavailable for {ticker}: {exc}")

    if "SPY" not in output:
        raise RuntimeError("SPY prices are required for market-adjusted returns; check Yahoo access or its cache.")
    if not any(ticker != "SPY" for ticker in output):
        raise RuntimeError("No stock price series were available.")
    return output
