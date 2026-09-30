"""Change TICKERS here when expanding the study."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EARNINGS_DIR = DATA / "raw" / "earnings"
PRICES_DIR = DATA / "raw" / "prices"
PROCESSED_DIR = DATA / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "figures"

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "XOM", "JNJ", "WMT"]
PRICE_TICKERS = [*TICKERS, "SPY"]
PRICE_START = "2015-11-01"
PRICE_END_EXCLUSIVE = "2026-02-01"  # yfinance's end argument is exclusive.
EVENT_START = "2016-01-01"
EVENT_END = "2025-12-31"

EARNINGS_COLUMNS = [
    "ticker", "fiscalDateEnding", "reportedDate", "reportedEPS",
    "estimatedEPS", "surprise", "surprisePercentage",
]
PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume"]
FEATURE_COLUMNS = [
    "earnings_surprise_pct", "reported_eps", "estimated_eps", "earnings_surprise",
    "reaction_day_return", "abnormal_volume", "return_5d_before",
    "return_20d_before", "volatility_20d", "spy_return_5d_before",
    "previous_earnings_surprise_pct",
]
TARGET_COLUMNS = [
    "future_5d_return", "positive_return", "positive_return_amount",
    "spy_future_5d_return", "market_adjusted_5d_return",
]


def ensure_directories() -> None:
    for directory in (EARNINGS_DIR, PRICES_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
