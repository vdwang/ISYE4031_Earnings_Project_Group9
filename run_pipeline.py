"""Run downloads, event construction, feature creation, cleaning, and summaries."""

import argparse

from src.build_dataset import build_events
from src.clean_data import clean_and_save
from src.config import PRICE_TICKERS, TICKERS, ensure_directories
from src.download_earnings import download_earnings
from src.download_prices import download_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ISYE 4031 earnings event dataset")
    parser.add_argument("--refresh", action="store_true", help="Redownload valid cached responses and prices")
    args = parser.parse_args()
    ensure_directories()
    earnings = download_earnings(TICKERS, refresh=args.refresh)
    prices = download_prices(PRICE_TICKERS, refresh=args.refresh)
    events, counts = build_events(earnings, prices)
    clean_and_save(events, counts)
    print("Pipeline complete. Open notebooks/01_preliminary_analysis.ipynb for plots and baseline.")


if __name__ == "__main__":
    main()
