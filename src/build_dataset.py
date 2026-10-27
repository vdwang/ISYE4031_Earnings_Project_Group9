"""Join quarterly EPS events to the first subsequent stock trading session."""

import numpy as np
import pandas as pd

from .config import EVENT_END, EVENT_START, FEATURE_COLUMNS, TARGET_COLUMNS
from .features import add_price_features, prepare_prices
from .utils import numeric

EVENT_COLUMNS = [
    "ticker", "fiscal_date_ending", "reported_date", "reaction_day",
    "prediction_start_date", "prediction_end_date", *FEATURE_COLUMNS, *TARGET_COLUMNS,
]


def build_events(earnings: pd.DataFrame, prices: dict[str, pd.DataFrame],
                 event_start: str = EVENT_START, event_end: str = EVENT_END) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_count = len(earnings)
    work = earnings.copy()
    work = work.drop_duplicates()
    work["reported_date"] = pd.to_datetime(work["reportedDate"], errors="coerce").dt.normalize()
    work["fiscal_date_ending"] = pd.to_datetime(work["fiscalDateEnding"], errors="coerce").dt.normalize()
    for original in ("reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"):
        work[original] = numeric(work[original])
    work = work.sort_values(["ticker", "reported_date", "fiscal_date_ending"], kind="stable")
    work["previous_earnings_surprise_pct"] = work.groupby("ticker")["surprisePercentage"].shift(1)
    # Keep earlier records long enough to supply the first in-window lag.
    work = work.loc[work["reported_date"].between(event_start, event_end)].copy()

    prepared = {ticker: prepare_prices(frame) for ticker, frame in prices.items()}
    spy = prepared["SPY"]
    rows: list[dict] = []
    matched_count = 0
    for _, source in work.iterrows():
        event = {column: np.nan for column in EVENT_COLUMNS}
        event.update({
            "ticker": source["ticker"],
            "fiscal_date_ending": source["fiscal_date_ending"],
            "reported_date": source["reported_date"],
            "earnings_surprise_pct": source["surprisePercentage"],
            "reported_eps": source["reportedEPS"],
            "estimated_eps": source["estimatedEPS"],
            "earnings_surprise": source["surprise"],
            "previous_earnings_surprise_pct": source["previous_earnings_surprise_pct"],
        })
        stock = prepared.get(source["ticker"])
        if stock is not None and not stock.empty:
            candidates = np.flatnonzero(stock["date"].gt(source["reported_date"]).to_numpy())
            if len(candidates):
                i = int(candidates[0])
                # A long gap is almost certainly a missing price history, not a reaction day.
                if (stock.loc[i, "date"] - source["reported_date"]).days <= 7:
                    event = add_price_features(event, stock, spy, i)
                    matched_count += 1
        rows.append(event)

    result = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    counts = {"raw_event_count": raw_count, "events_2016_2025": len(work), "events_matched_to_prices": matched_count}
    return result, counts
