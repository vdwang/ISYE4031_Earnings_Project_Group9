"""Calendar lags and completed-event lags with explicit availability dates."""

import numpy as np
import pandas as pd

from src.features import prepare_prices

AR_FEATURES = ["return_lag1", "return_lag2", "return_lag5"]
CALENDAR_FEATURES = ["quarter_return_lag1", "quarter_return_lag4", "year_return_lag1"]
EXOG_FEATURES = [*CALENDAR_FEATURES, "spy_return_known"]
DAILY_FEATURES = [*AR_FEATURES, *EXOG_FEATURES]
EVENT_FEATURES = ["completed_event_return_lag1", "completed_event_return_lag4", *CALENDAR_FEATURES,
    "earnings_surprise_pct", "reaction_day_return"]


def split_labels(dates, target_ends):
    result = pd.Series("purged_boundary", index=dates.index, dtype="object")
    for name, start, end in [("train", "2010-01-01", "2023-01-01"),
        ("validation", "2023-01-01", "2025-01-01"), ("exploratory_test", "2025-01-01", "2026-01-01")]:
        keep = dates.ge(start) & dates.lt(end) & target_ends.lt(end)
        result.loc[keep] = name
    return result


def completed_periods(price, market, frequency):
    """Use a period only if its final stock observation matches market period end."""
    groups = price.groupby(price.index.to_period(frequency))
    close = groups.agg(lambda x: x.iloc[-1])
    source = pd.Series(price.index, index=price.index).groupby(price.index.to_period(frequency)).last()
    expected = pd.Series(market.index, index=market.index).groupby(market.index.to_period(frequency)).last()
    closed = source.index.to_timestamp(how="end").normalize() <= market.index.max().normalize()
    complete = source.eq(expected.reindex(source.index)) & closed
    index = pd.period_range(close.index.min(), close.index.max(), freq=frequency)
    close = close.where(complete).reindex(index)
    source = source.where(complete).reindex(index)
    return pd.DataFrame({"return": close / close.shift(1) - 1, "source_date": source})


def company_daily(raw, spy_raw):
    stock = prepare_prices(raw).set_index("date")
    market = prepare_prices(spy_raw).set_index("date").price
    price = stock.price.where(stock.price.gt(0))
    returns = price.pct_change(fill_method=None)
    result = pd.DataFrame(index=price.index)
    result["return_lag1"] = returns
    result["return_lag2"] = returns.shift(1)
    result["return_lag5"] = returns.shift(4)
    result["spy_return_known"] = market.pct_change(fill_method=None).reindex(price.index)
    quarter = completed_periods(price, market, "Q")
    year = completed_periods(price, market, "Y")
    for lag, name in [(1, "quarter_return_lag1"), (4, "quarter_return_lag4")]:
        previous = price.index.to_period("Q") - lag
        result[name] = quarter["return"].reindex(previous).to_numpy()
        result[name + "_source_date"] = quarter.source_date.reindex(previous).to_numpy()
    previous_year = price.index.to_period("Y") - 1
    result["year_return_lag1"] = year["return"].reindex(previous_year).to_numpy()
    result["year_return_lag1_source_date"] = year.source_date.reindex(previous_year).to_numpy()
    result["future_return_1d"] = price.shift(-1) / price - 1
    result["future_return_5d"] = price.shift(-5) / price - 1
    result["target_end_date_1d"] = pd.Series(price.index, index=price.index).shift(-1)
    result["target_end_date_5d"] = pd.Series(price.index, index=price.index).shift(-5)
    result = result.replace([np.inf, -np.inf], np.nan).rename_axis("date").reset_index()
    return result


def completed_event_lags(current, history):
    """Lag realized event returns by availability, never by a naive row shift."""
    result = current.copy().reset_index(drop=True)
    result["_row"] = np.arange(len(result))
    frames = []
    for ticker, sample in result.groupby("ticker"):
        past = history.loc[history.ticker.eq(ticker)].dropna(subset=["prediction_end_date", "future_5d_return"])
        past = past.sort_values(["prediction_end_date", "reported_date"]).copy()
        values = pd.DataFrame({"available_date": past.prediction_end_date,
            "completed_event_return_lag1": past.future_5d_return,
            "completed_event_return_lag4": past.future_5d_return.shift(3),
            "event_lag1_source_date": past.prediction_end_date,
            "event_lag4_source_date": past.prediction_end_date.shift(3)})
        joined = pd.merge_asof(sample.sort_values("reaction_day"), values.sort_values("available_date"),
            left_on="reaction_day", right_on="available_date", direction="backward", allow_exact_matches=True)
        frames.append(joined)
    return pd.concat(frames).sort_values("_row").drop(columns=["_row", "available_date"]).reset_index(drop=True)
