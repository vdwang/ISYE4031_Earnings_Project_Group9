"""Event-level features computed with observations available by reaction-day close."""

import numpy as np
import pandas as pd

from .utils import numeric, price_series


def prepare_prices(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    result["price"] = price_series(result)
    result["volume"] = numeric(result["volume"])
    return result


def _ratio_return(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0 or numerator <= 0:
        return np.nan
    return float(numerator / denominator - 1)


def add_price_features(event: dict, stock: pd.DataFrame, spy: pd.DataFrame, reaction_index: int) -> dict:
    """Use zero-based stock indices; the five target sessions are i+1 through i+5."""
    i = reaction_index
    prices = stock["price"]
    volumes = stock["volume"]
    reaction = stock.loc[i, "date"]
    event["reaction_day"] = reaction

    if i + 5 < len(stock):
        end_day = stock.loc[i + 5, "date"]
        event["prediction_start_date"] = stock.loc[i + 1, "date"]
        event["prediction_end_date"] = end_day
        event["future_5d_return"] = _ratio_return(prices.iloc[i + 5], prices.iloc[i])
    else:
        end_day = pd.NaT
        event["prediction_start_date"] = stock.loc[i + 1, "date"] if i + 1 < len(stock) else pd.NaT
        event["prediction_end_date"] = pd.NaT
        event["future_5d_return"] = np.nan

    event["reaction_day_return"] = _ratio_return(prices.iloc[i], prices.iloc[i - 1]) if i >= 1 else np.nan
    if i >= 20:
        prior_volume = volumes.iloc[i - 20:i]
        mean_volume = prior_volume.mean() if prior_volume.notna().all() else np.nan
        event["abnormal_volume"] = float(volumes.iloc[i] / mean_volume) if pd.notna(mean_volume) and mean_volume > 0 and pd.notna(volumes.iloc[i]) else np.nan
    else:
        event["abnormal_volume"] = np.nan

    # A five-day lookback ends before the reaction session: close[i-1] / close[i-6].
    event["return_5d_before"] = _ratio_return(prices.iloc[i - 1], prices.iloc[i - 6]) if i >= 6 else np.nan
    event["return_20d_before"] = _ratio_return(prices.iloc[i - 1], prices.iloc[i - 21]) if i >= 21 else np.nan
    if i >= 21:
        prior_prices = prices.iloc[i - 21:i]
        daily_returns = prior_prices.pct_change(fill_method=None).iloc[1:]
        event["volatility_20d"] = float(daily_returns.std(ddof=1)) if daily_returns.notna().all() else np.nan
    else:
        event["volatility_20d"] = np.nan

    spy_by_date = spy.set_index("date")["price"]
    if reaction in spy_by_date.index:
        spy_i = spy.index[spy["date"].eq(reaction)][0]
        event["spy_return_5d_before"] = _ratio_return(spy["price"].iloc[spy_i - 1], spy["price"].iloc[spy_i - 6]) if spy_i >= 6 else np.nan
        event["spy_future_5d_return"] = _ratio_return(spy_by_date.get(end_day, np.nan), spy_by_date.loc[reaction]) if pd.notna(end_day) else np.nan
    else:
        event["spy_return_5d_before"] = np.nan
        event["spy_future_5d_return"] = np.nan

    future = event["future_5d_return"]
    event["positive_return"] = int(future > 0) if pd.notna(future) else np.nan
    event["positive_return_amount"] = max(float(future), 0.0) if pd.notna(future) else np.nan
    spy_future = event["spy_future_5d_return"]
    event["market_adjusted_5d_return"] = future - spy_future if pd.notna(future) and pd.notna(spy_future) else np.nan
    return event
