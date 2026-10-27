"""Vectorized daily features for a large, irregularly spaced earnings panel."""

import numpy as np
import pandas as pd

from src.features import prepare_prices


def daily_features(stock_raw, spy_raw, sector_raw, fallback_raw=None):
    stock = prepare_prices(stock_raw).set_index("date")
    spy = prepare_prices(spy_raw).set_index("date").price
    p = stock.price
    r = p.pct_change(fill_method=None)
    m = spy.reindex(stock.index)
    mr = spy.pct_change(fill_method=None).reindex(stock.index)
    out = pd.DataFrame(index=stock.index)
    out["reaction_day_return"] = r
    out["abnormal_volume"] = stock.volume / stock.volume.rolling(20, min_periods=20).mean().shift(1)
    out["return_5d_before"] = p.shift(1) / p.shift(6) - 1
    out["return_20d_before"] = p.shift(1) / p.shift(21) - 1
    out["volatility_20d"] = r.rolling(20, min_periods=20).std(ddof=1).shift(1)
    out["spy_return_5d_before"] = (spy.shift(1) / spy.shift(6) - 1).reindex(stock.index)
    out["stock_return_60d_before"] = p.shift(1) / p.shift(61) - 1
    out["stock_volatility_60d_before"] = r.rolling(60, min_periods=60).std(ddof=1).shift(1)
    cov = r.rolling(60, min_periods=60).cov(mr)
    variance = mr.rolling(60, min_periods=60).var(ddof=1)
    out["market_beta_60d_before"] = (cov / variance.where(variance.gt(0))).shift(1)
    out["reaction_day_market_adjusted_return"] = r - mr
    proxy_frames = []
    for raw in [sector_raw, fallback_raw]:
        if raw is None or raw.empty:
            continue
        sector = prepare_prices(raw).set_index("date")
        sp = sector.price
        values = pd.DataFrame(index=sp.index)
        values["prior_return"] = sp.shift(1) / sp.shift(21) - 1
        values["reaction_return"] = sp.pct_change(fill_method=None)
        values["sector_source_date"] = pd.Series(sp.index, index=sp.index).shift(1)
        values["sector_proxy_used"] = str(sector.ticker.iloc[0])
        valid = values[["prior_return", "reaction_return"]].notna().all(axis=1)
        values.loc[~valid, :] = np.nan
        proxy_frames.append(values.reindex(stock.index))
    if proxy_frames:
        sector_values = proxy_frames[0]
        for other in proxy_frames[1:]:
            sector_values = sector_values.combine_first(other)
        out["sector_relative_return_20d_before"] = out.return_20d_before - sector_values.prior_return
        out["reaction_day_sector_adjusted_return"] = r - sector_values.reaction_return
        out["sector_source_date"] = sector_values.sector_source_date
        out["sector_proxy_used"] = sector_values.sector_proxy_used
    else:
        out["sector_relative_return_20d_before"] = np.nan
        out["reaction_day_sector_adjusted_return"] = np.nan
        out["sector_source_date"] = pd.NaT
        out["sector_proxy_used"] = None
    out["future_5d_return"] = p.shift(-5) / p - 1
    out["spy_future_5d_return"] = m.shift(-5) / m - 1
    out["market_adjusted_5d_return"] = out.future_5d_return - out.spy_future_5d_return
    out["positive_return"] = np.where(out.future_5d_return.notna(), out.future_5d_return.gt(0).astype(float), np.nan)
    out["prediction_start_date"] = pd.Series(stock.index, index=stock.index).shift(-1)
    out["prediction_end_date"] = pd.Series(stock.index, index=stock.index).shift(-5)
    return out.rename_axis("reaction_day").reset_index()


def add_earnings_history(frame):
    result = frame.sort_values(["ticker", "reported_date"]).copy()
    if result.duplicated(["ticker", "reported_date"]).any():
        raise ValueError("Duplicate issuer announcement dates must be resolved first")
    for ticker, group in result.groupby("ticker"):
        lag = group.earnings_surprise_pct.shift(1)
        result.loc[group.index, "previous_earnings_surprise_pct"] = lag
        result.loc[group.index, "prior_surprise_mean_4q"] = lag.rolling(4, min_periods=4).mean()
        result.loc[group.index, "prior_surprise_std_4q"] = lag.rolling(4, min_periods=4).std(ddof=1)
        beats = lag.gt(0).astype(float).where(lag.notna())
        result.loc[group.index, "prior_beat_rate_4q"] = beats.rolling(4, min_periods=4).mean()
        result.loc[group.index, "days_since_previous_earnings"] = group.reported_date.diff().dt.days
    return result
