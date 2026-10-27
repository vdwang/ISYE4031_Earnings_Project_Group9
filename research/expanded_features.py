"""Time-ordered enrichment for the expanded earnings panel.

All future returns remain targets. Every added historical window stops before
reaction day; earnings histories exclude the current announcement.
"""

import numpy as np
import pandas as pd

from src.features import prepare_prices

# Raw EPS levels and dollar surprise are kept in the private audit table, not
# counted as additional independent model inputs in the shared derived table.
BASE_FEATURES = ["earnings_surprise_pct", "reaction_day_return", "abnormal_volume",
    "return_5d_before", "return_20d_before", "volatility_20d",
    "spy_return_5d_before", "previous_earnings_surprise_pct"]
ADDED_FEATURES = ["prior_surprise_mean_4q", "prior_surprise_std_4q", "prior_beat_rate_4q",
    "days_since_previous_earnings", "stock_return_60d_before", "stock_volatility_60d_before",
    "market_beta_60d_before", "sector_relative_return_20d_before",
    "reaction_day_market_adjusted_return", "reaction_day_sector_adjusted_return",
    "vix_prior_close", "vix_change_5_sessions", "vvix_prior_close",
    "treasury_10y_lag7d", "yield_curve_10y_minus_2y", "vix_reaction_interaction",
    "surprise_reaction_interaction"]
MODEL_FEATURES = BASE_FEATURES + ADDED_FEATURES


def earnings_history(earnings):
    frame = earnings.copy()
    frame["reported_date"] = pd.to_datetime(frame.reportedDate, errors="coerce")
    frame["surprise_pct"] = pd.to_numeric(frame.surprisePercentage, errors="coerce")
    frame["fiscal_date_ending"] = pd.to_datetime(frame.fiscalDateEnding, errors="coerce")
    frame = frame.sort_values(["ticker", "reported_date", "fiscal_date_ending"])
    rows = []
    for ticker, group in frame.groupby("ticker"):
        # Group same-date announcements: none may enter another's historical features.
        for report, current in group.groupby("reported_date"):
            past = group.loc[group.reported_date.lt(report)].tail(4)
            surprises = past.surprise_pct
            valid_four = len(past) == 4 and np.isfinite(surprises).all()
            previous = group.loc[group.reported_date.lt(report), "reported_date"].max()
            for _, event in current.iterrows():
                rows.append({"ticker": ticker, "reported_date": report,
                    "fiscal_date_ending": event.fiscal_date_ending,
                    "prior_surprise_mean_4q": surprises.mean() if valid_four else np.nan,
                    "prior_surprise_std_4q": surprises.std(ddof=1) if valid_four else np.nan,
                    "prior_beat_rate_4q": surprises.gt(0).mean() if valid_four else np.nan,
                    "days_since_previous_earnings": (report - previous).days if pd.notna(previous) else np.nan})
    return pd.DataFrame(rows)


def normalize_context(frame, name):
    date_column = next(c for c in frame if c.upper() in ["DATE", "OBSERVATION_DATE"])
    value_column = "CLOSE" if "CLOSE" in frame else name
    output = pd.DataFrame({"date": pd.to_datetime(frame[date_column], errors="coerce"),
        "value": pd.to_numeric(frame[value_column], errors="coerce")})
    return output.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)


def lagged_context(events, context):
    result = events.copy()
    for name, lag_days, max_age in [("VIX", 1, 7), ("VVIX", 1, 7), ("DGS10", 7, 14), ("DGS2", 7, 14)]:
        source = context[name].copy()
        if name == "VIX":
            source["vix_change_5_sessions"] = source.value / source.value.shift(5) - 1
        source = source.rename(columns={"date": name.lower() + "_source_date", "value": name.lower() + "_value"})
        source_date = name.lower() + "_source_date"
        left = result[["reaction_day"]].copy()
        left["_row"] = np.arange(len(result))
        left["_cutoff"] = left.reaction_day.to_numpy(dtype="datetime64[ns]") - np.timedelta64(lag_days, "D")
        usable = left.dropna(subset=["_cutoff"]).sort_values("_cutoff")
        joined = pd.merge_asof(usable, source.sort_values(source_date), left_on="_cutoff", right_on=source_date,
            direction="backward", allow_exact_matches=True)
        joined["age"] = (joined.reaction_day - joined[source_date]).dt.days
        valid = joined.age.between(lag_days, max_age)
        for column in [name.lower() + "_value", "vix_change_5_sessions"] if name == "VIX" else [name.lower() + "_value"]:
            result[column] = np.nan
            result.loc[joined.loc[valid, "_row"], column] = joined.loc[valid, column].to_numpy()
        result[source_date] = pd.NaT
        result.loc[joined.loc[valid, "_row"], source_date] = joined.loc[valid, source_date].to_numpy()
    result = result.rename(columns={"vix_value": "vix_prior_close", "vvix_value": "vvix_prior_close",
        "dgs10_value": "treasury_10y_lag7d", "dgs2_value": "treasury_2y_lag7d"})
    result["yield_curve_10y_minus_2y"] = result.treasury_10y_lag7d - result.treasury_2y_lag7d
    return result


def enrich_events(events, earnings, prices, universe, context):
    result = events.copy()
    for name in ["reported_date", "fiscal_date_ending", "reaction_day", "prediction_end_date"]:
        result[name] = pd.to_datetime(result[name])
    history = earnings_history(earnings)
    result = result.merge(history, on=["ticker", "reported_date", "fiscal_date_ending"], how="left", validate="one_to_one")
    result = result.merge(universe, on="ticker", how="left", validate="many_to_one")
    prepared = {ticker: prepare_prices(frame) for ticker, frame in prices.items()}
    spy = prepared["SPY"].set_index("date").price
    spy_returns = spy.pct_change(fill_method=None)
    records = []
    for _, event in result.iterrows():
        row = {}
        stock = prepared.get(event.ticker)
        reaction = event.reaction_day
        if stock is not None and pd.notna(reaction):
            past = stock.loc[stock.date.lt(reaction)].set_index("date")
            if len(past) >= 61:
                row["stock_return_60d_before"] = past.price.iloc[-1] / past.price.iloc[-61] - 1
                daily = past.price.pct_change(fill_method=None).tail(60)
                row["stock_volatility_60d_before"] = daily.std(ddof=1) if daily.notna().all() else np.nan
                aligned = pd.concat([daily.rename("stock"), spy_returns.rename("market")], axis=1).reindex(daily.index)
                if aligned.notna().all().all() and aligned.market.var(ddof=1) > 0:
                    row["market_beta_60d_before"] = aligned.stock.cov(aligned.market) / aligned.market.var(ddof=1)
            if reaction in spy.index:
                previous_spy = spy.loc[spy.index < reaction]
                if len(previous_spy):
                    row["reaction_day_market_adjusted_return"] = event.reaction_day_return - (spy.loc[reaction] / previous_spy.iloc[-1] - 1)
            candidates = [event.sector_etf]
            if event.sector_etf == "XLC":
                candidates.append("VOX")
            if event.sector_etf == "XLRE":
                candidates.append("IYR")
            for symbol in candidates:
                sector = prepared.get(symbol)
                if sector is None:
                    continue
                sector = sector.set_index("date").price
                prior = sector.loc[sector.index < reaction]
                if len(prior) >= 21 and reaction in sector.index and prior.iloc[-1] > 0 and prior.iloc[-21] > 0:
                    sector_return = prior.iloc[-1] / prior.iloc[-21] - 1
                    row["sector_relative_return_20d_before"] = event.return_20d_before - sector_return
                    row["reaction_day_sector_adjusted_return"] = event.reaction_day_return - (sector.loc[reaction] / prior.iloc[-1] - 1)
                    row["sector_proxy_used"] = symbol
                    row["sector_source_date"] = prior.index[-1]
                    break
        records.append(row)
    extra = pd.DataFrame(records, index=result.index)
    result = pd.concat([result, extra], axis=1)
    result = lagged_context(result, context)
    result["vix_reaction_interaction"] = result.vix_prior_close * result.reaction_day_return
    result["surprise_reaction_interaction"] = result.earnings_surprise_pct * result.reaction_day_return
    for name in MODEL_FEATURES:
        if name not in result:
            result[name] = np.nan
    result[MODEL_FEATURES] = result[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    return result
