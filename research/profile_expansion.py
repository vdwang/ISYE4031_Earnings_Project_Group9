"""Verify every shared event and create the schema and coverage figure."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.expanded_features import MODEL_FEATURES

OUT = ROOT / "outputs/dataset_research/expanded"
DATA = ROOT / "research/datasets/expanded_modeling_dataset.csv"

FEATURE_DOCS = {
    "earnings_surprise_pct": ("percent", "Alpha Vantage", "Current announced EPS surprise percentage; potentially unstable near a zero estimate"),
    "reaction_day_return": ("fraction", "Yahoo Finance", "Adjusted close on reaction day divided by the preceding close minus one"),
    "abnormal_volume": ("ratio", "Yahoo Finance", "Reaction-day volume divided by the mean volume of the previous 20 sessions"),
    "return_5d_before": ("fraction", "Yahoo Finance", "Five-session return ending at the close before reaction day"),
    "return_20d_before": ("fraction", "Yahoo Finance", "Twenty-session return ending at the close before reaction day"),
    "volatility_20d": ("fraction per session", "Yahoo Finance", "Sample standard deviation of the previous 20 daily returns excluding reaction day"),
    "spy_return_5d_before": ("fraction", "Yahoo Finance", "Five-session SPY return ending before reaction day"),
    "previous_earnings_surprise_pct": ("percent", "Alpha Vantage", "Surprise at the preceding earnings announcement for this company"),
    "prior_surprise_mean_4q": ("percent", "Alpha Vantage", "Mean surprise of the previous four reported earnings records; excludes current and same-date announcements"),
    "prior_surprise_std_4q": ("percentage points", "Alpha Vantage", "Sample standard deviation of the previous four surprises; all four required"),
    "prior_beat_rate_4q": ("fraction", "Alpha Vantage", "Fraction of the previous four reported events with strictly positive surprise; ties are not beats"),
    "days_since_previous_earnings": ("calendar days", "Alpha Vantage", "Days since the preceding strictly earlier announcement"),
    "stock_return_60d_before": ("fraction", "Yahoo Finance", "Sixty-session return ending before reaction day"),
    "stock_volatility_60d_before": ("fraction per session", "Yahoo Finance", "Sample standard deviation of the previous 60 daily returns excluding reaction day"),
    "market_beta_60d_before": ("ratio", "Yahoo Finance", "Stock/SPY return covariance divided by SPY return variance on the same 60 prior stock-session dates"),
    "sector_relative_return_20d_before": ("fraction", "Yahoo Finance", "Prior 20-session stock return minus prior 20-session proxy ETF return"),
    "reaction_day_market_adjusted_return": ("fraction", "Yahoo Finance", "Reaction-day stock return minus same-day SPY return"),
    "reaction_day_sector_adjusted_return": ("fraction", "Yahoo Finance", "Reaction-day stock return minus same-day proxy ETF return"),
    "vix_prior_close": ("index points", "Cboe", "Latest VIX close dated strictly before reaction day; rejected if more than seven calendar days old"),
    "vix_change_5_sessions": ("fraction", "Cboe", "Five-observation VIX change ending on the selected prior VIX date"),
    "vvix_prior_close": ("index points", "Cboe", "Latest VVIX close dated strictly before reaction day; rejected if more than seven calendar days old"),
    "treasury_10y_lag7d": ("percent yield", "Federal Reserve via FRED", "Latest 10-year Treasury yield dated at least seven calendar days before reaction; reject if older than 14 days"),
    "yield_curve_10y_minus_2y": ("percentage points", "Federal Reserve via FRED", "Lagged 10-year yield minus lagged 2-year yield; selected dates are retained separately"),
    "vix_reaction_interaction": ("index points times fractional return", "Derived", "Prior VIX close multiplied by reaction-day stock return"),
    "surprise_reaction_interaction": ("percent surprise times fractional return", "Derived", "Current announced surprise percentage multiplied by reaction-day stock return"),
}


def main():
    frame = pd.read_csv(DATA)
    dates = [c for c in frame if c.endswith("_date") or c == "reaction_day"]
    for name in dates:
        frame[name] = pd.to_datetime(frame[name])
    prices = {}
    for ticker in [*frame.ticker.unique(), "SPY"]:
        source = pd.read_csv(ROOT / "data/raw/research/expanded/prices" / (ticker + ".csv"))
        source["date"] = pd.to_datetime(source.date)
        prices[ticker] = source.set_index("date").sort_index()
    target_errors = horizon_errors = spy_errors = 0
    for _, event in frame.iterrows():
        stock = prices[event.ticker]
        i = stock.index.get_loc(event.reaction_day)
        target = stock.adjusted_close.iloc[i + 5] / stock.adjusted_close.iloc[i] - 1
        market = prices["SPY"].adjusted_close
        spy_target = market.loc[event.prediction_end_date] / market.loc[event.reaction_day] - 1
        target_errors += int(not np.isclose(target, event.future_5d_return, rtol=0, atol=1e-9))
        spy_errors += int(not np.isclose(spy_target, event.spy_future_5d_return, rtol=0, atol=1e-9))
        horizon_errors += int(stock.index[i + 1] != event.prediction_start_date or stock.index[i + 5] != event.prediction_end_date)
    checks = {"checked_events": len(frame), "target_recalculation_errors": target_errors,
        "spy_target_recalculation_errors": spy_errors, "five_session_horizon_errors": horizon_errors,
        "duplicate_event_ids": int(frame.event_id.duplicated().sum()),
        "nonfinite_predictor_cells": int((~np.isfinite(frame[MODEL_FEATURES])).sum().sum()),
        "date_order_errors": int((~(frame.reported_date.lt(frame.reaction_day) & frame.reaction_day.lt(frame.prediction_start_date)
            & frame.prediction_start_date.le(frame.prediction_end_date))).sum()),
        "market_adjusted_target_errors": int((~np.isclose(frame.future_5d_return - frame.spy_future_5d_return,
            frame.market_adjusted_5d_return, rtol=0, atol=1e-9)).sum()),
        "direction_target_errors": int(frame.positive_return.ne(frame.future_5d_return.gt(0).astype(int)).sum())}
    for name, lag, max_age in [("vix", 1, 7), ("vvix", 1, 7), ("dgs10", 7, 14), ("dgs2", 7, 14)]:
        age = (frame.reaction_day - frame[name + "_source_date"]).dt.days
        checks[name + "_lag_errors"] = int((~age.between(lag, max_age)).sum())
    (OUT / "integrity_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if any(v for k, v in checks.items() if k != "checked_events"):
        raise ValueError("Expanded dataset integrity checks failed; see integrity_checks.json")

    dictionary = []
    for column in frame:
        if column in FEATURE_DOCS:
            units, source, definition = FEATURE_DOCS[column]
            role = "numeric_predictor"
        elif column in ["future_5d_return", "spy_future_5d_return", "market_adjusted_5d_return", "positive_return"]:
            role = "target_never_predictor"
            units = "binary" if column == "positive_return" else "fraction"
            source = "Derived from Yahoo Finance adjusted closes"
            definition = {"future_5d_return": "Stock adjusted close at fifth session after reaction divided by reaction close minus one",
                "spy_future_5d_return": "SPY return between the same reaction/end calendar dates",
                "market_adjusted_5d_return": "Stock target minus SPY target",
                "positive_return": "One if the stock target is strictly positive; zero otherwise"}[column]
        elif column.endswith("_flag"):
            role, units, source = "audit_only", "boolean", "Derived from Alpha Vantage"
            definition = {"large_surprise_flag": "Absolute current surprise exceeds 100 percent; retained",
                "small_estimate_flag": "Absolute current estimated EPS below 0.05 USD; retained",
                "eps_difference_flag": "Reported EPS minus estimated EPS differs from provider dollar surprise by more than 0.001 USD; retained"}[column]
        else:
            role, units, source = "metadata_not_default_predictor", "text or date", "Project construction"
            definition = {"event_id": "Ticker and reported date identifier", "ticker": "Vendor company symbol; descriptive identifier",
                "sector": "Fixed descriptive industry grouping; not a reconstructed historical GICS membership",
                "cohort": "Original ten-company cohort or newly added expansion cohort",
                "fiscal_date_ending": "Vendor fiscal-period end; not an availability date",
                "reported_date": "Vendor announcement calendar date; exact release time unverified",
                "reaction_day": "First available stock session strictly after the announcement date; prediction at close",
                "prediction_start_date": "First stock session after reaction; target window starts here",
                "prediction_end_date": "Fifth stock session after reaction; target window ends here",
                "analysis_split": "Train through 2022; validation 2023-2024; exploratory test 2025; boundary-crossing labels purged",
                "sector_proxy_used": "Actual ETF used; VOX before sufficient XLC history and IYR before sufficient XLRE history",
                "sector_source_date": "Last historical proxy close used for the pre-reaction momentum window",
                "vix_source_date": "Date of selected lagged Cboe VIX close",
                "vvix_source_date": "Date of selected lagged Cboe VVIX close",
                "dgs10_source_date": "Date of selected lagged 10-year Treasury observation",
                "dgs2_source_date": "Date of selected lagged 2-year Treasury observation"}[column]
        dictionary.append({"column": column, "role": role, "units": units, "source": source, "definition": definition})
    pd.DataFrame(dictionary).to_csv(ROOT / "research/datasets/feature_dictionary.csv", index=False)

    yearly = frame.groupby(frame.reported_date.dt.year).size()
    sectors = frame.groupby("sector").size().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1.05, 1]})
    years = yearly.index.to_numpy()
    axes[0].bar(years - .18, [40 if y >= 2016 else 0 for y in years], .36, color="#b8bdc7", label="Original 400 events")
    axes[0].bar(years + .18, yearly.values, .36, color="#29677f", label=f"Expanded {len(frame):,} events")
    axes[0].set(title="More history and more companies", xlabel="Announcement year", ylabel="Complete earnings events")
    axes[0].set_xticks(years[::2])
    axes[0].set_ylim(0, yearly.max() * 1.3)
    axes[0].legend(frameon=False)
    axes[1].barh(sectors.index, sectors.values, color="#29677f")
    axes[1].set(title="Coverage across 11 industry groups", xlabel="Complete earnings events")
    for i, value in enumerate(sectors):
        axes[1].text(value + 3, i, str(value), va="center", fontsize=9)
    axes[1].set_xlim(0, sectors.max() * 1.15)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Expanded earnings research dataset 2010–2025", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "dataset_coverage.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
