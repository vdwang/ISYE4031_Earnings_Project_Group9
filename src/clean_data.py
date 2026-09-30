"""Keep data gaps visible and create a complete-case modeling table."""

import pandas as pd

from .config import FEATURE_COLUMNS, OUTPUT_DIR, PROCESSED_DIR, TARGET_COLUMNS


def clean_and_save(events: pd.DataFrame, counts: dict[str, int]) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_events = events.copy()
    all_events["required_predictors_missing"] = all_events[FEATURE_COLUMNS].isna().any(axis=1)
    all_events["valid_target"] = all_events[["future_5d_return", "spy_future_5d_return"]].notna().all(axis=1)
    all_events.to_csv(PROCESSED_DIR / "earnings_events_all.csv", index=False)

    modeling = all_events.loc[all_events["valid_target"] & ~all_events["required_predictors_missing"]].copy()
    modeling = modeling.sort_values(["reported_date", "ticker"]).reset_index(drop=True)
    modeling.to_csv(PROCESSED_DIR / "modeling_dataset.csv", index=False)

    important = [
        "reported_date", "reaction_day", "prediction_start_date", "prediction_end_date",
        *FEATURE_COLUMNS, *TARGET_COLUMNS,
    ]
    missing = pd.DataFrame({
        "variable": important,
        "missing_count": [int(all_events[column].isna().sum()) for column in important],
    })
    missing["missing_percent"] = 100 * missing["missing_count"] / len(all_events) if len(all_events) else 0.0
    missing.to_csv(OUTPUT_DIR / "missingness_summary.csv", index=False)

    signed = modeling["future_5d_return"]
    summary = {
        **counts,
        "events_with_valid_target": int(all_events["valid_target"].sum()),
        "clean_event_count": len(modeling),
        "number_of_tickers": int(modeling["ticker"].nunique()),
        "start_date": modeling["reported_date"].min() if len(modeling) else "",
        "end_date": modeling["reported_date"].max() if len(modeling) else "",
        "positive_return_count": int((signed > 0).sum()),
        "negative_return_count": int((signed < 0).sum()),
        "zero_return_count": int((signed == 0).sum()),
        "percent_positive_returns": float(100 * (signed > 0).mean()) if len(signed) else float("nan"),
        "percent_negative_returns": float(100 * (signed < 0).mean()) if len(signed) else float("nan"),
        "mean_5d_return": signed.mean(),
        "median_5d_return": signed.median(),
        "std_5d_return": signed.std(),
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "data_summary.csv", index=False)
    sample_columns = [
        "ticker", "reported_date", "reaction_day", "prediction_start_date", "prediction_end_date",
        "earnings_surprise_pct", "reaction_day_return", "abnormal_volume", "return_5d_before",
        "future_5d_return", "positive_return", "market_adjusted_5d_return",
    ]
    modeling[sample_columns].head(20).to_csv(OUTPUT_DIR / "event_sample.csv", index=False)

    print(f"Raw earnings events: {counts['raw_event_count']}")
    print(f"Events in 2016-2025: {counts['events_2016_2025']}")
    print(f"Events matched to prices: {counts['events_matched_to_prices']}")
    print(f"Events after complete-case cleaning: {len(modeling)}")
    print(f"Companies: {summary['number_of_tickers']}")
    print(f"Earliest event: {summary['start_date']}")
    print(f"Latest event: {summary['end_date']}")
    print(f"Positive returns: {summary['percent_positive_returns']:.1f}%")
    print(f"Negative returns: {summary['percent_negative_returns']:.1f}%")
    return modeling
