"""Recalculate every shared S&P 500 target and publish a data-quality profile."""

import json
import hashlib
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
from research.profile_expansion import FEATURE_DOCS

OUT = ROOT / "outputs/dataset_research/sp500"
DATA = ROOT / "research/datasets/sp500_modeling_dataset.csv"


def main():
    frame = pd.read_csv(DATA, dtype={"cik": str})
    for column in [c for c in frame if c.endswith("_date") or c == "reaction_day"]:
        frame[column] = pd.to_datetime(frame[column])
    spy = pd.read_csv(ROOT / "data/raw/research/sp500/prices/SPY.csv", parse_dates=["date"]).set_index("date").adjusted_close
    errors = {"checked_events": len(frame), "stock_target_errors": 0, "spy_target_errors": 0,
        "five_session_errors": 0, "reaction_alignment_errors": 0,
        "duplicate_issuer_date_keys": int(frame.duplicated(["cik", "reported_date"]).sum()),
        "nonfinite_predictor_cells": int((~np.isfinite(frame[MODEL_FEATURES])).sum().sum())}
    for ticker, sample in frame.groupby("ticker"):
        stock = pd.read_csv(ROOT / "data/raw/research/sp500/prices" / (ticker + ".csv"), parse_dates=["date"]).set_index("date").sort_index()
        indexes = stock.index.get_indexer(sample.reaction_day)
        expected_reactions = stock.index.searchsorted(sample.reported_date, side="right")
        errors["reaction_alignment_errors"] += int((indexes != expected_reactions).sum())
        predicted = stock.adjusted_close.to_numpy()[indexes + 5] / stock.adjusted_close.to_numpy()[indexes] - 1
        errors["stock_target_errors"] += int((~np.isclose(predicted, sample.future_5d_return, atol=1e-9, rtol=0)).sum())
        errors["five_session_errors"] += int(((stock.index[indexes + 1] != sample.prediction_start_date)
            | (stock.index[indexes + 5] != sample.prediction_end_date)).sum())
        market = spy.reindex(sample.prediction_end_date).to_numpy() / spy.reindex(sample.reaction_day).to_numpy() - 1
        errors["spy_target_errors"] += int((~np.isclose(market, sample.spy_future_5d_return, atol=1e-9, rtol=0)).sum())
    errors["adjusted_target_errors"] = int((~np.isclose(frame.future_5d_return - frame.spy_future_5d_return,
        frame.market_adjusted_5d_return, atol=1e-9, rtol=0)).sum())
    errors["direction_errors"] = int(frame.positive_return.ne(frame.future_5d_return.gt(0).astype(int)).sum())
    errors["reaction_gap_errors"] = int((~(frame.reaction_day - frame.reported_date).dt.days.between(1, 7)).sum())
    ages = []
    for name, lag, max_age in [("vix", 1, 7), ("vvix", 1, 7), ("dgs10", 7, 14), ("dgs2", 7, 14)]:
        age = (frame.reaction_day - frame[name + "_source_date"]).dt.days
        errors[name + "_date_errors"] = int((~age.between(lag, max_age)).sum())
        ages.append({"source": name, "min_age_days": int(age.min()), "max_age_days": int(age.max())})
    errors["boundary_errors"] = 0
    for label, start, end in [("train", "2010-01-01", "2023-01-01"), ("validation", "2023-01-01", "2025-01-01"), ("exploratory_test", "2025-01-01", "2026-01-01")]:
        sample = frame.loc[frame.analysis_split.eq(label)]
        errors["boundary_errors"] += int((~(sample.reaction_day.ge(start) & sample.reaction_day.lt(end) & sample.prediction_end_date.lt(end))).sum())
    (OUT / "integrity_checks.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    (OUT / "verified_dataset_hash.txt").write_text(hashlib.sha256(DATA.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    pd.DataFrame(ages).to_csv(OUT / "context_age_audit.csv", index=False)
    if any(v for k, v in errors.items() if k != "checked_events"):
        raise ValueError("Dataset integrity failed; inspect integrity_checks.json")
    dictionary = []
    meta = {"event_id": "Canonical ticker and reported date", "ticker": "One canonical stock class per issuer; Yahoo symbol format",
        "cik": "SEC issuer identifier; use for company grouping", "company_name": "Name in the downloaded constituent snapshot",
        "sector": "Snapshot sector descriptor, not historical membership", "reported_date": "Announcement calendar date; exact release time not verified",
        "reaction_day": "First stock session strictly after reported date; prediction at close",
        "prediction_start_date": "First session after reaction", "prediction_end_date": "Fifth session after reaction",
        "analysis_split": "Chronological split or purged_boundary; 2025 is exploratory",
        "earnings_source": "Chosen vendor/archive lineage; use for sensitivity comparisons",
        "source_symbol": "Original archive ticker before CIK-linked renaming/class consolidation",
        "sector_proxy_used": "Actual proxy ETF; IYR/VOX used before sufficient XLRE/XLC history",
        "sector_source_date": "Last pre-reaction proxy close date", "vix_source_date": "Selected prior VIX observation date",
        "vvix_source_date": "Selected prior VVIX observation date", "dgs10_source_date": "Selected Treasury 10-year observation date",
        "dgs2_source_date": "Selected Treasury 2-year observation date"}
    for column in frame:
        if column in FEATURE_DOCS:
            units, source, definition = FEATURE_DOCS[column]
            if column == "earnings_surprise_pct":
                source = "Chosen earnings archive"
                definition = "100 times (actual EPS minus estimated EPS) divided by absolute estimated EPS; zero denominator stays missing; no clipping"
            elif source == "Alpha Vantage":
                source = "Chosen earnings archive"
            role = "numeric_predictor"
        elif column in meta:
            role, units, source, definition = "metadata_not_default_predictor", "text or date", "Project construction", meta[column]
        elif column.endswith("_flag"):
            role, units, source = "audit_only", "boolean", "Derived from earnings"
            definition = {"large_surprise_flag": "Absolute surprise above 100 percent; retained", "small_estimate_flag": "Absolute estimated EPS below 0.05 USD; retained",
                "zero_estimate_flag": "Zero estimated EPS; percentage surprise undefined so not complete", "short_earnings_gap_flag": "Less than 45 days since prior announcement; retained for review"}[column]
        else:
            role, units, source = "target_never_predictor", "binary" if column == "positive_return" else "fraction", "Recalculated Yahoo adjusted closes"
            definition = {"future_5d_return": "Signed stock return from reaction close to the fifth later session close",
                "spy_future_5d_return": "SPY return over the same calendar endpoints", "market_adjusted_5d_return": "Stock target minus SPY target",
                "positive_return": "One for strictly positive stock target; zero for negative or zero"}[column]
        dictionary.append({"column": column, "role": role, "units": units, "source": source, "definition": definition})
    pd.DataFrame(dictionary).to_csv(ROOT / "research/datasets/sp500_feature_dictionary.csv", index=False)
    yearly = frame.groupby(frame.reported_date.dt.year).size()
    sectors = frame.groupby("sector").size().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    axes[0].bar(yearly.index, yearly.values, color="#29677f")
    axes[0].set(title="Complete events by announcement year", xlabel="Year", ylabel="Earnings events")
    axes[0].set_xticks(yearly.index[::2])
    axes[1].barh(sectors.index, sectors.values, color="#29677f")
    axes[1].set(title="Coverage across industry groups", xlabel="Complete events")
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"S&P 500 snapshot research panel  |  {len(frame):,} events  |  {frame.cik.nunique()} issuers", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "dataset_coverage.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(errors, indent=2))


if __name__ == "__main__":
    main()
