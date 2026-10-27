"""Build and audit the larger multi-source research dataset from cached inputs."""

import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.build_dataset import build_events
from src.config import EARNINGS_COLUMNS, FEATURE_COLUMNS
from research.expanded_features import MODEL_FEATURES, enrich_events, normalize_context

CACHE = ROOT / "data/raw/research/expanded"
PRIVATE = ROOT / "data/processed/research"
OUT = ROOT / "outputs/dataset_research/expanded"
SHARED = ROOT / "research/datasets"


def read_inputs():
    universe = pd.read_csv(ROOT / "research/universe.csv")
    records = []
    for ticker in universe.ticker:
        path = ROOT / "data/raw/earnings" / (ticker + ".json")
        if not path.exists():
            path = CACHE / "earnings" / (ticker + ".json")
        if path.exists():
            payload = json.loads(path.read_text())
            for entry in payload.get("quarterlyEarnings", []):
                records.append({c: ticker if c == "ticker" else entry.get(c) for c in EARNINGS_COLUMNS})
    earnings = pd.DataFrame(records, columns=EARNINGS_COLUMNS)
    earnings = earnings.drop_duplicates()
    # A conflicting key needs explicit resolution rather than arbitrary row selection.
    if earnings.duplicated(["ticker", "reportedDate", "fiscalDateEnding"]).any():
        raise ValueError("Conflicting raw earnings keys require review")
    prices = {p.stem: pd.read_csv(p) for p in (CACHE / "prices").glob("*.csv")}
    context = {name: normalize_context(pd.read_csv(CACHE / "context" / (name + ".csv")), name)
        for name in ["VIX", "VVIX", "DGS10", "DGS2"]}
    return universe, earnings, prices, context


def label_splits(frame):
    frame = frame.copy()
    frame["analysis_split"] = "purged_boundary"
    for name, start, end in [("train", "2010-01-01", "2023-01-01"),
        ("validation", "2023-01-01", "2025-01-01"),
        ("exploratory_test", "2025-01-01", "2026-01-01")]:
        keep = frame.reaction_day.ge(start) & frame.reaction_day.lt(end) & frame.prediction_end_date.lt(end)
        frame.loc[keep, "analysis_split"] = name
    return frame


def main():
    for folder in [PRIVATE, OUT, SHARED]:
        folder.mkdir(parents=True, exist_ok=True)
    primary = ROOT / "data/processed/modeling_dataset.csv"
    before_hash = hashlib.sha256(primary.read_bytes()).hexdigest() if primary.exists() else None
    universe, earnings, prices, context = read_inputs()
    events, _ = build_events(earnings, prices, event_start="2010-01-01", event_end="2025-12-31")
    events = enrich_events(events, earnings, prices, universe, context)
    events["event_id"] = events.ticker + "_" + events.reported_date.dt.strftime("%Y%m%d")
    if events.event_id.duplicated().any():
        raise ValueError("Multiple events for one ticker/date require explicit resolution")
    events["invalid_fiscal_timing"] = events.fiscal_date_ending.gt(events.reported_date) | events.fiscal_date_ending.isna()
    events["large_surprise_flag"] = events.earnings_surprise_pct.abs().gt(100)
    events["small_estimate_flag"] = events.estimated_eps.abs().lt(.05)
    events["eps_difference_flag"] = ((events.reported_eps - events.estimated_eps) - events.earnings_surprise).abs().gt(.001)
    original_required = [*FEATURE_COLUMNS, "future_5d_return", "spy_future_5d_return"]
    events["core_complete"] = np.isfinite(events[original_required]).all(axis=1) & ~events.invalid_fiscal_timing
    events["enriched_complete"] = events.core_complete & np.isfinite(events[MODEL_FEATURES]).all(axis=1)
    events = label_splits(events).sort_values(["reaction_day", "ticker"]).reset_index(drop=True)
    events.to_csv(PRIVATE / "expanded_events_all.csv", index=False)
    events.loc[events.enriched_complete].to_csv(PRIVATE / "expanded_modeling_full_audit.csv", index=False)

    metadata = ["event_id", "ticker", "sector", "cohort", "fiscal_date_ending", "reported_date", "reaction_day",
        "prediction_start_date", "prediction_end_date", "analysis_split", "sector_proxy_used", "sector_source_date",
        "vix_source_date", "vvix_source_date", "dgs10_source_date", "dgs2_source_date"]
    targets = ["future_5d_return", "spy_future_5d_return", "market_adjusted_5d_return", "positive_return"]
    flags = ["large_surprise_flag", "small_estimate_flag", "eps_difference_flag"]
    shared = events.loc[events.enriched_complete, metadata + MODEL_FEATURES + targets + flags].copy()
    shared.to_csv(SHARED / "expanded_modeling_dataset.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d")

    rows = []
    for ticker, group in events.groupby("ticker"):
        rows.append({"ticker": ticker, "sector": group.sector.iloc[0], "candidate_events": len(group),
            "core_complete_events": int(group.core_complete.sum()), "enriched_complete_events": int(group.enriched_complete.sum()),
            "first_report": str(group.reported_date.min().date()), "last_report": str(group.reported_date.max().date())})
    pd.DataFrame(rows).to_csv(OUT / "coverage_by_company.csv", index=False)
    events["year"] = events.reported_date.dt.year
    events.groupby("year").agg(candidate_events=("event_id", "size"), core_complete_events=("core_complete", "sum"),
        enriched_complete_events=("enriched_complete", "sum")).to_csv(OUT / "coverage_by_year.csv")
    shared.groupby("sector").agg(events=("event_id", "size"), companies=("ticker", "nunique")).to_csv(OUT / "coverage_by_sector.csv")
    shared.groupby("analysis_split").agg(events=("event_id", "size"), companies=("ticker", "nunique")).to_csv(OUT / "split_counts.csv")
    pd.DataFrame({"variable": MODEL_FEATURES, "missing_count": events[MODEL_FEATURES].isna().sum().to_numpy(),
        "missing_percent": 100 * events[MODEL_FEATURES].isna().mean().to_numpy()}).to_csv(OUT / "feature_missingness.csv", index=False)
    excluded = events.loc[~events.enriched_complete, ["event_id", "ticker", "reported_date", "invalid_fiscal_timing", "core_complete"]].copy()
    excluded["missing_variables"] = events.loc[~events.enriched_complete].apply(
        lambda row: ";".join(c for c in list(dict.fromkeys(original_required + MODEL_FEATURES)) if pd.isna(row[c])), axis=1)
    excluded.to_csv(OUT / "excluded_events.csv", index=False, date_format="%Y-%m-%d")
    flagged = events.loc[events.enriched_complete & (events.large_surprise_flag | events.small_estimate_flag | events.eps_difference_flag),
        ["event_id", "ticker", "reported_date", "earnings_surprise_pct", *flags]]
    flagged.to_csv(OUT / "flagged_events.csv", index=False, date_format="%Y-%m-%d")
    ages = []
    for source in ["vix", "vvix", "dgs10", "dgs2"]:
        age = (shared.reaction_day - shared[source + "_source_date"]).dt.days
        ages.append({"source": source, "min_age_calendar_days": int(age.min()), "max_age_calendar_days": int(age.max()),
            "missing_on_complete_events": int(age.isna().sum())})
    pd.DataFrame(ages).to_csv(OUT / "context_age_audit.csv", index=False)
    signed = shared.future_5d_return
    summary = {"audit_date_utc": datetime.now(timezone.utc).date().isoformat(),
        "planned_companies": len(universe), "companies_with_events": int(events.ticker.nunique()),
        "companies_in_complete_dataset": int(shared.ticker.nunique()), "sector_groups": int(shared.sector.nunique()),
        "candidate_events_2010_2025": len(events), "core_complete_events": int(events.core_complete.sum()),
        "enriched_complete_events": len(shared), "excluded_events": len(excluded),
        "numeric_predictors": len(MODEL_FEATURES), "first_report": str(shared.reported_date.min().date()),
        "last_report": str(shared.reported_date.max().date()), "duplicate_event_ids": int(shared.event_id.duplicated().sum()),
        "nonfinite_predictor_cells": int((~np.isfinite(shared[MODEL_FEATURES])).sum().sum()),
        "large_surprise_flags": int(shared.large_surprise_flag.sum()), "small_estimate_flags": int(shared.small_estimate_flag.sum()),
        "eps_difference_flags": int(shared.eps_difference_flag.sum()), "mean_target_percent": float(signed.mean() * 100),
        "std_target_percentage_points": float(signed.std() * 100), "positive_targets": int(signed.gt(0).sum()),
        "negative_targets": int(signed.lt(0).sum()), "zero_targets": int(signed.eq(0).sum()),
        "primary_400_event_csv_unchanged": before_hash == (hashlib.sha256(primary.read_bytes()).hexdigest() if primary.exists() else None),
        "models_refitted": False, "true_point_in_time_vintages_verified": False,
        "dataset_sha256": hashlib.sha256((SHARED / "expanded_modeling_dataset.csv").read_bytes()).hexdigest()}
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "model_features.json").write_text(json.dumps(MODEL_FEATURES, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
