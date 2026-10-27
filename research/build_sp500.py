"""Build the S&P 500 snapshot panel with explicitly chosen earnings sources."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.expanded_features import MODEL_FEATURES, lagged_context, normalize_context
from research.build_expanded_dataset import label_splits
from research.sp500_features import daily_features, add_earnings_history

CACHE = ROOT / "data/raw/research/sp500"
OUT = ROOT / "outputs/dataset_research/sp500"
PRIVATE = ROOT / "data/processed/research"
DATA = ROOT / "research/datasets/sp500_modeling_dataset.csv"


def choose_earnings():
    alias = pd.read_csv(OUT / "issuer_aliases.csv").set_index("source_symbol").canonical_ticker.to_dict()
    hf = pd.read_parquet(CACHE / "hf_earnings.parquet").rename(columns={"symbol": "source_symbol", "eps_estimate": "estimated_eps"})
    # The archive's fixed EST timestamps are used only as dates, never as verified release times.
    hf["reported_date"] = hf.date.dt.tz_localize(None).dt.normalize()
    hf["earnings_source"] = "hf_archive"
    pead = pd.read_csv(CACHE / "pead_results.csv").rename(columns={"ticker": "source_symbol", "eps_estimate": "estimated_eps", "eps_actual": "reported_eps"})
    pead["reported_date"] = pd.to_datetime(pead.earnings_date).dt.normalize()
    pead["earnings_source"] = "published_yahoo_archive"
    columns = ["source_symbol", "reported_date", "estimated_eps", "reported_eps", "earnings_source"]
    both = pd.concat([hf[columns], pead[columns]], ignore_index=True)
    both["ticker"] = both.source_symbol.map(alias)
    both = both.loc[both.ticker.notna() & both.reported_eps.notna() & both.estimated_eps.notna()].copy()
    both["canonical_class"] = both.source_symbol.str.replace(".", "-", regex=False).eq(both.ticker)
    both = both.sort_values("canonical_class", ascending=False).drop_duplicates(["ticker", "reported_date", "earnings_source"])
    pead_tickers = set(both.loc[both.earnings_source.eq("published_yahoo_archive"), "ticker"])
    keep = both.earnings_source.eq("published_yahoo_archive") & both.reported_date.ge("2015-01-01")
    keep |= both.earnings_source.eq("hf_archive") & (both.reported_date.lt("2015-01-01") | ~both.ticker.isin(pead_tickers))
    chosen = both.loc[keep].copy()
    gap_path = CACHE / "gap_earnings.csv"
    if gap_path.exists():
        gap = pd.read_csv(gap_path, parse_dates=["reported_date"])
        gap = gap.loc[~gap.ticker.isin(chosen.ticker)]
        chosen = pd.concat([chosen, gap], ignore_index=True)
    # Keep each pilot company's full existing source rather than mixing dates within that company.
    av = []
    for folder in [ROOT / "data/raw/earnings", ROOT / "data/raw/research/expanded/earnings"]:
        for path in folder.glob("*.json"):
            ticker = alias.get(path.stem)
            if ticker is None:
                continue
            payload = json.loads(path.read_text())
            for item in payload.get("quarterlyEarnings", []):
                av.append({"ticker": ticker, "source_symbol": path.stem,
                    "reported_date": pd.to_datetime(item.get("reportedDate")),
                    "estimated_eps": pd.to_numeric(item.get("estimatedEPS"), errors="coerce"),
                    "reported_eps": pd.to_numeric(item.get("reportedEPS"), errors="coerce"),
                    "earnings_source": "alpha_pilot_cache"})
    av = pd.DataFrame(av).drop_duplicates(["ticker", "reported_date"])
    chosen = pd.concat([chosen.loc[~chosen.ticker.isin(av.ticker)], av], ignore_index=True)
    chosen = chosen.drop_duplicates(["ticker", "reported_date"]).sort_values(["ticker", "reported_date"])
    denominator = chosen.estimated_eps.abs().where(chosen.estimated_eps.ne(0))
    chosen["earnings_surprise_pct"] = 100 * (chosen.reported_eps - chosen.estimated_eps) / denominator
    chosen["earnings_surprise_pct"] = chosen.earnings_surprise_pct.replace([np.inf, -np.inf], np.nan)
    chosen["small_estimate_flag"] = chosen.estimated_eps.abs().lt(.05)
    chosen["zero_estimate_flag"] = chosen.estimated_eps.eq(0)
    chosen["large_surprise_flag"] = chosen.earnings_surprise_pct.abs().gt(100)
    chosen = add_earnings_history(chosen)
    chosen["short_earnings_gap_flag"] = chosen.days_since_previous_earnings.lt(45)
    # Same-day source agreement is an audit, never a way of multiplying observations.
    overlap = both.pivot_table(index=["ticker", "reported_date"], columns="earnings_source", values=["estimated_eps", "reported_eps"], aggfunc="first")
    audit = []
    for value in ["estimated_eps", "reported_eps"]:
        comparable = overlap[value].dropna()
        delta = (comparable.hf_archive - comparable.published_yahoo_archive).abs()
        audit.append({"variable": value, "same_date_overlaps": len(comparable), "difference_over_0_01": int(delta.gt(.01).sum())})
    pct = {}
    for source in ["hf_archive", "published_yahoo_archive"]:
        denominator = overlap["estimated_eps"][source].abs().where(overlap["estimated_eps"][source].ne(0))
        pct[source] = 100 * (overlap["reported_eps"][source] - overlap["estimated_eps"][source]) / denominator
    comparable = pd.DataFrame(pct).replace([np.inf, -np.inf], np.nan).dropna()
    delta = (comparable.hf_archive - comparable.published_yahoo_archive).abs()
    audit.append({"variable": "derived_surprise_percentage_points", "same_date_overlaps": len(comparable),
        "difference_over_0_01": int(delta.gt(.01).sum()), "difference_over_5_points": int(delta.gt(5).sum())})
    pd.DataFrame(audit).to_csv(OUT / "source_overlap_audit.csv", index=False)
    return chosen.loc[chosen.reported_date.between("2010-01-01", "2025-12-31")].copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(ROOT / "research/sp500_universe.csv", dtype={"cik": str})
    earnings = choose_earnings()
    prices = {p.stem: pd.read_csv(p) for p in (CACHE / "prices").glob("*.csv")}
    frames = []
    coverage = []
    for _, company in universe.iterrows():
        sample = earnings.loc[earnings.ticker.eq(company.ticker)].copy()
        if sample.empty or company.ticker not in prices:
            coverage.append({"ticker": company.ticker, "cik": company.cik, "sector": company.sector,
                "candidate_events": len(sample), "matched_events": 0, "complete_events": 0,
                "status": "no_earnings" if sample.empty else "no_prices"})
            continue
        fallback = "VOX" if company.sector_etf == "XLC" else "IYR" if company.sector_etf == "XLRE" else None
        features = daily_features(prices[company.ticker], prices["SPY"], prices.get(company.sector_etf), prices.get(fallback))
        merged = pd.merge_asof(sample.sort_values("reported_date"), features,
            left_on="reported_date", right_on="reaction_day", direction="forward",
            allow_exact_matches=False)
        too_late = (merged.reaction_day - merged.reported_date).dt.days.gt(7)
        merged.loc[too_late, features.columns] = np.nan
        merged["cik"] = company.cik
        merged["sector"] = company.sector
        merged["company_name"] = company.Security
        frames.append(merged)
    events = pd.concat(frames, ignore_index=True)
    context_folder = ROOT / "data/raw/research/expanded/context"
    context = {name: normalize_context(pd.read_csv(context_folder / (name + ".csv")), name) for name in ["VIX", "VVIX", "DGS10", "DGS2"]}
    events = lagged_context(events, context)
    events["vix_reaction_interaction"] = events.vix_prior_close * events.reaction_day_return
    events["surprise_reaction_interaction"] = events.earnings_surprise_pct * events.reaction_day_return
    events[MODEL_FEATURES] = events[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    targets = ["future_5d_return", "spy_future_5d_return", "market_adjusted_5d_return", "positive_return"]
    events["complete_case"] = np.isfinite(events[MODEL_FEATURES + targets]).all(axis=1)
    events["event_id"] = events.ticker + "_" + events.reported_date.dt.strftime("%Y%m%d")
    events = label_splits(events).sort_values(["reaction_day", "ticker"]).reset_index(drop=True)
    for ticker, group in events.groupby("ticker"):
        coverage.append({"ticker": ticker, "cik": group.cik.iloc[0], "sector": group.sector.iloc[0],
            "candidate_events": len(group), "matched_events": int(group.reaction_day.notna().sum()),
            "complete_events": int(group.complete_case.sum()), "status": "complete_events" if group.complete_case.any() else "no_complete_events"})
    pd.DataFrame(coverage).sort_values("ticker").to_csv(OUT / "coverage_by_company.csv", index=False)
    flags = ["large_surprise_flag", "small_estimate_flag", "zero_estimate_flag", "short_earnings_gap_flag"]
    metadata = ["event_id", "ticker", "cik", "company_name", "sector", "reported_date", "reaction_day",
        "prediction_start_date", "prediction_end_date", "analysis_split", "earnings_source", "source_symbol",
        "sector_proxy_used", "sector_source_date", "vix_source_date", "vvix_source_date", "dgs10_source_date", "dgs2_source_date"]
    modeling = events.loc[events.complete_case, metadata + MODEL_FEATURES + targets + flags].copy()
    modeling.to_csv(DATA, index=False, float_format="%.10g", date_format="%Y-%m-%d", lineterminator="\n")
    events.to_csv(PRIVATE / "sp500_all_events.csv", index=False, date_format="%Y-%m-%d")
    events["year"] = events.reported_date.dt.year
    events.groupby("year").agg(candidate_events=("event_id", "size"), complete_events=("complete_case", "sum")).to_csv(OUT / "coverage_by_year.csv")
    modeling.groupby("sector").agg(events=("event_id", "size"), companies=("cik", "nunique")).to_csv(OUT / "coverage_by_sector.csv")
    modeling.groupby(["earnings_source"]).size().rename("complete_events").to_csv(OUT / "coverage_by_source.csv")
    modeling.groupby("analysis_split").size().rename("events").to_csv(OUT / "split_counts.csv")
    missing = events[MODEL_FEATURES + targets].isna().sum().rename("missing_count").rename_axis("variable").reset_index()
    missing.to_csv(OUT / "feature_missingness.csv", index=False)
    excluded = events.loc[~events.complete_case, ["event_id", "ticker", "reported_date"]].copy()
    excluded["missing_variables"] = events.loc[~events.complete_case].apply(lambda r: ";".join(c for c in MODEL_FEATURES + targets if pd.isna(r[c])), axis=1)
    excluded.to_csv(OUT / "excluded_events.csv", index=False, date_format="%Y-%m-%d")
    modeling.loc[modeling[flags].any(axis=1), ["event_id", "ticker", "reported_date", "earnings_surprise_pct", *flags]].to_csv(OUT / "flagged_events.csv", index=False, date_format="%Y-%m-%d")
    summary = {"roster_listings": 503, "roster_issuers": len(universe), "issuers_with_complete_events": int(modeling.cik.nunique()),
        "candidate_events": len(earnings), "events_with_price_table": len(events), "complete_events": len(modeling),
        "numeric_predictors": len(MODEL_FEATURES), "sectors": int(modeling.sector.nunique()),
        "first_report": str(modeling.reported_date.min().date()), "last_report": str(modeling.reported_date.max().date()),
        "duplicate_issuer_date_keys": int(modeling.duplicated(["cik", "reported_date"]).sum()),
        "large_surprise_flags": int(modeling.large_surprise_flag.sum()), "short_gap_flags": int(modeling.short_earnings_gap_flag.sum()),
        "meets_10000_event_target": len(modeling) >= 10000, "current_snapshot_not_historical_membership": True,
        "models_refitted": False, "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest()}
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
