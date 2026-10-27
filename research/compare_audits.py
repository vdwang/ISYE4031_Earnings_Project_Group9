"""Summarize the separate pilot histories and compare overlapping vendor values."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dataset_research"


def main():
    primary_path = ROOT / "data/processed/modeling_dataset.csv"
    primary = pd.read_csv(primary_path)
    keys = ["ticker", "reported_date"]
    summaries = []
    for start in [1996, 2010, 2016]:
        frame = pd.read_csv(ROOT / f"data/processed/research/modeling_{start}_2025.csv")
        summaries.append({"window": f"{start}_2025", "rows": len(frame),
            "duplicate_event_keys": int(frame.duplicated(keys).sum()),
            "nonfinite_numeric_cells": int((~np.isfinite(frame.select_dtypes("number"))).sum().sum()),
            "abs_surprise_above_100_pct": int(frame.earnings_surprise_pct.abs().gt(100).sum()),
            "max_abs_target_pct": float(frame.future_5d_return.abs().max() * 100)})
    pd.DataFrame(summaries).to_csv(OUT / "extended_event_quality.csv", index=False)

    shadow = pd.read_csv(ROOT / "data/processed/research/modeling_2016_2025.csv")
    merged = primary.merge(shadow, on=keys, suffixes=("_original", "_fresh"), validate="one_to_one", how="outer", indicator=True)
    rows = []
    for column in ["future_5d_return", "spy_future_5d_return", "market_adjusted_5d_return", "reaction_day_return", "return_20d_before"]:
        delta = (merged[column + "_original"] - merged[column + "_fresh"]).abs()
        rows.append({"variable": column, "matched_events": int(merged._merge.eq("both").sum()),
            "unmatched_events": int(merged._merge.ne("both").sum()),
            "changed_above_1e_minus_8": int(delta.gt(1e-8).sum()),
            "max_abs_difference_percentage_points": float(delta.max() * 100)})
    pd.DataFrame(rows).to_csv(OUT / "fresh_vendor_comparison.csv", index=False)
    (OUT / "primary_snapshot.json").write_text(json.dumps({"path": "data/processed/modeling_dataset.csv",
        "rows": len(primary), "sha256": hashlib.sha256(primary_path.read_bytes()).hexdigest(),
        "comparison_only": True, "primary_file_written": False}, indent=2), encoding="utf-8")

    # Measure join coverage only. Historical publication availability is not yet audited.
    vix = pd.read_csv(ROOT / "data/raw/research/fred/VIXCLS.csv")
    vix = vix.rename(columns={vix.columns[0]: "vix_date"})
    vix["vix_date"] = pd.to_datetime(vix.vix_date)
    vix["VIXCLS"] = pd.to_numeric(vix.VIXCLS, errors="coerce")
    vix = vix.dropna(subset=["VIXCLS"]).sort_values("vix_date")
    coverage = []
    for start in [2010, 2016]:
        frame = pd.read_csv(ROOT / f"data/processed/research/modeling_{start}_2025.csv")
        frame["reaction_day"] = pd.to_datetime(frame.reaction_day)
        joined = pd.merge_asof(frame.sort_values("reaction_day"), vix,
            left_on="reaction_day", right_on="vix_date", allow_exact_matches=False, direction="backward")
        age = (joined.reaction_day - joined.vix_date).dt.days
        coverage.append({"window": f"{start}_2025", "events": len(joined),
            "matched_prior_observation": int(joined.VIXCLS.notna().sum()),
            "max_calendar_days_old": int(age.max()), "publication_availability_verified": False})
    pd.DataFrame(coverage).to_csv(OUT / "vix_join_coverage.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    main()
