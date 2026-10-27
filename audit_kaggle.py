"""Audit the suggested Kaggle source without downloading its 2 GB price file.

python audit_kaggle.py --download fetches only the public summary and earnings
files for version 7. Without --download, existing candidate caches are required.
This never replaces the main event dataset or imputes unavailable observations.
"""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import quote
from zipfile import is_zipfile, ZipFile

import pandas as pd
import requests

from src.config import ROOT, TICKERS

CACHE = ROOT / "data/raw/kaggle_candidate"
OUT = ROOT / "outputs/milestone2"
SLUG = "tsaustin/us-historical-stock-prices-with-earnings-data"
FILES = ["dataset_summary.csv", "stocks_latest/earnings_latest.csv"]


def download_small_files():
    CACHE.mkdir(parents=True, exist_ok=True)
    response = requests.get("https://www.kaggle.com/api/v1/datasets/list/" + SLUG, timeout=30)
    response.raise_for_status()
    (CACHE / "file_metadata.json").write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
    for name in FILES:
        dest = CACHE / Path(name).name
        if dest.exists():
            continue
        url = "https://www.kaggle.com/api/v1/datasets/download/" + SLUG + "/" + quote(name, safe="")
        chunks = []
        downloaded = 0
        with requests.get(url, params={"datasetVersionNumber": 7}, timeout=60, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=65_536):
                downloaded += len(chunk)
                if downloaded > 20_000_000:
                    raise ValueError("Unexpected candidate file size; stopped before a bulk price download")
                chunks.append(chunk)
        payload = b"".join(chunks)
        stream = BytesIO(payload)
        if is_zipfile(stream):
            with ZipFile(stream) as archive:
                matches = [p for p in archive.namelist() if Path(p).name == dest.name]
                if len(matches) != 1:
                    raise ValueError("Candidate file could not be identified uniquely")
                if archive.getinfo(matches[0]).file_size > 20_000_000:
                    raise ValueError("Unexpected uncompressed candidate file size")
                payload = archive.read(matches[0])
        dest.write_bytes(payload)
        print("Downloaded candidate audit file:", dest.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download_small_files()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(CACHE / "dataset_summary.csv")
    earnings = pd.read_csv(CACHE / "earnings_latest.csv")
    dates = pd.to_datetime(earnings["date"], errors="coerce")
    eps = pd.to_numeric(earnings["eps"], errors="coerce")
    estimate = pd.to_numeric(earnings["eps_est"], errors="coerce")
    print("Summary columns:", summary.columns.tolist())
    print("Earnings columns:", earnings.columns.tolist())
    print("Earnings rows:", len(earnings))
    print(earnings.head(3).to_string(index=False))
    # Preserve raw column names until the actual vendor schema has been inspected.
    metadata = {"dataset": SLUG, "version_requested": 7, "audit_date": "2026-10-27",
        "summary_rows": len(summary), "earnings_rows": len(earnings),
        "summary_columns": summary.columns.tolist(), "earnings_columns": earnings.columns.tolist(),
        "earnings_missing_cells": {c: int(earnings[c].isna().sum()) for c in earnings.columns},
        "earnings_exact_duplicate_rows": int(earnings.duplicated().sum()),
        "earnings_duplicate_symbol_date_keys": int(earnings.duplicated(["symbol", "date"]).sum()),
        "earnings_first_date": str(dates.min().date()), "earnings_last_date": str(dates.max().date()),
        "invalid_earnings_dates": int(dates.isna().sum()),
        "events_with_both_eps_values": int((eps.notna() & estimate.notna()).sum()),
        "zero_eps_estimates": int(estimate.eq(0).sum()),
        "release_time_counts": {str(k): int(v) for k, v in earnings["release_time"].fillna("missing").value_counts().items()},
        "events_after_2021": int((dates >= "2022-01-01").sum()),
        "main_model_uses_candidate": False, "price_file_downloaded": False,
        "license_status": "Dataset page lists license as Unknown; redistribution permission not established",
        "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
            for p in CACHE.glob("*.csv")]}
    (OUT / "kaggle_candidate_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # FB is an explicit historical alias to inspect, not a silent merge into META.
    summary.loc[summary["symbol"].isin([*TICKERS, "FB"])].to_csv(OUT / "kaggle_pilot_coverage.csv", index=False)
    print("Candidate audit saved; the 400-event primary dataset was not changed.")


if __name__ == "__main__":
    main()
