"""Download bounded public samples and audit candidate data without modeling them.

Run with the project environment: python research/audit_candidates.py
The primary data, figures and model results are never overwritten.
"""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote
from zipfile import is_zipfile, ZipFile

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data/raw/research"
OUT = ROOT / "outputs/dataset_research"
KAGGLE = ["adarsh1077/epsclassification", "anthonypm/nasdaq-earnings-calendar-auto-updating",
    "kaziaishikuzzaman/earningssignal-earnings-call-tone-and-guidance", "hexsyro/earnings-and-corporate-news-reactions"]


def fetch(url, path, session=None, params=None, limit=20_000_000):
    """Cache a bounded public response; no authentication or payment is attempted."""
    if path.exists():
        return path.read_bytes()
    client = session or requests
    data = bytearray()
    with client.get(url, params=params, stream=True, timeout=45) as response:
        response.raise_for_status()
        for chunk in response.iter_content(65_536):
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError("Response exceeded the audit download limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return bytes(data)


def audit_kaggle_candidates():
    rows = []
    for ref in KAGGLE:
        folder = CACHE / ref.replace("/", "__")
        meta = json.loads(fetch("https://www.kaggle.com/api/v1/datasets/view/" + ref, folder / "metadata.json"))
        files = json.loads(fetch("https://www.kaggle.com/api/v1/datasets/list/" + ref, folder / "files.json"))
        description = meta.get("description", "")
        row = {"ref": ref, "url": "https://www.kaggle.com/datasets/" + ref,
            "version": meta.get("currentVersionNumber"), "title": meta.get("title"),
            "license": meta.get("licenseName"),
            "sample_audited": False, "primary_model_uses_source": False}
        if "synthetic dataset" in description.lower():
            row["publisher_declares_synthetic"] = True
            row["decision"] = "Reject as real financial observations: publisher explicitly describes synthetic data"
        elif ref.startswith("hexsyro/"):
            row["decision"] = "Not an earnings-event table; optional social-text supplement with first-seen/engagement timing and license checks"
        else:
            candidates = [f for f in files.get("datasetFiles", []) if f["name"].lower().endswith(".csv")]
            if len(candidates) != 1:
                row["decision"] = "Schema review needed before sample selection"
            else:
                item = candidates[0]
                payload = fetch("https://www.kaggle.com/api/v1/datasets/download/" + ref + "/" + quote(item["name"], safe=""),
                    folder / "sample_download", params={"datasetVersionNumber": row["version"]})
                stream = BytesIO(payload)
                if is_zipfile(stream):
                    with ZipFile(stream) as archive:
                        names = [name for name in archive.namelist() if Path(name).name == Path(item["name"]).name]
                        if len(names) != 1 or archive.getinfo(names[0]).file_size > 20_000_000:
                            raise ValueError("Candidate CSV is ambiguous or too large")
                        payload = archive.read(names[0])
                frame = pd.read_csv(BytesIO(payload))
                row.update({"sample_audited": True, "file": item["name"], "rows": len(frame),
                    "columns": frame.columns.tolist(), "exact_duplicate_rows": int(frame.duplicated().sum()),
                    "missing_cells": {c: int(frame[c].isna().sum()) for c in frame},
                    "sha256": hashlib.sha256(payload).hexdigest(), "date_columns": {}})
                for column in frame:
                    if "date" in column.lower():
                        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
                        row["date_columns"][column] = {"min": str(parsed.min()), "max": str(parsed.max()),
                            "invalid_count": int(parsed.isna().sum())}
                    if column.lower() in ["ticker", "symbol"]:
                        row["companies"] = int(frame[column].nunique())
                row["decision"] = "Needs target/provenance review; sample audited, not approved for modeling"
        rows.append(row)
        print(ref, "sample rows:", row.get("rows"), "decision:", row["decision"])
    (OUT / "kaggle_candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def audit_fred():
    payload = fetch("https://fred.stlouisfed.org/graph/fredgraph.csv", CACHE / "fred/VIXCLS.csv",
        params={"id": "VIXCLS", "cosd": "1995-01-01", "coed": "2025-12-31"})
    frame = pd.read_csv(BytesIO(payload))
    date_column = frame.columns[0]
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    values = pd.to_numeric(frame["VIXCLS"], errors="coerce")
    row = {"source": "FRED VIXCLS", "url": "https://fred.stlouisfed.org/series/VIXCLS", "rows": len(frame),
        "first_date": str(dates.min().date()), "last_date": str(dates.max().date()),
        "missing_values": int(values.isna().sum()), "nonmissing_values": int(values.notna().sum()),
        "sha256": hashlib.sha256(payload).hexdigest(), "role": "Market-regime supplement, not earnings-event replacement",
        "timing_rule": "Use the previous available observation date, strictly before reaction day; publication-time/vintage audit still required",
        "primary_model_uses_source": False}
    (OUT / "fred_audit.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print("FRED", row["rows"], "rows;", row["nonmissing_values"], "observed values")


def audit_sec():
    session = requests.Session()
    session.headers["User-Agent"] = "ISYE4031Group9 academic research https://github.com/vdwang/ISYE4031_Earnings_Project_Group9"
    rows = []
    for ticker, cik in [("AAPL", "0000320193"), ("JPM", "0000019617")]:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            payload = fetch(url, CACHE / "sec" / f"{ticker}_companyfacts.json", session=session)
            facts = json.loads(payload).get("facts", {}).get("us-gaap", {})
            for concept in ["Assets", "Liabilities", "NetIncomeLoss", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "EarningsPerShareDiluted"]:
                entries = [dict(value, unit=unit) for unit, values in facts.get(concept, {}).get("units", {}).items() for value in values]
                filing_dates = [x["filed"] for x in entries if x.get("filed")]
                rows.append({"ticker": ticker, "cik": cik, "concept": concept, "facts_including_repeated_filings": len(entries),
                    "first_filed": min(filing_dates) if filing_dates else None,
                    "last_filed": max(filing_dates) if filing_dates else None,
                    "status": "Actual concept records audited; no event-level feature merge yet", "source_url": url})
            print("SEC sample:", ticker, len(facts), "us-gaap concepts")
        except requests.HTTPError as exc:
            rows.append({"ticker": ticker, "cik": cik, "status": "Access failed: " + str(exc.response.status_code), "source_url": url})
            print("SEC access failed; no bypass attempted:", ticker, exc.response.status_code)
        time.sleep(1)
    pd.DataFrame(rows).to_csv(OUT / "sec_sample_audit.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, task in [("Kaggle", audit_kaggle_candidates), ("FRED", audit_fred), ("SEC", audit_sec)]:
        try:
            task()
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(name, "audit unavailable:", str(exc)[:180])
            (OUT / (name.lower() + "_audit_error.txt")).write_text(str(exc), encoding="utf-8")


if __name__ == "__main__":
    main()
