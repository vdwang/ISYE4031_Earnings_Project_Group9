"""Collect a dated S&P 500 issuer roster, bulk earnings and resumable prices.

No paid APIs or additional Alpha Vantage calls are used. Existing pilot caches
remain unchanged. Raw downloads stay in the ignored research cache.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import warnings

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import PRICE_COLUMNS
from src.utils import valid_price_frame

warnings.filterwarnings("ignore", message="'raise_errors' deprecated.*", category=DeprecationWarning)

CACHE = ROOT / "data/raw/research/sp500"
OUT = ROOT / "outputs/dataset_research/sp500"
SNAPSHOT = ROOT / "research/datasets/sp500_constituents.csv"
STOP = threading.Event()
SECTORS = {"Information Technology": ("Technology", "XLK"), "Communication Services": ("Communication Services", "XLC"),
    "Consumer Discretionary": ("Consumer Discretionary", "XLY"), "Consumer Staples": ("Consumer Staples", "XLP"),
    "Energy": ("Energy", "XLE"), "Financials": ("Financials", "XLF"), "Health Care": ("Health Care", "XLV"),
    "Industrials": ("Industrials", "XLI"), "Materials": ("Materials", "XLB"), "Real Estate": ("Real Estate", "XLRE"),
    "Utilities": ("Utilities", "XLU")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_file(url, path, maximum=25_000_000):
    if not path.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        if len(response.content) > maximum:
            raise ValueError("Unexpected public file size")
        path.write_bytes(response.content)
    return path


def github_file(repo, name, path):
    # The GitHub API is used because the raw-content host is unavailable here.
    if path.exists():
        return
    response = requests.get(f"https://api.github.com/repos/{repo}/contents/{name}", timeout=45)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("content"):
        response = requests.get(f"https://api.github.com/repos/{repo}/git/blobs/{payload['sha']}", timeout=60)
        response.raise_for_status()
        payload = response.json()
    path.write_bytes(base64.b64decode(payload["content"]))


def inputs():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    github_file("datasets/s-and-p-500-companies", "data/constituents.csv", CACHE / "roster.txt")
    public_file("https://huggingface.co/datasets/jwigginton/earnings-sp500/resolve/main/data/train-00000-of-00001.parquet", CACHE / "hf_earnings.parquet")
    public_file("https://huggingface.co/datasets/jwigginton/index-constituents-sp500/resolve/main/data/train-00000-of-00001.parquet", CACHE / "hf_constituents.parquet")
    github_file("dylan-jeon/earnings-surprise-analysis", "output/pead_full_results.csv", CACHE / "pead_results.csv")
    snapshot_path = CACHE / "snapshot_metadata.json"
    if not snapshot_path.exists():
        snapshot_path.write_text(json.dumps({"downloaded_date_utc": datetime.now(timezone.utc).date().isoformat(),
            "roster_source": "https://github.com/datasets/s-and-p-500-companies",
            "roster_is_current_snapshot_not_historical_membership": True}, indent=2), encoding="utf-8")
    snapshot = json.loads(snapshot_path.read_text())
    (OUT / "snapshot_metadata.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    roster = pd.read_csv(CACHE / "roster.txt", dtype={"CIK": str})
    roster["cik"] = roster.CIK.str.lstrip("0").str.zfill(10)
    preferred = {"GOOGL", "FOXA", "NWSA"}
    roster["preferred"] = roster.Symbol.isin(preferred)
    chosen = roster.sort_values(["cik", "preferred", "Symbol"], ascending=[True, False, True]).drop_duplicates("cik")
    canonical = chosen.set_index("cik").Symbol
    roster["canonical_symbol"] = roster.cik.map(canonical)
    roster["canonical_ticker"] = roster.canonical_symbol.str.replace(".", "-", regex=False)
    roster["is_canonical"] = roster.Symbol.eq(roster.canonical_symbol)
    roster["snapshot_date"] = snapshot["downloaded_date_utc"]
    roster.drop(columns="preferred").to_csv(SNAPSHOT, index=False)
    universe = chosen[["cik", "Symbol", "Security", "GICS Sector"]].copy()
    universe["ticker"] = universe.Symbol.str.replace(".", "-", regex=False)
    universe["sector"] = universe["GICS Sector"].map(lambda x: SECTORS[x][0])
    universe["sector_etf"] = universe["GICS Sector"].map(lambda x: SECTORS[x][1])
    universe["cohort"] = "sp500_snapshot"
    universe.to_csv(ROOT / "research/sp500_universe.csv", index=False)
    old = pd.read_parquet(CACHE / "hf_constituents.parquet")
    old["cik"] = old.cik.astype(str).str.lstrip("0").str.zfill(10)
    alias = dict(zip(old.symbol, old.cik.map(canonical)))
    alias.update(dict(zip(roster.Symbol, roster.canonical_symbol)))
    alias.update({k.replace(".", "-"): v for k, v in list(alias.items()) if isinstance(v, str)})
    pd.DataFrame([{"source_symbol": k, "canonical_symbol": v, "canonical_ticker": v.replace(".", "-")}
        for k, v in sorted(alias.items()) if isinstance(v, str)]).to_csv(OUT / "issuer_aliases.csv", index=False)
    manifest = []
    for name, url in [("roster.txt", "https://github.com/datasets/s-and-p-500-companies"),
        ("hf_earnings.parquet", "https://huggingface.co/datasets/jwigginton/earnings-sp500"),
        ("hf_constituents.parquet", "https://huggingface.co/datasets/jwigginton/index-constituents-sp500"),
        ("pead_results.csv", "https://github.com/dylan-jeon/earnings-surprise-analysis")]:
        p = CACHE / name
        manifest.append({"file": name, "sha256": sha(p), "bytes": p.stat().st_size, "source_url": url})
    pd.DataFrame(manifest).to_csv(OUT / "bulk_source_manifest.csv", index=False)
    print("Roster:", len(roster), "listings;", len(universe), "canonical issuers", flush=True)
    return universe


def price_one(ticker):
    path = CACHE / "prices" / (ticker + ".csv")
    prior = ROOT / "data/raw/research/expanded/prices" / (ticker + ".csv")
    if STOP.is_set() and not path.exists() and not prior.exists():
        return {"ticker": ticker, "status": "not_requested_after_rate_limit", "rows": 0}
    try:
        if path.exists():
            frame = pd.read_csv(path)
            status = "research_cache"
        elif prior.exists():
            frame = pd.read_csv(prior)
            frame.to_csv(path, index=False)
            status = "reused_pilot_cache"
        else:
            raw = yf.Ticker(ticker).history(start="2008-01-01", end="2026-02-01", interval="1d",
                auto_adjust=False, actions=False, raise_errors=True)
            if raw.empty:
                raise ValueError("No prices returned")
            frame = raw.reset_index()
            frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame]
            frame = frame.rename(columns={"adj_close": "adjusted_close"})
            frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
            frame["ticker"] = ticker
            frame = frame.reindex(columns=PRICE_COLUMNS)
            if not valid_price_frame(frame, ticker):
                raise ValueError("Invalid price response")
            frame.to_csv(path, index=False)
            status = "downloaded"
            time.sleep(.6)
        if not valid_price_frame(frame, ticker):
            raise ValueError("Invalid cached prices")
        return {"ticker": ticker, "status": status, "rows": len(frame), "first_date": frame.date.min(),
            "last_date": frame.date.max(), "sha256": sha(path)}
    except Exception as exc:
        if "ratelimit" in type(exc).__name__.lower() or "too many" in str(exc).lower():
            STOP.set()
        return {"ticker": ticker, "status": "failed_" + type(exc).__name__, "rows": 0}


def prices(universe, limit=None):
    (CACHE / "prices").mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(["SPY", *universe.sector_etf, "VOX", "IYR", *universe.ticker]))
    if limit:
        names = names[:limit]
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(price_one, name): name for name in names}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if len(rows) % 10 == 0 or row["rows"] == 0:
                pd.DataFrame(rows).sort_values("ticker").to_csv(OUT / "price_download_manifest.csv", index=False)
                print("Price progress", len(rows), "/", len(names), row["ticker"], row["status"], flush=True)
    pd.DataFrame(rows).sort_values("ticker").to_csv(OUT / "price_download_manifest.csv", index=False)
    print("Prices available:", sum(x["rows"] > 0 for x in rows), "/", len(names), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--limit-prices", type=int)
    args = parser.parse_args()
    universe = inputs()
    if not args.inputs_only:
        prices(universe, args.limit_prices)


if __name__ == "__main__":
    main()
