# ISYE 4031 post-earnings returns project

## S&P 500 dataset with more than 10000 events

The current research dataset has **28,496 complete events from 495 companies**,
25 numeric predictors and 11 industry groups during 2010–2025. All 500 issuers
in the downloaded S&P 500 snapshot were attempted; five lack enough usable
history. Multiple stock classes are consolidated by issuer so they do not
inflate event counts. This is a snapshot-constituent panel, not historical
index membership.

Start with the **[S&P 500 dataset guide](research/datasets/SP500_README.md)**,
download the **[actual CSV](research/datasets/sp500_modeling_dataset.csv)**,
or read the **[source and quality report](reports/SP500_DATASET.md)**.
Build commands are in [research/README.md](research/README.md).
The 400-event Milestone 2 report and 1,714-event pilot are retained as earlier
snapshots; no S&P 500 model has been evaluated yet.

## Earlier 27 company dataset

The earlier research snapshot contains **1,714 complete earnings events from
27 companies in 11 industry groups during 2010–2025**, with **25 numeric
predictors**. It combines earnings, stock/sector prices, Cboe volatility indexes
and Federal Reserve Treasury yields. Start with the
[dataset guide](research/datasets/README.md), open the
[actual dataset CSV](research/datasets/expanded_modeling_dataset.csv), or read the
[expansion findings and limitations](reports/EXPANDED_DATASET.md).
Reproduction commands are in [research/README.md](research/README.md).

This is a separate expanded research dataset. The completed Milestone 2 report
and original pipeline retain their 400-event snapshot; no expanded model has
been fitted or claimed to outperform it.

## Milestone 2

Meeting 2 updates are on branch `Milestone2updates`: training-only outlier
sensitivity, year/company/volatility diagnostics, a history-coverage audit and
a separate audit of the suggested Kaggle source. The new packet is
`output/docx/ISYE4031_Group9_Meeting2_Packet.docx`. See
`reports/MEETING1_FEEDBACK.md` and `reports/DATA_EXPANSION_REVIEW.md`.

The completed pilot analysis, baseline comparisons, diagnostics, four-page report,
and reproducibility instructions are described in [MILESTONE2_README.md](MILESTONE2_README.md).
After building the dataset, run `python run_milestone2.py` or open
`notebooks/02_milestone2_analysis.ipynb`. It uses the raw signed return as the primary
target and SPY-adjusted return as a secondary analysis. Both are compared with
training-mean and zero forecasts. The 2025 results are exploratory and have been
inspected, so they cannot serve as an untouched test for subsequent tuning.

## Question

After the initial market reaction to a quarterly earnings announcement, can information about earnings, price, volume, momentum, and volatility predict a stock's signed return over the following five trading days? `positive_return` and `positive_return_amount` permit a focused analysis of upward moves without discarding negative returns.

This is a preliminary analysis project for TA discussion, not an investment recommendation. The ten starting tickers are in `src/config.py`; replace or expand `TICKERS` there for a roughly 100-stock study. At four quarters per stock per year over 2016-2025, 100 stocks would yield about 4,000 raw events before filtering, subject to data availability and ticker-history issues.

## Data sources

- [Alpha Vantage EARNINGS](https://www.alphavantage.co/documentation/#earnings): quarterly EPS, estimates, and surprise. You need a personal API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key). API usage limits may delay a large download.
- [Yahoo Finance through yfinance](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html): daily OHLC, adjusted close when available, and volume for stocks and SPY.

The data window is 2016-2025 for earnings events. Prices are requested from 2015-11-01 through 2026-01-31 to provide earlier lookbacks and five later sessions. The yfinance end date is exclusive, so the code passes 2026-02-01. Alpha Vantage raw JSON and Yahoo price CSVs are cached locally. A valid cache is reused; `--refresh` requests data again. The key and generated data are ignored by Git.

## Setup and run

Use Python 3.10 or newer. From this project folder:

```bash
python -m venv .venv
```

Activate the environment: macOS/Linux `source .venv/bin/activate`; Windows PowerShell `.venv\Scripts\Activate.ps1`.

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then replace the placeholder with your Alpha Vantage key:

```text
ALPHA_VANTAGE_API_KEY=YOUR_REAL_KEY
```

Run:

```bash
python run_pipeline.py
```

The run downloads earnings, downloads stock/SPY prices, constructs events and features, cleans the data, and writes summary CSVs. Then open `notebooks/01_preliminary_analysis.ipynb` with Jupyter and run its cells. The notebook does not need to run to build the dataset. To retry or update already valid caches, run `python run_pipeline.py --refresh` (mind the Alpha Vantage rate limit).

## Event timing and definitions

One observation is one company quarterly earnings event. `reportedDate` is a calendar date without a reliable release time. We define `reaction_day` as the **first available stock trading date strictly after** `reportedDate`, even when a premarket release might have moved the price on the reported date. The prediction is made at that reaction day's close. A match more than seven calendar days later is rejected as likely missing stock history. The prediction window contains the next five stock trading days; `prediction_start_date` and `prediction_end_date` identify its first and fifth sessions. The stock target is `adjusted_close[end] / adjusted_close[reaction_day] - 1` when adjusted closes exist, otherwise the corresponding raw closes. Thus the reaction day's jump is not part of the target. SPY is measured between those **same calendar dates**. A missing SPY close makes the market-adjusted target missing.

The 5- and 20-day prior stock returns end at the close **before** reaction day. Volatility uses the 20 daily returns before reaction day. Abnormal volume divides reaction-day volume by the average of the previous 20 sessions, excluding reaction day. Reaction-day return uses that day's and the preceding day's close. The prior quarterly surprise is based on earlier reported events for the same ticker. All predictors are available no later than reaction-day close. If adjusted close is supplied for a ticker, returns use it consistently and rows with missing adjusted values remain missing; raw close is used only if the entire adjusted series is unavailable. `future_5d_return` remains signed; `positive_return` is 1 only when it exceeds zero, so a zero return is coded 0. `positive_return_amount` is the nonnegative part. No absolute-return target or random split is used.

## Files

| Path | Purpose |
| --- | --- |
| `run_pipeline.py` | Orchestrates the full pipeline. |
| `src/config.py` | Tickers, dates, paths, and required columns. |
| `src/download_earnings.py` | Fetches and caches quarterly earnings JSON; combines CSV. |
| `src/download_prices.py` | Fetches and caches stock and SPY daily prices. |
| `src/build_dataset.py` | Joins earnings to strictly later trading dates. |
| `src/features.py` | Calculates historical predictors and future targets. |
| `src/clean_data.py` | Flags gaps, writes both event tables and summaries. |
| `src/utils.py` | Shared conversion and price validation. |
| `tests/test_event_timing.py` | Synthetic checks for event timing and future-price leakage. |
| `notebooks/01_preliminary_analysis.ipynb` | TA-ready overview, plots, optional chronological baseline, and questions. |
| `data/README.md` | Data file descriptions. |
| `figures/` | Saved notebook plots. |
| `outputs/` | Summary CSVs and a 20-event sample. |

## Outputs and limitations

The pipeline creates `data/processed/earnings_events_all.csv`, `data/processed/modeling_dataset.csv`, `outputs/data_summary.csv`, `outputs/missingness_summary.csv`, and `outputs/event_sample.csv`. The three output CSVs initially contain headers only and are populated by the first successful run. The notebook saves three plots in `figures/`. Its optional linear regression uses 2016-2022 for training, 2023-2024 for validation, and 2025 for testing when those periods have data. Events whose target window crosses a split boundary are excluded from the earlier split.

The exact earnings release time is unknown, so the reaction-day convention can misclassify premarket releases. Current ticker symbols may not represent a survivorship-free historical large-cap universe, and delisted companies are absent. Vendor data may be incomplete, revised, or adjusted differently; daily adjusted closes mix price and dividend effects. Dates with missing features stay missing in the all-events file and are excluded from the complete-case modeling file. This can introduce selection bias. The baseline is exploratory only, and overlapping event horizons or market regimes can make errors dependent.

For the course proposal, document actual source access and sample size after the real download. The course instructions also call for source attribution and disclosure of AI assistance, as directed by the syllabus.
