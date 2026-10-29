# Milestone 2 daily and earnings event datasets

The time-series branch contains two separately labeled datasets for 2010–2025:

| Dataset | Observation | Stored rows | Purpose |
|---|---|---:|---|
| [Daily Parquet files](timeseries_daily/) | One company on one observed stock session | 1,846,178 | Basic daily autoregression and quarter/year predictors |
| [Event CSV](timeseries_events.csv) | One earnings announcement | 28,496 | Original five-session post-earnings question with temporal predictors |

The daily dataset covers 492 companies, compared with 495 in the earlier event
dataset. The original 500-issuer snapshot was retained. Five recent listings
(RDDT, SOLV, GEV, SNDK and Q) lack the required completed-year history, while
FDXF, HONA and VYLR lack study-window prices. These omissions and the 31,508
incomplete candidate stock-days are listed in
[company coverage](../../outputs/milestone2_timeseries/daily_company_coverage.csv).
No missing observations are replaced with artificial values.

**Daily stock-days are not extra earnings announcements.** The 1.85 million
observations are repeated, dependent stock observations; they are not 1.85
million independent experiments. The daily and event counts must not be added.

## What to open

- [Main Milestone 2 report](../../reports/MILESTONE2_TIMESERIES.md): models, results and diagnostics.
- [Feature dictionary](timeseries_dictionary.csv): field definitions and availability.
- [Small daily preview](timeseries_daily_preview.csv): ten rows per year for browser inspection; not the full dataset.
- [AAPL Q Q plot](../../outputs/milestone2_timeseries/aapl_autoreg_qq.png).
- [AAPL residual plots](../../outputs/milestone2_timeseries/aapl_autoreg_residuals.png).
- [Event model Q Q plot](../../outputs/milestone2_timeseries/event_arx_qq.png).
- [Data checks](../../outputs/milestone2_timeseries/data_verification.json).

## Read the actual daily data

Parquet is a compressed table format. The sixteen yearly files total about
68 MB and contain derived returns/features, not raw price histories. GitHub
cannot display them as a normal CSV preview. Clone/download the branch and use:

```python
from pathlib import Path
import pandas as pd

daily = pd.concat(
    [pd.read_parquet(p) for p in sorted(Path("research/datasets/timeseries_daily").glob("*.parquet"))],
    ignore_index=True,
)
```

The model can be reproduced without API access:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-timeseries.txt
.\.venv\Scripts\python.exe run_milestone2_timeseries.py --fit-only
```

`--build-only` reconstructs and verifies the data using the same private source
caches as the earlier S&P 500 pipeline. Source acquisition instructions remain
in [research/README.md](../README.md). Fresh vendor downloads can differ from
the cached snapshot. The default new command rebuilds, verifies and fits; the
older `run_pipeline.py` retains the original 400-event study.

## Forecast dates and past periods

Daily prediction occurs at date `t` close. `return_lag1` is the just-observed
return, lag 2 is the preceding observed session's return, and lag 5 is four
sessions earlier. The model forecasts the next session's return. These are
lags 1, 2 and 5 relative to the next-session response.

Calendar predictors use the last completed quarter, the same calendar quarter
one year earlier, and the previous completed calendar year. For example, a
prediction in 2020 Q2 uses 2020 Q1, 2019 Q2 and calendar year 2019. The current
unfinished quarter/year is not used. Source dates are stored for auditing.

Daily rows contain one-day and five-day targets, with separate target-end
dates and split fields. `analysis_split_1d` is used for daily model evaluation;
`analysis_split_5d` records the stricter five-day boundary rule. Training is
2010–2022, validation 2023–2024 and exploratory testing 2025. Exclude
`purged_boundary` rows for the chosen horizon. The next-day splits contain
1,477,215 training, 244,990 validation and 122,507 exploratory-test rows, with
1,466 otherwise complete rows purged at boundaries.

All 28,496 event rows are preserved. The new `model_complete` flag identifies
28,368 rows with all seven time-series predictors. Earlier completed earnings
outcomes enter the event model only once their five-session target window has
ended. Calendar quarters, fiscal quarters and the last four available events
are not assumed to be interchangeable.

Returns are fractions: `0.01` means 1%. Earnings surprise is a percentage:
`10` means a 10% surprise. Daily predictors use float32 storage for compactness;
targets retain float64 precision. Rounding tolerances are explicitly recorded.

The source roster is a current snapshot, not historical S&P membership.
Survivorship, vendor revisions, incomplete announcement times, shared market
shocks and company dependence remain relevant limits. More rows alone do not
establish useful forecasts.
