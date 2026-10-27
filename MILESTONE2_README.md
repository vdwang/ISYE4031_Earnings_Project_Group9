# Group 9 Milestone 2: reproducible analysis

Georgia Tech ISYE 4031, Fall 2026. Bill Nguyen, Sarah Wang, Cindy Le, Phoenix Moore, Victor Wang.

The four-page report is `output/pdf/ISYE4031_Group9_Milestone2_Report.pdf`.
The editable Word version is `output/docx/ISYE4031_Group9_Milestone2_Report.docx`.
The three-page Meeting 1 packet is `output/docx/ISYE4031_Group9_Meeting1_Packet.docx`.
The Meeting 2 packet is `output/docx/ISYE4031_Group9_Meeting2_Packet.docx`.
The report text is `reports/ISYE4031_Group9_Milestone2_Word_Source.md`.
`notebooks/02_milestone2_analysis.ipynb` displays the analysis and diagnostics.
`reports/TA_MEETING_MATERIALS.md` contains progress notes and meeting questions.
The analysis uses the existing cached snapshot; the report and Meeting 2 packet
were revised October 27, 2026. Meeting 1 notes supplied by the group are recorded
in `reports/MEETING1_FEEDBACK.md`, with a response to each requested focus.
Proposed task owners are explicitly labeled. The short contribution statements in
`reports/TEAM_CONTRIBUTIONS.md` require group confirmation before submission.

## Obtain the original data

The repository is https://github.com/vdwang/ISYE4031_Earnings_Project_Group9.
Use the included scripts in this submission package for the Milestone 2 additions.
All paths in code are relative to the project directory or derived from `__file__`.

1. Obtain a free Alpha Vantage key at https://www.alphavantage.co/support/#api-key.
2. Copy `.env.example` to `.env` and set `ALPHA_VANTAGE_API_KEY` locally.
3. `run_pipeline.py` requests `function=EARNINGS` for the 10 symbols in
   `src/config.py`. Keep quarterly records with reported dates in 2016-2025;
   earlier records supply the first lagged surprise. No intraday subscription is used.
4. yfinance requests daily stock and SPY prices with `auto_adjust=False`,
   `actions=False`, start `2015-11-01`, exclusive end `2026-02-01`.
5. Valid original JSON/CSV caches are reused. The reported analysis reused
   existing caches; no new download was needed. Exact original retrieval
   timestamps were not recorded. `outputs/milestone2/source_manifest.json`
   records source paths, sizes and SHA-256 hashes for the analyzed snapshot.

Source documentation: [Alpha Vantage EARNINGS](https://www.alphavantage.co/documentation/#earnings),
[yfinance PriceHistory](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html).
API availability, free-access limits and revised histories may change. The
downloader reports provider errors and does not substitute invented observations.
Do not purchase data for this project. A changed snapshot can change results;
compare hashes before describing a rerun as identical. This package excludes raw
vendor data and credentials; precise source-access instructions are provided instead.
The supplied team PDF is not included in the package.

## Environment and run order

Tested analysis environment: Python 3.12.7. Exact versions are recorded in
`requirements-milestone2.txt`; `requirements.txt` gives broader ranges.
The pinned requirements reproduce the environment used for the results.

From the project directory, Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-milestone2.txt
Copy-Item .env.example .env  # only for a new setup; then edit .env locally
.\.venv\Scripts\python.exe run_pipeline.py
.\.venv\Scripts\python.exe run_milestone2.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The existing `.env` must not be overwritten. Do not print or commit its key.
On macOS/Linux use `.venv/bin/python` after creating the virtual environment.
Optional notebook regeneration (also requires nbconvert 7.17.1 and nbformat 5.11.1):

```powershell
.\.venv\Scripts\python.exe -m pip install nbconvert==7.17.1 nbformat==5.11.1
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name isye4031-milestone2 --display-name "ISYE 4031 Milestone 2"
$env:JUPYTER_PATH = (Resolve-Path .venv/share/jupyter).Path
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks/02_milestone2_analysis.ipynb --ExecutePreprocessor.kernel_name=isye4031-milestone2 --ExecutePreprocessor.timeout=600
```

Optional editable Word-document rebuild (requires python-docx):

```powershell
.\.venv\Scripts\python.exe -m pip install python-docx
.\.venv\Scripts\python.exe reports/update_meeting2_sources.py
.\.venv\Scripts\python.exe reports/build_word_documents.py
```

Export the rebuilt report from Word or Google Docs as PDF when needed. The
committed PDF was exported from the visually verified four-page Word version.

The kernel is stored inside the project environment, so it uses the project's
installed packages rather than an unrelated global Jupyter installation.
The analysis is deterministic; no stochastic estimator, shuffle or sampling is
used, so no random seed is required. OLS and scaler are fitted once on training
only; validation/test do not choose predictors, transformations or tuning values.
Both targets and all reference predictions use the same observations and split.

## Variables and timing

For report date t, reaction day i is the first stock session strictly later than t,
within seven calendar days. Prediction occurs at close i; the outcome sessions
are i+1 through i+5. Adjusted closes are used for every ticker in this snapshot.
The raw-close fallback applies only when an entire adjusted series is unavailable.
Missing adjusted values are not individually filled from raw close.

| Variable | Definition / stored units |
| --- | --- |
| `future_5d_return` | P[i+5]/P[i]-1, signed decimal; primary target |
| `spy_future_5d_return` | SPY end/reaction-date price ratio minus 1, decimal |
| `market_adjusted_5d_return` | Stock target minus contemporaneous SPY target; secondary target |
| `positive_return` | 1 if raw target > 0, else 0; missing stays missing |
| `positive_return_amount` | max(raw target, 0); auxiliary, not a predictor |
| `reported_eps`, `estimated_eps` | Provider reported/estimated USD per share |
| `earnings_surprise` | USD/share difference; preserved in dataset, excluded from OLS |
| `earnings_surprise_pct` | Provider surprise percentage, e.g. 10 means 10% |
| `previous_earnings_surprise_pct` | Prior event for same ticker, percentage |
| `reaction_day_return` | P[i]/P[i-1]-1, decimal |
| `abnormal_volume` | V[i]/mean(V[i-20:i]), dimensionless ratio; end excludes i |
| `return_5d_before` | P[i-1]/P[i-6]-1, decimal |
| `return_20d_before` | P[i-1]/P[i-21]-1, decimal |
| `volatility_20d` | Sample SD of 20 prior daily returns, decimal |
| `spy_return_5d_before` | SPY five-session return ending before reaction day, decimal |

All 11 candidate features have prediction-time cutoffs by reaction-day close.
`earnings_surprise = reported_eps - estimated_eps` exactly within floating-point
precision, so the baseline retains 10 predictors plus an intercept. StandardScaler
fits within 2016-2022 only. Future targets, positive-return columns, future SPY
returns, outcome end dates and quality flags are never predictors.

Splits use reaction dates: 2016-2022 train (280), 2023-2024 validation (80),
2025 exploratory test (40). Target ends must precede the next period boundary;
none of these observations require purging. The train model is fixed for both
later periods. Do not re-label the inspected 2025 sample as an untouched test
for Milestone 3; use rolling evaluation and a new prespecified evaluation design.

## Generated outputs and verification

`run_pipeline.py` regenerates the less-filtered and complete-case event tables,
pipeline counts and missingness from original caches or newly downloaded sources.
`run_milestone2.py` independently recalculates every event date, feature and target
and fails on a mismatch above 1e-10. The present maximum difference is below 1e-12.

| Output under `outputs/milestone2/` | Purpose |
| --- | --- |
| `data_quality.csv`, `price_quality.csv`, `missingness.csv` | Counts, duplicates, missingness, timing, nonfinite values, overlapping windows |
| `formula_checks.csv` | All 400 events independently reconstructed from sources |
| `descriptive_statistics.csv`, `correlations.csv` | Full-sample exploration; never used to tune this baseline |
| `year_summary.csv`, `ticker_summary.csv` | Regime/group patterns and sample counts |
| `split_summary.csv` | Chronological dates/counts and target-end boundaries |
| `baseline_metrics.csv` | MAE/RMSE in percentage points; R2, reference skill, direction accuracy |
| `baseline_predictions.csv` | Local event-level forecasts and residuals for audit |
| `training_influence.csv`, `coefficients.csv` | Cook's distance, leverage, standardized residuals and scaled coefficients |
| `analysis_summary.json`, `source_manifest.json` | Report inputs, exact versions, source hashes |

Figures 01-03 are in `figures/milestone2/`; Meeting 2 adds figures 04-05. All 400
observations remain in the primary exploration; no surprise tails are clipped.
Cook's distance > 4/n and leverage
> 2p/n are diagnostic screens, not deletion rules or hypothesis tests.
Eight unit tests cover event timing, future-price invariance, split-boundary label
purging, simultaneous dates, held-out metric units, target separation and
training-only outlier transformations.

## Meeting 2 additions

The same `run_milestone2.py` command now also creates:

- `outlier_sensitivity.csv` and `outlier_sensitivity_rules.json`: original OLS,
  training-only exclusion of absolute current surprises above 100%, exclusion
  of original training Cook's-distance flags above 4/n, and clipping current/prior
  surprises to training 1st/99th percentiles. Every variant scores all 80 validation
  and 40 test events. It never clips outcomes or deletes difficult test rows.
- `large_surprise_events.csv`, `largest_training_influence.csv` and
  `largest_holdout_errors.csv`: explicit source-review and diagnostic cases.
- `performance_by_year.csv`, `performance_by_ticker.csv` and
  `performance_by_volatility.csv`: group-level errors; the volatility cut uses
  training median volatility only. Small company test samples limit conclusions.
- `holdout_residual_summary.csv` and `leave_one_quarter_out.csv`: tail errors and
  stability when a whole calendar quarter is omitted. The latter is a deterministic
  check, not a significance test or confidence interval.
- `history_coverage_audit.csv` and `earnings_coverage_by_year.csv`: cached-history
  counts. There are 631 candidate earnings events in 2010-2025, not 631 validated
  modeling rows; older price lookbacks and targets still need verification.
- Figures `04_baseline_comparison.png` and `05_outlier_sensitivity.png` under
  `figures/milestone2/`.

The original full-data OLS remains primary. No sensitivity treatment consistently
improves both later periods. These are checks on already inspected evaluation
data, not an untouched assessment for subsequent model selection. No random
sampling or stochastic estimator is used.

The separate `audit_kaggle.py --download` command fetches only the summary and
earnings files from version 7 of the suggested dataset; it does not fetch the
approximately 2 GB price file or change the primary modeling data. The actual
earnings file ends in June 2021 and has missing EPS and release-time values.
Raw candidate files are ignored by Git and excluded from the submission ZIP.
See `reports/DATA_EXPANSION_REVIEW.md` for source links, limitations and access.

Raw-return OLS RMSE is 3.838 pp (validation) / 4.253 pp (2025), versus
3.572 / 4.135 for the training mean. SPY-adjusted OLS is 3.085 / 3.606,
versus 2.818 / 3.651 for the training mean. Forecasting usefulness is unestablished.

Important limitations: retrospective vendor revisions, uncertain release times,
survivorship, 10-company scope, company/market dependence, influential events and
no tradable execution/cost evaluation. Chronological order cannot establish
point-in-time availability of retrospectively obtained earnings estimates.

## Course completion record

The supplied project proposal records prior contributions. New individual
Milestone 2 work must be confirmed by the group. Group 9's check-ins are scheduled
for October 22 and 29, 2:00-3:00 p.m. Eastern, George Tower 802 Collab; the deadline
is October 30, 2026, 11:59 p.m. Eastern. Record actual meeting outcomes and completed
work as they occur. Upload the report PDF and code/access package to Canvas once
per group.

AI disclosure: OpenAI Codex assisted with implementation, analysis, writing and
packaging. The team must review the results, understand the methods, confirm
contributions and apply the syllabus's disclosure requirements.
