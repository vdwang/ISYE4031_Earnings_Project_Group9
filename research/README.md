# Dataset quality research

## Current S&P 500 dataset

The requested scale-up is complete: **28,496 complete events, 495 companies,
25 numeric predictors, 2010–2025**. All 500 snapshot issuers were attempted.
Open the **[S&P 500 guide](datasets/SP500_README.md)** or
**[actual dataset](datasets/sp500_modeling_dataset.csv)**. The earlier 1,714-event
pilot below is retained separately. Read the
[S&P 500 source/quality report](../reports/SP500_DATASET.md) before modeling.

### Run the larger dataset models

The completed [model report](../reports/SP500_MODEL_RESULTS.md) compares mean/zero
forecasts, core/expanded OLS, raw/clipped Ridge, gradient boosting and logistic
direction classification. Scaling and clipping use training data only, with
purged rolling training folds for penalty selection. The already-inspected 2025
period stays exploratory. The small core-OLS raw-return gain is not clearly
separated from zero by the approximate monthly bootstrap diagnostic.

```powershell
.\.venv\Scripts\python.exe research\run_sp500_models.py
.\.venv\Scripts\python.exe research\summarize_sp500_models.py
```

These commands use the committed CSV and root dependencies, with no API calls
or raw caches required. Results, predictions, selected parameters, grouped
diagnostics and graphs are in `outputs/sp500_models/`. The summary checks the
saved forecasts against the reported regression/classification metrics and the
locked dataset hash. Model outputs are separate from the construction-audit
metadata under `outputs/dataset_research/sp500/`.

### Rebuild the S&P 500 panel

```powershell
.\.venv\Scripts\python.exe -m pip install -r research\requirements-sp500.txt
.\.venv\Scripts\python.exe research\collect_expansion.py --only context
.\.venv\Scripts\python.exe research\collect_sp500.py
.\.venv\Scripts\python.exe research\fill_sp500_gaps.py
.\.venv\Scripts\python.exe research\build_sp500.py
.\.venv\Scripts\python.exe research\profile_sp500.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The committed CSV can be used immediately. Rebuilding uses public bulk EPS
archives, a dated constituent registry, individual cached Yahoo prices and
Cboe/FRED context. No new Alpha Vantage calls are needed. For the exact submitted
source mix, the original private pilot earnings caches must also be present;
otherwise the builder chooses available bulk sources and resulting counts or
values may differ. This difference is visible in `earnings_source`.

`collect_sp500.py --inputs-only` collects/normalizes just the roster and bulk
inputs. The full command collects prices with two workers, reuses caches and
stops new network calls after a rate-limit exception. Failed symbols remain
listed in its manifest. Rerun to resume failed/missing requests when access
returns. `fill_sp500_gaps.py` attempts only companies missing from bulk earnings.
Current-snapshot sector labels and CIK-linked aliases are preserved explicitly.

Raw inputs are ignored under `data/raw/research/sp500/`; all-event audit data
are private under `data/processed/research/sp500_all_events.csv`. The shared
complete CSV and dictionary are under `research/datasets/`; all issuer statuses,
source hashes, missingness, flags and checks are under
`outputs/dataset_research/sp500/`. The profiler verifies every target, not just
a sample. No source-provided CAR/return column or clipped surprise is reused.

### Earlier pilot

This branch includes an actual expanded dataset: **1,714 complete events from
27 companies, 11 industry groups and 25 numeric predictors, 2010–2025**.
Open [the dataset guide](datasets/README.md),
[the dataset CSV](datasets/expanded_modeling_dataset.csv) or
[the current findings](../reports/EXPANDED_DATASET.md).
The original 400-event Milestone 2 snapshot is preserved. No new model has been
selected or fitted as part of this dataset expansion.

## Build the expanded dataset

From the project directory, using the existing environment and local `.env` key:

```powershell
.\.venv\Scripts\python.exe research\collect_expansion.py
.\.venv\Scripts\python.exe research\build_expanded_dataset.py
.\.venv\Scripts\python.exe research\profile_expansion.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The committed derived CSV is ready to inspect without downloading anything.
Rebuilding requires Alpha Vantage access, Yahoo Finance access and the public
Cboe/FRED CSVs. Install the root `requirements.txt` if setting up a new Python
environment. The collector limits new earnings requests to 20 per run and at
most 25 UTC-day attempts tracked by this research script; external uses of the
same key may consume the provider's remaining quota. It stops on a provider
decline rather than bypassing limits. A rerun reuses successful caches and can
fill unavailable companies when service access returns. Thirty companies are
planned in `universe.csv`; 27 actually supplied usable earnings in this run.
Check the download manifests for the three unavailable companies.

The builder writes a derived, complete-case CSV under `research/datasets/` and
private fuller audit tables under `data/processed/research/`. It saves coverage,
missingness, excluded rows, flagged observations and source-date ages under
`outputs/dataset_research/expanded/`. The profiler recalculates every target and
checks date horizons against cached prices, then creates the feature dictionary
and coverage figure. Source files have hashes in the download manifests.
The original `run_pipeline.py` still rebuilds the original 400-event study.

## Earlier source comparison scripts

The earlier [source comparison](../reports/DATASET_QUALITY_RESEARCH.md) and
ten-company history pilot can also be reproduced:

From the project directory with its existing environment:

```powershell
.\.venv\Scripts\python.exe research\audit_candidates.py
.\.venv\Scripts\python.exe research\audit_extended_history.py
.\.venv\Scripts\python.exe research\compare_audits.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The first script audits two bounded public Kaggle CSVs, reviews two other
publishers' metadata, downloads FRED VIX observations and attempts two SEC
Company Facts requests. It uses cached responses when available. The SEC
requests returned HTTP 403 during this audit; no SEC sample coverage is claimed.
If a source fails, inspect the error/status output before using its results.

The second script requires the existing private
`data/raw/earnings/quarterly_earnings.csv`. It obtains separate stock/SPY price
histories and builds candidate events for 1996–2025, 2010–2025 and 2016–2025.
Changing the optional `build_events` window leaves the normal pipeline defaults
at 2016–2025. The older count key `events_2016_2025` is retained for compatibility;
research summaries count the actual returned events instead of using that key.

The third script compares overlapping primary/pilot values, checks complete
event keys and finite numeric values, and measures coverage of a strictly prior
VIX observation. An observation-date join is not proof of publication-time
availability, so the joined VIX data are not approved modeling predictors yet.

Raw downloads are private caches under `data/raw/research/`; fuller candidate
event tables are under `data/processed/research/`. Both are ignored by Git.
The derived expanded dataset, schema and aggregate audit results are committed;
raw API payloads, keys and vendor price histories are not.
Existing vendor caches, an environment with the project dependencies and network
access are needed to reproduce this research; no subscription is purchased.
