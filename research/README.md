# Dataset quality research

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
