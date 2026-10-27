# Dataset quality research

This branch compares public sources and audits a longer-history pilot. It does
not replace the submitted 400-event dataset or select a new model. See
[the findings](../reports/DATASET_QUALITY_RESEARCH.md) for the recommendation,
source links, measured coverage and limitations.

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

Raw downloads are private caches under `data/raw/research/`; candidate event
tables are under `data/processed/research/`. Both are ignored by Git. Only
aggregate audit results are committed under `outputs/dataset_research/`.
Existing vendor caches, an environment with the project dependencies and network
access are needed to reproduce this research; no subscription is purchased.
