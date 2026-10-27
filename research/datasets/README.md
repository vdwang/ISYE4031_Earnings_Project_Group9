# Expanded earnings dataset

Open **[expanded_modeling_dataset.csv](expanded_modeling_dataset.csv)** for the
current research dataset. It contains **1,714 real earnings events from 27
companies in 11 descriptive industry groups during 2010–2025**. Each event has
25 numeric research predictors, four outcome/target columns, timing/provenance
metadata and quality flags. The CSV contains no missing numeric predictors or
targets and no duplicate event IDs.

One row is one company earnings announcement. The prediction is made at the
close of the first stock trading day strictly after its reported date. The
main target is the signed return from that close to the close five sessions
later, excluding the reaction-day jump. Negative and zero returns are retained.
All target columns are future outcomes and must be excluded from predictors.

## What is different

Compared with the original 400-event study, the dataset has more companies,
six additional years and linked information about market uncertainty, interest
rates, sector performance, historical market sensitivity and previous earnings.
The sources are Alpha Vantage, Yahoo Finance, Cboe and the Federal Reserve via
FRED. No Kaggle records are used in this expanded dataset.

The new features support questions such as whether a positive earnings reaction
continues differently in a volatile market, or after repeated earnings beats.
An economic hypothesis makes this useful; adding columns by itself does not
establish predictive value or novelty.

## What to open

- [Dataset CSV](expanded_modeling_dataset.csv): the actual complete event table.
- [Feature dictionary](feature_dictionary.csv): every column, units, source and role.
- [Expansion report](../../reports/EXPANDED_DATASET.md): sources, checks and limitations.
- [Coverage figure](../../outputs/dataset_research/expanded/dataset_coverage.png): size and industry coverage.
- [Company coverage](../../outputs/dataset_research/expanded/coverage_by_company.csv): observations for each company.
- [Missingness and exclusions](../../outputs/dataset_research/expanded/excluded_events.csv): the five excluded candidates and reasons.
- [Outlier flags](../../outputs/dataset_research/expanded/flagged_events.csv): observations needing sensitivity review.

## Use the columns correctly

Most returns are **fractions**: `0.01` means a 1% return. EPS surprise is already
in **percent**: `10` means a 10% surprise. Treasury yields are percentage yields;
the yield spread is in percentage points. VIX and VVIX are index levels.

Use the 25 features listed in
[`model_features.json`](../../outputs/dataset_research/expanded/model_features.json).
The fixed `sector` grouping is an optional categorical descriptor, not one of
those numeric inputs. Fit any encoding or scaling on training data only. Ticker,
dates, outcomes and audit flags are not default model predictors. The source
dates let you inspect the observation selected for each contextual feature.
Prior-quarter summaries exclude the current announcement and require four
available preceding earnings records.

`analysis_split` uses 2010–2022 for training, 2023–2024 for validation and 2025
as an **exploratory** test. There are 1,390, 216 and 108 rows respectively.
The code purges labels that cross period boundaries; none of these complete
events cross one. The 2025 outcomes were already inspected in earlier work and
are not an untouched final test. A downloaded CSV row is not an independent
observation: companies repeat and return windows overlap across companies.

VIX/VVIX use a strictly prior dated close. Treasury yields use an observation
at least seven calendar days old. Source vintages remain retrospective; these
lags reduce timing risk but do not certify a point-in-time trading backtest.
Exact announcement times and a survivorship-free historical universe remain
unverified. The groups use fixed descriptive labels, and sector ETF proxies
change where XLC/XLRE history is unavailable. See the expansion report before
interpreting these as historical sector memberships.

Raw vendor payloads and price histories are not included here. This is a derived
course-research table with source attribution in the report, not a new license
for the underlying providers' data. Reproduction commands and dependencies are
in [research/README.md](../README.md).
