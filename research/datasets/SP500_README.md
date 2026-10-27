# S&P 500 earnings dataset

Use **[sp500_modeling_dataset.csv](sp500_modeling_dataset.csv)** for the current
large research dataset. It contains **28,496 complete event rows from 495
companies**, with **25 numeric predictors** across **11 industry groups** during
**2010–2025**. This exceeds the requested 10,000-event target. The earlier
1,714-event file is retained as a smaller reproducibility snapshot.

## What to click

1. [Actual S&P 500 dataset](sp500_modeling_dataset.csv). On GitHub, use **Download
   raw file** if the large CSV preview is truncated.
2. [Column dictionary](sp500_feature_dictionary.csv) for definitions, units and
   which columns are outcomes rather than predictors.
3. [Full roster](sp500_constituents.csv) for all 503 listings and their canonical
   issuer mappings, including companies with no complete events.
4. [Company coverage](../../outputs/dataset_research/sp500/coverage_by_company.csv)
   for the exact count and status of every one of the 500 issuers.
5. [Research report](../../reports/SP500_DATASET.md) for sources, checks and limits.
6. [Coverage graph](../../outputs/dataset_research/sp500/dataset_coverage.png).

## What the rows mean

One row is one canonical company's earnings announcement and its subsequent
stock return. We consolidate multiple stock classes using the snapshot issuer
CIK, choosing one class per company. The roster contains 503 listings but 500
issuers; share classes are not counted as extra company announcements.

The roster was downloaded October 27, 2026. It is a **current-constituent snapshot**,
not a list of companies that belonged to the S&P 500 at every historical date.
It therefore retains survivorship/selection bias. CIK and ticker mappings describe
the snapshot issuer; predecessor businesses and reorganizations need separate
review before interpreting the historical rows as one unchanged legal entity.

All 500 issuers were attempted. FDXF, HONA and VYLR supplied no usable history
within 2010–2025. Q and SNDK supplied only one and three candidate announcements
respectively, insufficient for the required four prior earnings observations.
Their missing rows are not filled with invented history. The complete dataset
therefore covers **495 of the 500 issuers**, not all 500 with complete records.

## Use the columns correctly

The prediction cutoff is reaction-day close, where reaction day is the first
stock trading day strictly after the announcement date. The main target is
`future_5d_return`, the signed return from that close to the fifth later stock
session close. Negative and zero returns are retained. Returns are fractions:
`0.01` means 1%. Surprise is already a percentage: `10` means a 10% surprise.

Use the 25 numeric features described in the dictionary. They include earnings
history, momentum, volume, realized volatility, market beta, sector-relative
performance, lagged VIX/VVIX, lagged Treasury yields and two prespecified
interactions. Ticker, CIK, dates, source labels and audit flags are metadata by
default. All four outcome columns must be excluded from predictors.

`analysis_split` identifies 22,663 training rows, 3,876 validation rows and 1,956
exploratory-test rows. **Exclude the one `purged_boundary` row** when fitting or
scoring: its label crosses a period boundary. Fit preprocessing only on training
data. The already-inspected 2025 period is not a fresh final holdout.

Sources are selected before inspecting our forecast results. Each pilot company
retains its cached Alpha Vantage history. Elsewhere, a published Yahoo earnings
archive supplies 2015 onward, with the Hugging Face archive supplying earlier
history or entire companies unavailable in that published archive. Direct Yahoo
queries fill seven otherwise missing companies. `earnings_source` and
`source_symbol` make the choice visible. No precomputed returns or clipped
surprises from another study are used: we recompute both from raw EPS and prices.

Source archives disagree on some historical EPS values. For a useful robustness
comparison, filtering to dates from 2015 onward and excluding `hf_archive` leaves
**20,384 complete rows**, still above 10,000:

```python
data = pd.read_csv("research/datasets/sp500_modeling_dataset.csv")
analysis = data[data["analysis_split"] != "purged_boundary"].copy()
recent_sources = data[
    (data["reported_date"] >= "2015-01-01")
    & (data["earnings_source"] != "hf_archive")
].copy()
```

The four-prior-earnings features near a source transition can still incorporate
earlier archive values; use the report's source/period sensitivity guidance.
Exact release times, point-in-time estimate vintages and vendor corporate-action
consistency remain unverified. All target calculations and specified lag rules
were checked, but completeness does not establish error-free financial data.

Build instructions are in [research/README.md](../README.md). No new model has
been fitted on this dataset and no performance improvement is claimed yet.
