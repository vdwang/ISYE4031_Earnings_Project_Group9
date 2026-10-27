# S&P 500 dataset expansion

**Modeling update:** the completed [large-data model comparison](SP500_MODEL_RESULTS.md)
is now available. This document records the dataset construction and its limits.

The dataset research branch now provides **28,496 complete event rows from 495
companies in the downloaded S&P 500 snapshot**, covering January 5, 2010 through
December 19, 2025. There are 25 numeric predictors and 11 industry groups. The
dataset is 16.6 times the 1,714-event pilot and exceeds the requested minimum of
10,000 usable events. No model was refitted or evaluated during this expansion.

Start with [the dataset guide](../research/datasets/SP500_README.md) or download
[the actual CSV](../research/datasets/sp500_modeling_dataset.csv).

## Universe and coverage

The October 27, 2026 download of the maintained constituent list contains 503
stock listings representing 500 unique CIKs. We choose one canonical class per
issuer; GOOGL, FOXA and NWSA are preferred when their second class is also listed.
The full list and mapping are committed. The roster source's latest constituent
file commit inspected during collection was dated October 26, 2026.
[Maintained constituent data](https://github.com/datasets/s-and-p-500-companies).

This is a **snapshot-constituent study**, not a historically reconstructed S&P 500
membership panel. Backdating today's roster does not recover firms that exited,
failed or were acquired, and many selected companies joined the index later.
Current sector labels are descriptive, not historical GICS assignments. The
`Date added` column alone is insufficient to reconstruct entry/exit intervals.

| Coverage measure | Count |
|---|---:|
| Snapshot issuers attempted | 500 |
| Stock listings in the source roster | 503 |
| Issuers with candidate earnings | 497 |
| Candidate earnings during 2010–2025 | 28,941 |
| Issuers with complete enriched events | 495 |
| Complete enriched event rows | 28,496 |
| Excluded candidate events | 445 |
| Duplicate issuer/date keys | 0 |
| Numeric predictor columns | 25 |

All 500 canonical tickers and 14 market/sector proxy tickers were attempted for
prices. Prices were available for 511 of the 514 requested symbols. FDXF, HONA
and VYLR had no usable earnings or price history in the study window. Q had one
candidate announcement and SNDK had three; neither has the four prior
announcements needed for the full feature set. No data were invented to obtain
500 complete companies. `coverage_by_company.csv` lists every issuer's status.

We map historical symbols only where the old and current constituent files
share a CIK, including punctuation normalization for vendor symbols. We do not
guess mappings based on a similar company name. Direct vendor histories can
still include predecessor businesses across reorganizations; a snapshot CIK
does not certify historical legal/business continuity. Such changes require
review before longitudinal company-effect interpretations.

## Sources and selection

| Earnings lineage | Complete events | Selection rule |
|---|---:|---|
| Existing Alpha Vantage pilot caches | 1,714 | Retain each pilot company's whole cached history |
| Published Yahoo earnings archive | 19,045 | Prefer for other available companies from 2015 onward |
| Hugging Face earnings archive | 7,570 | Earlier history, or companies absent from the published archive |
| Direct Yahoo gap queries | 167 | Previously missing companies only |

The Hugging Face file contains 44,503 raw records, of which 42,036 have both
estimated and reported EPS. Future/provisional rows with missing reported EPS
are excluded. Fixed EST timestamps are used only to recover calendar dates;
they are not treated as verified release times. Its company registry provides
the historical CIK mapping. Publisher provenance and historical estimate-vintage
documentation are incomplete.
[Earnings archive](https://huggingface.co/datasets/jwigginton/earnings-sp500),
[company registry](https://huggingface.co/datasets/jwigginton/index-constituents-sp500).

The public earnings-study repository supplies date and raw actual/estimated EPS
fields. Its collection code identifies Yahoo/yfinance as the earnings source.
Its analysis clips its own surprise variable, so we discard that variable and
all supplied return outcomes. We independently calculate the untrimmed EPS
surprise and our five-session post-reaction targets. No conclusions, fitted
coefficients, code or forecast results from that study are adopted.
[Published source data and collection code](https://github.com/dylan-jeon/earnings-surprise-analysis).

Direct public Yahoo earnings queries recovered CRH, TWLO, P, BE, RDDT, FERG and
SKYD. Queries for FDXF, HONA and VYLR returned no usable study-window records.
Alpha Vantage's individual-call limits were not bypassed: no additional Alpha
Vantage requests were made for this expansion. The price collector uses two
workers, caches successful results, and stops new requests after a detected
rate-limit response. Source hashes and download statuses are committed; raw
vendor files and credentials stay private.

Stock, SPY and sector/proxy price histories are obtained independently through
yfinance. Cboe VIX/VVIX and Federal Reserve/FRED Treasury series come from the
previous verified downloads. The latter add market-risk and rate-environment
context. VIX/VVIX source observations precede reaction by one to five calendar
days; Treasury source observations precede it by seven to ten days. These rules
address timing conservatively, not historical data revisions.
[Cboe historical indexes](https://www.cboe.com/tradable_products/vix/vix_historical_data),
[FRED DGS10](https://fred.stlouisfed.org/series/DGS10),
[FRED DGS2](https://fred.stlouisfed.org/series/DGS2).

## Quality checks and source disagreements

EPS surprise is calculated consistently as
`100 * (actual EPS - estimated EPS) / abs(estimated EPS)`.
A zero estimate leaves percentage surprise missing; missing quarters are not
skipped when constructing the prior-four summaries. No surprise or target is
clipped, and no held-out outlier is removed. There are 834 current surprises
above 100% in absolute value, 538 small estimates below 0.05 USD in absolute
value, and 29 announcements within 45 days of the preceding announcement.
Those groups overlap and are review flags, not established errors.

The two bulk archives have 15,771 same-date overlaps. Estimated EPS differs by
more than 0.01 USD in 6,739, and reported EPS differs by that threshold in 2,616.
Derived percentage surprise differs by more than five percentage points in
2,091. These discrepancies may reflect estimate vintages, precision, accounting
basis or adjustment treatment; this audit does not resolve which record is
correct. Adding rows must not be described as eliminating those quality issues.

Sources are chosen by company and a fixed 2015 boundary rather than appending
all records from every provider. This prevents multiplying same-quarter records
because vendors report slightly different dates. The chosen lineage remains
visible in every shared row. The lack of trustworthy fiscal-quarter identifiers
in the bulk archives prevents a complete quarterly identity audit. Our key is
issuer plus reported date; short announcement gaps require manual review.

A useful robustness subset retains dates from 2015 onward and excludes rows
whose current source is `hf_archive`: it has **20,384 complete rows**. It still
uses retrospective vendor data. Prior-four summaries near 2015 can inherit
earlier-source inputs; a stricter sensitivity could start in 2016 and require
the preceding four records to share a source. This has not been fitted or
selected using forecast performance.

Every shared event was independently checked against its cached price series:
reaction is the first later stock session, its gap is at most seven calendar
days, the target ends at the fifth following stock session, and both stock
and SPY targets recalculate within numerical tolerance. Direction labels,
market-adjusted targets, macro source-date lags and split boundaries also pass.
No duplicate issuer/date keys or nonfinite predictor cells were found.
Thirteen automated tests pass, including equivalence of the scalable features
to the earlier implementation and invariance to future prices.

## Modeling implications

The larger panel brings repeated firms, changing market conditions, unequal
histories, common calendar shocks and overlapping return windows. Those are
substantive difficulties for Milestone 3, rather than a reason to add a complex
model without a hypothesis. The 25 numeric predictors cover earnings history,
price/volume, longer volatility and beta, sector-relative performance, market
uncertainty, rates and two prespecified interactions. The complete dictionary
defines units, timing and roles for every column.

There are 22,663 training rows in 2010–2022, 3,876 validation rows in 2023–2024,
and 1,956 exploratory-test rows in 2025. One otherwise complete row is labeled
`purged_boundary` because its target crosses a split cutoff; exclude it from
fitting/scoring. The 2025 period has already been inspected and remains
exploratory. Fit preprocessing on training only and use rolling chronological
validation, grouped uncertainty and feature/source ablations for later work.

The main unresolved limits are snapshot survivorship, unverified exact release
times, retrospective EPS estimates, corporate-action/business continuity,
historical source disagreement and dependent observations. Larger sample size
does not by itself establish significance, independence or forecast gains.

See [reproduction instructions](../research/README.md),
[the column dictionary](../research/datasets/sp500_feature_dictionary.csv), and
[all integrity results](../outputs/dataset_research/sp500/integrity_checks.json).
