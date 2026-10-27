# Data expansion and informativeness review

## Current study and the first expansion step

The primary model remains the 400-event, ten-company 2016-2025 study. Its
distinctive question is whether information after the initial earnings reaction
helps forecast the following signed five-session return. We do not claim that
the public source data themselves are unique or that more rows ensure useful
prediction.

`history_coverage_audit.csv` counts 675 cached earnings records before 2016.
Restricting the cached earnings to 2010-2025 would yield 631 candidate events,
including one event missing at least one earnings value. These are not 631
validated modeling rows. The current stock/SPY price caches begin in November
2015. Older adjusted-price histories, lookbacks, prior surprises and future
target windows must be obtained and verified before any expanded analysis.
META begins later than the other companies; a longer panel would be unbalanced.
Report observations per company and year instead of assuming four usable events
per company every year.

Prioritize deeper history within the existing companies, then add companies
using a historical selection rule with a documented mix of industries. Keep
delisted firms and ticker changes visible where source access permits. Preserve
an original-data snapshot and make each expanded-data cleaning exclusion explicit.

## Suggested Kaggle candidate

Source: [US historical stock prices with earnings data](https://www.kaggle.com/datasets/tsaustin/us-historical-stock-prices-with-earnings-data),
tsaustin, requested version 7. Audit performed October 27, 2026.

The actual downloaded summary has 7,786 symbols. The earnings file has 168,603
rows dated April 16, 2009-June 14, 2021; none are dated 2022 or later.
Of these rows, 110,994 have both reported and estimated EPS. Estimated EPS is
missing in 57,256 rows, reported EPS in 32,387, and release-time labels in 60,203.
There are 774 zero EPS estimates. No exact duplicate rows or duplicate symbol/date
keys were found. Zero estimates need special handling before calculating a
percentage surprise; they must not be silently replaced.

The release-time column contains `pre` and `post`, which could help audit the
reaction-day convention after validation against original announcements.
They are categorical labels, not exact announcement timestamps. Historical FB
appears instead of META, requiring an explicit ticker mapping. The summary lists
GOOGL prices ending February 3, 2020 while its earnings continue into April 2021,
so a longer earnings series does not guarantee matching prices.

This candidate can support an older-history study, but it cannot by itself
provide the current 2023-2025 evaluation periods. Its underlying sources overlap
our vendors, so it is not an independent confirmation. The dataset page lists
its license as Unknown. Raw candidate files are excluded from Git; confirm
appropriate use and redistribution before sharing them. The approximately 2 GB
price file was not downloaded, so its completeness and adjustments remain
unverified. This audit does not claim an integrated Kaggle modeling dataset.

## Reproduce the candidate audit

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe audit_kaggle.py --download
```

Only the public summary and earnings files for version 7 are requested. Once
cached, use `audit_kaggle.py` without `--download`. Failures must be reported
rather than replaced with invented records. Audit JSON and pilot coverage tables
are in `outputs/milestone2/`; private candidate caches are in
`data/raw/kaggle_candidate/`. Hashes identify the two analyzed CSV files.

## Significance and Milestone 3

Statistical significance, better prediction and practical importance are
different questions. Removing outliers or adding observations solely to obtain
a favorable p-value would not answer the research question. The full-data OLS
does not consistently improve on reference forecasts, and sensitivity treatments
do not reverse that conclusion across both later periods.

For Milestone 3, use rolling chronological evaluation, label-window purging,
company/year error summaries and uncertainty methods suited to the repeated
company/calendar structure. Consider company effects or a simple longitudinal
model and regularization before complex time-series models. Earnings-event rows
are irregularly spaced and should not be treated as an equally spaced daily
series. Specify which question a model change answers. Check whether any apparent
gain persists across periods and companies, and evaluate realistic entry time and
transaction costs before discussing trading deployment.
