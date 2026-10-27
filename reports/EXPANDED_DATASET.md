# Expanded earnings dataset findings

**Earlier pilot snapshot:** the current larger implementation is the
[28,496-event S&P 500 dataset](SP500_DATASET.md). The 1,714-event analysis below
is retained for reproducibility.

The research branch now contains **1,714 complete earnings events from 27
companies in 11 industry groups, covering 2010–2025**, with **25 numeric research
predictors**. This is 4.29 times the original 400-event sample and 2.73 times
the earlier 629-event history pilot. The expanded dataset is built and checked;
model performance has not yet been estimated on it.

Open [the actual CSV](../research/datasets/expanded_modeling_dataset.csv) or
[the dataset guide](../research/datasets/README.md). The original Milestone 2
400-event data and report remain reproducible as a separate snapshot.

## Sample and quality

| Item | Actual result |
|---|---:|
| Companies planned | 30 |
| Companies with usable earnings | 27 |
| Candidate earnings events during 2010–2025 | 1,719 |
| Complete events using original required fields | 1,717 |
| Complete events with all enriched predictors | 1,714 |
| Excluded events | 5 |
| Numeric predictors | 25 |
| Descriptive industry groups | 11 |
| Duplicate event IDs | 0 |
| Nonfinite cells among the 25 predictors | 0 |
| Current EPS surprises above 100 percent in absolute value | 28 |
| Current estimated EPS below 0.05 USD in absolute value | 73 |

The sample adds BAC, GS, CVX, COP, PFE, MRK, PG, KO, HD, MCD, CAT, UNP, NUE,
FCX, NEE, DUK and SPG to the original ten companies. Twenty-six companies
have 64 complete events; META has 50. All five excluded candidates are early
META records with missing earnings/price lookbacks or insufficient prior
earnings history. Each exclusion is recorded with the missing variables.
No large surprises or negative-return observations were removed.

Thirty names were chosen before examining the expanded return outcomes to add
industry coverage and longer company histories. This is a purposive large-company
sample, not a random or historically representative universe. EQR returned no
quarterly records, so the collector stopped new earnings requests; CSCO and IBM
were left unrequested after that response. EQR price history also failed.
Those companies contribute no events. The script does not claim a specific
reason for the provider decline or bypass it with extra accounts/keys.

There are 939 positive, 773 negative and two zero stock targets. Mean signed
five-session return is 0.260%, with standard deviation 3.502 percentage points.
These describe the selected dataset and do not establish predictability.

## Sources beyond Kaggle

The implemented dataset uses four provider families. No Kaggle observations
are used in it. Download manifests record success/failure and source-file hashes.
Raw response files and keys remain private; the derived event table is committed.

| Source | What was actually obtained | What it adds |
|---|---|---|
| Alpha Vantage earnings | Existing ten-company caches plus successful new quarterly records for 17 companies | Announcement dates, actual/estimated EPS and surprises; prior earnings histories |
| Yahoo Finance via yfinance | Daily adjusted prices and volumes for companies, SPY and sector/proxy ETFs | Targets, momentum, volatility, beta and sector-relative performance |
| Cboe VIX and VVIX | Actual public CSV downloads; 9,302 VIX rows and 5,134 VVIX rows, including dates outside the event window | Market uncertainty and uncertainty about volatility |
| Federal Reserve via FRED DGS10 and DGS2 | Actual daily series downloads; each 8,087 date rows during 1995–2025, including 332 missing values | Interest-rate level and yield-curve context |

Alpha Vantage documents quarterly EPS estimates and surprise metrics. Its
standard free allowance is 25 requests per day; this collector caches results,
tracks its own UTC-day attempts and stops at provider declines. It does not
assume the key has no other daily usage. Historical estimates downloaded now
are not automatically the consensus vintage available before each announcement.
[Earnings documentation](https://www.alphavantage.co/documentation/#earnings),
[API allowance](https://www.alphavantage.co/support/#api-key).

Cboe directly publishes VIX and VVIX historical series. We select only a dated
close before each reaction day and compute the VIX change ending on that selected
date. Selected closes are one to five calendar days old in the complete sample.
This adds forward-looking market-risk context that a company's realized price
volatility alone does not measure. We retain Cboe attribution and do not treat
its data as an independent confirmation of earnings records.
[Cboe historical data](https://www.cboe.com/tradable_products/vix/vix_historical_data).

The DGS10 and DGS2 series are daily Treasury constant-maturity yields sourced
from the Federal Reserve's H.15 release. The current H.15 schedule posts after
the stock-market close, so the pipeline deliberately uses yields dated at least
seven calendar days before prediction. Actual selected observations are seven
to ten days old. The spread uses the selected 10-year minus 2-year observations.
Missing dates are matched backward, never forward. Retrospective revisions and
historical release vintages remain unresolved despite this conservative lag.
[DGS10](https://fred.stlouisfed.org/series/DGS10),
[DGS2](https://fred.stlouisfed.org/series/DGS2),
[H15 release schedule](https://www.federalreserve.gov/releases/h15/).

## What makes the added complexity useful

The numeric inputs comprise eight retained derived predictors and seventeen
additional predictors. The shared table omits raw reported/estimated EPS levels
and dollar surprise as default inputs, avoiding the exact dollar-EPS identity
and reducing raw split-scale comparability problems. Full source values remain
in the private audit table. This does not eliminate near-zero-estimate effects.

| Feature group | Added information | Question it can support |
|---|---|---|
| Previous four earnings | Prior mean/std surprise, beat rate, time since previous announcement | Does a repeated beat convey different information than an isolated beat? |
| Longer price history | Prior 60-session return, volatility and market beta | Does continuation depend on momentum or market sensitivity? |
| Sector and market context | Sector-relative prior return; market/sector-adjusted reaction | Was the reaction company-specific or shared with the broader market? |
| Options-implied uncertainty | Prior VIX, prior VVIX and five-observation VIX change | Does the same reaction behave differently in calm and uncertain markets? |
| Interest-rate environment | Lagged 10-year yield and 10-year minus 2-year spread | Do financing and macroeconomic conditions alter the earnings relationship? |
| Prespecified interactions | VIX times reaction return; surprise times reaction return | Does the reaction's association with later returns vary with uncertainty or surprise? |

This is a custom linked panel with a specific post-reaction cutoff, not a claim
that public source data are unique. Interactions and closely related volatility
features may be redundant; future modeling should compare feature groups and
regularization rather than presume that every additional column helps.

The fixed industry groups are descriptive labels, not reconstructed historical
GICS classifications. XLC and XLRE have shorter histories; the code uses VOX
and IYR respectively until enough primary-ETF history exists, recording the
actual proxy used. There are 54 VOX and 24 IYR event matches. Earlier VOX's
telecom exposure is an imperfect benchmark for today's communication-service
companies. ETF holdings and group definitions change; inspect results by proxy
and period before attributing a difference to a sector effect.

## Other researched sources and decisions

SEC Company Facts remains a promising source of accounting context, but the
earlier AAPL/JPM API tests returned HTTP 403. No SEC variables or coverage are
claimed in this dataset. Point-in-time filing joins would still need original
filing dates, revision handling and concept comparability.
[SEC APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

FINRA short-sale volume was investigated as a possible investor-positioning
feature. It is transaction activity, not outstanding short interest; interpreting
it as bearish positioning would be misleading. A valid extension would require
separate scope, venue and publication-time work. It was not downloaded or merged.
[FINRA explanation](https://www.finra.org/investors/insights/short-interest).

The Kenneth French Data Library is another possible research benchmark for
market/style exposures. Its published factor files are not a replacement for
company earnings events. We did not download or merge them in this build;
availability and version treatment need to be specified first.
[French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

The earlier Kaggle, SimFin and news-source comparison is retained in
[Dataset quality research](DATASET_QUALITY_RESEARCH.md). Adding a third-party
row count without matching actual earnings, prices, dates and required fields
would overstate usable sample size.

## Validation and limits

All **1,714** shared rows were checked against cached adjusted prices: stock and
SPY targets recalculate correctly, target horizons contain exactly five later
stock sessions, and date ordering/direction labels are correct. All contextual
source dates meet their specified lag rules. These checks found zero errors.
Twelve automated tests pass, including checks that future prices cannot change
predictors, current/same-date earnings cannot enter prior-four summaries, and
newer or stale macro observations cannot slip into the joins.

Chronological split labels identify 1,390 training events in 2010–2022, 216
validation events in 2023–2024 and 108 exploratory-test events in 2025. The code
checks and purges crossing target windows; none cross in this complete sample.
No preprocessing or model fitting has occurred. The 2025 period has already
been inspected in earlier work and is not a fresh final holdout.

Complete cases do not guarantee correct source values. Exact earnings release
times, historical estimate vintages, corporate-action consistency, near-zero
EPS denominators, survivorship and company/calendar dependence remain concerns.
Prior-four surprise summaries can inherit very large historical surprises, so
training-only robust transformations and sensitivity analyses remain necessary.
The 28 current large-surprise flags and 73 small-estimate flags overlap; they
are not counts of known errors. No EPS dollar-identity mismatch was found at
the 0.001 USD audit tolerance.

For the next modeling step, prespecify feature-group comparisons, use rolling
chronological validation with label-window purging and report company/industry
and period-specific performance. Compare simple regularized or longitudinal
models against the original simple forecasts. Assess uncertainty with repeated
company/calendar structure in mind. Extra events and complexity do not imply
better forecasts or statistical significance.

Reproduction commands are in [research/README.md](../research/README.md);
the schema and timing definitions are in
[feature_dictionary.csv](../research/datasets/feature_dictionary.csv).
