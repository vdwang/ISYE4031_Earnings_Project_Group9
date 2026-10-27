# Dataset quality and expansion findings

Audited October 27, 2026. Research branch: `codex/dataset-quality-research`.

The best next pilot is to extend the existing companies to 2010–2025 and add
carefully timed market context. The actual download-and-join audit produced
**629 complete events, up from 400**, a 57% increase. A 1996–2025 pilot produced
1,048 complete events, but longer history adds changes in accounting, market
structure and source reliability that require investigation. These counts show
technical completeness, not verified point-in-time accuracy or improved prediction.

The original Milestone 2 data, models, figures and Word documents remain the
400-event analysis. No model was refitted or selected using these pilot data.
The three requested Word copies are in the user's Downloads folder.

## Measured history expansion

These are observed results from the existing Alpha Vantage earnings cache and
separately downloaded Yahoo Finance/yfinance adjusted price histories.

| Event window | Candidate earnings events | Matched to stock prices | Complete events | Excluded |
|---|---:|---:|---:|---:|
| 2016–2025 | 400 | 400 | 400 | 0 |
| 2010–2025 | 631 | 631 | 629 | 2 |
| 1996–2025 | 1,075 | 1,075 | 1,048 | 27 |

Completeness requires all eleven candidate predictors plus stock and SPY future
five-session returns. Complete pilot rows have no duplicate ticker/report-date
keys or nonfinite numeric cells. Counts and missingness are saved in
`outputs/dataset_research/extended_event_totals.csv`, `extended_event_coverage.csv`,
`extended_event_quality.csv` and the three `extended_missingness` tables.

The 2010 pilot has 64 complete events per company for nine companies and 53 for
META, an unbalanced panel. One candidate lacks estimated EPS and percentage
surprise; two lack prior surprise, and one lacks sufficient early price history
for several predictors. These missingness categories overlap across the two
excluded events. Early META announcement dates deserve source verification.

The 1996 pilot has 104–118 complete events for most companies, 84 for GOOGL and
53 for META. Twelve candidate events lack estimated EPS/percentage surprise and
22 lack prior surprise. Earlier listing dates and availability limit the panel;
do not invent earlier quarters to balance it. The 2010 complete pilot contains
13 absolute percentage surprises above 100%, versus seven in the original
sample and 22 in the 1996 pilot. These are flags for review, not automatic errors.

Fresh vendor prices reproduce all 400 overlapping event keys. Small numerical
changes occur: the maximum stock target difference is about **0.000069 percentage
points**, and the maximum market-adjusted target difference is 0.000101 points.
The original CSV was not overwritten. The comparison and original file hash are
recorded in `fresh_vendor_comparison.csv` and `primary_snapshot.json`.

## Comparison of alternative sources

Counts below distinguish downloaded observations from publisher claims.
The original tsaustin audit is recorded in `reports/DATA_EXPANSION_REVIEW.md`.

| Source | Size and dates checked | Fit for the project | Decision |
|---|---|---|---|
| Existing earnings plus extended prices | Actual audit: 629 complete events in 2010–2025; 1,048 in 1996–2025; same ten companies | Preserves EPS surprise and signed post-reaction target | Best immediate expansion pilot; verify older records |
| tsaustin Kaggle earnings, version 7 | Actual earnings file: 168,603 rows, 2009–2021; 110,994 with actual and estimated EPS | Useful older events and pre/post release labels; no 2023–2025 evaluation coverage | Older-history candidate only; price file and license unresolved |
| adarsh1077 EPS classification, version 1 | Actual CSV: 11,282 rows, 501 tickers, April 2020–March 2026; two exact duplicates | EPS-beat labels and precomputed features; lacks raw actual/estimated EPS and the required stock-return outcome | Do not adopt as a replacement without source and timing reconstruction |
| anthonypm Nasdaq calendar, version 248 | Actual CSV: 2,603 symbols, October 28–November 10, 2026 | Upcoming announcements, forecasts and time categories; no realized current-quarter EPS outcome | Useful for prospective collection, not this historical backtest |
| EarningsSignal | Publisher explicitly describes 3,500 synthetic excerpts; metadata reviewed | Simulated text-classification task | Reject as real financial observations |
| SEC Company Facts | Documentation reviewed; both sample API requests returned 403 | Financial statements and filing context, not analyst EPS consensus | Strong contextual supplement to investigate; not yet sample-validated |
| FRED VIXCLS | Actual CSV: 7,828 observed values from 1995–2025, plus 259 missing entries | Market uncertainty context across companies | Most feasible new information source; audit historical availability |
| SimFin | Documentation only: free plan lists 5,000 US stocks and five years of fundamentals | Financial-statement context; longer advertised histories require paid tiers | Free history insufficient as our full-window replacement |
| GDELT Global Knowledge Graph | Documentation reviewed; no historical news sample integrated | News coverage/tone could add information | Potential later extension; substantial entity/time matching work |
| Social Intel corporate-news reactions | Publisher claims 6,779 posts; metadata reviewed only | Social posts and derived signals, not company-quarter events | Low priority; first-seen timing, engagement leakage and license unresolved |

The EPS-classification CSV has no missing cells in its ten supplied columns,
but this does not establish source accuracy or feature availability. In
particular, a historical beat rate or streak can leak the current/future outcome
if its construction does not stop at the prediction cutoff. Its CC0 label does
not independently establish the rights to underlying vendor data.

The upcoming-calendar CSV has 410 missing EPS forecasts, 125 missing previous-year
EPS values, 411 missing previous-year report dates and 72 missing estimate counts.
There are no exact duplicate rows. A calendar row cannot be counted as a usable
historical earnings event with a future-return label. Kaggle file version and
SHA256 hashes are recorded in `kaggle_candidates.json`; raw files stay private.

## What would make the study more informative

Size and distinctiveness should address an economic question. A useful proposed
extension is: **Does the relationship between the initial earnings reaction and
the next five-session return depend on market uncertainty and whether the company
has repeatedly beaten expectations?** This builds on our timing design and
uses information that the current price/volume-only context may miss.

Start with the 2010 pilot and a lagged VIX measure. A strictly prior-observation
join covers all 629 pilot events, using a value no more than four calendar days
old. FRED identifies this as a daily closing CBOE index and states that values
can be revised. The CSV does not prove when each historical value was published.
Check publication/vintage availability and use a conservative lag before fitting.
Do not fill missing holidays with future observations.
[FRED series and attribution](https://fred.stlouisfed.org/series/VIXCLS).

For a second contextual feature, investigate profitability or leverage from
SEC filings already public before prediction. SEC provides financial-statement
XBRL APIs without an API key, with standardized reporting introduced in 2009.
Use original filing availability and preserve revisions; a period-end date is
not an availability date. Repeated reported facts are not additional events.
Our requests failed with 403, so no SEC feature coverage or successful merge is
claimed. Company Facts does not supply the analyst consensus needed for the
existing surprise measure.
[SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

GDELT offers downloadable news data and a Global Knowledge Graph. An optional
news-volume or tone measure would need company-entity matching, deduplication,
publication timestamps and a strict cutoff before prediction. Historical bulk
archives need a separate pilot; a rolling search API should not be assumed to
cover 2016–2025. This is more work than adding a lagged market index, so defer it
until a reproducible earnings panel is established.
[GDELT data](https://gdeltproject.org/data.html).

SimFin's current free plan lists five years of fundamentals; the longer histories
on its site are paid-tier features. No account or paid access was used.
[SimFin plans](https://www.simfin.com/en/prices/).

## Recommended sequence and remaining shortcomings

1. Keep the original 400-event results as a reproducible Milestone 2 snapshot.
2. Verify a sample of older earnings dates, large surprises and split treatment,
   especially the earliest META records. Record each exclusion and its reason.
3. Predefine the 2010 panel as a pilot, audit lagged VIX availability, and compare
   a small, motivated set of additional features using chronological validation.
4. Add companies under a documented historical selection rule across industries
   after the within-company history checks. The present ten-company sample still
   emphasizes surviving large companies; more years do not fix that bias.
5. Ask the TA/library whether existing institutional I/B/E/S/CRSP access is
   available. Georgia Tech access and permitted usage have not been verified.

Exact release timestamps, historically available EPS estimates, vendor revisions,
corporate actions, repeated companies and overlapping target windows remain
unresolved concerns. Prefer original announcements/filings when checking dates.
The strictly-next-trading-session rule is a reproducible convention, but can be
late for a before-open announcement. Compare timing definitions once trustworthy
release labels are available.

Increasing rows does not guarantee statistical significance, independent
observations or lower forecast error. Use grouped uncertainty appropriate to
company/calendar dependence, purge label windows crossing split boundaries and
avoid selecting cleaning rules using held-out outcomes. The 2025 period has
already been inspected and must remain an exploratory test. If expansion changes
the modeling choices, disclose that history and use rolling validation or a
genuinely unseen later period for future evaluation.

## Candidate publisher links

- [tsaustin historical stock prices with earnings](https://www.kaggle.com/datasets/tsaustin/us-historical-stock-prices-with-earnings-data)
- [EPS Beat Prediction](https://www.kaggle.com/datasets/adarsh1077/epsclassification)
- [Nasdaq earnings calendar](https://www.kaggle.com/datasets/anthonypm/nasdaq-earnings-calendar-auto-updating)
- [EarningsSignal synthetic text](https://www.kaggle.com/datasets/kaziaishikuzzaman/earningssignal-earnings-call-tone-and-guidance)
- [Corporate news reactions](https://www.kaggle.com/datasets/hexsyro/earnings-and-corporate-news-reactions)

Reproduction commands and private-cache locations are in `research/README.md`.
