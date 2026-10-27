# Meeting 1 feedback and actions

Source: Meeting 1 notes supplied by Victor on October 27, 2026, including the
group's October 26 discussion of a possible Kaggle source. These notes do not
establish a separate attendance record or individual completed contributions.

| Requested focus | Work completed on Milestone2updates | Remaining work |
| --- | --- | --- |
| Get more history per company; more companies is secondary | Audited cached history: 675 pre-2016 earnings records and 631 potential events in 2010-2025, including one with a missing earnings value | Obtain older price lookbacks, validate targets and lags, then assess usable counts before changing scope |
| Identify large surprises and potential outliers | Listed all seven absolute surprises over 100%, training Cook's distance flags, and the largest held-out errors | Inspect source records and release timing; do not automatically label genuine moves as errors |
| Test whether outlier removal improves the model | Compared original OLS, training-only surprise exclusion, training-only influence exclusion, and training-fitted surprise clipping | No treatment consistently improves both later periods; retain original baseline as primary |
| Make the dataset informative and explain its distinctive question | Added year/company/volatility error summaries and calendar-quarter stability checks; clarified the after-reaction signed-return question | Broaden historical conditions and document feature rationale; row count alone does not measure information |
| Explain significance and difficulty | Separated predictive usefulness, statistical significance and practical relevance; documented dependence and weak reference comparisons | Dependence-aware inference and rolling evaluation belong in the next stage; no significance claim is supported here |
| Consider time-series or longitudinal methods | Documented an event panel with repeated companies and overlapping windows | Compare simple panel/company effects and regularized regression within rolling splits before complex methods |
| Review the suggested Kaggle data | Audited version 7 summary and earnings files; preserved primary data and reported actual missingness and date coverage | Full price-file audit, licensing, release-time consistency, ticker aliases and corporate-action alignment are required before integration |

The current Milestone 2 report retains its 400 complete events and signed target.
No new companies, synthetic observations or selectively filtered test outcomes
have been added. The sensitivity results are diagnostics on already inspected
evaluation periods, not new evidence from an untouched test.
