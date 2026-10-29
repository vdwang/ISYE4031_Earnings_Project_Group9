# Professor Milestone 2 feedback and implementation

Source: notes supplied by Victor on October 29, 2026. The notes request a basic
time-series model using prior quarters/years, a fitted regression with basic
diagnostics including a Q–Q plot and residual plot, and a larger dataset.

Victor also explicitly selected a separate daily time-series dataset alongside
the earnings-event analysis after the quarterly-event count limit was explained.

Branch `codex/milestone2-time-series` implements the following:

- 1,846,178 complete stock-day observations from 492 companies in 2010–2025,
  stored as sixteen public derived Parquet files; 2008–2009 provide warm-up only.
- Preservation of all 28,496 earnings events, with 28,368 complete temporal-model
  rows. Daily observations are not relabeled as earnings events.
- Explicit returns from the previous completed calendar quarter, the same
  calendar quarter last year and the previous completed calendar year.
- A basic within-company lag regression and AR-X panel model; an actual
  statsmodels AutoReg example for preselected AAPL; a separate event AR-X model
  for the original post-earnings five-session target.
- Q–Q and residual plots, AAPL residual ACF, basic normality/variance/serial
  diagnostics, chronological evaluation and source-date checks.

See [the completed report](MILESTONE2_TIMESERIES.md) and
[dataset guide](../research/datasets/TIMESERIES_README.md).
