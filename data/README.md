# Data files

`raw/earnings/` holds one Alpha Vantage JSON response per ticker and the combined `quarterly_earnings.csv`.
`raw/prices/` holds one Yahoo Finance daily price CSV and cache metadata JSON per ticker, including SPY.
`processed/earnings_events_all.csv` retains in-window events, including events with missing targets or predictors.
`processed/modeling_dataset.csv` contains events with a valid stock and SPY target and all required predictors.

Run `python run_pipeline.py` from the project root to populate these folders. Raw and processed data are ignored by Git because the source services may impose usage terms and because the files can be reproduced. Data gaps are left missing; no financial values are invented or imputed.
