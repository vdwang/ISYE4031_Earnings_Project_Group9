# S&P 500 model results

The larger dataset supports a small improvement from the simpler eight-feature linear regression for signed stock returns. Adding all 25 predictors, regularization or gradient boosting does not produce consistent gains across the later periods. Direction classification is mixed, and the monthly uncertainty diagnostics do not establish a dependable forecasting advantage.

These comparisons are exploratory. The coefficient tables describe the fitted models and do not provide feature-level significance tests.

## Data and evaluation

The locked dataset has 28,496 events from 495 snapshot S&P 500 issuers. We use 22,663 events in 2010–2022 for training, 3,876 in 2023–2024 for validation and 1,956 in 2025 for exploratory testing. One boundary-crossing event is purged. All models are fitted through 2022; none are refitted on validation or test observations.

The raw training-mean reference always predicts +0.288%; the SPY-adjusted training-mean reference predicts +0.083%. The zero reference predicts no change. Core OLS uses the eight retained derived pilot features, rather than the old pilot's raw EPS levels. The 25-feature candidates add historical earnings, market/sector context, rates and interactions.

Ridge penalties and logistic C are chosen only within training using purged rolling folds evaluating 2017–2018, 2019–2020 and 2021–2022. Feature scaling and clipping are refitted inside each fold. Ridge selection minimizes pooled squared error; logistic selection minimizes log loss. All selected Ridge penalties were 1,000 and logistic C was 0.01. These were the tested grid boundaries; no expanded search was made after viewing holdout results.

The candidate rules and dataset hash are in [run_specification.json](../outputs/sp500_models/run_specification.json). The 2025 outcomes had already been inspected in earlier project work, so this is not a fresh final holdout.

## Forecasting the signed stock return

MAE and RMSE below are in percentage points; lower is better. For example, RMSE 4.219 means a root-mean-square error of about 4.22 return percentage points.

| Model | 2023–24 MAE | 2023–24 RMSE | 2025 MAE | 2025 RMSE |
| --- | --- | --- | --- | --- |
| Training mean | 2.991 | 4.234 | 3.317 | 4.650 |
| Zero | 3.003 | 4.261 | 3.336 | 4.669 |
| OLS core 8 | 2.984 | 4.219 | 3.297 | 4.612 |
| OLS expanded 25 | 3.037 | 4.289 | 3.294 | 4.615 |
| Ridge 25 | 3.025 | 4.275 | 3.285 | 4.605 |
| Clipped Ridge 25 | 3.009 | 4.265 | 3.300 | 4.628 |
| Gradient boosting 25 | 3.138 | 4.400 | 3.623 | 4.944 |

Core OLS reduces RMSE by 0.015 percentage points (0.36%) in validation and 0.038 points (0.81%) in 2025, relative to the training-mean forecast. The error remains large relative to the small improvement. Core OLS also has slightly lower RMSE than the mean in each of 2023, 2024 and 2025 individually.

The 25-feature Ridge has the lowest trained-model RMSE in 2025, but performs worse than the mean in validation; choosing it based on 2025 alone would be hindsight selection. The core OLS is the trained model preferred by validation RMSE. Gradient boosting is worse than the simple references in both later periods.

## Stock returns minus SPY

The SPY-adjusted target uses the same reaction/end dates. A zero forecast is a particularly relevant reference for this target.

| Model | 2023–24 MAE | 2023–24 RMSE | 2025 MAE | 2025 RMSE |
| --- | --- | --- | --- | --- |
| Training mean | 2.732 | 3.885 | 3.105 | 4.275 |
| Zero | 2.727 | 3.883 | 3.102 | 4.272 |
| OLS core 8 | 2.736 | 3.884 | 3.112 | 4.280 |
| OLS expanded 25 | 2.763 | 3.907 | 3.125 | 4.304 |
| Ridge 25 | 2.762 | 3.905 | 3.124 | 4.302 |
| Clipped Ridge 25 | 2.761 | 3.910 | 3.117 | 4.300 |
| Gradient boosting 25 | 2.772 | 3.916 | 3.177 | 4.347 |

No fitted candidate consistently improves on the zero forecast for the SPY-adjusted target. The modest raw-return gain does not establish an ability to predict returns beyond the market benchmark.

## Predicting up versus down

Logistic regression predicts whether the stock's next-five-session return is strictly positive. Zero is grouped with nonpositive returns. The reference probability is the training positive frequency, 53.766%; its 0.5 classification threshold predicts up for every event. This majority reference is more informative than comparing accuracy with 50%.

| Period | Model | Accuracy | Balanced accuracy | ROC AUC | Log loss |
| --- | --- | --- | --- | --- | --- |
| 2023–24 | Training prevalence | 54.08% | 50.00% | 0.500 | 0.6898 |
| 2023–24 | Logistic regression 25 | 52.19% | 50.26% | 0.512 | 0.6927 |
| 2025 exploratory | Training prevalence | 55.27% | 50.00% | 0.500 | 0.6880 |
| 2025 exploratory | Logistic regression 25 | 55.47% | 52.99% | 0.563 | 0.6837 |

The classifier is worse than the majority reference in validation accuracy and probability loss. In 2025 it correctly classifies only four more events than the majority forecast, although its ROC AUC and probability loss improve. Its validation-to-test change prevents a claim of stable directional skill.

## Stability and source checks

The [yearly chart](../outputs/sp500_models/yearly_performance.png) and [grouped table](../outputs/sp500_models/grouped_performance.csv) report later-period errors by year, sector, company, source and current surprise flag. Small company groups are descriptive; R-squared is omitted for single-observation groups.

A separate sensitivity refits fixed core OLS and clipped-Ridge specifications on 2016 onward rows whose current source is not the Hugging Face archive, without retuning. It changes both the training population and the eligible evaluation population; it is not an isolated causal effect of changing sources. The current-source filter cannot certify every historical feature's vintage or provenance.

| Period | Recent-source fit | Training rows | Evaluation rows | RMSE pp |
| --- | --- | --- | --- | --- |
| 2023–24 | Training mean | 12898 | 3865 | 4.222 |
| 2023–24 | Zero | 12898 | 3865 | 4.255 |
| 2023–24 | OLS core 8 | 12898 | 3865 | 4.206 |
| 2023–24 | Clipped Ridge 25 | 12898 | 3865 | 4.224 |
| 2025 | Training mean | 12898 | 1956 | 4.646 |
| 2025 | Zero | 12898 | 1956 | 4.669 |
| 2025 | OLS core 8 | 12898 | 1956 | 4.620 |
| 2025 | Clipped Ridge 25 | 12898 | 1956 | 4.582 |

[Complete source-sensitivity metrics](../outputs/sp500_models/recent_source_sensitivity.csv) include both targets. Source disagreement, snapshot survivorship and incomplete release-time histories remain limitations despite the larger sample.

## Approximate uncertainty

The following descriptive paired bootstrap resamples calendar months, retaining all events within a sampled month. This preserves within-month market/firm dependence but does not fully account for cross-month overlap, persistent company effects or model-selection uncertainty. There are only 24 validation and 12 test month clusters. Intervals are exploratory diagnostics, not a definitive hypothesis test.

| Analysis | Period | Gain in percentage points | Approximate 95% interval |
| --- | --- | --- | --- |
| future_5d_return | 2025 | 0.038 | [-0.077, 0.164] |
| future_5d_return | 2023–24 | 0.015 | [-0.009, 0.040] |
| positive_return_classification | 2025 | 0.204 | [-6.769, 6.450] |
| positive_return_classification | 2023–24 | -1.883 | [-7.857, 2.468] |

For raw returns, gain is reference RMSE minus core-OLS RMSE; for classification it is logistic accuracy minus majority accuracy. Positive values favor the fitted model. Interpret the interval and its limitations rather than treating the point improvement alone as established skill.

All displayed intervals include zero. Under this approximation, the small core-OLS and classification point gains are not clearly distinguishable from no improvement.

## Reproduce and next steps

```powershell
.\.venv\Scripts\python.exe research\run_sp500_models.py
.\.venv\Scripts\python.exe research\summarize_sp500_models.py
```

The model script reads only the committed CSV and needs the root requirements; no new API downloads or raw caches are needed. The summary script recalculates saved holdout metrics and verifies that every model scores the same events. Fifteen automated tests pass, including training-only clipping, purged tuning folds and future-feature invariance.

Keep the core linear model and simple forecasts as the reference for subsequent work. A useful next investigation is source/date quality and predefined company/regime structure, with rolling validation and grouped uncertainty, before increasing model complexity. Do not select features, outlier rules or models to make the already-inspected 2025 result look better.
