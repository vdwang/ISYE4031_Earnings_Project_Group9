# ISYE 4031 Group 9 · Milestone 3 Check-in 2

**Research question.** Once the five-session outcomes become observable, does updating the earnings-event AR-X slopes improve the next forecasts? A secondary question from Check-in 1 asks whether stable issuer differences justify a partially pooled intercept.

- [Working report (PDF)](output/pdf/ISYE4031_Group9_Milestone3_Checkin2.pdf)
- [Editable report text](reports/ISYE4031_Group9_Milestone3_Checkin2.md)
- [Eight-minute TA notes, results table, and proposed owners](reports/MILESTONE3_TA_NOTES.md)
- [Complete comparison table](outputs/milestone3/comparison.csv)

## Reproduce

From the repository root, use Python 3.13. The committed [event snapshot](research/datasets/timeseries_events.csv) and [Milestone 2 predictions](outputs/milestone2_timeseries/event_predictions.csv) are sufficient. No downloads from market-data providers or API keys are needed. See the [event feature dictionary](research/datasets/timeseries_dictionary.csv) and [dataset provenance](research/datasets/TIMESERIES_README.md).

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-milestone3.txt
python run_milestone3.py
python run_milestone3.py --demo 2024-02-15
python -m unittest discover -s tests -p test_milestone3.py -v
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Use `--out PATH` on both commands to write and read a separate output directory. The driver validates the event and lag dates, reproduces the Milestone 2 forecasts, fits the models, runs the replay and follow-ups, and writes tables and plots to `outputs/milestone3/`. It uses one BLAS thread. The archive contains past forecasts with known outcomes, so the demo recreates what could have been computed at each forecast close.

## Inspect the evidence

| Question | Evidence |
| --- | --- |
| Which labels changed the coefficients and when? | [Label updates](outputs/milestone3/label_updates.csv), [coefficient trace](outputs/milestone3/coefficient_trace.csv), [coefficient plot](outputs/milestone3/online_coefficients.png), [replay code](milestone3/online.py) |
| Were small forecast differences stable? | [Yearly scores](outputs/milestone3/by_year.csv), [paired date-block intervals](outputs/milestone3/paired_intervals.csv), [memory tuning](outputs/milestone3/memory_tuning.csv), [annual refits](outputs/milestone3/annual_refit.csv) |
| Why did changes help or hurt? | [Squared-error terms](outputs/milestone3/loss_decomposition.csv), [per-event forecasts](outputs/milestone3/forecasts.csv) |
| Did issuer adjustments help in short histories? | [Natural history groups](outputs/milestone3/by_history.csv), [controlled caps](outputs/milestone3/history_caps.csv), [offset ablation](outputs/milestone3/issuer_offset_ablation.csv), [issuer residual stability](outputs/milestone3/issuer_residual_persistence.csv) |
| What settings were used? | [Run specification and snapshot hash](outputs/milestone3/run_specification.json), [fit diagnostics](outputs/milestone3/fit_diagnostics.csv), [cap fit diagnostics](outputs/milestone3/history_cap_fits.csv), [neural seed sensitivity](outputs/milestone3/seed_sensitivity.csv) |

The primary frozen fits use 2010–2022 completed labels. For the online models, the 2023–2025 forecasts use the same initial fit and then update after outcomes mature. The best finite half-life was selected by mean RMSE over three pre-2023 rolling-origin windows. No forgetting had lower RMSE than any finite setting in every tuning window. The 30-day setting is a stress test. See the report for the two different historical evaluation statuses. The 2025 outcome scores were already seen in earlier project work.

The pooled and mixed neural models use the same single eight-unit tanh layer, weight penalty 0.001, and Gaussian objective, with issuer variance fixed at zero for the pooled fit. L-BFGS-B uses a 1,600-iteration budget, relative objective tolerance `1e-10`, and projected gradient tolerance `1e-6`. Seed 17 is the displayed reference, with paired runs at 29 and 53. Intervals use 2,000 circular calendar-date block resamples, seed 4031, at 14, 28 and 56 days for full-sample contrasts. These are approximate uncertainty estimates for differences in historical forecast performance conditional on the fitted models.

## Experiment log

1. Reproduced the Milestone 2 pooled event AR-X and training mean on the exact scored IDs. Added a forecast-by-forecast audit to show the day a realized return enters the regression.
2. Tested all-history updates and five finite memories before 2023. Kept the best finite setting and the short-memory stress test to see what happens when a model reacts strongly to recent news. One 2022 boundary event was available to the replay only after its January 2023 return closed.
3. Retained the original Check-in 1 pooled/mixed linear and neural comparisons. Training issuer variance was near zero. Tested whether removing the fitted offsets changed the same mixed network's predictions, checked earlier versus later issuer residual means, and kept fixed forecast IDs while capping history at 5, 10 and 20 events.
4. Checked yearly errors, annual refits, multiple neural starts and paired date-block intervals. The short-memory stress test was weaker, slow updates had mixed results, and the caps did not produce a consistent mixed-model advantage.

**Sources.** The timing, feature list and frozen comparator come from the team's [Milestone 2 time-series analysis](reports/MILESTONE2_TIMESERIES.md). The methodological adaptations are described in the report. [Goel, Bruce and Bernstein (2020)](https://doi.org/10.1109/MCS.2020.2990516) and [Simchoni and Rosset (2023)](https://www.jmlr.org/papers/v24/22-0501.html).

**AI disclosure for group review.** OpenAI Codex assisted with method research, implementation, execution, testing, and drafting the working materials. The group should rerun and review the results, confirm individual contributions, and revise the wording to reflect its own understanding before submission.
