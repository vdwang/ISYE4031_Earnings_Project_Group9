# Group 9 Meeting 2 progress note

Prepared October 27, 2026 for the October 29 check-in.

Bill Nguyen, Sarah Wang, Cindy Le, Phoenix Moore, Victor Wang.

## What changed after Meeting 1

The TA asked for deeper history per company, identification of large earnings
surprises and influential observations, outlier sensitivity, and a more informative
dataset. The branch now includes explicit case tables, training-only sensitivity
comparisons, residual/group/quarter checks, a history-coverage audit and a separate
audit of the Kaggle candidate discussed by the group. See MEETING1_FEEDBACK.md.

## Baseline and comparison

The unchanged main question predicts signed five-session return after
reaction-day close. The cleaned dataset has 400 complete company-quarter events.
OLS uses 10 predictors and training-only scaling; its split is 280 training,
80 validation and 40 exploratory-test events. Future targets cannot cross an
earlier split boundary. MAE and RMSE are in percentage points (pp).

| Target and period | OLS MAE / RMSE | Training mean MAE / RMSE |
| --- | --- | --- |
| Raw return, 2023-24 | 2.906 / 3.838 | 2.662 / 3.572 |
| Raw return, 2025 | 3.316 / 4.253 | 3.161 / 4.135 |
| SPY adjusted, 2023-24 | 2.381 / 3.085 | 2.099 / 2.818 |
| SPY adjusted, 2025 | 2.876 / 3.606 | 2.974 / 3.651 |

The full-data model remains the primary baseline. It does not consistently beat
simple forecasts. The inspected 2025 period is not a fresh test for later tuning.

## Outliers and sensitivity

There are seven training events with absolute surprise above 100%; five are
AMZN, one XOM and one WMT. The raw model flags 17 training events by Cook's
distance >4/n. We compared training-only large-surprise exclusion, influence
exclusion and training-fitted percentile clipping. All original evaluation
events remain scored; outcomes are not clipped.

Raw RMSE after large-surprise exclusion is 3.733/4.288 pp versus original
3.838/4.253. Influence exclusion gives 3.898/4.108; clipping gives 3.824/4.264.
None consistently improves both periods. These are sensitivity checks, not
grounds for removing difficult test outcomes or claiming significance.

## Data expansion and remaining questions

Cached earnings provide 631 candidate events in 2010-2025, including one missing
an earnings value; older prices, lags and targets still need validation. The
Kaggle earnings file ends in June 2021 and has many missing estimates/release
labels. It may support an older study but cannot supply our 2023-2025 evaluation.
See DATA_EXPANSION_REVIEW.md for the actual file audit and source access.

Ask whether these training-only sensitivity rules are appropriate, whether
deeper existing-company history is preferable to the older Kaggle candidate,
and which dependence-aware evaluation or longitudinal model fits Milestone 3.
Confirm actual member contributions and the proposed owners in the Word packet
before submission. The course deadline is October 30 at 11:59 p.m. Eastern.

## Materials and run order

- Working report: `output/docx/ISYE4031_Group9_Milestone2_Report.docx` and the matching PDF.
- Meeting packet: `output/docx/ISYE4031_Group9_Meeting2_Packet.docx`.
- Code: run `run_pipeline.py`, then `run_milestone2.py`, using the project's `.venv`.
- Candidate audit: optionally run `audit_kaggle.py --download`.
- Notebook: `notebooks/02_milestone2_analysis.ipynb`.
- Figures and detailed tables: `figures/milestone2/` and `outputs/milestone2/`.

OpenAI Codex assisted with implementation, analysis and writing. The group must
review the work, confirm contributions and follow course disclosure requirements.
