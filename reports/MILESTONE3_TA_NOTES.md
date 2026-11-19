# Group 9 · Check-in 2 meeting notes

**Group 9 meeting window.** Thursday, November 19, 2–3 p.m. Eastern, George Tower 802 Collab. The rubric allocates approximately eight minutes per group. Confirm your actual slot with the group. The working materials go to the TA before the meeting.

Send the [working report](../output/pdf/ISYE4031_Group9_Milestone3_Checkin2.pdf), the [GitHub branch and reproduction instructions](https://github.com/vdwang/ISYE4031_Earnings_Project_Group9/tree/milestone3-checkin2), and the [results table](../outputs/milestone3/comparison.csv). Use these notes to rehearse the reasoning and prepare a question. Confirm each member's own contributions before describing them to the TA.

## Eight minutes

**0:00–1:00 · Research question and response to feedback.** “At the first trading close after the earnings date, can we forecast the return over the next five sessions? Milestone 2’s static AR-X gave mixed results. After Check-in 1, we kept the issuer-effects question and tested whether letting the AR-X coefficients update when returns complete would improve forecasts.”

**1:00–2:30 · Demonstrate a real update.** Run `python run_milestone3.py --demo 2024-02-15` from the branch root. The 31 returns that ended February 14 entered before the February 15 forecasts. The reaction-return coefficient moved from −0.039984 to −0.040165 pp per training standard deviation. At that forecast close, ALB’s forecast was −0.218821 pp with frozen coefficients and −0.107808 pp with the 730-day online fit. The difference reflects the full updated coefficient vector and the history before this batch. Point to [the replay loop](../milestone3/online.py) and [the coefficient plot](../outputs/milestone3/online_coefficients.png). To show a boundary example, run `python run_milestone3.py --demo`. A December PAYX event enters only after its outcome ends on January 3.

**2:30–4:15 · Compare and explain.** The training mean was better in 2023–2024. The 730-day replay beat frozen AR-X by 0.0038 RMSE pp in that window, then lost by 0.0221 pp in exploratory 2025. Both intervals cross zero. The 30-day replay made much larger changes. Its forecast spread was 1.275 pp versus 0.326 for frozen AR-X in 2023–2024. The change-size term added 1.694 pp² and its alignment with the frozen model's misses was negative, raising squared error further. The same diagnostic shows why the frozen model gained on the mean in 2025. Its forecasts moved by a similar amount across periods, but those moves lined up better with actual returns in 2025.

**4:15–5:45 · Explain the Check-in 1 follow-up.** The mixed linear model was almost identical to frozen AR-X. The neural network can capture interactions but its extra forecast changes raised error slightly in 2023–2024. Mixed effects address a persistent issuer bias. Short histories tell us to shrink noisy estimates, but the data also need evidence of persistent issuer variance. The mixed net estimated issuer SD of 0.0035 pp against residual SD of 4.06 pp. Across two training eras, issuer residual means correlated only −0.02 among 411 sufficiently observed issuers. The mixed net’s small advantage with a ten-event cap vanished as evidence for issuer effects when we switched off the fitted offsets. The same mixed network’s RMSE moved by less than 0.000006 pp. The change came from the shared nonlinear fit. The 5- and 20-event comparisons had no gain.

**5:45–7:00 · Reflection and limits.** “We expected recency or issuer adjustments to rescue some predictions. The strongest experimental difference was that aggressive recency made forecasts more variable and worsened 2023–2024 errors. Slow updating and issuer offsets gave little repeatable benefit. Our assignment uses 2025 as exploratory because the team had already inspected it. Coefficient movement is evidence that the procedure updates. The data leave open whether the true return relationship changed.”

**7:00–8:00 · Ask for feedback and agree on revisions.** “Should our final report emphasize the 30-day failure and error decomposition, and keep the issuer-history test as the secondary finding? If a comparable point-in-time 2026 sample is available, can it serve as the prospective check?” Close with the proposed ownership below.

## Results to keep open

RMSE in return percentage points. All models share 3,869 forecast events in 2023–2024 and 1,947 events in exploratory 2025. Lower is better. MAE and sample breakdowns are in the linked tables. [MAE and RMSE](../outputs/milestone3/comparison.csv), [paired intervals](../outputs/milestone3/paired_intervals.csv).

| Model | 2023–2024 | 2025 exploratory |
|:---|---:|---:|
| Training mean | **4.2346** | 4.6373 |
| Frozen AR-X | 4.2655 | 4.6035 |
| Mixed AR-X | 4.2655 | 4.6035 |
| Pooled neural net | 4.2693 | **4.5931** |
| Mixed neural net | 4.2692 | 4.5951 |
| Online AR-X, all history | 4.2619 | 4.6191 |
| Online AR-X, 730-day memory | 4.2617 | 4.6256 |
| Online AR-X, 30-day memory | 4.4725 | 4.6951 |

## Questions the TA may ask

- **Where does the improvement come from mathematically?** A candidate forecast differs from its reference by δ. If the reference miss is e, then the candidate miss is e − δ. Squaring gives the change δ² − 2δe. A model needs enough alignment with the reference's misses to justify moving the prediction. [Loss decomposition](../outputs/milestone3/loss_decomposition.csv).
- **Why did you show an all-history model and two half-lives?** All-history updating tests the effect of new information. Finite memory additionally reduces old information's weight. The 730-day model was best among finite settings in pre-2023 tuning, and all history was better still in every tuning fold. The 30-day setting tests whether aggressive adaptation helps. [Rolling-origin tuning](../outputs/milestone3/memory_tuning.csv), [Goel, Bruce and Bernstein](https://doi.org/10.1109/MCS.2020.2990516).
- **Could the model have known this return on the forecast day?** It knows the earnings information and close used to construct predictors by the forecast cutoff. A five-session outcome remains unavailable to the fitting loop until an earlier window ends strictly before that forecast date. Same-close outcomes wait. Every event on a forecast date sees the same state. A lagged event return also needs an earlier completed window. [Replay code](../milestone3/online.py), [timing tests](../tests/test_milestone3.py).
- **Why did you use partial pooling for short histories?** A company with few events has an unstable average residual. The model borrows strength across companies through the fitted issuer variance. A near-zero variance produces near-zero adjustments even for many firms with long histories. The controlled caps keep the scored firms and events fixed. Capping keeps their latest events, so it varies calendar composition along with training size. [Cap table](../outputs/milestone3/history_caps.csv), [offset ablation](../outputs/milestone3/issuer_offset_ablation.csv), [Simchoni and Rosset](https://www.jmlr.org/papers/v24/22-0501.html).
- **How would you distinguish time variation from noisy outcomes?** Score the frozen and all-history methods prospectively on the same dates and consider whether a predeclared regime or sector adjustment produces repeatable alignment. The 28-day bootstrap speaks to uncertainty in average historical forecast differences and retains nearby dates together. It does not describe uncertainty around an individual future return.

## Remaining work before November 22

These owners follow the Check-in 1 proposals and need group confirmation. Add each person’s actual completed contribution before submission.

| Member | Proposed revision |
|:---|:---|
| Bill Nguyen | Verify coverage and the comparison between natural short histories and the controlled caps. |
| Sarah Wang | Confirm 2026 source coverage and whether release timestamps and estimate vintages can be audited. |
| Cindy Le | Recheck the delayed-label example, target close and cutoff logic against the dataset. |
| Phoenix Moore | Explain the shrinkage formula and review the interpretation of the ablation. |
| Victor Wang | Rerun the evaluation and verify the paired intervals and final report numbers. |
