"""Build an executable notebook exposing the Milestone 2 analysis outputs."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md("""# ISYE 4031 Group 9: Milestone 2
Bill Nguyen, Sarah Wang, Cindy Le, Phoenix Moore, Victor Wang.

**Question:** After the initial earnings reaction, can available information predict
the signed stock return over the following five trading sessions?

Prediction is at the close of the first stock trading day strictly after the
reported date. The raw signed return is primary; SPY-adjusted return is secondary.
The reaction-day jump is excluded, and negative outcomes are retained. This
notebook uses actual cached observations. Run `run_pipeline.py` first.

Prepared with Codex assistance using the existing cached snapshot. Individual
Milestone 2 work and proposed task assignments must be confirmed before submission.
""")
    code("""from pathlib import Path
import contextlib
import io
import json
import sys
import pandas as pd
from IPython.display import display, Image

ROOT = Path.cwd()
if not (ROOT / 'run_milestone2.py').exists():
    ROOT = ROOT.parent
assert (ROOT / 'run_milestone2.py').exists(), 'Open from project root or notebooks/'
sys.path.insert(0, str(ROOT))
import run_milestone2
with contextlib.redirect_stdout(io.StringIO()):
    run_milestone2.main()
OUT = ROOT / 'outputs/milestone2'
FIG = ROOT / 'figures/milestone2'
summary = json.loads((OUT / 'analysis_summary.json').read_text())
print('All Milestone 2 outputs rebuilt from processed data; source formulas checked.')
print('Python used:', summary['python'])
""")
    md("""## Data pipeline and quality
EARNINGS JSON -> parsed event dates and numbers -> restrict 2016-2025 ->
first-later-session match -> historical predictors and stock/SPY targets ->
all-events table and complete-case modeling table. No missing financial values
are imputed. Outside-window records are excluded; genuine extremes are retained.
Price caches cover 2015-11-02 to 2026-01-30 for each symbol.

All 400 event dates, earnings values, historical price/volume features and targets
are independently recalculated by the script, with an assertion on agreement.
This checks mechanics, not point-in-time availability of vendor values.
""")
    code("""display(pd.read_csv(OUT / 'data_quality.csv'))
display(pd.read_csv(OUT / 'price_quality.csv'))
display(pd.read_csv(OUT / 'missingness.csv'))
display(pd.read_csv(OUT / 'formula_checks.csv'))
""")
    md("""## Exploratory findings
All 400 events remain in the figures, including surprise tails. 223 returns are
positive (55.75%), 177 are negative, and none is zero. Mean +0.242%, median
+0.319%, SD 4.085 pp. The reaction-day relationship is modest (r=0.187); surprise
is weaker (r=0.045). Descriptive fitted lines are full-sample exploration and do
not set the held-out model. Associations do not establish causation.
""")
    code("""display(pd.read_csv(OUT / 'descriptive_statistics.csv', index_col=0))
display(Image(filename=str(FIG / '01_return_distribution.png')))
display(Image(filename=str(FIG / '02_predictor_relationships.png')))
display(pd.read_csv(OUT / 'year_summary.csv'))
display(pd.read_csv(OUT / 'ticker_summary.csv'))
""")
    md("""## Simple baseline and chronological evaluation
OLS uses an intercept and 10 predictors. Dollar earnings surprise is omitted
because it is exactly reported minus estimated EPS. The data retain that variable.
StandardScaler fits on training only. The model is fixed using 2016-2022 (n=280),
validation is 2023-2024 (n=80), and the exploratory test is 2025 (n=40).
Boundary-crossing target windows are purged from the earlier period; none occur
in this snapshot. No random split, tuning, or validation refitting is used.

Training-mean and zero forecasts provide references. MAE and RMSE are in
percentage points; R2 uses the evaluation-period mean as its denominator.
`mse_skill_vs_training_mean` compares squared error with the training reference.
Zero has no positive direction predictions; sign accuracy treats zero as nonpositive.
""")
    code("""print('Model predictors:', summary['model_features'])
display(pd.read_csv(OUT / 'split_summary.csv'))
metrics = pd.read_csv(OUT / 'baseline_metrics.csv')
display(metrics.loc[metrics['split'] != 'train'].round(4))
display(pd.read_csv(OUT / 'coefficients.csv').round(4))
""")
    md("""## Residuals and influential observations
The raw baseline is worse than the training mean in both held-out periods.
SPY-adjusted OLS is worse in validation and marginally better in 2025. This does
not establish robust predictive usefulness. Direction accuracy also does not
improve consistently on always forecasting a positive raw return.

The training design has full rank (11 columns including the intercept). Cook's
distance >4/n is an influence screen; flagged events remain in the data. The raw
model flags 17 training events. AMZN 2022-04-28 is most influential (D=0.191).
Held-out errors include AAPL 2025-07-31: +13.326% observed vs -0.649% predicted.
The residual plot shows large errors and time clusters; it does not verify
independence, constant variance or causality.
""")
    code("""display(Image(filename=str(FIG / '03_holdout_residuals.png')))
influence = pd.read_csv(OUT / 'training_influence.csv')
display(influence.loc[influence['target'] == 'future_5d_return']
        .sort_values('cooks_distance', ascending=False).head(10).round(4))
display(pd.DataFrame(summary['diagnostics']).T)
""")
    md("""## Preliminary answer and Milestone 3 priorities
Weak predictor associations did not produce consistent improvements over naive
forecasts. The 10-company pilot is not ready for a trading application. Repeated
companies, 597 overlapping target-window pairs and retrospectively revised data
limit conclusions. Date-only reports imperfectly separate premarket reactions.

Prioritize rolling chronological evaluation with label purging, dependence-aware
uncertainty, company/year performance, influence sensitivity, and a small set of
simple regularized models. Verify release timing and historical availability.
The inspected 2025 data must not become a fresh test after model tuning. Expansion
needs a historical ticker-selection rule and free access. Deployment needs a
tradable entry rule, spreads/costs and evidence of practical benefit.

## Questions for MTA
1. Is the first trading day after a date-only report an acceptable reaction-day convention?
2. Should raw return remain primary and SPY-adjusted return secondary?
3. How should signed forecasts inform positive-return decisions without outcome filtering?
4. What historical selection rule should guide an approximately 100-stock expansion?
5. How should rolling validation and uncertainty handle overlapping market windows?

Milestone 1 contributions are described in the supplied `ISYE 4031 Project.pdf`.
No TA feedback has been received or incorporated. Confirm each member's actual
Milestone 2 work; see `reports/TEAM_CONTRIBUTIONS.md` and
`reports/TA_MEETING_MATERIALS.md`. Source documentation and exact run instructions
are in `MILESTONE2_README.md`.
""")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "ISYE 4031 Milestone 2", "language": "python", "name": "isye4031-milestone2"},
        "language_info": {"name": "python", "version": "3.12.7"}})
    path = ROOT / "notebooks/02_milestone2_analysis.ipynb"
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
