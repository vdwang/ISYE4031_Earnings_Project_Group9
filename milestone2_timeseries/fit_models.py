"""Basic daily AR-X and event-lag regressions with Q-Q/residual diagnostics."""

import importlib.metadata
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.ar_model import AutoReg
from threadpoolctl import threadpool_limits

from milestone2_timeseries.build_data import ROOT, DAILY, EVENT, OUT
from milestone2_timeseries.features import AR_FEATURES, EXOG_FEATURES, DAILY_FEATURES, EVENT_FEATURES


def score(actual, prediction):
    error = np.asarray(actual) - np.asarray(prediction)
    return {"mae_pp": float(100 * np.abs(error).mean()), "rmse_pp": float(100 * np.sqrt(np.mean(error ** 2))),
        "direction_accuracy": float(np.mean((np.asarray(actual) > 0) == (np.asarray(prediction) > 0)))}


def pooled_fit(train, features, target, date_column, name):
    scaler = StandardScaler().fit(train[features])
    x = sm.add_constant(scaler.transform(train[features]).astype(float), has_constant="add")
    # Two-way clustering preserves company and common-date groups for coefficient SEs.
    groups = np.column_stack([pd.factorize(train.ticker)[0], pd.factorize(train[date_column])[0]])
    result = sm.OLS(train[target].to_numpy(), x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    table = pd.DataFrame({"feature": ["intercept", *features], "coefficient": result.params,
        "two_way_cluster_se": result.bse, "approximate_p_value": result.pvalues})
    table["coefficient_pp_per_training_sd"] = 100 * table.coefficient
    table.to_csv(OUT / (name + "_coefficients.csv"), index=False)
    (OUT / (name + "_fit_statistics.json")).write_text(json.dumps({"training_observations": int(result.nobs),
        "r_squared": float(result.rsquared), "adjusted_r_squared": float(result.rsquared_adj),
        "design_rank": int(np.linalg.matrix_rank(x)), "design_columns": x.shape[1],
        "standardized_condition_number": float(np.linalg.cond(x)),
        "nonfinite_cluster_standard_errors": int((~np.isfinite(result.bse)).sum())}, indent=2), encoding="utf-8")
    (OUT / (name + "_parameters.json")).write_text(json.dumps({"features": features,
        "training_means": scaler.mean_.tolist(), "training_scales": scaler.scale_.tolist(),
        "coefficients": result.params.tolist(), "covariance": "two-way clustered by company and prediction date"}, indent=2), encoding="utf-8")
    return scaler, result


def predict(fit, sample, features):
    scaler, result = fit
    return result.predict(sm.add_constant(scaler.transform(sample[features]).astype(float), has_constant="add"))


def residual_plots(name, residual, fitted, dates, title):
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    sm.qqplot(100 * np.asarray(residual), line="s", ax=ax, marker=".", markersize=3, alpha=.6)
    ax.set_title(title + "\nTraining residual normal Q–Q plot")
    ax.set_ylabel("Residual quantiles (percentage points)")
    fig.tight_layout()
    fig.savefig(OUT / (name + "_qq.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(100 * np.asarray(fitted), 100 * np.asarray(residual), s=5, alpha=.3, color="#29677f")
    axes[0].set(xlabel="Fitted return (%)", ylabel="Observed minus fitted (percentage points)")
    axes[1].scatter(dates, 100 * np.asarray(residual), s=5, alpha=.3, color="#29677f")
    axes[1].set(xlabel="Prediction date", ylabel="Observed minus fitted (percentage points)")
    for ax in axes:
        ax.axhline(0, color="#777777", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title + "  |  Training residuals")
    fig.tight_layout()
    fig.savefig(OUT / (name + "_residuals.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def daily_panel(panel):
    train = panel.loc[panel.analysis_split_1d.eq("train")].copy()
    fits = {"Pooled AR lags 1 2 5": (AR_FEATURES, pooled_fit(train, AR_FEATURES, "future_return_1d", "date", "daily_ar")),
        "Pooled ARX with quarter year": (DAILY_FEATURES, pooled_fit(train, DAILY_FEATURES, "future_return_1d", "date", "daily_arx"))}
    mean = float(train.future_return_1d.mean())
    metrics, by_year, examples, residual_series = [], [], [], []
    for split in ["train", "validation", "exploratory_test"]:
        sample = panel.loc[panel.analysis_split_1d.eq(split)]
        predictions = {"Training mean": np.full(len(sample), mean), "Zero": np.zeros(len(sample))}
        for name, (features, fit) in fits.items():
            predictions[name] = predict(fit, sample, features)
        for name, forecast in predictions.items():
            metrics.append({"analysis": "daily_next_session", "split": split, "model": name, "n": len(sample),
                **score(sample.future_return_1d, forecast)})
            work = pd.DataFrame({"date": sample.date.to_numpy(), "observed": sample.future_return_1d.to_numpy(), "predicted": forecast})
            for year, group in work.groupby(work.date.dt.year):
                by_year.append({"analysis": "daily_next_session", "year": year, "model": name, "n": len(group), **score(group.observed, group.predicted)})
            if name == "Pooled ARX with quarter year":
                work["squared_error"] = (work.observed - work.predicted) ** 2
                work["residual"] = work.observed - work.predicted
                grouped = work.groupby("date").agg(n=("observed", "size"), mean_observed=("observed", "mean"),
                    mean_prediction=("predicted", "mean"), mean_residual=("residual", "mean"), mean_squared_error=("squared_error", "mean")).reset_index()
                grouped["split"] = split
                residual_series.append(grouped)
                chosen = sample.ticker.eq("AAPL").to_numpy()
                example = work.loc[chosen].copy()
                example["split"] = split
                examples.append(example)
    pd.DataFrame(metrics).to_csv(OUT / "daily_metrics.csv", index=False)
    pd.DataFrame(by_year).to_csv(OUT / "daily_year_metrics.csv", index=False)
    pd.concat(residual_series).to_csv(OUT / "daily_date_aggregate.csv", index=False, date_format="%Y-%m-%d")
    pd.concat(examples).to_csv(OUT / "aapl_pooled_predictions.csv", index=False, date_format="%Y-%m-%d")
    return pd.DataFrame(metrics)


def aapl_autoreg(panel):
    """A genuine single-series AutoReg example; OOS predictions are rolling one-step."""
    sample = panel.loc[panel.ticker.eq("AAPL")].sort_values("date").copy().reset_index(drop=True)
    # y_t is the return realized at the next close. Its lag 1 is today's known return.
    for lag in [1, 2, 5]:
        exact = sample.future_return_1d.shift(lag)
        common = exact.notna()
        if not np.allclose(exact.loc[common], sample.loc[common, f"return_lag{lag}"], atol=1e-7, rtol=1e-7):
            raise ValueError("AAPL series has a gap; AutoReg would require explicit handling")
        sample[f"return_lag{lag}"] = exact
    train = sample.loc[sample.analysis_split_1d.eq("train")].copy()
    result = AutoReg(train.future_return_1d.reset_index(drop=True), lags=[1, 2, 5],
        exog=train[EXOG_FEATURES].reset_index(drop=True), trend="c").fit(
            cov_type="HAC", cov_kwds={"maxlags": 5})
    parameters = result.params.to_numpy()
    fitted_train = train.iloc[5:].copy()
    matrix = sm.add_constant(fitted_train[DAILY_FEATURES].to_numpy(dtype=float), has_constant="add")
    fit_error = float(np.max(np.abs(matrix @ parameters - result.fittedvalues.to_numpy())))
    if fit_error > 1e-9:
        raise ValueError("AutoReg and manual rolling one-step predictors disagree")
    summary_text = "\n".join(line.rstrip() for line in str(result.summary()).splitlines()) + "\n"
    (OUT / "aapl_autoreg_summary.txt").write_text(summary_text, encoding="utf-8")
    pd.DataFrame({"feature": ["intercept", *DAILY_FEATURES], "coefficient": parameters,
        "hac_se": result.bse.to_numpy(), "approximate_p_value": result.pvalues.to_numpy()}).to_csv(OUT / "aapl_autoreg_coefficients.csv", index=False)
    rows, forecasts = [], []
    mean = float(train.future_return_1d.mean())
    for split in ["train", "validation", "exploratory_test"]:
        group = sample.loc[sample.analysis_split_1d.eq(split)].copy()
        if split == "train":
            group = group.iloc[5:]
        prediction = sm.add_constant(group[DAILY_FEATURES].to_numpy(dtype=float), has_constant="add") @ parameters
        for name, values in [("Training mean", np.full(len(group), mean)), ("Zero", np.zeros(len(group))), ("AAPL AutoReg ARX", prediction)]:
            rows.append({"analysis": "aapl_next_session", "split": split, "model": name, "n": len(group), **score(group.future_return_1d, values)})
        forecasts.append(pd.DataFrame({"date": group.date, "split": split, "observed": group.future_return_1d, "prediction": prediction,
            "residual": group.future_return_1d - prediction}))
    residual = np.asarray(result.resid)
    residual_plots("aapl_autoreg", residual, result.fittedvalues, fitted_train.date, "AAPL AR-X")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    plot_acf(residual, lags=20, zero=False, ax=ax)
    ax.set_title("AAPL AR-X training residual autocorrelation")
    ax.set_xlabel("Lag in observed trading sessions")
    fig.tight_layout()
    fig.savefig(OUT / "aapl_autoreg_acf.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    lb = acorr_ljungbox(residual, lags=[5, 10, 20], model_df=3, return_df=True)
    lb.to_csv(OUT / "aapl_ljung_box.csv", index_label="lag")
    jb, jb_p, skew, kurtosis = jarque_bera(residual)
    bp = het_breuschpagan(residual, result.model._x)
    arch = het_arch(residual, nlags=5, ddof=8)
    diagnostics = {"n_training_residuals": len(residual), "autoreg_manual_prediction_max_error": fit_error,
        "jarque_bera": float(jb), "jarque_bera_p": float(jb_p), "skewness": float(skew), "kurtosis": float(kurtosis),
        "durbin_watson": float(durbin_watson(residual)), "breusch_pagan_lm": float(bp[0]), "breusch_pagan_p": float(bp[1]),
        "arch_lm": float(arch[0]), "arch_p": float(arch[1]), "covariance": "HAC with five lags",
        "acf_and_ljung_box_scope": "AAPL only; no cross-company concatenation", "ljung_box_model_df": 3}
    (OUT / "aapl_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "aapl_metrics.csv", index=False)
    pd.concat(forecasts).to_csv(OUT / "aapl_autoreg_predictions.csv", index=False, date_format="%Y-%m-%d")
    return pd.DataFrame(rows), diagnostics


def event_regression():
    event = pd.read_csv(EVENT, parse_dates=["reaction_day", "prediction_end_date"])
    event = event.loc[event.model_complete].copy()
    train = event.loc[event.analysis_split.eq("train")]
    lag_features = ["completed_event_return_lag1", "completed_event_return_lag4"]
    fits = {"Event autoregression": (lag_features, pooled_fit(train, lag_features, "future_5d_return", "reaction_day", "event_ar")),
        "Event ARX quarter year": (EVENT_FEATURES, pooled_fit(train, EVENT_FEATURES, "future_5d_return", "reaction_day", "event_arx"))}
    mean = float(train.future_5d_return.mean())
    rows, forecasts = [], []
    for split in ["train", "validation", "exploratory_test"]:
        sample = event.loc[event.analysis_split.eq(split)]
        predictions = {"Training mean": np.full(len(sample), mean), "Zero": np.zeros(len(sample))}
        for name, (features, fit) in fits.items():
            predictions[name] = predict(fit, sample, features)
        for name, prediction in predictions.items():
            rows.append({"analysis": "earnings_event_five_sessions", "split": split, "model": name, "n": len(sample),
                **score(sample.future_5d_return, prediction)})
        output = sample[["event_id", "ticker", "reaction_day", "future_5d_return"]].copy()
        output["split"] = split
        for name, prediction in predictions.items():
            output[name] = prediction
        forecasts.append(output)
    fit = fits["Event ARX quarter year"][1]
    fitted = predict(fit, train, EVENT_FEATURES)
    residual_plots("event_arx", train.future_5d_return.to_numpy() - fitted, fitted, train.reaction_day, "Earnings-event AR-X")
    pd.DataFrame(rows).to_csv(OUT / "event_metrics.csv", index=False)
    pd.concat(forecasts).to_csv(OUT / "event_predictions.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d")
    result = fit[1]
    (OUT / "event_model_diagnostics.json").write_text(json.dumps({"training_events": len(train),
        "design_rank": int(np.linalg.matrix_rank(result.model.exog)), "design_columns": result.model.exog.shape[1],
        "standardized_condition_number": float(np.linalg.cond(result.model.exog)),
        "marginal_residual_mean": float(np.mean(result.resid)), "marginal_residual_sd": float(np.std(result.resid, ddof=1)),
        "covariance": "Two-way clustered by company and reaction date",
        "no_pooled_acf": "Events are irregular within companies; concatenating companies would create false time adjacency"}, indent=2), encoding="utf-8")
    return pd.DataFrame(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(OUT / "daily_file_manifest.csv")
    for row in manifest.itertuples():
        if hashlib.sha256((DAILY / (str(row.year) + ".parquet")).read_bytes()).hexdigest() != row.sha256:
            raise ValueError("Daily data changed from the verified snapshot")
    summary = json.loads((OUT / "data_summary.json").read_text())
    if hashlib.sha256(EVENT.read_bytes()).hexdigest() != summary["event_dataset_sha256"]:
        raise ValueError("Event data changed from the verified snapshot")
    panel = pd.concat([pd.read_parquet(p) for p in sorted(DAILY.glob("*.parquet"))], ignore_index=True)
    specification = {"daily_target": "Next observed stock-session return", "daily_features": DAILY_FEATURES,
        "event_target": "Five-session stock return after earnings reaction close", "event_features": EVENT_FEATURES,
        "fit_window": "2010-2022", "validation": "2023-2024", "exploratory_test": "2025",
        "daily_models": ["Zero", "Training mean", "Pooled AR lags 1 2 5", "Pooled ARX with quarter year", "AAPL AutoReg ARX"],
        "aapl_selection": "AAPL fixed before model results; illustrative single-series diagnostics, not best-company selection",
        "forecast_mode": "Fixed training coefficients, rolling one-step forecasts using actually observed earlier returns",
        "no_tuning": "Lags and predictors fixed from professor notes; no holdout parameter search",
        "versions": {name: importlib.metadata.version(name) for name in ["statsmodels", "pandas", "numpy", "pyarrow", "scikit-learn"]}}
    (OUT / "model_specification.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    with threadpool_limits(limits=2):
        print("Fitting daily panel models", len(panel), "stock-days", flush=True)
        daily = daily_panel(panel)
        print("Fitting actual AutoReg example for AAPL", flush=True)
        aapl, diagnostic = aapl_autoreg(panel)
        print("Fitting event lag regressions", flush=True)
        event = event_regression()
    from milestone2_timeseries.verify_models import verify_models
    verify_models(panel)
    from milestone2_timeseries.report import write_report
    write_report(daily, aapl, event, diagnostic)
    print(daily.loc[daily.split.ne("train")].to_string(index=False))
    print(event.loc[event.split.ne("train")].to_string(index=False))
