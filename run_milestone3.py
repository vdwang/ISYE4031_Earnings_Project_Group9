"""Reproduce Group 9's delayed-label study and Check-in 2 evidence."""

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from milestone2_timeseries.features import EVENT_FEATURES, split_labels
from milestone3 import mixed
from milestone3.evaluation import (
    COMPARISONS, EXPANDING, FAST, FROZEN, MEAN, MIXED, MIXED_NEURAL, NEURAL, TARGET, TUNED,
    bootstrap_indices, loss_terms, metrics, paired_interval, scores,
)
from milestone3.online import OnlineARX, replay
from research.modeling import rolling_folds

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "research/datasets/timeseries_events.csv"
OUT = ROOT / "outputs/milestone3"
ORIGIN = "2023-01-01"
PERIODS = {"2023–2024": (ORIGIN, "2025-01-01"), "2025 exploratory": ("2025-01-01", "2026-01-01")}
HALF_LIVES = (np.inf, 730, 365, 180, 90, 30)
COEFFICIENTS = ["intercept", *EVENT_FEATURES]
STATIC_MODELS = (MEAN, FROZEN, MIXED, NEURAL, MIXED_NEURAL)
MAIN_MODELS = (*STATIC_MODELS, EXPANDING, TUNED, FAST)
SHARED_MIXED_NEURAL = "Mixed neural net, zero-offset prediction"
FEATURE_DATES = [
    "event_lag1_source_date", "event_lag4_source_date", "quarter_return_lag1_source_date",
    "quarter_return_lag4_source_date", "year_return_lag1_source_date",
]


def read_panel():
    frame = pd.read_csv(DATA, dtype={"cik": str}, parse_dates=[
        "reported_date", "reaction_day", "prediction_end_date", *FEATURE_DATES])
    total_rows = len(frame)
    frame = frame.loc[frame.model_complete].sort_values(["reaction_day", "event_id"]).reset_index(drop=True)
    if (frame.event_id.duplicated().any() or frame.cik.isna().any()
            or not np.isfinite(frame[[*EVENT_FEATURES, "future_5d_return"]]).all().all()):
        raise ValueError("Expected unique complete events")
    if (not frame[FEATURE_DATES].lt(frame.reaction_day, axis=0).all().all()
            or not frame.reported_date.lt(frame.reaction_day).all()
            or not frame.reaction_day.lt(frame.prediction_end_date).all()):
        raise ValueError("Event or predictor dates violate the timing protocol")
    if not split_labels(frame.reaction_day, frame.prediction_end_date).eq(frame.analysis_split).all():
        raise ValueError("Snapshot split labels disagree with date rules")
    frame[TARGET] = 100 * frame.future_5d_return
    return frame, total_rows


def get_design(train, *samples):
    scaler = StandardScaler().fit(train[EVENT_FEATURES])
    return [np.column_stack((np.ones(len(frame)), scaler.transform(frame[EVENT_FEATURES])))
            for frame in (train, *samples)]


def online_prediction(train, stream, origin, half_life):
    if not train.prediction_end_date.lt(origin).all() or not stream.prediction_end_date.ge(origin).all():
        raise ValueError("Initialization and streaming outcomes overlap the origin incorrectly")
    past, future = get_design(train, stream)
    model = OnlineARX.initialize(past, train[TARGET].to_numpy(), origin, half_life)
    return replay(model, stream, future, stream[TARGET].to_numpy())


def tune_memory(train):
    tuning = []
    for fold, past, future in rolling_folds(train):
        origin = fold[:4] + "-01-01"
        eligible = train.loc[train.prediction_end_date.ge(origin)
                             & train.reaction_day.le(future.reaction_day.max())].reset_index(drop=True)
        for half_life in HALF_LIVES:
            forecast, _, _ = online_prediction(past, eligible, origin, half_life)
            result = metrics(future[TARGET], pd.Series(forecast, index=eligible.event_id).loc[future.event_id])
            tuning.append({"fold": fold, "half_life_days": half_life, **result})
    table = pd.DataFrame(tuning)
    # Select a finite memory to test the recency hypothesis, with infinity as its control.
    finite = table.loc[np.isfinite(table.half_life_days)]
    selected = finite.groupby("half_life_days").rmse_pp.mean().idxmin()
    return selected, table


def static_prediction(train, evaluation, seed=mixed.SEEDS[0]):
    design, future = get_design(train, evaluation)
    target = train[TARGET].to_numpy()
    fitted_arx = np.linalg.lstsq(design, target, rcond=None)[0]
    predictions = evaluation[["event_id", "ticker", "cik", "reaction_day",
                              "prediction_end_date", "analysis_split", TARGET]].copy()
    predictions[MEAN] = target.mean()
    predictions[FROZEN] = future @ fitted_arx
    count = train.groupby("cik").size()
    predictions["training_events"] = evaluation.cik.map(count).fillna(0).astype(int)
    predictions["training_history"] = pd.cut(predictions.training_events, [-1, 10, 30, np.inf],
                                            labels=["0–10", "11–30", "31+"])
    diagnostics = []
    for name, nonlinear, use_mixed in [(MIXED, False, True), (NEURAL, True, False),
                                       (MIXED_NEURAL, True, True)]:
        fit = mixed.fit(design, target, train.cik.to_numpy(), nonlinear=nonlinear,
                        mixed=use_mixed, seed=seed)
        predictions[name] = fit.predict(future, evaluation.cik)
        if name == MIXED_NEURAL:
            predictions[SHARED_MIXED_NEURAL] = fit.shared_prediction(future)
        diagnostics.append({"model": name, **fit.diagnostics})
    return predictions, diagnostics, fitted_arx


def verify_baseline(predictions):
    reference = pd.read_csv(ROOT / "outputs/milestone2_timeseries/event_predictions.csv")
    matched = predictions.merge(reference[["event_id", "Training mean", "Event ARX quarter year"]],
                                on="event_id", suffixes=("", "_m2"), validate="one_to_one")
    if len(matched) != len(predictions):
        raise ValueError("Milestone 2 forecast rows do not match")
    max_diff = max((matched[MEAN] - 100 * matched[MEAN + "_m2"]).abs().max(),
                   (matched[FROZEN] - 100 * matched["Event ARX quarter year"]).abs().max())
    if max_diff > 1e-7:
        raise ValueError("The frozen forecasts disagree with Milestone 2")
    return max_diff


def cap_experiment(train, evaluation):
    """Thin well-covered issuers; retain all other issuers and fixed score rows."""
    eligible = train.groupby("cik").size().loc[lambda count: count > 30].index
    selected = evaluation.loc[evaluation.cik.isin(eligible) & evaluation.reaction_day.lt("2025")]
    results, fit_info, offset_checks = [], [], []
    ordered = train.loc[train.cik.isin(eligible)].sort_values(["prediction_end_date", "event_id"])
    indices = bootstrap_indices(ORIGIN, "2025-01-01", 28)
    for cap in (5, 10, 20, None):
        truncated = ordered.groupby("cik").tail(cap) if cap is not None else ordered
        capped = pd.concat([train.loc[~train.cik.isin(eligible)], truncated]).sort_index()
        predictions, diagnostics, _ = static_prediction(capped, selected)
        meta = {"per_issuer_cap": cap if cap is not None else "full", "train_n": len(capped),
                "eval_issuers": selected.cik.nunique()}
        results.extend({**meta, **row} for row in scores(predictions, STATIC_MODELS))
        fit_info.extend({**meta, **row} for row in diagnostics)
        offset_checks.append({"period": "2023–2024 capped cohort", **meta,
                              **offset_check(predictions)})
        for reference in (NEURAL, MEAN):
            offset_checks.append({"period": "2023–2024 capped cohort", **meta,
                                  **offset_check(predictions, reference, indices)})
    return results, fit_info, offset_checks


def offset_check(sample, reference=SHARED_MIXED_NEURAL, indices=None):
    row = {"reference": reference, **loss_terms(sample, MIXED_NEURAL, reference)}
    if indices is not None:
        row.update(paired_interval(sample, MIXED_NEURAL, reference, ORIGIN, indices))
    else:
        row["delta_rmse_pp"] = (metrics(sample[TARGET], sample[MIXED_NEURAL])["rmse_pp"]
                                - metrics(sample[TARGET], sample[reference])["rmse_pp"])
    return row


def split_results(predictions):
    main, yearly, history, intervals, decomposition, offsets = [], [], [], [], [], []
    for period, (start, end) in PERIODS.items():
        sample = predictions.loc[predictions.reaction_day.ge(start) & predictions.reaction_day.lt(end)]
        main.extend({"period": period, **row} for row in scores(sample, MAIN_MODELS))
        offsets.append({"period": period, **offset_check(sample)})
        for year, group in sample.groupby(sample.reaction_day.dt.year):
            yearly.extend({"year": year, **row} for row in scores(group, MAIN_MODELS))
        for bucket, group in sample.groupby("training_history", observed=True):
            history.extend({"period": period, "initial_training_events": bucket,
                            "issuers": group.cik.nunique(), **row} for row in scores(group, MAIN_MODELS))
        for block_days in (14, 28, 56):
            indices = bootstrap_indices(start, end, block_days)
            for model, reference in COMPARISONS:
                intervals.append({"period": period, "group": "all", "block_days": block_days,
                                  "model": model, "reference": reference,
                                  **paired_interval(sample, model, reference, start, indices)})
            if block_days == 28:
                short = sample.loc[sample.training_history.eq("0–10")]
                intervals.append({"period": period, "group": "0–10", "block_days": block_days,
                                  "model": MIXED_NEURAL, "reference": NEURAL,
                                  **paired_interval(short, MIXED_NEURAL, NEURAL, start, indices)})
        for model, reference in COMPARISONS:
            decomposition.append({"period": period, "model": model, "reference": reference,
                                  **loss_terms(sample, model, reference)})
    return {"comparison": main, "by_year": yearly, "by_history": history,
            "paired_intervals": intervals, "loss_decomposition": decomposition,
            "issuer_offset_ablation": offsets}


def issuer_persistence(train):
    early = train.loc[train.prediction_end_date.lt("2017-01-01")]
    later = train.loc[train.reaction_day.ge("2017-01-01")]
    design, future = get_design(early, later)
    beta = np.linalg.lstsq(design, early[TARGET], rcond=None)[0]
    joined = pd.DataFrame()
    for period, frame, array in [("2010_2016", early, design), ("2017_2022", later, future)]:
        error = frame.assign(residual=frame[TARGET] - array @ beta).groupby("cik").residual
        joined = pd.concat([joined, error.agg(["size", "mean"]).add_suffix("_" + period)], axis=1)
    return joined.dropna().reset_index()


def flatten_trace(rows, trace_type, model):
    flat = []
    for row in rows:
        record = {"model": model, **row}
        if trace_type == "update":
            before, after = record.pop("beta_before"), record.pop("beta_after")
            record.update({f"before_{key}": value for key, value in zip(COEFFICIENTS, before)})
            record.update({f"after_{key}": value for key, value in zip(COEFFICIENTS, after)})
        else:
            record.update(dict(zip(COEFFICIENTS, record.pop("beta"))))
        flat.append(record)
    return flat


def run(out):
    frame, total_rows = read_panel()
    train = frame.loc[frame.analysis_split.eq("train")]
    evaluation = frame.loc[frame.analysis_split.isin(["validation", "exploratory_test"])]
    stream = frame.loc[frame.prediction_end_date.ge(ORIGIN)].reset_index(drop=True)
    selected, tuning = tune_memory(train)
    predictions, diagnostics, beta = static_prediction(train, evaluation)
    diagnostics = [{"origin": 2023, **row} for row in diagnostics]
    baseline_error = verify_baseline(predictions)
    traces, updates = [], []
    for model, half_life in [(EXPANDING, np.inf), (TUNED, selected), (FAST, 30)]:
        forecasts, trace, journal = online_prediction(train, stream, ORIGIN, half_life)
        predictions[model] = pd.Series(forecasts, index=stream.event_id).loc[predictions.event_id].to_numpy()
        traces.extend(flatten_trace(trace, "forecast", model))
        updates.extend(flatten_trace(journal, "update", model))
    if not np.isfinite(predictions[list(MAIN_MODELS)]).all().all():
        raise ValueError("Comparison contains missing or invalid forecasts")

    annual = []
    for year in (2023, 2024, 2025):
        origin = f"{year}-01-01"
        available = frame.loc[frame.prediction_end_date.lt(origin)]
        scored = evaluation.loc[evaluation.reaction_day.dt.year.eq(year)]
        if year == 2023:
            refit = predictions.loc[predictions.reaction_day.dt.year.eq(year)]
        else:
            refit, fit_info, _ = static_prediction(available, scored)
            diagnostics.extend({"origin": year, **row} for row in fit_info)
        annual.extend({"year": year, "train_n": len(available), **row}
                      for row in scores(refit, STATIC_MODELS))

    seed_scores = []
    for seed in mixed.SEEDS:
        candidate = predictions
        if seed != mixed.SEEDS[0]:
            candidate, fit_info, _ = static_prediction(train, evaluation, seed=seed)
            diagnostics.extend({"origin": 2023, **row} for row in fit_info if row["seed"] is not None)
        for split, group in candidate.groupby("analysis_split"):
            seed_scores.extend({"seed": seed, "split": split, **row} for row in scores(group, (NEURAL, MIXED_NEURAL)))

    counts = train.groupby("cik").size()
    capped, cap_fits, cap_checks = cap_experiment(train, evaluation)
    tables = {**split_results(predictions), "memory_tuning": tuning, "forecasts": predictions,
              "annual_refit": annual, "seed_sensitivity": seed_scores,
              "fit_diagnostics": diagnostics, "history_caps": capped, "history_cap_fits": cap_fits,
              "issuer_residual_persistence": issuer_persistence(train),
              "coefficient_trace": traces, "label_updates": updates}
    tables["issuer_offset_ablation"].extend(cap_checks)
    diagnostics_count = len(set(evaluation.cik) - set(counts.index))
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        pd.DataFrame(table).to_csv(out / f"{name}.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d")
    spec = {"input": str(DATA.relative_to(ROOT)), "sha256": sha256(DATA.read_bytes()).hexdigest(),
            "initial_fit_rows": len(train), "evaluated_rows": len(evaluation), "issuer_count": frame.cik.nunique(),
            "eval_issuers_unseen_at_initial_fit": diagnostics_count,
            "excluded_incomplete_rows": total_rows - len(frame), "boundary_rows_unscored": len(frame) - len(train) - len(evaluation),
            "max_pp_difference_vs_milestone2_frozen_forecasts": baseline_error,
            "origin": ORIGIN, "features": EVENT_FEATURES, "selected_finite_memory_days": selected,
            "frozen_coefficients_pp_per_training_sd": dict(zip(COEFFICIENTS, beta.tolist())),
            "neural_hidden_width": mixed.WIDTH, "neural_l2": mixed.NEURAL_L2, "neural_seeds": mixed.SEEDS,
            "bootstrap_seed": 4031, "bootstrap_draws": 2000, "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl"]}}
    (out / "run_specification.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    from milestone3.figures import render
    render(out, selected)
    print(pd.DataFrame(tables["comparison"]).round(4).to_string(index=False))
    print(f"\nSelected finite memory: {selected:g} days. Wrote results to {out}")


def demo(out, date):
    trace = pd.read_csv(out / "coefficient_trace.csv")
    trace = trace.loc[trace.model.eq(TUNED)]
    if date is None:
        date = trace.loc[trace.labels_since_start.gt(0), "reaction_day"].iloc[0]
    print(trace.loc[trace.reaction_day.le(date), ["reaction_day", "latest_label_date",
          "labels_since_start", "intercept", "reaction_day_return"]].tail(3).to_string(index=False))
    if date not in trace.reaction_day.values:
        raise ValueError(f"No forecast at {date}. Use a reaction date from coefficient_trace.csv.")
    journal = pd.read_csv(out / "label_updates.csv")
    arrivals = journal.loc[journal.model.eq(TUNED) & journal.first_forecast_using_update.eq(date)]
    if len(arrivals):
        print("\nNewly available return labels and their effect on the reaction-day-return slope")
        print(arrivals[["completed_date", "example_event_id", "new_labels", "mean_new_return_pp",
                        "before_reaction_day_return",
                       "after_reaction_day_return"]].to_string(index=False))
    else:
        print("\nNo new labels for this forecast close.")
    forecasts = pd.read_csv(out / "forecasts.csv")
    current = forecasts.loc[forecasts.reaction_day.eq(date), ["event_id", FROZEN, TUNED]].head(5)
    if len(current):
        print("\nForecasts in return percentage points")
        print(current.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--demo", nargs="?", const="", metavar="YYYY-MM-DD",
                        help="Show recent updates and forecasts from a completed run")
    args = parser.parse_args()
    if args.demo is not None:
        demo(args.out, args.demo or None)
    else:
        with threadpool_limits(limits=1):
            run(args.out)


if __name__ == "__main__":
    main()
