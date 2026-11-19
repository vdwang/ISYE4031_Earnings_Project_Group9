"""Comparable forecast metrics and diagnostics for a dated earnings panel."""

import numpy as np
import pandas as pd

TARGET = "actual_pp"
MEAN = "Training mean"
FROZEN = "Frozen AR-X"
EXPANDING = "Online AR-X, all history"
TUNED = "Online AR-X, best finite memory"
FAST = "Online AR-X, 30-day memory"
MIXED = "Mixed AR-X"
NEURAL = "Pooled neural net"
MIXED_NEURAL = "Mixed neural net"

COMPARISONS = (
    (FROZEN, MEAN), (EXPANDING, FROZEN), (TUNED, FROZEN), (FAST, FROZEN),
    (MIXED, FROZEN), (NEURAL, FROZEN), (MIXED_NEURAL, NEURAL), (MIXED_NEURAL, MEAN),
)


def metrics(actual, forecast):
    error = np.asarray(actual) - np.asarray(forecast)
    return {"n": len(error), "mae_pp": np.abs(error).mean(),
            "rmse_pp": np.sqrt(np.mean(error ** 2)),
            "forecast_sd_pp": np.std(forecast)}


def scores(frame, models):
    return [{"model": model, **metrics(frame[TARGET], frame[model])} for model in models]


def loss_terms(frame, model, reference):
    """Exact squared-error identity with the reference forecast as the origin."""
    shift = frame[model] - frame[reference]
    reference_error = frame[TARGET] - frame[reference]
    return {"n": len(frame), "rms_forecast_change_pp": np.sqrt((shift ** 2).mean()),
            "shift_size_cost_pp2": (shift ** 2).mean(),
            "alignment_benefit_pp2": (2 * shift * reference_error).mean(),
            "delta_mse_pp2": (shift ** 2 - 2 * shift * reference_error).mean()}


def bootstrap_indices(start, end, block_days, draws=2000, seed=4031):
    """Circular moving blocks of calendar days, including days without events."""
    n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_days / block_days))
    anchors = rng.integers(0, n_days, size=(draws, n_blocks, 1))
    offsets = np.arange(block_days)
    return ((anchors + offsets) % n_days).reshape(draws, -1)[:, :n_days]


def paired_interval(frame, model, reference, start, indices):
    work = pd.DataFrame({
        "offset": (frame.reaction_day - pd.Timestamp(start)).dt.days,
        "a": (frame[TARGET] - frame[model]) ** 2,
        "b": (frame[TARGET] - frame[reference]) ** 2,
    })
    daily = work.groupby("offset").agg(n=("a", "size"), a=("a", "sum"), b=("b", "sum"))
    daily = daily.reindex(range(indices.shape[1]), fill_value=0)
    totals = daily.to_numpy()[indices].sum(axis=1)
    nonempty = totals[:, 0] > 0
    bootstrap_rmse = np.sqrt(totals[nonempty, 1:] / totals[nonempty, :1])
    low, high = np.quantile(bootstrap_rmse[:, 0] - bootstrap_rmse[:, 1], [.025, .975])
    point = metrics(frame[TARGET], frame[model])["rmse_pp"] - metrics(
        frame[TARGET], frame[reference])["rmse_pp"]
    return {"n": len(frame), "delta_rmse_pp": point, "lower_95_pp": low,
            "upper_95_pp": high, "usable_bootstrap_draws": nonempty.sum()}
