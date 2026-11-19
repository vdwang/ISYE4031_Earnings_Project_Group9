"""Delayed-label online least squares with a calendar-day memory."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OnlineARX:
    information: np.ndarray
    response: np.ndarray
    date: pd.Timestamp
    half_life: float
    beta: np.ndarray
    count: int = 0

    @classmethod
    def initialize(cls, design, outcome, start, half_life):
        """Start with the exact same least-squares coefficients as frozen AR-X."""
        if half_life <= 0 or np.isnan(half_life):
            raise ValueError("Half-life must be positive")
        return cls(design.T @ design, design.T @ outcome, pd.Timestamp(start),
                   half_life, np.linalg.lstsq(design, outcome, rcond=None)[0])

    def update(self, design, outcome, completed_date):
        """Consume one batch, with all labels ending on the same close."""
        date = pd.Timestamp(completed_date)
        if date < self.date:
            raise ValueError("Labels must arrive in chronological order")
        decay = 2 ** (-(date - self.date).days / self.half_life)
        self.information = decay * self.information + design.T @ design
        self.response = decay * self.response + design.T @ outcome
        self.beta = np.linalg.solve(self.information, self.response)
        self.date = date
        self.count += len(outcome)


def replay(model, stream, design, target):
    """Predict each close using only labels ending strictly before that close.

    The runner aligns arrays to the same integer index as the input frame.
    End-date batches also include eligible labels from the purged boundary row
    when it becomes available, even though that event is never scored.
    """
    if not stream.index.equals(pd.RangeIndex(len(stream))):
        raise ValueError("Replay inputs must share a consecutive integer index")
    predictions = np.full(len(stream), np.nan)
    forecast_log, update_log = [], []
    origin = model.date
    releases = iter(stream.groupby("prediction_end_date", sort=True).indices.items())
    release = next(releases, None)
    for date, rows in stream.groupby("reaction_day", sort=True).indices.items():
        if date < origin:
            # An event forecast before the origin can still mature after it.
            continue
        while release is not None and release[0] < date:
            end_date, arrived = release
            before = model.beta.copy()
            model.update(design[arrived], target[arrived], end_date)
            update_log.append({
                "completed_date": end_date, "first_forecast_using_update": date,
                "new_labels": len(arrived), "labels_since_start": model.count,
                "mean_new_return_pp": target[arrived].mean(),
                "max_input_reaction_day": stream.iloc[arrived].reaction_day.max(),
                "example_event_id": stream.iloc[arrived[0]].event_id,
                "beta_before": before, "beta_after": model.beta.copy(),
            })
            release = next(releases, None)
        predictions[rows] = design[rows] @ model.beta
        forecast_log.append({"reaction_day": date, "n_forecasts": len(rows),
                             "labels_since_start": model.count,
                             "latest_label_date": model.date if model.count else pd.NaT,
                             "beta": model.beta.copy()})
    return predictions, forecast_log, update_log
