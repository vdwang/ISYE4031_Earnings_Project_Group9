"""Training-only preprocessing and purged rolling-origin folds for this study."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TrainingQuantileClipper(TransformerMixin, BaseEstimator):
    """Learn feature limits from fit data; retain every evaluation row/outcome."""

    def __init__(self, lower=.01, upper=.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Invalid clipping quantiles")
        self.n_features_in_ = values.shape[1]
        self.lower_bounds_ = np.quantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.quantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed")
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def rolling_folds(train):
    """Use only pre-2023 rows and purge targets crossing each fold boundary."""
    if train.reaction_day.ge("2023-01-01").any() or train.prediction_end_date.ge("2023-01-01").any():
        raise ValueError("Rolling tuning must receive training-period rows only")
    for start, end in [("2017-01-01", "2019-01-01"), ("2019-01-01", "2021-01-01"), ("2021-01-01", "2023-01-01")]:
        past = train.loc[train.reaction_day.lt(start) & train.prediction_end_date.lt(start)]
        later = train.loc[train.reaction_day.ge(start) & train.reaction_day.lt(end) & train.prediction_end_date.lt(end)]
        if len(past) and len(later):
            yield start[:4] + "_" + str(int(end[:4]) - 1), past.copy(), later.copy()
