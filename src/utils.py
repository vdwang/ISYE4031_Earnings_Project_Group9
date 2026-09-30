"""Shared small data-validation helpers."""

import numpy as np
import pandas as pd


def numeric(series: pd.Series) -> pd.Series:
    """Convert API strings (including 'None') to numbers without filling gaps."""
    return pd.to_numeric(series.replace({"None": None, "": None}), errors="coerce").replace([np.inf, -np.inf], np.nan)


def valid_price_frame(frame: pd.DataFrame, ticker: str) -> bool:
    required = {"ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume"}
    if frame.empty or not required.issubset(frame.columns):
        return False
    dates = pd.to_datetime(frame["date"], errors="coerce")
    closes = numeric(frame["close"])
    return (
        frame["ticker"].eq(ticker).all()
        and dates.notna().all()
        and dates.is_unique
        and closes.notna().any()
        and (closes.dropna() > 0).all()
    )


def price_series(frame: pd.DataFrame) -> pd.Series:
    """Use adjusted closes when supplied; never mix price scales within a series."""
    adjusted = numeric(frame["adjusted_close"]) if "adjusted_close" in frame else pd.Series(index=frame.index, dtype=float)
    return adjusted if adjusted.notna().any() else numeric(frame["close"])
