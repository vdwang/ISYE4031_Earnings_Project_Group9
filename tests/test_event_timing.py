"""A small synthetic fixture checks alignment; it is never saved as project data."""

import unittest

import numpy as np
import pandas as pd

from src.build_dataset import build_events


class EventTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.bdate_range("2015-11-02", "2016-03-01")
        indexes = np.arange(len(dates), dtype=float)
        self.stock = pd.DataFrame({
            "ticker": "TEST", "date": dates.strftime("%Y-%m-%d"),
            "close": 100 + indexes, "adjusted_close": 100 + indexes,
            "volume": np.full(len(dates), 1000.0),
        })
        self.spy = pd.DataFrame({
            "ticker": "SPY", "date": dates.strftime("%Y-%m-%d"),
            "close": 200 + indexes / 2, "adjusted_close": 200 + indexes / 2,
            "volume": np.full(len(dates), 5000.0),
        })
        self.earnings = pd.DataFrame([
            {"ticker": "TEST", "fiscalDateEnding": "2015-09-30", "reportedDate": "2015-12-31", "reportedEPS": "1", "estimatedEPS": "0.9", "surprise": "0.1", "surprisePercentage": "8"},
            {"ticker": "TEST", "fiscalDateEnding": "2015-12-31", "reportedDate": "2016-01-22", "reportedEPS": "2", "estimatedEPS": "1.8", "surprise": "0.2", "surprisePercentage": "10"},
        ])

    def test_strictly_later_reaction_and_five_session_target(self) -> None:
        events, counts = build_events(self.earnings, {"TEST": self.stock, "SPY": self.spy})
        self.assertEqual(counts["events_2016_2025"], 1)
        event = events.iloc[0]
        self.assertEqual(event["reaction_day"], pd.Timestamp("2016-01-25"))
        i = self.stock.index[self.stock["date"].eq("2016-01-25")][0]
        self.assertEqual(event["prediction_start_date"], pd.Timestamp(self.stock.loc[i + 1, "date"]))
        self.assertEqual(event["prediction_end_date"], pd.Timestamp(self.stock.loc[i + 5, "date"]))
        self.assertAlmostEqual(event["future_5d_return"], self.stock.loc[i + 5, "close"] / self.stock.loc[i, "close"] - 1)
        self.assertAlmostEqual(event["return_5d_before"], self.stock.loc[i - 1, "close"] / self.stock.loc[i - 6, "close"] - 1)
        self.assertAlmostEqual(event["abnormal_volume"], 1.0)
        self.assertEqual(event["previous_earnings_surprise_pct"], 8)

    def test_future_prices_do_not_change_predictors(self) -> None:
        first, _ = build_events(self.earnings, {"TEST": self.stock, "SPY": self.spy})
        changed = self.stock.copy()
        changed.loc[changed["date"] > "2016-01-25", "adjusted_close"] *= 3
        second, _ = build_events(self.earnings, {"TEST": changed, "SPY": self.spy})
        predictors = ["reaction_day_return", "abnormal_volume", "return_5d_before", "return_20d_before", "volatility_20d"]
        for name in predictors:
            self.assertAlmostEqual(first.loc[0, name], second.loc[0, name])
        self.assertNotEqual(first.loc[0, "future_5d_return"], second.loc[0, "future_5d_return"])


if __name__ == "__main__":
    unittest.main()
