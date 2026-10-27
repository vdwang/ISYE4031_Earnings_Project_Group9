"""Check the new historical features against current/future information leakage."""

import unittest

import numpy as np
import pandas as pd

from research.expanded_features import earnings_history, lagged_context, enrich_events, MODEL_FEATURES
from src.build_dataset import build_events
from research.sp500_features import daily_features, add_earnings_history


class ExpandedFeatureTests(unittest.TestCase):
    def test_earnings_history_excludes_current_and_same_date_announcements(self):
        dates = ["2015-01-15", "2015-04-15", "2015-07-15", "2015-10-15", "2016-01-15", "2016-01-15"]
        frame = pd.DataFrame({"ticker": "TEST", "reportedDate": dates,
            "fiscalDateEnding": ["2014-12-31", "2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31", "2015-12-30"],
            "surprisePercentage": [1, 2, 3, 4, 1000, -1000]})
        out = earnings_history(frame)
        current = out.loc[out.reported_date.eq("2016-01-15")]
        np.testing.assert_allclose(current.prior_surprise_mean_4q, 2.5)
        np.testing.assert_allclose(current.prior_beat_rate_4q, 1)
        self.assertEqual(len(current), 2)
        self.assertTrue(out.loc[out.reported_date.eq("2015-10-15"), "prior_surprise_mean_4q"].isna().all())

    def test_macro_uses_conservative_lag_and_rejects_stale_values(self):
        events = pd.DataFrame({"reaction_day": pd.to_datetime(["2020-02-10", "2020-04-10"])})
        dates = pd.date_range("2020-01-01", "2020-02-10", freq="D")
        context = {name: pd.DataFrame({"date": dates, "value": np.arange(len(dates), dtype=float) + 1})
            for name in ["VIX", "VVIX", "DGS10", "DGS2"]}
        out = lagged_context(events, context)
        self.assertEqual(out.loc[0, "vix_source_date"], pd.Timestamp("2020-02-09"))
        self.assertEqual(out.loc[0, "dgs10_source_date"], pd.Timestamp("2020-02-03"))
        self.assertTrue(pd.isna(out.loc[1, "vix_prior_close"]))
        self.assertTrue(pd.isna(out.loc[1, "treasury_10y_lag7d"]))
        changed = {k: v.copy() for k, v in context.items()}
        for key, values in changed.items():
            cutoff = "2020-02-09" if key in ["VIX", "VVIX"] else "2020-02-03"
            values.loc[values.date.gt(cutoff), "value"] = 99999
        other = lagged_context(events, changed)
        for name in ["vix_prior_close", "vix_change_5_sessions", "vvix_prior_close", "treasury_10y_lag7d", "yield_curve_10y_minus_2y"]:
            self.assertAlmostEqual(out.loc[0, name], other.loc[0, name])

    def test_added_price_predictors_ignore_post_reaction_prices(self):
        dates = pd.bdate_range("2015-01-01", "2016-03-01")
        step = np.arange(len(dates), dtype=float)
        prices = {}
        for symbol, scale in [("TEST", 1), ("SPY", .5), ("XLK", .2)]:
            prices[symbol] = pd.DataFrame({"ticker": symbol, "date": dates,
                "close": 100 + scale * step, "adjusted_close": 100 + scale * step, "volume": 1000.})
        earnings = pd.DataFrame({"ticker": "TEST", "reportedDate": ["2015-01-15", "2015-04-15", "2015-07-15", "2015-10-15", "2016-01-15"],
            "fiscalDateEnding": ["2014-12-31", "2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31"],
            "reportedEPS": "1.1", "estimatedEPS": "1", "surprise": ".1", "surprisePercentage": "10"})
        universe = pd.DataFrame({"ticker": ["TEST"], "sector": ["Technology"], "sector_etf": ["XLK"], "cohort": ["test"]})
        context = {name: pd.DataFrame({"date": dates, "value": 20 + step / 100}) for name in ["VIX", "VVIX", "DGS10", "DGS2"]}
        first, _ = build_events(earnings, prices)
        original = enrich_events(first, earnings, prices, universe, context)
        changed = {k: v.copy() for k, v in prices.items()}
        for frame in changed.values():
            frame.loc[frame.date.gt(first.loc[0, "reaction_day"]), "adjusted_close"] *= 3
        second, _ = build_events(earnings, changed)
        other = enrich_events(second, earnings, changed, universe, context)
        np.testing.assert_allclose(original[MODEL_FEATURES], other[MODEL_FEATURES], equal_nan=True)
        self.assertNotEqual(original.loc[0, "future_5d_return"], other.loc[0, "future_5d_return"])
        # The scalable implementation must reproduce the independently calculated pilot features.
        vector = daily_features(prices["TEST"], prices["SPY"], prices["XLK"])
        row = vector.loc[vector.reaction_day.eq(first.loc[0, "reaction_day"])].iloc[0]
        common = [c for c in MODEL_FEATURES if c in vector]
        np.testing.assert_allclose(row[common].to_numpy(dtype=float), original.loc[0, common].to_numpy(dtype=float), atol=1e-9)
        future_changed = daily_features(changed["TEST"], changed["SPY"], changed["XLK"])
        row2 = future_changed.loc[future_changed.reaction_day.eq(first.loc[0, "reaction_day"])].iloc[0]
        np.testing.assert_allclose(row[common].to_numpy(dtype=float), row2[common].to_numpy(dtype=float), atol=1e-9)

    def test_bulk_history_preserves_missing_quarters_and_rejects_duplicate_announcements(self):
        frame = pd.DataFrame({"ticker": "TEST", "reported_date": pd.date_range("2020-01-01", periods=6, freq="QS"),
            "earnings_surprise_pct": [1, np.nan, 3, 4, 1000, -1000]})
        result = add_earnings_history(frame)
        self.assertTrue(pd.isna(result.loc[4, "prior_surprise_mean_4q"]))
        self.assertTrue(pd.isna(result.loc[5, "prior_surprise_mean_4q"]))
        self.assertEqual(result.loc[5, "previous_earnings_surprise_pct"], 1000)
        with self.assertRaises(ValueError):
            add_earnings_history(pd.concat([frame, frame.iloc[[0]]]))


if __name__ == "__main__":
    unittest.main()
