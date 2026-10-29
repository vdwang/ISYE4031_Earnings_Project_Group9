"""Check completed-quarter/year predictors and availability-aware outcome lags."""

import unittest

import numpy as np
import pandas as pd

from milestone2_timeseries.features import company_daily, completed_event_lags, completed_periods, DAILY_FEATURES, split_labels


class TimeSeriesMilestoneTests(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range("2017-01-02", "2021-02-01")
        price = 100 * np.exp(np.arange(len(dates)) * .001)
        raw = pd.DataFrame({"ticker": "TEST", "date": dates, "close": price, "adjusted_close": price, "volume": 1000.})
        return raw, raw.assign(ticker="SPY")

    def test_calendar_features_use_completed_quarters_and_years(self):
        raw, market = self.fixture()
        out = company_daily(raw, market).set_index("date")
        row = out.loc["2020-05-01"]
        p = raw.set_index("date").adjusted_close
        q1 = p.loc[:"2020-03-31"].iloc[-1] / p.loc[:"2019-12-31"].iloc[-1] - 1
        q4 = p.loc[:"2019-06-30"].iloc[-1] / p.loc[:"2019-03-31"].iloc[-1] - 1
        year = p.loc[:"2019-12-31"].iloc[-1] / p.loc[:"2018-12-31"].iloc[-1] - 1
        self.assertAlmostEqual(row.quarter_return_lag1, q1)
        self.assertAlmostEqual(row.quarter_return_lag4, q4)
        self.assertAlmostEqual(row.year_return_lag1, year)
        self.assertEqual(row.quarter_return_lag1_source_date, pd.Timestamp("2020-03-31"))
        self.assertEqual(row.quarter_return_lag4_source_date, pd.Timestamp("2019-06-28"))
        self.assertTrue(pd.isna(out.loc["2017-05-01", "year_return_lag1"]))

    def test_future_prices_do_not_enter_calendar_or_daily_predictors(self):
        raw, market = self.fixture()
        first = company_daily(raw, market).set_index("date").loc["2020-05-01"]
        changed = raw.copy()
        changed.loc[changed.date.gt("2020-05-01"), "adjusted_close"] *= 3
        other = company_daily(changed, market).set_index("date").loc["2020-05-01"]
        np.testing.assert_allclose(first[DAILY_FEATURES].to_numpy(dtype=float), other[DAILY_FEATURES].to_numpy(dtype=float))
        self.assertNotEqual(first.future_return_5d, other.future_return_5d)

    def test_truncated_cache_does_not_make_an_unfinished_period_complete(self):
        raw, _ = self.fixture()
        price = raw.set_index("date").adjusted_close
        quarter = completed_periods(price, price, "Q")
        year = completed_periods(price, price, "Y")
        self.assertTrue(pd.isna(quarter.loc[pd.Period("2021Q1"), "return"]))
        self.assertTrue(pd.isna(year.loc[pd.Period("2021"), "return"]))
        self.assertTrue(pd.notna(quarter.loc[pd.Period("2020Q4"), "return"]))

    def test_event_lags_require_outcomes_to_have_finished(self):
        current = pd.DataFrame({"ticker": "TEST", "reaction_day": pd.to_datetime(["2020-01-12", "2020-01-20"])})
        history = pd.DataFrame({"ticker": "TEST", "reported_date": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-21"]),
            "prediction_end_date": pd.to_datetime(["2020-01-10", "2020-01-15", "2020-01-16", "2020-01-17", "2020-01-30"]),
            "future_5d_return": [.1, .2, .3, .4, 999]})
        out = completed_event_lags(current, history)
        self.assertAlmostEqual(out.loc[0, "completed_event_return_lag1"], .1)
        self.assertTrue(pd.isna(out.loc[0, "completed_event_return_lag4"]))
        self.assertAlmostEqual(out.loc[1, "completed_event_return_lag1"], .4)
        self.assertAlmostEqual(out.loc[1, "completed_event_return_lag4"], .1)

    def test_one_day_and_five_day_splits_purge_their_own_boundary_labels(self):
        dates = pd.Series(pd.to_datetime(["2022-12-29", "2023-01-03"]))
        one = pd.Series(pd.to_datetime(["2022-12-30", "2023-01-04"]))
        five = pd.Series(pd.to_datetime(["2023-01-06", "2023-01-10"]))
        self.assertEqual(split_labels(dates, one).tolist(), ["train", "validation"])
        self.assertEqual(split_labels(dates, five).tolist(), ["purged_boundary", "validation"])


if __name__ == "__main__":
    unittest.main()
