"""Check that sensitivity transformations cannot learn from held-out outcomes."""

import unittest
import numpy as np
import pandas as pd
from src.sensitivity import fit_variant


class OutlierSensitivityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4031)
        self.features = ["earnings_surprise_pct", "previous_earnings_surprise_pct", "momentum"]
        self.train = pd.DataFrame(rng.normal(size=(40, 3)), columns=self.features)
        self.train["target"] = .02 * self.train["momentum"] + rng.normal(0, .005, 40)
        self.train.loc[0, "earnings_surprise_pct"] = 500
        self.sample = self.train.iloc[1:4].copy()
        self.sample.loc[self.sample.index[0], "earnings_surprise_pct"] = 1000

    def test_clipping_limits_use_training_only_and_predictions_ignore_test_outcomes(self):
        pred, details = fit_variant(self.train, self.sample, self.features, "target",
            "Clip surprise to training 1st and 99th pct")
        changed = self.sample.copy()
        changed["target"] = 999
        other, changed_details = fit_variant(self.train, changed, self.features, "target",
            "Clip surprise to training 1st and 99th pct")
        np.testing.assert_allclose(pred, other)
        np.testing.assert_allclose(details["clip_limits"]["earnings_surprise_pct"],
            self.train["earnings_surprise_pct"].quantile([.01, .99]).to_numpy())
        self.assertEqual(details, changed_details)
        self.assertEqual(len(pred), len(self.sample))

    def test_surprise_exclusion_keeps_extreme_test_events(self):
        pred, details = fit_variant(self.train, self.sample, self.features, "target",
            "Exclude surprise over 100 pct")
        self.assertEqual(details["n_train_used"], 39)
        self.assertEqual(details["n_train_excluded"], 1)
        self.assertEqual(len(pred), 3)
        self.assertTrue(np.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
