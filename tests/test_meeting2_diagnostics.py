"""Check units and holdout separation in the descriptive diagnostic summaries."""

import unittest
import numpy as np
import pandas as pd
from src.diagnostics import grouped_performance


class Meeting2DiagnosticsTests(unittest.TestCase):
    def test_training_errors_cannot_enter_holdout_metrics(self):
        predictions = pd.DataFrame({
            "target": ["raw", "raw", "raw"], "split": ["train", "validation", "validation"],
            "year": [2022, 2023, 2023], "observed": [999, .03, -.01],
            "ols_prediction": [0, .01, .01], "training_mean_prediction": [.01, .01, .01],
        })
        result = grouped_performance(predictions, ["year"])
        ols = result.loc[result["model"].eq("OLS")].iloc[0]
        self.assertEqual(ols["n"], 2)
        self.assertAlmostEqual(ols["mae_pp"], 2)
        self.assertAlmostEqual(ols["rmse_pp"], 2)
        self.assertAlmostEqual(ols["mean_error_pp"], 0)
        self.assertAlmostEqual(ols["direction_accuracy"], .5)
        self.assertEqual(set(result["split"]), {"validation"})

    def test_targets_are_scored_separately_and_zero_is_nonpositive(self):
        predictions = pd.DataFrame({
            "target": ["raw", "adjusted"], "split": ["test", "test"], "year": [2025, 2025],
            "observed": [.01, -.02], "ols_prediction": [0, 0], "training_mean_prediction": [0, 0],
        })
        result = grouped_performance(predictions, ["year"])
        zero = result.loc[result["model"].eq("Zero")].set_index("target")
        self.assertAlmostEqual(zero.loc["raw", "mae_pp"], 1)
        self.assertAlmostEqual(zero.loc["adjusted", "mae_pp"], 2)
        self.assertEqual(zero.loc["raw", "direction_accuracy"], 0)
        self.assertEqual(zero.loc["adjusted", "direction_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
