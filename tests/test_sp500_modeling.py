"""Check that tuning and clipping cannot learn from later evaluation data."""

import unittest

import numpy as np
import pandas as pd

from research.modeling import TrainingQuantileClipper, rolling_folds


class SP500ModelingTests(unittest.TestCase):
    def test_inner_folds_purge_crossing_labels_and_reject_outer_holdouts(self):
        data = pd.DataFrame({"reaction_day": pd.to_datetime(["2016-12-20", "2016-12-30", "2017-01-03", "2018-12-28", "2022-12-20"]),
            "prediction_end_date": pd.to_datetime(["2016-12-28", "2017-01-09", "2017-01-10", "2019-01-07", "2022-12-28"])})
        name, train, validation = list(rolling_folds(data))[0]
        self.assertEqual(name, "2017_2018")
        self.assertEqual(train.index.tolist(), [0])
        self.assertEqual(validation.index.tolist(), [2])
        invalid = pd.DataFrame({"reaction_day": pd.to_datetime(["2023-01-03"]), "prediction_end_date": pd.to_datetime(["2023-01-10"])})
        with self.assertRaises(ValueError):
            list(rolling_folds(invalid))

    def test_clipping_keeps_rows_and_uses_training_limits_only(self):
        training = np.arange(100, dtype=float).reshape(-1, 1)
        sample = np.array([[-1e8], [50.], [1e8]])
        clipper = TrainingQuantileClipper().fit(training)
        before = clipper.upper_bounds_.copy()
        result = clipper.transform(sample)
        clipper.transform(np.array([[1e12]]))
        np.testing.assert_allclose(clipper.upper_bounds_, before)
        np.testing.assert_allclose(result[:, 0], [.99, 50, 98.01])
        self.assertEqual(result.shape, sample.shape)


if __name__ == "__main__":
    unittest.main()
