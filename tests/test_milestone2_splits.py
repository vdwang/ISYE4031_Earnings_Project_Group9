"""Check that labels cannot cross the chronological evaluation boundaries."""

import unittest
import pandas as pd
from run_milestone2 import chronological_splits


class Milestone2SplitTests(unittest.TestCase):
    def test_boundary_crossing_labels_are_purged(self):
        data = pd.DataFrame({
            "reaction_day": pd.to_datetime(["2022-12-20", "2022-12-29", "2024-12-20", "2024-12-30", "2025-12-30"]),
            "prediction_end_date": pd.to_datetime(["2022-12-28", "2023-01-06", "2024-12-30", "2025-01-07", "2026-01-07"]),
        })
        splits = chronological_splits(data)
        self.assertEqual(splits["train"].index.tolist(), [0])
        self.assertEqual(splits["validation"].index.tolist(), [2])
        self.assertTrue(splits["test"].empty)

    def test_same_prediction_date_stays_in_one_period(self):
        data = pd.DataFrame({
            "ticker": ["A", "B", "A", "B"],
            "reaction_day": pd.to_datetime(["2023-01-03", "2023-01-03", "2025-01-02", "2025-01-02"]),
            "prediction_end_date": pd.to_datetime(["2023-01-10", "2023-01-10", "2025-01-09", "2025-01-09"]),
        })
        splits = chronological_splits(data)
        self.assertTrue(splits["train"].empty)
        self.assertEqual(splits["validation"].index.tolist(), [0, 1])
        self.assertEqual(splits["test"].index.tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
