"""Regression tests for delayed labels and the research models' numerical identities."""

import unittest

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from milestone3.evaluation import loss_terms
from milestone3.mixed import gaussian_loss
from milestone3.online import OnlineARX, replay


class TestDelayedOnlineUpdate(unittest.TestCase):
    def test_update_matches_direct_calendar_weighted_fit(self):
        design = np.array([[1., -1.], [1., 0.], [1., 1.]])
        targets = np.array([1., 0., 3.])
        new = np.array([[1., 3.], [1., 3.]])
        outcomes = np.array([8., 2.])
        for half_life, weights in [(10, [.25] * 3 + [.5, 1.]), (np.inf, [1.] * 5)]:
            model = OnlineARX.initialize(design, targets, "2023-01-01", half_life)
            for idx, date in enumerate(["2023-01-11", "2023-01-21"]):
                model.update(new[idx:idx + 1], outcomes[idx:idx + 1], date)
            scale = np.sqrt(weights)
            expected = np.linalg.lstsq(np.vstack([design, new]) * scale[:, None],
                                       np.r_[targets, outcomes] * scale, rcond=None)[0]
            np.testing.assert_allclose(model.beta, expected, rtol=1e-10)

    def test_boundary_label_arrives_after_same_close(self):
        frame = pd.DataFrame({
            "event_id": ["boundary", "A", "B", "C"],
            "reaction_day": pd.to_datetime(["2022-12-27", "2023-01-04", "2023-01-05", "2023-01-05"]),
            "prediction_end_date": pd.to_datetime(["2023-01-04", "2023-01-11", "2023-01-12", "2023-01-12"]),
        })
        x = np.ones((4, 1))

        def run(target):
            model = OnlineARX.initialize(np.ones((3, 1)), np.zeros(3), "2023-01-01", np.inf)
            return replay(model, frame, x, np.array(target))

        predictions, trace, updates = run([4., 9., 90., 900.])
        edited, _, _ = run([4., -9., -90., -900.])
        np.testing.assert_allclose(predictions, [np.nan, 0., 1., 1.], equal_nan=True)
        np.testing.assert_allclose(edited, predictions, equal_nan=True)
        self.assertEqual(trace[1]["n_forecasts"], 2)
        self.assertEqual([row["example_event_id"] for row in updates], ["boundary"])
        self.assertLess(updates[0]["completed_date"], updates[0]["first_forecast_using_update"])


class TestModelIdentities(unittest.TestCase):
    def test_mixed_loss_matches_dense_gaussian_and_gradient(self):
        group = np.array([0, 0, 1, 2, 2, 2])
        counts = np.bincount(group)
        residual = np.array([0.5, -1., 1., 2., 3., -0.5])
        params = np.r_[residual, 0.7, -1.6]
        variance, ratio = np.exp(params[-2:])
        covariance = variance * (np.eye(len(group)) + ratio * (group[:, None] == group))
        dense_loss = (np.linalg.slogdet(covariance)[1] + residual @ np.linalg.solve(covariance, residual)) / 12
        loss, residual_grad, variance_grad, ratio_grad, offset = gaussian_loss(
            residual, group, counts, *params[-2:])
        self.assertAlmostEqual(loss, dense_loss)
        self.assertAlmostEqual(offset[1], ratio / (1 + ratio))
        numerical = approx_fprime(params, lambda p: gaussian_loss(p[:-2], group, counts, *p[-2:])[0])
        np.testing.assert_allclose(np.r_[residual_grad, variance_grad, ratio_grad], numerical, atol=2e-7)

    def test_loss_decomposition_reconciles_squared_error(self):
        frame = pd.DataFrame({"actual_pp": [1., 2., -1.], "new": [1., 0., 1.],
                              "reference": [2., 1., 1.]})
        delta = (frame.actual_pp - frame.new).pow(2) - (frame.actual_pp - frame.reference).pow(2)
        self.assertAlmostEqual(loss_terms(frame, "new", "reference")["delta_mse_pp2"], delta.mean())


if __name__ == "__main__":
    unittest.main()
