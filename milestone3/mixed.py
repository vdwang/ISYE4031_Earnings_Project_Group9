"""Full-batch Gaussian random-intercept regressions for earnings events."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

WIDTH = 8
NEURAL_L2 = 0.001
SEEDS = (17, 29, 53)


def gaussian_loss(residual, group, counts, log_variance, log_ratio):
    """Per-row Gaussian NLL and derivatives, using one covariance block per issuer."""
    variance = np.exp(log_variance)
    ratio = np.exp(log_ratio)  # issuer variance / residual variance
    sums = np.bincount(group, weights=residual, minlength=len(counts))
    denominator = 1 + counts * ratio
    adjustment = ratio * sums / denominator
    inverse_residual = (residual - adjustment[group]) / variance
    quadratic = residual @ inverse_residual
    n = len(residual)
    loss = (quadratic + n * log_variance + np.log(denominator).sum()) / (2 * n)
    variance_gradient = (n - quadratic) / (2 * n)
    ratio_gradient = ratio * np.sum(counts / denominator
                                     - (sums / denominator) ** 2 / variance) / (2 * n)
    return loss, inverse_residual / n, variance_gradient, ratio_gradient, adjustment


def mean_and_gradient(design, parameters, nonlinear, derivative=None):
    if not nonlinear:
        mean = design @ parameters
        return mean, None if derivative is None else design.T @ derivative
    features = design[:, 1:]
    n_inputs = features.shape[1]
    weight = parameters[:n_inputs * WIDTH].reshape(n_inputs, WIDTH)
    hidden_bias = parameters[n_inputs * WIDTH:-WIDTH - 1]
    output_weight = parameters[-WIDTH - 1:-1]
    hidden = np.tanh(features @ weight + hidden_bias)
    mean = parameters[-1] + hidden @ output_weight
    if derivative is None:
        return mean, None
    backprop = derivative[:, None] * output_weight * (1 - hidden ** 2)
    gradient = np.r_[np.ravel(features.T @ backprop), backprop.sum(axis=0),
                     hidden.T @ derivative, derivative.sum()]
    return mean, gradient


@dataclass
class FittedModel:
    parameters: np.ndarray
    offsets: dict
    nonlinear: bool
    diagnostics: dict

    def shared_prediction(self, design):
        mean, _ = mean_and_gradient(design, self.parameters, self.nonlinear)
        return mean

    def predict(self, design, issuers):
        return self.shared_prediction(design) + np.array([self.offsets.get(issuer, 0.0) for issuer in issuers])


def fit(design, outcome, issuers, *, nonlinear=False, mixed=False, seed=SEEDS[0]):
    """Fit the shared mean jointly with covariance parameters by maximum likelihood.

    Neural fits add a small fixed weight penalty. Runs with ratio zero supply
    the pooled neural control. Unknown issuer IDs fall back to a zero offset.
    """
    ids, group, counts = np.unique(issuers, return_inverse=True, return_counts=True)
    if nonlinear:
        rng = np.random.default_rng(seed)
        input_weights = rng.normal(0, 0.25, (design.shape[1] - 1) * WIDTH)
        parameters = np.r_[input_weights, np.zeros(WIDTH), rng.normal(0, 0.1, WIDTH),
                           outcome.mean()]
        penalty_mask = np.ones(len(parameters))
        penalty_mask[len(input_weights):len(input_weights) + WIDTH] = 0
        penalty_mask[-1] = 0
    else:
        parameters = np.linalg.lstsq(design, outcome, rcond=None)[0]
        penalty_mask = np.zeros(len(parameters))

    def objective(theta):
        weights = theta[:len(parameters)]
        log_variance = theta[len(parameters)]
        log_ratio = theta[-1] if mixed else -np.inf
        mean, _ = mean_and_gradient(design, weights, nonlinear)
        loss, dr, dv, dt, _ = gaussian_loss(outcome - mean, group, counts, log_variance, log_ratio)
        _, gradient = mean_and_gradient(design, weights, nonlinear, -dr)
        penalized = weights * penalty_mask
        return (loss + NEURAL_L2 / 2 * (penalized @ penalized),
                np.r_[gradient + NEURAL_L2 * penalized, dv, [dt] if mixed else []])

    initial = np.r_[parameters, np.log(outcome.var()), [-5.] if mixed else []]
    result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                      bounds=[(None, None)] * len(parameters) + [(-8, 8)]
                      + ([(-16, 4)] if mixed else []),
                      options={"maxiter": 1600, "ftol": 1e-10, "gtol": 1e-6})
    if not result.success:
        raise RuntimeError(f"Model fit failed (nonlinear={nonlinear}, mixed={mixed}): {result.message}")
    fitted = result.x[:len(parameters)]
    log_variance = result.x[len(parameters)]
    log_ratio = result.x[-1] if mixed else -np.inf
    mean, _ = mean_and_gradient(design, fitted, nonlinear)
    _, _, _, _, adjustment = gaussian_loss(outcome - mean, group, counts, log_variance, log_ratio)
    return FittedModel(fitted, dict(zip(ids, adjustment)) if mixed else {}, nonlinear,
                       {"seed": seed if nonlinear else None, "iterations": result.nit,
                        "residual_sd_pp": np.exp(log_variance / 2),
                        "issuer_sd_pp": np.exp((log_variance + log_ratio) / 2),
                        "variance_ratio": np.exp(log_ratio), "objective": result.fun})
