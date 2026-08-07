"""Tests for tandemmp.fitting.minimize — target API."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tandemmp.fitting.minimize import (
    CurveFitResult,
    FitResult,
    curve_fit_model,
    minimize_model,
)

# ---------------------------------------------------------------------------
# minimize_model
# ---------------------------------------------------------------------------


def _quadratic(v: np.ndarray) -> float:
    """Minimised at x=3, y=-2, value 0."""
    x, y = v
    return float((x - 3.0) ** 2 + (y + 2.0) ** 2)


class TestMinimizeModel:
    def test_recovers_minimum(self):
        result = minimize_model(_quadratic, x0=[0.0, 0.0])
        assert result.params[0] == pytest.approx(3.0, abs=1e-4)
        assert result.params[1] == pytest.approx(-2.0, abs=1e-4)

    def test_fun_near_zero_at_minimum(self):
        result = minimize_model(_quadratic, x0=[0.0, 0.0])
        assert result.fun == pytest.approx(0.0, abs=1e-6)

    def test_success_true(self):
        result = minimize_model(_quadratic, x0=[0.0, 0.0])
        assert result.success is True

    def test_returns_fit_result(self):
        result = minimize_model(_quadratic, x0=[0.0, 0.0])
        assert isinstance(result, FitResult)

    def test_raw_is_optimize_result(self):
        result = minimize_model(_quadratic, x0=[0.0, 0.0])
        assert isinstance(result.raw, OptimizeResult)

    def test_method_recorded(self):
        result = minimize_model(_quadratic, x0=[0.0, 0.0], method="Nelder-Mead")
        assert result.method == "Nelder-Mead"

    def test_bounds_respected(self):
        # Unconstrained minimum is at x=3; clamp it below 1.
        result = minimize_model(
            _quadratic,
            x0=[0.0, 0.0],
            method="Powell",
            bounds=[(None, 1.0), (None, None)],
        )
        assert result.params[0] <= 1.0 + 1e-6
        assert result.params[1] == pytest.approx(-2.0, abs=1e-3)

    def test_success_false_surfaces_through_wrapper(self):
        # Starve the optimiser of iterations/evaluations so it can't converge.
        result = minimize_model(
            _quadratic,
            x0=[0.0, 0.0],
            method="Nelder-Mead",
            options={"maxiter": 0, "maxfev": 0},
        )
        assert result.success is False
        assert result.raw.success is False


# ---------------------------------------------------------------------------
# curve_fit_model
# ---------------------------------------------------------------------------


def _line(x: np.ndarray, m: float, c: float) -> np.ndarray:
    return m * x + c


def _gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - mean) ** 2) / (2.0 * sigma**2))


class TestCurveFitModel:
    def test_recovers_linear_params(self):
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 5.0
        result = curve_fit_model(_line, x, y, p0=[1.0, 1.0])
        assert result.params[0] == pytest.approx(2.0, abs=1e-6)
        assert result.params[1] == pytest.approx(5.0, abs=1e-6)

    def test_perfect_fit_r_squared_is_one(self):
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 5.0
        result = curve_fit_model(_line, x, y, p0=[1.0, 1.0])
        assert result.r_squared == pytest.approx(1.0, abs=1e-8)

    def test_perfect_fit_residual_sum_near_zero(self):
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 5.0
        result = curve_fit_model(_line, x, y, p0=[1.0, 1.0])
        assert result.residual_sum_of_squares == pytest.approx(0.0, abs=1e-12)

    def test_covariance_shape(self):
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 5.0
        result = curve_fit_model(_line, x, y, p0=[1.0, 1.0])
        assert result.covariance.shape == (2, 2)

    def test_param_errors_shape(self):
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 5.0
        result = curve_fit_model(_line, x, y, p0=[1.0, 1.0])
        assert result.param_errors.shape == (2,)

    def test_returns_curve_fit_result(self):
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 5.0
        result = curve_fit_model(_line, x, y, p0=[1.0, 1.0])
        assert isinstance(result, CurveFitResult)

    def test_recovers_gaussian_params(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-10, 10, 201)
        y = _gaussian(x, amplitude=5.0, mean=1.0, sigma=2.0)
        y_noisy = y + rng.normal(0, 0.01, x.shape)
        result = curve_fit_model(_gaussian, x, y_noisy, p0=[4.0, 0.0, 1.5])
        assert result.params[0] == pytest.approx(5.0, rel=0.05)
        assert result.params[1] == pytest.approx(1.0, abs=0.1)
        assert result.params[2] == pytest.approx(2.0, rel=0.05)

    def test_r_squared_high_for_good_gaussian_fit(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-10, 10, 201)
        y = _gaussian(x, amplitude=5.0, mean=1.0, sigma=2.0)
        y_noisy = y + rng.normal(0, 0.01, x.shape)
        result = curve_fit_model(_gaussian, x, y_noisy, p0=[4.0, 0.0, 1.5])
        assert result.r_squared > 0.99

    def test_weighted_fit_with_sigma_and_absolute_sigma(self):
        # Known dataset with per-point uncertainties.
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2x + 1
        sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result = curve_fit_model(
            _line, x, y, p0=[1.0, 1.0], sigma=sigma, absolute_sigma=True
        )
        assert result.params[0] == pytest.approx(2.0, abs=1e-6)
        assert result.params[1] == pytest.approx(1.0, abs=1e-6)
        # Uniform sigma of 1.0 with absolute_sigma=True on an exact linear
        # relationship still yields a well-defined covariance matrix.
        assert result.covariance.shape == (2, 2)
        assert np.all(np.isfinite(result.param_errors))

    def test_unweighted_vs_weighted_params_match_for_uniform_sigma(self):
        x = np.linspace(0, 10, 20)
        y = 3.0 * x - 1.0
        sigma = np.full_like(x, 2.0)

        unweighted = curve_fit_model(_line, x, y, p0=[1.0, 1.0])
        weighted = curve_fit_model(
            _line, x, y, p0=[1.0, 1.0], sigma=sigma, absolute_sigma=True
        )
        assert unweighted.params[0] == pytest.approx(weighted.params[0], abs=1e-8)
        assert unweighted.params[1] == pytest.approx(weighted.params[1], abs=1e-8)

    def test_r_squared_zero_for_constant_y_data(self):
        # y is constant, so ss_tot == 0; curve_fit_model must return the
        # defined 0.0 rather than propagating NumPy's 0/0 -> nan.
        x = np.linspace(0, 10, 20)
        y = np.full_like(x, 7.0)
        result = curve_fit_model(_line, x, y, p0=[0.0, 1.0])
        assert result.r_squared == 0.0

    def test_non_convergence_propagates_from_scipy(self, monkeypatch):
        # curve_fit_model must not swallow scipy.optimize.curve_fit's
        # RuntimeError on non-convergence (e.g. maxfev exceeded).
        def _raise_non_convergence(*args, **kwargs):
            msg = "Optimal parameters not found: Number of calls to function has reached maxfev"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "tandemmp.fitting.minimize.scipy.optimize.curve_fit",
            _raise_non_convergence,
        )
        x = np.linspace(0, 10, 20)
        y = 2.0 * x + 5.0
        with pytest.raises(RuntimeError, match="maxfev"):
            curve_fit_model(_line, x, y, p0=[1.0, 1.0])
