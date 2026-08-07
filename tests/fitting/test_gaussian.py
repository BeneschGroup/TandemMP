"""Tests for tandemmp.fitting.gaussian — target API."""

import numpy as np
import pytest

from tandemmp.fitting.gaussian import (
    fit_gaussian,
    gaussian,
    gaussian_fit_histogram,
)

# ---------------------------------------------------------------------------
# gaussian
# ---------------------------------------------------------------------------


class TestGaussian:
    def test_peak_at_center(self):
        x = np.linspace(-5, 5, 101)
        y = gaussian(x, a0=1.0, x0=0.0, sigma=1.0)
        assert y[50] == pytest.approx(1.0)

    def test_symmetry(self):
        x = np.array([-1.0, 1.0])
        y = gaussian(x, a0=1.0, x0=0.0, sigma=1.0)
        assert y[0] == pytest.approx(y[1])

    def test_amplitude_scaling(self):
        x = np.array([0.0])
        assert gaussian(x, a0=3.0, x0=0.0, sigma=1.0) == pytest.approx(3.0)

    def test_negative_amplitude_uses_abs(self):
        x = np.array([0.0])
        assert gaussian(x, a0=-2.0, x0=0.0, sigma=1.0) == pytest.approx(2.0)

    def test_zero_far_from_center(self):
        x = np.array([100.0])
        y = gaussian(x, a0=1.0, x0=0.0, sigma=1.0)
        assert y[0] == pytest.approx(0.0, abs=1e-10)

    def test_output_shape(self):
        x = np.linspace(-3, 3, 50)
        y = gaussian(x, a0=1.0, x0=0.0, sigma=1.0)
        assert y.shape == (50,)


# ---------------------------------------------------------------------------
# fit_gaussian
# ---------------------------------------------------------------------------


class TestFitGaussian:
    def test_recovers_known_params(self):
        x = np.linspace(-10, 10, 201)
        y = gaussian(x, a0=5.0, x0=1.0, sigma=2.0)
        popt = fit_gaussian(x, y)
        assert popt[0] == pytest.approx(5.0, rel=0.01)
        assert popt[1] == pytest.approx(1.0, abs=0.1)
        assert abs(popt[2]) == pytest.approx(2.0, rel=0.01)

    def test_noisy_data(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-10, 10, 201)
        y = gaussian(x, a0=5.0, x0=0.0, sigma=2.0) + rng.normal(0, 0.1, 201)
        popt = fit_gaussian(x, y)
        assert popt[0] == pytest.approx(5.0, rel=0.1)
        assert popt[1] == pytest.approx(0.0, abs=0.5)

    def test_returns_three_params(self):
        x = np.linspace(-5, 5, 101)
        y = gaussian(x, a0=1.0, x0=0.0, sigma=1.0)
        popt = fit_gaussian(x, y)
        assert len(popt) == 3


# ---------------------------------------------------------------------------
# gaussian_fit_histogram
# ---------------------------------------------------------------------------


class TestGaussianFitHistogram:
    def test_recovers_distribution(self):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=100.0, scale=10.0, size=10_000)
        a, x0, sigma = gaussian_fit_histogram(data, bins=50)
        assert x0 == pytest.approx(100.0, abs=2.0)
        assert abs(sigma) == pytest.approx(10.0, rel=0.2)

    def test_returns_three_values(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=0.0, scale=1.0, size=1000)
        result = gaussian_fit_histogram(data)
        assert len(result) == 3
