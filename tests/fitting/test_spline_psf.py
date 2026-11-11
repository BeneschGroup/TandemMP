"""Tests for :mod:`tandemmp.fitting.spline_psf` and its Rust extension."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BivariateSpline

from tandemmp.fitting.psf_interface import _initial_position_guess, square_mesh
from tandemmp.fitting.psf_refit import refit_psf_parameters
from tandemmp.fitting.spline_psf import SplinePsfModel

sp = pytest.importorskip("tandemmp._rust.spline_psf")


@pytest.fixture(scope="module")
def scipy_spline() -> BivariateSpline:
    return BivariateSpline._from_tck(sp.tck())  # noqa: SLF001


class TestExtension:
    @pytest.mark.parametrize("shape", [(5,), (13, 13), (3, 4, 5)])
    def test_matches_scipy(self, scipy_spline, shape):
        rng = np.random.default_rng(0)
        u, v = rng.uniform(-14, 14, (2, *shape))
        expected = scipy_spline(u.ravel(), v.ravel(), grid=False).reshape(shape)
        np.testing.assert_allclose(sp.eval(u, v), expected, rtol=0, atol=1e-12)

    def test_non_contiguous_input(self, scipy_spline):
        g = np.linspace(-5, 5, 11)
        yy, xx = np.meshgrid(g, g, indexing="ij")
        np.testing.assert_allclose(
            sp.eval(yy.T, xx.T), scipy_spline(yy.T, xx.T, grid=False), atol=1e-12
        )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            sp.eval(np.zeros(3), np.zeros(4))


class TestSplinePsfModel:
    def test_peak_is_contrast(self):
        f = SplinePsfModel().get_psf_function()
        assert float(f(np.array(0.0), np.array(0.0), c=-0.02)) == pytest.approx(
            -0.02, rel=2e-3
        )

    def test_axis_scaling(self):
        x, y = square_mesh(13)
        base = SplinePsfModel().get_psf_function()
        scaled = SplinePsfModel(a=1.3, b=0.8).get_psf_function()
        np.testing.assert_allclose(scaled(x, y, c=-1.0), base(1.3 * x, 0.8 * y, c=-1.0))

    def test_with_params_and_bounds(self):
        m = SplinePsfModel().with_params(b=1.1)
        assert m.psf_params == {"a": 1.0, "b": 1.1}
        assert set(m.param_bounds) == {"a", "b"}
        with pytest.raises(ValueError, match="Unknown PSF parameter"):
            m.with_params(sigma=1.0)

    def test_refit_recovers_scales(self):
        grid = np.arange(-8.0, 8.5, 0.5)
        xx, yy = np.meshgrid(grid, grid)
        target = SplinePsfModel(a=1.2, b=0.9).get_psf_function()(xx, yy, c=-1.0)
        model, _, sse = refit_psf_parameters(SplinePsfModel(), target, grid)
        assert model.psf_params["a"] == pytest.approx(1.2, abs=1e-3)
        assert model.psf_params["b"] == pytest.approx(0.9, abs=1e-3)
        assert sse < 1e-8

    def test_fit_peak_recovers_position_and_contrast(self):
        model = SplinePsfModel()
        peak = model.psf_2d_offset(13, 0.6, -0.4, -0.03)
        result = model.fit_peak(peak)
        assert result.dx == pytest.approx(0.6, abs=1e-3)
        assert result.dy == pytest.approx(-0.4, abs=1e-3)
        assert result.contrast == pytest.approx(-0.03, rel=1e-3)


def test_position_guess_ignores_masked_pixels():
    peak = SplinePsfModel().psf_2d_offset(13, 0.0, 0.0, -1.0)
    peak[:, -3:] = -5.0
    mask = np.zeros(peak.shape, dtype=bool)
    mask[:, -3:] = True
    dx, dy = _initial_position_guess(np.ma.array(peak, mask=mask))
    unmasked = _initial_position_guess(np.where(mask, 0.0, peak))
    assert (dx, dy) == pytest.approx(unmasked)
