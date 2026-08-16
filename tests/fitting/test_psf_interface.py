"""Contract tests for :mod:`tandemmp.fitting.psf_interface`.

Runs the same checks against every concrete ``PsfInterface`` implementation to
pin the shared structural contract, independent of PSF shape.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from tandemmp.fitting.gaussian_psf import GaussianPsfModel
from tandemmp.fitting.psf import PsfModel
from tandemmp.fitting.psf_interface import PsfInterface, error_metric


def _make_gaussian_spline(
    sigma: float = 2.0, grid_size: int = 101
) -> RectBivariateSpline:
    """Build a RectBivariateSpline from a 2-D Gaussian (mirrors test_psf.py)."""
    half = grid_size // 2
    coords = np.linspace(-half, half, grid_size)
    x, y = np.meshgrid(coords, coords)
    z = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return RectBivariateSpline(coords, coords, z)


def _spline_psf_model() -> PsfModel:
    model = PsfModel()
    model.set_spline_model(_make_gaussian_spline())
    return model


def _gaussian_psf_model() -> GaussianPsfModel:
    return GaussianPsfModel(sigma=2.0)


MODEL_FACTORIES = [_spline_psf_model, _gaussian_psf_model]
MODEL_IDS = ["PsfModel[spline]", "GaussianPsfModel"]


@pytest.fixture(params=MODEL_FACTORIES, ids=MODEL_IDS)
def psf_model(request: pytest.FixtureRequest) -> PsfInterface:
    return request.param()


# ---------------------------------------------------------------------------
# Structural conformance
# ---------------------------------------------------------------------------


class TestStructuralConformance:
    def test_isinstance_of_psf_interface(self, psf_model: PsfInterface) -> None:
        assert isinstance(psf_model, PsfInterface)


# ---------------------------------------------------------------------------
# psf_params
# ---------------------------------------------------------------------------


class TestPsfParams:
    def test_returns_dict_of_floats(self, psf_model: PsfInterface) -> None:
        params = psf_model.psf_params
        assert isinstance(params, dict)
        assert all(isinstance(v, float) for v in params.values())


# ---------------------------------------------------------------------------
# get_psf_function
# ---------------------------------------------------------------------------


class TestGetPsfFunction:
    def test_returns_callable(self, psf_model: PsfInterface) -> None:
        psf_fn = psf_model.get_psf_function()
        assert callable(psf_fn)

    def test_callable_shape(self, psf_model: PsfInterface) -> None:
        psf_fn = psf_model.get_psf_function()
        x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        result = psf_fn(x=x, y=y, c=-0.1)
        assert result.shape == x.shape

    def test_contrast_scaling(self, psf_model: PsfInterface) -> None:
        psf_fn = psf_model.get_psf_function()
        x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        a = psf_fn(x=x, y=y, c=-0.1)
        b = psf_fn(x=x, y=y, c=-0.2)
        np.testing.assert_allclose(b, 2 * a, atol=1e-10)


# ---------------------------------------------------------------------------
# render / psf_2d_offset
# ---------------------------------------------------------------------------


class TestRenderAndOffset:
    def test_psf_2d_offset_shape(self, psf_model: PsfInterface) -> None:
        image = psf_model.psf_2d_offset(13, 0.0, 0.0, -0.05)
        assert image.shape == (13, 13)

    def test_render_matches_psf_2d_offset(self, psf_model: PsfInterface) -> None:
        rendered = psf_model.render(0.1, -0.2, -0.05, 13)
        offset = psf_model.psf_2d_offset(13, 0.1, -0.2, -0.05)
        np.testing.assert_allclose(rendered, offset)

    def test_render_contrast_scaling(self, psf_model: PsfInterface) -> None:
        a = psf_model.render(0.0, 0.0, -0.05, 13)
        b = psf_model.render(0.0, 0.0, -0.10, 13)
        np.testing.assert_allclose(b, 2 * a, atol=1e-10)


# ---------------------------------------------------------------------------
# fit_peak
# ---------------------------------------------------------------------------


class TestFitPeak:
    def test_zero_image_returns_none_backend(self, psf_model: PsfInterface) -> None:
        peak = np.zeros((13, 13))
        result = psf_model.fit_peak(peak)
        assert result.contrast == 0.0
        assert result.converged is False
        assert result.backend == "none"

    def test_recovers_centred_peak(self, psf_model: PsfInterface) -> None:
        peak = psf_model.psf_2d_offset(13, 0.0, 0.0, -0.05)
        result = psf_model.fit_peak(peak, c_guess=-0.05, dx_guess=0.0, dy_guess=0.0)
        assert result.contrast == pytest.approx(-0.05, abs=0.005)
        assert abs(result.dx) < 0.5
        assert abs(result.dy) < 0.5
        assert result.error < 1e-6

    def test_recovers_offset_peak(self, psf_model: PsfInterface) -> None:
        peak = psf_model.psf_2d_offset(13, 0.5, -0.3, -0.08)
        result = psf_model.fit_peak(peak, dx_guess=0.5, dy_guess=-0.3)
        assert result.contrast == pytest.approx(-0.08, abs=0.01)
        assert result.dx == pytest.approx(0.5, abs=0.3)
        assert result.dy == pytest.approx(-0.3, abs=0.3)


# ---------------------------------------------------------------------------
# fit_peak_exclusion
# ---------------------------------------------------------------------------


class TestFitPeakExclusion:
    def test_shape_mismatch_raises(self, psf_model: PsfInterface) -> None:
        peak = np.zeros((13, 13))
        excl = np.zeros((11, 11), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            psf_model.fit_peak_exclusion(peak, excl)

    def test_fits_with_exclusion(self, psf_model: PsfInterface) -> None:
        peak = psf_model.psf_2d_offset(13, 0.0, 0.0, -0.06)
        excl = np.zeros((13, 13), dtype=bool)
        excl[:3, :] = True  # exclude top rows
        result = psf_model.fit_peak_exclusion(peak, excl)
        assert result.contrast == pytest.approx(-0.06, abs=0.02)
        # Error is reported over the full (unmasked) peak.
        full_fit = psf_model.psf_2d_offset(13, result.dx, result.dy, result.contrast)
        assert result.error == pytest.approx(error_metric(peak, full_fit))
