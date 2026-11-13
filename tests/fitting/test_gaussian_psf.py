"""Tests for tandemmp.fitting.gaussian_psf — analytic 2-D Gaussian PSF model."""

from __future__ import annotations

import numpy as np
import pytest

from tandemmp.fitting.gaussian_psf import _DEFAULT_SIGMA, GaussianPsfModel
from tandemmp.fitting.psf_interface import PsfInterface


class TestConstruction:
    def test_default_sigma(self) -> None:
        model = GaussianPsfModel()
        assert model.sigma == _DEFAULT_SIGMA

    def test_custom_sigma(self) -> None:
        model = GaussianPsfModel(sigma=3.0)
        assert model.sigma == 3.0

    def test_satisfies_psf_interface(self) -> None:
        assert isinstance(GaussianPsfModel(), PsfInterface)


class TestPsfParams:
    def test_contains_sigma(self) -> None:
        model = GaussianPsfModel(sigma=2.5)
        assert model.psf_params == {"sigma": 2.5}


class TestGetPsfFunction:
    def test_peak_value_at_origin(self) -> None:
        model = GaussianPsfModel(sigma=2.0)
        psf_fn = model.get_psf_function()
        value = psf_fn(x=np.array([0.0]), y=np.array([0.0]), c=-0.1)
        assert value[0] == pytest.approx(-0.1)

    def test_decays_with_distance(self) -> None:
        model = GaussianPsfModel(sigma=2.0)
        psf_fn = model.get_psf_function()
        centre = psf_fn(x=np.array([0.0]), y=np.array([0.0]), c=1.0)[0]
        edge = psf_fn(x=np.array([5.0]), y=np.array([0.0]), c=1.0)[0]
        assert edge < centre

    def test_matches_analytic_formula(self) -> None:
        sigma = 1.5
        model = GaussianPsfModel(sigma=sigma)
        psf_fn = model.get_psf_function()
        x = np.array([1.0, 2.0, -1.5])
        y = np.array([0.5, -1.0, 3.0])
        c = -0.05
        expected = c * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        np.testing.assert_allclose(psf_fn(x=x, y=y, c=c), expected)

    def test_default_contrast_is_one(self) -> None:
        model = GaussianPsfModel(sigma=1.5)
        psf_fn = model.get_psf_function()
        value = psf_fn(x=np.array([0.0]), y=np.array([0.0]))
        assert value[0] == pytest.approx(1.0)


class TestPsf2dOffsetAndRender:
    def test_shape(self) -> None:
        model = GaussianPsfModel()
        image = model.psf_2d_offset(13, 0.0, 0.0, -0.05)
        assert image.shape == (13, 13)

    def test_peak_at_offset(self) -> None:
        model = GaussianPsfModel(sigma=2.0)
        image = model.psf_2d_offset(15, 2.0, -1.0, -0.1)
        min_idx = np.unravel_index(np.argmin(image), image.shape)
        half = 15 // 2
        # min_idx is (row=y, col=x); offset applied as coords - (dx, dy).
        assert min_idx == (half - 1, half + 2)

    def test_render_delegates_to_psf_2d_offset(self) -> None:
        model = GaussianPsfModel()
        rendered = model.render(0.2, -0.3, -0.05, 13)
        offset = model.psf_2d_offset(13, 0.2, -0.3, -0.05)
        np.testing.assert_allclose(rendered, offset)


class TestFitPeak:
    def test_zero_image(self) -> None:
        model = GaussianPsfModel()
        peak = np.zeros((13, 13))
        result = model.fit_peak(peak)
        assert result.contrast == 0.0
        assert result.converged is False
        assert result.backend == "none"

    def test_recovers_centred_peak(self) -> None:
        model = GaussianPsfModel(sigma=2.0)
        peak = model.psf_2d_offset(13, 0.0, 0.0, -0.05)
        result = model.fit_peak(peak, c_guess=-0.05, dx_guess=0.0, dy_guess=0.0)
        assert result.contrast == pytest.approx(-0.05, abs=0.005)
        assert abs(result.dx) < 0.5
        assert abs(result.dy) < 0.5
        assert result.error < 1e-6
        assert result.backend == "python"

    def test_recovers_offset_peak(self) -> None:
        model = GaussianPsfModel(sigma=2.0)
        peak = model.psf_2d_offset(13, 0.5, -0.3, -0.08)
        result = model.fit_peak(peak, dx_guess=0.5, dy_guess=-0.3)
        assert result.contrast == pytest.approx(-0.08, abs=0.01)
        assert result.dx == pytest.approx(0.5, abs=0.3)
        assert result.dy == pytest.approx(-0.3, abs=0.3)

    def test_non_square_raises(self) -> None:
        model = GaussianPsfModel()
        peak = np.zeros((10, 12))
        with pytest.raises(ValueError, match="square"):
            model.fit_peak(peak)

    def test_ignores_backend_specific_kwarg(self) -> None:
        """`rust=` has no analogue here, but `PsfInterface` callers may pass it.

        Either backend must accept it without a `TypeError`.
        """
        model = GaussianPsfModel(sigma=2.0)
        peak = model.psf_2d_offset(13, 0.0, 0.0, -0.05)
        result = model.fit_peak(peak, rust=True)
        assert result.contrast == pytest.approx(-0.05, abs=0.005)


class TestFitPeakExclusion:
    def test_shape_mismatch_raises(self) -> None:
        model = GaussianPsfModel()
        peak = np.zeros((13, 13))
        excl = np.zeros((11, 11), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            model.fit_peak_exclusion(peak, excl)

    def test_fits_with_exclusion(self) -> None:
        model = GaussianPsfModel(sigma=2.0)
        peak = model.psf_2d_offset(13, 0.0, 0.0, -0.06)
        excl = np.zeros((13, 13), dtype=bool)
        excl[:3, :] = True  # exclude top rows
        result = model.fit_peak_exclusion(peak, excl)
        assert result.contrast == pytest.approx(-0.06, abs=0.02)
        assert result.backend == "python"

    def test_ignores_backend_specific_kwarg(self) -> None:
        model = GaussianPsfModel(sigma=2.0)
        peak = model.psf_2d_offset(13, 0.0, 0.0, -0.06)
        excl = np.zeros((13, 13), dtype=bool)
        excl[:3, :] = True
        result = model.fit_peak_exclusion(peak, excl, rust=True)
        assert result.contrast == pytest.approx(-0.06, abs=0.02)
