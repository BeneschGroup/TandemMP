"""Tests for tandemmp.fitting.psf — PSF fitting module."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from tandemmp.core.classifications import EvClassification
from tandemmp.fitting.gaussian_psf import GaussianPsfModel
from tandemmp.fitting.psf import (
    PeakFitResult,
    PsfModel,
    correct_known_positions,
    dynamic_mesh,
    error_metric,
    fit_3d_ratiometric,
    get_psf_model,
    make_3d_image,
    native_impression,
    psf_2d_cached,
    square_mesh,
)

# Helpers


def _make_gaussian_spline(
    sigma: float = 2.0, grid_size: int = 101
) -> RectBivariateSpline:
    """Build a RectBivariateSpline from a 2-D Gaussian."""
    half = grid_size // 2
    coords = np.linspace(-half, half, grid_size)
    x, y = np.meshgrid(coords, coords)
    z = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return RectBivariateSpline(coords, coords, z)


@pytest.fixture()
def spline_model():
    """PsfModel backed by a Gaussian spline."""
    model = PsfModel()
    model.set_spline_model(_make_gaussian_spline())
    return model


def _make_synthetic_peak(
    model: PsfModel,
    width: int = 13,
    dx: float = 0.0,
    dy: float = 0.0,
    c: float = -0.05,
) -> np.ndarray:
    """Generate a noise-free PSF peak for testing."""
    return model.psf_2d_offset(width, dx, dy, c)


# PeakFitResult


class TestPeakFitResult:
    def test_fields(self):
        r = PeakFitResult(
            contrast=-0.05,
            dx=0.1,
            dy=-0.2,
            error=0.001,
            converged=True,
            backend="spline",
        )
        assert r.contrast == -0.05
        assert r.dx == 0.1
        assert r.dy == -0.2
        assert r.error == 0.001
        assert r.converged is True
        assert r.backend == "spline"

    def test_frozen(self):
        r = PeakFitResult(
            contrast=0.0, dx=0.0, dy=0.0, error=0.0, converged=False, backend="none"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.contrast = 1.0  # type: ignore[misc]


# square_mesh


class TestSquareMesh:
    def test_shape(self):
        x, y = square_mesh(13)
        assert x.shape == (13, 13)
        assert y.shape == (13, 13)

    def test_centre_is_zero(self):
        x, y = square_mesh(13)
        mid = 13 // 2
        assert x[mid, mid] == 0
        assert y[mid, mid] == 0

    def test_symmetry(self):
        x, y = square_mesh(11)
        np.testing.assert_array_equal(x, -x[:, ::-1])
        np.testing.assert_array_equal(y, -y[::-1, :])

    def test_cached(self):
        a = square_mesh(7)
        b = square_mesh(7)
        assert a[0] is b[0]


# dynamic_mesh


class TestDynamicMesh:
    def test_odd_dimensions(self):
        x, y = dynamic_mesh(11, 13)
        assert x.shape == (11, 13)

    def test_even_dimensions(self):
        x, y = dynamic_mesh(10, 12)
        assert x.shape == (10, 12)

    def test_mixed(self):
        x, y = dynamic_mesh(10, 13)
        assert x.shape == (10, 13)


# error_metric


class TestErrorMetric:
    def test_identical(self):
        a = np.random.default_rng(0).random((5, 5))
        assert error_metric(a, a) == 0.0

    def test_known_value(self):
        a = np.zeros((3, 3))
        b = np.ones((3, 3))
        assert error_metric(a, b) == pytest.approx(9.0)

    def test_with_exclusion(self):
        a = np.zeros((3, 3))
        b = np.ones((3, 3))
        excl = np.zeros((3, 3), dtype=bool)
        excl[0, :] = True  # exclude top row (3 pixels)
        assert error_metric(a, b, excl) == pytest.approx(6.0)


# PsfModel


class TestPsfModel:
    def test_defaults(self):
        m = PsfModel()
        assert m.a12 == pytest.approx(-5.1844, abs=0.01)
        assert not m.has_spline_model
        assert m.spline_model is None
        assert m.use_rust is False

    def test_custom_params(self):
        m = PsfModel(a12=-1.0, w=2.0, s=3.0, use_rust=True)
        assert m.psf_params == {"a12": -1.0, "w": 2.0, "s": 3.0}
        assert m.use_rust is True

    def test_set_spline_model(self, spline_model):
        assert spline_model.has_spline_model
        assert spline_model.spline_model is not None

    def test_psf_2d_offset_shape(self, spline_model):
        img = spline_model.psf_2d_offset(13, 0.0, 0.0, 1.0)
        assert img.shape == (13, 13)

    def test_psf_2d_offset_centred(self, spline_model):
        """Centred PSF should be symmetric."""
        img = spline_model.psf_2d_offset(13, 0.0, 0.0, 1.0)
        np.testing.assert_allclose(img, img[::-1, :], atol=1e-10)
        np.testing.assert_allclose(img, img[:, ::-1], atol=1e-10)

    def test_psf_2d_offset_contrast_scaling(self, spline_model):
        img1 = spline_model.psf_2d_offset(13, 0.0, 0.0, 1.0)
        img2 = spline_model.psf_2d_offset(13, 0.0, 0.0, 2.0)
        np.testing.assert_allclose(img2, img1 * 2.0, atol=1e-10)

    def test_get_psf_function_spline(self, spline_model):
        fn = spline_model.get_psf_function()
        x, y = square_mesh(13)
        result = fn(x, y, c=1.0)
        assert result.shape == (13, 13)

    def test_hashable_by_identity(self):
        """PsfModel uses identity-based hashing so mutation via
        set_spline_model does not corrupt caches keyed on the model."""
        m1 = PsfModel(a12=-1.0, w=2.0, s=3.0)
        m2 = PsfModel(a12=-1.0, w=2.0, s=3.0)
        d = {m1: "a", m2: "b"}
        assert len(d) == 2
        assert hash(m1) == hash(m1)


# PsfModel.fit_peak


class TestFitPeak:
    def test_zero_image(self, spline_model):
        peak = np.zeros((13, 13))
        result = spline_model.fit_peak(peak)
        assert result.contrast == 0.0
        assert result.converged is False
        assert result.backend == "none"

    def test_non_square_raises(self, spline_model):
        peak = np.zeros((10, 12))
        with pytest.raises(ValueError, match="square"):
            spline_model.fit_peak(peak)

    def test_recovers_centred_peak(self, spline_model):
        peak = _make_synthetic_peak(spline_model, c=-0.05)
        result = spline_model.fit_peak(peak, c_guess=-0.05, dx_guess=0.0, dy_guess=0.0)
        assert result.contrast == pytest.approx(-0.05, abs=0.005)
        assert abs(result.dx) < 0.5
        assert abs(result.dy) < 0.5
        assert result.error < 1e-6
        assert result.backend == "spline"

    def test_recovers_offset_peak(self, spline_model):
        peak = _make_synthetic_peak(spline_model, dx=0.5, dy=-0.3, c=-0.08)
        result = spline_model.fit_peak(peak, dx_guess=0.5, dy_guess=-0.3)
        assert result.contrast == pytest.approx(-0.08, abs=0.01)
        assert result.dx == pytest.approx(0.5, abs=0.3)
        assert result.dy == pytest.approx(-0.3, abs=0.3)

    def test_allow_offset(self, spline_model):
        peak = _make_synthetic_peak(
            spline_model, c=-0.8
        )  # large contrast triggers offset
        result = spline_model.fit_peak(peak, c_guess=-0.8, dx_guess=0.0, dy_guess=0.0)
        assert result.backend == "spline"


# PsfModel.fit_peak_exclusion


class TestFitPeakExclusion:
    def test_shape_mismatch_raises(self, spline_model):
        peak = np.zeros((13, 13))
        excl = np.zeros((11, 11), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            spline_model.fit_peak_exclusion(peak, excl)

    def test_fits_with_exclusion(self, spline_model):
        peak = _make_synthetic_peak(spline_model, c=-0.06)
        excl = np.zeros((13, 13), dtype=bool)
        excl[:3, :] = True  # exclude top rows
        result = spline_model.fit_peak_exclusion(peak, excl)
        assert result.contrast == pytest.approx(-0.06, abs=0.02)
        assert result.backend == "spline"


# get_psf_model


class TestGetPsfModel:
    def test_default_kind_is_bessel(self):
        model = get_psf_model()
        assert isinstance(model, PsfModel)

    def test_kind_bessel_returns_psf_model(self):
        model = get_psf_model(kind="bessel")
        assert isinstance(model, PsfModel)

    def test_kind_gaussian_returns_gaussian_model(self):
        model = get_psf_model(kind="gaussian")
        assert isinstance(model, GaussianPsfModel)

    def test_gaussian_model_is_cached(self):
        first = get_psf_model(kind="gaussian")
        second = get_psf_model(kind="gaussian")
        assert first is second

    def test_bessel_and_gaussian_are_distinct_instances(self):
        bessel = get_psf_model(kind="bessel")
        gaussian = get_psf_model(kind="gaussian")
        assert bessel is not gaussian


# psf_2d_cached


class TestPsf2dCached:
    def test_returns_array(self, spline_model):
        img = psf_2d_cached(0.0, 0.0, 1.0, 13, spline_model)
        assert isinstance(img, np.ndarray)
        assert img.shape == (13, 13)

    def test_contrast_scaling(self, spline_model):
        img1 = psf_2d_cached(0.0, 0.0, 1.0, 13, spline_model)
        img2 = psf_2d_cached(0.0, 0.0, 2.0, 13, spline_model)
        np.testing.assert_allclose(img2, img1 * 2.0, atol=1e-10)


# correct_known_positions


class TestCorrectKnownPositions:
    def test_subtraction(self, spline_model):
        peak = psf_2d_cached(0.0, 0.0, -0.05, 13, spline_model)
        corrected = correct_known_positions(peak, [0.0], [0.0], [-0.05], spline_model)
        np.testing.assert_allclose(corrected, 0.0, atol=1e-10)


# make_3d_image


class TestMake3dImage:
    def _dx_dy(self, n=10):
        """Constant position series (n frames)."""
        return [0.0] * n, [0.0] * n

    def test_binding(self, spline_model):
        dx, dy = self._dx_dy()
        img = make_3d_image(
            dx,
            dy,
            -0.05,
            EvClassification.Binding,
            navg=5,
            l_thum=13,
            psf_model=spline_model,
        )
        assert img.shape == (13, 13)
        # Binding = after only → should have negative centre (PSF * -c is positive for negative c)
        assert img[6, 6] != 0.0

    def test_unbinding(self, spline_model):
        dx, dy = self._dx_dy()
        img = make_3d_image(
            dx,
            dy,
            -0.05,
            EvClassification.Unbinding,
            navg=5,
            l_thum=13,
            psf_model=spline_model,
        )
        assert img.shape == (13, 13)

    def test_movement(self, spline_model):
        dx, dy = self._dx_dy()
        img = make_3d_image(
            dx,
            dy,
            -0.05,
            EvClassification.Movement,
            navg=5,
            l_thum=13,
            psf_model=spline_model,
        )
        assert img.shape == (13, 13)
        # Same position and contrast → after - before ≈ 0
        np.testing.assert_allclose(img, 0.0, atol=1e-10)

    def test_two_contrasts(self, spline_model):
        dx, dy = self._dx_dy()
        img = make_3d_image(
            dx,
            dy,
            [-0.03, -0.06],
            EvClassification.Movement,
            navg=5,
            l_thum=13,
            psf_model=spline_model,
        )
        assert img.shape == (13, 13)
        # Different contrasts → non-zero difference
        assert np.max(np.abs(img)) > 0


# fit_3d_ratiometric


class TestFit3dRatiometric:
    def test_binding_recovers_contrast(self, spline_model):
        dx = [0.0] * 10
        dy = [0.0] * 10
        c_true = -0.05
        peak = make_3d_image(
            dx,
            dy,
            c_true,
            EvClassification.Binding,
            navg=5,
            l_thum=13,
            psf_model=spline_model,
        )
        result, fit_img = fit_3d_ratiometric(
            peak,
            dx,
            dy,
            EvClassification.Binding,
            c_guess=c_true,
            psf_model=spline_model,
        )
        assert result.fun < 1e-6
        assert fit_img.shape == (13, 13)

    def test_lock_requires_particle_c(self, spline_model):
        with pytest.raises(ValueError, match="particle_c"):
            fit_3d_ratiometric(
                np.zeros((13, 13)),
                [0.0] * 10,
                [0.0] * 10,
                EvClassification.Movement,
                lock=True,
                psf_model=spline_model,
            )


# native_impression


class TestNativeImpression:
    def test_returns_image(self, spline_model):
        dx = [0.0] * 10
        dy = [0.0] * 10
        img = native_impression(
            dx, dy, [-0.05], EvClassification.Binding, l_thum=13, psf_model=spline_model
        )
        assert img.shape == (13, 13)
