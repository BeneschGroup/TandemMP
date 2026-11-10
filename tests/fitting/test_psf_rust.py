"""Parity tests: Rust spline PSF (tandemmp._rust.psf.SplinePsf) vs SciPy/Python."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

pytest.importorskip("tandemmp._rust.psf")

from tandemmp._rust.psf import SplinePsf  # noqa: E402
from tandemmp.fitting.psf import PsfModel, get_psf_model  # noqa: E402
from tandemmp.fitting.psf_interface import square_mesh  # noqa: E402
from tandemmp.fitting.spline_psf import get_experimental_psf_model  # noqa: E402

L_THUM = 21


def _make_gaussian_spline(
    sigma: float = 2.0, grid_size: int = 61
) -> RectBivariateSpline:
    half = grid_size // 2
    coords = np.linspace(-half, half, grid_size)
    x, y = np.meshgrid(coords, coords)
    z = -np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return RectBivariateSpline(coords, coords, z)


@pytest.fixture()
def spline_model() -> PsfModel:
    model = PsfModel(use_rust=True)
    model.set_spline_model(_make_gaussian_spline())
    return model


@pytest.fixture()
def experimental_model() -> PsfModel:
    return get_experimental_psf_model(use_rust=True)


# Eval parity vs RectBivariateSpline


class TestEvalParity:
    def test_synthetic_gaussian_spline(self, spline_model: PsfModel) -> None:
        rust_model = spline_model._rust_model()  # noqa: SLF001
        rng = np.random.default_rng(0)
        for dx, dy in zip(rng.uniform(-5, 5, 20), rng.uniform(-5, 5, 20), strict=True):
            py_img = spline_model.psf_2d_offset(L_THUM, dx, dy, 1.0)
            rust_img = rust_model.render(dx, dy, 1.0, L_THUM)
            np.testing.assert_allclose(rust_img, py_img, atol=1e-10)

    def test_bundled_experimental_psf(self, experimental_model: PsfModel) -> None:
        rust_model = experimental_model._rust_model()  # noqa: SLF001
        for dx, dy, c in [(0.0, 0.0, -0.05), (0.3, -0.7, -0.08), (-2.4, 3.1, -0.02)]:
            py_img = experimental_model.psf_2d_offset(L_THUM, dx, dy, c)
            rust_img = rust_model.render(dx, dy, c, L_THUM)
            np.testing.assert_allclose(rust_img, py_img, atol=1e-10)

    def test_near_edge_clamping(self, experimental_model: PsfModel) -> None:
        """Query points outside the fitted domain should clamp identically."""
        rust_model = experimental_model._rust_model()  # noqa: SLF001
        for dx, dy in [(50.0, 0.0), (0.0, -50.0), (30.0, -30.0)]:
            py_img = experimental_model.psf_2d_offset(L_THUM, dx, dy, -0.05)
            rust_img = rust_model.render(dx, dy, -0.05, L_THUM)
            np.testing.assert_allclose(rust_img, py_img, atol=1e-10)


# Sign convention


class TestSignConvention:
    def test_binding_event_recovers_negative_contrast(
        self, experimental_model: PsfModel
    ) -> None:
        """A synthetic binding event (negative contrast, TandemMP convention)
        must fit to a negative contrast, not the positive mirror.
        """
        c_true, dx_true, dy_true = -0.06, 0.2, -0.35
        thumb = experimental_model.psf_2d_offset(L_THUM, dx_true, dy_true, c_true)
        result = experimental_model.fit_peak(
            thumb, c_guess=c_true, dx_guess=0.0, dy_guess=0.0
        )
        assert result.backend == "rust"
        assert result.contrast < 0
        assert result.contrast == pytest.approx(c_true, abs=1e-4)
        assert result.dx == pytest.approx(dx_true, abs=0.05)
        assert result.dy == pytest.approx(dy_true, abs=0.05)

    def test_centre_pixel_is_negative(self, experimental_model: PsfModel) -> None:
        """The bundled spline's centre value is ~-1, so a centred peak with
        negative c directly matches TandemMP's contrast convention.
        """
        img = experimental_model.psf_2d_offset(L_THUM, 0.0, 0.0, -0.05)
        mid = L_THUM // 2
        assert img[mid, mid] == pytest.approx(-0.05, abs=1e-3)


# Fit parity: Rust vs Python/SciPy BFGS


class TestFitParity:
    @pytest.mark.parametrize(
        ("dx_true", "dy_true", "c_true", "label"),
        [
            (0.0, 0.0, -0.05, "centred"),
            (0.4, -0.3, -0.06, "subpixel"),
            (-8.5, 7.8, -0.04, "near_edge"),
        ],
    )
    def test_noise_free(
        self,
        experimental_model: PsfModel,
        dx_true: float,
        dy_true: float,
        c_true: float,
        label: str,
    ) -> None:
        thumb = experimental_model.psf_2d_offset(L_THUM, dx_true, dy_true, c_true)

        rust_result = experimental_model.fit_peak(thumb, rust=True)
        py_result = experimental_model.fit_peak(thumb, rust=False)

        assert abs(rust_result.dx - py_result.dx) < 0.05, label
        assert abs(rust_result.dy - py_result.dy) < 0.05, label
        assert abs(rust_result.contrast - py_result.contrast) < 0.01 * abs(c_true), (
            label
        )

    def test_noisy(self, experimental_model: PsfModel) -> None:
        rng = np.random.default_rng(1)
        dx_true, dy_true, c_true = 0.6, -0.2, -0.07
        thumb = experimental_model.psf_2d_offset(L_THUM, dx_true, dy_true, c_true)
        thumb = thumb + rng.normal(scale=0.002, size=thumb.shape)

        rust_result = experimental_model.fit_peak(thumb, rust=True)
        py_result = experimental_model.fit_peak(thumb, rust=False)

        assert abs(rust_result.dx - py_result.dx) < 0.05
        assert abs(rust_result.dy - py_result.dy) < 0.05
        assert abs(rust_result.contrast - py_result.contrast) < 0.01 * abs(c_true)


# SplinePsf (direct Rust binding)


class TestSplinePsfBinding:
    def test_render_matches_scipy(self) -> None:
        spline = _make_gaussian_spline()
        tx, ty, c = spline.tck
        kx, ky = spline.degrees
        rust = SplinePsf(
            np.asarray(tx), np.asarray(ty), np.asarray(c), int(kx), int(ky)
        )

        x_coords, y_coords = square_mesh(L_THUM)
        expected = spline(y_coords - 0.4, x_coords - (-0.6), grid=False) * -1.0
        got = rust.render(-0.6, 0.4, 1.0, L_THUM)
        np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_fit_recovers_truth(self) -> None:
        spline = _make_gaussian_spline()
        tx, ty, c = spline.tck
        kx, ky = spline.degrees
        rust = SplinePsf(
            np.asarray(tx), np.asarray(ty), np.asarray(c), int(kx), int(ky)
        )

        truth = (-0.07, 0.5, -0.4)
        thumb = rust.render(truth[1], truth[2], truth[0], L_THUM)
        c_fit, dx_fit, dy_fit, error, converged = rust.fit(thumb, (-0.05, 0.0, 0.0))
        assert converged
        assert c_fit == pytest.approx(truth[0], abs=1e-6)
        assert dx_fit == pytest.approx(truth[1], abs=1e-6)
        assert dy_fit == pytest.approx(truth[2], abs=1e-6)
        assert error < 1e-12

    def test_fit_batch_matches_scalar(self) -> None:
        spline = _make_gaussian_spline()
        tx, ty, c = spline.tck
        kx, ky = spline.degrees
        rust = SplinePsf(
            np.asarray(tx), np.asarray(ty), np.asarray(c), int(kx), int(ky)
        )

        rng = np.random.default_rng(2)
        n = 8
        truths = np.stack(
            [rng.uniform(-0.1, -0.02, n), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)],
            axis=1,
        )
        thumbs = np.stack([rust.render(dx, dy, c_, L_THUM) for c_, dx, dy in truths])
        inits = np.stack([truths[:, 0] * 0.8, np.zeros(n), np.zeros(n)], axis=1)

        batch = rust.fit_batch(thumbs, inits)
        assert batch.shape == (n, 5)
        for i in range(n):
            scalar = rust.fit(thumbs[i], tuple(inits[i]))
            np.testing.assert_allclose(batch[i], scalar, atol=1e-8)
            np.testing.assert_allclose(batch[i, :3], truths[i], atol=1e-6)


# get_psf_model default wiring


class TestGetPsfModelDefault:
    def test_default_is_experimental_spline_with_rust(self) -> None:
        model = get_psf_model()
        assert isinstance(model, PsfModel)
        assert model.has_spline_model
        assert model.use_rust is True
