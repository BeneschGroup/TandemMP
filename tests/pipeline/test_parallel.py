"""Tests for tandemmp.pipeline.parallel — parallel event processing utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import OptimizeResult

from tandemmp.core.classifications import EvClassification, ProcessingMethod
from tandemmp.events._base import PeakImage
from tandemmp.fitting.psf import PsfModel
from tandemmp.pipeline.parallel import (
    fit_peak,
    make_ratiometric_event_from_peak,
    make_subtraction_event,
    make_subtraction_event_from_peak,
    median_frame,
    parallel_find_classification,
)

# Helpers


def _make_gaussian_spline(
    sigma: float = 2.0, grid_size: int = 101
) -> RectBivariateSpline:
    half = grid_size // 2
    coords = np.linspace(-half, half, grid_size)
    x, y = np.meshgrid(coords, coords)
    z = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return RectBivariateSpline(coords, coords, z)


@pytest.fixture()
def spline_model() -> PsfModel:
    model = PsfModel()
    model.set_spline_model(_make_gaussian_spline())
    return model


def _make_synthetic_peak(
    model: PsfModel,
    width: int = 13,
    c: float = -0.05,
) -> np.ndarray:
    return model.psf_2d_offset(width, 0.0, 0.0, c)


def _make_peak_image(
    width: int = 13,
    method: ProcessingMethod = ProcessingMethod.subtraction,
    midpoint: tuple[int, int] = (50, 50),
    frame: int = 10,
    image: np.ndarray | None = None,
) -> PeakImage:
    if image is None:
        image = np.random.default_rng(0).normal(0, 0.001, (width, width))
    return PeakImage(
        image=image,
        method=method,
        midpoint=midpoint,
        cutout=[slice(44, 57), slice(44, 57)],
        i=frame,
    )


# fit_peak


class TestFitPeak:
    def test_delegates_to_fit_peak(self, spline_model):
        peak = _make_synthetic_peak(spline_model)
        result = fit_peak(peak, spline_model, method="BFGS")
        assert result.contrast == pytest.approx(-0.05, abs=0.01)


# median_frame


class TestMedianFrame:
    def test_returns_same_shape(self):
        frame = np.random.default_rng(0).random((50, 50))
        result = median_frame(frame, 3)
        assert result.shape == frame.shape

    def test_smooths_noise(self):
        rng = np.random.default_rng(42)
        frame = rng.random((30, 30))
        result = median_frame(frame, 5)
        # Median filter should reduce variance
        assert np.std(result) < np.std(frame)

    def test_uniform_unchanged(self):
        frame = np.ones((20, 20)) * 5.0
        result = median_frame(frame, 3)
        np.testing.assert_allclose(result, 5.0)


# make_subtraction_event_from_peak


class TestMakeSubtractionEventFromPeak:
    def test_creates_event(self, spline_model):
        peak_img = _make_synthetic_peak(spline_model)
        peak = _make_peak_image(image=peak_img)
        event = make_subtraction_event_from_peak(
            peak, subtracted_i=5, fit=True, peak_model=spline_model
        )
        assert event.subtracted_i == 5
        assert event.i == peak.i

    def test_no_fit(self, spline_model):
        peak = _make_peak_image()
        event = make_subtraction_event_from_peak(
            peak, subtracted_i=3, fit=False, peak_model=spline_model
        )
        assert event.subtracted_i == 3


# make_ratiometric_event_from_peak


class TestMakeRatiometricEventFromPeak:
    def test_creates_event(self, spline_model):
        peak_img = _make_synthetic_peak(spline_model)
        peak = _make_peak_image(image=peak_img, method=ProcessingMethod.ratiometric)
        event = make_ratiometric_event_from_peak(peak, navg=5, peak_model=spline_model)
        assert event.navg == 5
        assert event.i == peak.i


# make_subtraction_event


class TestMakeSubtractionEvent:
    def test_creates_from_raw_args(self, spline_model):
        image = _make_synthetic_peak(spline_model)
        event = make_subtraction_event(
            image=image,
            index=10,
            midpoint=(50, 50),
            cutout=[slice(44, 57), slice(44, 57)],
            subtracted_i=5,
            fit=True,
            peak_model=spline_model,
        )
        assert event.i == 10
        assert event.subtracted_i == 5
        assert event.peak.method == ProcessingMethod.subtraction

    def test_no_fit(self, spline_model):
        image = np.zeros((13, 13))
        event = make_subtraction_event(
            image=image,
            index=7,
            midpoint=(50, 50),
            cutout=[slice(44, 57), slice(44, 57)],
            subtracted_i=2,
            fit=False,
            peak_model=spline_model,
        )
        assert event.i == 7


# parallel_find_classification


def _make_mock_series(navg: int = 3) -> MagicMock:
    """Build a SubtractionSeries-like mock with the attributes the function reads."""
    length = navg * 2
    series = MagicMock()
    series.dx = np.zeros(length)
    series.dy = np.zeros(length)
    series.smooth_dx = np.zeros(length)
    series.smooth_dy = np.zeros(length)
    series.m = np.array([50.0] * navg + [100.0] * navg)
    series.avg_residual_noise = 0.01
    series.is_bad = np.zeros(length, dtype=bool)
    series.__getitem__ = lambda self_, idx: _make_mock_event(-0.02)
    # fit_to and impression are used for native fits
    series.fit_to.return_value = OptimizeResult(fun=5.0, x=np.array([-0.02, -0.04]))
    series.impression.return_value = np.zeros((13, 13))
    return series


def _make_mock_event(c: float) -> MagicMock:
    ev = MagicMock()
    ev.c = c
    return ev


def _stub_fit_3d_ratiometric(peak, dx, dy, classification, **kwargs):
    """Deterministic stub: fun depends on classification so argmin is predictable."""
    del peak, dx, dy, kwargs
    score_map = {
        EvClassification.Binding: 3.0,
        EvClassification.Movement: 1.0,
        EvClassification.Unbinding: 2.0,
    }
    result = OptimizeResult(
        fun=score_map.get(classification, 4.0),
        x=np.array([-0.03, -0.05]),
    )
    return result, np.zeros((13, 13))


class TestParallelFindClassification:
    def test_binding_only_when_particle_c_is_none(self, spline_model, monkeypatch):
        """particle_c=None → only Binding hypothesis is evaluated."""
        monkeypatch.setattr(
            "tandemmp.pipeline.parallel.fit_3d_ratiometric",
            _stub_fit_3d_ratiometric,
        )
        peak = np.zeros((13, 13))
        series = _make_mock_series()
        result = parallel_find_classification(
            peak=peak,
            series=series,
            minimum_mass=10.0,
            peak_model=spline_model,
            particle_c=None,
            allow_native=False,
        )
        assert result["classification"] == EvClassification.Binding
        assert result["fit_type"] == "untracked simple"

    def test_all_hypotheses_when_particle_c_nan(self, spline_model, monkeypatch):
        """particle_c=NaN → Binding/Movement/Unbinding; best = Movement (lowest stub fun)."""
        monkeypatch.setattr(
            "tandemmp.pipeline.parallel.fit_3d_ratiometric",
            _stub_fit_3d_ratiometric,
        )
        peak = np.zeros((13, 13))
        series = _make_mock_series()
        result = parallel_find_classification(
            peak=peak,
            series=series,
            minimum_mass=10.0,
            peak_model=spline_model,
            particle_c=float("nan"),
            allow_native=False,
        )
        # Movement with a 2-param fit that resolves to association by default
        # (signals: before=50, after=100 → mass increase → Association).
        # Stub returns len(x) == 2, so classify_mass_change runs and returns
        # Association.
        assert result["classification"] == EvClassification.Association
        assert result["fit_type"].startswith("untracked")

    def test_movement_and_unbinding_when_particle_c_set(
        self, spline_model, monkeypatch
    ):
        """particle_c is a number → only Movement and Unbinding are evaluated."""
        monkeypatch.setattr(
            "tandemmp.pipeline.parallel.fit_3d_ratiometric",
            _stub_fit_3d_ratiometric,
        )
        peak = np.zeros((13, 13))
        series = _make_mock_series()
        result = parallel_find_classification(
            peak=peak,
            series=series,
            minimum_mass=10.0,
            peak_model=spline_model,
            particle_c=-0.03,
            allow_native=False,
        )
        # Movement has lower stub fun than Unbinding → Movement wins (resolved
        # to Association via classify_mass_change as above).
        assert result["classification"] == EvClassification.Association
        assert result["fit_type"].startswith("tracked")

    def test_ghost_unbind_series_restricts_to_unbinding(
        self, spline_model, monkeypatch
    ):
        """ghost_unbind_series=True forces Unbinding as the sole candidate."""
        monkeypatch.setattr(
            "tandemmp.pipeline.parallel.fit_3d_ratiometric",
            _stub_fit_3d_ratiometric,
        )
        peak = np.zeros((13, 13))
        series = _make_mock_series()
        result = parallel_find_classification(
            peak=peak,
            series=series,
            minimum_mass=10.0,
            peak_model=spline_model,
            ghost_unbind_series=True,
            particle_c=float("nan"),
            allow_native=False,
        )
        assert result["classification"] == EvClassification.Unbinding
