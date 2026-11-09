"""Tests for tandemmp.plotting.events — event/series rendering."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import OptimizeResult

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt  # noqa: E402

from tandemmp.core.classifications import EvClassification, ProcessingMethod
from tandemmp.events import (
    PeakImage,
    RatiometricEvent,
    SubtractionEvent,
    SubtractionSeries,
)
from tandemmp.fitting.psf import PsfModel
from tandemmp.pipeline.investigator import MultiFitResult
from tandemmp.plotting.events import (
    assignment_check,
    fit_reports,
    ranking_plot,
    report,
    report_and_draw,
    show,
    show_3d_fit,
    show_basic_fit,
    show_fits,
    show_fits_improved,
    show_series,
    show_series_big,
    show_series_corrections,
    show_series_impression,
    show_subtraction_corrections,
)

# Helpers / fixtures


def _make_gaussian_spline(
    sigma: float = 2.0, grid_size: int = 101
) -> RectBivariateSpline:
    """Build a RectBivariateSpline from a 2-D Gaussian."""
    half = grid_size // 2
    coords = np.linspace(-half, half, grid_size)
    x, y = np.meshgrid(coords, coords)
    z = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return RectBivariateSpline(coords, coords, z)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture()
def spline_model() -> PsfModel:
    """PsfModel backed by a Gaussian spline (no open_iscat needed)."""
    model = PsfModel()
    model.set_spline_model(_make_gaussian_spline())
    return model


def _make_peak_image(
    model: PsfModel,
    method: ProcessingMethod = ProcessingMethod.ratiometric,
    width: int = 13,
    dx: float = 0.0,
    dy: float = 0.0,
    c: float = -0.05,
    frame: int = 50,
) -> PeakImage:
    """Create a synthetic PeakImage from a PsfModel."""
    image = model.psf_2d_offset(width, dx, dy, c)
    mid = (100, 100)
    cutout = [
        slice(mid[0] - width // 2, mid[0] + width // 2 + 1),
        slice(mid[1] - width // 2, mid[1] + width // 2 + 1),
    ]
    return PeakImage(image=image, method=method, midpoint=mid, cutout=cutout, i=frame)


def _make_ratiometric_event(
    spline_model: PsfModel, *, navg: int = 5, frame: int = 50
) -> RatiometricEvent:
    peak = _make_peak_image(spline_model, c=-0.05, frame=frame)
    return RatiometricEvent(peak, fit=True, navg=navg, peak_model=spline_model)


def _make_subtraction_series(
    spline_model: PsfModel, *, n: int = 5, start_frame: int = 50
) -> SubtractionSeries:
    events = []
    for frame in range(start_frame, start_frame + n):
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.subtraction, c=-0.05, frame=frame
        )
        ev = SubtractionEvent(
            peak, subtracted_i=frame - 5, fit=True, peak_model=spline_model
        )
        events.append(ev)
    return SubtractionSeries(events)


def _make_fit_result(
    series: SubtractionSeries,
    *,
    distance: float = 1.0,
    fun: float = 1e-3,
    penalty: float = 0.01,
    image_error: float = 0.001,
) -> MultiFitResult:
    return MultiFitResult(
        series=series,
        classification=EvClassification.Movement,  # full-series slice
        fit=OptimizeResult(fun=fun, x=series.c),
        fit_type="untracked simple",
        c=series.c,
        penalty=penalty,
        image_error=image_error,
        fit_image=np.zeros_like(series.events[0].peak.image),
        distance=distance,
    )


def _make_fit3d_event(spline_model: PsfModel, *, navg: int = 5) -> RatiometricEvent:
    """A RatiometricEvent with an applied 3-D fit and recorded attempts."""
    n = navg * 2 + 1
    ev = _make_ratiometric_event(spline_model, navg=navg)
    series = _make_subtraction_series(spline_model, n=n)
    fit_result = _make_fit_result(series, distance=1.0)
    ev.apply_3D_fit(fit_result)
    ev._first_attempt = series
    ev._corrections_attempts = [
        fit_result,
        _make_fit_result(
            series, distance=2.0, fun=2e-3, penalty=0.02, image_error=0.002
        ),
    ]
    return ev


# AbstractEvent-level: show / show_basic_fit / show_fits


class TestShow:
    def test_runs_without_error(self, spline_model):
        ev = _make_ratiometric_event(spline_model)
        show(ev)

    def test_with_explicit_axes(self, spline_model):
        ev = _make_ratiometric_event(spline_model)
        _, ax = plt.subplots()
        show(ev, axes=ax)

    def test_with_string_title(self, spline_model):
        ev = _make_ratiometric_event(spline_model)
        show(ev, title="custom title")


class TestShowBasicFit:
    def test_runs_without_error(self, spline_model):
        ev = _make_ratiometric_event(spline_model)
        show_basic_fit(ev)


class TestShowFits:
    def test_basic_dispatch(self, spline_model):
        """No 3-D fit applied -> dispatches to the basic-fit renderer."""
        ev = _make_ratiometric_event(spline_model)
        assert ev.fit_3D is False
        show_fits(ev)

    def test_improved_dispatch(self, spline_model):
        """3-D fit applied -> dispatches to show_fits_improved."""
        ev = _make_fit3d_event(spline_model)
        assert ev.fit_3D is True
        show_fits(ev)


# SubtractionEvent / SubtractionSeries


class TestShowSubtractionCorrections:
    def test_runs_without_error(self, spline_model):
        series = _make_subtraction_series(spline_model, n=1)
        show_subtraction_corrections(series.events[0])


class TestShowSeries:
    def test_runs_without_error(self, spline_model):
        series = _make_subtraction_series(spline_model, n=3)
        show_series(series)


class TestShowSeriesImpression:
    def test_runs_without_error(self, spline_model, monkeypatch):
        # impression() falls back to get_psf_model() when no model is
        # passed explicitly; point that at our open_iscat-free spline model.
        monkeypatch.setattr(
            "tandemmp.fitting.psf.get_psf_model", lambda **_: spline_model
        )
        series = _make_subtraction_series(spline_model, n=3)
        show_series_impression(series)


class TestShowSeriesBig:
    def test_runs_without_error(self, spline_model):
        series = _make_subtraction_series(spline_model, n=3)
        show_series_big(series)


class TestShowSeriesCorrections:
    def test_runs_without_error(self, spline_model):
        series = _make_subtraction_series(spline_model, n=3)
        show_series_corrections(series)


# RatiometricEvent — 3-D fit and reports


class TestShow3DFit:
    def test_runs_without_error(self, spline_model):
        ev = _make_fit3d_event(spline_model)
        show_3d_fit(ev)


class TestShowFitsImproved:
    def test_runs_without_error(self, spline_model):
        ev = _make_fit3d_event(spline_model)
        show_fits_improved(ev)


class TestReport:
    def test_runs_without_error(self, spline_model):
        ev = _make_fit3d_event(spline_model)
        report(ev)

    def test_without_connection(self, spline_model):
        """The no-3D-fit branch of report() (event.has_connection is False)."""
        ev = _make_ratiometric_event(spline_model, navg=2)
        ev._first_attempt = _make_subtraction_series(spline_model, n=5)
        report(ev)


class TestReportAndDraw:
    def test_runs_without_error(self, spline_model):
        ev = _make_fit3d_event(spline_model)
        report_and_draw(ev)


class TestFitReports:
    def test_runs_without_error(self, spline_model):
        ev = _make_fit3d_event(spline_model)
        fit_reports(ev)

    def test_unsorted(self, spline_model):
        ev = _make_fit3d_event(spline_model)
        fit_reports(ev, sort=False)


class TestAssignmentCheck:
    def test_runs_without_error(self, spline_model):
        ev = _make_fit3d_event(spline_model)
        assignment_check(ev)


class TestRankingPlot:
    def test_runs_without_error(self, spline_model):
        ev = _make_fit3d_event(spline_model)
        ranking_plot(ev)
