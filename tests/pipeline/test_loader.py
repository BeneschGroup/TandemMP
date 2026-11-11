"""Tests for tandemmp.pipeline.loader.

Pure helper functions are exercised directly with synthetic data; ``load_mp_file``
and pipeline-backed event finding are tested via bypasses/fakes to avoid a real
``open_iscat`` dependency.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift

from tandemmp.core.classifications import ProcessingMethod
from tandemmp.core.parameters import (
    DriftCorrectionParams,
    ExperimentalPsfParams,
    get_parameters,
)
from tandemmp.events import PeakImage, RatiometricEvent
from tandemmp.fitting.gaussian_psf import GaussianPsfModel
from tandemmp.fitting.psf import PsfModel
from tandemmp.pipeline.loader import (
    Loader,
    bin_movie,
    build_first_pass_events,
    compute_post_binning_factor,
    correct_movements,
    determine_experimental_psf_model,
    filter_events_by_lid,
    filter_events_by_mass,
    find_lid_open_exclusion_range,
    get_peak,
    mark_overlaps,
)
from tandemmp.pipeline.open_iscat_pipeline import OpenIscatPipeline
from tandemmp.pipeline.pipeline_interface import PipelineResults

# Fixtures / helpers


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


def _make_bump(model: PsfModel, width: int = 9, c: float = -0.05) -> np.ndarray:
    return model.psf_2d_offset(width, 0.0, 0.0, c)


def _make_drifting_movie(
    n_frames: int = 10,
    size: int = 31,
    drift_per_frame: tuple[float, float] = (0.4, -0.3),
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(size, size)).astype(float)
    yy, xx = np.mgrid[:size, :size]
    for cx, cy, amp in [(10, 12, 8.0), (20, 18, -6.0)]:
        base += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 4.0)
    dxs = np.arange(n_frames) * drift_per_frame[0]
    dys = np.arange(n_frames) * drift_per_frame[1]
    return np.array([shift(base, (dy, dx)) for dx, dy in zip(dxs, dys, strict=True)])


def _event_stub(
    *, i: int, navg: int, x_slice: slice, y_slice: slice, event_id: int
) -> SimpleNamespace:
    return SimpleNamespace(
        i=i,
        navg=navg,
        overlaps=[],
        peak=SimpleNamespace(cutout=[y_slice, x_slice]),
        _ID=event_id,
    )


def _build_mismatched_event(
    l_thum: int,
    midpoint: tuple[int, int],
    frame_i: int,
    true_sigma: float,
    narrow_model: PsfModel,
    size: int = 40,
) -> tuple[RatiometricEvent, np.ndarray]:
    """Build an event whose data (wide gaussian) mismatches *narrow_model*.

    Used to force a large ``basic_fit_error`` so
    ``determine_experimental_psf_model`` has room to improve on it.
    """
    coords = np.arange(size)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    mx, my = midpoint
    true_bump = -0.05 * np.exp(
        -(((xx - mx) ** 2 + (yy - my) ** 2) / (2 * true_sigma**2))
    )
    ratiometric = np.zeros((frame_i + 1, size, size))
    ratiometric[frame_i] = true_bump

    peak = get_peak(ratiometric, x=mx, y=my, i=frame_i, l_thum=l_thum)
    event = RatiometricEvent(peak, fit=True, navg=5, peak_model=narrow_model)
    return event, ratiometric


# bin_movie / compute_post_binning_factor


class TestBinMovie:
    def test_averages_full_windows(self):
        arr = np.arange(7 * 2 * 2, dtype=float).reshape(7, 2, 2)
        result = bin_movie(arr, binning=2, dtype=np.float64)
        # indices = arange(0, 5, 2) = [0, 2, 4] -> 3 windows, last frame (6) dropped
        assert result.shape == (3, 2, 2)
        np.testing.assert_allclose(result[0], arr[0:2].mean(axis=0))
        np.testing.assert_allclose(result[1], arr[2:4].mean(axis=0))
        np.testing.assert_allclose(result[2], arr[4:6].mean(axis=0))

    def test_drops_last_full_window_on_exact_multiple(self):
        """Matches the legacy ``utils.bin_array`` quirk: when the
        array length is an exact multiple of *binning*, the final window is
        dropped too (``arange(0, len - binning, binning)`` excludes it)."""
        arr = np.arange(6 * 2 * 2, dtype=float).reshape(6, 2, 2)
        result = bin_movie(arr, binning=2, dtype=np.float64)
        assert result.shape == (2, 2, 2)

    def test_dtype_cast(self):
        arr = np.ones((4, 2, 2)) * 3.7
        result = bin_movie(arr, binning=2, dtype=np.uint16)
        assert result.dtype == np.uint16


class TestComputePostBinningFactor:
    def test_basic(self):
        result = compute_post_binning_factor(
            raw_fps=1000, pre_binning=2, navg=5, target_integration_time=0.1
        )
        assert result == 10

    def test_clamped_to_one(self):
        result = compute_post_binning_factor(
            raw_fps=1000, pre_binning=1, navg=1, target_integration_time=0.0001
        )
        assert result == 1


# get_peak


class TestGetPeak:
    def test_normal_cutout(self):
        ratiometric = np.arange(5 * 20 * 20, dtype=float).reshape(5, 20, 20)
        peak = get_peak(ratiometric, x=10, y=10, i=2, l_thum=5)
        assert peak.midpoint == (10, 10)
        assert peak.cutout == [slice(8, 13), slice(8, 13)]
        np.testing.assert_array_equal(peak.image, ratiometric[2, 8:13, 8:13])
        assert peak.method == ProcessingMethod.ratiometric
        assert peak.i == 2

    def test_clamped_at_low_x(self):
        ratiometric = np.zeros((3, 20, 20))
        peak = get_peak(ratiometric, x=1, y=10, i=0, l_thum=5)
        assert peak.cutout[1] == slice(0, 5)

    def test_clamped_at_high_y(self):
        ratiometric = np.zeros((3, 20, 20))
        peak = get_peak(ratiometric, x=10, y=19, i=0, l_thum=5)
        assert peak.cutout[0] == slice(15, 20)

    def test_nanopore_flips_sign(self):
        ratiometric = np.ones((3, 20, 20)) * 2.0
        peak = get_peak(ratiometric, x=10, y=10, i=0, l_thum=5, nanopore=True)
        np.testing.assert_allclose(peak.image, -2.0)


# build_first_pass_events


class TestBuildFirstPassEvents:
    def test_filters_by_navg_and_sets_navg(self):
        ratiometric = np.zeros((10, 30, 30))
        xs = np.array([10, 15, 20])
        ys = np.array([10, 15, 20])
        indices = np.array([1, 6, 8])
        events = build_first_pass_events(
            ratiometric,
            xs,
            ys,
            indices,
            navg=5,
            l_thum=9,
            peak_model=PsfModel(),
            parallel=False,
        )
        assert len(events) == 2
        assert {e.i for e in events} == {6, 8}
        assert all(e.navg == 5 for e in events)


# find_lid_open_exclusion_range / filter_events_by_lid


class TestFindLidOpenExclusionRange:
    def test_no_dip_returns_empty(self):
        raw_movie = np.ones((10, 5, 5)) * 100
        result = find_lid_open_exclusion_range(raw_movie, navg=2, threshold=1.0)
        assert result == set()

    def test_dip_returns_expected_range(self):
        raw_movie = np.ones((10, 5, 5)) * 100
        raw_movie[4:7] = 0
        result = find_lid_open_exclusion_range(
            raw_movie, navg=2, threshold=1.0, margin=1
        )
        # dip at raw frames 4,5,6 -> shifted by navg*2=4 -> 8,9,10
        assert result == set(range(8 - 1, 10 + 1))


class TestFilterEventsByLid:
    def test_empty_exclusion_keeps_all(self):
        events = [SimpleNamespace(i=1), SimpleNamespace(i=2)]
        kept, excluded = filter_events_by_lid(events, set())
        assert kept == events
        assert excluded == []

    def test_splits_by_exclusion_range(self):
        events = [SimpleNamespace(i=1), SimpleNamespace(i=5), SimpleNamespace(i=10)]
        kept, excluded = filter_events_by_lid(events, {5, 6, 7})
        assert [e.i for e in kept] == [1, 10]
        assert [e.i for e in excluded] == [5]


# correct_movements


class TestCorrectMovements:
    def test_no_merge_when_far_apart(self, spline_model, monkeypatch):
        monkeypatch.setattr("tandemmp.events._base.get_psf_model", lambda: spline_model)
        ratiometric = np.zeros((10, 40, 40))
        bump = _make_bump(spline_model)
        peak1 = PeakImage(
            image=bump,
            method=ProcessingMethod.ratiometric,
            midpoint=(10, 10),
            cutout=[slice(6, 15), slice(6, 15)],
            i=5,
        )
        peak2 = PeakImage(
            image=bump,
            method=ProcessingMethod.ratiometric,
            midpoint=(30, 30),
            cutout=[slice(26, 35), slice(26, 35)],
            i=5,
        )
        e1 = RatiometricEvent(peak1, fit=True, navg=5, peak_model=spline_model)
        e2 = RatiometricEvent(peak2, fit=True, navg=5, peak_model=spline_model)
        result = correct_movements([e1, e2], ratiometric, l_thum=9, navg=5)
        assert result == [e1, e2]

    def test_merges_close_events(self, spline_model, monkeypatch):
        monkeypatch.setattr("tandemmp.events._base.get_psf_model", lambda: spline_model)
        ratiometric = np.zeros((10, 40, 40))
        bump = _make_bump(spline_model)
        peak1 = PeakImage(
            image=bump,
            method=ProcessingMethod.ratiometric,
            midpoint=(10, 10),
            cutout=[slice(6, 15), slice(6, 15)],
            i=5,
        )
        peak2 = PeakImage(
            image=bump,
            method=ProcessingMethod.ratiometric,
            midpoint=(11, 11),
            cutout=[slice(7, 16), slice(7, 16)],
            i=5,
        )
        e1 = RatiometricEvent(peak1, fit=True, navg=5, peak_model=spline_model)
        e2 = RatiometricEvent(peak2, fit=True, navg=5, peak_model=spline_model)
        result = correct_movements([e1, e2], ratiometric, l_thum=9, navg=5)
        assert len(result) == 1
        merged = result[0]
        assert isinstance(merged, RatiometricEvent)
        assert merged.navg == 5
        assert merged.i == 5


# mark_overlaps


class TestMarkOverlaps:
    def test_spatial_only_overlap_marks_non_temporal(self):
        a = _event_stub(
            i=0, navg=5, x_slice=slice(0, 13), y_slice=slice(0, 13), event_id=0
        )
        b = _event_stub(
            i=20, navg=5, x_slice=slice(5, 18), y_slice=slice(5, 18), event_id=1
        )
        mark_overlaps([a, b])
        assert len(a.overlaps) == 1
        ov = a.overlaps[0]
        assert ov.event is b
        assert ov.temporal is False
        assert ov.frames == 15
        assert b.overlaps == []

    def test_spatiotemporal_overlap_marks_temporal_with_frame_intersection(self):
        a = _event_stub(
            i=0, navg=5, x_slice=slice(0, 13), y_slice=slice(0, 13), event_id=0
        )
        b = _event_stub(
            i=3, navg=5, x_slice=slice(5, 18), y_slice=slice(5, 18), event_id=1
        )
        mark_overlaps([a, b])
        assert len(a.overlaps) == 1
        ov = a.overlaps[0]
        assert ov.event is b
        assert ov.temporal is True
        assert ov.frames == sorted(set(range(-5, 6)) & set(range(-2, 9)))

    def test_no_spatial_overlap_no_marks(self):
        a = _event_stub(
            i=0, navg=5, x_slice=slice(0, 13), y_slice=slice(0, 13), event_id=0
        )
        b = _event_stub(
            i=1, navg=5, x_slice=slice(50, 63), y_slice=slice(50, 63), event_id=1
        )
        mark_overlaps([a, b])
        assert a.overlaps == []
        assert b.overlaps == []


# filter_events_by_mass


class TestFilterEventsByMass:
    @staticmethod
    def _stub(m: float, snr: float, image: np.ndarray | None = None) -> SimpleNamespace:
        if image is None:
            image = np.zeros((3, 3))
        return SimpleNamespace(m=m, snr=snr, peak=SimpleNamespace(image=image))

    def test_keeps_large_mass_and_snr(self):
        e = self._stub(m=-1000, snr=1.0)
        kept, discarded = filter_events_by_mass([e], minimum_mass=50, snr_limit=0.2)
        assert kept == [e]
        assert discarded == []

    def test_discards_small_mass(self):
        e = self._stub(m=-10, snr=1.0)
        kept, discarded = filter_events_by_mass([e], minimum_mass=50, snr_limit=0.2)
        assert kept == []
        assert discarded == [e]

    def test_discards_low_snr(self):
        e = self._stub(m=-1000, snr=0.01)
        kept, discarded = filter_events_by_mass([e], minimum_mass=50, snr_limit=0.2)
        assert kept == []
        assert discarded == [e]

    def test_high_signal_image_overrides_filter(self):
        image = np.zeros((3, 3))
        image[1, 1] = -0.01
        e = self._stub(m=-10, snr=0.01, image=image)
        kept, discarded = filter_events_by_mass([e], minimum_mass=50, snr_limit=0.2)
        assert kept == [e]
        assert discarded == []


# determine_experimental_psf_model


def _small_exp_psf_params(**overrides):
    defaults = {
        "sampling_points": 1.0,
        "selection_percentile": 0.0,
        "spatial_expansion": 0,
        "report": False,
    }
    defaults.update(overrides)
    return ExperimentalPsfParams(**defaults)


class TestDetermineExperimentalPsfModel:
    def test_no_events_returns_none_model(self):
        params = get_parameters()
        peak_model = PsfModel()
        result_model, exp_model, refit_model = determine_experimental_psf_model(
            movie=np.zeros((3, 10, 10)),
            ratiometric=np.zeros((3, 10, 10)),
            events=[],
            parameters=params,
            peak_model=peak_model,
        )
        assert result_model is peak_model
        assert exp_model is None
        assert refit_model is None

    def test_disabled_experimental_psf_model_returns_none(self):
        params = get_parameters(experimental_psf_model=False)
        peak_model = PsfModel()
        result_model, exp_model, refit_model = determine_experimental_psf_model(
            movie=np.zeros((3, 10, 10)),
            ratiometric=np.zeros((3, 10, 10)),
            events=[SimpleNamespace()],
            parameters=params,
            peak_model=peak_model,
        )
        assert result_model is peak_model
        assert exp_model is None
        assert refit_model is None

    @staticmethod
    def _run(peak_model, use_rust=False, **exp_overrides):
        l_thum = 9
        event, ratiometric = _build_mismatched_event(
            l_thum, (20, 20), 2, true_sigma=4.0, narrow_model=peak_model
        )
        params = get_parameters(
            l_thum=l_thum,
            experimental_psf_model=_small_exp_psf_params(**exp_overrides),
            use_rust=use_rust,
        )
        out = determine_experimental_psf_model(
            movie=ratiometric,
            ratiometric=ratiometric,
            events=[event],
            parameters=params,
            peak_model=peak_model,
        )
        return event, out

    @pytest.mark.parametrize("use_rust", [False, True])
    def test_adopts_refit_model_when_mismatched(self, use_rust):
        narrow = GaussianPsfModel(sigma=1.0)
        event, (result, exp_model, refit) = self._run(narrow, use_rust=use_rust)
        assert exp_model is not None
        assert exp_model.has_spline_model
        assert result is refit
        assert result is not exp_model
        assert result.psf_params["sigma"] == pytest.approx(4.0, rel=0.05)
        assert event.peak_model is result

    def test_keeps_old_model_when_refit_is_worse(self, monkeypatch):
        import tandemmp.pipeline.loader as loader_mod

        bad = GaussianPsfModel(sigma=0.3)
        monkeypatch.setattr(
            loader_mod, "refit_psf_parameters", lambda *a, **k: (bad, None, 0.0)
        )
        narrow = GaussianPsfModel(sigma=1.0)
        event, (result, exp_model, refit) = self._run(narrow)
        assert result is narrow
        assert refit is bad
        assert event.peak_model is narrow

    def test_refit_disabled(self):
        narrow = GaussianPsfModel(sigma=1.0)
        _, (result, exp_model, refit) = self._run(
            narrow, attempt_parametric_refit=False
        )
        assert result is narrow
        assert exp_model is not None
        assert refit is None


# Loader


class _FakeEventsFound:
    """Minimal EventsFound double — no peaks found."""

    def get_X(self) -> np.ndarray:  # noqa: N802
        return np.array([])

    def get_Y(self) -> np.ndarray:  # noqa: N802
        return np.array([])

    def get_I(self) -> np.ndarray:  # noqa: N802
        return np.array([], dtype=int)

    def get_C(self) -> np.ndarray:  # noqa: N802
        return np.array([])


class _FakePipelineRunner:
    """Minimal PipelineRunner double, exercising Loader's full event-finding
    path without needing ``open_iscat``."""

    def __init__(self, movie: np.ndarray) -> None:
        self.movie = movie
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        filepath: str | Path,
        *,
        frame_bypass: np.ndarray | None = None,
        parallel: bool = True,
        **kwargs: object,
    ) -> PipelineResults:
        self.calls.append(
            {
                "filepath": filepath,
                "frame_bypass": frame_bypass,
                "parallel": parallel,
                **kwargs,
            }
        )
        return PipelineResults(
            frames=self.movie, rframes=self.movie, events=_FakeEventsFound()
        )


class TestLoader:
    def test_just_movie_with_special_bypass_no_drift_correction(self):
        movie = np.arange(5 * 4 * 4, dtype=np.uint16).reshape(5, 4, 4)
        params = get_parameters(drift_correction=False)
        loader = Loader(
            "fake/path.mp",
            special_bypass=movie,
            parameters=params,
            just_movie=True,
        )
        np.testing.assert_array_equal(loader.movie, movie)
        assert loader.params is None
        assert loader._fps is None
        assert loader.vector is None
        assert repr(loader) == f"Loaded movie {Path('fake/path.mp')} - ? events"

    def test_frame_start_end_slicing(self):
        movie = np.arange(10 * 4 * 4, dtype=np.uint16).reshape(10, 4, 4)
        params = get_parameters(drift_correction=False)
        loader = Loader(
            "fake/path.mp",
            special_bypass=movie,
            parameters=params,
            just_movie=True,
            frame_start=2,
            frame_end=6,
        )
        np.testing.assert_array_equal(loader.movie, movie[2:6])

    def test_drift_correction_true_bool_raises_type_error(self):
        movie = np.zeros((5, 10, 10), dtype=np.uint16)
        params = get_parameters(drift_correction=True)
        with pytest.raises(TypeError):
            Loader(
                "fake/path.mp",
                special_bypass=movie,
                parameters=params,
                just_movie=True,
            )

    def test_drift_correction_sets_vector(self):
        movie = _make_drifting_movie()
        params = get_parameters(
            drift_correction=DriftCorrectionParams(frame_gap=2, z_drift=False),
            parallel=False,
            lid_open_threshold=0.0,
        )
        loader = Loader(
            "fake/path.mp",
            special_bypass=movie,
            parameters=params,
            just_movie=True,
            seed=42,
        )
        assert loader.vector is not None
        assert loader.movie.shape[0] == movie.shape[0]
        assert loader.movie.shape[1] <= movie.shape[1]
        assert loader.movie.shape[2] <= movie.shape[2]

    def test_defaults_to_open_iscat_pipeline(self):
        movie = np.arange(5 * 4 * 4, dtype=np.uint16).reshape(5, 4, 4)
        params = get_parameters(drift_correction=False)
        loader = Loader(
            "fake/path.mp",
            special_bypass=movie,
            parameters=params,
            just_movie=True,
        )
        assert isinstance(loader.pipeline_runner, OpenIscatPipeline)

    def test_uses_injected_peak_model(self):
        movie = np.zeros((5, 8, 8), dtype=np.uint16)
        model = GaussianPsfModel(sigma=2.0)
        loader = Loader(
            "fake/path.mp",
            special_bypass=movie,
            parameters=get_parameters(drift_correction=False),
            just_movie=True,
            peak_model=model,
        )
        assert loader.peak_model is model

    def test_full_analysis_uses_injected_pipeline_runner(self):
        movie = np.zeros((5, 8, 8), dtype=np.uint16)
        fake_runner = _FakePipelineRunner(movie.astype(float))
        params = get_parameters(drift_correction=False, lid_open_threshold=0.0)
        loader = Loader(
            "fake/path.mp",
            special_bypass=movie,
            parameters=params,
            pipeline_runner=fake_runner,
        )
        assert loader.pipeline_runner is fake_runner
        assert loader.events == []
        assert len(fake_runner.calls) == 1
        call = fake_runner.calls[0]
        assert call["frame_bypass"] is not None
        assert call["navg"] == params.first_analysis.navg
        assert call["l_thum"] == params.first_analysis.l_thum
