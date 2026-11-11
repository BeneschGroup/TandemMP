"""Tests for tandemmp.events — event containers and event types."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from tandemmp.core.classifications import EvClassification, ProcessingMethod
from tandemmp.core.conversions import c_to_mass
from tandemmp.events import (
    AbstractEvent,
    Connectable,
    Correction,
    GhostEvent,
    PeakImage,
    RatiometricEvent,
    SubtractionEvent,
    SubtractionSeries,
    get_ghost_model,
)
from tandemmp.fitting.psf import PsfModel

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
    noise_std: float = 0.0,
) -> PeakImage:
    """Create a synthetic PeakImage from a PsfModel."""
    image = model.psf_2d_offset(width, dx, dy, c)
    if noise_std > 0:
        rng = np.random.default_rng(42)
        image = image + rng.normal(0, noise_std, image.shape)
    mid = (100, 100)
    cutout = [
        slice(mid[0] - width // 2, mid[0] + width // 2 + 1),
        slice(mid[1] - width // 2, mid[1] + width // 2 + 1),
    ]
    return PeakImage(
        image=image,
        method=method,
        midpoint=mid,
        cutout=cutout,
        i=frame,
    )


# Correction


class TestCorrection:
    def test_frozen(self):
        c = Correction(
            sliced_image=np.zeros((5, 5)),
            orig_image=np.ones((5, 5)),
            particle=0,
            frame_requested=10,
            background=False,
        )
        with pytest.raises(AttributeError):
            c.particle = 1  # type: ignore[misc]

    def test_fields(self):
        img = np.zeros((5, 5))
        c = Correction(
            sliced_image=img,
            orig_image=img,
            particle=3,
            frame_requested=20,
            background=True,
            data_from=15,
        )
        assert c.particle == 3
        assert c.frame_requested == 20
        assert c.background is True
        assert c.data_from == 15

    def test_default_data_from(self):
        c = Correction(
            sliced_image=None,
            orig_image=np.zeros((3, 3)),
            particle=0,
            frame_requested=5,
            background=False,
        )
        assert c.data_from is None


# PeakImage


class TestPeakImage:
    def test_frozen(self):
        pi = PeakImage(
            image=np.zeros((5, 5)),
            method=ProcessingMethod.ratiometric,
            midpoint=(50, 50),
            cutout=[slice(48, 53), slice(48, 53)],
            i=10,
        )
        with pytest.raises(AttributeError):
            pi.i = 20  # type: ignore[misc]

    def test_as_reduced_memory(self):
        pi = PeakImage(
            image=np.ones((5, 5), dtype=np.float64),
            method=ProcessingMethod.ratiometric,
            midpoint=(50, 50),
            cutout=[slice(48, 53), slice(48, 53)],
            i=10,
        )
        reduced = pi.as_reduced_memory()
        assert reduced.image.dtype == np.float16
        assert reduced.i == 10
        assert reduced.method == ProcessingMethod.ratiometric

    def test_optional_fields_default_none(self):
        pi = PeakImage(
            image=np.zeros((5, 5)),
            method=ProcessingMethod.subtraction,
            midpoint=(10, 20),
            cutout=[slice(0, 5)],
            i=0,
        )
        assert pi.original is None
        assert pi.background is None
        assert pi.corrections is None


# Connectable


class TestConnectable:
    def test_initial_state(self):
        c = Connectable()
        assert c.has_connection is False
        assert c.connected_events == []

    def test_connect_bidirectional(self):
        a, b = Connectable(), Connectable()
        a.connect(b)
        assert a.has_connection is True
        assert b.has_connection is True
        assert b in a.connected_events
        assert a in b.connected_events

    def test_connect_one_way(self):
        a, b = Connectable(), Connectable()
        a.connect(b, do_back=False)
        assert a.has_connection is True
        assert b.has_connection is False
        assert b in a.connected_events
        assert a not in b.connected_events

    def test_connect_rejects_non_connectable(self):
        c = Connectable()
        with pytest.raises(TypeError, match="must inherit"):
            c.connect("not_connectable")  # type: ignore[arg-type]

    def test_connect_rejects_duplicate(self):
        a, b = Connectable(), Connectable()
        a.connect(b)
        with pytest.raises(ValueError, match="already connected"):
            a.connect(b)

    def test_disconnect_all(self):
        a, b, c = Connectable(), Connectable(), Connectable()
        a.connect(b)
        b.connect(c)
        a.disconnect_all()
        assert a.has_connection is False
        assert b.has_connection is False
        assert c.has_connection is False
        assert a.connected_events == []

    def test_disconnect_chain(self):
        nodes = [Connectable() for _ in range(5)]
        for i in range(4):
            nodes[i].connect(nodes[i + 1])
        nodes[0].disconnect_all()
        for n in nodes:
            assert n.has_connection is False
            assert n.connected_events == []


# AbstractEvent


class TestAbstractEvent:
    def test_creates_with_fit(self, spline_model):
        peak = _make_peak_image(spline_model)
        event = AbstractEvent(peak, fit=True, peak_model=spline_model)
        assert hasattr(event, "dx")
        assert hasattr(event, "dy")
        assert hasattr(event, "c")
        assert hasattr(event, "m")
        assert hasattr(event, "_basic_fit")

    def test_creates_without_fit(self, spline_model):
        peak = _make_peak_image(spline_model)
        event = AbstractEvent(peak, fit=False, peak_model=spline_model)
        assert event.c == 0.0

    def test_frame_index(self, spline_model):
        peak = _make_peak_image(spline_model, frame=42)
        event = AbstractEvent(peak, fit=False, peak_model=spline_model)
        assert event.i == 42

    def test_basic_fit_recovers_contrast(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.04)
        event = AbstractEvent(peak, fit=True, peak_model=spline_model)
        assert abs(event.c - (-0.04)) < 0.005

    def test_basic_fit_recovers_position(self, spline_model):
        # Centred peak, so the result doesn't depend on the initial guess.
        peak = _make_peak_image(spline_model, dx=0.0, dy=0.0, c=-0.04)
        event = AbstractEvent(peak, fit=True, peak_model=spline_model)
        assert abs(event.dx) < 0.5
        assert abs(event.dy) < 0.5

    def test_snr_positive(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.05)
        event = AbstractEvent(peak, fit=True, peak_model=spline_model)
        assert event.snr > 0

    def test_noise_nonnegative(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.05)
        event = AbstractEvent(peak, fit=True, peak_model=spline_model)
        assert event.noise >= 0

    def test_basic_replicate_fit_shapes(self, spline_model):
        peak = _make_peak_image(spline_model)
        event = AbstractEvent(peak, fit=True, peak_model=spline_model)
        fit_im, residual = event._basic_replicate_fit()
        assert fit_im.shape == peak.image.shape
        assert residual.shape == peak.image.shape

    def test_repr_unfitted(self, spline_model):
        peak = _make_peak_image(spline_model)
        event = AbstractEvent(peak, fit=False, peak_model=spline_model)
        assert "Not fitted" in repr(event)

    def test_mass_consistent_with_contrast(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.04)
        event = AbstractEvent(peak, fit=True, peak_model=spline_model)
        expected_m = float(c_to_mass(event.c))
        assert abs(event.m - expected_m) < 0.1

    def test_connectable_inheritance(self, spline_model):
        peak = _make_peak_image(spline_model)
        event = AbstractEvent(peak, fit=False, peak_model=spline_model)
        assert isinstance(event, Connectable)
        assert event.has_connection is False


# RatiometricEvent


class TestRatiometricEvent:
    def test_rejects_wrong_method(self, spline_model):
        peak = _make_peak_image(spline_model, method=ProcessingMethod.subtraction)
        with pytest.raises(ValueError, match="ratiometric"):
            RatiometricEvent(peak, fit=False, peak_model=spline_model)

    def test_creates_with_fit(self, spline_model):
        peak = _make_peak_image(spline_model)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        assert hasattr(ev, "c")
        assert ev.navg == 5
        assert ev.classification is None

    def test_self_classify_binding(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.05)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        ev.self_classify()
        assert ev.classification == EvClassification.Binding
        assert ev.c < 0

    def test_self_classify_unbinding(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.05)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        ev.c = 0.05  # force positive contrast
        ev.self_classify()
        assert ev.classification == EvClassification.Unbinding

    def test_unique_ids(self, spline_model):
        peak = _make_peak_image(spline_model)
        ev1 = RatiometricEvent(peak, fit=False, peak_model=spline_model)
        ev2 = RatiometricEvent(peak, fit=False, peak_model=spline_model)
        assert ev1._ID != ev2._ID

    def test_equality(self, spline_model):
        peak = _make_peak_image(spline_model)
        ev = RatiometricEvent(peak, fit=False, peak_model=spline_model)
        assert ev == ev
        ev2 = RatiometricEvent(peak, fit=False, peak_model=spline_model)
        assert ev != ev2

    def test_frame_range(self, spline_model):
        peak = _make_peak_image(spline_model, frame=50)
        ev = RatiometricEvent(peak, fit=False, navg=5, peak_model=spline_model)
        fr = ev.frame_range
        assert fr == set(range(45, 56))

    def test_particle_i_binding(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.05, frame=50)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        ev.self_classify()
        assert ev.particle_i == 51  # binding: i + 1

    def test_particle_i_unbinding(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.05, frame=50)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        ev.c = 0.05
        ev.self_classify()
        assert ev.particle_i == 49  # unbinding: i - 1

    def test_fit_3d_false_by_default(self, spline_model):
        peak = _make_peak_image(spline_model)
        ev = RatiometricEvent(peak, fit=False, peak_model=spline_model)
        assert ev.fit_3D is False

    def test_particle_c_scalar(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.04)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        assert isinstance(ev.particle_c, float)

    def test_particle_m_consistent(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.04)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        expected = float(c_to_mass(ev.particle_c))
        assert abs(ev.particle_m - expected) < 0.1

    def test_particle_x_scalar(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.04)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        assert isinstance(ev.particle_x, float)
        assert ev.particle_x == ev.x

    def test_particle_y_scalar(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.04)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        assert isinstance(ev.particle_y, float)
        assert ev.particle_y == ev.y

    def test_particle_x_reduces_array_to_last_element(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.04)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        ev.x = np.array([1.0, 2.0, 3.0])
        assert ev.particle_x == 3.0

    def test_particle_y_reduces_array_to_last_element(self, spline_model):
        peak = _make_peak_image(spline_model, c=-0.04)
        ev = RatiometricEvent(peak, fit=True, peak_model=spline_model)
        ev.y = np.array([1.0, 2.0, 3.0])
        assert ev.particle_y == 3.0

    def test_temporal_overlaps_empty(self, spline_model):
        peak = _make_peak_image(spline_model)
        ev = RatiometricEvent(peak, fit=False, peak_model=spline_model)
        assert ev.temporal_overlaps == []

    def test_recalculate_exclusion_zone_no_overlaps(self, spline_model):
        peak = _make_peak_image(spline_model)
        ev = RatiometricEvent(peak, fit=False, navg=2, peak_model=spline_model)
        zone = ev.recalculate_exclusion_zone([])
        assert zone.shape[0] == 2 * 2 + 1  # 2*navg + 1 frames
        assert not zone.any()

    def test_hashable(self, spline_model):
        peak = _make_peak_image(spline_model)
        ev = RatiometricEvent(peak, fit=False, peak_model=spline_model)
        s = {ev}
        assert ev in s


# SubtractionEvent


class TestSubtractionEvent:
    def test_rejects_wrong_method(self, spline_model):
        peak = _make_peak_image(spline_model, method=ProcessingMethod.ratiometric)
        with pytest.raises(ValueError, match="subtraction"):
            SubtractionEvent(peak, subtracted_i=40, fit=False, peak_model=spline_model)

    def test_creates_with_fit(self, spline_model):
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.subtraction, c=-0.05, frame=50
        )
        ev = SubtractionEvent(peak, subtracted_i=45, fit=True, peak_model=spline_model)
        assert hasattr(ev, "c")
        assert ev.bleed_time == 5
        assert ev.subtracted_i == 45
        assert ev.navg == 0

    def test_creates_without_fit(self, spline_model):
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.subtraction, frame=50
        )
        ev = SubtractionEvent(peak, subtracted_i=48, fit=False, peak_model=spline_model)
        assert ev._is_bad is False
        assert ev.interp_dx is None
        assert ev.interp_dy is None
        assert ev.bleed_time == 2

    def test_bad_fit_positive_contrast(self, spline_model):
        """A positive contrast should mark the event as bad."""
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.subtraction, c=-0.05, frame=50
        )
        ev = SubtractionEvent(peak, subtracted_i=45, fit=True, peak_model=spline_model)
        # Force positive contrast and re-test
        ev.c = 0.05
        ev.test_bad()
        assert ev._is_bad is True

    def test_mean_c_computed(self, spline_model):
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.subtraction, c=-0.05, frame=50
        )
        ev = SubtractionEvent(peak, subtracted_i=45, fit=False, peak_model=spline_model)
        assert isinstance(ev.mean_c, float)

    def test_particle_m(self, spline_model):
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.subtraction, c=-0.05, frame=50
        )
        ev = SubtractionEvent(peak, subtracted_i=45, fit=True, peak_model=spline_model)
        assert ev.particle_m == ev.m

    def test_reduce_memory(self, spline_model):
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.subtraction, c=-0.05, frame=50
        )
        ev = SubtractionEvent(peak, subtracted_i=45, fit=False, peak_model=spline_model)
        assert ev.peak.image.dtype == np.float64
        ev.reduce_memory()
        assert ev.peak.image.dtype == np.float16

    def test_fit_with_exclusion(self, spline_model):
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.subtraction, c=-0.05, frame=50
        )
        exclusion = np.zeros(peak.image.shape, dtype=bool)
        exclusion[0, :] = True  # exclude top row
        ev = SubtractionEvent(
            peak,
            subtracted_i=45,
            fit=True,
            exclusion=exclusion,
            peak_model=spline_model,
        )
        assert ev.exc_fit is True
        assert hasattr(ev, "c")

    def test_fit_with_large_exclusion_falls_back(self, spline_model):
        """When exclusion covers >= 50%, basic_fit is used instead."""
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.subtraction, c=-0.05, frame=50
        )
        exclusion = np.ones(peak.image.shape, dtype=bool)  # 100% excluded
        ev = SubtractionEvent(
            peak,
            subtracted_i=45,
            fit=True,
            exclusion=exclusion,
            peak_model=spline_model,
        )
        assert ev.exc_fit is False


# SubtractionSeries


class TestSubtractionSeries:
    def _make_series(self, spline_model, n=5, fit=True):
        events = []
        for frame in range(50, 50 + n):
            peak = _make_peak_image(
                spline_model,
                method=ProcessingMethod.subtraction,
                c=-0.05,
                frame=frame,
            )
            ev = SubtractionEvent(
                peak, subtracted_i=frame - 5, fit=fit, peak_model=spline_model
            )
            events.append(ev)
        return SubtractionSeries(events)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SubtractionSeries([])

    def test_length(self, spline_model):
        series = self._make_series(spline_model, n=7)
        assert len(series) == 7

    def test_iteration(self, spline_model):
        series = self._make_series(spline_model, n=3)
        items = list(series)
        assert len(items) == 3
        assert all(isinstance(e, SubtractionEvent) for e in items)

    def test_indexing(self, spline_model):
        series = self._make_series(spline_model, n=5)
        first = series[0]
        assert isinstance(first, SubtractionEvent)
        last = series[-1]
        assert isinstance(last, SubtractionEvent)

    def test_get_by_frame(self, spline_model):
        series = self._make_series(spline_model, n=5)
        ev = series.get_by_frame(52)
        assert ev is not None
        assert ev.i == 52

    def test_get_by_frame_missing(self, spline_model):
        series = self._make_series(spline_model, n=5)
        assert series.get_by_frame(999) is None

    def test_smooth_dx_shape(self, spline_model):
        series = self._make_series(spline_model, n=5)
        assert series.smooth_dx.shape == (5,)

    def test_smooth_dy_shape(self, spline_model):
        series = self._make_series(spline_model, n=5)
        assert series.smooth_dy.shape == (5,)

    def test_reduce_memory(self, spline_model):
        series = self._make_series(spline_model, n=3)
        series.reduce_memory()
        for ev in series:
            assert ev.peak.image.dtype == np.float16

    def test_connectable(self, spline_model):
        series = self._make_series(spline_model, n=3)
        assert isinstance(series, Connectable)
        assert series.has_connection is False

    def test_append_original(self, spline_model):
        s1 = self._make_series(spline_model, n=3)
        s2 = self._make_series(spline_model, n=3)
        s1.append_original(s2)
        assert s1.original is s2

    def test_ghost_signal_heuristic(self, spline_model):
        series = self._make_series(spline_model, n=5)
        sig = series.ghost_signal(min_mass=100.0)
        assert isinstance(sig, float)
        assert 0.0 <= sig <= 1.0

    def test_avg_residual_noise(self, spline_model):
        series = self._make_series(spline_model, n=5)
        noise = series.avg_residual_noise
        assert isinstance(noise, float)
        assert noise >= 0

    def test_unreliability(self, spline_model):
        series = self._make_series(spline_model, n=5)
        u = series.unreliability(EvClassification.Complex)
        assert isinstance(u, float)


# GhostEvent


class TestGhostEvent:
    def test_rejects_wrong_method(self, spline_model):
        peak = _make_peak_image(spline_model, method=ProcessingMethod.ratiometric)
        with pytest.raises(ValueError, match="inferred"):
            GhostEvent(peak, peak_fit=(0.0, 0.0, -0.05))

    def test_creates_correctly(self, spline_model):
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.inferred, c=-0.05, frame=30
        )
        ev = GhostEvent(peak, peak_fit=(0.1, -0.2, -0.04))
        assert ev.dx == 0.1
        assert ev.dy == -0.2
        assert ev.c == -0.04
        assert ev.i == 30
        assert ev.classification == EvClassification.Binding
        assert ev.navg == 0
        assert ev.fit_3D is False

    def test_position_from_midpoint(self, spline_model):
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.inferred, frame=30
        )
        ev = GhostEvent(peak, peak_fit=(1.0, 2.0, -0.03))
        assert ev.x == peak.midpoint[0] + 1.0
        assert ev.y == peak.midpoint[1] + 2.0

    def test_mass_from_contrast(self, spline_model):
        peak = _make_peak_image(spline_model, method=ProcessingMethod.inferred)
        ev = GhostEvent(peak, peak_fit=(0.0, 0.0, -0.05))
        expected = float(c_to_mass(-0.05))
        assert abs(ev.m - expected) < 0.1
        assert ev.particle_m == ev.m

    def test_particle_properties(self, spline_model):
        peak = _make_peak_image(
            spline_model, method=ProcessingMethod.inferred, frame=20
        )
        ev = GhostEvent(peak, peak_fit=(0.0, 0.0, -0.05))
        assert ev.particle_c == -0.05
        assert ev.particle_i == 20

    def test_no_basic_fit(self, spline_model):
        peak = _make_peak_image(spline_model, method=ProcessingMethod.inferred)
        ev = GhostEvent(peak, peak_fit=(0.0, 0.0, -0.05))
        assert ev._basic_fit is None
        assert ev.basic_fit_error is None


# Ghost model loader


class TestGetGhostModel:
    def test_model_file_exists(self):
        """The GP_model.h5 file should be bundled in resources."""
        import importlib.resources

        ref = importlib.resources.files("tandemmp.resources").joinpath("GP_model.h5")
        with importlib.resources.as_file(ref) as path:
            assert path.exists()
            assert path.stat().st_size > 0

    def test_lazy_import_no_tensorflow_at_module_level(self):
        """Importing tandemmp.events should not import tensorflow.

        Runs in a fresh subprocess so the check is deterministic regardless
        of whether an earlier test in this session already imported
        tensorflow. PYTHONPATH is set explicitly
        since pytest's `pythonpath = ["src"]` setting (pyproject.toml)
        doesn't propagate to a subprocess.
        """
        src_dir = Path(__file__).resolve().parents[2] / "src"
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = os.pathsep.join(filter(None, [str(src_dir), existing_pythonpath]))
        env = {**os.environ, "PYTHONPATH": pythonpath}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tandemmp.events; "
                "assert 'tensorflow' not in __import__('sys').modules",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.skipif(
        importlib.util.find_spec("tensorflow") is None,
        reason="TensorFlow not installed",
    )
    def test_loads_model(self):
        """If TF is available, the model should load and be cached."""
        get_ghost_model.cache_clear()
        model = get_ghost_model()
        assert model is not None
        # Second call should return same object (caching)
        assert get_ghost_model() is model


# Public API re-exports


class TestPublicAPI:
    def test_all_classes_importable(self):
        from tandemmp.events import (
            AbstractEvent,
            Connectable,
            Correction,
            GhostEvent,
            PeakImage,
            RatiometricEvent,
            SubtractionEvent,
            SubtractionSeries,
            get_ghost_model,
        )

        assert all(
            cls is not None
            for cls in [
                AbstractEvent,
                Connectable,
                Correction,
                GhostEvent,
                PeakImage,
                RatiometricEvent,
                SubtractionEvent,
                SubtractionSeries,
                get_ghost_model,
            ]
        )

    def test_dunder_all(self):
        import tandemmp.events

        assert hasattr(tandemmp.events, "__all__")
        assert "RatiometricEvent" in tandemmp.events.__all__
        assert "SubtractionEvent" in tandemmp.events.__all__
        assert "GhostEvent" in tandemmp.events.__all__
        assert "get_ghost_model" in tandemmp.events.__all__
