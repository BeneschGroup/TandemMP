"""Tests for tandemmp.plotting.animation — animation and mosaic utilities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for CI

from tandemmp.plotting.animation import (
    create_mosaic_movie,
    create_raw_mosaic,
    reconstruct_particle_movie,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _movie(n_frames: int = 10, h: int = 20, w: int = 20) -> np.ndarray:
    return (RNG.random((n_frames, h, w)) + 1.0).astype(float)


def _make_peak(
    image: np.ndarray, row: int, col: int, size: int, frame: int
) -> MagicMock:
    """Minimal mock of a peak-image-like object."""
    peak = MagicMock()
    peak.image = image
    peak.cutout = [slice(row, row + size), slice(col, col + size)]
    peak.i = frame
    return peak


def _make_event(
    image: np.ndarray, row: int, col: int, size: int, frame: int
) -> MagicMock:
    ev = MagicMock()
    ev.peak = _make_peak(image, row, col, size, frame)
    ev.i = frame
    return ev


def _make_particle(
    events: list[MagicMock],
    unbinding_frame: int | None = None,
    frame_data: dict[int, Any] | None = None,
) -> MagicMock:
    """Mock particle with fixed_background_events and __getitem__."""
    p = MagicMock()
    p.fixed_background_events = events
    p.unbinding_frame = unbinding_frame

    data = frame_data or {}

    def getitem(frame: int) -> Any:
        return data.get(frame)

    p.__getitem__ = MagicMock(side_effect=getitem)
    return p


def _make_experiment(
    movie: np.ndarray,
    particles: list[MagicMock],
    events: list[MagicMock] | None = None,
) -> MagicMock:
    exp = MagicMock()
    exp.movie = movie
    exp.particles = particles
    exp.events = events or []
    return exp


# ---------------------------------------------------------------------------
# create_raw_mosaic
# ---------------------------------------------------------------------------


class TestCreateRawMosaic:
    def test_returns_same_shape_as_movie(self) -> None:
        movie = _movie()
        exp = _make_experiment(movie, particles=[])
        result = create_raw_mosaic(exp)
        assert result.shape == movie.shape

    def test_no_particles_returns_nan_array(self) -> None:
        movie = _movie()
        exp = _make_experiment(movie, particles=[])
        result = create_raw_mosaic(exp)
        assert np.all(np.isnan(result))

    def test_stamps_event_image_into_correct_position(self) -> None:
        movie = np.full((10, 20, 20), fill_value=np.nan)
        size = 3
        patch = np.ones((size, size)) * 0.05
        ev0 = _make_event(patch, row=5, col=7, size=size, frame=2)
        ev1 = _make_event(patch, row=5, col=7, size=size, frame=6)
        # two events in one particle — ev0 spans frames 2..6, ev1 spans 6..end
        particle = _make_particle([ev0, ev1])
        exp = _make_experiment(movie, particles=[particle])

        result = create_raw_mosaic(exp)

        # frame 2 should have the patch at [5:8, 7:10]
        np.testing.assert_allclose(result[2, 5:8, 7:10], 0.05)

    def test_uses_provided_new_movie(self) -> None:
        movie = _movie()
        provided = np.zeros_like(movie)
        exp = _make_experiment(movie, particles=[])
        result = create_raw_mosaic(exp, new_movie=provided)
        assert result is provided

    def test_cumulative_zeros_before_adding(self) -> None:
        """In cumulative mode, the region is zeroed before the event image is added."""
        movie = np.full((10, 20, 20), fill_value=1.0)
        size = 3
        patch = np.ones((size, size)) * 0.05
        ev = _make_event(patch, row=0, col=0, size=size, frame=0)
        particle = _make_particle([ev])
        exp = _make_experiment(movie, particles=[particle])

        new_movie = np.full_like(movie, fill_value=np.nan)
        result = create_raw_mosaic(exp, new_movie=new_movie, cumulative=True)
        # The region should contain the patch value, not nan or the original 1.0
        np.testing.assert_allclose(result[0, 0:3, 0:3], 0.05)

    def test_multiple_particles(self) -> None:
        movie = np.full((10, 30, 30), fill_value=np.nan)
        size = 4
        patch_a = np.ones((size, size)) * 0.1
        patch_b = np.ones((size, size)) * 0.2

        ev_a = _make_event(patch_a, row=0, col=0, size=size, frame=1)
        ev_b = _make_event(patch_b, row=20, col=20, size=size, frame=1)
        p_a = _make_particle([ev_a])
        p_b = _make_particle([ev_b])

        exp = _make_experiment(movie, particles=[p_a, p_b])
        result = create_raw_mosaic(exp)

        np.testing.assert_allclose(result[1, 0:4, 0:4], 0.1)
        np.testing.assert_allclose(result[1, 20:24, 20:24], 0.2)


# ---------------------------------------------------------------------------
# create_mosaic_movie
# ---------------------------------------------------------------------------


class TestCreateMosaicMovie:
    def test_returns_same_shape_as_movie(self) -> None:
        movie = _movie()
        exp = _make_experiment(movie, particles=[])
        result = create_mosaic_movie(exp)
        assert result.shape == movie.shape

    def test_no_particles_returns_nan_array(self) -> None:
        movie = _movie()
        exp = _make_experiment(movie, particles=[])
        result = create_mosaic_movie(exp)
        assert np.all(np.isnan(result))

    def test_without_corrections_stamps_from_first_attempt(self) -> None:
        movie = np.full((10, 20, 20), fill_value=np.nan)
        size = 3
        patch = np.ones((size, size)) * 0.03

        sub_event = MagicMock()
        sub_event.peak = _make_peak(patch, row=2, col=2, size=size, frame=3)
        sub_event.i = 3

        ev = MagicMock()
        ev._first_attempt = [sub_event]

        exp = _make_experiment(movie, particles=[], events=[ev])
        result = create_mosaic_movie(exp, corrections=False)

        # Without corrections, stamps from sub_event.i onwards
        np.testing.assert_allclose(result[3:, 2:5, 2:5], 0.03)

    def test_without_corrections_skips_none_first_attempt(self) -> None:
        movie = np.full((10, 20, 20), fill_value=np.nan)
        ev = MagicMock()
        ev._first_attempt = None
        exp = _make_experiment(movie, particles=[], events=[ev])
        result = create_mosaic_movie(exp, corrections=False)
        assert np.all(np.isnan(result))

    def test_with_corrections_uses_history(self) -> None:
        movie = np.full((10, 20, 20), fill_value=np.nan)
        size = 3
        patch = np.ones((size, size)) * 0.07

        ev = MagicMock()
        ev.peak = _make_peak(patch, row=1, col=1, size=size, frame=0)

        particle = MagicMock()
        particle.history.event_map.connected.items.return_value = [(0, ev)].copy()
        exp = _make_experiment(movie, particles=[particle])
        result = create_mosaic_movie(exp, corrections=True)

        # frames 0 to None (end) should carry the patch
        np.testing.assert_allclose(result[0, 1:4, 1:4], 0.07)

    def test_with_corrections_skips_empty_history(self) -> None:
        """Particles with empty history raise ValueError from np.transpose; should skip."""
        movie = _movie()
        particle = MagicMock()
        particle.history.event_map.connected.items.return_value = []
        exp = _make_experiment(movie, particles=[particle])
        result = create_mosaic_movie(exp, corrections=True)
        assert np.all(np.isnan(result))

    def test_add_mode_accumulates(self) -> None:
        movie = np.full((10, 20, 20), fill_value=np.nan)
        size = 2
        patch = np.ones((size, size)) * 0.1

        ev = MagicMock()
        ev.peak = _make_peak(patch, row=0, col=0, size=size, frame=0)

        particle = MagicMock()
        particle.history.event_map.connected.items.return_value = [(0, ev)]
        exp = _make_experiment(movie, particles=[particle])

        provided = np.zeros_like(movie)
        result = create_mosaic_movie(
            exp, corrections=True, new_movie=provided, add=True
        )
        # patch should be added to the initial 0.0
        np.testing.assert_allclose(result[0, 0:2, 0:2], 0.1)


# ---------------------------------------------------------------------------
# reconstruct_particle_movie
# ---------------------------------------------------------------------------


class TestReconstructParticleMovie:
    def test_returns_correct_shape(self) -> None:
        movie = _movie(n_frames=5, h=10, w=10)
        exp = _make_experiment(movie, particles=[])
        result = reconstruct_particle_movie(exp)
        assert result.shape == movie.shape

    def test_no_particles_gives_zeros(self) -> None:
        movie = _movie(n_frames=5, h=8, w=8)
        exp = _make_experiment(movie, particles=[])
        result = reconstruct_particle_movie(exp)
        np.testing.assert_array_equal(result, 0.0)

    def test_particle_with_no_data_gives_zeros(self) -> None:
        """A particle whose __getitem__ always returns None contributes nothing."""
        movie = _movie(n_frames=3, h=6, w=6)
        particle = _make_particle(events=[], frame_data={})
        exp = _make_experiment(movie, particles=[particle])
        result = reconstruct_particle_movie(exp)
        np.testing.assert_array_equal(result, 0.0)

    def test_particle_data_affects_output(self) -> None:
        """A particle with non-None frame data should produce non-zero output."""
        from scipy.interpolate import RectBivariateSpline

        from tandemmp.fitting.psf import PsfModel

        # Build a spline-based PSF model so open_iscat is not required
        grid_size = 101
        half = grid_size // 2
        coords = np.linspace(-half, half, grid_size)
        x, y = np.meshgrid(coords, coords)
        z = np.exp(-(x**2 + y**2) / (2 * 2.0**2))
        spline = RectBivariateSpline(coords, coords, z)

        model = PsfModel()
        model.set_spline_model(spline)

        with patch("tandemmp.plotting.animation.get_psf_model", return_value=model):
            movie = _movie(n_frames=3, h=10, w=10)
            data = MagicMock()
            data.x = 5.0
            data.y = 5.0
            data.c = -0.05

            particle = _make_particle(events=[], frame_data={0: data, 1: data, 2: data})
            exp = _make_experiment(movie, particles=[particle])
            result = reconstruct_particle_movie(exp)
            # At least one frame should be non-zero
            assert np.any(result != 0.0)
