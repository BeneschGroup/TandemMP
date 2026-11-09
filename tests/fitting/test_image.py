"""Tests for tandemmp.fitting.image — image correction utilities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from tandemmp.core.classifications import ProcessingMethod
from tandemmp.events._base import Correction, PeakImage
from tandemmp.fitting.image import (
    calculate_overlapping_slice,
    correct_ghost,
    correct_subtraction_fit,
    correct_subtraction_image,
    ghost_prediction_transform,
    remove_artefacts,
)
from tandemmp.fitting.psf import PsfModel, psf_2d_cached

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


def _make_peak_image(
    width: int = 13,
    midpoint: tuple[int, int] = (50, 50),
    frame: int = 10,
    image: np.ndarray | None = None,
) -> PeakImage:
    if image is None:
        image = np.random.default_rng(0).normal(0, 0.001, (width, width))
    return PeakImage(
        image=image,
        method=ProcessingMethod.subtraction,
        midpoint=midpoint,
        cutout=[slice(44, 57), slice(44, 57)],
        i=frame,
    )


def _make_subtraction_event(
    peak: PeakImage,
    subtracted_i: int = 5,
    navg: int = 5,
) -> MagicMock:
    """Create a mock SubtractionEvent with the attributes used by image.py."""
    ev = MagicMock()
    ev.peak = peak
    ev.i = peak.i
    ev.subtracted_i = subtracted_i
    ev.navg = navg
    return ev


def _make_mock_event_data(
    x: float, y: float, c: float, data_from: int | None = None
) -> MagicMock:
    """Minimal object returned by particle[frame]."""
    obj = MagicMock()
    obj.x = x
    obj.y = y
    obj.c = c
    obj.data_from = data_from
    return obj


def _make_mock_particle(
    pid: int,
    events: dict[int, Any],
    *,
    is_possible_split: bool = False,
    binding_frame: int = 0,
    best_events: dict[int, Any] | None = None,
) -> MagicMock:
    """Create a mock Particle with __getitem__ and get_best_event."""
    p = MagicMock()
    p.ID = pid
    p.is_possible_split = is_possible_split
    p.binding_frame = binding_frame

    def getitem(frame: int) -> Any:
        return events.get(frame)

    p.__getitem__ = MagicMock(side_effect=getitem)

    # get_best_event returns an object with .peak.image and .i
    if best_events is None:
        best_events = {}

    def get_best_event(frame: int) -> Any:
        return best_events.get(frame)

    p.get_best_event = MagicMock(side_effect=get_best_event)
    # Make hashable for set() operations
    p.__hash__ = MagicMock(return_value=hash(pid))
    p.__eq__ = MagicMock(side_effect=lambda other: getattr(other, "ID", None) == pid)
    return p


# ghost_prediction_transform


class TestGhostPredictionTransform:
    def test_returns_same_shape(self):
        peak = np.random.default_rng(0).normal(0, 0.01, (13, 13))
        result = ghost_prediction_transform(peak)
        assert result.shape == peak.shape

    def test_zero_input(self):
        peak = np.zeros((7, 7))
        result = ghost_prediction_transform(peak)
        # All-zero distances → nan_normalise produces NaN → result is NaN
        assert np.all(np.isnan(result))

    def test_output_does_not_flip_sign(self):
        """The transform never flips a pixel's sign (factor is non-negative).

        The factor ``1 - nan_normalise(distances)**0.1`` is in [0, 1],
        so the result can be zero but never the opposite sign.
        """
        peak = np.random.default_rng(1).normal(0, 0.01, (5, 5))
        result = ghost_prediction_transform(peak)
        # Where input is positive, output should be >= 0 (never negative)
        assert np.all(result[peak > 0] >= 0)
        # Where input is negative, output should be <= 0 (never positive)
        assert np.all(result[peak < 0] <= 0)

    def test_symmetric_input(self):
        """Symmetric input should produce symmetric output."""
        peak = np.zeros((5, 5))
        peak[2, 2] = -0.05  # centre only
        result = ghost_prediction_transform(peak)
        np.testing.assert_allclose(result, result[::-1, :], atol=1e-10)
        np.testing.assert_allclose(result, result[:, ::-1], atol=1e-10)


# calculate_overlapping_slice


class TestCalculateOverlappingSlice:
    def test_no_overlap(self):
        """Non-overlapping cutouts → None."""
        c1 = (slice(0, 10), slice(0, 10))
        c2 = (slice(20, 30), slice(20, 30))
        assert calculate_overlapping_slice(c1, c2) is None

    def test_full_overlap(self):
        """Identical cutouts → full slice."""
        c = (slice(10, 23), slice(10, 23))
        result = calculate_overlapping_slice(c, c)
        assert result is not None

    def test_partial_overlap_x(self):
        """Cutouts that overlap only in x."""
        c1 = (slice(10, 23), slice(10, 23))
        c2 = (slice(10, 23), slice(15, 28))
        result = calculate_overlapping_slice(c1, c2)
        assert result is not None

    def test_partial_overlap_y(self):
        """Cutouts that overlap only in y."""
        c1 = (slice(10, 23), slice(10, 23))
        c2 = (slice(15, 28), slice(10, 23))
        result = calculate_overlapping_slice(c1, c2)
        assert result is not None

    def test_result_is_list_of_slices(self):
        c1 = (slice(10, 23), slice(10, 23))
        c2 = (slice(12, 25), slice(12, 25))
        result = calculate_overlapping_slice(c1, c2)
        assert result is not None
        assert len(result) == 2
        assert isinstance(result[0], slice)
        assert isinstance(result[1], slice)

    def test_adjacent_no_overlap(self):
        """Cutouts that are exactly adjacent → None."""
        c1 = (slice(0, 10), slice(0, 10))
        c2 = (slice(10, 20), slice(0, 10))
        assert calculate_overlapping_slice(c1, c2) is None

    def test_one_pixel_overlap(self):
        """Cutouts that share exactly one column."""
        c1 = (slice(0, 10), slice(0, 10))
        c2 = (slice(0, 10), slice(8, 18))
        result = calculate_overlapping_slice(c1, c2)
        assert result is not None

    def test_overlap_corner_at_relative_offset_zero(self):
        """Bottom-right of cutout2 landing at cutout1's origin (offset 0).

        When ``x22 - x11 == 0`` (and likewise for y), the relative offset is
        0 — a valid pixel that must produce a slice with stop=1 rather than
        being treated as "out of range".
        """
        # cutout1 starts at (5, 5); cutout2 ends exactly at (5, 5).
        c1 = (slice(5, 13), slice(5, 13))
        c2 = (slice(0, 6), slice(0, 6))
        result = calculate_overlapping_slice(c1, c2)
        assert result is not None
        # Bottom-right corner at relative offset 0 → slice stop == 1.
        assert result[0].stop == 1
        assert result[1].stop == 1


# remove_artefacts


class TestRemoveArtefacts:
    def test_no_corrections(self):
        """With empty corrections list, original image is returned."""
        img = np.ones((13, 13)) * 0.01
        peak = PeakImage(
            image=img,
            method=ProcessingMethod.subtraction,
            midpoint=(50, 50),
            cutout=[slice(44, 57), slice(44, 57)],
            i=10,
            original=img,
            corrections=[],
        )
        result = remove_artefacts(peak)
        np.testing.assert_array_equal(result, img)

    def test_single_correction_removed(self):
        """A single correction (no paired duplicate) should be subtracted."""
        base = np.ones((5, 5)) * 0.1
        correction_img = np.ones((5, 5)) * 0.02
        corr = Correction(
            sliced_image=correction_img,
            orig_image=base,
            particle=1,
            frame_requested=10,
            background=False,
        )
        peak = PeakImage(
            image=base,
            method=ProcessingMethod.subtraction,
            midpoint=(50, 50),
            cutout=[slice(48, 53), slice(48, 53)],
            i=10,
            original=base,
            corrections=[corr],
        )
        result = remove_artefacts(peak)
        np.testing.assert_allclose(result, base - correction_img)

    def test_paired_corrections_excluded(self):
        """Two corrections from the same particle should both be excluded."""
        base = np.ones((5, 5)) * 0.1
        c1_img = np.ones((5, 5)) * 0.03
        c2_img = np.ones((5, 5)) * 0.01
        c1 = Correction(
            sliced_image=c1_img,
            orig_image=base,
            particle=1,
            frame_requested=10,
            background=False,
        )
        c2 = Correction(
            sliced_image=c2_img,
            orig_image=base,
            particle=1,
            frame_requested=5,
            background=True,
        )
        peak = PeakImage(
            image=base,
            method=ProcessingMethod.subtraction,
            midpoint=(50, 50),
            cutout=[slice(48, 53), slice(48, 53)],
            i=10,
            original=base,
            corrections=[c1, c2],
        )
        result = remove_artefacts(peak)
        # Both corrections from particle 1 are excluded (count==2)
        np.testing.assert_array_equal(result, base)


# correct_ghost


class TestCorrectGhost:
    def test_basic_subtraction(self, spline_model):
        """correct_ghost should subtract a ghost PSF from each peak.

        The ghost contrast is ``-c`` (negated real particle contrast).
        correct_ghost generates ``psf_2d_cached(dx, dy, -c, ...)`` and
        subtracts it, removing the ghost impression.
        """
        width = 13
        real_c = -0.05
        # Image contains the ghost impression (contrast = -real_c = 0.05)
        ghost_psf = psf_2d_cached(0.0, 0.0, -real_c, width, spline_model)
        noise = np.random.default_rng(0).normal(0, 0.001, (width, width))
        img = ghost_psf + noise
        peak = _make_peak_image(width=width, image=img)
        result = correct_ghost([peak], 0.0, 0.0, real_c, peak_model=spline_model)
        assert len(result) == 1
        assert result[0].image.shape == (width, width)
        # Corrected peak should have less energy (ghost removed)
        assert np.sum(np.abs(result[0].image)) < np.sum(np.abs(img))

    def test_preserves_metadata(self, spline_model):
        peak = _make_peak_image()
        result = correct_ghost([peak], 0.0, 0.0, -0.01, peak_model=spline_model)
        assert result[0].midpoint == peak.midpoint
        assert result[0].cutout == peak.cutout
        assert result[0].i == peak.i

    def test_save_mode(self, spline_model):
        peak = _make_peak_image()
        result = correct_ghost(
            [peak], 0.0, 0.0, -0.01, peak_model=spline_model, save=True
        )
        assert result[0].corrections is not None
        assert len(result[0].corrections) == 1
        assert result[0].corrections[0].background is True

    def test_multiple_peaks(self, spline_model):
        peaks = [_make_peak_image(frame=i) for i in range(5)]
        result = correct_ghost(peaks, 0.0, 0.0, -0.01, peak_model=spline_model)
        assert len(result) == 5

    def test_series_input(self, spline_model):
        """Should accept list of objects with .peak attribute."""
        peak = _make_peak_image()
        mock_item = MagicMock()
        mock_item.peak = peak
        result = correct_ghost([mock_item], 0.0, 0.0, -0.01, peak_model=spline_model)
        assert len(result) == 1


# correct_subtraction_fit


class TestCorrectSubtractionFit:
    def test_no_particles_returns_original(self, spline_model):
        """With no particles, the original peak is returned."""
        peak = _make_peak_image()
        ev = _make_subtraction_event(peak)
        result = correct_subtraction_fit(ev, particles=None, psf_model=spline_model)
        assert result is peak

    def test_single_particle_correction(self, spline_model):
        width = 13
        midpoint = (50, 50)
        peak = _make_peak_image(width=width, midpoint=midpoint, frame=10)
        ev = _make_subtraction_event(peak, subtracted_i=5)

        # Particle at frame 10 (foreground) only
        fg_data = _make_mock_event_data(50.5, 50.3, -0.03, data_from=10)

        # Create a mock best-event for the particle
        best_ev = MagicMock()
        best_ev.peak = _make_peak_image(width=width)
        best_ev.i = 10

        p = _make_mock_particle(
            pid=1,
            events={10: fg_data, 5: None},
            best_events={10: best_ev},
        )

        result = correct_subtraction_fit(ev, particles=[p], psf_model=spline_model)
        assert isinstance(result, PeakImage)
        assert result.method == ProcessingMethod.subtraction
        assert result.image.shape == (width, width)

    def test_save_mode_preserves_corrections(self, spline_model):
        width = 13
        peak = _make_peak_image(width=width, frame=10)
        ev = _make_subtraction_event(peak, subtracted_i=5)

        fg_data = _make_mock_event_data(50.5, 50.3, -0.03, data_from=10)
        best_ev = MagicMock()
        best_ev.peak = _make_peak_image(width=width)
        best_ev.i = 10

        p = _make_mock_particle(
            pid=1,
            events={10: fg_data, 5: None},
            best_events={10: best_ev},
        )

        result = correct_subtraction_fit(
            ev, particles=[p], save=True, psf_model=spline_model
        )
        assert result.corrections is not None
        assert result.original is not None
        assert result.background is not None


# correct_subtraction_image


class TestCorrectSubtractionImage:
    def test_no_particles_returns_original(self):
        peak = _make_peak_image()
        ev = _make_subtraction_event(peak)
        result = correct_subtraction_image(ev, particles=None)
        assert result is peak

    def test_gdb_raises(self):
        peak = _make_peak_image()
        ev = _make_subtraction_event(peak)
        with pytest.raises(Exception, match="implement"):
            correct_subtraction_image(ev, particles=[], gdb=True)

    def test_returns_peak_image(self):
        """With particles that don't overlap, result is still a PeakImage."""
        peak = _make_peak_image(width=13, frame=10)
        ev = _make_subtraction_event(peak, subtracted_i=5)

        # Particle with no events at the relevant frames
        p = _make_mock_particle(pid=1, events={})
        p.get_best_event = MagicMock(return_value=None)

        result = correct_subtraction_image(ev, particles=[p])
        assert isinstance(result, PeakImage)
        assert result.method == ProcessingMethod.subtraction

    def test_overlapping_particle_applies_correction(self):
        """Overlapping neighbour: foreground image is pasted into the background."""
        width = 13
        # Event peak at midpoint (50, 50) with cutout covering (44:57, 44:57).
        event_image = np.full((width, width), 0.05)
        event_peak = PeakImage(
            image=event_image,
            method=ProcessingMethod.subtraction,
            midpoint=(50, 50),
            cutout=[slice(44, 57), slice(44, 57)],
            i=10,
        )
        ev = _make_subtraction_event(event_peak, subtracted_i=5)

        # Neighbour particle with a cutout shifted by 4 pixels in x.
        neighbour_image = np.full((width, width), 0.02)
        neighbour_peak = PeakImage(
            image=neighbour_image,
            method=ProcessingMethod.subtraction,
            midpoint=(54, 50),
            cutout=[slice(44, 57), slice(48, 61)],
            i=10,
        )
        best_ev = MagicMock()
        best_ev.peak = neighbour_peak
        best_ev.i = 10

        p = _make_mock_particle(
            pid=1,
            events={},
            best_events={10: best_ev},
        )

        result = correct_subtraction_image(ev, particles=[p])
        assert result.background is not None
        # The background should be non-zero where the two cutouts overlap
        # (x columns 4-12 of the event peak = x columns 0-8 of the neighbour).
        assert np.any(result.background != 0)
        # Corrected image = event image - background
        np.testing.assert_allclose(
            result.image,
            event_peak.image - result.background,
        )
        # Corrections list should contain a single foreground entry
        # (particle is not the focus → foreground branch; no background
        # because get_best_event(subtracted_i=5) returns None).
        assert result.corrections is not None
        assert len(result.corrections) == 1
        assert result.corrections[0].background is False
