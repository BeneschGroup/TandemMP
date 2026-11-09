"""Tests for tandemmp.tracking.manager — ParticleManager."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from tandemmp.core.classifications import ProcessingMethod
from tandemmp.events import (
    GhostEvent,
    PeakImage,
    RatiometricEvent,
    SubtractionEvent,
)
from tandemmp.fitting.psf import PsfModel
from tandemmp.tracking.manager import ParticleManager
from tandemmp.tracking.particle import (
    GhostParticle,
    Particle,
    _reset_particle_ids,
)

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


@pytest.fixture(autouse=True)
def _reset_ids() -> None:
    _reset_particle_ids()


def _make_peak(
    model: PsfModel,
    *,
    method: ProcessingMethod = ProcessingMethod.ratiometric,
    width: int = 13,
    dx: float = 0.0,
    dy: float = 0.0,
    c: float = -0.05,
    frame: int = 50,
    midpoint: tuple[int, int] = (100, 100),
) -> PeakImage:
    image = model.psf_2d_offset(width, dx, dy, c)
    cutout = [
        slice(midpoint[0] - width // 2, midpoint[0] + width // 2 + 1),
        slice(midpoint[1] - width // 2, midpoint[1] + width // 2 + 1),
    ]
    return PeakImage(
        image=image,
        method=method,
        midpoint=midpoint,
        cutout=cutout,
        i=frame,
    )


def _make_binding_ratiometric(
    model: PsfModel,
    *,
    frame: int = 50,
    c: float = -0.05,
    midpoint: tuple[int, int] = (100, 100),
) -> RatiometricEvent:
    peak = _make_peak(model, frame=frame, c=c, midpoint=midpoint)
    ev = RatiometricEvent(peak, fit=True, peak_model=model)
    ev.self_classify()
    return ev


def _make_subtraction(
    model: PsfModel,
    *,
    frame: int,
    subtracted_i: int,
    c: float = -0.05,
    midpoint: tuple[int, int] = (100, 100),
) -> SubtractionEvent:
    peak = _make_peak(
        model,
        method=ProcessingMethod.subtraction,
        frame=frame,
        c=c,
        midpoint=midpoint,
    )
    return SubtractionEvent(
        peak,
        subtracted_i=subtracted_i,
        fit=True,
        peak_model=model,
    )


def _make_ghost_event(
    model: PsfModel,
    *,
    frame: int = 50,
    c: float = -0.05,
    midpoint: tuple[int, int] = (100, 100),
) -> GhostEvent:
    peak = _make_peak(
        model,
        method=ProcessingMethod.inferred,
        frame=frame,
        midpoint=midpoint,
    )
    return GhostEvent(peak, peak_fit=(0.0, 0.0, c))


def _make_particle(
    model: PsfModel,
    *,
    frame: int = 50,
    midpoint: tuple[int, int] = (100, 100),
) -> Particle:
    return Particle(_make_binding_ratiometric(model, frame=frame, midpoint=midpoint))


def _make_ghost_particle(
    model: PsfModel,
    *,
    frame: int = 50,
    midpoint: tuple[int, int] = (100, 100),
) -> GhostParticle:
    return GhostParticle(_make_ghost_event(model, frame=frame, midpoint=midpoint))


# Construction / container protocol


class TestConstruction:
    def test_default_empty(self):
        m = ParticleManager()
        assert len(m) == 0
        assert list(m) == []

    def test_seed_with_particles(self, spline_model):
        p1 = _make_particle(spline_model, frame=10)
        p2 = _make_particle(spline_model, frame=20)
        m = ParticleManager([p1, p2])
        assert len(m) == 2
        assert list(m) == [p1, p2]

    def test_seeded_list_is_copied(self, spline_model):
        p = _make_particle(spline_model, frame=10)
        seed = [p]
        m = ParticleManager(seed)
        seed.append(_make_particle(spline_model, frame=20))
        assert len(m) == 1

    def test_accepts_iterable(self, spline_model):
        p1 = _make_particle(spline_model, frame=10)
        p2 = _make_particle(spline_model, frame=20)
        m = ParticleManager(iter([p1, p2]))
        assert len(m) == 2

    def test_none_defaults_to_empty(self):
        m = ParticleManager(None)
        assert len(m) == 0


class TestContainerProtocol:
    def test_iter_yields_particles(self, spline_model):
        p1 = _make_particle(spline_model, frame=10)
        p2 = _make_particle(spline_model, frame=20)
        m = ParticleManager([p1, p2])
        assert list(iter(m)) == [p1, p2]

    def test_contains(self, spline_model):
        p = _make_particle(spline_model, frame=10)
        m = ParticleManager([p])
        assert p in m

    def test_not_contains_other_particle(self, spline_model):
        p1 = _make_particle(spline_model, frame=10)
        p2 = _make_particle(spline_model, frame=20)
        m = ParticleManager([p1])
        assert p2 not in m

    def test_repr_reports_count(self, spline_model):
        m = ParticleManager([_make_particle(spline_model, frame=10)])
        assert "1" in repr(m)
        assert "ParticleManager" in repr(m)

    def test_get_all_returns_copy(self, spline_model):
        p = _make_particle(spline_model, frame=10)
        m = ParticleManager([p])
        out = m.get_all()
        assert out == [p]
        out.append(_make_particle(spline_model, frame=20))
        assert len(m) == 1


# append


class TestAppend:
    def test_appends_particle(self, spline_model):
        m = ParticleManager()
        p = _make_particle(spline_model, frame=10)
        m.append(p)
        assert len(m) == 1
        assert p in m

    def test_rejects_non_particle(self):
        m = ParticleManager()
        with pytest.raises(TypeError):
            m.append("not a particle")  # type: ignore[arg-type]

    def test_rejects_duplicate(self, spline_model):
        m = ParticleManager()
        p = _make_particle(spline_model, frame=10)
        m.append(p)
        with pytest.raises(ValueError, match="already managed"):
            m.append(p)

    def test_append_ghost_without_nearby(self, spline_model, caplog):
        m = ParticleManager()
        m.append(_make_particle(spline_model, frame=10, midpoint=(500, 500)))
        ghost = _make_ghost_particle(spline_model, frame=20, midpoint=(100, 100))
        with caplog.at_level(logging.INFO):
            m.append(ghost)
        assert caplog.records == []
        assert ghost in m
        assert ghost.is_possible_split is False

    def test_append_ghost_with_nearby_marks_split(self, spline_model):
        m = ParticleManager()
        # Particle bound on frame 10 at (100, 100); was_bound is strict (>),
        # so it will be "present" for any frame > 10.
        p = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        m.append(p)

        ghost = _make_ghost_particle(spline_model, frame=15, midpoint=(100, 100))
        m.append(ghost)

        assert ghost.is_possible_split is True
        assert p.split_mass is not None
        split_frame, _ = p.split_mass
        assert split_frame == 15

    def test_ghost_split_radius_is_tight(self, spline_model):
        m = ParticleManager()
        # Place particle just outside the 0.5 px split radius.
        p = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        m.append(p)

        ghost = _make_ghost_particle(spline_model, frame=15, midpoint=(102, 102))
        m.append(ghost)

        assert ghost.is_possible_split is False
        assert p.split_mass is None


# remove


class TestRemove:
    def test_removes_existing(self, spline_model):
        m = ParticleManager()
        p = _make_particle(spline_model, frame=10)
        m.append(p)
        assert m.remove(p) is True
        assert p not in m
        assert len(m) == 0

    def test_missing_particle_is_noop(self, spline_model, caplog):
        m = ParticleManager()
        p = _make_particle(spline_model, frame=10)
        with caplog.at_level(logging.WARNING):
            result = m.remove(p)
        assert result is False
        assert any("not found" in rec.message for rec in caplog.records)

    def test_double_remove_does_not_raise(self, spline_model):
        m = ParticleManager()
        p = _make_particle(spline_model, frame=10)
        m.append(p)
        m.remove(p)
        assert m.remove(p) is False


# get_present_particles


class TestGetPresentParticles:
    def test_empty_before_any_bind(self, spline_model):
        m = ParticleManager([_make_particle(spline_model, frame=50)])
        assert m.get_present_particles(10) == []

    def test_includes_particles_after_binding(self, spline_model):
        p = _make_particle(spline_model, frame=10)
        m = ParticleManager([p])
        assert m.get_present_particles(20) == [p]

    def test_excludes_particles_after_unbinding(self, spline_model):
        p = _make_particle(spline_model, frame=10)
        p.unbinding_frame = 20
        m = ParticleManager([p])
        assert m.get_present_particles(25) == []

    def test_mixed_particles(self, spline_model):
        p1 = _make_particle(spline_model, frame=10)
        p2 = _make_particle(spline_model, frame=30)
        m = ParticleManager([p1, p2])
        assert m.get_present_particles(20) == [p1]


# get_close_particles


class TestGetCloseParticles:
    def test_returns_nearby_particle(self, spline_model):
        p = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        m = ParticleManager([p])
        close = m.get_close_particles(frame=15, x=100.0, y=100.0, distance=5.0)
        assert close == [p]

    def test_excludes_far_particle(self, spline_model):
        p = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        m = ParticleManager([p])
        close = m.get_close_particles(frame=15, x=200.0, y=200.0, distance=5.0)
        assert close == []

    def test_excludes_particles_not_present_at_frame(self, spline_model):
        p = _make_particle(spline_model, frame=50, midpoint=(100, 100))
        m = ParticleManager([p])
        close = m.get_close_particles(frame=10, x=100.0, y=100.0, distance=5.0)
        assert close == []

    def test_distance_is_exclusive_boundary(self, spline_model):
        p = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        m = ParticleManager([p])
        # Query point exactly at the threshold distance should not match.
        close = m.get_close_particles(frame=15, x=103.0, y=100.0, distance=3.0)
        assert close == []

    def test_multiple_particles_only_close_ones(self, spline_model):
        near = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        far = _make_particle(spline_model, frame=10, midpoint=(300, 300))
        m = ParticleManager([near, far])
        close = m.get_close_particles(frame=15, x=101.0, y=101.0, distance=5.0)
        assert close == [near]


# background_particle_check / particles_in_thumbnail


class TestBackgroundParticleCheck:
    def test_returns_particle_inside_box(self, spline_model):
        p = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        m = ParticleManager([p])
        event = _make_binding_ratiometric(spline_model, frame=20, midpoint=(100, 100))
        suitable = m.background_particle_check(
            event, frame=15, l_thum=20, psf_radius=5.0
        )
        assert suitable == [p]

    def test_excludes_particle_outside_box(self, spline_model):
        p = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        m = ParticleManager([p])
        event = _make_binding_ratiometric(spline_model, frame=20, midpoint=(300, 300))
        suitable = m.background_particle_check(
            event, frame=15, l_thum=20, psf_radius=5.0
        )
        assert suitable == []

    def test_uses_event_peak_midpoint(self, spline_model):
        p = _make_particle(spline_model, frame=10, midpoint=(50, 50))
        m = ParticleManager([p])
        event = _make_binding_ratiometric(spline_model, frame=20, midpoint=(50, 50))
        suitable = m.background_particle_check(
            event, frame=15, l_thum=10, psf_radius=3.0
        )
        assert suitable == [p]


class TestParticlesInThumbnail:
    def test_returns_list_per_frame_in_navg_window(self, spline_model):
        p = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        m = ParticleManager([p])
        event = _make_binding_ratiometric(spline_model, frame=30, midpoint=(100, 100))
        per_frame = m.particles_in_thumbnail(event, l_thum=20, psf_radius=5.0)
        assert len(per_frame) == 2 * event.navg + 1
        for frame_particles in per_frame:
            assert frame_particles == [p]

    def test_particle_not_present_dropped(self, spline_model):
        p = _make_particle(spline_model, frame=100, midpoint=(100, 100))
        m = ParticleManager([p])
        event = _make_binding_ratiometric(spline_model, frame=30, midpoint=(100, 100))
        per_frame = m.particles_in_thumbnail(event, l_thum=20, psf_radius=5.0)
        assert all(lst == [] for lst in per_frame)

    def test_excludes_particle_outside_thumbnail(self, spline_model):
        p = _make_particle(spline_model, frame=10, midpoint=(500, 500))
        m = ParticleManager([p])
        event = _make_binding_ratiometric(spline_model, frame=30, midpoint=(100, 100))
        per_frame = m.particles_in_thumbnail(event, l_thum=20, psf_radius=5.0)
        assert all(lst == [] for lst in per_frame)


# find_best_matches


class TestFindBestMatches:
    def test_empty_when_no_present(self, spline_model):
        m = ParticleManager([_make_particle(spline_model, frame=100)])
        assert m.find_best_matches(x=0.0, y=0.0, i=10, c=0.0) == []

    def test_single_present_particle(self, spline_model):
        p = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        m = ParticleManager([p])
        matches = m.find_best_matches(x=100.0, y=100.0, i=15, c=-0.05)
        assert matches == [p]

    def test_ranks_by_3d_distance(self, spline_model):
        near = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        far = _make_particle(spline_model, frame=10, midpoint=(200, 200))
        m = ParticleManager([far, near])
        matches = m.find_best_matches(x=101.0, y=101.0, i=15, c=-0.05)
        assert matches[0] is near
        assert matches[1] is far

    def test_contrast_component_affects_ordering(self, spline_model):
        # Two particles at the same (x, y), distinguished by contrast.
        p_low = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        p_high = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        # Override the contrast returned from particle[i] by mutating the
        # binding event (used as the fallback entry).
        p_low.binding_event.c = -0.02  # type: ignore[misc]
        p_high.binding_event.c = -0.10  # type: ignore[misc]
        m = ParticleManager([p_low, p_high])

        matches_high = m.find_best_matches(x=100.0, y=100.0, i=15, c=-0.10)
        assert matches_high[0] is p_high

        matches_low = m.find_best_matches(x=100.0, y=100.0, i=15, c=-0.02)
        assert matches_low[0] is p_low

    def test_skips_particles_not_bound_at_queried_frame(self, spline_model):
        p_present = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        p_absent = _make_particle(spline_model, frame=200, midpoint=(101, 101))
        m = ParticleManager([p_absent, p_present])
        matches = m.find_best_matches(x=100.0, y=100.0, i=15, c=-0.05)
        assert matches == [p_present]

    def test_result_only_contains_particles_with_entries(self, spline_model):
        p1 = _make_particle(spline_model, frame=10, midpoint=(100, 100))
        p2 = _make_particle(spline_model, frame=10, midpoint=(150, 150))
        m = ParticleManager([p1, p2])
        matches = m.find_best_matches(x=100.0, y=100.0, i=15, c=-0.05)
        assert set(matches) == {p1, p2}
        assert len(matches) == 2
