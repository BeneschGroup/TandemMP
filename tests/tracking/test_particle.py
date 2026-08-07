"""Tests for tandemmp.tracking.particle — Particle and GhostParticle."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from tandemmp.core.classifications import EvClassification, ProcessingMethod
from tandemmp.events import (
    GhostEvent,
    PeakImage,
    RatiometricEvent,
    SubtractionEvent,
)
from tandemmp.fitting.psf import PsfModel
from tandemmp.tracking.particle import (
    EventMap,
    GhostParticle,
    HistoryEntry,
    MassChangeWarning,
    Particle,
    _reset_particle_ids,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


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
    """Reset module-level ID counters before each test."""
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


def _make_ghost(
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


# ===========================================================================
# HistoryEntry
# ===========================================================================


class TestHistoryEntry:
    def test_fields(self):
        h = HistoryEntry(i=10, x=1.0, y=2.0, m=100.0, c=-0.05, sigma=0.5, data_from=10)
        assert h.i == 10
        assert h.x == 1.0
        assert h.m == 100.0
        assert h.data_from == 10

    def test_frozen(self):
        h = HistoryEntry(i=0, x=0.0, y=0.0, m=0.0, c=0.0, sigma=None, data_from=0)
        with pytest.raises(AttributeError):
            h.x = 5.0  # type: ignore[misc]

    def test_sigma_optional(self):
        h = HistoryEntry(i=0, x=0.0, y=0.0, m=0.0, c=0.0, sigma=None, data_from=0)
        assert h.sigma is None


# ===========================================================================
# EventMap
# ===========================================================================


class TestEventMap:
    def test_initial_state(self):
        em = EventMap()
        assert em.ratiometric == {}
        assert em.connected == {}
        assert em.disconnected == {}

    def test_repr_counts(self):
        em = EventMap()
        em.ratiometric[5] = object()  # type: ignore[assignment]
        text = repr(em)
        assert "Ratiometric 1" in text
        assert "Connected 0" in text
        assert "Disconnected 0" in text

    def test_get_ratiometric_returns_none_when_empty(self):
        em = EventMap()
        assert em.get_ratiometric(5) is None

    def test_get_connected_returns_latest_before_frame(self):
        em = EventMap()
        e1 = object()
        e2 = object()
        em.connected[10] = e1  # type: ignore[assignment]
        em.connected[20] = e2  # type: ignore[assignment]
        assert em.get_connected(25) is e2
        assert em.get_connected(15) is e1
        assert em.get_connected(5) is None

    def test_get_disconnected_independent_of_connected(self):
        em = EventMap()
        em.disconnected[7] = "d"  # type: ignore[assignment]
        em.connected[8] = "c"  # type: ignore[assignment]
        assert em.get_disconnected(10) == "d"
        assert em.get_connected(10) == "c"


# ===========================================================================
# Particle construction
# ===========================================================================


class TestParticleConstruction:
    def test_creates_from_binding_event(self, spline_model):
        ev = _make_binding_ratiometric(spline_model, frame=50)
        p = Particle(ev)
        assert p.binding_frame == 50
        assert p.binding_event is ev
        assert p.unbinding_frame is None
        assert p.on_slide is True
        assert p.strikes == 0
        assert p.rejected == []

    def test_rejects_non_binding_event(self, spline_model):
        ev = _make_binding_ratiometric(spline_model)
        # Force the classification away from binding
        ev.classification = EvClassification.Unbinding
        with pytest.raises(ValueError, match="Binding"):
            Particle(ev)

    def test_sequential_ids(self, spline_model):
        p1 = Particle(_make_binding_ratiometric(spline_model, frame=10))
        p2 = Particle(_make_binding_ratiometric(spline_model, frame=20))
        assert p1.ID == 0
        assert p2.ID == 1

    def test_overwrite_id(self, spline_model):
        ev = _make_binding_ratiometric(spline_model)
        p = Particle(ev, overwrite_id=42)
        assert p.ID == 42

    def test_preferred_background_offset(self, spline_model):
        ev = _make_binding_ratiometric(spline_model, frame=100)
        p = Particle(ev)
        assert p.preferred_background == 97

    def test_adds_binding_event_to_map(self, spline_model):
        ev = _make_binding_ratiometric(spline_model, frame=50)
        p = Particle(ev)
        assert 50 in p._event_map.ratiometric
        assert p._event_map.ratiometric[50] is ev

    def test_sets_particle_back_reference(self, spline_model):
        ev = _make_binding_ratiometric(spline_model, frame=50)
        p = Particle(ev)
        assert ev.particle == p.ID

    def test_initial_tracking_flags(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model))
        assert p.possible_ghost is False
        assert p.is_possible_split is False
        assert p.split_mass is None
        assert p.lost_tracking is False


# ===========================================================================
# Identity / equality / repr
# ===========================================================================


class TestParticleIdentity:
    def test_equality_by_id(self, spline_model):
        p1 = Particle(_make_binding_ratiometric(spline_model, frame=1))
        p2 = Particle(_make_binding_ratiometric(spline_model, frame=2))
        assert p1 == p1
        assert p1 != p2

    def test_not_equal_to_other_type(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model))
        assert (p == "not a particle") is False

    def test_hashable(self, spline_model):
        p1 = Particle(_make_binding_ratiometric(spline_model, frame=1))
        p2 = Particle(_make_binding_ratiometric(spline_model, frame=2))
        s = {p1, p2}
        assert len(s) == 2

    def test_repr_contains_id(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model))
        assert str(p.ID) in repr(p)


# ===========================================================================
# was_bound / on_slide / residency_time
# ===========================================================================


class TestWasBound:
    def test_bound_after_binding_frame(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=50))
        assert p.was_bound(51) is True

    def test_not_bound_on_binding_frame(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=50))
        # Strict less-than: not yet bound on the frame itself.
        assert p.was_bound(50) is False

    def test_not_bound_after_unbinding(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        p.unbinding_frame = 20
        assert p.was_bound(15) is True
        assert p.was_bound(21) is False

    def test_on_slide_toggles_with_unbinding_frame(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model))
        assert p.on_slide is True
        p.unbinding_frame = 99
        assert p.on_slide is False

    def test_residency_time_nan_while_on_slide(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        assert np.isnan(p.residency_time)

    def test_residency_time_after_unbinding(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        p.unbinding_frame = 35
        assert p.residency_time == 25


# ===========================================================================
# __getitem__
# ===========================================================================


class TestParticleGetItem:
    def test_none_before_binding(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=50))
        assert p[10] is None

    def test_returns_history_entry(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=50))
        entry = p[55]
        assert isinstance(entry, HistoryEntry)
        assert entry.i == 55
        assert entry.data_from == 50

    def test_history_entry_has_mass(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=50, c=-0.05))
        entry = p[55]
        assert entry is not None
        assert entry.m > 0
        assert entry.sigma is not None

    def test_none_after_unbinding(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        p.unbinding_frame = 20
        assert p[21] is None


# ===========================================================================
# get_best_event
# ===========================================================================


class TestGetBestEvent:
    def test_none_when_not_bound(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=50))
        assert p.get_best_event(10) is None

    def test_falls_back_to_binding_ratiometric(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=50))
        ev = p.get_best_event(60)
        assert ev is p.binding_event

    def test_prefers_disconnected_subtraction_event(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        sub = _make_subtraction(spline_model, frame=30, subtracted_i=25)
        p._event_map.disconnected[sub.i] = sub
        assert p.get_best_event(31) is sub

    def test_disconnected_only_skips_ratiometric(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        assert p.get_best_event(20, disconnected_only=True) is None


# ===========================================================================
# choose_background
# ===========================================================================


class TestChooseBackground:
    def test_returns_preferred_when_no_trajectory(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=50))
        # No fixed events → trajectory falls back to binding event
        out = p.choose_background(55, midpoint=(100, 100), l_thum=5, psf_radius=2)
        assert out == p.preferred_background

    def test_returns_clear_frame_when_distant(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        sub = _make_subtraction(
            spline_model, frame=40, subtracted_i=35, midpoint=(200, 200)
        )
        p._event_map.disconnected[sub.i] = sub
        # Midpoint at origin → sub is very far away → suitable
        out = p.choose_background(frame=50, midpoint=(0, 0), l_thum=10, psf_radius=2)
        assert out == sub.i


# ===========================================================================
# Event addition / fixed_background_events / trajectory
# ===========================================================================


class TestFixedBackgroundEvents:
    def test_empty_by_default(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model))
        assert p.fixed_background_events == []

    def test_sorted_by_frame(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        e1 = _make_subtraction(spline_model, frame=30, subtracted_i=25)
        e2 = _make_subtraction(spline_model, frame=20, subtracted_i=15)
        p._event_map.disconnected[e1.i] = e1
        p._event_map.disconnected[e2.i] = e2
        evs = p.fixed_background_events
        assert [e.i for e in evs] == [20, 30]

    def test_fixed_data_returns_history_entries(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        e = _make_subtraction(spline_model, frame=20, subtracted_i=15)
        p._event_map.disconnected[e.i] = e
        data = p.fixed_data
        assert len(data) == 1
        assert isinstance(data[0], HistoryEntry)
        assert data[0].i == 20

    def test_last_continuation_is_latest(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        e1 = _make_subtraction(spline_model, frame=20, subtracted_i=15)
        e2 = _make_subtraction(spline_model, frame=30, subtracted_i=25)
        p._event_map.disconnected[e1.i] = e1
        p._event_map.disconnected[e2.i] = e2
        assert p.last_continuation is e2


class TestTrajectory:
    def test_fallback_to_binding_event(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=50))
        (xs, ys), frames = p.trajectory
        assert len(xs) == 1
        assert len(frames) == 1
        assert frames[0] == 50

    def test_uses_fixed_events_when_available(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        e = _make_subtraction(spline_model, frame=20, subtracted_i=15)
        p._event_map.disconnected[e.i] = e
        (xs, ys), frames = p.trajectory
        assert 20 in frames.tolist()


class TestEventsProperty:
    def test_returns_ratiometric_events(self, spline_model):
        ev = _make_binding_ratiometric(spline_model, frame=50)
        p = Particle(ev)
        assert ev in p.events


# ===========================================================================
# propose_new_event
# ===========================================================================


class TestProposeNewEvent:
    def test_rejects_when_off_slide(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        p.unbinding_frame = 15
        sub = _make_subtraction(spline_model, frame=20, subtracted_i=15)
        assert p.propose_new_event(sub, minimum_mass=10.0) is False

    def test_accepts_compatible_subtraction(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10, c=-0.05))
        sub = _make_subtraction(spline_model, frame=20, subtracted_i=15, c=-0.05)
        accepted = p.propose_new_event(sub, minimum_mass=1.0)
        assert accepted is True
        assert sub in p._event_map.disconnected.values()

    def test_rejects_when_mass_below_minimum(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10, c=-0.05))
        sub = _make_subtraction(spline_model, frame=20, subtracted_i=15)
        # Force last_known mass below minimum via huge minimum_mass
        accepted = p.propose_new_event(sub, minimum_mass=1e12)
        assert accepted is False
        assert p.unbinding_frame == 20

    def test_records_minimum_mass_rejection(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10, c=-0.05))
        sub = _make_subtraction(spline_model, frame=20, subtracted_i=15)
        p.propose_new_event(sub, minimum_mass=1e12)
        assert sub in p.rejected

    def test_unsupported_event_type_raises(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        ghost = _make_ghost(spline_model, frame=20)
        with pytest.raises(TypeError):
            p.propose_new_event(ghost, minimum_mass=1.0)

    def test_strikes_unbind_after_max(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10, c=-0.05))
        # Already add one fixed event so last_continuation works
        anchor = _make_subtraction(spline_model, frame=15, subtracted_i=12, c=-0.05)
        p._event_map.disconnected[anchor.i] = anchor
        # Drive strikes by rejecting far-away events
        p.strikes = 100
        # Give a "bad" event to trigger the lost_tracking unbinding path
        bad = _make_subtraction(spline_model, frame=25, subtracted_i=20, c=-0.05)
        bad._is_bad = True
        accepted = p.propose_new_event(bad, minimum_mass=1.0, max_strikes=5)
        assert accepted is False
        assert p.lost_tracking is True
        assert p.unbinding_frame is not None


class TestProposeRatiometricMassError:
    def test_warns_on_unexplainable_mass_drop(self, spline_model):
        # Build a heavy binding ratiometric event
        ev = _make_binding_ratiometric(spline_model, frame=10, c=-0.05)
        p = Particle(ev)

        # Create an unbinding ratiometric event whose "proposed" mass
        # (last_known.m - particle_m) is hugely negative.
        unbinder = _make_binding_ratiometric(spline_model, frame=20, c=-0.05)
        unbinder.classification = EvClassification.Unbinding
        # Force particle_m massively larger than last_known.m
        unbinder.c = -5.0  # very negative → huge mass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", MassChangeWarning)
            p.propose_new_event(unbinder, minimum_mass=1.0)
        assert any(issubclass(w.category, MassChangeWarning) for w in caught)


# ===========================================================================
# Spatial queries
# ===========================================================================


class TestSpatialQueries:
    def test_is_close_true_when_same_position(self, spline_model):
        ev = _make_binding_ratiometric(spline_model, frame=10)
        p = Particle(ev)
        # Another event at the same pixel position on a later frame
        later = _make_binding_ratiometric(spline_model, frame=20)
        assert p.is_close(later, threshold=1.0) is True

    def test_is_close_false_when_not_bound(self, spline_model):
        ev = _make_binding_ratiometric(spline_model, frame=50)
        p = Particle(ev)
        probe = _make_binding_ratiometric(spline_model, frame=5)
        assert p.is_close(probe, threshold=100.0) is False

    def test_distance_to_none_when_not_bound(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=50))
        assert p.distance_to(100.0, 100.0, frame=5) is None

    def test_distance_to_zero_for_exact_match(self, spline_model):
        ev = _make_binding_ratiometric(spline_model, frame=10, midpoint=(100, 100))
        p = Particle(ev)
        entry = p[20]
        assert entry is not None
        d = p.distance_to(entry.x, entry.y, 20)
        assert d == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# GhostParticle
# ===========================================================================


class TestGhostParticleConstruction:
    def test_rejects_non_binding_event(self, spline_model):
        ev = _make_ghost(spline_model)
        ev.classification = EvClassification.Unbinding
        with pytest.raises(ValueError, match="Binding"):
            GhostParticle(ev)

    def test_flags_and_defaults(self, spline_model):
        ev = _make_ghost(spline_model, frame=50)
        p = GhostParticle(ev)
        assert p.possible_ghost is True
        assert p.preferred_background == 50
        assert p.binding_frame == 50
        assert p.double_checked is False
        assert p.self_destruct is False
        assert p._bad_corrections == 0

    def test_impression_args_uses_inverse_contrast(self, spline_model):
        ev = _make_ghost(spline_model, c=-0.03)
        p = GhostParticle(ev)
        _, _, bc = p._impression_args
        assert bc == pytest.approx(0.03)

    def test_overwrite_id_accepts_string(self, spline_model):
        ev = _make_ghost(spline_model)
        p = GhostParticle(ev, overwrite_id="synthetic-1")
        assert p.ID == "synthetic-1"

    def test_was_bound_inclusive_at_binding_frame(self, spline_model):
        ev = _make_ghost(spline_model, frame=50)
        p = GhostParticle(ev)
        # Ghost particles are bound starting on their binding frame.
        assert p.was_bound(50) is True
        assert p.was_bound(49) is False


class TestGhostParticleProposeFixed:
    def test_first_subtraction_accepted_unchecked(self, spline_model):
        ev = _make_ghost(spline_model, frame=10, c=-0.05)
        p = GhostParticle(ev)
        sub = _make_subtraction(spline_model, frame=20, subtracted_i=15)
        accepted = p._propose_fixed(
            sub,
            last_known=HistoryEntry(
                i=15, x=100.0, y=100.0, m=1.0, c=-0.01, sigma=1.0, data_from=10
            ),
            minimum_mass=1.0,
        )
        assert accepted is True
        assert sub in p._event_map.disconnected.values()

    def test_propose_fixed_on_off_slide_particle(self, spline_model):
        ev = _make_ghost(spline_model, frame=10, c=-0.05)
        p = GhostParticle(ev)
        # Off-slide ghost particles still reject through propose_new_event
        p.unbinding_frame = 15
        sub = _make_subtraction(spline_model, frame=20, subtracted_i=15)
        assert p.propose_new_event(sub, minimum_mass=1.0) is False

    def test_rejects_spatially_inconsistent_event(self, spline_model):
        ev = _make_ghost(spline_model, frame=10, c=-0.05)
        p = GhostParticle(ev)
        # Add a second ratiometric event so the unchecked first-acceptance path is skipped.
        ghost2 = _make_ghost(spline_model, frame=20, c=-0.05)
        p._event_map.ratiometric[ghost2.i] = ghost2
        # Subtraction event is very far from the particle at (100, 100).
        sub = _make_subtraction(
            spline_model, frame=30, subtracted_i=25, midpoint=(999, 999)
        )
        result = p._propose_fixed(
            sub,
            last_known=HistoryEntry(
                i=20, x=100.0, y=100.0, m=1.0, c=-0.01, sigma=None, data_from=10
            ),
            minimum_mass=0.01,
            spatial_threshold=2.0,
        )
        assert result is False
        assert p.strikes >= 1


# ===========================================================================
# Additional propose_new_event / _propose_ratiometric coverage
# ===========================================================================


class TestProposeRatiometricWithThresholds:
    def test_accepted_when_spatially_and_mass_consistent(self, spline_model):
        binding = _make_binding_ratiometric(spline_model, frame=10, midpoint=(100, 100))
        p = Particle(binding)
        entry = p[15]
        assert entry is not None
        # Continuation event at the same position with similar parameters.
        continuation = _make_binding_ratiometric(
            spline_model, frame=30, midpoint=(100, 100)
        )
        accepted = p._propose_ratiometric(
            continuation,
            last_known=entry,
            minimum_mass=0.1,
            spatial_threshold=5.0,
            mass_fluctuation=1.0,
        )
        assert accepted is True
        assert continuation in p._event_map.ratiometric.values()


class TestCoincidenceChecking:
    def test_raises_when_unbinding_frame_not_set(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        sub = _make_subtraction(spline_model, frame=20, subtracted_i=15)
        with pytest.raises(ValueError, match="coincidence_checking"):
            p.propose_new_event(sub, minimum_mass=1.0, coincidence_checking=True)

    def test_proceeds_for_off_slide_particle(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10, c=-0.05))
        # Add a fixed event so last_continuation works if strikes hit max.
        anchor = _make_subtraction(spline_model, frame=15, subtracted_i=12, c=-0.05)
        p._event_map.disconnected[anchor.i] = anchor
        p.unbinding_frame = 20
        # Without coincidence_checking the off-slide particle immediately rejects.
        sub = _make_subtraction(spline_model, frame=25, subtracted_i=20, c=-0.05)
        assert p.propose_new_event(sub, minimum_mass=1.0) is False
        # With coincidence_checking it gets evaluated (result may be True or False
        # depending on mass/space, but it must not immediately return False).
        p2 = Particle(_make_binding_ratiometric(spline_model, frame=10, c=-0.05))
        p2._event_map.disconnected[anchor.i] = anchor
        p2.unbinding_frame = 20
        # Calling with coincidence_checking must not raise.
        p2.propose_new_event(sub, minimum_mass=1.0, coincidence_checking=True)


class TestLastContinuationEmptyRaises:
    def test_raises_index_error_when_no_fixed_events(self, spline_model):
        p = Particle(_make_binding_ratiometric(spline_model, frame=10))
        with pytest.raises(IndexError):
            _ = p.last_continuation
