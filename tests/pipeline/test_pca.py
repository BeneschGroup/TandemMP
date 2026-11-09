"""Tests for tandemmp.pipeline.pca — target API.

Covers ``Experiment`` wiring (with lightweight ``Loader``/``Investigator`` fakes)
and property/method behavior on a bare, minimally-constructed ``Experiment``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from tandemmp.core.classifications import ProcessingMethod
from tandemmp.core.conversions import (
    DEFAULT_CALIBRATION,
    Calibration,
    get_calibration,
    set_calibration,
)
from tandemmp.core.parameters import FullAnalysisParams, get_parameters
from tandemmp.events import GhostEvent, PeakImage, RatiometricEvent
from tandemmp.fitting.psf import PsfModel
from tandemmp.io.save import load_results
from tandemmp.pipeline import pca
from tandemmp.pipeline.investigator import Action, Decision
from tandemmp.pipeline.open_iscat_pipeline import Info
from tandemmp.tracking.manager import ParticleManager
from tandemmp.tracking.particle import GhostParticle, Particle, _reset_particle_ids

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
    midpoint: tuple[int, int] = (25, 25),
) -> PeakImage:
    image = model.psf_2d_offset(width, dx, dy, c)
    half = width // 2
    cutout = [
        slice(midpoint[0] - half, midpoint[0] + half + 1),
        slice(midpoint[1] - half, midpoint[1] + half + 1),
    ]
    return PeakImage(
        image=image,
        method=method,
        midpoint=midpoint,
        cutout=cutout,
        i=frame,
    )


def _make_event(
    model: PsfModel,
    *,
    frame: int = 50,
    c: float = -0.05,
    dx: float = 0.0,
    dy: float = 0.0,
    midpoint: tuple[int, int] = (25, 25),
    navg: int = 3,
) -> RatiometricEvent:
    peak = _make_peak(model, frame=frame, c=c, dx=dx, dy=dy, midpoint=midpoint)
    ev = RatiometricEvent(peak, fit=True, peak_model=model, navg=navg)
    ev.self_classify()
    return ev


def _make_ghost_event(
    model: PsfModel,
    *,
    frame: int = 50,
    c: float = -0.05,
    midpoint: tuple[int, int] = (25, 25),
) -> GhostEvent:
    peak = _make_peak(
        model,
        method=ProcessingMethod.inferred,
        frame=frame,
        midpoint=midpoint,
    )
    return GhostEvent(peak, peak_fit=(0.0, 0.0, c))


def _make_particle(model: PsfModel, **kwargs: Any) -> Particle:
    return Particle(_make_event(model, **kwargs))


def _make_ghost_particle(model: PsfModel, **kwargs: Any) -> GhostParticle:
    return GhostParticle(_make_ghost_event(model, **kwargs))


def _bare_experiment(**attrs: Any) -> pca.Experiment:
    """Construct an ``Experiment`` bypassing ``__init__``.

    Only the attributes under test need to be set, isolating each method
    from the (heavy, ``open_iscat``-dependent) construction path.
    """
    exp = object.__new__(pca.Experiment)
    for key, value in attrs.items():
        setattr(exp, key, value)
    return exp


class _StubInvestigator:
    """Records calls and returns pre-programmed decisions, keyed by event ID."""

    def __init__(self, decisions: dict[int, Decision]) -> None:
        self._decisions = decisions
        self.find_and_fit_calls: list[RatiometricEvent] = []
        self.multi_continuation_calls: list[RatiometricEvent] = []

    def find_and_fit(self, *, event: RatiometricEvent) -> Decision:
        self.find_and_fit_calls.append(event)
        return self._decisions[event._ID]  # noqa: SLF001

    def multi_continuation(self, *, latest_event: RatiometricEvent) -> None:
        self.multi_continuation_calls.append(latest_event)


def _analyse_experiment(
    events: list[RatiometricEvent],
    decisions: dict[int, Decision],
    *,
    particles: ParticleManager | None = None,
    parameters: FullAnalysisParams | None = None,
) -> tuple[pca.Experiment, _StubInvestigator]:
    investigator = _StubInvestigator(decisions)
    exp = _bare_experiment(
        _events=events,
        particles=particles if particles is not None else ParticleManager(),
        investigator=investigator,
        parameters=parameters if parameters is not None else get_parameters(),
        _index=0,
        interesting=[],
    )
    return exp, investigator


class _FakeLoader:
    """Stand-in for ``pipeline.loader.Loader`` capturing constructor kwargs."""

    def __init__(
        self,
        mp_filepath: str | Path,
        *,
        parameters: Any,
        just_movie: bool,
        special_bypass: Any,
        pipeline_runner: Any = None,
    ) -> None:
        self.mp_filepath = mp_filepath
        self.parameters = parameters
        self.just_movie = just_movie
        self.special_bypass = special_bypass
        self.pipeline_runner = pipeline_runner
        self.folder = Path("fake_folder")
        self.movie = np.zeros((3, 4, 4))
        self.params: Info | None = (
            None
            if just_movie
            else Info(
                scatter_interference_destructive=True,
                raw_camera_frame_rate=1000.0,
                frame_binning=1,
                pixel_binning=1,
                full_well_capacity=None,
                grey_levels=None,
            )
        )
        self.events: list[RatiometricEvent] = []
        self.peak_model = MagicMock()
        self.exclusion_range: set[int] = {1, 2}
        self.vector = None
        self.old_mass = None


class _FakeInvestigator:
    """Stand-in for ``pipeline.investigator.Investigator`` capturing init kwargs."""

    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs

    def find_and_fit(self, *, event: RatiometricEvent) -> Decision:
        raise NotImplementedError

    def multi_continuation(self, *, latest_event: RatiometricEvent) -> None:
        raise NotImplementedError


# __init__


class TestInit:
    def test_just_movie_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(pca, "Loader", _FakeLoader)
        exp = pca.Experiment(tmp_path / "movie.mp", just_movie=True)
        assert exp.movie is not None
        assert not hasattr(exp, "particles")
        assert not hasattr(exp, "investigator")

    def test_raises_without_movie_params(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        class _NoParamsLoader(_FakeLoader):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.params = None

        monkeypatch.setattr(pca, "Loader", _NoParamsLoader)
        with pytest.raises(ValueError, match="movie metadata"):
            pca.Experiment(tmp_path / "movie.mp")

    def test_wires_investigator_with_loader_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(pca, "Loader", _FakeLoader)
        monkeypatch.setattr(pca, "Investigator", _FakeInvestigator)

        exp = pca.Experiment(tmp_path / "movie.mp")

        assert isinstance(exp.investigator, _FakeInvestigator)
        kwargs = _FakeInvestigator.last_kwargs
        assert kwargs is not None
        assert kwargs["movie"] is exp.movie
        assert kwargs["movie_params"] is exp.movie_params
        assert kwargs["parameters"] is exp.parameters
        assert kwargs["particle_manager"] is exp.particles
        assert kwargs["peak_model"] is exp.loader.peak_model
        assert sorted(kwargs["open_lid_frames"]) == [1, 2]
        assert kwargs["event_veto_frames"] == []
        assert isinstance(exp.particles, ParticleManager)
        assert exp._index == 0
        assert exp.interesting == []


# events / clean_events


class TestEventsProperty:
    def test_sorted_by_id(self, spline_model: PsfModel) -> None:
        e1 = _make_event(spline_model, frame=10)
        e2 = _make_event(spline_model, frame=20)
        e3 = _make_event(spline_model, frame=30)
        exp = _bare_experiment(_events=[e3, e1, e2])
        assert exp.events == [e1, e2, e3]


class TestCleanEvents:
    def test_filters_none_and_ignored_decisions(self, spline_model: PsfModel) -> None:
        e1 = _make_event(spline_model, frame=10)
        e2 = _make_event(spline_model, frame=20)
        e3 = _make_event(spline_model, frame=30)
        e1.decision = None
        e2.decision = Decision(action=Action.ignore, particle=None)
        e3.decision = Decision(action=Action.create_new, particle=None)
        exp = _bare_experiment(_events=[e1, e2, e3])
        assert exp.clean_events == [e3]


# simple_analysis


class TestSimpleAnalysis:
    def test_converts_events_and_delegates(
        self, monkeypatch: pytest.MonkeyPatch, spline_model: PsfModel
    ) -> None:
        e1 = _make_event(spline_model, frame=10)
        e2 = _make_event(spline_model, frame=20)
        exp = _bare_experiment(_events=[e1, e2])

        captured: dict[str, Any] = {}

        def _fake_match(
            simple_events: Any,
            *,
            sigma_multiplier: float,
            max_pairs: int,
            analyse_distance: float,
        ) -> list[Any]:
            captured["simple_events"] = simple_events
            captured["sigma_multiplier"] = sigma_multiplier
            captured["max_pairs"] = max_pairs
            captured["analyse_distance"] = analyse_distance
            return []

        monkeypatch.setattr(pca, "match_pairs_sigma", _fake_match)
        result = exp.simple_analysis(
            sigma_multiplier=2.0, max_pairs=3, analyse_distance=5.0
        )

        assert result == []
        assert captured["sigma_multiplier"] == 2.0
        assert captured["max_pairs"] == 3
        assert captured["analyse_distance"] == 5.0
        simple = captured["simple_events"]
        assert len(simple) == 2
        assert simple[0].c == pytest.approx(e1.particle_c)
        assert simple[0].m == pytest.approx(e1.particle_m)
        assert simple[0].x == pytest.approx(e1.particle_x)
        assert simple[0].y == pytest.approx(e1.particle_y)
        assert simple[0].i == pytest.approx(e1.i)


# reset


class TestReset:
    def test_refits_recent_events_and_clears_particles(
        self, spline_model: PsfModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events = [_make_event(spline_model, frame=10 * i) for i in range(5)]
        exp = _bare_experiment(_events=events, particles=ParticleManager(), _index=2)

        disconnect_calls: list[RatiometricEvent] = []
        fit_calls: list[RatiometricEvent] = []
        for e in events:
            monkeypatch.setattr(
                e, "disconnect_all", lambda ev=e: disconnect_calls.append(ev)
            )
            monkeypatch.setattr(e, "basic_fit", lambda ev=e: fit_calls.append(ev))

        exp.reset()

        assert disconnect_calls == events[:4]
        assert fit_calls == events[:4]
        assert exp._index == 0
        assert isinstance(exp.particles, ParticleManager)
        assert exp.particles.get_all() == []


# analyse


class TestAnalyseCreateNew:
    def test_uses_particle_supplied_by_decision(self, spline_model: PsfModel) -> None:
        event = _make_event(spline_model, frame=10)
        particle = _make_ghost_particle(spline_model, frame=10)
        decisions = {event._ID: Decision(action=Action.create_new, particle=particle)}
        exp, _ = _analyse_experiment(
            [event], decisions, parameters=get_parameters(particle_continuation=False)
        )

        exp.analyse()

        assert exp.particles.get_all() == [particle]
        assert exp._index == 1

    def test_creates_particle_from_binding_event_above_minimum_mass(
        self, spline_model: PsfModel
    ) -> None:
        event = _make_event(spline_model, frame=10, c=-0.05)
        decisions = {event._ID: Decision(action=Action.create_new, particle=None)}
        exp, _ = _analyse_experiment(
            [event], decisions, parameters=get_parameters(particle_continuation=False)
        )

        exp.analyse()

        particles = exp.particles.get_all()
        assert len(particles) == 1
        assert particles[0].binding_event is event

    def test_below_minimum_mass_is_ignored_without_a_particle(
        self, spline_model: PsfModel
    ) -> None:
        event = _make_event(spline_model, frame=10, c=-0.001)  # mass ~31 < default 50
        decisions = {event._ID: Decision(action=Action.create_new, particle=None)}
        exp, _ = _analyse_experiment(
            [event], decisions, parameters=get_parameters(particle_continuation=False)
        )

        exp.analyse()

        assert exp.particles.get_all() == []
        assert event.decision == Decision(action=Action.ignore, particle=None)

    def test_non_binding_event_without_particle_raises(
        self, spline_model: PsfModel
    ) -> None:
        event = _make_event(spline_model, frame=10, c=0.05)  # -> Unbinding
        decisions = {event._ID: Decision(action=Action.create_new, particle=None)}
        exp, _ = _analyse_experiment(
            [event], decisions, parameters=get_parameters(particle_continuation=False)
        )

        with pytest.raises(RuntimeError, match="non-Binding"):
            exp.analyse()


class TestAnalyseFailed:
    def test_qualifying_failed_event_creates_particle(
        self, spline_model: PsfModel
    ) -> None:
        event = _make_event(spline_model, frame=10, c=-0.05)
        decisions = {event._ID: Decision(action=Action.failed, particle=None)}
        exp, _ = _analyse_experiment(
            [event], decisions, parameters=get_parameters(particle_continuation=False)
        )

        exp.analyse()

        particles = exp.particles.get_all()
        assert len(particles) == 1
        assert particles[0].binding_event is event

    def test_non_qualifying_failed_event_creates_no_particle(
        self, spline_model: PsfModel
    ) -> None:
        event = _make_event(spline_model, frame=10, c=0.05)  # Unbinding, disqualifies
        decisions = {event._ID: Decision(action=Action.failed, particle=None)}
        exp, _ = _analyse_experiment(
            [event], decisions, parameters=get_parameters(particle_continuation=False)
        )

        exp.analyse()

        assert exp.particles.get_all() == []


class TestAnalyseIgnore:
    def test_disconnects_ignored_event(
        self, spline_model: PsfModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        event = _make_event(spline_model, frame=10)
        calls: list[RatiometricEvent] = []
        monkeypatch.setattr(event, "disconnect_all", lambda: calls.append(event))
        decisions = {event._ID: Decision(action=Action.ignore, particle=None)}
        exp, _ = _analyse_experiment(
            [event], decisions, parameters=get_parameters(particle_continuation=False)
        )

        exp.analyse()

        assert calls == [event]


class TestAnalyseUpdated:
    def test_no_particle_created_or_removed(self, spline_model: PsfModel) -> None:
        event = _make_event(spline_model, frame=10)
        decisions = {event._ID: Decision(action=Action.updated, particle=None)}
        exp, _ = _analyse_experiment(
            [event], decisions, parameters=get_parameters(particle_continuation=False)
        )

        exp.analyse()

        assert exp.particles.get_all() == []
        assert exp._index == 1

    def test_multi_continuation_still_runs(self, spline_model: PsfModel) -> None:
        event = _make_event(spline_model, frame=10)
        decisions = {event._ID: Decision(action=Action.updated, particle=None)}
        exp, investigator = _analyse_experiment(
            [event], decisions, parameters=get_parameters()
        )

        exp.analyse()

        assert investigator.multi_continuation_calls == [event]


class TestAnalyseLoopControl:
    def test_interesting_events_are_recorded(self, spline_model: PsfModel) -> None:
        event = _make_event(spline_model, frame=10)
        decisions = {
            event._ID: Decision(action=Action.ignore, particle=None, interesting=True)
        }
        exp, _ = _analyse_experiment(
            [event], decisions, parameters=get_parameters(particle_continuation=False)
        )

        exp.analyse()

        assert exp.interesting == [(event, None)]

    def test_stop_at_halts_before_processing_that_index(
        self, spline_model: PsfModel
    ) -> None:
        events = [_make_event(spline_model, frame=10 * i) for i in range(3)]
        decisions = {
            e._ID: Decision(action=Action.ignore, particle=None) for e in events
        }
        exp, investigator = _analyse_experiment(
            events, decisions, parameters=get_parameters(particle_continuation=False)
        )

        exp.analyse(stop_at=1)

        assert investigator.find_and_fit_calls == [events[0]]
        assert exp._index == 1

    def test_refresh_calls_reset_before_looping(
        self, spline_model: PsfModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        event = _make_event(spline_model, frame=10)
        decisions = {event._ID: Decision(action=Action.ignore, particle=None)}
        exp, _ = _analyse_experiment(
            [event], decisions, parameters=get_parameters(particle_continuation=False)
        )
        reset_mock = MagicMock()
        monkeypatch.setattr(exp, "reset", reset_mock)

        exp.analyse(refresh=True, stop_at=0)

        reset_mock.assert_called_once()

    def test_continuation_disabled_skips_multi_continuation(
        self, spline_model: PsfModel
    ) -> None:
        events = [_make_event(spline_model, frame=10 * i) for i in range(2)]
        decisions = {
            e._ID: Decision(action=Action.ignore, particle=None) for e in events
        }
        exp, investigator = _analyse_experiment(
            events, decisions, parameters=get_parameters(particle_continuation=False)
        )

        exp.analyse()

        assert investigator.multi_continuation_calls == []

    def test_continuation_enabled_calls_multi_continuation_per_event(
        self, spline_model: PsfModel
    ) -> None:
        events = [_make_event(spline_model, frame=10 * i) for i in range(2)]
        decisions = {
            e._ID: Decision(action=Action.ignore, particle=None) for e in events
        }
        exp, investigator = _analyse_experiment(
            events, decisions, parameters=get_parameters()
        )

        exp.analyse()

        assert investigator.multi_continuation_calls == events


class TestAnalyseCoincidence:
    def test_accepted_coincidence_removes_new_particle_and_restores_candidate(
        self, spline_model: PsfModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binding_event = _make_event(spline_model, frame=100, c=-0.05)
        candidate = _make_particle(spline_model, frame=90)
        candidate.lost_tracking = True
        candidate.unbinding_frame = 100

        propose_mock = MagicMock(return_value=True)
        monkeypatch.setattr(candidate, "propose_new_event", propose_mock)

        manager = ParticleManager()
        manager.append(candidate)

        decisions = {
            binding_event._ID: Decision(action=Action.create_new, particle=None)
        }
        params = get_parameters()
        exp, _ = _analyse_experiment(
            [binding_event], decisions, particles=manager, parameters=params
        )

        exp.analyse()

        propose_mock.assert_called_once()
        kwargs = propose_mock.call_args.kwargs
        assert kwargs["event"] is binding_event
        assert kwargs["minimum_mass"] == params.minimum_mass
        assert kwargs["spatial_threshold"] == 30
        assert kwargs["coincidence_checking"] is True
        assert kwargs["mass_fluctuation"] == pytest.approx(
            params.tandemmp_analysis.particle_continuation.mass_fluctuation / 2  # type: ignore[union-attr]
        )
        assert kwargs["max_strikes"] == (
            params.tandemmp_analysis.particle_continuation.max_strikes + 1  # type: ignore[union-attr]
        )

        remaining = exp.particles.get_all()
        assert remaining == [candidate]
        assert candidate.unbinding_frame is None
        assert candidate.lost_tracking is False

    def test_rejected_coincidence_keeps_new_particle(
        self, spline_model: PsfModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binding_event = _make_event(spline_model, frame=100, c=-0.05)
        candidate = _make_particle(spline_model, frame=90)
        candidate.lost_tracking = True
        candidate.unbinding_frame = 100

        propose_mock = MagicMock(return_value=False)
        monkeypatch.setattr(candidate, "propose_new_event", propose_mock)

        manager = ParticleManager()
        manager.append(candidate)

        decisions = {
            binding_event._ID: Decision(action=Action.create_new, particle=None)
        }
        exp, _ = _analyse_experiment(
            [binding_event], decisions, particles=manager, parameters=get_parameters()
        )

        exp.analyse()

        propose_mock.assert_called_once()
        remaining = exp.particles.get_all()
        assert candidate in remaining
        assert len(remaining) == 2
        assert candidate.unbinding_frame == 100
        assert candidate.lost_tracking is True

    def test_ghost_particle_skips_coincidence_check(
        self, spline_model: PsfModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binding_event = _make_event(spline_model, frame=100, c=-0.05)
        ghost = _make_ghost_particle(spline_model, frame=100)
        candidate = _make_particle(spline_model, frame=90)
        candidate.lost_tracking = True
        candidate.unbinding_frame = 100

        propose_mock = MagicMock(return_value=True)
        monkeypatch.setattr(candidate, "propose_new_event", propose_mock)

        manager = ParticleManager()
        manager.append(candidate)

        decisions = {
            binding_event._ID: Decision(action=Action.create_new, particle=ghost)
        }
        exp, _ = _analyse_experiment(
            [binding_event], decisions, particles=manager, parameters=get_parameters()
        )

        exp.analyse()

        propose_mock.assert_not_called()
        assert ghost in exp.particles.get_all()
        assert candidate in exp.particles.get_all()


# save


class TestSave:
    def test_round_trips_events_and_particles(
        self, spline_model: PsfModel, tmp_path: Path
    ) -> None:
        event = _make_event(spline_model, frame=10, c=-0.05)
        particle = Particle(event)
        manager = ParticleManager()
        manager.append(particle)

        loader = SimpleNamespace(
            mp_filepath="fake_movie.mp", vector=None, old_mass=None
        )
        exp = _bare_experiment(
            loader=loader,
            parameters=get_parameters(),
            particles=manager,
            _events=[event],
            movie=np.zeros((2, 4, 4)),
        )

        out_path = exp.save(tmp_path / "results.pkl")

        results = load_results(out_path)
        assert results.source_filepath == "fake_movie.mp"
        assert len(results.events) == 1
        assert len(results.particles) == 1
        assert results.particles[0].particle == particle.ID
        assert results.movie is not None

    def test_save_movie_false_omits_movie(
        self, spline_model: PsfModel, tmp_path: Path
    ) -> None:
        event = _make_event(spline_model, frame=10, c=-0.05)
        loader = SimpleNamespace(
            mp_filepath="fake_movie.mp", vector=None, old_mass=None
        )
        exp = _bare_experiment(
            loader=loader,
            parameters=get_parameters(),
            particles=ParticleManager(),
            _events=[event],
            movie=np.zeros((2, 4, 4)),
        )

        out_path = exp.save(tmp_path / "results.pkl", save_movie=False)

        results = load_results(out_path)
        assert results.movie is None

    def test_records_active_calibration(
        self, spline_model: PsfModel, tmp_path: Path
    ) -> None:
        event = _make_event(spline_model, frame=10, c=-0.05)
        loader = SimpleNamespace(
            mp_filepath="fake_movie.mp", vector=None, old_mass=None
        )
        exp = _bare_experiment(
            loader=loader,
            parameters=get_parameters(),
            particles=ParticleManager(),
            _events=[event],
            movie=np.zeros((2, 4, 4)),
        )

        custom = Calibration(slope=-0.00005, offset=-0.0001)
        set_calibration(custom)
        try:
            out_path = exp.save(tmp_path / "results.pkl")
        finally:
            set_calibration(None)

        results = load_results(out_path)
        assert results.calibration == custom

    def test_defaults_to_active_calibration_when_none_overridden(
        self, spline_model: PsfModel, tmp_path: Path
    ) -> None:
        event = _make_event(spline_model, frame=10, c=-0.05)
        loader = SimpleNamespace(
            mp_filepath="fake_movie.mp", vector=None, old_mass=None
        )
        exp = _bare_experiment(
            loader=loader,
            parameters=get_parameters(),
            particles=ParticleManager(),
            _events=[event],
            movie=np.zeros((2, 4, 4)),
        )

        assert get_calibration() == DEFAULT_CALIBRATION
        out_path = exp.save(tmp_path / "results.pkl")

        results = load_results(out_path)
        assert results.calibration == DEFAULT_CALIBRATION
