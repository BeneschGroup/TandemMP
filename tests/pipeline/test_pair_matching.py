"""Tests for tandemmp.pipeline.pair_matching — target API.

``run_pair_matching`` is a thin wrapper around
:class:`~tandemmp.pipeline.pca.Experiment`/``.simple_analysis`` — exercised
here with ``Experiment`` monkeypatched, avoiding a real ``.mp`` file
dependency.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tandemmp.core.conversions import Calibration, get_calibration
from tandemmp.pipeline import pair_matching as pm
from tandemmp.pipeline.calibration import load_mc
from tandemmp.pipeline.pair_matching import PairMatchingResult, run_pair_matching
from tandemmp.pipeline.simple_analysis import (
    Pair,
    SimpleEvent,
    build_simple_events,
    match_pairs_sigma,
)

MC_FIXTURE = Path(__file__).parent / "fixtures" / "20260329_mfp1.mc"


@dataclasses.dataclass
class _FakeEvent:
    """Stands in for a RatiometricEvent's particle_c/m/x/y/i interface."""

    particle_c: float
    particle_m: float
    particle_x: float
    particle_y: float
    i: int


def _binding(i: int, x: float = 0.0, y: float = 0.0, m: float = 100.0) -> _FakeEvent:
    return _FakeEvent(particle_c=-0.05, particle_m=m, particle_x=x, particle_y=y, i=i)


def _unbinding(i: int, x: float = 0.0, y: float = 0.0, m: float = 100.0) -> _FakeEvent:
    return _FakeEvent(particle_c=0.05, particle_m=m, particle_x=x, particle_y=y, i=i)


class _FakeExperiment:
    """Stand-in for ``pipeline.pca.Experiment``, capturing constructor kwargs."""

    events_for_test: list[Any] = [_binding(0), _unbinding(10)]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.events = type(self).events_for_test
        self.simple_analysis_calls: list[dict[str, Any]] = []

    def simple_analysis(self, **kwargs: Any) -> list[Pair]:
        self.simple_analysis_calls.append(kwargs)
        return []


@pytest.fixture(autouse=True)
def _reset_fake_experiment() -> None:
    _FakeExperiment.events_for_test = [_binding(0), _unbinding(10)]


# run_pair_matching — wiring


class TestRunPairMatchingWiring:
    def test_wires_experiment_events_into_simple_analysis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pm, "Experiment", _FakeExperiment)

        result = run_pair_matching(
            "movie.mp",
            sigma_multiplier=2.0,
            max_pairs=3,
            analyse_distance=5.0,
            minimum_overlap=0.02,
            match_binding=True,
        )

        assert isinstance(result, PairMatchingResult)
        assert result.events == _FakeExperiment.events_for_test

    def test_forwards_sigma_multiplier_max_pairs_etc_to_simple_analysis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pm, "Experiment", _FakeExperiment)

        experiments: list[_FakeExperiment] = []
        real_init = _FakeExperiment.__init__

        def _capturing_init(self: _FakeExperiment, *a: Any, **k: Any) -> None:
            real_init(self, *a, **k)
            experiments.append(self)

        monkeypatch.setattr(_FakeExperiment, "__init__", _capturing_init)

        run_pair_matching(
            "movie.mp",
            sigma_multiplier=2.0,
            max_pairs=3,
            analyse_distance=5.0,
            minimum_overlap=0.02,
            match_binding=True,
        )

        call = experiments[0].simple_analysis_calls[0]
        assert call["sigma_multiplier"] == 2.0
        assert call["max_pairs"] == 3
        assert call["analyse_distance"] == 5.0
        assert call["minimum_overlap"] == 0.02
        assert call["match_binding"] is True

    def test_forwards_frame_start_end_seed_and_pipeline_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        class _CapturingExperiment(_FakeExperiment):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["args"] = args
                seen["kwargs"] = kwargs
                super().__init__(*args, **kwargs)

        sentinel_runner = object()
        monkeypatch.setattr(pm, "Experiment", _CapturingExperiment)

        run_pair_matching(
            "movie.mp",
            frame_start=5,
            frame_end=15,
            seed=42,
            pipeline_runner=sentinel_runner,  # type: ignore[arg-type]
        )

        assert seen["args"] == ("movie.mp",)
        assert seen["kwargs"]["frame_start"] == 5
        assert seen["kwargs"]["frame_end"] == 15
        assert seen["kwargs"]["seed"] == 42
        assert seen["kwargs"]["pipeline_runner"] is sentinel_runner

    def test_events_csv_forwarded_when_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        class _CapturingExperiment(_FakeExperiment):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["kwargs"] = kwargs
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(pm, "Experiment", _CapturingExperiment)

        run_pair_matching("movie.mp", events_csv="events.csv")

        assert seen["kwargs"]["events_csv"] == "events.csv"

    def test_no_events_csv_omits_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        class _CapturingExperiment(_FakeExperiment):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["kwargs"] = kwargs
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(pm, "Experiment", _CapturingExperiment)

        run_pair_matching("movie.mp")

        assert "events_csv" not in seen["kwargs"]

    def test_forwards_parameter_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        class _CapturingExperiment(_FakeExperiment):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["kwargs"] = kwargs
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(pm, "Experiment", _CapturingExperiment)

        run_pair_matching("movie.mp", parallel=False, drift_correction=False)

        assert seen["kwargs"]["parallel"] is False
        assert seen["kwargs"]["drift_correction"] is False

    def test_default_parallel_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        class _CapturingExperiment(_FakeExperiment):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["kwargs"] = kwargs
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(pm, "Experiment", _CapturingExperiment)

        run_pair_matching("movie.mp")

        assert seen["kwargs"]["parallel"] is True

    def test_result_contains_source_filepath_and_events(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pm, "Experiment", _FakeExperiment)

        result = run_pair_matching("some/movie.mp")

        assert result.source_filepath == "some/movie.mp"
        assert len(result.events) == 2

    def test_no_calibration_leaves_active_calibration_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pm, "Loader", _FakeLoader)
        monkeypatch.setattr(pm, "match_pairs_sigma", lambda *a, **k: [])
        custom = Calibration(slope=-0.00005, offset=-0.0001)
        pm.set_calibration(custom)

        run_pair_matching("movie.mp")

        assert get_calibration() == custom

    def test_calibration_sets_active_calibration_before_loader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        class _CapturingLoader(_FakeLoader):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["calibration"] = get_calibration()
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(pm, "Loader", _CapturingLoader)
        monkeypatch.setattr(pm, "match_pairs_sigma", lambda *a, **k: [])
        custom = Calibration(slope=-0.00005, offset=-0.0001)

        run_pair_matching("movie.mp", calibration=custom)

        assert seen["calibration"] == custom
        assert get_calibration() == custom

    def test_calibration_accepts_mc_filepath(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pm, "Loader", _FakeLoader)
        monkeypatch.setattr(pm, "match_pairs_sigma", lambda *a, **k: [])

        run_pair_matching("movie.mp", calibration=MC_FIXTURE)

        assert get_calibration() == load_mc(MC_FIXTURE).calibration


# run_pair_matching — end to end (real Experiment.simple_analysis via a bare fake)


class TestRunPairMatchingEndToEnd:
    def test_matches_a_close_binding_unbinding_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _RealSimpleAnalysisExperiment(_FakeExperiment):
            def simple_analysis(self, **kwargs: Any) -> list[Pair]:
                simple_events = build_simple_events(self.events)  # type: ignore[arg-type]
                return match_pairs_sigma(simple_events, **kwargs)

        _FakeExperiment.events_for_test = [
            _binding(i=0, x=0.0, y=0.0, m=100.0),
            _unbinding(i=5, x=0.5, y=0.5, m=100.0),
        ]
        monkeypatch.setattr(pm, "Experiment", _RealSimpleAnalysisExperiment)

        result = run_pair_matching("movie.mp")

        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.binding.i == pytest.approx(0.0)
        assert pair.unbinding.i == pytest.approx(5.0)

    def test_no_pairs_when_events_too_far_apart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _RealSimpleAnalysisExperiment(_FakeExperiment):
            def simple_analysis(self, **kwargs: Any) -> list[Pair]:
                simple_events = build_simple_events(self.events)  # type: ignore[arg-type]
                return match_pairs_sigma(simple_events, **kwargs)

        _FakeExperiment.events_for_test = [
            _binding(i=0, x=0.0, y=0.0),
            _unbinding(i=5, x=100.0, y=100.0),
        ]
        monkeypatch.setattr(pm, "Experiment", _RealSimpleAnalysisExperiment)

        result = run_pair_matching("movie.mp")

        assert result.pairs == []

    def test_zero_events_yields_zero_pairs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _RealSimpleAnalysisExperiment(_FakeExperiment):
            def simple_analysis(self, **kwargs: Any) -> list[Pair]:
                simple_events = build_simple_events(self.events)  # type: ignore[arg-type]
                return match_pairs_sigma(simple_events, **kwargs)

        _FakeExperiment.events_for_test = []
        monkeypatch.setattr(pm, "Experiment", _RealSimpleAnalysisExperiment)

        result = run_pair_matching("movie.mp")

        assert result.pairs == []
        assert result.events == []
        assert result.initial_masses.shape == (0,)
        assert result.secondary_masses.shape == (0,)


# PairMatchingResult


class TestPairMatchingResult:
    def test_initial_and_secondary_masses(self) -> None:
        b1 = SimpleEvent(c=-0.05, m=100.0, x=0.0, y=0.0, i=0)
        u1 = SimpleEvent(c=0.03, m=60.0, x=0.0, y=0.0, i=5)
        b2 = SimpleEvent(c=-0.02, m=40.0, x=1.0, y=1.0, i=1)
        u2 = SimpleEvent(c=0.02, m=40.0, x=1.0, y=1.0, i=6)

        pairs = [
            Pair(
                binding=b1, unbinding=u1, time=5.0, distance=0.0, overlap=1.0, order=0
            ),
            Pair(
                binding=b2, unbinding=u2, time=5.0, distance=0.0, overlap=1.0, order=0
            ),
        ]
        result = PairMatchingResult(pairs=pairs, events=[], source_filepath="foo.mp")

        np.testing.assert_allclose(result.initial_masses, [100.0, 40.0])
        np.testing.assert_allclose(result.secondary_masses, [60.0, 40.0])

    def test_masses_empty_when_no_pairs(self) -> None:
        result = PairMatchingResult(pairs=[], events=[], source_filepath="foo.mp")

        assert result.initial_masses.shape == (0,)
        assert result.secondary_masses.shape == (0,)

    def test_frozen(self) -> None:
        result = PairMatchingResult(pairs=[], events=[], source_filepath="foo.mp")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.source_filepath = "bar.mp"  # type: ignore[misc]
