"""Tests for tandemmp.pipeline.pair_matching — target API.

``run_pair_matching`` is exercised with ``Loader``/``match_pairs_sigma``
monkeypatched, avoiding a real ``.mp``/``open_iscat`` dependency.
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
from tandemmp.pipeline.simple_analysis import Pair, SimpleEvent

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


class _FakeLoader:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.events: list[_FakeEvent] = [_binding(0), _unbinding(10)]


# run_pair_matching — wiring


class TestRunPairMatchingWiring:
    def test_wires_loader_events_into_match_pairs_sigma(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        def _spy_match(simple_events: Any, **kwargs: Any) -> list[Pair]:
            calls["simple_events"] = simple_events
            calls.update(kwargs)
            return []

        monkeypatch.setattr(pm, "Loader", _FakeLoader)
        monkeypatch.setattr(pm, "match_pairs_sigma", _spy_match)

        result = run_pair_matching(
            "movie.mp",
            sigma_multiplier=2.0,
            max_pairs=3,
            analyse_distance=5.0,
            minimum_overlap=0.02,
            match_binding=True,
        )

        assert isinstance(result, PairMatchingResult)
        assert calls["sigma_multiplier"] == 2.0
        assert calls["max_pairs"] == 3
        assert calls["analyse_distance"] == 5.0
        assert calls["minimum_overlap"] == 0.02
        assert calls["match_binding"] is True

        simple_events = calls["simple_events"]
        assert len(simple_events) == 2
        assert all(isinstance(e, SimpleEvent) for e in simple_events)
        assert simple_events[0].c == pytest.approx(-0.05)
        assert simple_events[0].m == pytest.approx(100.0)
        assert simple_events[0].i == pytest.approx(0.0)
        assert simple_events[1].c == pytest.approx(0.05)
        assert simple_events[1].i == pytest.approx(10.0)

    def test_forwards_frame_start_end_seed_and_pipeline_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        class _CapturingLoader(_FakeLoader):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["args"] = args
                seen["kwargs"] = kwargs
                super().__init__(*args, **kwargs)

        sentinel_runner = object()
        monkeypatch.setattr(pm, "Loader", _CapturingLoader)
        monkeypatch.setattr(pm, "match_pairs_sigma", lambda *a, **k: [])

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

    def test_forwards_parameter_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        class _CapturingLoader(_FakeLoader):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["parameters"] = kwargs["parameters"]
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(pm, "Loader", _CapturingLoader)
        monkeypatch.setattr(pm, "match_pairs_sigma", lambda *a, **k: [])

        run_pair_matching("movie.mp", parallel=False, drift_correction=False)

        assert seen["parameters"].parallel is False
        assert seen["parameters"].drift_correction is False

    def test_default_parallel_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        class _CapturingLoader(_FakeLoader):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["parameters"] = kwargs["parameters"]
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(pm, "Loader", _CapturingLoader)
        monkeypatch.setattr(pm, "match_pairs_sigma", lambda *a, **k: [])

        run_pair_matching("movie.mp")

        assert seen["parameters"].parallel is True

    def test_result_contains_source_filepath_and_events(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pm, "Loader", _FakeLoader)
        monkeypatch.setattr(pm, "match_pairs_sigma", lambda *a, **k: [])

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


# run_pair_matching — end to end (real match_pairs_sigma)


class TestRunPairMatchingEndToEnd:
    def test_matches_a_close_binding_unbinding_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _CloseFakeLoader(_FakeLoader):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.events = [
                    _binding(i=0, x=0.0, y=0.0, m=100.0),
                    _unbinding(i=5, x=0.5, y=0.5, m=100.0),
                ]

        monkeypatch.setattr(pm, "Loader", _CloseFakeLoader)

        result = run_pair_matching("movie.mp")

        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.binding.i == pytest.approx(0.0)
        assert pair.unbinding.i == pytest.approx(5.0)

    def test_no_pairs_when_events_too_far_apart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FarFakeLoader(_FakeLoader):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.events = [
                    _binding(i=0, x=0.0, y=0.0),
                    _unbinding(i=5, x=100.0, y=100.0),
                ]

        monkeypatch.setattr(pm, "Loader", _FarFakeLoader)

        result = run_pair_matching("movie.mp")

        assert result.pairs == []

    def test_zero_events_yields_zero_pairs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _EmptyFakeLoader(_FakeLoader):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.events: list[_FakeEvent] = []

        monkeypatch.setattr(pm, "Loader", _EmptyFakeLoader)

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
