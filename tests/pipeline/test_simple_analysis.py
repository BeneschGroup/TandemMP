"""Tests for tandemmp.pipeline.simple_analysis — target API.

Covers CSV loading + drift correction, pair-matching algorithms, and RDF/ring-area
helpers; ``load_events`` uses a monkeypatched ``Loader``/``get_vector`` to avoid a
real ``open_iscat`` dependency.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tandemmp.core.conversions import Calibration, c_to_mass, get_calibration
from tandemmp.pipeline import simple_analysis as sa
from tandemmp.pipeline.calibration import load_mc
from tandemmp.pipeline.simple_analysis import (
    NewPair,
    Pair,
    SimpleEvent,
    area_of_ring_in_box,
    build_simple_events,
    calculate_rdf,
    convert_to_product_precursor,
    load_events,
    match_pairs_sigma,
    match_pairs_tally,
)

MC_FIXTURE = Path(__file__).parent / "fixtures" / "20260329_mfp1.mc"


@pytest.fixture(autouse=True)
def _reset_ids() -> None:
    sa._reset_simple_event_ids()
    yield
    sa._reset_simple_event_ids()


def _event(c: float, m: float, x: float, y: float, i: int) -> SimpleEvent:
    return SimpleEvent(c=c, m=m, x=x, y=y, i=i)


# SimpleEvent


class TestSimpleEvent:
    def test_frozen(self) -> None:
        event = _event(c=-0.05, m=100.0, x=1.0, y=2.0, i=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.m = 200.0  # type: ignore[misc]

    def test_auto_increment_id_unique(self) -> None:
        a = _event(c=-0.05, m=100.0, x=1.0, y=2.0, i=0)
        b = _event(c=-0.05, m=100.0, x=1.0, y=2.0, i=0)
        assert a.id != b.id
        assert a != b
        assert hash(a) != hash(b)

    def test_explicit_id_overrides_default(self) -> None:
        event = SimpleEvent(c=-0.05, m=100.0, x=1.0, y=2.0, i=0, id=42)
        assert event.id == 42

    def test_equal_fields_and_id_compare_equal(self) -> None:
        a = SimpleEvent(c=-0.05, m=100.0, x=1.0, y=2.0, i=0, id=7)
        b = SimpleEvent(c=-0.05, m=100.0, x=1.0, y=2.0, i=0, id=7)
        assert a == b
        assert hash(a) == hash(b)


# build_simple_events


@dataclasses.dataclass
class _FakeRatiometricEvent:
    """Stands in for a RatiometricEvent's particle_c/m/x/y/i interface."""

    particle_c: float
    particle_m: float
    particle_x: float
    particle_y: float
    i: int


class TestBuildSimpleEvents:
    def test_converts_particle_fields(self) -> None:
        events = [
            _FakeRatiometricEvent(
                particle_c=-0.05, particle_m=100.0, particle_x=1.0, particle_y=2.0, i=10
            ),
            _FakeRatiometricEvent(
                particle_c=0.03, particle_m=60.0, particle_x=3.0, particle_y=4.0, i=20
            ),
        ]

        simple_events = build_simple_events(events)  # type: ignore[arg-type]

        assert len(simple_events) == 2
        assert all(isinstance(e, SimpleEvent) for e in simple_events)
        assert simple_events[0].c == pytest.approx(-0.05)
        assert simple_events[0].m == pytest.approx(100.0)
        assert simple_events[0].x == pytest.approx(1.0)
        assert simple_events[0].y == pytest.approx(2.0)
        assert simple_events[0].i == pytest.approx(10.0)
        assert simple_events[1].c == pytest.approx(0.03)
        assert simple_events[1].i == pytest.approx(20.0)

    def test_empty_input(self) -> None:
        assert build_simple_events([]) == []


# load_events


class TestLoadEvents:
    def _write_csv(self, tmp_path: Path, rows: list[tuple[float, ...]]) -> Path:
        path = tmp_path / "events.csv"
        header = "frame,x,y,contrast,mass\n"
        body = "\n".join(",".join(str(v) for v in row) for row in rows)
        path.write_text(header + body + "\n")
        return path

    def test_loads_without_movie(self, tmp_path: Path) -> None:
        csv_path = self._write_csv(
            tmp_path,
            [
                (0, 10.0, 20.0, -0.05, 100.0),
                (5, 30.0, 40.0, 0.05, -100.0),
            ],
        )
        events = load_events(csv_path)
        assert len(events) == 2
        assert events[0].x == pytest.approx(10.0)
        assert events[0].y == pytest.approx(20.0)
        assert events[0].m == pytest.approx(100.0)
        assert events[1].m == pytest.approx(-100.0)

    def test_missing_mass_column_falls_back_to_contrast(self, tmp_path: Path) -> None:
        contrasts = [-0.01, -0.02]
        csv_path = self._write_csv(
            tmp_path,
            [
                (0, 10.0, 20.0, contrasts[0], float("nan")),
                (1, 11.0, 21.0, contrasts[1], float("nan")),
            ],
        )
        events = load_events(csv_path)
        np.testing.assert_allclose(
            [e.m for e in events], c_to_mass(np.array(contrasts))
        )

    def test_missing_mass_column_uses_given_calibration(
        self, tmp_path: Path
    ) -> None:
        contrasts = [-0.01, -0.02]
        csv_path = self._write_csv(
            tmp_path,
            [
                (0, 10.0, 20.0, contrasts[0], float("nan")),
                (1, 11.0, 21.0, contrasts[1], float("nan")),
            ],
        )
        custom = Calibration(slope=-0.00005, offset=-0.0001)

        events = load_events(csv_path, calibration=custom)

        expected = (np.array(contrasts) - custom.offset) / custom.slope
        np.testing.assert_allclose([e.m for e in events], expected)
        # The process-wide calibration is untouched by load_events.
        assert get_calibration() != custom

    def test_missing_mass_column_accepts_mc_filepath_as_calibration(
        self, tmp_path: Path
    ) -> None:
        contrasts = [-0.01, -0.02]
        csv_path = self._write_csv(
            tmp_path,
            [
                (0, 10.0, 20.0, contrasts[0], float("nan")),
                (1, 11.0, 21.0, contrasts[1], float("nan")),
            ],
        )

        events = load_events(csv_path, calibration=MC_FIXTURE)

        cal = load_mc(MC_FIXTURE).calibration
        expected = (np.array(contrasts) - cal.offset) / cal.slope
        np.testing.assert_allclose([e.m for e in events], expected)

    def test_movie_filepath_applies_drift_correction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = self._write_csv(
            tmp_path,
            [
                (0, 10.0, 20.0, -0.05, 100.0),
                (1, 10.0, 20.0, -0.05, 100.0),
            ],
        )

        class _FakeLoader:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.movie = np.zeros((2, 8, 8))

        shifts = {0: (1.0, -2.0), 1: (3.0, -4.0)}

        def _fake_get_vector(*, movie: Any, **kwargs: Any) -> Any:
            class _Vector:
                def __getitem__(self, key: int) -> tuple[float, float]:
                    return shifts[key]

            return _Vector()

        monkeypatch.setattr(sa, "Loader", _FakeLoader)
        monkeypatch.setattr(sa, "get_vector", _fake_get_vector)

        events = load_events(csv_path, movie_filepath="dummy.mp")

        assert events[0].x == pytest.approx(10.0 - 1.0)
        assert events[0].y == pytest.approx(20.0 - -2.0)
        assert events[1].x == pytest.approx(10.0 - 3.0)
        assert events[1].y == pytest.approx(20.0 - -4.0)


# match_pairs_sigma / match_pairs_tally


class TestMatchPairsSigma:
    def test_matches_close_overlapping_events(self) -> None:
        binding = _event(c=-0.05, m=100.0, x=10.0, y=10.0, i=0)
        unbinding = _event(c=0.05, m=-100.0, x=10.1, y=10.1, i=5)
        pairs = match_pairs_sigma([binding, unbinding], analyse_distance=4)
        assert len(pairs) == 1
        pair = pairs[0]
        assert isinstance(pair, Pair)
        assert pair.binding is binding
        assert pair.unbinding is unbinding
        assert pair.time == 5

    def test_no_match_when_too_far(self) -> None:
        binding = _event(c=-0.05, m=100.0, x=0.0, y=0.0, i=0)
        unbinding = _event(c=0.05, m=-100.0, x=100.0, y=100.0, i=5)
        pairs = match_pairs_sigma([binding, unbinding], analyse_distance=4)
        assert pairs == []

    def test_no_match_when_unbinding_before_binding(self) -> None:
        binding = _event(c=-0.05, m=100.0, x=10.0, y=10.0, i=5)
        unbinding = _event(c=0.05, m=-100.0, x=10.0, y=10.0, i=0)
        pairs = match_pairs_sigma([binding, unbinding], analyse_distance=4)
        assert pairs == []

    def test_max_pairs_limits_reuse_of_binding_event(self) -> None:
        binding = _event(c=-0.05, m=100.0, x=10.0, y=10.0, i=0)
        unbindings = [_event(c=0.05, m=-100.0, x=10.0, y=10.0, i=i) for i in (1, 2, 3)]
        pairs = match_pairs_sigma(
            [binding, *unbindings], analyse_distance=4, max_pairs=2
        )
        assert len(pairs) == 2
        assert all(p.binding is binding for p in pairs)

    def test_repr_includes_ids(self) -> None:
        binding = _event(c=-0.05, m=100.0, x=10.0, y=10.0, i=0)
        unbinding = _event(c=0.05, m=-100.0, x=10.0, y=10.0, i=5)
        pairs = match_pairs_sigma([binding, unbinding], analyse_distance=4)
        assert f"id={binding.id}" in repr(pairs[0])

    def test_match_binding_pairs_binding_events_together(self) -> None:
        # All three events are "binding" (negative contrast); match_binding
        # re-targets the matching to pair binding events with each other
        # instead of with (nonexistent, here) unbinding events, while still
        # excluding an event from matching itself.
        first = _event(c=-0.05, m=100.0, x=0.0, y=0.0, i=0)
        second = _event(c=-0.05, m=100.0, x=0.0, y=0.0, i=1)
        third = _event(c=-0.05, m=100.0, x=0.0, y=0.0, i=2)

        pairs = match_pairs_sigma(
            [first, second, third], analyse_distance=4, match_binding=True
        )

        assert len(pairs) == 2
        assert all(p.binding is first for p in pairs)
        assert {p.unbinding for p in pairs} == {second, third}

    def test_match_binding_off_finds_no_pairs_among_binding_only_events(
        self,
    ) -> None:
        first = _event(c=-0.05, m=100.0, x=0.0, y=0.0, i=0)
        second = _event(c=-0.05, m=100.0, x=0.0, y=0.0, i=1)
        pairs = match_pairs_sigma([first, second], analyse_distance=4)
        assert pairs == []


class TestMatchPairsTally:
    def test_builds_running_precursor_product_tally(self) -> None:
        binding = _event(c=-0.05, m=100.0, x=10.0, y=10.0, i=0)
        unbinding = _event(c=0.05, m=-40.0, x=10.0, y=10.0, i=5)
        pairs = match_pairs_tally([binding, unbinding], analyse_distance=4)
        assert len(pairs) == 1
        pair = pairs[0]
        assert isinstance(pair, NewPair)
        assert pair.precursor == pytest.approx(100.0)
        assert pair.product == pytest.approx(60.0)

    def test_excludes_self_pairing(self) -> None:
        binding = _event(c=-0.05, m=100.0, x=10.0, y=10.0, i=0)
        pairs = match_pairs_tally([binding], analyse_distance=4)
        assert pairs == []

    def test_max_pairs_caps_chain_length(self) -> None:
        binding = _event(c=-0.05, m=100.0, x=0.0, y=0.0, i=0)
        unbindings = [_event(c=0.05, m=-10.0, x=0.0, y=0.0, i=t) for t in (1, 2, 3, 4)]
        pairs = match_pairs_tally(
            [binding, *unbindings], analyse_distance=4, max_pairs=2
        )
        assert len(pairs) == 2
        assert [p.time for p in pairs] == [1, 2]


# convert_to_product_precursor


class TestConvertToProductPrecursor:
    def test_chains_sequential_unbindings_onto_running_totals(self) -> None:
        binding = _event(c=-0.1, m=200.0, x=0.0, y=0.0, i=0)
        unbinding_1 = _event(c=0.02, m=-40.0, x=0.0, y=0.0, i=1)
        unbinding_2 = _event(c=0.03, m=-60.0, x=0.0, y=0.0, i=2)
        pairs = [
            Pair(
                binding=binding,
                unbinding=unbinding_1,
                time=1,
                distance=0.0,
                overlap=1.0,
                order=0,
            ),
            Pair(
                binding=binding,
                unbinding=unbinding_2,
                time=2,
                distance=0.0,
                overlap=1.0,
                order=1,
            ),
        ]
        new_pairs = convert_to_product_precursor(pairs)
        assert len(new_pairs) == 2
        first, second = sorted(new_pairs, key=lambda p: p.order)
        # Each step's precursor is the previous step's product — a running
        # chain, not every step re-based on the original binding mass.
        assert first.binding.m == pytest.approx(200.0)
        assert first.unbinding.m == pytest.approx(160.0)
        assert second.binding.m == pytest.approx(160.0)
        assert second.unbinding.m == pytest.approx(100.0)


# area_of_ring_in_box / calculate_rdf


class TestAreaOfRingInBox:
    def test_full_ring_within_box_matches_annulus_area(self) -> None:
        area = area_of_ring_in_box(
            origin=(50.0, 50.0),
            inner_radius=5.0,
            outer_radius=10.0,
            box_size=(100.0, 100.0),
            resolution=2000,
        )
        expected = np.pi * (10.0**2 - 5.0**2)
        assert area == pytest.approx(expected, rel=0.01)

    def test_zero_width_ring_has_zero_area(self) -> None:
        area = area_of_ring_in_box(
            origin=(50.0, 50.0),
            inner_radius=5.0,
            outer_radius=5.0,
            box_size=(100.0, 100.0),
            resolution=500,
        )
        assert area == pytest.approx(0.0, abs=1e-6)

    def test_annulus_clipped_by_box_edge(self) -> None:
        # Origin at the box corner: only a quarter of the full circle lies
        # within the box, pinning the docstring's box-clipping behaviour.
        area = area_of_ring_in_box(
            origin=(0.0, 0.0),
            inner_radius=0.0,
            outer_radius=10.0,
            box_size=(20.0, 20.0),
            resolution=2000,
        )
        expected = np.pi * 10.0**2 / 4
        assert area == pytest.approx(expected, rel=0.01)


class TestCalculateRdf:
    def test_uniform_random_points_have_rdf_near_one(self) -> None:
        # calculate_rdf normalises each query point's local counts by the
        # candidate-pool size rather than the number of query points, so
        # its g(r) ~ 1 approximation only holds when sel_first and
        # sel_after select comparably-sized pools of the same data (the
        # common case of an unrestricted, self-referential RDF) — as here,
        # where both default to "every point".
        rng = np.random.default_rng(0)
        n = 3000
        x = rng.uniform(0, 300, n)
        y = rng.uniform(0, 300, n)
        i = np.zeros(n)
        bins = np.linspace(5, 15, 6)
        rdf, bin_centres = calculate_rdf(x, y, i, bins=bins)
        assert rdf.shape == (len(bins) - 1,)
        assert bin_centres.shape == (len(bins) - 1,)
        assert np.mean(rdf) == pytest.approx(1.0, rel=0.2)

    def test_sel_first_and_sel_after_restrict_pairs(self) -> None:
        x = np.array([0.0, 0.0, 50.0])
        y = np.array([0.0, 0.0, 50.0])
        i = np.array([0, 1, 2])
        sel_first = np.array([True, False, False])
        sel_after = np.array([False, True, False])
        bins = np.linspace(0, 5, 3)
        rdf, _ = calculate_rdf(
            x, y, i, bins=bins, sel_first=sel_first, sel_after=sel_after
        )
        assert rdf.shape == (2,)
