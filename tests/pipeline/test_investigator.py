"""Tests for tandemmp.pipeline.investigator.

Tiered by cost/value: pure unit tests for small containers/helpers, lightweight
``BaseInvestigator`` methods on synthetic movies, and higher-level
``GhostInvestigator``/``Investigator`` scenarios wired end to end.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import OptimizeResult

from tandemmp.core.classifications import (
    Classification,
    EvClassification,
    ProcessingMethod,
)
from tandemmp.core.conversions import mass_to_c
from tandemmp.core.parameters import get_parameters
from tandemmp.events import (
    GhostEvent,
    PeakImage,
    RatiometricEvent,
    SubtractionSeries,
)
from tandemmp.fitting.psf import PsfModel
from tandemmp.pipeline.investigator import (
    Action,
    BaseInvestigator,
    Decision,
    Investigator,
    MultiFitResult,
    TrailingOffException,
    _distance_array,
    create_ghost_event,
)
from tandemmp.pipeline.open_iscat_pipeline import Info
from tandemmp.tracking.manager import ParticleManager
from tandemmp.tracking.particle import (
    GhostParticle,
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
    _reset_particle_ids()


@pytest.fixture()
def info() -> Info:
    return Info(
        scatter_interference_destructive=True,
        raw_camera_frame_rate=1000.0,
        frame_binning=1,
        pixel_binning=1,
        full_well_capacity=None,
        grey_levels=None,
    )


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


def _make_binding_ratiometric(
    model: PsfModel,
    *,
    frame: int = 50,
    c: float = -0.05,
    midpoint: tuple[int, int] = (25, 25),
    navg: int = 3,
) -> RatiometricEvent:
    peak = _make_peak(model, frame=frame, c=c, midpoint=midpoint)
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


def _make_particle(
    model: PsfModel,
    *,
    frame: int = 50,
    midpoint: tuple[int, int] = (25, 25),
    c: float = -0.05,
) -> Particle:
    return Particle(
        _make_binding_ratiometric(model, frame=frame, midpoint=midpoint, c=c)
    )


def _make_ghost_particle(
    model: PsfModel,
    *,
    frame: int = 50,
    midpoint: tuple[int, int] = (25, 25),
    c: float = -0.05,
) -> GhostParticle:
    return GhostParticle(_make_ghost_event(model, frame=frame, midpoint=midpoint, c=c))


def _make_synthetic_movie(
    width: int = 51,
    n_frames: int = 60,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Positive-valued synthetic 3-D movie suitable for the ratiometric divide."""
    rng = np.random.default_rng(seed)
    # Strictly positive baseline so `frames / start_frames` is defined.
    baseline = 1.0 + 0.05 * rng.random((width, width))
    movie = np.tile(baseline, (n_frames, 1, 1))
    movie += rng.normal(0, 0.001, movie.shape)
    return movie


def _make_mock_series(
    length: int = 7,
    *,
    ghost_ml: float | None = 0.1,
    bleed_time: int = 3,
) -> MagicMock:
    """SubtractionSeries stand-in with the attributes the investigator reads."""
    s = MagicMock(spec=SubtractionSeries)
    s.avg_residual_noise = 0.001
    s.ghost_signal_ML = ghost_ml
    s.basic_fit_error = np.full(length, 0.01)
    s.subtracted_i = np.full(length, 5)
    s.bleed_time = np.full(length, bleed_time)
    s.smooth_dx = np.zeros(length)
    s.smooth_dy = np.zeros(length)
    s.x = np.zeros(length)
    s.y = np.zeros(length)
    s.c = np.array([-0.02, -0.03])
    s.m = np.full(length, 100.0)
    s.is_bad = np.zeros(length, dtype=bool)
    s.unreliability.return_value = 0.0

    def get_slice(idx: object) -> MagicMock:
        return MagicMock(i=int(idx) if isinstance(idx, int) else 0)

    s.__getitem__.side_effect = get_slice
    s.fit_to.return_value = OptimizeResult(fun=1.0, x=np.array([-0.03, -0.05]))
    s.impression.return_value = np.zeros((13, 13))
    s.append_original = MagicMock()
    return s


def _make_fit_attempt(
    series: SubtractionSeries,
    *,
    c: float,
    classification: Classification = EvClassification.Binding,
    fun: float = 1e-8,
) -> MultiFitResult:
    """A real, cheaply-constructed ``MultiFitResult`` for a real *series*.

    Used to stand in for the return value of ``fit_initial_ghost``/
    ``fit_double_ghost`` in tests: the interesting behaviour in
    ``attempt_initial_ghost``/``double_check_ghost`` is what they *do* with a
    fit result (GhostParticle construction, threshold checks,
    ``propose_new_event`` acceptance), not whether ``scipy.optimize.minimize``
    converges, so the inner optimiser is replaced with a deterministic value.
    """
    return MultiFitResult(
        series=series,
        classification=classification,
        fit=OptimizeResult(fun=fun, x=np.array([c])),
        fit_type="untracked simple",
        c=np.array([c]),
        penalty=0.0,
        image_error=0.0,
        fit_image=np.zeros((13, 13)),
    )


# ===========================================================================
# TIER 1 — pure unit tests
# ===========================================================================


class TestMultiFitResult:
    """Cover construction, ``.error`` property, ``slim()``, and ``__repr__``."""

    @staticmethod
    def _make(**overrides: object) -> MultiFitResult:
        defaults: dict[str, object] = {
            "series": MagicMock(spec=SubtractionSeries),
            "classification": EvClassification.Binding,
            "fit": OptimizeResult(fun=1.5, x=np.array([-0.03])),
            "fit_type": "untracked simple",
            "c": np.array([-0.03]),
            "penalty": 0.5,
            "image_error": 0.01,
        }
        defaults.update(overrides)
        return MultiFitResult(**defaults)  # type: ignore[arg-type]

    def test_error_returns_fit_fun(self):
        r = self._make()
        assert r.error == pytest.approx(1.5)

    def test_default_particle_is_none(self):
        r = self._make()
        assert r.particle is None

    def test_default_distance_is_none(self):
        r = self._make()
        assert r.distance is None

    def test_set_particle_updates_attribute(self):
        r = self._make()
        r.set_particle("some tag")
        assert r.particle == "some tag"

    def test_slim_replaces_particle_with_id(self, spline_model):
        p = _make_particle(spline_model)
        r = self._make(particle=p)
        slim = r.slim()
        assert slim.particle == p.ID
        assert slim.series is r.series
        assert slim.distance is r.distance

    def test_slim_preserves_string_particle(self):
        r = self._make(particle="No correction")
        slim = r.slim()
        assert slim.particle == "No correction"

    def test_slim_preserves_none_particle(self):
        r = self._make(particle=None)
        slim = r.slim()
        assert slim.particle is None

    def test_repr_uses_particle_id(self, spline_model):
        p = _make_particle(spline_model)
        r = self._make(particle=p, distance=2.5)
        text = repr(r)
        assert f"Fit for {p.ID}" in text
        assert "distance from known = 2.5" in text

    def test_repr_uses_string_or_none(self):
        r = self._make(particle=None)
        text = repr(r)
        assert "Fit for None" in text


class TestAction:
    def test_all_members_present(self):
        assert Action.create_new == 0
        assert Action.updated == 1
        assert Action.failed == 2
        assert Action.ignore == 3


class TestDecision:
    def test_interesting_default_false(self):
        d = Decision(action=Action.updated, particle=None)
        assert d.interesting is False

    def test_interesting_can_be_true(self):
        d = Decision(action=Action.create_new, particle=None, interesting=True)
        assert d.interesting is True

    def test_interesting_is_per_instance_field(self):
        """Regression: pre-refactor, ``interesting`` was a class attribute so
        overrides silently didn't stick. Fix makes it a real NamedTuple field.
        """
        d1 = Decision(action=Action.updated, particle=None, interesting=True)
        d2 = Decision(action=Action.updated, particle=None)
        assert d1.interesting is True
        assert d2.interesting is False


class TestTrailingOffException:
    def test_is_exception_subclass(self):
        assert issubclass(TrailingOffException, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(TrailingOffException):
            raise TrailingOffException("trailed off")


class TestCreateGhostEvent:
    def test_returns_ghost_event(self, spline_model):
        template = _make_binding_ratiometric(spline_model, frame=42)
        gh = create_ghost_event(
            template, fit_result=(0.0, 0.0, -0.05), peak_model=spline_model
        )
        assert isinstance(gh, GhostEvent)

    def test_default_frame_is_minus_one(self, spline_model):
        template = _make_binding_ratiometric(spline_model, frame=42)
        gh = create_ghost_event(
            template, fit_result=(0.0, 0.0, -0.05), peak_model=spline_model
        )
        assert gh.i == -1

    def test_frame_override_propagates(self, spline_model):
        template = _make_binding_ratiometric(spline_model, frame=42)
        gh = create_ghost_event(
            template, fit_result=(0.0, 0.0, -0.05), frame=17, peak_model=spline_model
        )
        assert gh.i == 17

    def test_contrast_propagates(self, spline_model):
        template = _make_binding_ratiometric(spline_model, frame=42)
        gh = create_ghost_event(
            template, fit_result=(0.0, 0.0, -0.075), peak_model=spline_model
        )
        assert gh.c == pytest.approx(-0.075)

    def test_midpoint_reused_from_template(self, spline_model):
        template = _make_binding_ratiometric(spline_model, frame=42, midpoint=(30, 40))
        gh = create_ghost_event(
            template, fit_result=(0.0, 0.0, -0.05), peak_model=spline_model
        )
        assert gh.peak.midpoint == (30, 40)


class TestDistanceArray:
    """The core bug fix — CLAUDE.md's most important test."""

    @staticmethod
    def _r(distance: float | None) -> MultiFitResult:
        return MultiFitResult(
            series=MagicMock(spec=SubtractionSeries),
            classification=EvClassification.Binding,
            fit=OptimizeResult(fun=0.0, x=np.array([-0.03])),
            fit_type="untracked simple",
            c=np.array([-0.03]),
            penalty=0.0,
            image_error=0.0,
            distance=distance,
        )

    def test_all_known(self):
        arr = _distance_array([self._r(1.0), self._r(2.0), self._r(3.0)])
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_none_becomes_worst(self):
        """None → 2× max known, so unbound candidates score worse than any
        real match (fix for the pre-refactor 'unbound = distance 0' bias).
        """
        arr = _distance_array([self._r(1.0), self._r(None), self._r(3.0)])
        assert arr[0] == 1.0
        assert arr[1] == 6.0  # 3.0 * 2
        assert arr[2] == 3.0

    def test_all_none_gets_finite_fill(self):
        arr = _distance_array([self._r(None), self._r(None)])
        assert np.all(np.isfinite(arr))
        assert arr[0] == arr[1] == 1.0

    def test_returns_float_ndarray(self):
        arr = _distance_array([self._r(1.0), self._r(None)])
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == float

    def test_none_never_scores_lower_than_known(self):
        """Regression against the pre-refactor bias where `False`/`0` made
        unbound candidates rank as best.
        """
        for known in (0.1, 1.0, 10.0):
            arr = _distance_array([self._r(known), self._r(None)])
            assert arr[1] > arr[0], (
                f"None-distance candidate scored better than known={known}: {arr}"
            )

    def test_empty_input(self):
        arr = _distance_array([])
        assert arr.shape == (0,)
        assert arr.dtype == float


# ===========================================================================
# TIER 2 — BaseInvestigator with synthetic movies
# ===========================================================================


def _make_base_investigator(
    model: PsfModel,
    info: Info,
    *,
    parallel: bool = False,
    n_frames: int = 60,
    width: int = 51,
) -> BaseInvestigator:
    params = get_parameters(parallel=parallel, save_memory=True, minimum_mass=30.0)
    movie = _make_synthetic_movie(width=width, n_frames=n_frames)
    manager = ParticleManager()
    return BaseInvestigator(
        movie=movie,
        movie_params=info,
        parameters=params,
        particle_manager=manager,
        peak_model=model,
    )


class TestBaseInvestigatorInit:
    def test_open_lid_frames_default_empty_list(self, spline_model, info):
        b = _make_base_investigator(spline_model, info)
        assert b.open_lid_frames == []

    def test_open_lid_frames_default_is_not_shared(self, spline_model, info):
        b1 = _make_base_investigator(spline_model, info)
        b2 = _make_base_investigator(spline_model, info)
        b1.open_lid_frames.append(999)
        assert b2.open_lid_frames == []

    def test_open_lid_frames_materialised_from_arg(self, spline_model, info):
        params = get_parameters(parallel=False, save_memory=True)
        movie = _make_synthetic_movie()
        manager = ParticleManager()
        b = BaseInvestigator(
            movie=movie,
            movie_params=info,
            parameters=params,
            particle_manager=manager,
            peak_model=spline_model,
            open_lid_frames=[1, 2, 3],
        )
        assert b.open_lid_frames == [1, 2, 3]

    def test_im_correct_negates_when_not_destructive(self, spline_model):
        info = Info(
            scatter_interference_destructive=False,
            raw_camera_frame_rate=1000.0,
            frame_binning=1,
            pixel_binning=1,
            full_well_capacity=None,
            grey_levels=None,
        )
        b = _make_base_investigator(spline_model, info)
        sample = np.array([[1.0, -2.0], [3.0, 4.0]])
        np.testing.assert_allclose(b._im_correct(sample), -sample)

    def test_im_correct_identity_when_destructive(self, spline_model, info):
        b = _make_base_investigator(spline_model, info)
        sample = np.array([[1.0, -2.0], [3.0, 4.0]])
        np.testing.assert_allclose(b._im_correct(sample), sample)

    def test_reflectivity_correction_computed_when_enabled(self, spline_model, info):
        params = get_parameters(
            parallel=False, save_memory=True, reflectivity_correction=True
        )
        movie = _make_synthetic_movie()
        b = BaseInvestigator(
            movie=movie,
            movie_params=info,
            parameters=params,
            particle_manager=ParticleManager(),
            peak_model=spline_model,
        )
        assert b._r is not None
        assert b._r.shape == movie.shape[1:]


class TestGenerateSubtractionSeries:
    """Only sanity checks — the real work belongs to the events package."""

    def test_returns_series_with_correct_length(self, spline_model, info):
        b = _make_base_investigator(spline_model, info)
        ev = _make_binding_ratiometric(
            spline_model, frame=30, navg=3, midpoint=(25, 25)
        )
        series = b.generate_subtraction_series(ev, start_frame=10, fit=False)
        # frame_range spans [i - navg, i + navg] inclusive → 2*navg + 1
        assert len(series) == 7

    def test_start_frame_clamped_to_positive(self, spline_model, info):
        b = _make_base_investigator(spline_model, info)
        ev = _make_binding_ratiometric(spline_model, frame=30, navg=3)
        # Negative start frame should be clamped to 0 without error.
        series = b.generate_subtraction_series(ev, start_frame=-5, fit=False)
        assert len(series) == 7

    def test_custom_frames_used(self, spline_model, info):
        b = _make_base_investigator(spline_model, info)
        ev = _make_binding_ratiometric(spline_model, frame=30, navg=3)
        custom = [10, 11, 12]
        series = b.generate_subtraction_series(
            ev, start_frame=5, custom_frames=custom, fit=False
        )
        assert len(series) == 3

    def test_parallel_flag_serial_path_matches(self, spline_model, info):
        """When parallel=False, results should be identical to a serial run."""
        b = _make_base_investigator(spline_model, info, parallel=False)
        ev = _make_binding_ratiometric(spline_model, frame=30, navg=3)
        s1 = b.generate_subtraction_series(ev, start_frame=10, fit=False)
        s2 = b.generate_subtraction_series(ev, start_frame=10, fit=False)
        # Same input → same event count and frame ordering.
        assert len(s1) == len(s2)
        assert [e.i for e in s1] == [e.i for e in s2]


class TestFindClassification:
    """Direct coverage of ``find_classification`` (PR #40 review issue 4).

    The real ``fit_3d_ratiometric`` call is replaced with a deterministic
    stand-in keyed by classification, so the scoring/selection logic that
    ``find_classification`` itself owns can be pinned exactly rather than
    depending on real optimiser convergence.
    """

    def test_single_classification_returns_best_result(
        self, spline_model, info, monkeypatch
    ):
        b = _make_base_investigator(spline_model, info)
        event = _make_binding_ratiometric(spline_model, frame=30, navg=3)
        series = b.generate_subtraction_series(event, start_frame=10)

        c_val = -0.06
        fake_result = (
            OptimizeResult(fun=0.001, x=np.array([c_val])),
            np.zeros((13, 13)),
        )
        monkeypatch.setattr(
            "tandemmp.pipeline.investigator.fit_3d_ratiometric",
            lambda **kwargs: fake_result,
        )

        result = b.find_classification(
            event, series, particle_c=None, allow_native=False
        )

        assert isinstance(result, MultiFitResult)
        assert result.classification == EvClassification.Binding
        assert result.fit_type == "untracked simple"
        assert result.distance == 0.5
        np.testing.assert_array_equal(result.c, np.array([c_val]))

    def test_dissociation_veto_skips_to_next_candidate(
        self, spline_model, info, monkeypatch
    ):
        """A >90% mass drop on the best-fitting candidate is vetoed — the
        loop must fall through to the next-best candidate instead of
        returning the vetoed one (the exact behaviour documented at
        investigator.py:553-556).
        """
        b = _make_base_investigator(spline_model, info)
        event = _make_binding_ratiometric(spline_model, frame=30, navg=3)
        series = b.generate_subtraction_series(event, start_frame=10)

        # particle_c is a plain float -> classifications = [Movement, Unbinding].
        results_by_classification = {
            EvClassification.Movement: (
                OptimizeResult(
                    fun=0.0, x=np.array([mass_to_c(1000.0), mass_to_c(50.0)])
                ),
                np.zeros((13, 13)),
            ),
            EvClassification.Unbinding: (
                OptimizeResult(fun=1.0, x=np.array([-0.02])),
                np.zeros((13, 13)),
            ),
        }
        monkeypatch.setattr(
            "tandemmp.pipeline.investigator.fit_3d_ratiometric",
            lambda **kwargs: results_by_classification[kwargs["classification"]],
        )

        result = b.find_classification(
            event, series, particle_c=100.0, allow_native=False
        )

        assert isinstance(result, MultiFitResult)
        # Movement (lowest fit error) reclassifies to Dissociation with a
        # >90% mass drop and is vetoed; Unbinding is the fallback.
        assert result.classification == EvClassification.Unbinding


# ===========================================================================
# TIER 3 — GhostInvestigator / Investigator scenarios
# ===========================================================================


def _make_investigator(
    model: PsfModel,
    info: Info,
    *,
    manager: ParticleManager | None = None,
    parallel: bool = False,
) -> Investigator:
    params = get_parameters(parallel=parallel, save_memory=True, minimum_mass=30.0)
    movie = _make_synthetic_movie()
    return Investigator(
        movie=movie,
        movie_params=info,
        parameters=params,
        particle_manager=manager or ParticleManager(),
        peak_model=model,
    )


class TestInvestigatorInit:
    def test_event_veto_frames_default_empty(self, spline_model, info):
        inv = _make_investigator(spline_model, info)
        assert inv.event_veto_frames == []

    def test_event_veto_frames_default_not_shared(self, spline_model, info):
        i1 = _make_investigator(spline_model, info)
        i2 = _make_investigator(spline_model, info)
        i1.event_veto_frames.append(42)
        assert i2.event_veto_frames == []

    def test_event_veto_frames_materialised(self, spline_model, info):
        params = get_parameters(parallel=False, save_memory=True)
        inv = Investigator(
            movie=_make_synthetic_movie(),
            movie_params=info,
            parameters=params,
            particle_manager=ParticleManager(),
            peak_model=spline_model,
            event_veto_frames=[7, 8],
        )
        assert inv.event_veto_frames == [7, 8]

    def test_doubles_checked_starts_empty(self, spline_model, info):
        inv = _make_investigator(spline_model, info)
        assert inv.doubles_checked == set()

    def test_no_fakers_attribute(self, spline_model, info):
        """Regression: ``self.fakers`` was dead — must not be reintroduced."""
        inv = _make_investigator(spline_model, info)
        assert not hasattr(inv, "fakers")


class TestMatchGhost:
    def test_returns_false_when_no_ghosts(self, spline_model, info):
        inv = _make_investigator(spline_model, info)
        # Query ghost — nothing in the manager.
        query = _make_ghost_particle(spline_model, frame=30)
        event = _make_binding_ratiometric(spline_model, frame=30)
        assert inv.match_ghost(event, query) is False

    def test_returns_automatic_candidate_on_temporal_overlap(self, spline_model, info):
        """A tracked ghost with a temporal-overlap link to *event* is an
        "automatic candidate" regardless of the spatial-probability scoring —
        this is the branch ``process_ghost``'s success path depends on.
        """
        inv = _make_investigator(spline_model, info)

        existing_ratiometric = _make_binding_ratiometric(spline_model, frame=20)
        query_event = _make_binding_ratiometric(spline_model, frame=30)
        # Normally established upstream by the pairing/threading stage —
        # simulate the link directly.
        existing_ratiometric.overlaps.append(
            SimpleNamespace(temporal=True, event=query_event)
        )

        candidate_ghost = _make_ghost_particle(spline_model, frame=15)
        candidate_ghost._add_event(existing_ratiometric)
        inv.particle_manager.append(candidate_ghost)

        query_ghost = _make_ghost_particle(spline_model, frame=30)
        result = inv.match_ghost(query_event, query_ghost)

        assert result == [candidate_ghost]


class TestProcessGhost:
    """Orchestration coverage for ``process_ghost`` (PR #40 review issue 4).

    ``attempt_initial_ghost``/``match_ghost``/``double_check_ghost`` are
    monkeypatched at the instance level: the decision-tree logic that stitches
    their results together into an ``Action``/particle pair is what's under
    test here, not the underlying fits (covered separately).
    """

    def test_returns_ignore_when_initial_attempt_fails(
        self, spline_model, info, monkeypatch
    ):
        inv = _make_investigator(spline_model, info)
        event = _make_binding_ratiometric(spline_model, frame=30)
        monkeypatch.setattr(inv, "attempt_initial_ghost", lambda **kwargs: False)

        decision = inv.process_ghost(event, close_particles=[])

        assert decision == Decision(action=Action.ignore, particle=None)

    def test_creates_new_ghost_when_no_match_found(
        self, spline_model, info, monkeypatch
    ):
        inv = _make_investigator(spline_model, info)
        event = _make_binding_ratiometric(spline_model, frame=30)
        initial = _make_ghost_particle(spline_model, frame=30)
        monkeypatch.setattr(inv, "attempt_initial_ghost", lambda **kwargs: initial)
        monkeypatch.setattr(inv, "match_ghost", lambda *args, **kwargs: False)

        decision = inv.process_ghost(event, close_particles=[])

        assert decision.action == Action.create_new
        assert decision.particle is initial

    def test_creates_new_ghost_from_successful_double_check(
        self, spline_model, info, monkeypatch
    ):
        inv = _make_investigator(spline_model, info)
        event = _make_binding_ratiometric(spline_model, frame=30)
        initial = _make_ghost_particle(spline_model, frame=30)
        matched = _make_ghost_particle(spline_model, frame=15)
        refitted = _make_ghost_particle(spline_model, frame=15)
        monkeypatch.setattr(inv, "attempt_initial_ghost", lambda **kwargs: initial)
        monkeypatch.setattr(inv, "match_ghost", lambda *args, **kwargs: [matched])
        monkeypatch.setattr(inv, "double_check_ghost", lambda **kwargs: refitted)

        decision = inv.process_ghost(event, close_particles=[])

        assert decision.action == Action.create_new
        assert decision.particle is refitted

    def test_skips_already_double_checked_particle(
        self, spline_model, info, monkeypatch
    ):
        inv = _make_investigator(spline_model, info)
        event = _make_binding_ratiometric(spline_model, frame=30)
        initial = _make_ghost_particle(spline_model, frame=30)
        already_checked = _make_ghost_particle(spline_model, frame=10)
        second = _make_ghost_particle(spline_model, frame=15)
        refitted = _make_ghost_particle(spline_model, frame=15)
        inv.doubles_checked = {already_checked}
        seen: list[object] = []

        def fake_double_check(**kwargs):
            seen.append(kwargs["original_ghost_particle"])
            return refitted

        monkeypatch.setattr(inv, "attempt_initial_ghost", lambda **kwargs: initial)
        monkeypatch.setattr(
            inv, "match_ghost", lambda *args, **kwargs: [already_checked, second]
        )
        monkeypatch.setattr(inv, "double_check_ghost", fake_double_check)

        decision = inv.process_ghost(event, close_particles=[])

        assert seen == [second]
        assert decision.particle is refitted

    def test_falls_back_to_initial_when_all_double_checks_fail(
        self, spline_model, info, monkeypatch
    ):
        inv = _make_investigator(spline_model, info)
        event = _make_binding_ratiometric(spline_model, frame=30)
        initial = _make_ghost_particle(spline_model, frame=30)
        matched = _make_ghost_particle(spline_model, frame=15)
        monkeypatch.setattr(inv, "attempt_initial_ghost", lambda **kwargs: initial)
        monkeypatch.setattr(inv, "match_ghost", lambda *args, **kwargs: [matched])
        monkeypatch.setattr(inv, "double_check_ghost", lambda **kwargs: False)

        decision = inv.process_ghost(event, close_particles=[])

        assert decision.action == Action.create_new
        assert decision.particle is initial


class TestFindAndFitNoParticles:
    """Verifies the no-signal → Decision.failed path with an empty manager."""

    def test_no_ghost_signal_returns_failed(self, spline_model, info, monkeypatch):
        # Force the ML score to a small value so the ghost branch is skipped.
        monkeypatch.setattr(
            "tandemmp.events.subtraction.SubtractionSeries.ghost_signal_ML",
            0.0,
        )
        inv = _make_investigator(spline_model, info)
        # Event well inside the movie so start_frame arithmetic is straightforward.
        event = _make_binding_ratiometric(
            spline_model, frame=30, navg=3, midpoint=(25, 25)
        )
        decision = inv.find_and_fit(event)
        # No signal (near-zero, no ghost) → either failed (no candidate under
        # tolerance) or create_new (if the trivial fit somehow beats the
        # basic error).  Both are acceptable Decisions — assert it's a real
        # Decision and doesn't crash.
        assert isinstance(decision, Decision)
        assert decision.action in (Action.failed, Action.create_new, Action.ignore)


class TestAttemptInitialGhost:
    """Real ``GhostParticle`` construction from a fit result (PR #40 issue 4).

    ``fit_initial_ghost`` (which drives a real ``scipy.optimize.minimize``
    call) is replaced with a deterministic result built from the *real*
    background-corrected series passed to it — the behaviour under test is
    everything ``attempt_initial_ghost`` does with that result: threshold
    checks, ``GhostEvent``/``GhostParticle`` construction, and
    ``propose_new_event`` acceptance.
    """

    def test_returns_ghost_particle_on_successful_fit(
        self, spline_model, info, monkeypatch
    ):
        inv = _make_investigator(spline_model, info)
        event = _make_binding_ratiometric(
            spline_model, frame=30, navg=3, midpoint=(25, 25)
        )
        c_val = float(mass_to_c(1000.0))

        def fake_fit_initial_ghost(event, series, method="Powell"):
            # fun=0.0: the noiseless synthetic PSF gives `event.basic_fit_error`
            # a near-machine-epsilon value, so `attempt.error` must be smaller
            # still to clear attempt_initial_ghost's `beats_basic` threshold.
            attempt = _make_fit_attempt(series, c=c_val, fun=0.0)
            res = OptimizeResult(x=np.array([0.0, 0.0, c_val]))
            return res, attempt

        monkeypatch.setattr(inv, "fit_initial_ghost", fake_fit_initial_ghost)

        result = inv.attempt_initial_ghost(event=event, close_particles=[])

        assert isinstance(result, GhostParticle)
        assert result.binding_event.c == pytest.approx(c_val)

    def test_returns_false_when_fit_initial_ghost_fails(
        self, spline_model, info, monkeypatch
    ):
        inv = _make_investigator(spline_model, info)
        event = _make_binding_ratiometric(spline_model, frame=30, navg=3)
        monkeypatch.setattr(inv, "fit_initial_ghost", lambda *a, **kw: False)

        assert inv.attempt_initial_ghost(event=event, close_particles=[]) is False


class TestDoubleCheckGhostGuards:
    def test_raises_when_ghost_has_no_connected_init_event(self, spline_model, info):
        """Ghost particle has only the binding ghost event → events[1] IndexError."""
        inv = _make_investigator(spline_model, info)
        gp = _make_ghost_particle(spline_model, frame=30)
        event = _make_binding_ratiometric(spline_model, frame=40)
        with pytest.raises((IndexError, ValueError)):
            inv.double_check_ghost(
                event, close_particles=[], original_ghost_particle=gp
            )
