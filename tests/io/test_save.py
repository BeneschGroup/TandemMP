"""Tests for :mod:`tandemmp.io.save`."""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from tandemmp.core.conversions import Calibration
from tandemmp.io.save import (
    EventRecord,
    ParticleRecord,
    ResultsFile,
    build_event_records,
    build_particle_records,
    legacy_pickle_load,
    load_results,
    save_results,
)

# Fixtures / fakes


def _make_event(
    *,
    c: float = -5e-4,
    m: float = 120.0,
    i: int = 10,
    x: float = 20.5,
    y: float = 15.2,
    dx: float = 0.1,
    dy: float = -0.2,
    basic_fit_error: float = 1e-4,
    fit_3d: bool = False,
    fit_3d_error: float = 2e-4,
    snr: float = 3.5,
    classification_rep: str | None = "Binding",
    decision_action_name: str | None = "accept",
    particle: int | None = 7,
    n_overlaps: int = 0,
    image: np.ndarray | None = None,
    old_basic_error: float = 1.1e-4,
) -> types.SimpleNamespace:
    if image is None:
        image = np.zeros((5, 5), dtype=float)
    classification = (
        None
        if classification_rep is None
        else types.SimpleNamespace(rep=classification_rep)
    )
    decision = (
        None
        if decision_action_name is None
        else types.SimpleNamespace(
            action=types.SimpleNamespace(name=decision_action_name)
        )
    )
    basic_fit = types.SimpleNamespace(error=old_basic_error)
    peak = types.SimpleNamespace(image=image)
    return types.SimpleNamespace(
        c=c,
        m=m,
        i=i,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        basic_fit_error=basic_fit_error,
        fit_3D=fit_3d,
        fit_3D_error=fit_3d_error,
        snr=snr,
        classification=classification,
        decision=decision,
        particle=particle,
        temporal_overlaps=[object()] * n_overlaps,
        peak=peak,
        _basic_fit=basic_fit,
    )


def _make_particle(
    *,
    particle_id: int,
    frames: list[int],
    binding_frame: int = 5,
    unbinding_frame: int | None = 50,
) -> types.SimpleNamespace:
    fixed_events = [
        _make_event(i=f, x=1.0 + f, y=2.0 + f, basic_fit_error=1e-4, snr=4.0)
        for f in frames
    ]
    p = types.SimpleNamespace(
        ID=particle_id,
        binding_frame=binding_frame,
        unbinding_frame=unbinding_frame,
        fixed_background_events=fixed_events,
    )
    return p


# EventRecord / ParticleRecord / ResultsFile basics


class TestRecordDataclasses:
    def test_event_record_is_frozen(self) -> None:
        rec = EventRecord(
            contrast=-1e-4,
            mass=100.0,
            frame=5,
            x_pos=1.0,
            y_pos=2.0,
            dx=0.0,
            dy=0.0,
            error=1e-4,
            basic_fit_error=1e-4,
            snr=3.0,
            classification="Binding",
            decision="accept",
            particle=0,
            direct_overlap=False,
            image=np.zeros((3, 3)),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.mass = 200.0  # type: ignore[misc]

    def test_particle_record_is_frozen(self) -> None:
        rec = ParticleRecord(
            particle=1,
            binding_frame=0,
            unbinding_frame=10,
            is_ghost=False,
            contrasts=[0.0, 0.0, 0.0],
            masses=[0.0, 0.0, 0.0],
            x_pos=[0.0, 0.0, 0.0],
            y_pos=[0.0, 0.0, 0.0],
            dx=[0.0, 0.0, 0.0],
            dy=[0.0, 0.0, 0.0],
            frames=[0, 1, 2],
            errors=[0.0, 0.0, 0.0],
            snrs=[0.0, 0.0, 0.0],
            images=[np.zeros((3, 3))],
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.particle = 2  # type: ignore[misc]

    def test_results_file_repr_without_movie(self) -> None:
        rf = ResultsFile(
            source_filepath="foo.mp",
            parameters=None,
            drift_vector=None,
            events=[],
            particles=[],
            movie=None,
        )
        r = repr(rf)
        assert "foo.mp" in r
        assert "no movie" in r
        assert "0 events" in r
        assert "0 particles" in r

    def test_results_file_repr_with_movie(self) -> None:
        rf = ResultsFile(
            source_filepath="foo.mp",
            parameters=None,
            drift_vector=None,
            events=[],
            particles=[],
            movie=np.zeros((7, 4, 4)),
        )
        assert "7 frames" in repr(rf)

    def test_results_file_calibration_defaults_to_none(self) -> None:
        rf = ResultsFile(
            source_filepath="foo.mp",
            parameters=None,
            drift_vector=None,
            events=[],
            particles=[],
        )
        assert rf.calibration is None


# ParticleRecord.interpolate


class TestParticleRecordInterpolate:
    def _rec(self, frames: np.ndarray) -> ParticleRecord:
        n = len(frames)
        zeros = [0.0] * n
        return ParticleRecord(
            particle=0,
            binding_frame=int(frames[0]),
            unbinding_frame=int(frames[-1]),
            is_ghost=False,
            contrasts=zeros,
            masses=zeros,
            x_pos=zeros,
            y_pos=zeros,
            dx=zeros,
            dy=zeros,
            frames=[int(f) for f in frames],
            errors=zeros,
            snrs=zeros,
            images=[],
        )

    def test_linear_interpolation_of_gaps(self) -> None:
        rec = self._rec(np.array([0, 2, 4]))
        values = np.array([0.0, 2.0, 4.0])
        new_frames, new_vals = rec.interpolate(values)
        np.testing.assert_array_equal(new_frames, np.arange(0, 5))
        np.testing.assert_allclose(new_vals, np.arange(0.0, 5.0))

    def test_nans_are_dropped(self) -> None:
        rec = self._rec(np.array([0, 1, 2]))
        new_frames, new_vals = rec.interpolate(np.array([0.0, np.nan, 2.0]))
        np.testing.assert_array_equal(new_frames, np.arange(0, 3))
        np.testing.assert_allclose(new_vals, np.array([0.0, 1.0, 2.0]))

    def test_all_nan_returns_empty(self) -> None:
        rec = self._rec(np.array([0, 1, 2]))
        new_frames, new_vals = rec.interpolate(np.full(3, np.nan))
        assert new_frames.size == 0
        assert new_vals.size == 0

    def test_shape_mismatch_raises(self) -> None:
        rec = self._rec(np.array([0, 1, 2]))
        with pytest.raises(ValueError):
            rec.interpolate(np.array([0.0, 1.0]))


# build_event_records


class TestBuildEventRecords:
    def test_basic_build(self) -> None:
        events = [_make_event(i=1), _make_event(i=2, c=1e-4)]
        records = build_event_records(events)
        assert len(records) == 2
        assert all(isinstance(r, EventRecord) for r in records)
        assert records[0].frame == 1
        assert records[1].frame == 2
        assert records[0].old_mass is None

    def test_3d_fit_uses_fit3d_error(self) -> None:
        ev = _make_event(fit_3d=True, fit_3d_error=9e-5, basic_fit_error=1e-3)
        [rec] = build_event_records([ev])
        assert rec.error == pytest.approx(9e-5)
        assert rec.basic_fit_error == pytest.approx(1e-3)

    def test_basic_fit_uses_basic_error(self) -> None:
        ev = _make_event(fit_3d=False, basic_fit_error=2e-4)
        [rec] = build_event_records([ev])
        assert rec.error == pytest.approx(2e-4)

    def test_classification_none_is_none(self) -> None:
        ev = _make_event(classification_rep=None)
        [rec] = build_event_records([ev])
        assert rec.classification is None

    def test_decision_none_is_none(self) -> None:
        ev = _make_event(decision_action_name=None)
        [rec] = build_event_records([ev])
        assert rec.decision is None

    def test_direct_overlap_flag(self) -> None:
        ev_with = _make_event(n_overlaps=2)
        ev_without = _make_event(n_overlaps=0)
        r1, r2 = build_event_records([ev_with, ev_without])
        assert r1.direct_overlap is True
        assert r2.direct_overlap is False

    def test_old_masses_attached(self) -> None:
        events = [_make_event(i=1), _make_event(i=2)]
        records = build_event_records(events, old_masses=[100.0, 200.0])
        assert records[0].old_mass == pytest.approx(100.0)
        assert records[1].old_mass == pytest.approx(200.0)

    def test_old_masses_wrong_length_raises(self) -> None:
        events = [_make_event(), _make_event()]
        with pytest.raises(ValueError):
            build_event_records(events, old_masses=[1.0])

    def test_old_error_mirrors_basic_fit_error(self) -> None:
        """`old_error` duplicates `basic_fit_error`, as legacy ``old_errors`` always did.

        That held whether or not `old_masses` was supplied.
        """
        ev = _make_event(basic_fit_error=3.3e-4)
        [rec] = build_event_records([ev])
        assert rec.old_error == pytest.approx(3.3e-4)
        assert rec.old_error == pytest.approx(rec.basic_fit_error)

    def test_array_contrast_preserved(self) -> None:
        """3-D-fit events store array contrast/mass; the record passes them through.

        Like legacy ``RatiometricData.contrasts`` (``[e.c for e in events]``), an array
        ``e.c`` lands on disk as an array, with no scalar collapse.
        """
        arr = np.array([-1e-4, -2e-4, -3e-4])
        ev = _make_event(c=arr)
        [rec] = build_event_records([ev])
        assert isinstance(rec.contrast, np.ndarray)
        np.testing.assert_array_equal(rec.contrast, arr)


# build_particle_records


class TestBuildParticleRecords:
    def test_basic_build(self) -> None:
        ps = [_make_particle(particle_id=0, frames=[5, 7, 9])]
        [rec] = build_particle_records(ps)
        assert rec.particle == 0
        assert rec.binding_frame == 5
        assert rec.unbinding_frame == 50
        assert rec.is_ghost is False
        np.testing.assert_array_equal(rec.frames, [5, 7, 9])
        assert len(rec.images) == 3

    def test_ghost_detection(self) -> None:
        from tandemmp.tracking.particle import GhostParticle

        # object.__new__ bypasses __init__ (which requires a GhostEvent).
        # fixed_background_events is a property that reads _event_map, so we
        # mock that with a SimpleNamespace carrying the right dict attributes.
        fake_ev = _make_event(i=1)
        p = object.__new__(GhostParticle)
        p.ID = 3
        p.binding_frame = 1
        p.unbinding_frame = None
        p._event_map = types.SimpleNamespace(
            connected={},
            disconnected={1: fake_ev},
        )
        [rec] = build_particle_records([p])
        assert rec.is_ghost is True

    def test_unbinding_none_preserved(self) -> None:
        p = _make_particle(particle_id=0, frames=[1, 2], unbinding_frame=None)
        [rec] = build_particle_records([p])
        assert rec.unbinding_frame is None


# Pickle round-trip


class TestSaveLoadRoundTrip:
    def test_round_trip_preserves_content(self, tmp_path) -> None:
        events = build_event_records([_make_event(i=1), _make_event(i=2)])
        particles = build_particle_records(
            [_make_particle(particle_id=0, frames=[1, 2])]
        )
        rf = ResultsFile(
            source_filepath="/tmp/some.mp",
            parameters={"navg": 5},
            drift_vector=None,
            events=events,
            particles=particles,
            movie=None,
        )
        path = tmp_path / "results.pkl"
        returned = save_results(path, rf)
        assert returned == path
        loaded = load_results(path)
        assert isinstance(loaded, ResultsFile)
        assert loaded.source_filepath == rf.source_filepath
        assert len(loaded.events) == 2
        assert len(loaded.particles) == 1

    def test_round_trip_preserves_calibration(self, tmp_path) -> None:
        calibration = Calibration(slope=-0.00005, offset=-0.0001)
        rf = ResultsFile(
            source_filepath="/tmp/some.mp",
            parameters=None,
            drift_vector=None,
            events=[],
            particles=[],
            movie=None,
            calibration=calibration,
        )
        path = tmp_path / "results.pkl"
        save_results(path, rf)
        loaded = load_results(path)
        assert loaded.calibration == calibration

    def test_save_non_results_raises(self, tmp_path) -> None:
        with pytest.raises(TypeError):
            save_results(tmp_path / "x.pkl", {"not": "a ResultsFile"})  # type: ignore[arg-type]

    def test_save_creates_missing_parent_directories(self, tmp_path) -> None:
        rf = ResultsFile(
            source_filepath="/tmp/some.mp",
            parameters=None,
            drift_vector=None,
            events=[],
            particles=[],
            movie=None,
        )
        path = tmp_path / "nested" / "subdir" / "results.pkl"
        assert not path.parent.exists()
        returned = save_results(path, rf)
        assert returned == path
        assert path.exists()
        assert isinstance(load_results(path), ResultsFile)

    def test_load_wrong_type_raises(self, tmp_path) -> None:
        import pickle

        path = tmp_path / "bad.pkl"
        with path.open("wb") as fh:
            pickle.dump({"wrong": "object"}, fh)
        with pytest.raises(TypeError):
            load_results(path)


# Backward compatibility: loading pre-refactor SaveFile pickles


def _write_legacy_pickle(path, *, n_events: int, n_particles: int) -> None:
    """Write a pickle mimicking the legacy ``tandemmp.io.save`` format.

    Recreates ``SaveFile``, ``RatiometricData`` and ``PartData`` under their legacy
    module paths, pickles, then removes the temporary classes.
    """
    import pickle as _pickle
    import sys
    from typing import NamedTuple

    class _RatiometricData(NamedTuple):
        old_masses: list
        old_errors: list
        contrasts: list
        x_pos: list
        y_pos: list
        dx: list
        dy: list
        masses: list
        frames: list
        classifications: list
        decisions: list
        particle: list
        errors: list
        direct_overlap: list
        snrs: list
        images: list

    class _PartData:
        pass

    class _SaveFile(NamedTuple):
        mp_filepath: str
        movie: object
        parameters: object
        vector: object
        r_events: _RatiometricData
        particles: list

    save_mod = sys.modules["tandemmp.io.save"]

    _RatiometricData.__module__ = "tandemmp.io.save"
    _RatiometricData.__qualname__ = "RatiometricData"
    _PartData.__module__ = "tandemmp.io.save"
    _PartData.__qualname__ = "PartData"
    _SaveFile.__module__ = "tandemmp.io.save"
    _SaveFile.__qualname__ = "SaveFile"

    save_mod.RatiometricData = _RatiometricData  # type: ignore[attr-defined]
    save_mod.PartData = _PartData  # type: ignore[attr-defined]
    save_mod.SaveFile = _SaveFile  # type: ignore[attr-defined]
    try:
        rdata = _RatiometricData(
            old_masses=[100.0 + i for i in range(n_events)],
            old_errors=[1.1e-4] * n_events,
            contrasts=[-(i + 1) * 1e-4 for i in range(n_events)],
            x_pos=[float(i) for i in range(n_events)],
            y_pos=[float(i) + 0.5 for i in range(n_events)],
            dx=[0.1] * n_events,
            dy=[-0.1] * n_events,
            masses=[110.0 + i for i in range(n_events)],
            frames=list(range(n_events)),
            classifications=["Binding"] * n_events,
            decisions=["accept"] * n_events,
            particle=list(range(n_events)),
            errors=[1e-4] * n_events,
            direct_overlap=[False] * n_events,
            snrs=[3.5] * n_events,
            images=[np.zeros((3, 3)) for _ in range(n_events)],
        )

        particles: list = []
        for pid in range(n_particles):
            p = _PartData()
            p.particle = pid
            p.contrasts = [-1e-4, -2e-4]
            p.x_pos = [1.0, 1.5]
            p.y_pos = [2.0, 2.5]
            p.dx = [0.0, 0.0]
            p.dy = [0.0, 0.0]
            p.masses = [100.0, 110.0]
            p.frames = [pid * 10, pid * 10 + 1]
            p.errors = [1e-4, 1e-4]
            p.hess_inv = [None, None]
            p.snrs = [3.5, 3.6]
            p.images = [np.zeros((3, 3)), np.zeros((3, 3))]
            p.binding_frame = pid * 10
            p.unbinding_frame = pid * 10 + 5
            p.is_ghost = pid == 1
            p.continuous = True
            particles.append(p)

        sf = _SaveFile(
            mp_filepath="/legacy/path.mp",
            movie=None,
            parameters={"navg": 5},
            vector=None,
            r_events=rdata,
            particles=particles,
        )
        with path.open("wb") as fh:
            _pickle.dump(sf, fh)
    finally:
        del save_mod.RatiometricData
        del save_mod.PartData
        del save_mod.SaveFile


def _write_legacy_pickle_with_params(path) -> None:
    """Write a legacy ``SaveFile`` pickle referencing the removed parameter modules.

    Its ``parameters``/``vector`` fields use the legacy ``tandemmp.parameters`` and
    ``tandemmp.DriftCorrector`` module paths. Both were removed with no forwarding
    shim, so a real legacy file fails to unpickle (``ModuleNotFoundError``) unless
    remapped. Uses the real ``Legacy*`` stubs so the pickled fields match their count
    and order exactly, as NamedTuple unpickling is positional.
    """
    import pickle as _pickle
    import sys
    import types

    from tandemmp.io._legacy import (
        LegacyContinuationParams,
        LegacyDriftCorrectionParams,
        LegacyExperimentalPsfParams,
        LegacyFirstAnalysisParams,
        LegacyFullAnalysisParams,
        LegacyPartData,
        LegacyRatiometricData,
        LegacySaveFile,
        LegacyTandemMPAnalysisParams,
        LegacyVector,
    )

    legacy_locations: dict[str, dict[str, type]] = {
        "tandemmp.io.save": {
            "SaveFile": LegacySaveFile,
            "RatiometricData": LegacyRatiometricData,
            "PartData": LegacyPartData,
        },
        "tandemmp.parameters": {
            "FullAnalysisParams": LegacyFullAnalysisParams,
            "DriftCorrectionParams": LegacyDriftCorrectionParams,
            "FirstAnalysisParams": LegacyFirstAnalysisParams,
            "ExperimentalPsfParams": LegacyExperimentalPsfParams,
            "TandemMPAnalysisParams": LegacyTandemMPAnalysisParams,
            "ContinuationParams": LegacyContinuationParams,
        },
        "tandemmp.DriftCorrector": {"Vector": LegacyVector},
    }

    original_identity: dict[type, tuple[str, str]] = {}
    installed_modules: list[str] = []
    save_mod = sys.modules["tandemmp.io.save"]
    for mod_name, classes in legacy_locations.items():
        target_mod = (
            save_mod if mod_name == "tandemmp.io.save" else types.ModuleType(mod_name)
        )
        for cls_name, cls in classes.items():
            original_identity[cls] = (cls.__module__, cls.__qualname__)
            cls.__module__ = mod_name
            cls.__qualname__ = cls_name
            setattr(target_mod, cls_name, cls)
        if mod_name != "tandemmp.io.save":
            sys.modules[mod_name] = target_mod
            installed_modules.append(mod_name)

    try:
        vector = LegacyVector()
        vector.__dict__.update(x=[0.1, 0.2, 0.3], y=[-0.1, -0.2, -0.3], _index=0)

        params = LegacyFullAnalysisParams(
            minimum_mass=75.0,
            first_analysis=LegacyFirstAnalysisParams(
                navg=7, experimental_psf_model=LegacyExperimentalPsfParams()
            ),
            tandemmp_analysis=LegacyTandemMPAnalysisParams(
                particle_continuation=LegacyContinuationParams(max_strikes=5)
            ),
        )

        rdata = LegacyRatiometricData(
            old_masses=[100.0],
            old_errors=[1.1e-4],
            contrasts=[-1e-4],
            x_pos=[1.0],
            y_pos=[2.0],
            dx=[0.1],
            dy=[-0.1],
            masses=[110.0],
            frames=[0],
            classifications=["Binding"],
            decisions=["accept"],
            particle=[0],
            errors=[1e-4],
            direct_overlap=[False],
            snrs=[3.5],
            images=[np.zeros((3, 3))],
        )
        sf = LegacySaveFile(
            mp_filepath="/legacy/path.mp",
            movie=None,
            parameters=params,
            vector=vector,
            r_events=rdata,
            particles=[],
        )
        with path.open("wb") as fh:
            _pickle.dump(sf, fh)
    finally:
        for cls, (mod, qualname) in original_identity.items():
            cls.__module__ = mod
            cls.__qualname__ = qualname
        for mod_name in installed_modules:
            del sys.modules[mod_name]


class TestLegacyLoad:
    def test_load_legacy_savefile_returns_results_file(self, tmp_path) -> None:
        path = tmp_path / "legacy.pkl"
        _write_legacy_pickle(path, n_events=3, n_particles=2)
        rf = load_results(path)
        assert isinstance(rf, ResultsFile)
        assert rf.source_filepath == "/legacy/path.mp"
        assert len(rf.events) == 3
        assert len(rf.particles) == 2
        assert rf.calibration is None

    def test_legacy_event_fields_migrated(self, tmp_path) -> None:
        path = tmp_path / "legacy.pkl"
        _write_legacy_pickle(path, n_events=2, n_particles=0)
        rf = load_results(path)
        ev = rf.events[1]
        assert ev.frame == 1
        assert ev.contrast == pytest.approx(-2e-4)
        assert ev.mass == pytest.approx(111.0)
        assert ev.classification == "Binding"
        assert ev.decision == "accept"
        assert ev.error == pytest.approx(1e-4)
        assert ev.basic_fit_error == pytest.approx(1.1e-4)
        assert ev.old_mass == pytest.approx(101.0)
        assert ev.old_error == pytest.approx(1.1e-4)
        assert ev.image.shape == (3, 3)

    def test_legacy_particle_fields_migrated(self, tmp_path) -> None:
        path = tmp_path / "legacy.pkl"
        _write_legacy_pickle(path, n_events=0, n_particles=2)
        rf = load_results(path)
        p0, p1 = rf.particles
        assert p0.particle == 0
        assert p0.is_ghost is False
        assert p1.is_ghost is True
        assert p0.binding_frame == 0
        assert p0.unbinding_frame == 5
        np.testing.assert_array_equal(p0.frames, [0, 1])
        np.testing.assert_allclose(p0.masses, [100.0, 110.0])
        assert len(p0.images) == 2

    def test_legacy_round_trip_re_saves_as_results_file(self, tmp_path) -> None:
        legacy_path = tmp_path / "legacy.pkl"
        _write_legacy_pickle(legacy_path, n_events=1, n_particles=1)
        rf = load_results(legacy_path)
        new_path = tmp_path / "migrated.pkl"
        save_results(new_path, rf)
        reloaded = load_results(new_path)
        assert isinstance(reloaded, ResultsFile)
        assert reloaded.source_filepath == "/legacy/path.mp"
        assert len(reloaded.events) == 1
        assert len(reloaded.particles) == 1


# Backward compatibility: legacy `tandemmp.parameters` / `DriftCorrector`


class TestLegacyParametersAndVectorLoad:
    """Load legacy SaveFile parameters and vectors through their old module paths."""

    def test_load_does_not_raise_module_not_found(self, tmp_path) -> None:
        path = tmp_path / "legacy_with_params.pkl"
        _write_legacy_pickle_with_params(path)
        rf = load_results(path)
        assert isinstance(rf, ResultsFile)

    def test_parameters_remapped_to_legacy_stub(self, tmp_path) -> None:
        from tandemmp.io._legacy import (
            LegacyContinuationParams,
            LegacyExperimentalPsfParams,
            LegacyFirstAnalysisParams,
            LegacyFullAnalysisParams,
            LegacyTandemMPAnalysisParams,
        )

        path = tmp_path / "legacy_with_params.pkl"
        _write_legacy_pickle_with_params(path)
        rf = load_results(path)
        params = rf.parameters
        assert isinstance(params, LegacyFullAnalysisParams)
        assert params.minimum_mass == pytest.approx(75.0)
        assert isinstance(params.first_analysis, LegacyFirstAnalysisParams)
        assert params.first_analysis.navg == 7
        assert isinstance(
            params.first_analysis.experimental_psf_model, LegacyExperimentalPsfParams
        )
        assert isinstance(params.tandemmp_analysis, LegacyTandemMPAnalysisParams)
        assert isinstance(
            params.tandemmp_analysis.particle_continuation, LegacyContinuationParams
        )
        assert params.tandemmp_analysis.particle_continuation.max_strikes == 5

    def test_vector_remapped_to_legacy_stub(self, tmp_path) -> None:
        from tandemmp.io._legacy import LegacyVector

        path = tmp_path / "legacy_with_params.pkl"
        _write_legacy_pickle_with_params(path)
        rf = load_results(path)
        assert isinstance(rf.drift_vector, LegacyVector)
        assert rf.drift_vector.x == [0.1, 0.2, 0.3]
        assert rf.drift_vector.y == [-0.1, -0.2, -0.3]

    def test_round_trip_re_saves_as_results_file(self, tmp_path) -> None:
        legacy_path = tmp_path / "legacy_with_params.pkl"
        _write_legacy_pickle_with_params(legacy_path)
        rf = load_results(legacy_path)
        new_path = tmp_path / "migrated.pkl"
        save_results(new_path, rf)
        reloaded = load_results(new_path)
        assert isinstance(reloaded, ResultsFile)
        assert reloaded.parameters.minimum_mass == pytest.approx(75.0)


# Backward compatibility: loading a bare legacy particle list


def _write_bare_particle_pickle(path, *, n_particles: int) -> None:
    """Write a pickle mimicking a legacy bare ``[PartData, ...]`` dump.

    Some old scripts pickled a particle list directly, with no enclosing ``SaveFile``
    (e.g. ``pickle.dump([p for p in particles], f)``).
    """
    import pickle as _pickle
    import sys

    class _PartData:
        pass

    save_mod = sys.modules["tandemmp.io.save"]
    _PartData.__module__ = "tandemmp.io.save"
    _PartData.__qualname__ = "PartData"
    save_mod.PartData = _PartData  # type: ignore[attr-defined]
    try:
        particles = []
        for pid in range(n_particles):
            p = _PartData()
            p.particle = pid
            p.contrasts = [-1e-4, -2e-4]
            p.x_pos = [1.0, 1.5]
            p.y_pos = [2.0, 2.5]
            p.dx = [0.0, 0.0]
            p.dy = [0.0, 0.0]
            p.masses = [100.0, 110.0]
            p.frames = [pid * 10, pid * 10 + 1]
            p.errors = [1e-4, 1e-4]
            p.hess_inv = [None, None]
            p.snrs = [3.5, 3.6]
            p.images = [np.zeros((3, 3)), np.zeros((3, 3))]
            p.binding_frame = pid * 10
            p.unbinding_frame = pid * 10 + 5
            p.is_ghost = pid == 1
            particles.append(p)

        with path.open("wb") as fh:
            _pickle.dump(particles, fh)
    finally:
        del save_mod.PartData


class TestBareParticleListLoad:
    def test_load_bare_particle_list_returns_list_of_particle_records(
        self, tmp_path
    ) -> None:
        path = tmp_path / "parts.pickle"
        _write_bare_particle_pickle(path, n_particles=2)
        result = load_results(path)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(p, ParticleRecord) for p in result)

    def test_bare_particle_fields_migrated(self, tmp_path) -> None:
        path = tmp_path / "parts.pickle"
        _write_bare_particle_pickle(path, n_particles=2)
        p0, p1 = load_results(path)
        assert p0.particle == 0
        assert p0.is_ghost is False
        assert p1.is_ghost is True
        assert p0.binding_frame == 0
        assert p0.unbinding_frame == 5
        np.testing.assert_array_equal(p0.frames, [0, 1])
        np.testing.assert_allclose(p0.masses, [100.0, 110.0])
        assert len(p0.images) == 2

    def test_bare_particle_list_can_be_wrapped_and_round_tripped(
        self, tmp_path
    ) -> None:
        legacy_path = tmp_path / "parts.pickle"
        _write_bare_particle_pickle(legacy_path, n_particles=1)
        particles = load_results(legacy_path)
        rf = ResultsFile(
            source_filepath=str(legacy_path),
            parameters=None,
            drift_vector=None,
            events=[],
            particles=particles,
            movie=None,
        )
        new_path = tmp_path / "migrated.pkl"
        save_results(new_path, rf)
        reloaded = load_results(new_path)
        assert isinstance(reloaded, ResultsFile)
        assert len(reloaded.particles) == 1
        assert reloaded.events == []

    def test_empty_list_is_not_treated_as_particle_list(self, tmp_path) -> None:
        import pickle

        path = tmp_path / "empty.pkl"
        with path.open("wb") as fh:
            pickle.dump([], fh)
        with pytest.raises(TypeError):
            load_results(path)


# legacy_pickle_load: general-purpose legacy-aware pickle.load replacement


class TestLegacyPickleLoad:
    """``legacy_pickle_load`` is a drop-in ``pickle.load`` replacement.

    For callers (e.g. ``utils.load_from_pickle``) that unpickle arbitrary objects
    which may reference legacy TandemMP classes anywhere in their graph. Unlike
    :func:`load_results` it assumes nothing about the pickled shape and returns the
    unmigrated object.
    """

    def test_load_returns_plain_object_unmigrated(self, tmp_path) -> None:
        import pickle

        path = tmp_path / "plain.pkl"
        with path.open("wb") as fh:
            pickle.dump({"a": 1, "b": [1, 2, 3]}, fh)
        with path.open("rb") as fh:
            obj = legacy_pickle_load(fh)
        assert obj == {"a": 1, "b": [1, 2, 3]}

    def test_load_remaps_legacy_class_nested_in_arbitrary_container(
        self, tmp_path
    ) -> None:
        """A legacy ``PartData`` nested in an unrelated container still resolves.

        The container is neither a bare particle list nor a ``SaveFile``, so
        ``load_results`` would reject it.
        """
        import pickle as _pickle
        import sys

        from tandemmp.io._legacy import LegacyPartData

        class _PartData:
            pass

        class _Container:
            def __init__(self, part_data) -> None:
                self.part_data = part_data
                self.nlogP = np.array([1.0, 2.0])

        save_mod = sys.modules["tandemmp.io.save"]
        _PartData.__module__ = "tandemmp.io.save"
        _PartData.__qualname__ = "PartData"
        save_mod.PartData = _PartData  # type: ignore[attr-defined]
        _Container.__module__ = "tandemmp.io.save"
        _Container.__qualname__ = "_TestContainer"
        save_mod._TestContainer = _Container  # type: ignore[attr-defined]

        try:
            p = _PartData()
            p.particle = 5
            p.frames = [0, 1, 2]
            p.masses = [100.0, 101.0, 102.0]

            container = _Container(p)
            path = tmp_path / "nested.pkl"
            with path.open("wb") as fh:
                _pickle.dump([container], fh)

            # `_TestContainer` (unlike `PartData`) has no remap entry, so it
            # must still resolve on the real module while unpickling.
            with path.open("rb") as fh:
                [loaded] = legacy_pickle_load(fh)
        finally:
            del save_mod.PartData
            del save_mod._TestContainer

        assert isinstance(loaded.part_data, LegacyPartData)
        assert loaded.part_data.particle == 5
        assert loaded.part_data.frames == [0, 1, 2]
