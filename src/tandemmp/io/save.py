"""Result serialisation for TandemMP analyses.

Picklable dataclass containers (:class:`EventRecord`, :class:`ParticleRecord`,
:class:`ResultsFile`) hold per-event and per-particle outputs, saved and loaded
with :func:`save_results` / :func:`load_results`. They depend on no pipeline
objects, so they unpickle in a minimal environment.
:func:`build_event_records` and :func:`build_particle_records` extract the
fields that persist.
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from tandemmp.core.conversions import Calibration
from tandemmp.io._legacy import (
    LegacyPartData,
    LegacyParticleData,
    LegacyRatiometricData,
    LegacySaveFile,
)

if TYPE_CHECKING:
    from tandemmp.events import RatiometricEvent
    from tandemmp.tracking.particle import Particle


# Data containers


_FloatOrArray = float | npt.NDArray[np.floating]


@dataclasses.dataclass(frozen=True)
class EventRecord:
    """Serialisable snapshot of one ratiometric event.

    Position/contrast/mass fields keep the source event's shape: scalars for
    basic-fit events, per-frame arrays for 3-D-fit events.
    """

    contrast: _FloatOrArray
    mass: _FloatOrArray
    frame: int
    x_pos: _FloatOrArray
    y_pos: _FloatOrArray
    dx: _FloatOrArray
    dy: _FloatOrArray
    error: float
    basic_fit_error: float
    snr: float
    classification: str | None
    decision: str | None
    particle: int | str | None
    direct_overlap: bool
    image: npt.NDArray[np.floating]
    old_mass: _FloatOrArray | None = None
    #: Duplicates ``basic_fit_error``; kept for compatibility with the legacy schema
    #: (``old_errors`` was always ``[e.basic_fit_error for e in events]``).
    old_error: float | None = None


@dataclasses.dataclass(frozen=True)
class ParticleRecord:
    """Serialisable snapshot of one tracked particle.

    Per-event fields are Python lists whose elements keep the source event's shape
    (scalar for basic-fit, per-frame array for 3-D-fit). :meth:`interpolate` gives a
    contiguous frame range with linearly interpolated values for any 1-D scalar
    field.
    """

    particle: int | str
    binding_frame: int
    unbinding_frame: int | None
    is_ghost: bool
    contrasts: list[_FloatOrArray]
    masses: list[_FloatOrArray]
    x_pos: list[_FloatOrArray]
    y_pos: list[_FloatOrArray]
    dx: list[_FloatOrArray]
    dy: list[_FloatOrArray]
    frames: list[int]
    errors: list[float]
    snrs: list[float]
    images: list[npt.NDArray[np.floating]]

    def interpolate(
        self, values: npt.ArrayLike
    ) -> tuple[
        npt.NDArray[np.int_],
        npt.NDArray[np.floating],
    ]:
        """Linearly interpolate *values* onto the continuous frame range.

        NaNs are dropped first. Returns ``(frames, interpolated_values)``.
        """
        arr = np.asarray(values, dtype=float)
        frames = np.asarray(self.frames, dtype=int)
        if arr.shape != frames.shape:
            msg = f"values shape {arr.shape} does not match frames shape {frames.shape}"
            raise ValueError(msg)
        sel = ~np.isnan(arr)
        if not sel.any():
            return np.array([], dtype=int), np.array([], dtype=float)
        times = frames[sel]
        vals = arr[sel]
        new_range = np.arange(times.min(), times.max() + 1)
        return new_range, np.interp(new_range, times, vals)

    def __repr__(self) -> str:
        return (
            f"ParticleRecord(particle={self.particle!r}, "
            f"frames={len(self.frames)} frames, "
            f"binding_frame={self.binding_frame}, "
            f"unbinding_frame={self.unbinding_frame}, "
            f"is_ghost={self.is_ghost})"
        )


@dataclasses.dataclass(frozen=True)
class ResultsFile:
    """Top-level pickled payload for a completed TandemMP analysis."""

    source_filepath: str
    parameters: Any
    drift_vector: Any | None
    events: list[EventRecord]
    particles: list[ParticleRecord]
    movie: npt.NDArray[np.floating] | None = None
    calibration: Calibration | None = None

    def __repr__(self) -> str:
        n_frames = "no movie" if self.movie is None else f"{len(self.movie)} frames"
        return (
            f"ResultsFile(source={self.source_filepath!r}, {n_frames}, "
            f"{len(self.events)} events, {len(self.particles)} particles)"
        )


# Builders


def _decision_name(decision: object) -> str | None:
    """``action.name`` of a Decision, if present."""
    if decision is None:
        return None
    action = getattr(decision, "action", None)
    if action is None:
        return None
    return getattr(action, "name", str(action))


def _classification_rep(classification: object) -> str | None:
    """``rep`` of a Classification, if present."""
    if classification is None:
        return None
    return getattr(classification, "rep", str(classification))


def build_event_records(
    events: list[RatiometricEvent],
    *,
    old_masses: list[_FloatOrArray] | None = None,
) -> list[EventRecord]:
    """Build an :class:`EventRecord` list from RatiometricEvents.

    Fields are pulled off each event without scalarisation, so 3-D-fit events keep
    their per-frame arrays.
    """
    records: list[EventRecord] = []
    if old_masses is not None and len(old_masses) != len(events):
        msg = "old_masses must match events length"
        raise ValueError(msg)

    for idx, ev in enumerate(events):
        records.append(
            EventRecord(
                contrast=ev.c,
                mass=ev.m,
                frame=int(ev.i),
                x_pos=ev.x,
                y_pos=ev.y,
                dx=ev.dx,
                dy=ev.dy,
                error=float(ev.fit_3D_error if ev.fit_3D else ev.basic_fit_error),
                basic_fit_error=float(ev.basic_fit_error),
                snr=float(ev.snr),
                classification=_classification_rep(ev.classification),
                decision=_decision_name(ev.decision),
                particle=ev.particle,
                direct_overlap=len(ev.temporal_overlaps) > 0,
                image=np.asarray(ev.peak.image),
                old_mass=None if old_masses is None else old_masses[idx],
                old_error=ev.basic_fit_error,
            )
        )
    return records


def build_particle_records(
    particles: list[Particle],
) -> list[ParticleRecord]:
    """Build a :class:`ParticleRecord` list from Particles.

    Per-event fields (``contrasts``, ``masses``, ``x_pos`` …) are stored as lists
    keeping each event's own shape, including 3-D-fit traces.
    """
    from tandemmp.tracking.particle import GhostParticle

    records: list[ParticleRecord] = []
    for p in particles:
        evs = p.fixed_background_events
        records.append(
            ParticleRecord(
                particle=p.ID,
                binding_frame=p.binding_frame,
                unbinding_frame=p.unbinding_frame,
                is_ghost=isinstance(p, GhostParticle),
                contrasts=[e.c for e in evs],
                masses=[e.m for e in evs],
                x_pos=[e.x for e in evs],
                y_pos=[e.y for e in evs],
                dx=[e.dx for e in evs],
                dy=[e.dy for e in evs],
                frames=[e.i for e in evs],
                errors=[e.basic_fit_error for e in evs],
                snrs=[e.snr for e in evs],
                images=[e.peak.image for e in evs],
            )
        )
    return records


# Pickle helpers


def save_results(filepath: str | Path, results: ResultsFile) -> Path:
    """Pickle *results* to *filepath*, creating parent directories as needed.

    Returns the resolved path.
    """
    if not isinstance(results, ResultsFile):
        msg = f"results must be a ResultsFile, got {type(results).__name__}"
        raise TypeError(msg)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(results, fh)
    return path


_LEGACY_CLASS_REMAP: dict[tuple[str, str], tuple[str, str]] = {
    ("tandemmp.io.save", "SaveFile"): ("tandemmp.io._legacy", "LegacySaveFile"),
    ("tandemmp.io.save", "RatiometricData"): (
        "tandemmp.io._legacy",
        "LegacyRatiometricData",
    ),
    ("tandemmp.io.save", "PartData"): ("tandemmp.io._legacy", "LegacyPartData"),
    ("tandemmp.io.save", "ParticleData"): (
        "tandemmp.io._legacy",
        "LegacyParticleData",
    ),
    ("tandemmp.parameters", "FullAnalysisParams"): (
        "tandemmp.io._legacy",
        "LegacyFullAnalysisParams",
    ),
    ("tandemmp.parameters", "TandemMPAnalysisParams"): (
        "tandemmp.io._legacy",
        "LegacyTandemMPAnalysisParams",
    ),
    ("tandemmp.parameters", "DriftCorrectionParams"): (
        "tandemmp.io._legacy",
        "LegacyDriftCorrectionParams",
    ),
    ("tandemmp.parameters", "FirstAnalysisParams"): (
        "tandemmp.io._legacy",
        "LegacyFirstAnalysisParams",
    ),
    ("tandemmp.parameters", "ExperimentalPsfParams"): (
        "tandemmp.io._legacy",
        "LegacyExperimentalPsfParams",
    ),
    ("tandemmp.parameters", "ContinuationParams"): (
        "tandemmp.io._legacy",
        "LegacyContinuationParams",
    ),
    ("tandemmp.DriftCorrector", "Vector"): (
        "tandemmp.io._legacy",
        "LegacyVector",
    ),
}


class _LegacyUnpickler(pickle.Unpickler):
    """Unpickler mapping legacy class paths onto stubs in :mod:`tandemmp.io._legacy`.

    Old result files reference ``tandemmp.io.save.SaveFile`` and the removed
    ``tandemmp.parameters`` / ``tandemmp.DriftCorrector`` modules (inside a
    ``SaveFile``'s ``parameters``/``vector`` fields); ``find_class`` reroutes them so
    unpickling succeeds.
    """

    def find_class(self, module: str, name: str) -> type:
        remapped = _LEGACY_CLASS_REMAP.get((module, name))
        if remapped is not None:
            module, name = remapped
        return cast(type, super().find_class(module, name))


def legacy_pickle_load(fh: IO[bytes]) -> object:
    """Unpickle *fh*, mapping legacy TandemMP class paths onto stubs.

    A drop-in ``pickle.load`` for objects that may *reference* legacy classes
    anywhere in their graph (e.g. a post-processing result holding a legacy
    ``PartData``), not just top-level ``SaveFile``/``PartData`` payloads.
    :func:`load_results` is the schema-enforcing version for :class:`ResultsFile`.
    """
    return _LegacyUnpickler(fh).load()


def _migrate_event(rdata: LegacyRatiometricData, idx: int) -> EventRecord:
    """Build an :class:`EventRecord` from one slot of a legacy RatiometricData."""

    def _at(seq: list[Any], default: Any = None) -> Any:  # noqa: ANN401
        if seq is None or idx >= len(seq):
            return default
        return seq[idx]

    return EventRecord(
        contrast=_at(rdata.contrasts, 0.0),
        mass=_at(rdata.masses, 0.0),
        frame=int(_at(rdata.frames, 0)),
        x_pos=_at(rdata.x_pos, 0.0),
        y_pos=_at(rdata.y_pos, 0.0),
        dx=_at(rdata.dx, 0.0),
        dy=_at(rdata.dy, 0.0),
        error=float(_at(rdata.errors, 0.0)),
        basic_fit_error=float(_at(rdata.old_errors, _at(rdata.errors, 0.0))),
        snr=float(_at(rdata.snrs, 0.0)),
        classification=_at(rdata.classifications),
        decision=_at(rdata.decisions),
        particle=_at(rdata.particle),
        direct_overlap=bool(_at(rdata.direct_overlap, False)),
        image=np.asarray(_at(rdata.images, np.zeros((0, 0)))),
        old_mass=(
            None
            if rdata.old_masses is None or idx >= len(rdata.old_masses)
            else rdata.old_masses[idx]
        ),
        # `rdata.old_errors` is the legacy `basic_fit_error`; recovered so this matches
        # `build_event_records`, which always populates `old_error`.
        old_error=(
            None
            if rdata.old_errors is None or idx >= len(rdata.old_errors)
            else float(rdata.old_errors[idx])
        ),
    )


def _migrate_particle(p: object) -> ParticleRecord:
    """Build a :class:`ParticleRecord` from a legacy PartData/ParticleData.

    The legacy ``hess_inv`` field is dropped: it was always written as
    ``[None, ...]``, so there is nothing to carry forward.
    """
    return ParticleRecord(
        particle=getattr(p, "particle", 0),
        binding_frame=getattr(p, "binding_frame", 0),
        unbinding_frame=getattr(p, "unbinding_frame", None),
        is_ghost=bool(getattr(p, "is_ghost", False)),
        contrasts=list(getattr(p, "contrasts", [])),
        masses=list(getattr(p, "masses", [])),
        x_pos=list(getattr(p, "x_pos", [])),
        y_pos=list(getattr(p, "y_pos", [])),
        dx=list(getattr(p, "dx", [])),
        dy=list(getattr(p, "dy", [])),
        frames=list(getattr(p, "frames", [])),
        errors=list(getattr(p, "errors", [])),
        snrs=list(getattr(p, "snrs", [])),
        images=list(getattr(p, "images", [])),
    )


def _migrate(obj: object) -> ResultsFile:
    """Convert a legacy :class:`LegacySaveFile` into a current :class:`ResultsFile`."""
    if not isinstance(obj, LegacySaveFile):
        msg = (
            f"cannot migrate object of type {type(obj).__name__}; expected a "
            "ResultsFile or a legacy SaveFile"
        )
        raise TypeError(msg)

    rdata = obj.r_events
    n_events = 0 if rdata is None else len(rdata.contrasts)
    events = [_migrate_event(rdata, i) for i in range(n_events)]
    particles = [_migrate_particle(p) for p in (obj.particles or [])]

    return ResultsFile(
        source_filepath=str(obj.mp_filepath),
        parameters=obj.parameters,
        drift_vector=obj.vector,
        events=events,
        particles=particles,
        movie=obj.movie,
    )


def load_results(filepath: str | Path) -> ResultsFile | list[ParticleRecord]:
    """Unpickle a :class:`ResultsFile` from *filepath*.

    Legacy pickles are remapped onto stubs in :mod:`tandemmp.io._legacy` and
    converted to the current schema. Two legacy shapes are supported: a top-level
    ``SaveFile`` (events + particles + movie + parameters), and a bare
    ``[PartData, ...]`` list, which returns a plain ``list[ParticleRecord]`` since
    there is no other context to populate a :class:`ResultsFile`.
    """
    path = Path(filepath)
    with path.open("rb") as fh:
        obj = legacy_pickle_load(fh)  # noqa: S301 — trusted internal format
    if isinstance(obj, ResultsFile):
        return obj
    if isinstance(obj, LegacySaveFile):
        return _migrate(obj)
    if (
        isinstance(obj, list)
        and obj
        and all(isinstance(p, LegacyPartData | LegacyParticleData) for p in obj)
    ):
        return [_migrate_particle(p) for p in obj]
    msg = (
        f"{path} does not contain a ResultsFile, legacy SaveFile, "
        "or legacy particle list"
    )
    raise TypeError(msg)
