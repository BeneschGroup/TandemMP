"""Result serialisation for TandemMP analyses.

Provides picklable dataclass containers that capture the per-event and
per-particle outputs of an analysis, along with :func:`save_results` and
:func:`load_results` helpers.

The container types (:class:`EventRecord`, :class:`ParticleRecord`,
:class:`ResultsFile`) are plain dataclasses with no dependency on
higher-level pipeline objects, so they are safe to unpickle in a
minimal environment.

The builder helpers (:func:`build_event_records`,
:func:`build_particle_records`) accept event and particle
objects and extract the subset of fields that should persist on disk.
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
    """Serialisable snapshot of a single ratiometric event.

    Position/contrast/mass fields preserve whatever shape the source
    :class:`~tandemmp.events.RatiometricEvent` carries — scalars for
    basic-fit events and per-frame arrays for events that went through 3D
    ratiometric fitting. This mirrors the pre-refactor :class:`SaveFile`
    output, which stored ``e.c``, ``e.m`` etc. directly without flattening.
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
    #: Duplicates ``basic_fit_error``. Mirrors the pre-refactor
    #: ``RatiometricData.old_errors`` field, which was always populated as
    #: ``[e.basic_fit_error for e in events]`` regardless of whether
    #: ``old_masses`` was supplied — kept as a separate field for schema
    #: compatibility with the legacy format rather than being dropped.
    old_error: float | None = None


@dataclasses.dataclass(frozen=True)
class ParticleRecord:
    """Serialisable snapshot of a single tracked particle.

    Per-event fields are stored as Python lists, mirroring the pre-refactor
    ``PartData`` container. Each list element is whatever the source event
    held (a scalar for basic-fit events; a per-frame array for 3D-fit
    events). Use :meth:`interpolate` to obtain a contiguous frame range and
    linearly interpolated values for any 1-D scalar field.
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

        NaNs in *values* are dropped before interpolation. Returns
        ``(frames, interpolated_values)``.
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
    """Extract the ``action.name`` field off a Decision, if present."""
    if decision is None:
        return None
    action = getattr(decision, "action", None)
    if action is None:
        return None
    return getattr(action, "name", str(action))


def _classification_rep(classification: object) -> str | None:
    """Return the ``rep`` field off a Classification, if present."""
    if classification is None:
        return None
    return getattr(classification, "rep", str(classification))


def build_event_records(
    events: list[RatiometricEvent],
    *,
    old_masses: list[_FloatOrArray] | None = None,
) -> list[EventRecord]:
    """Build :class:`EventRecord` list from RatiometricEvents.

    Mirrors the pre-refactor ``save_experiment`` extraction: fields are
    pulled directly off the event without scalarisation, so 3D-fit events
    keep their per-frame arrays in ``contrast``/``mass``/``x_pos``/etc.
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
    """Build :class:`ParticleRecord` list from Particles.

    Per-event fields (``contrasts``, ``masses``, ``x_pos`` …) are stored as
    Python lists exactly as the pre-refactor ``PartData`` did before
    interpolation — preserving whatever per-event shape the source events
    carry, including 3D-fit traces.
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
    """Unpickler that rewrites pre-refactor class paths onto stub classes.

    Old result files reference ``tandemmp.io.save.SaveFile`` and friends, as
    well as the legacy ``tandemmp.parameters`` and
    ``tandemmp.DriftCorrector`` modules embedded in a
    ``SaveFile``'s ``parameters``/``vector`` fields. Those modules/classes
    were moved or deleted during the refactor with no forwarding shim left
    behind; ``find_class`` reroutes them to the stubs in
    :mod:`tandemmp.io._legacy` so unpickling succeeds.
    """

    def find_class(self, module: str, name: str) -> type:
        remapped = _LEGACY_CLASS_REMAP.get((module, name))
        if remapped is not None:
            module, name = remapped
        return cast(type, super().find_class(module, name))


def legacy_pickle_load(fh: IO[bytes]) -> object:
    """Unpickle *fh*, remapping pre-refactor TandemMP class paths onto stubs.

    A drop-in replacement for ``pickle.load`` for callers that unpickle
    arbitrary objects which may *reference* legacy TandemMP classes
    somewhere in their object graph — e.g. a post-processing
    result whose ``part_data`` field is a pre-refactor ``PartData`` — not
    just top-level ``SaveFile``/``PartData`` payloads. :func:`load_results`
    is the schema-enforcing counterpart for actual :class:`ResultsFile`
    payloads; this is the unopinionated primitive it's built on.
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
        # `rdata.old_errors` is the legacy analogue of `basic_fit_error`
        # (`[e.basic_fit_error for e in events]` in the pre-refactor
        # `save_experiment`); recovered here rather than discarded so it
        # matches `build_event_records`, which always populates `old_error`.
        old_error=(
            None
            if rdata.old_errors is None or idx >= len(rdata.old_errors)
            else float(rdata.old_errors[idx])
        ),
    )


def _migrate_particle(p: object) -> ParticleRecord:
    """Build a :class:`ParticleRecord` from a legacy PartData/ParticleData.

    The legacy ``hess_inv`` field (inverse Hessian from the optimizer) is
    intentionally dropped: pre-refactor ``save_experiment`` already wrote it
    as ``[None for e in fixed_events]`` — it was never populated with real
    data — so there is nothing to carry forward.
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

    Pre-refactor pickles are remapped onto stubs in :mod:`tandemmp.io._legacy`
    and then converted to the current schema. Two legacy shapes are
    supported: a top-level ``SaveFile`` NamedTuple (events + particles +
    movie + parameters), and a bare ``[PartData, ...]`` list — some
    pre-refactor scripts pickled particle lists directly with no enclosing
    ``SaveFile``. The bare-list case returns a plain ``list[ParticleRecord]``
    rather than a :class:`ResultsFile`, since there is no movie/parameters/
    events context to populate.
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
