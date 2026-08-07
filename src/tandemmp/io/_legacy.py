"""Stub classes for unpickling pre-refactor TandemMP result files.

These types mirror the field layout of the legacy containers that lived in
``tandemmp.io.save`` before the Layer 3 refactor. They carry no behaviour —
:func:`tandemmp.io.save._LegacyUnpickler` remaps the legacy class paths onto
these stubs so :mod:`pickle` can reconstruct old payloads, and
:func:`tandemmp.io.save._migrate` converts the result into a current
:class:`~tandemmp.io.save.ResultsFile`.

The stubs are deliberately permissive — they do not validate fields, do not
re-run interpolation, and accept arbitrary state via ``__dict__``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

__all__ = [
    "LegacyContinuationParams",
    "LegacyDriftCorrectionParams",
    "LegacyExperimentalPsfParams",
    "LegacyFirstAnalysisParams",
    "LegacyFullAnalysisParams",
    "LegacyParticleData",
    "LegacyPartData",
    "LegacyRatiometricData",
    "LegacySaveFile",
    "LegacyTandemMPAnalysisParams",
    "LegacyVector",
]


class LegacyRatiometricData(NamedTuple):
    """Stub for the old ``RatiometricData`` NamedTuple (16 parallel lists)."""

    old_masses: list[Any]
    old_errors: list[Any]
    contrasts: list[Any]
    x_pos: list[Any]
    y_pos: list[Any]
    dx: list[Any]
    dy: list[Any]
    masses: list[Any]
    frames: list[Any]
    classifications: list[Any]
    decisions: list[Any]
    particle: list[Any]
    errors: list[Any]
    direct_overlap: list[Any]
    snrs: list[Any]
    images: list[Any]


class LegacyParticleData(NamedTuple):
    """Stub for the old ``ParticleData`` NamedTuple."""

    particle: int
    contrasts: list[Any]
    x_pos: list[Any]
    y_pos: list[Any]
    dx: list[Any]
    dy: list[Any]
    masses: list[Any]
    frames: list[Any]
    errors: list[Any]
    hess_inv: list[Any]
    snrs: list[Any]
    images: list[Any]
    binding_frame: int
    unbinding_frame: int | None
    is_ghost: bool


class LegacyPartData:
    """Stub for the old ``PartData`` frozen dataclass.

    The original ran linear interpolation in ``__post_init__``; we deliberately
    skip that. Pickle bypasses ``__init__`` entirely and restores state through
    ``__dict__``, so this empty class is sufficient for the unpickler to land
    the saved attributes (``particle``, ``contrasts``, ``frames``, …).
    """


class LegacySaveFile(NamedTuple):
    """Stub for the old top-level ``SaveFile`` NamedTuple."""

    mp_filepath: str
    movie: Any
    parameters: Any
    vector: Any
    r_events: LegacyRatiometricData
    particles: list[Any]


# ---------------------------------------------------------------------------
# Legacy `tandemmp.parameters` NamedTuples
#
# Pre-refactor `SaveFile.parameters` payloads reference these classes by
# their old module path (`tandemmp.parameters`, deleted wholesale in Layer
# 0g's move to `core.parameters` — no forwarding shim was left behind). The
# field layout below is copied verbatim from the pre-refactor module so
# positional unpickling lands correctly; these stubs are not migrated onto
# the current `core.parameters` dataclasses; `ResultsFile.parameters` is
# typed `Any`, so passing the legacy value through unmigrated is sufficient
# to unblock loading.
# ---------------------------------------------------------------------------


class LegacyContinuationParams(NamedTuple):
    """Stub for the old ``ContinuationParams`` NamedTuple."""

    spatial_threshold: float = 4.0
    mass_fluctuation: float = 0.15
    max_strikes: int = 10


class LegacyExperimentalPsfParams(NamedTuple):
    """Stub for the old ``ExperimentalPsfParams`` NamedTuple."""

    sampling_points: float = 0.1
    selection_percentile: float = 95
    spatial_expansion: int = 2
    report: bool = True


class LegacyDriftCorrectionParams(NamedTuple):
    """Stub for the old ``DriftCorrectionParams`` NamedTuple."""

    frame_gap: int = 25
    z_drift: bool = False
    debug: bool = False
    show_vector: bool = True


class LegacyFirstAnalysisParams(NamedTuple):
    """Stub for the old ``FirstAnalysisParams`` NamedTuple."""

    navg: int = 5
    minimum_navg: int = 3
    l_thum: int = 13
    experimental_psf_model: bool | LegacyExperimentalPsfParams = (
        LegacyExperimentalPsfParams()
    )


class LegacyTandemMPAnalysisParams(NamedTuple):
    """Stub for the old ``TandemMPAnalysisParams`` NamedTuple."""

    particle_continuation: bool | LegacyContinuationParams = LegacyContinuationParams()
    psf_radius: float = 6.5
    minimum_matching_probability: float = 0.05
    ghost_signal_check: float = 0.45
    correct_with_images: bool = False
    complex_fit_error_tolerance: float = 1.04
    mass_similarity_flexibility: float = 1.01
    keep_background_low: bool = True
    correct_mass_shifts_on_binding: bool = False
    check_ghosts_from_start: bool = False
    max_ghost_matches: int = 3
    smart_averaging: bool = False
    constant_refresh: bool = False


class LegacyFullAnalysisParams(NamedTuple):
    """Stub for the old top-level ``FullAnalysisParams`` NamedTuple."""

    drift_correction: bool | LegacyDriftCorrectionParams = LegacyDriftCorrectionParams()
    lid_open_threshold: float = 1
    lip_open_margin: int = 100
    illumination_correction: bool = False
    minimum_mass: float = 50
    snr_limit: float = 0.2
    skip_overlaps: bool = False
    first_analysis: LegacyFirstAnalysisParams = LegacyFirstAnalysisParams()
    tandemmp_analysis: LegacyTandemMPAnalysisParams = LegacyTandemMPAnalysisParams()
    parallel: bool = True
    nanopore: bool = False
    save_memory: bool = False
    supervise: bool = False
    use_rust: bool = True
    reflectivity_correction: bool = False
    auto_bin: bool = True
    save_exp_psf: bool = False


class LegacyVector:
    """Stub for the old ``tandemmp.DriftCorrector.Vector`` plain class.

    The original stored ``x``/``y``/``_index`` via ``__init__``. Pickle
    bypasses ``__init__`` and restores state through ``__dict__``, so this
    empty class is sufficient to land the saved attributes — same pattern as
    :class:`LegacyPartData`.
    """
