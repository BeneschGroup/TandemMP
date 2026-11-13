"""Stub classes for unpickling legacy TandemMP result files.

They mirror the field layout of the old containers from ``tandemmp.io.save`` and
carry no behaviour: :func:`tandemmp.io.save._LegacyUnpickler` remaps legacy class
paths onto them so :mod:`pickle` can rebuild old payloads, and
:func:`tandemmp.io.save._migrate` converts the result to a current
:class:`~tandemmp.io.save.ResultsFile`.

The stubs are deliberately permissive: no field validation, no
re-interpolation, and arbitrary state via ``__dict__``.
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

    The original interpolated in ``__post_init__``; the stub deliberately does not.
    Pickle bypasses ``__init__`` and restores state through ``__dict__``, so an empty
    class lets the unpickler land the saved attributes (``particle``, ``contrasts``,
    ``frames``, …).
    """


class LegacySaveFile(NamedTuple):
    """Stub for the old top-level ``SaveFile`` NamedTuple."""

    mp_filepath: str
    movie: Any
    parameters: Any
    vector: Any
    r_events: LegacyRatiometricData
    particles: list[Any]


# Legacy `tandemmp.parameters` NamedTuples
#
# Old `SaveFile.parameters` payloads reference these classes by their old module path
# (`tandemmp.parameters`, deleted with no forwarding shim when it moved to
# `core.parameters`). Field layouts are copied verbatim so positional unpickling lands
# correctly. They are not migrated to the current dataclasses: `ResultsFile.parameters`
# is typed `Any`, so passing the legacy value through unmigrated is enough to load.


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

    The original set ``x``/``y``/``_index`` in ``__init__``; as with
    :class:`LegacyPartData`, pickle restores them through ``__dict__``, so an empty
    class suffices.
    """
