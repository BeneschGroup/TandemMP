"""Analysis parameters for TandemMP — frozen dataclass hierarchy."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Any


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value}")


def _validate_percentile(value: float, name: str) -> None:
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{name} must be a percentile in [0, 100], got {value}")


def _validate_positive(value: float | int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be positive, got {value}")


# ---------------------------------------------------------------------------
# Sub-parameter groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftCorrectionParams:
    """Parameters for drift correction."""

    #: Number of frames between drift vector sample points.
    frame_gap: int = 25
    #: Check against immediately preceding frames rather than the initial;
    #: useful when z-drift changes images between the start and current frame.
    z_drift: bool = False
    #: Show each error landscape for calculating the drift vector.
    debug: bool = False
    #: Display drift vector plot once calculated.
    show_vector: bool = True

    def __post_init__(self) -> None:
        _validate_positive(self.frame_gap, "frame_gap")


@dataclass(frozen=True)
class ExperimentalPsfParams:
    """Parameters for experimental PSF model construction."""

    #: Grid spacing for the experimental PSF model (pixels).
    sampling_points: float = 0.1
    #: Percentile for event selection criteria using a combination of
    #: mass and error.
    selection_percentile: float = 95.0
    #: Number of pixels to expand thumbnails if possible.
    spatial_expansion: int = 2
    #: Display a report of the model after it has been constructed.
    report: bool = True

    def __post_init__(self) -> None:
        _validate_percentile(self.selection_percentile, "selection_percentile")
        _validate_positive(self.sampling_points, "sampling_points")
        _validate_positive(self.spatial_expansion, "spatial_expansion")


@dataclass(frozen=True)
class ContinuationParams:
    """Parameters for particle continuation tracking."""

    #: Distance (pixels) between a new and previous particle position
    #: above which the fit is marked as bad.
    spatial_threshold: float = 4.0
    #: Tolerance for relative mass fluctuation.
    mass_fluctuation: float = 0.15
    #: Running tally of bad fits before a particle is removed.
    #: Good fits reduce the tally back down.
    max_strikes: int = 10

    def __post_init__(self) -> None:
        _validate_positive(self.spatial_threshold, "spatial_threshold")
        _validate_positive(self.mass_fluctuation, "mass_fluctuation")
        _validate_positive(self.max_strikes, "max_strikes")


@dataclass(frozen=True)
class FirstAnalysisParams:
    """Parameters for initial ratiometric analysis."""

    #: Number of averaged frames used to make and find ratiometric events.
    navg: int = 5
    #: Minimum navg size for events during attempted resizing for
    #: complexity minimisation. Events that overlap with less than double
    #: this temporally will be reduced down.
    minimum_navg: int = 3
    #: Event thumbnail size — side length in pixels.
    l_thum: int = 13
    #: Experimental PSF model config, or ``False`` to disable.
    experimental_psf_model: ExperimentalPsfParams | bool = dataclasses.field(
        default_factory=ExperimentalPsfParams,
    )

    def __post_init__(self) -> None:
        _validate_positive(self.navg, "navg")
        _validate_positive(self.minimum_navg, "minimum_navg")
        _validate_positive(self.l_thum, "l_thum")


@dataclass(frozen=True)
class TandemMPAnalysisParams:
    """Parameters for the core TandemMP analysis loop."""

    #: Particle continuation config, or ``False`` to disable.
    particle_continuation: ContinuationParams | bool = dataclasses.field(
        default_factory=ContinuationParams,
    )
    #: Size of the defined PSF radius (pixels); used to detect when
    #: particles are interfering with one another.
    psf_radius: float = 6.5
    #: Minimum overlap of two normalised Gaussian distributions
    #: representing location for a particle/event(s) to be matched.
    minimum_matching_probability: float = 0.05
    #: Minimum signal strength to check for ghosts.
    ghost_signal_check: float = 0.45
    #: Use images as particle correction rather than reconstructed fits.
    #: NOT RECOMMENDED.
    correct_with_images: bool = False
    #: Allows the complex fit error to be worse than the basic by this
    #: multiplier and still be considered an improvement.
    complex_fit_error_tolerance: float = 1.04
    #: If the complex mass and basic mass are similar, there is an extra
    #: 2x multiplier to the tolerance.
    mass_similarity_flexibility: float = 1.01
    #: Keep the background estimate low during tracking — never move it
    #: up. Makes more sense off but seems to give better results on.
    keep_background_low: bool = True
    #: NOT IMPLEMENTED.
    correct_mass_shifts_on_binding: bool = False
    #: Use only the start of the movie to explicitly confirm ghosts.
    #: When off, allows recovery of particles mid-movie.
    check_ghosts_from_start: bool = False
    #: Maximum number of particles to consider when spatially matching
    #: ghosts. Can be slow; you'd hope the closest is correct.
    max_ghost_matches: int = 3
    #: Smart averaging to increase SNR in fixed background, using
    #: particle information to avoid averaging over particles.
    smart_averaging: bool = False
    constant_refresh: bool = False

    def __post_init__(self) -> None:
        _validate_probability(
            self.minimum_matching_probability,
            "minimum_matching_probability",
        )
        _validate_probability(self.ghost_signal_check, "ghost_signal_check")
        _validate_positive(self.psf_radius, "psf_radius")
        _validate_positive(self.max_ghost_matches, "max_ghost_matches")


# ---------------------------------------------------------------------------
# Top-level parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FullAnalysisParams:
    """Complete parameter set for a TandemMP analysis run."""

    #: Drift correction config, or ``False`` to disable.
    drift_correction: DriftCorrectionParams | bool = dataclasses.field(
        default_factory=DriftCorrectionParams,
    )
    #: Arbitrary photon-count threshold that indicates the lid opening.
    lid_open_threshold: float = 1.0
    #: Number of frames before and after lid opening to subsample the
    #: drift vector.
    lid_open_margin: int = 100
    #: Apply median filter to every frame first.
    illumination_correction: bool = False
    #: Minimum expected mass to be tracked; helps define a noise floor to
    #: decide whether a detection is viable.
    minimum_mass: float = 40.0
    snr_limit: float = 0.5
    #: Skip marking overlapping particles.
    skip_overlaps: bool = False
    first_analysis: FirstAnalysisParams = dataclasses.field(
        default_factory=FirstAnalysisParams,
    )
    tandemmp_analysis: TandemMPAnalysisParams = dataclasses.field(
        default_factory=TandemMPAnalysisParams,
    )
    #: Use parallelisation where available.
    parallel: bool = False
    #: Special bypass for nanopore analysis.
    nanopore: bool = False
    #: Trim down memory usage everywhere; disables debugging, so only
    #: enable if the pipeline is trusted.
    save_memory: bool = True
    #: Display event report after every fit.
    supervise: bool = False
    use_rust: bool = False
    reflectivity_correction: bool = True
    #: Auto-bin to ~100 ms integration time.
    auto_bin: bool = True
    #: Save experimental PSF model to disk.
    save_exp_psf: bool = False

    def __post_init__(self) -> None:
        _validate_positive(self.minimum_mass, "minimum_mass")
        _validate_positive(self.lid_open_margin, "lid_open_margin")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _recursive_replace(obj: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Replace a field anywhere in the dataclass tree.

    Returns the updated root, or ``None`` if the key was not found.
    """
    key = next(iter(kwargs))

    # Direct field on this object?
    field_names = {f.name for f in fields(obj)}
    if key in field_names:
        return dataclasses.replace(obj, **kwargs)

    # Recurse into nested dataclass fields.
    for f in fields(obj):
        val = getattr(obj, f.name)
        if dataclasses.is_dataclass(val):
            result = _recursive_replace(val, **kwargs)
            if result is not None:
                return dataclasses.replace(obj, **{f.name: result})

    return None


def get_parameters(**overrides: Any) -> FullAnalysisParams:  # noqa: ANN401
    """Create a :class:`FullAnalysisParams` with keyword overrides.

    Overrides are applied recursively — a keyword matching a nested
    dataclass field (e.g. ``navg=10``) will update the correct sub-object.

    Raises :class:`ValueError` for unknown parameter names.
    """
    params: FullAnalysisParams = FullAnalysisParams()
    for key, value in overrides.items():
        result = _recursive_replace(params, **{key: value})
        if result is None:
            raise ValueError(
                f"Unknown parameter {key!r}; not found in "
                "FullAnalysisParams or its nested dataclasses"
            )
        params = result
    return params
