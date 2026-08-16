"""Picklable helpers for parallel event processing.

All public functions are module-level (not closures or methods) so they can
be serialised by :mod:`multiprocessing`.  They wrap event construction and
fitting operations that would otherwise be defined inline in the pipeline.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import median_filter

from tandemmp.core.classifications import (
    EvClassification,
    ProcessingMethod,
    classify_mass_change,
)
from tandemmp.core.conversions import c_to_mass, mass_to_c
from tandemmp.events._base import PeakImage
from tandemmp.events.ratiometric import RatiometricEvent
from tandemmp.events.subtraction import SubtractionEvent
from tandemmp.fitting.psf import fit_3d_ratiometric
from tandemmp.fitting.psf_interface import PsfInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple wrappers (picklable for multiprocessing)
# ---------------------------------------------------------------------------


def fit_peak(
    peak: npt.NDArray[np.floating],
    peak_model: PsfInterface,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Call ``peak_model.fit_peak(peak, **kwargs)`` — picklable wrapper."""
    return peak_model.fit_peak(peak, **kwargs)


def make_subtraction_event_from_peak(
    peak: PeakImage,
    subtracted_i: int,
    fit: bool = True,
    exclusion: npt.NDArray[np.bool_] | None = None,
    peak_model: PsfInterface | None = None,
) -> SubtractionEvent:
    """Create a :class:`SubtractionEvent` from a :class:`PeakImage`."""
    return SubtractionEvent(
        peak, subtracted_i, fit=fit, exclusion=exclusion, peak_model=peak_model
    )


def median_frame(
    frame: npt.NDArray[np.floating],
    size: int,
) -> npt.NDArray[np.floating]:
    """Apply a median filter to *frame* — picklable wrapper."""
    return median_filter(frame, size)  # type: ignore[no-any-return]


def make_ratiometric_event_from_peak(
    peak: PeakImage,
    navg: int,
    peak_model: PsfInterface | None = None,
) -> RatiometricEvent:
    """Create a :class:`RatiometricEvent` from a :class:`PeakImage`."""
    return RatiometricEvent(peak, fit=True, navg=navg, peak_model=peak_model)


def make_subtraction_event(
    image: npt.NDArray[np.floating],
    index: int,
    midpoint: tuple[int, int],
    cutout: list[slice],
    subtracted_i: int,
    fit: bool = True,
    exclusion: npt.NDArray[np.bool_] | None = None,
    peak_model: PsfInterface | None = None,
) -> SubtractionEvent:
    """Build a :class:`SubtractionEvent` from raw image data.

    Constructs a :class:`PeakImage` internally and passes it to
    :class:`SubtractionEvent`.  Best used to create a
    :class:`SubtractionSeries`.
    """
    peak = PeakImage(image, ProcessingMethod.subtraction, midpoint, cutout, index)
    return SubtractionEvent(
        peak=peak,
        subtracted_i=subtracted_i,
        fit=fit,
        exclusion=exclusion,
        peak_model=peak_model,
    )


# ---------------------------------------------------------------------------
# 3-D ratiometric classification
# ---------------------------------------------------------------------------


def parallel_find_classification(  # noqa: PLR0912, PLR0913
    peak: npt.NDArray[np.floating],
    series: Any,  # noqa: ANN401
    minimum_mass: float,
    peak_model: PsfInterface,
    *,
    ghost_unbind_series: bool = False,
    particle_c: float | None = None,
    lock: bool = False,
    method: str = "Powell",
    allow_native: bool = True,
    partial_fit: bool = False,
    particle_desc: Any = None,  # noqa: ANN401
) -> dict[str, Any]:
    """Find the best classification for a ratiometric event.

    Applies 3-D fitting across candidate classifications (binding, movement,
    unbinding) and selects the one with the lowest combined error and penalty.

    Args:
        peak: Ratiometric peak image.
        series: A :class:`SubtractionSeries` providing positional information.
        minimum_mass: Mass threshold below which a signal is treated as zero.
        peak_model: PSF model for generating synthetic impressions.
        ghost_unbind_series: If *True*, only consider unbinding classification.
        particle_c: Known contrast of the tracked particle. ``None`` for new
            particles.
        lock: Lock the before-contrast to ``particle_c`` during fitting.
        method: Optimisation method for :func:`scipy.optimize.minimize`.
        allow_native: Also evaluate native (non-minimised) fits.
        partial_fit: Use bad-frame mask from the series.
        particle_desc: Identifier for logging/output.

    Returns:
        Dict with keys ``series``, ``classification``, ``fit``, ``fit_type``,
        ``c``, ``particle``, ``penalty``, and ``fit_image``.
    """
    # Determine candidate classifications
    if particle_c is None:
        classifications = [EvClassification.Binding]
    elif np.isnan(particle_c):
        classifications = [
            EvClassification.Binding,
            EvClassification.Movement,
            EvClassification.Unbinding,
        ]
    else:
        classifications = [EvClassification.Movement, EvClassification.Unbinding]

    if ghost_unbind_series:
        classifications = [EvClassification.Unbinding]

    is_bad = series.is_bad if partial_fit else None
    navg = len(series.dx) // 2

    specific_3d_fit = functools.partial(
        fit_3d_ratiometric,
        peak=peak,
        dx=series.smooth_dx,
        dy=series.smooth_dy,
        lock=lock,
        particle_c=particle_c,
        c_guess=-abs(series[0].c),
        is_bad=is_bad,
        psf_model=peak_model,
        method=method,
    )

    _results = [specific_3d_fit(classification=c) for c in classifications]
    images = [r[1] for r in _results]
    results = [r[0] for r in _results]

    # Expand with native fits (no minimisation)
    if allow_native:
        results.extend([series.fit_to(peak, c, peak_model) for c in classifications])
        images.extend([series.impression(c, peak_model) for c in classifications])
        classifications = classifications + classifications

    # Expected signals
    signal_before = float(series.m[:navg].mean())
    signal_after = float(series.m[-navg:].mean())

    if abs(signal_before) < minimum_mass:
        signal_before = 0.0
    if abs(signal_after) < minimum_mass:
        signal_after = 0.0

    signal_before = abs(float(mass_to_c(signal_before)))
    signal_after = abs(float(mass_to_c(signal_after)))

    penalties: list[float] = []

    for res, classification in zip(results, classifications, strict=True):
        penalty = series.avg_residual_noise

        if len(res.x) > 2:
            fit_before = fit_after = None
        elif len(res.x) == 2:
            fit_before, fit_after = res.x
        elif classification.presence_before and classification.presence_after:
            fit_before = fit_after = res.x[0]
        elif not classification.presence_before:
            fit_before = None
            penalty += signal_before**2
            fit_after = res.x[0]
        elif not classification.presence_after:
            fit_before = res.x[0]
            penalty += signal_after**2
            fit_after = None
        else:
            msg = (
                f"Unreachable branch: single-parameter fit with classification "
                f"{classification!r} where both presence_before and "
                f"presence_after are True should have matched the earlier "
                f"branch."
            )
            raise AssertionError(msg)

        if fit_after is not None:
            penalty += (signal_after - abs(fit_after)) ** 2

        if fit_before is not None:
            if particle_c is not None and np.isnan(particle_c):
                penalty += (signal_before - abs(fit_before)) ** 2
            elif particle_c is not None:
                penalty += (signal_before - abs(particle_c)) ** 2

        penalties.append(penalty)

    # TODO: verify intended selection behaviour with upstream TandemMP.
    # `lowest_error_index` is computed from raw `r.fun` BEFORE the penalty
    # loop below folds penalties into `r.fun`, so the best classification is
    # chosen by raw fit error only — the computed penalties never influence
    # the selection.  This mirrors the original tandemmp/utils/parallel.py
    # behaviour but likely contradicts the docstring's "lowest combined error
    # and penalty" phrasing.  Issue to be filed against upstream
    # joshualiambishop/TandemMP; resolve once intent is confirmed (either
    # swap the two blocks or drop the unused penalty computation).
    lowest_error_index = int(np.argmin([r.fun for r in results]))

    for r, p in zip(results, penalties, strict=True):
        r.fun += p

    best_classification = classifications[lowest_error_index]
    best_result = results[lowest_error_index]
    penalty = penalties[lowest_error_index]
    best_image = images[lowest_error_index]

    if particle_c is None or np.isnan(particle_c):
        fit_type = "untracked"
    else:
        fit_type = "tracked"

    if len(best_result.x) > 2:
        fit_type += " native"
    else:
        fit_type += " simple"

    if lock:
        c: Any = [particle_c, best_result.x[0]]
    else:
        c = best_result.x

    if best_classification == EvClassification.Movement:
        if len(c) > 2:
            mass_before = float(c_to_mass(np.mean(c[:navg])))
            mass_after = float(c_to_mass(np.mean(c[-navg:])))
            best_classification = classify_mass_change(mass_before, mass_after)
        elif len(c) == 2:
            masses: npt.NDArray[np.floating] = c_to_mass(np.asarray(c))  # type: ignore[assignment]
            mass_before, mass_after = float(masses[0]), float(masses[1])
            best_classification = classify_mass_change(
                float(mass_before), float(mass_after)
            )

    if partial_fit:
        fit_type += "\npartial"

    return {
        "series": series,
        "classification": best_classification,
        "fit": best_result,
        "fit_type": fit_type,
        "c": c,
        "particle": particle_desc,
        "penalty": penalty,
        "fit_image": best_image,
    }
