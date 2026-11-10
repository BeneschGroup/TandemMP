"""Image correction utilities for iSCAT peak analysis.

Provides ghost-prediction transforms, cutout overlap calculations, and
PSF-based image corrections for subtraction events.

Functions that accept ``particles`` require objects with these attributes:

- ``p.ID: int``
- ``p[frame] -> event_data | None``  (with ``.x``, ``.y``, ``.c``, ``.data_from``)
- ``p.get_best_event(frame) -> event | None``  (with ``.peak.image``, ``.i``)
- ``p.is_possible_split: bool``
- ``p.binding_frame: int``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from tandemmp.core.classifications import ProcessingMethod
from tandemmp.core.functions import nan_normalise
from tandemmp.events._base import Correction, PeakImage
from tandemmp.fitting.psf import psf_2d_cached
from tandemmp.fitting.psf_interface import PsfInterface

logger = logging.getLogger(__name__)


# Ghost-prediction transform


def ghost_prediction_transform(
    peak: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Apply a distance-based transform to a peak image for ghost prediction.

    For each pixel, compute the minimum distance to the negated image and
    scale the original peak by a factor derived from those distances.
    """
    flat_neg = np.ravel(-peak)
    distances = np.min(np.abs(flat_neg - peak[:, :, np.newaxis]), axis=-1)

    norm = nan_normalise(distances)
    transform: npt.NDArray[np.floating] = 1 - norm**0.1
    return peak * transform


# Cutout overlap


def calculate_overlapping_slice(
    cutout1: tuple[slice, slice],
    cutout2: tuple[slice, slice],
) -> list[slice] | None:
    """Compute the relative overlap of *cutout2* within the frame of *cutout1*.

    Returns a ``[y_slice, x_slice]`` pair in *cutout1*'s local coordinates,
    or ``None`` if there is no spatial intersection.
    """
    ys1, xs1 = cutout1
    ys2, xs2 = cutout2

    x11, x12 = xs1.start, xs1.stop - 1
    y11, y12 = ys1.start, ys1.stop - 1
    x21, x22 = xs2.start, xs2.stop - 1
    y21, y22 = ys2.start, ys2.stop - 1

    l_thum = x12 - x11

    x_check = (x11 <= x21 <= x12) or (x11 <= x22 <= x12)
    y_check = (y11 <= y21 <= y12) or (y11 <= y22 <= y12)

    if not (x_check and y_check):
        return None

    coords = np.array([[x21, y21], [x22, y22]], dtype=object)
    rel_coords = coords - np.array([x11, y11])
    rel_coords[(rel_coords >= l_thum) | (rel_coords < 0)] = None

    x2 = rel_coords[1][0] + 1 if rel_coords[1][0] is not None else None
    y2 = rel_coords[1][1] + 1 if rel_coords[1][1] is not None else None

    return [slice(rel_coords[0][1], y2), slice(rel_coords[0][0], x2)]


# Artefact removal


def remove_artefacts(peak: PeakImage) -> npt.NDArray[np.floating]:
    """Subtract corrections from the original image, skipping paired entries.

    When a particle has *two* corrections in the peak (foreground + background),
    both are excluded from the subtraction.
    """
    pids = [c.particle for c in peak.corrections]  # type: ignore[union-attr]
    unique = np.array(list(set(pids)))
    p_with_both = unique[[pids.count(i) == 2 for i in set(pids)]]
    corrections = [
        p.sliced_image
        for p in peak.corrections  # type: ignore[union-attr]
        if p.particle not in p_with_both
    ]
    return peak.original - np.sum(corrections, axis=0)  # type: ignore[no-any-return, arg-type]


# Ghost correction


def correct_ghost(
    series: list[PeakImage] | list[Any],
    dx: float,
    dy: float,
    c: float,
    peak_model: PsfInterface | None = None,
    *,
    save: bool = False,
    desc: str = "Ghost",
) -> list[PeakImage]:
    """Subtract a ghost PSF impression from each peak in *series*.

    *series* can hold :class:`PeakImage` objects, or objects with a ``.peak``
    attribute (e.g. events, or a ``SubtractionSeries``).
    """
    peaks: list[PeakImage] = [
        p if isinstance(p, PeakImage) else p.peak  # type: ignore[union-attr]
        for p in series
    ]

    images = [p.image for p in peaks]
    l_thum = images[0].shape[0]
    psf = psf_2d_cached(dx, dy, -c, l_thum, peak_model)
    corrected = np.array(images) - psf

    new_peaks: list[PeakImage] = []
    for p, new_im in zip(peaks, corrected, strict=True):
        extra_kwargs: dict[str, Any] = {}
        if save:
            extra_kwargs = {
                "original": p.original if p.original is not None else p.image,
                "background": (
                    p.background if p.background is not None else np.zeros_like(p.image)
                )
                - psf,
                "corrections": list(p.corrections or [])
                + [
                    Correction(
                        sliced_image=psf,
                        orig_image=p.image,
                        particle=desc,  # type: ignore[arg-type]
                        frame_requested=p.i,
                        background=True,
                        data_from=None,
                    )
                ],
            }

        new_peaks.append(
            PeakImage(
                image=new_im,
                method=p.method,
                midpoint=p.midpoint,
                cutout=p.cutout,
                i=p.i,
                **extra_kwargs,
            )
        )

    return new_peaks


# PSF-fit based subtraction correction


def correct_subtraction_fit(  # noqa: PLR0913
    event: Any,  # noqa: ANN401
    particles: list[Any] | None = None,
    focus: Any | None = None,  # noqa: ANN401
    *,
    gdb: bool = False,
    save: bool = False,
    psf_model: PsfInterface | None = None,
) -> PeakImage:
    """Correct a subtraction event by subtracting modelled PSF impressions.

    For each particle, a foreground and/or background PSF is generated from
    the particle's fitted position and subtracted from the event image.
    The ``focus`` particle's foreground contribution is skipped (it is the
    particle being measured).

    Args:
        event: Subtraction event (needs ``.peak``, ``.i``, ``.subtracted_i``, ``.navg``)
        particles: List of particle objects. If *None*, returns the original peak.
        focus: The particle being measured — its foreground is not subtracted.
        gdb: Ghost Double Bypass — if *True*, skip the focus particle's background.
        save: Store correction metadata on the returned :class:`PeakImage`.
        psf_model: PSF model for generating synthetic impressions.
    """
    if particles is None:
        return event.peak  # type: ignore[no-any-return]

    particles = list(set(particles))
    midpoint = event.peak.midpoint
    l_thum = event.peak.image.shape[0]
    corrections: list[Correction] = []

    if focus is not None and focus not in particles:
        logger.warning(
            "Focus particle %s not in particles %s",
            focus.ID,
            [p.ID for p in particles],
        )

    for p in particles:
        foreground = None if p is focus else p[event.i]

        if (p is focus) and gdb:
            background = None
        else:
            background = p[event.subtracted_i]

        no_bg_for_split = (
            p.is_possible_split and (event.subtracted_i - p.binding_frame) <= event.navg
        )

        if foreground is not None:
            dx = foreground.x - midpoint[0]
            dy = foreground.y - midpoint[1]
            c = foreground.c

            f_image = psf_2d_cached(dx, dy, c, l_thum, psf_model)

            corrections.append(
                Correction(
                    sliced_image=f_image,
                    orig_image=p.get_best_event(event.i).peak.image,
                    particle=p.ID,
                    frame_requested=event.i,
                    background=False,
                    data_from=foreground.data_from,
                )
            )

        if background is not None and not no_bg_for_split:
            b_image = -psf_2d_cached(
                background.x - midpoint[0],
                background.y - midpoint[1],
                background.c,
                l_thum,
                psf_model,
            )

            corrections.append(
                Correction(
                    sliced_image=b_image,
                    orig_image=p.get_best_event(event.subtracted_i).peak.image,
                    particle=p.ID,
                    frame_requested=event.subtracted_i,
                    background=True,
                    data_from=background.data_from,
                )
            )

    slices: list[npt.NDArray[np.floating]] = [
        c.sliced_image for c in corrections if c.sliced_image is not None
    ]
    working_background: npt.NDArray[np.floating] = np.sum(slices, axis=0)
    corrected_peak = event.peak.image - working_background

    extra_kwargs: dict[str, Any] = {}
    if save:
        extra_kwargs = {
            "original": event.peak.image,
            "background": working_background,
            "corrections": corrections,
        }

    return PeakImage(
        image=corrected_peak,
        method=ProcessingMethod.subtraction,
        midpoint=midpoint,
        cutout=event.peak.cutout,
        i=event.i,
        **extra_kwargs,
    )


# Image-based subtraction correction


def correct_subtraction_image(
    event: Any,  # noqa: ANN401
    particles: list[Any] | None = None,
    focus: Any | None = None,  # noqa: ANN401
    *,
    gdb: bool = False,
) -> PeakImage:
    """Correct a subtraction event using overlapping image data.

    Unlike :func:`correct_subtraction_fit`, this function uses actual peak
    images from neighbouring particles rather than PSF models, computing
    spatial overlaps between cutouts.  Correction metadata (``original``,
    ``background``, ``corrections``) is always attached to the returned
    :class:`PeakImage`.

    Args:
        event: Subtraction event.
        particles: List of particle objects.
        focus: The particle being measured.
        gdb: Ghost Double Bypass. Not yet implemented for the image-based
            correction path — raises :class:`NotImplementedError` if *True*.
    """
    if gdb:
        msg = (
            "Ghost Double Bypass (gdb) is not implemented for "
            "correct_subtraction_image; use correct_subtraction_fit for "
            "this code path."
        )
        raise NotImplementedError(msg)

    if particles is None:
        return event.peak  # type: ignore[no-any-return]

    particles = list(set(particles))
    midpoint = event.peak.midpoint
    corrections: list[Correction] = []

    for p in particles:
        # Background (inverse image)
        pev = p.get_best_event(event.subtracted_i)

        if pev:
            if pev.peak.corrections is None:
                pev_im = -pev.peak.image
            else:
                pev_im = -remove_artefacts(pev.peak)

            slice_main = calculate_overlapping_slice(event.peak.cutout, pev.peak.cutout)
            slice_other = calculate_overlapping_slice(
                pev.peak.cutout, event.peak.cutout
            )

            if slice_main and slice_other:
                bckg = np.zeros_like(event.peak.image)
                bckg[tuple(slice_main)] += pev_im[tuple(slice_other)]

                corrections.append(
                    Correction(
                        sliced_image=bckg,
                        orig_image=pev_im,
                        particle=p.ID,
                        frame_requested=event.subtracted_i,
                        background=True,
                        data_from=pev.i,
                    )
                )

        # Foreground
        if p is not focus:
            pev = p.get_best_event(event.i)
            if pev:
                if pev.peak.corrections is None:
                    pev_im = pev.peak.image
                else:
                    pev_im = remove_artefacts(pev.peak)

                slice_main = calculate_overlapping_slice(
                    event.peak.cutout, pev.peak.cutout
                )
                slice_other = calculate_overlapping_slice(
                    pev.peak.cutout, event.peak.cutout
                )
                if slice_main and slice_other:
                    bckg = np.zeros_like(event.peak.image)
                    bckg[tuple(slice_main)] += pev_im[tuple(slice_other)]

                    corrections.append(
                        Correction(
                            sliced_image=bckg,
                            orig_image=pev.peak.image,
                            particle=p.ID,
                            frame_requested=event.i,
                            background=False,
                            data_from=pev.i,
                        )
                    )

    slices_img: list[npt.NDArray[np.floating]] = [
        c.sliced_image for c in corrections if c.sliced_image is not None
    ]
    working_background: npt.NDArray[np.floating] = np.sum(slices_img, axis=0)
    corrected_peak = event.peak.image - working_background

    return PeakImage(
        image=corrected_peak,
        method=ProcessingMethod.subtraction,
        midpoint=midpoint,
        cutout=event.peak.cutout,
        i=event.i,
        original=event.peak.image,
        background=working_background,
        corrections=corrections,
    )
