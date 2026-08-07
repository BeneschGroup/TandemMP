"""Movie loading and first-pass event detection for the TandemMP pipeline.

:class:`Loader` reads a raw movie (from a file or an injected array), applies
optional drift correction, runs an event-finding pipeline conforming to
:class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner` (defaulting to
:class:`~tandemmp.pipeline.open_iscat_pipeline.OpenIscatPipeline`), fits a
first-pass :class:`~tandemmp.events.RatiometricEvent` for every detected
peak, and then merges/filters/marks-overlaps on those events ready for
:class:`tandemmp.pipeline.investigator.Investigator`.

The ``open_iscat.utils.parallel`` dependency is replaced with
:func:`tandemmp.core.parallel.dispatch_kwargs`, the same shared local
:mod:`multiprocessing` dispatcher used by :mod:`tandemmp.drift.corrector` and
:mod:`tandemmp.pipeline.investigator`. ``determine_experimental_psf_model``
only adopts the spline-based experimental PSF model; porting the parametric
a12/w/s refit branch onto the
:class:`~tandemmp.fitting.psf_interface.PsfInterface` abstraction (layer 6m,
see CLAUDE.md) remains a separate, not-yet-done task.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import tqdm
from scipy.interpolate import RectBivariateSpline

from tandemmp.core.classifications import ProcessingMethod
from tandemmp.core.conversions import c_to_mass
from tandemmp.core.parallel import dispatch_kwargs
from tandemmp.core.parameters import DriftCorrectionParams, FullAnalysisParams
from tandemmp.core.types import FloatArray, UInt16Array
from tandemmp.drift.corrector import get_corrected_movie
from tandemmp.events import PeakImage, RatiometricEvent
from tandemmp.fitting.psf import PsfModel, get_psf_model
from tandemmp.fitting.psf_interface import PsfInterface
from tandemmp.io.movie import final_polish_fix
from tandemmp.pipeline.open_iscat_pipeline import Info, OpenIscatPipeline, load_mp_file
from tandemmp.pipeline.parallel import make_ratiometric_event_from_peak, median_frame
from tandemmp.pipeline.pipeline_interface import PipelineResults, PipelineRunner

logger = logging.getLogger(__name__)

Events = list[RatiometricEvent]
Corner = tuple[tuple[int, int, int], tuple[int, int, int]]


class OverlapArea(NamedTuple):
    """A spatio-temporal overlap between two ratiometric events."""

    event: RatiometricEvent
    area: list[slice]
    frames: list[int] | int
    temporal: bool


# ---------------------------------------------------------------------------
# Movie loading helpers
# ---------------------------------------------------------------------------


def bin_movie(
    arr: npt.NDArray[Any], binning: int, dtype: type = np.uint16
) -> npt.NDArray[Any]:
    """Temporally bin *arr* by averaging non-overlapping windows of *binning* frames.

    The final partial window (if ``len(arr)`` is not a multiple of *binning*)
    is dropped, matching the original ``utils.bin_array`` behaviour.
    """
    indices = np.arange(0, len(arr) - binning, binning)
    return np.array([arr[i : i + binning].mean(axis=0) for i in indices]).astype(dtype)


def compute_post_binning_factor(
    raw_fps: float, pre_binning: int, navg: int, target_integration_time: float = 0.1
) -> int:
    """Return the extra temporal binning factor to reach ``target_integration_time``.

    Used to auto-bin a movie to roughly a 100 ms integration window (the
    default) once *navg*-frame ratiometric averaging is taken into account.
    """
    return int(
        np.round(max(1, target_integration_time / (navg / (raw_fps / pre_binning))))
    )


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------


def get_peak(
    ratiometric: FloatArray,
    x: float,
    y: float,
    i: int,
    l_thum: int,
    *,
    nanopore: bool = False,
) -> PeakImage:
    """Cut a thumbnail out of *ratiometric* centred on ``(x, y)`` at frame *i*.

    The thumbnail is clamped to stay within the frame if it would otherwise
    run off the edge. ``nanopore`` inverts the sign of the cutout, since the
    destructive-interference convention is flipped for nanopore movies.
    """
    shape = ratiometric[0].shape
    width = l_thum // 2

    x, y = int(round(x)), int(round(y))
    x_min, x_max = x - width, x + width + 1
    y_min, y_max = y - width, y + width + 1

    if x_min < 0:
        x_min, x_max = 0, width * 2 + 1
    if y_min < 0:
        y_min, y_max = 0, width * 2 + 1
    if x_max > shape[1]:
        x_max, x_min = shape[1], shape[1] - (width * 2 + 1)
    if y_max > shape[0]:
        y_max, y_min = shape[0], shape[0] - (width * 2 + 1)

    crop = ratiometric[i, y_min:y_max, x_min:x_max]
    if nanopore:
        crop = -crop

    return PeakImage(
        image=crop,
        method=ProcessingMethod.ratiometric,
        midpoint=(x, y),
        cutout=[slice(y_min, y_max), slice(x_min, x_max)],
        i=i,
    )


def build_first_pass_events(
    ratiometric: FloatArray,
    xs: npt.NDArray[np.floating[Any]],
    ys: npt.NDArray[np.floating[Any]],
    indices: npt.NDArray[np.integer[Any]],
    *,
    navg: int,
    l_thum: int,
    peak_model: PsfInterface,
    nanopore: bool = False,
    parallel: bool = False,
    n_jobs: int | None = None,
) -> Events:
    """Fit a :class:`RatiometricEvent` for every detected peak past frame *navg*."""
    peak_images = [
        get_peak(ratiometric, x, y, int(i), l_thum, nanopore=nanopore)
        for x, y, i in zip(xs, ys, indices, strict=True)
        if int(i) > navg
    ]
    args = [{"peak": p, "navg": navg, "peak_model": peak_model} for p in peak_images]
    return dispatch_kwargs(
        make_ratiometric_event_from_peak, args, parallel=parallel, n_jobs=n_jobs
    )


# ---------------------------------------------------------------------------
# Lid-open exclusion
# ---------------------------------------------------------------------------


def find_lid_open_exclusion_range(
    raw_movie: FloatArray,
    navg: int,
    threshold: float,
    margin: int = 5,
) -> set[int]:
    """Return frame indices near a "lid open" region, in ``final_polish_fix``-space.

    *raw_movie* is the movie **before** :func:`~tandemmp.io.movie.final_polish_fix`
    prepends ``navg * 2`` synthetic frames. Those prepended frames never count
    as lid-open, so the returned indices are directly comparable against
    events found in the polished movie.
    """
    below_threshold = np.sum(raw_movie, axis=(1, 2)) < threshold
    below_threshold = np.concatenate([np.zeros(navg * 2, dtype=bool), below_threshold])
    if not below_threshold.any():
        return set()

    lid_open_frames = np.arange(len(raw_movie) + navg * 2)[below_threshold]
    return set(
        range(int(lid_open_frames.min()) - margin, int(lid_open_frames.max()) + margin)
    )


def filter_events_by_lid(
    events: Events, exclusion_range: set[int]
) -> tuple[Events, Events]:
    """Split *events* into ``(kept, excluded)`` based on ``exclusion_range``."""
    if not exclusion_range:
        return list(events), []
    excluded = [e for e in events if e.i in exclusion_range]
    kept = [e for e in events if e.i not in exclusion_range]
    return kept, excluded


# ---------------------------------------------------------------------------
# Movement correction (merges spatio-temporally coincident events)
# ---------------------------------------------------------------------------


def correct_movements(
    events: Events,
    ratiometric: FloatArray,
    *,
    l_thum: int,
    navg: int,
    nanopore: bool = False,
) -> Events:
    """Merge events that fire on the same particle across adjacent frames.

    Nearby events (within ``0.8 * l_thum`` pixels, at most one frame apart)
    are averaged into a single re-fitted event, weighted by whether they are
    signal (contrast < 0) or a likely background fluctuation.
    """
    if not events:
        return events
    xs = np.array([e.x for e in events])
    ys = np.array([e.y for e in events])
    cs = np.array([e.c for e in events])
    midx, midy = np.array([e.peak.midpoint for e in events]).T
    frames = np.array([e.i for e in events])
    indexes = np.arange(len(frames))

    new_events: Events = []
    merged_events: list[int] = []
    n_merged = 0
    for index, event in enumerate(tqdm.tqdm(events, desc="Correcting movements:")):
        if index in merged_events:
            continue

        not_self = indexes != index
        same_frame = (frames <= event.i + 1) & (frames >= event.i - 1)
        close = np.hypot(xs - event.x, ys - event.y) < int(l_thum * 0.8)
        sel = not_self & same_frame & close

        if sel.any():
            weights = np.array(cs[sel].tolist() + [event.c])
            weights[weights > 0] = 1
            weights[weights < 0] = 3
            group_x = np.average(
                midx[sel].tolist() + [event.peak.midpoint[0]], weights=weights
            )
            group_y = np.average(
                midy[sel].tolist() + [event.peak.midpoint[1]], weights=weights
            )
            group_i = int(np.average(frames[sel].tolist() + [event.i]))

            image = get_peak(
                ratiometric, group_x, group_y, group_i, l_thum, nanopore=nanopore
            )
            new_event = RatiometricEvent(peak=image, fit=True, navg=navg)

            for other_index in sel.nonzero()[0]:
                merged_events.append(int(other_index))
            n_merged += 2
            new_events.append(new_event)
        else:
            new_events.append(event)

    logger.info("Merged %d events into %d", n_merged, n_merged // 2)
    return new_events


# ---------------------------------------------------------------------------
# Overlap marking
# ---------------------------------------------------------------------------


def mark_overlaps(events: Events) -> None:
    """Populate ``event.overlaps`` for every spatio-temporally overlapping pair.

    Mutates *events* in place, appending :class:`OverlapArea` entries.
    """
    n_spatial, n_spatiotemporal = 0, 0

    corners: list[Corner] = [
        (
            (e.peak.cutout[1].start, e.peak.cutout[0].start, e.i - e.navg),
            (e.peak.cutout[1].stop - 1, e.peak.cutout[0].stop - 1, e.i + e.navg),
        )
        for e in events
    ]

    def x_inside(c1: Corner, c2: Corner) -> bool:
        return (c2[0][0] <= c1[0][0] <= c2[1][0]) or (c2[0][0] <= c1[1][0] <= c2[1][0])

    def y_inside(c1: Corner, c2: Corner) -> bool:
        return (c2[0][1] <= c1[0][1] <= c2[1][1]) or (c2[0][1] <= c1[1][1] <= c2[1][1])

    def frames_inside(c1: Corner, c2: Corner) -> bool:
        return (c2[0][2] <= c1[0][2] <= c2[1][2]) or (c2[0][2] <= c1[1][2] <= c2[1][2])

    def spatial_check(c1: Corner, c2: Corner) -> bool:
        return x_inside(c1, c2) and y_inside(c1, c2)

    for index, event in enumerate(tqdm.tqdm(events, desc="Marking overlaps...")):
        for other_index, other_event in enumerate(events[index + 1 :], start=index + 1):
            if not spatial_check(corners[index], corners[other_index]):
                continue

            # Other event is entirely before this one (captured the other way around).
            if corners[other_index][1][2] <= corners[index][0][2]:
                continue

            reference = corners[index][0][:-1]
            relative = np.subtract(
                [corners[other_index][0][:-1], corners[other_index][1][:-1]],
                reference,
                dtype=object,
            )

            event_width = corners[index][1][0] - corners[index][0][0]
            relative[(relative >= event_width) | (relative <= 0)] = None

            if relative[1][0]:
                relative[1][0] += 1
            if relative[1][1]:
                relative[1][1] += 1

            area = [
                slice(relative[0][1], relative[1][1]),
                slice(relative[0][0], relative[1][0]),
            ]

            if frames_inside(corners[index], corners[other_index]):
                n_spatiotemporal += 1
                overlapping_frames = set(
                    range(corners[index][0][-1], corners[index][1][-1] + 1)
                ) & set(
                    range(corners[other_index][0][-1], corners[other_index][1][-1] + 1)
                )
                event.overlaps.append(
                    OverlapArea(
                        event=other_event,
                        area=area,
                        frames=sorted(overlapping_frames),
                        temporal=True,
                    )
                )
            elif corners[other_index][0][-1] >= corners[index][1][-1]:
                n_spatial += 1
                next_frame = corners[other_index][0][-1]
                event.overlaps.append(
                    OverlapArea(
                        event=other_event, area=area, frames=next_frame, temporal=False
                    )
                )
            else:
                msg = (
                    f"Unreachable overlap geometry between events "
                    f"{event._ID!r} and {other_event._ID!r}"
                )
                raise RuntimeError(msg)

    logger.info(
        "%d spatial overlaps marked, %d direct intersections marked",
        n_spatial,
        n_spatiotemporal,
    )


# ---------------------------------------------------------------------------
# Mass / SNR filtering
# ---------------------------------------------------------------------------


def filter_events_by_mass(
    events: Events, minimum_mass: float, snr_limit: float
) -> tuple[Events, Events]:
    """Split *events* into ``(kept, discarded)`` by minimum mass and SNR.

    An event with a suspiciously high signal in its raw image is kept
    regardless of SNR/mass — it likely represents a genuine, if oddly-fit,
    particle rather than noise.
    """
    high_signal = np.array(
        [np.max(np.abs(c_to_mass(e.peak.image))) > minimum_mass * 3 for e in events],
        dtype=bool,
    )
    large_enough = (np.array([e.snr for e in events]) >= snr_limit) & (
        np.abs([e.m for e in events]) >= minimum_mass
    )
    selection = large_enough | high_signal
    kept = [e for e, sel in zip(events, selection, strict=True) if sel]
    discarded = [e for e, sel in zip(events, selection, strict=True) if not sel]
    return kept, discarded


# ---------------------------------------------------------------------------
# Experimental PSF model
# ---------------------------------------------------------------------------


def determine_experimental_psf_model(
    movie: FloatArray,
    ratiometric: FloatArray,
    events: Events,
    parameters: FullAnalysisParams,
    peak_model: PsfModel,
    *,
    nanopore: bool = False,
) -> tuple[PsfModel, PsfModel | None]:
    """Derive an experimental PSF spline from well-fit events and adopt it if it helps.

    Returns ``(new_peak_model, exp_model)``. ``exp_model`` is the raw spline
    :class:`PsfModel` built from the data (or ``None`` if there are no events
    to build one from); ``new_peak_model`` is *peak_model* unchanged unless a
    switch to the spline model measurably reduces total fit error.

    Only the spline-based model is adopted here — see the module docstring
    for why the parametric a12/w/s refit branch (the default when
    ``parameters.use_rust`` is ``True``) is not yet ported.
    """
    if not events:
        return peak_model, None

    model_params = parameters.first_analysis.experimental_psf_model
    if not model_params:
        return peak_model, None

    hw = parameters.first_analysis.l_thum // 2
    n = model_params.spatial_expansion  # type: ignore[union-attr]

    def expand_by_n(s: list[slice]) -> tuple[slice, slice]:
        return (
            slice(s[0].start - n, s[0].stop + n),
            slice(s[1].start - n, s[1].stop + n),
        )

    dxs, dys, cs, errs = np.transpose(
        [[e.dx, e.dy, e.c, e.basic_fit_error] for e in events]
    )
    xs, ys = np.transpose([e.peak.midpoint for e in events])

    x_in = ((movie.shape[2] - hw - n - 1) > xs) & (xs > (hw + n + 1))
    y_in = ((movie.shape[1] - hw - n - 1) > ys) & (ys > (hw + n + 1))
    within_margin = x_in & y_in

    snr = np.array([e.snr for e in events])
    limit = np.percentile(snr, model_params.selection_percentile)  # type: ignore[union-attr]
    good = snr >= limit

    sign = -1 if nanopore else 1
    _peaks = np.array(
        [
            (
                sign * ratiometric[e.i][expand_by_n(e.peak.cutout)]
                if inside
                else np.pad(e.peak.image, n)
            )
            for e, inside in zip(events, within_margin, strict=True)
        ]
    )
    _peaks[np.sign(cs) < 0] *= -1
    peaks = _peaks[good]

    old_dim = np.arange(-hw - n, hw + 1 + n)
    sampling_points = model_params.sampling_points  # type: ignore[union-attr]
    new_dim = np.arange(-hw - n, hw + n + sampling_points, sampling_points)
    scaling_factor = len(new_dim) / len(old_dim)
    x_shifts, y_shifts = dxs * scaling_factor, dys * scaling_factor

    splines = [RectBivariateSpline(old_dim, old_dim, p) for p in peaks]
    quality = np.array(
        [
            s(new_dim + dy, new_dim + dx)
            for s, dx, dy in zip(splines, dxs[good], dys[good], strict=True)
        ]
    )
    quality = np.array([-(q / q.max()) for q in quality])

    def float_to_slice(value: float) -> slice:
        if value > 0:
            return slice(int(value), None)
        if int(value) == 0:
            return slice(None, None)
        return slice(None, int(value))

    not_visible = np.full_like(quality, fill_value=True, dtype=bool)
    for p, y, x in zip(not_visible, y_shifts[good], x_shifts[good], strict=True):
        p[float_to_slice(y), float_to_slice(x)] = False

    quality[not_visible] = np.nan
    model = np.nanmedian(quality, axis=0)
    model[np.isnan(model)] = 0

    spline_model = RectBivariateSpline(new_dim, new_dim, model)
    exp_model = PsfModel(use_rust=parameters.use_rust)
    exp_model.set_spline_model(spline_model)

    if parameters.use_rust:
        logger.info(
            "use_rust is set; skipping spline-based PSF model refit "
            "(parametric a12/w/s refit onto PsfInterface not yet ported)"
        )
        return peak_model, exp_model

    new_error = float(
        np.sum(
            [
                exp_model.fit_peak(ev.peak.image, method="BFGS").error
                for ev in np.array(events)[good]
            ]
        )
    )
    old_error = float(np.sum(errs[good]))
    improvement = old_error - new_error

    logger.info(
        "Experimental PSF model: combined error before=%.6g after=%.6g "
        "improvement=%.2f%%",
        old_error,
        new_error,
        (improvement / old_error) * 100 if old_error else float("nan"),
    )

    if improvement <= 0:
        logger.info("Experimental PSF model did not improve fit; keeping default")
        return peak_model, exp_model

    logger.info("Applying experimental PSF model")
    new_model = PsfModel(use_rust=parameters.use_rust)
    new_model.set_spline_model(spline_model)
    for ev in tqdm.tqdm(events, desc="Refitting events..."):
        ev.peak_model = new_model
        ev.basic_fit()

    return new_model, exp_model


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class Loader:
    """Load a movie, optionally drift-correct it, and find/fit first-pass events.

    Parameters
    ----------
    mp_filepath:
        Path to the ``.mp``/``.tdms`` movie file. Still required (and used
        for naming outputs) even when *special_bypass* supplies the frames
        directly.
    special_bypass:
        Use these frames directly instead of reading *mp_filepath* — for
        injecting simulated movies. Must be a ``uint16`` array, matching the
        dtype of real camera-captured movies. Event finding is still driven
        through ``mp_filepath``.
    parameters:
        Analysis configuration. Defaults to
        :func:`~tandemmp.core.parameters.get_parameters`.
    just_movie:
        If ``True``, load (and optionally drift-correct) the movie and
        return without running event detection.
    skip_exp_model:
        If ``True``, skip fitting/adopting an experimental PSF model even
        when *special_bypass* is not set.
    frame_start, frame_end:
        Slice applied to the raw movie before any processing.
    seed:
        Forwarded to the drift-correction search for reproducibility (see
        :func:`tandemmp.drift.corrector.get_vector`). Left ``None`` for
        real usage.
    pipeline_runner:
        The event-finding backend, conforming to
        :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`.
        Defaults to
        :class:`~tandemmp.pipeline.open_iscat_pipeline.OpenIscatPipeline`.
        A different backend can be dropped in here without any other
        changes to :class:`Loader`.
    """

    def __init__(  # noqa: PLR0913
        self,
        mp_filepath: str | Path,
        *,
        special_bypass: UInt16Array | None = None,
        parameters: FullAnalysisParams | None = None,
        just_movie: bool = False,
        skip_exp_model: bool = False,
        frame_start: int | None = None,
        frame_end: int | None = None,
        seed: int | np.random.Generator | np.random.RandomState | None = None,
        pipeline_runner: PipelineRunner | None = None,
    ) -> None:
        self.parameters = parameters if parameters is not None else FullAnalysisParams()
        self.peak_model = get_psf_model(use_rust=self.parameters.use_rust)
        self.pipeline_runner: PipelineRunner = (
            pipeline_runner if pipeline_runner is not None else OpenIscatPipeline()
        )
        self.mp_filepath = Path(mp_filepath)
        self.folder = self.mp_filepath.parent
        self.vector = None
        self.exp_model: PsfModel | None = None
        self.movie: npt.NDArray[Any]

        if special_bypass is not None:
            self.movie = special_bypass[frame_start:frame_end]
            self.params: Info | None = None
            self._fps: float | None = None
        else:
            movie = load_mp_file(self.mp_filepath)
            info = movie.info

            pre_binning = info.frame_binning
            navg = self.parameters.first_analysis.navg
            post_binning = compute_post_binning_factor(
                info.raw_camera_frame_rate, pre_binning, navg
            )

            if self.parameters.auto_bin and post_binning > 1:
                self.movie = bin_movie(
                    np.asarray(movie)[frame_start:frame_end], binning=post_binning
                )
                info = dataclasses.replace(
                    info, frame_binning=pre_binning * post_binning
                )
            else:
                self.movie = np.asarray(movie)[frame_start:frame_end]

            self.params = info
            self._fps = info.raw_camera_frame_rate / info.frame_binning

        if self.parameters.drift_correction:
            logger.info("Starting drift correction")
            self._correct_drift(seed=seed)

        if just_movie:
            return

        # special_bypass's uint16 branch narrows self.movie's flow-type; widen
        # it back so it type-checks against the floating-array-typed helpers
        # below (they accept any numeric dtype at runtime via promotion).
        self.movie = cast("npt.NDArray[Any]", self.movie)

        navg = self.parameters.first_analysis.navg
        raw_movie = self.movie
        self.movie = final_polish_fix(self.movie, navg)

        self.exclusion_range = find_lid_open_exclusion_range(
            raw_movie, navg, self.parameters.lid_open_threshold
        )

        self._run_pipeline()

        xs = self.eventsFound.get_X()
        ys = self.eventsFound.get_Y()
        indices = self.eventsFound.get_I()

        events = build_first_pass_events(
            self.ratiometric,
            xs,
            ys,
            indices,
            navg=navg,
            l_thum=self.parameters.first_analysis.l_thum,
            peak_model=self.peak_model,
            nanopore=self.parameters.nanopore,
            parallel=self.parameters.parallel,
        )

        events, self.lid_events = filter_events_by_lid(events, self.exclusion_range)
        if self.lid_events:
            logger.info("Removing %d events in lid open region", len(self.lid_events))

        events = correct_movements(
            events,
            self.ratiometric,
            l_thum=self.parameters.first_analysis.l_thum,
            navg=navg,
            nanopore=self.parameters.nanopore,
        )

        n_before = len(events)
        events, self.discarded_events = filter_events_by_mass(
            events, self.parameters.minimum_mass, self.parameters.snr_limit
        )
        logger.info(
            "Removing %d events with mass under %gkDa",
            n_before - len(events),
            self.parameters.minimum_mass,
        )

        if not self.parameters.skip_overlaps:
            mark_overlaps(events)
        else:
            logger.warning(
                "Event overlaps are unmarked, this will not perform properly!"
            )

        self.events = sorted(events, key=lambda e: e.i)
        for index, ev in enumerate(self.events):
            ev._ID = index

        if special_bypass is None and not skip_exp_model:
            logger.info("Optimising PSF model...")
            self.peak_model, self.exp_model = determine_experimental_psf_model(
                self.movie,
                self.ratiometric,
                self.events,
                self.parameters,
                self.peak_model,
                nanopore=self.parameters.nanopore,
            )

        if self.parameters.save_exp_psf:
            if self.exp_model is None or not self.exp_model.has_spline_model:
                msg = "No experimental PSF model found to save"
                raise ValueError(msg)
            name = (
                f"{self.mp_filepath.stem}"
                f"_exp_psf_l_thum{self.parameters.first_analysis.l_thum}.pickle"
            )
            with (self.folder / name).open("wb") as f:
                pickle.dump(self.exp_model.spline_model, f)

        if self.parameters.illumination_correction:
            args = [{"frame": f, "size": 23} for f in self.movie]
            median = dispatch_kwargs(
                median_frame, args, parallel=self.parameters.parallel
            )
            self.movie = self.movie / np.asarray(median)

        self.old_mass = [e.m for e in events]

    def _correct_drift(
        self,
        *,
        seed: int | np.random.Generator | np.random.RandomState | None = None,
    ) -> bool:
        dc = self.parameters.drift_correction
        if not isinstance(dc, DriftCorrectionParams):
            msg = "drift_correction must be a DriftCorrectionParams instance"
            raise TypeError(msg)

        corrected_movie, vector, sel3d = get_corrected_movie(  # type: ignore[misc]
            movie=self.movie,
            frame_gap=dc.frame_gap,
            rolling_baseline=dc.z_drift,
            threshold=self.parameters.lid_open_threshold,
            threshold_margin=self.parameters.lid_open_margin,
            parallel=self.parameters.parallel,
            full_output=True,
            seed=seed,
        )
        self.vector = vector

        old_error = float(np.std(self.movie[sel3d][-1] / self.movie[sel3d][0]))
        new_error = float(
            np.std(
                corrected_movie[-1]
                / corrected_movie[self.parameters.first_analysis.navg]
            )
        )
        improvement = old_error - new_error

        logger.info(
            "St.dev = %.6g\nDrift corrected st.dev = %.6g\nImprovement = %.2f%%",
            old_error,
            new_error,
            (improvement / old_error) * 100 if old_error else float("nan"),
        )

        if improvement > 0:
            logger.info("Accepting drift correction")
            self.movie = corrected_movie
            return True

        logger.info("Refusing drift correction")
        return False

    def _run_pipeline(self) -> None:
        pipeline_results = self.pipeline_runner.run(
            self.mp_filepath,
            frame_bypass=self.movie,
            navg=self.parameters.first_analysis.navg,
            l_thum=self.parameters.first_analysis.l_thum,
            parallel=self.parameters.parallel,
        )
        if not isinstance(pipeline_results, PipelineResults):
            msg = (
                f"{type(self.pipeline_runner).__name__}.run() returned "
                f"{type(pipeline_results).__name__}; expected PipelineResults"
            )
            raise TypeError(msg)
        self.movie = pipeline_results.frames
        self.ratiometric = pipeline_results.rframes
        self.eventsFound = pipeline_results.events  # noqa: N815

    def __repr__(self) -> str:
        n_events = len(self.events) if hasattr(self, "events") else "?"
        return f"Loaded movie {self.mp_filepath} - {n_events} events"
