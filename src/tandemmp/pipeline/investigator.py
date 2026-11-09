"""Per-event decision-making for the TandemMP pipeline.

Given a detected event, an :class:`Investigator` decides whether it represents
a new particle, an update to a tracked one, a ghost, or noise. The class
hierarchy is:

- :class:`BaseInvestigator` — series generation, classification, and correction
  primitives shared by all decision paths.
- :class:`GhostInvestigator` — ghost fitting, matching, and double-checking.
- :class:`Investigator` — top-level ``find_and_fit`` / ``multi_continuation``
  driver that combines the above.

Parallel work uses :func:`tandemmp.core.parallel.dispatch_kwargs`.
"""

from __future__ import annotations

import enum
import functools
import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import numpy.typing as npt
import tqdm
from scipy.optimize import OptimizeResult

from tandemmp.core.classifications import (
    EvClassification,
    classify_mass_change,
)
from tandemmp.core.conversions import c_to_mass, mass_to_c
from tandemmp.core.functions import get_overlap_probability, normalise
from tandemmp.core.parallel import dispatch_kwargs
from tandemmp.core.parameters import FullAnalysisParams
from tandemmp.events import (
    GhostEvent,
    PeakImage,
    RatiometricEvent,
    SubtractionEvent,
    SubtractionSeries,
)
from tandemmp.fitting.image import (
    correct_ghost,
    correct_subtraction_fit,
    correct_subtraction_image,
)
from tandemmp.fitting.minimize import minimize_model
from tandemmp.fitting.psf import (
    fit_3d_ratiometric,
    psf_2d_cached,
)
from tandemmp.fitting.psf_interface import PsfInterface
from tandemmp.pipeline.open_iscat_pipeline import Info
from tandemmp.pipeline.parallel import (
    make_subtraction_event,
    make_subtraction_event_from_peak,
)
from tandemmp.tracking.manager import ParticleManager
from tandemmp.tracking.particle import GhostParticle, HistoryEntry, Particle

if TYPE_CHECKING:
    from tandemmp.core.classifications import Classification

logger = logging.getLogger(__name__)


# Exceptions


class TrailingOffException(Exception):  # noqa: N818 — name preserved for API stability
    """Raised when a fit trails off to physically implausible mass."""


# MultiFitResult


@dataclass
class MultiFitResult:
    """Container for a candidate 3-D ratiometric fit of a single event.

    ``particle`` can be a real :class:`Particle`, a descriptive string
    (e.g. ``"No correction"``, ``"Initial ghost attempt"``), or ``None``
    when the fit has no associated particle context.
    """

    series: SubtractionSeries
    classification: Classification
    fit: OptimizeResult
    fit_type: str
    c: Any
    penalty: float
    image_error: float
    fit_image: npt.NDArray[np.floating] | None = None
    particle: Particle | str | None = None
    distance: float | None = None

    @property
    def error(self) -> float:
        """Optimisation error from the underlying fit."""
        return float(self.fit.fun)

    def set_particle(self, particle: Particle | str | None) -> None:
        """Replace the associated particle (or descriptive string)."""
        self.particle = particle

    def slim(self) -> MultiFitResult:
        """Return a shallow copy with nested particle replaced by its ID."""
        if isinstance(self.particle, Particle):
            slim_particle: Particle | str | int | None = self.particle.ID
        else:
            slim_particle = self.particle

        return MultiFitResult(
            series=self.series,
            classification=self.classification,
            fit=self.fit,
            fit_type=self.fit_type,
            c=self.c,
            penalty=self.penalty,
            image_error=self.image_error,
            fit_image=self.fit_image,
            particle=slim_particle,  # type: ignore[arg-type]
            distance=self.distance,
        )

    def __repr__(self) -> str:
        if isinstance(self.particle, Particle):
            particle: Any = self.particle.ID
        else:
            particle = self.particle
        return (
            f"Fit for {particle}, error = {self.error}, "
            f"mass penalty = {self.penalty}, avg noise = {self.image_error}, "
            f"distance from known = {self.distance} pixels"
        )


# Ghost event helper


def create_ghost_event(
    event_template: RatiometricEvent,
    fit_result: tuple[float, float, float] | npt.NDArray[np.floating],
    frame: int | None = None,
    peak_model: PsfInterface | None = None,
) -> GhostEvent:
    """Recreate a ghost impression as a :class:`GhostEvent`.

    ``fit_result`` is a ``(dx, dy, c)`` tuple; ``frame`` defaults to ``-1``.
    """
    if frame is None:
        frame = -1

    dx, dy, c = float(fit_result[0]), float(fit_result[1]), float(fit_result[2])

    fit_image = psf_2d_cached(
        dx, dy, c, event_template.peak.image.shape[0], psf_model=peak_model
    )

    from tandemmp.core.classifications import ProcessingMethod  # local to avoid cycles

    im = PeakImage(
        image=fit_image,
        method=ProcessingMethod.inferred,
        midpoint=event_template.peak.midpoint,
        cutout=event_template.peak.cutout,
        i=frame,
    )
    return GhostEvent(im, (dx, dy, c))


# Action / Decision


class Action(enum.IntFlag):
    """Action returned by an :class:`Investigator` for a single event."""

    create_new = 0
    updated = 1
    failed = 2
    ignore = 3


class Decision(NamedTuple):
    """Outcome of investigating an event."""

    action: Action
    particle: Particle | None
    interesting: bool = False


# Scoring helper


def _distance_array(
    results: list[MultiFitResult],
) -> npt.NDArray[np.floating]:
    """Extract distances from *results*, replacing ``None`` with a value
    strictly larger than the maximum known distance.

    :meth:`Particle.distance_to` returns ``None`` for unbound candidates.
    These receive ``max_known * 2`` (or ``1.0`` when all are unbound), so
    they rank behind candidates with a known positive distance.
    """
    raw = [r.distance for r in results]
    known = [d for d in raw if d is not None]

    if not known:
        fill = 1.0
    else:
        fill = max(known) * 2.0

    return np.asarray([fill if d is None else float(d) for d in raw], dtype=float)


# BaseInvestigator


class BaseInvestigator:
    """Movie/event manipulation primitives shared by all investigator paths."""

    def __init__(
        self,
        movie: npt.NDArray[np.floating],
        movie_params: Info,
        parameters: FullAnalysisParams,
        particle_manager: ParticleManager,
        peak_model: PsfInterface,
        open_lid_frames: list[int] | None = None,
    ) -> None:
        self.movie = movie
        self.movie_params = movie_params
        self.parameters = parameters

        if self.parameters.reflectivity_correction:
            logger.info("Applying reflectivity correction")
            self._r: npt.NDArray[np.floating] | None = np.sqrt(
                np.median(self.movie, axis=0) / np.median(self.movie)
            )
        else:
            self._r = None

        self.particle_manager = particle_manager
        self._mm: float = self.parameters.minimum_mass
        self.peak_model = peak_model
        self.open_lid_frames: list[int] = (
            [] if not open_lid_frames else list(open_lid_frames)
        )

        self.background_check: Callable[..., list[Particle]] = functools.partial(
            self.particle_manager.background_particle_check,
            l_thum=self.parameters.first_analysis.l_thum,
            psf_radius=self.parameters.tandemmp_analysis.psf_radius,
        )

        # Movie inversion — nanopore movies bypass the destructive-interference
        # marker, which is unreliable on that pipeline.
        if (
            not movie_params.scatter_interference_destructive
            or self.parameters.nanopore
        ):
            self._im_correct: Callable[
                [npt.NDArray[np.floating]], npt.NDArray[np.floating]
            ] = lambda im: -im
        else:
            self._im_correct = lambda im: im

    # Subtraction series generation

    def generate_subtraction_series(
        self,
        event: RatiometricEvent,
        start_frame: int,
        navg: int = 25,
        custom_frames: list[int] | None = None,
        fit: bool = True,
    ) -> SubtractionSeries:
        """Build a :class:`SubtractionSeries` of frames responsible for
        *event*, sharing *event*'s cutout.
        """
        if start_frame < 0:
            start_frame = 0
        if start_frame >= len(self.movie):
            start_frame = len(self.movie) - 1
        if start_frame < navg:
            navg = start_frame

        if custom_frames is None:
            start_frame = min(event.i - event.navg, start_frame)
            indexes = sorted(event.frame_range)
        else:
            indexes = list(custom_frames)

        if start_frame in self.open_lid_frames:
            start_frame = self.open_lid_frames[0] - 1

        midpoint = event.peak.midpoint
        cutout = event.peak.cutout

        cutout_frames: npt.NDArray[np.floating] = self.movie[
            indexes[0] : indexes[-1] + 1, cutout[0], cutout[1]
        ]
        if navg:
            start_frames = self.movie[
                start_frame - navg : start_frame, cutout[0], cutout[1]
            ].mean(axis=0)
        else:
            start_frames = self.movie[start_frame, cutout[0], cutout[1]]

        # This one line builds all fixed-background images.
        frames = self._im_correct((cutout_frames / start_frames) - 1).astype(np.float64)

        if self.parameters.reflectivity_correction and self._r is not None:
            frames *= self._r[cutout[0], cutout[1]]

        method = functools.partial(
            make_subtraction_event,
            midpoint=midpoint,
            cutout=cutout,
            subtracted_i=start_frame,
            fit=fit,
            peak_model=self.peak_model,
        )

        if event.exclusion_zone.any() and (custom_frames is None):
            args: list[dict[str, Any]] = [
                {"image": im, "index": i, "exclusion": e}
                for im, i, e in zip(frames, indexes, event.exclusion_zone, strict=True)
            ]
        else:
            args = [
                {"image": im, "index": i} for im, i in zip(frames, indexes, strict=True)
            ]

        # Parallelism only helps when fitting; otherwise the overhead dominates.
        events = dispatch_kwargs(
            method, args, parallel=self.parameters.parallel and fit
        )
        return SubtractionSeries(events)

    # Classification

    def find_classification(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        event: RatiometricEvent,
        series: SubtractionSeries,
        ghost_unbind_series: bool = False,
        manual_contrast: npt.NDArray[np.floating] | None = None,
        particle_c: float | None = None,
        lock: bool = False,
        method: str = "Powell",
        full_output: bool = False,
        allow_native: bool = True,
        partial_fit: bool = False,
        particle_desc: Particle | str | None = None,
        as_fit: bool = False,
    ) -> MultiFitResult | tuple[Any, list[Classification]]:
        """Run 3-D ratiometric fits for candidate classifications and return
        the best one.
        """
        if particle_c is None:
            classifications = [EvClassification.Binding]
        elif np.isnan(particle_c):
            classifications = [
                EvClassification.Binding,
                EvClassification.Movement,
                EvClassification.Unbinding,
            ]
        else:
            classifications = [
                EvClassification.Movement,
                EvClassification.Unbinding,
            ]

        if ghost_unbind_series:
            classifications = [EvClassification.Unbinding]

        is_bad = series.is_bad if partial_fit else None

        specific_3d_fit = functools.partial(
            fit_3d_ratiometric,
            peak=event.peak.image,
            dx=series.smooth_dx,  # type: ignore[arg-type]
            dy=series.smooth_dy,  # type: ignore[arg-type]
            lock=lock,
            particle_c=particle_c,
            c_guess=None,
            is_bad=is_bad,  # type: ignore[arg-type]
            psf_model=self.peak_model,
            method=method,
        )

        _results = [specific_3d_fit(classification=c) for c in classifications]
        images = [r[1] for r in _results]
        results = [r[0] for r in _results]

        if allow_native:
            results.extend(
                [
                    series.fit_to(event.peak.image, c, self.peak_model)
                    for c in classifications
                ]
            )
            images.extend(
                [series.impression(c, self.peak_model) for c in classifications]
            )
            classifications = classifications + classifications

        if full_output:
            if manual_contrast is None:
                return results, classifications
            return _results, classifications

        # Expected signals for penalty scoring
        signal_before = float(np.median(series.m[: event.navg]))
        signal_after = float(np.median(series.m[-event.navg :]))

        if abs(signal_before) < self._mm:
            signal_before = 0.0
        if abs(signal_after) < self._mm:
            signal_after = 0.0

        signal_before = abs(float(mass_to_c(signal_before)))
        signal_after = abs(float(mass_to_c(signal_after)))

        penalties: list[float] = []
        for res, classification in zip(results, classifications, strict=True):
            penalty = 0.0

            fit_before: float | None
            fit_after: float | None

            if len(res.x) > 2:
                fit_before = fit_after = None
                if particle_c and not (np.isnan(particle_c) or ghost_unbind_series):
                    penalty += (signal_before - abs(particle_c)) ** 2
            elif len(res.x) == 2:
                fit_before, fit_after = float(res.x[0]), float(res.x[1])
            elif classification.presence_before and classification.presence_after:
                fit_before = fit_after = float(res.x[0])
            elif not classification.presence_before:
                fit_before = None
                penalty += signal_before**2
                fit_after = float(res.x[0])
            elif not classification.presence_after:
                fit_before = float(res.x[0])
                penalty += signal_after**2
                fit_after = None
            else:
                fit_before = fit_after = None

            if fit_after:
                penalty += (signal_after - abs(fit_after)) ** 2

            if fit_before:
                if (
                    particle_c is not None and np.isnan(particle_c)
                ) or ghost_unbind_series:
                    penalty += (signal_before - abs(fit_before)) ** 2
                elif particle_c is not None:
                    penalty += (signal_before - abs(particle_c)) ** 2

            penalties.append(penalty)

        sorted_by_error = np.argsort([r.fun for r in results])

        for lowest_error_index in sorted_by_error:
            best_classification = classifications[lowest_error_index]
            best_result = results[lowest_error_index]
            penalty_v = penalties[lowest_error_index]
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
                c: Any = [particle_c, float(best_result.x[0])]
            else:
                c = best_result.x

            if best_classification == EvClassification.Movement:
                mass_before = mass_after = 0.0
                if len(c) > 2:
                    mass_before = float(c_to_mass(np.median(c[: event.navg])))
                    mass_after = float(c_to_mass(np.median(c[-event.navg :])))
                    best_classification = classify_mass_change(mass_before, mass_after)
                elif len(c) == 2:
                    masses = c_to_mass(np.asarray(c))
                    mass_before, mass_after = float(masses[0]), float(masses[1])  # type: ignore[index]
                    best_classification = classify_mass_change(mass_before, mass_after)
                if (
                    best_classification == EvClassification.Dissociation
                    and mass_after < mass_before * 0.1
                ):
                    if not as_fit:
                        logger.info("Huge dissociation noticed (>90%), moving to next")
                    continue

            distance: float | None
            if isinstance(particle_desc, Particle):
                distance = particle_desc.distance_to(
                    float(series.x[0]),
                    float(series.y[0]),
                    int(series[0].i),
                )
            else:
                distance = 0.5

            if partial_fit:
                fit_type += "\npartial"

            return MultiFitResult(
                series=series,
                classification=best_classification,
                fit=best_result,
                fit_type=fit_type,
                c=c,
                particle=particle_desc,
                penalty=penalty_v,
                image_error=series.avg_residual_noise,
                fit_image=best_image,
                distance=distance,
            )

        # All classifications were dissociation-vetoed. The old code fell off
        # the end of the loop here and returned None implicitly, which every
        # caller then dereferenced as a MultiFitResult anyway; raise
        # explicitly instead so the failure is self-describing.
        raise RuntimeError("find_classification produced no viable candidate")

    # Series correction

    def correct_series(  # noqa: PLR0913
        self,
        event: RatiometricEvent,
        series: SubtractionSeries,
        particles: list[Particle],
        focus: Particle | None = None,
        gdb: bool = False,
        return_peak_args: bool = False,
    ) -> SubtractionSeries | list[dict[str, Any]]:
        """Correct *series* by subtracting known-particle impressions."""
        if self.parameters.tandemmp_analysis.correct_with_images:
            correction: Callable[..., PeakImage] = correct_subtraction_image
        else:
            correction = functools.partial(
                correct_subtraction_fit, psf_model=self.peak_model
            )

        save: bool = not self.parameters.save_memory

        new_peaks: list[PeakImage] = [
            correction(sub_event, particles, focus, gdb=gdb, save=save)
            for sub_event in series
        ]
        subtracted_i: int = int(series.subtracted_i[0])

        exclusion_zone = event.exclusion_zone

        if focus:
            focus_event = focus._event_map.get_ratiometric(event.i + 1)
            affecting_ez = focus_event in [i.event for i in event.temporal_overlaps]
            if affecting_ez:
                exclusion_zone = event.recalculate_exclusion_zone(
                    overlaps=[
                        i for i in event.temporal_overlaps if i.event is not focus_event
                    ]
                )

        method = functools.partial(
            make_subtraction_event_from_peak, peak_model=self.peak_model
        )

        if exclusion_zone.any():
            args: list[dict[str, Any]] = [
                {"peak": peak, "subtracted_i": subtracted_i, "exclusion": e}
                for peak, e in zip(new_peaks, exclusion_zone, strict=True)
            ]
        else:
            args = [{"peak": peak, "subtracted_i": subtracted_i} for peak in new_peaks]

        if return_peak_args:
            return args

        new_events = dispatch_kwargs(method, args, parallel=self.parameters.parallel)
        return SubtractionSeries(new_events)


# GhostInvestigator


class GhostInvestigator(BaseInvestigator):
    """Ghost-particle fitting, matching, and double-checking."""

    def __init__(
        self,
        movie: npt.NDArray[np.floating],
        movie_params: Info,
        parameters: FullAnalysisParams,
        particle_manager: ParticleManager,
        peak_model: PsfInterface,
        open_lid_frames: list[int] | None = None,
    ) -> None:
        super().__init__(
            movie=movie,
            movie_params=movie_params,
            parameters=parameters,
            particle_manager=particle_manager,
            peak_model=peak_model,
            open_lid_frames=open_lid_frames,
        )
        self.doubles_checked: set[Particle] = set()

    # Top-level ghost processing

    def process_ghost(
        self,
        event: RatiometricEvent,
        close_particles: list[Particle],
        best_error: float | None = None,
        preferred_background: int | None = None,
    ) -> Decision:
        """Fit *event* as a ghost — either as new, or matched to existing."""
        initial_attempt: bool | GhostParticle = self.attempt_initial_ghost(
            event=event,
            close_particles=close_particles,
            preferred_background=preferred_background,
            best_error=best_error,
            method="Powell",
        )

        if initial_attempt is False:
            return Decision(action=Action.ignore, particle=None)

        if not self.parameters.tandemmp_analysis.check_ghosts_from_start:
            logger.info("Looking for present ghost")
            matched_ghost: bool | list[GhostParticle] = self.match_ghost(
                event,
                initial_attempt,  # type: ignore[arg-type]
            )
        else:
            matched_ghost = False

        if matched_ghost is False:
            logger.info("No matches found, creating new ghost")
            return Decision(action=Action.create_new, particle=initial_attempt)  # type: ignore[arg-type]

        matched_ids: list[int | str] = [p.ID for p in matched_ghost]  # type: ignore[union-attr]
        logger.info("Matches found! Particle IDs = %s", matched_ids)

        for potential_match in matched_ghost[  # type: ignore[index]
            : self.parameters.tandemmp_analysis.max_ghost_matches
        ]:
            if potential_match in self.doubles_checked:
                logger.info("Already tried particle %s - skipping", potential_match.ID)
                continue

            if (
                hasattr(potential_match, "double_checked")
                and potential_match.double_checked
            ):
                logger.info("Matched particle is double-checked, attempting again")

            refitted_ghost = self.double_check_ghost(
                event=event,
                close_particles=close_particles,
                original_ghost_particle=potential_match,
                best_error=best_error,
            )

            if refitted_ghost is not False:
                logger.info("Creating double ghost")
                return Decision(action=Action.create_new, particle=refitted_ghost)  # type: ignore[arg-type]

        logger.info("Failed all double fits, assuming new ghost")
        return Decision(action=Action.create_new, particle=initial_attempt)  # type: ignore[arg-type]

    # Initial ghost fit

    def attempt_initial_ghost(  # noqa: PLR0912, PLR0915
        self,
        event: RatiometricEvent,
        close_particles: list[Particle] | None = None,
        preferred_background: int | None = None,
        best_error: float | None = None,
        method: str = "Powell",
    ) -> bool | GhostParticle:
        """Boilerplate for ghost-particle fitting from a single event."""
        close: list[Particle] = list(close_particles) if close_particles else []
        logger.info("Attempting initial Ghost Particle fitting")

        if self.parameters.tandemmp_analysis.check_ghosts_from_start:
            start_frame = min(event.i - event.navg - 3, event.navg)
        else:
            start_frame = max(
                event.i - event.navg - 3, min(event.navg, event.i - event.navg - 1)
            )

        if preferred_background is not None and preferred_background < (
            event.i - event.navg - 1
        ):
            start_frame = preferred_background

        new_particles = set(self.background_check(event=event, frame=start_frame))
        need_correcting = list(new_particles.union(set(close)))

        series: SubtractionSeries = self.correct_series(  # type: ignore[assignment]
            event=event,
            series=self.generate_subtraction_series(event, start_frame, fit=False),
            particles=need_correcting,
        )

        fit = self.fit_initial_ghost(event, series, method=method)

        if fit is False:
            logger.info("Ghost fit trailed off to high mass - premature failure")
            return False
        assert isinstance(fit, tuple)
        res, attempt = fit

        if not self.parameters.save_memory:
            event._corrections_attempts.append(attempt)

        attempt.set_particle("Initial ghost attempt")

        tol: float = self.parameters.tandemmp_analysis.complex_fit_error_tolerance
        beats_basic: bool = attempt.error < (event.basic_fit_error * tol)
        under_max: bool = bool(np.all(c_to_mass(np.asarray(attempt.c)) < 29900))

        beats_fit: bool = (
            beats_basic
            if best_error is None
            else attempt.error <= min(best_error, event.basic_fit_error)
        )

        if not (beats_fit and under_max):
            return False

        logger.info("Initial ghost particle fitted")
        attempt.series.append_original(series)

        ghost_event = create_ghost_event(
            event_template=event,
            fit_result=res.x,
            frame=start_frame,
            peak_model=self.peak_model,
        )
        ghost_particle = GhostParticle(ghost_event)

        event.apply_3D_fit(attempt)
        success = ghost_particle.propose_new_event(event, self._mm, None, None)

        if not success:
            logger.info("Ghost particle rejected its own event")
            event.disconnect_all()
            return False

        if attempt.classification == EvClassification.Dissociation:
            logger.info("Initial ghost fit is dissociation - attempting pure unbinding")

            series_unbind: SubtractionSeries = self.correct_series(  # type: ignore[assignment]
                event=event,
                series=self.generate_subtraction_series(
                    event=event,
                    start_frame=event.i + event.navg + 1,
                    fit=False,
                    custom_frames=sorted(event.frame_range),
                ),
                particles=need_correcting,
            )

            attempt_s = self.find_classification(
                event,
                series_unbind,
                method=method,
                ghost_unbind_series=True,
                particle_c=np.nan,
            )
            assert isinstance(attempt_s, MultiFitResult)
            attempt_s.set_particle("Initial ghost attempt")

            if not self.parameters.save_memory:
                event._corrections_attempts.append(attempt_s)

            if (
                attempt_s.error + attempt_s.penalty + attempt_s.image_error
                < attempt.error + attempt.penalty + attempt.image_error
            ):
                attempt_s.set_particle(
                    f"Initial ghost attempt particle {ghost_particle.ID}"
                )
                logger.info("Better than initial - making removed particle")
                event.disconnect_all()
                event.apply_3D_fit(attempt_s)
                ghost_particle = GhostParticle(ghost_event)
                ghost_particle._add_event(event)
                ghost_particle.unbinding_frame = event.i
                return ghost_particle

            logger.info("Dissociation was better")

        attempt.set_particle(f"Initial ghost attempt particle {ghost_particle.ID}")
        return ghost_particle

    # Ghost fit inner loop

    def fit_initial_ghost(  # noqa: PLR0915
        self,
        event: RatiometricEvent,
        series: SubtractionSeries,
        method: str = "Powell",
    ) -> tuple[OptimizeResult, MultiFitResult] | bool:
        """Fit a candidate ghost impression through the minimiser."""
        original_peaks = [e.peak.image for e in series]

        def make_corrected_series(
            g_dx: float, g_dy: float, c: float, save: bool = False
        ) -> SubtractionSeries:
            x = event.peak.midpoint[0] + g_dx
            y = event.peak.midpoint[1] + g_dy
            out = self._correct_ghost(
                event=event, series=series, x=x, y=y, c=c, save=save
            )
            assert isinstance(out, SubtractionSeries)
            return out

        @functools.lru_cache(maxsize=16)
        def test_correction(
            g_dx: float,
            g_dy: float,
            c: float,
            adjust_err: bool = False,
            no_penalty: bool = False,
            save: bool = False,
        ) -> MultiFitResult:
            new_series = make_corrected_series(g_dx, g_dy, c, save=save)
            fit_result = self.find_classification(
                event,
                new_series,
                particle_c=c,
                allow_native=False,
                lock=False,
                as_fit=True,
            )
            assert isinstance(fit_result, MultiFitResult)

            if c_to_mass(c) < 0:
                fit_result.fit.fun *= 1000

            if not no_penalty:
                fit_result.fit.fun += fit_result.penalty

            if abs(float(c_to_mass(c))) > 900000:
                logger.warning("Caught trailing off")
                raise TrailingOffException()

            if adjust_err:
                fit_result.fit.fun += new_series.basic_fit_error.mean()

            return fit_result

        average = np.mean(original_peaks, axis=0)
        dy_guess, dx_guess = np.transpose(
            np.unravel_index(int(average.argmax()), average.shape)
        )
        dy_guessr, dx_guessr = np.transpose(
            np.unravel_index(int(average.argmin()), average.shape)
        )
        dist_guess: float = float(np.hypot(dx_guess - dx_guessr, dy_guess - dy_guessr))

        if dist_guess >= self.parameters.tandemmp_analysis.psf_radius * 0.9:

            def err(v: npt.NDArray[np.floating]) -> float:
                return test_correction(
                    float(v[0]),
                    float(v[1]),
                    float(v[2]),
                    adjust_err=True,
                    no_penalty=True,
                ).error

        else:

            def err(v: npt.NDArray[np.floating]) -> float:
                return test_correction(
                    float(v[0]),
                    float(v[1]),
                    float(v[2]),
                    adjust_err=False,
                    no_penalty=False,
                ).error

        c_guess = -float(np.max(original_peaks))
        hw = event.peak.image.shape[0] // 2
        guesses = (float(dx_guess - hw), float(dy_guess - hw), c_guess * 1.3)

        try:
            fit = minimize_model(err, x0=list(guesses), method=method)
            res = fit.raw
            fit_result = test_correction(
                float(res.x[0]),
                float(res.x[1]),
                float(res.x[2]),
                save=not self.parameters.save_memory,
            )

            if float(c_to_mass(res.x[-1])) > 30000:
                logger.info("Fitted ghost > 30MDa - ignoring")
                return False
            return res, fit_result
        except (TrailingOffException, KeyboardInterrupt) as exc:
            logger.info("Ghost fitting aborted: %s", exc)
            return False

    # Ghost correction helpers

    def _correct_ghost(  # noqa: PLR0913
        self,
        event: RatiometricEvent,
        x: float,
        y: float,
        c: float,
        series: SubtractionSeries | None = None,
        return_args: bool = False,
        save: bool = False,
    ) -> SubtractionSeries | list[dict[str, Any]]:
        if series is None:
            if event.has_connection:
                series = event.connected_events[0]
            else:
                series = event._first_attempt

        assert series is not None
        start_frame = int(series[0].subtracted_i)
        new_peaks: list[PeakImage] = correct_ghost(
            series=series,  # type: ignore[arg-type]
            dx=x - event.peak.midpoint[0],
            dy=y - event.peak.midpoint[1],
            c=c,
            peak_model=self.peak_model,
            save=save,
            desc="ghost corrected",
        )

        args = [{"peak": p, "subtracted_i": start_frame} for p in new_peaks]
        if return_args:
            return args

        method = functools.partial(
            make_subtraction_event_from_peak, peak_model=self.peak_model
        )
        new_events = dispatch_kwargs(method, args, parallel=self.parameters.parallel)
        return SubtractionSeries(new_events)

    def _recorrect_ghost(  # noqa: PLR0913
        self,
        ghost_particle: GhostParticle,
        event: RatiometricEvent,
        x: float,
        y: float,
        c: float,
        series: SubtractionSeries | None = None,
        return_args: bool = False,
        ez_applicable: bool = True,
        save: bool = False,
    ) -> SubtractionSeries | list[dict[str, Any]]:
        if series is None:
            if event.has_connection:
                series = event.connected_events[0]
            else:
                series = event._first_attempt

        assert series is not None
        start_frame = int(series[0].subtracted_i)

        ghost_event = ghost_particle.binding_event
        if not isinstance(ghost_event, GhostEvent):
            msg = "recorrect_ghost requires a GhostEvent binding_event"
            raise TypeError(msg)
        oldx, oldy, oldc = ghost_event.x, ghost_event.y, ghost_event.c

        reversed_correction: list[PeakImage] = correct_ghost(
            series=[i.peak for i in series],
            dx=oldx - event.peak.midpoint[0],
            dy=oldy - event.peak.midpoint[1],
            c=-oldc,
            peak_model=self.peak_model,
            save=save,
            desc=f"{ghost_particle.ID} reversed ghost",
        )

        new_peaks = correct_ghost(
            series=reversed_correction,
            dx=x - event.peak.midpoint[0],
            dy=y - event.peak.midpoint[1],
            c=c,
            peak_model=self.peak_model,
            save=save,
            desc="Recorrected ghost",
        )

        if event.exclusion_zone.any() and ez_applicable:
            args = [
                {"peak": p, "subtracted_i": start_frame, "exclusion": e}
                for p, e in zip(new_peaks, event.exclusion_zone, strict=True)
            ]
        else:
            args = [{"peak": p, "subtracted_i": start_frame} for p in new_peaks]

        if return_args:
            return args

        method = functools.partial(
            make_subtraction_event_from_peak, peak_model=self.peak_model
        )
        new_events = dispatch_kwargs(method, args, parallel=self.parameters.parallel)
        return SubtractionSeries(new_events)

    def _make_pseudo_event(
        self,
        event: RatiometricEvent,
        series: SubtractionSeries,
        **fc_args: Any,  # noqa: ANN401
    ) -> tuple[RatiometricEvent, bool]:
        new_event = RatiometricEvent(
            event.peak, navg=event.navg, peak_model=event.peak_model
        )
        new_event.overlaps = event.overlaps
        new_event._ID = event._ID
        new_event._first_attempt = event._first_attempt
        attempt = self.find_classification(new_event, series, **fc_args)
        assert isinstance(attempt, MultiFitResult)

        tol = self.parameters.tandemmp_analysis.complex_fit_error_tolerance
        beats_basic: bool = attempt.error < (event.basic_fit_error * tol)

        new_event.apply_3D_fit(attempt)
        return new_event, beats_basic

    # Double ghost fit

    def fit_double_ghost(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        init_event: RatiometricEvent,
        new_event: RatiometricEvent,
        new_series: SubtractionSeries,
        ghost_particle: GhostParticle,
        lock_to_unbinding: bool = False,
        method: str = "Powell",
    ) -> tuple[OptimizeResult, list[MultiFitResult]] | bool:
        """Confirm a ghost via a joint two-event fit."""
        init_im = [ev.peak.image for ev in init_event.connected_events[0]]
        start_frame = int(init_event.connected_events[0].subtracted_i[0])
        new_im = [ev.peak.image for ev in new_series]

        if new_event is init_event:
            msg = "init_event and new_event must be distinct"
            raise ValueError(msg)

        if int(new_series.subtracted_i[0]) != start_frame:
            msg = "new_series must share the same background frame as init_event"
            raise ValueError(msg)

        overlapping_events = [i.event for i in init_event.temporal_overlaps]
        ez_applicable = new_event not in overlapping_events

        if type(ghost_particle) is not GhostParticle:
            logger.info("Performing advanced check on possibly mistaken particle")
            first_correction: Callable[..., Any] = functools.partial(
                self._correct_ghost, event=init_event, return_args=True
            )
        else:
            logger.info("Performing double fit on ghost particle")
            first_correction = functools.partial(
                self._recorrect_ghost,
                ghost_particle=ghost_particle,
                event=init_event,
                ez_applicable=ez_applicable,
                return_args=True,
            )

        def make_corrected_series(
            x: float, y: float, c: float, save: bool = False
        ) -> tuple[SubtractionSeries, SubtractionSeries]:
            first_args = first_correction(x=x, y=y, c=c, save=save)
            second_args = self._correct_ghost(
                event=new_event,
                series=new_series,
                x=x,
                y=y,
                c=c,
                save=save,
                return_args=True,
            )
            all_args = list(first_args) + list(second_args)

            _method = functools.partial(
                make_subtraction_event_from_peak, peak_model=self.peak_model
            )
            all_events = dispatch_kwargs(
                _method, all_args, parallel=self.parameters.parallel
            )
            first_events, second_events = np.reshape(all_events, (2, -1))
            return SubtractionSeries(first_events), SubtractionSeries(second_events)

        def test_correction(  # noqa: PLR0913
            x: float,
            y: float,
            c: float,
            adjust_err: bool = False,
            no_penalty: bool = False,
            save: bool = False,
            as_fit: bool = True,
        ) -> list[MultiFitResult] | bool:
            first_series, second_series = make_corrected_series(x, y, c, save=save)

            fit_result1 = self.find_classification(
                init_event, first_series, particle_c=c, allow_native=True, as_fit=True
            )
            assert isinstance(fit_result1, MultiFitResult)
            fit_result2 = self.find_classification(
                new_event,
                second_series,
                particle_c=fit_result1.series.c[-1],
                allow_native=True,
                as_fit=True,
                ghost_unbind_series=lock_to_unbinding,
            )
            assert isinstance(fit_result2, MultiFitResult)

            if float(c_to_mass(c)) < 0:
                fit_result1.fit.fun *= 1000

            if abs(float(c_to_mass(c))) > 100000:
                logger.info("Caught trailing off")
                return False

            if as_fit:
                if not no_penalty:
                    fit_result1.fit.fun += fit_result1.penalty
                    fit_result2.fit.fun += fit_result2.penalty
                if adjust_err:
                    fit_result1.fit.fun += fit_result1.image_error
                    fit_result2.fit.fun += fit_result2.image_error
                if lock_to_unbinding:
                    fit_result2.fit.fun += fit_result2.image_error

            return [fit_result1, fit_result2]

        average = np.mean(new_im, axis=0)
        dy_guess, dx_guess = np.transpose(
            np.unravel_index(int(average.argmax()), average.shape)
        )
        dy_guessr, dx_guessr = np.transpose(
            np.unravel_index(int(average.argmin()), average.shape)
        )
        hw = average.shape[0] // 2
        dist_guess: float = float(np.hypot(dx_guess - dx_guessr, dy_guess - dy_guessr))

        if dist_guess >= self.parameters.tandemmp_analysis.psf_radius * 0.9:

            def err(v: npt.NDArray[np.floating]) -> float:
                out = test_correction(
                    float(v[0]),
                    float(v[1]),
                    float(v[2]),
                    adjust_err=True,
                    no_penalty=True,
                )
                if out is False:
                    return float("inf")
                return float(np.sum([r.error for r in out]))  # type: ignore[union-attr]

        else:

            def err(v: npt.NDArray[np.floating]) -> float:
                out = test_correction(
                    float(v[0]),
                    float(v[1]),
                    float(v[2]),
                    adjust_err=True,
                    no_penalty=False,
                )
                if out is False:
                    return float("inf")
                return float(np.sum([r.error for r in out]))  # type: ignore[union-attr]

        c_guess = -float(np.max(new_im + init_im))

        if type(ghost_particle) is not GhostParticle:
            guesses = (
                float(new_event.peak.midpoint[0] - (dx_guess - hw)),
                float(new_event.peak.midpoint[1] - (dy_guess - hw)),
                c_guess,
            )
        else:
            guesses = (
                float(ghost_particle.events[0].x),
                float(ghost_particle.events[0].y),
                c_guess,
            )

        fit = minimize_model(
            err,
            x0=list(guesses),
            method=method,
            bounds=[(None, None), (None, None), (-1.5, None)],
        )
        res = fit.raw
        fit_results = test_correction(
            float(res.x[0]),
            float(res.x[1]),
            float(res.x[2]),
            save=not self.parameters.save_memory,
            as_fit=False,
        )
        if fit_results is False:
            return False
        return res, fit_results  # type: ignore[return-value]

    # Ghost matching

    def match_ghost(
        self, event: RatiometricEvent, ghost_particle: GhostParticle
    ) -> bool | list[GhostParticle]:
        """Find the most probable existing ghost particles to match *event*."""
        fit = ghost_particle.events[0]
        # GhostEvent never applies a 3-D fit, so x/y are always plain floats
        # despite AbstractEvent's wider declared type.
        x, y, i = float(fit.x), float(fit.y), fit.i

        ghosts = [g for g in self.particle_manager if g.possible_ghost]
        entries: list[HistoryEntry | None] = [g[i] for g in ghosts]
        has_history = np.array([e is not None for e in entries])
        still_bound = np.array([g.on_slide for g in ghosts])

        if len(entries) == 0:
            return False

        prev_events = [
            [ev for ev in g.events if not isinstance(ev, GhostEvent)] for g in ghosts
        ]
        has_overlap = [
            event in [ovl.event for ev in evs for ovl in ev.temporal_overlaps]  # type: ignore[attr-defined]
            for evs in prev_events
        ]
        automatic_candidates = [
            g
            for g, bound, ovl in zip(ghosts, still_bound, has_overlap, strict=True)
            if bound and ovl
        ]

        suitable_entry = np.array(entries, dtype=object)[has_history & still_bound]
        fake_sigma = 1.5
        probabilities = np.array(
            [
                get_overlap_probability(x, y, fake_sigma, e.x, e.y, fake_sigma)
                for e in suitable_entry
            ]
        )
        viable = (
            probabilities
            > self.parameters.tandemmp_analysis.minimum_matching_probability
        )

        if viable.sum() > 1:
            logger.warning(
                "%d valid ghosts to match to (should ideally be one)",
                int(viable.sum()),
            )
        best_probabilities = np.argsort(probabilities[viable])[::-1]

        pre_candidates = list(
            itertools.compress(
                list(itertools.compress(automatic_candidates, viable)),
                best_probabilities,
            )
        )
        candidates = list(automatic_candidates) + list(pre_candidates)

        logger.info("Matching to %d candidates", len(set(candidates)))

        if len(candidates) > 0:
            return candidates  # type: ignore[return-value]
        return False

    # Double check

    def double_check_ghost(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        event: RatiometricEvent,
        close_particles: list[Particle],
        original_ghost_particle: GhostParticle,
        best_error: float | None = None,
        method: str = "Powell",
        lock_to_unbinding: bool = False,
    ) -> bool | GhostParticle:
        """Confirm or reject a match to *original_ghost_particle*.

        TODO: add success-path coverage for correction, pseudo-event creation,
        classification, and particle removal. ``TestDoubleCheckGhostGuards``
        covers only guard and exception paths.
        """
        pc = self.parameters.tandemmp_analysis.particle_continuation
        init_event: RatiometricEvent
        if not isinstance(original_ghost_particle, GhostParticle):
            init_event = [
                i
                for i in original_ghost_particle.events
                if i.has_connection and i.connected_events[0].ghost_signal_ML > 0.5
            ][0]
        else:
            init_event = original_ghost_particle.events[1]  # type: ignore[assignment]

        init_start_frame: int = int(init_event.connected_events[0].subtracted_i[0])

        if not init_event.has_connection:
            msg = (
                f"Particle {original_ghost_particle.ID} trying to double fit "
                f"and checking against first event ID = {init_event._ID}"
            )
            raise ValueError(msg)

        new_series_uncorrected = self.generate_subtraction_series(
            event, init_start_frame, fit=False
        )

        logger.info(
            "Initiating double check on particle %s", original_ghost_particle.ID
        )

        new_particles = set(self.background_check(event=event, frame=init_start_frame))
        need_correcting = list(new_particles.union(set(close_particles)))

        new_series: SubtractionSeries = self.correct_series(  # type: ignore[assignment]
            event=event,
            series=new_series_uncorrected,
            particles=need_correcting,
            focus=original_ghost_particle,
            gdb=True,
        )

        fit = self.fit_double_ghost(
            init_event=init_event,
            new_event=event,
            new_series=new_series,
            ghost_particle=original_ghost_particle,
            method=method,
            lock_to_unbinding=lock_to_unbinding,
        )

        if fit is False:
            logger.info("Double ghost fit trailed off - premature failure")
            return False
        assert isinstance(fit, tuple)
        res, (new_init_attempt, new_attempt) = fit

        new_attempt.set_particle(f"Double fit {original_ghost_particle.ID}")
        new_init_attempt.set_particle(f"Double fit {original_ghost_particle.ID}")

        if not self.parameters.save_memory:
            event._corrections_attempts.append(new_attempt)
            init_event._corrections_attempts.append(new_init_attempt)
            new_attempt.series.append_original(new_series)

        err1 = new_init_attempt.error
        err2 = new_attempt.error
        tol: float = self.parameters.tandemmp_analysis.complex_fit_error_tolerance
        beats_basic1 = err1 <= init_event.basic_fit_error * tol
        beats_complex1 = err1 <= init_event.fit_3D_error * tol
        beats_series1 = (
            new_init_attempt.series.basic_fit_error.sum()
            <= init_event.connected_events[0].basic_fit_error.sum() * tol
        )
        beats_basic2 = err2 <= event.basic_fit_error * tol
        beats_complex2 = err2 <= best_error if best_error is not None else beats_basic2
        beats_complex2 = beats_complex2 or lock_to_unbinding

        logger.info(
            "Double fit criteria — first: basic=%s complex=%s series=%s; "
            "second: basic=%s complex=%s",
            beats_basic1,
            beats_complex1,
            beats_series1,
            beats_basic2,
            beats_complex2,
        )

        success = beats_basic1 & beats_basic2 & (beats_complex2 | beats_series1)

        if not success:
            return False

        logger.info("Attempting pseudo particle")
        event.apply_3D_fit(new_attempt)

        fitx, fity, fitc = float(res.x[0]), float(res.x[1]), float(res.x[2])
        old_midpoint = init_event.peak.midpoint
        formatted_fit_result = (
            fitx - old_midpoint[0],
            fity - old_midpoint[1],
            fitc,
        )

        ghost_event = create_ghost_event(
            event_template=init_event,
            fit_result=formatted_fit_result,
            frame=init_start_frame,
            peak_model=self.peak_model,
        )
        test = GhostParticle(ghost_event, overwrite_id="PSEUDO")

        if type(original_ghost_particle) is not GhostParticle:
            apply_correction: Callable[..., Any] = functools.partial(
                self._correct_ghost, x=fitx, y=fity, c=fitc
            )
            old_events = original_ghost_particle.events
        else:
            apply_correction = functools.partial(
                self._recorrect_ghost,
                ghost_particle=original_ghost_particle,
                x=fitx,
                y=fity,
                c=fitc,
            )
            old_events = original_ghost_particle.events[1:]

        for index, old_event in enumerate(old_events):
            trial_series = apply_correction(event=old_event)
            expected_c = float(
                np.median(
                    [
                        test[i].c if test[i] else 0  # type: ignore[union-attr]
                        for i in old_event.frame_range  # type: ignore[attr-defined]
                    ]
                )
            )

            new, better_than_basic = self._make_pseudo_event(
                event=old_event,  # type: ignore[arg-type]
                series=trial_series,
                particle_c=expected_c,
            )

            st = None if index == 0 else pc.spatial_threshold  # type: ignore[union-attr]
            mf = None if index == 0 else pc.mass_fluctuation  # type: ignore[union-attr]

            success &= test.propose_new_event(
                event=new,
                minimum_mass=self._mm,
                spatial_threshold=st,
                mass_fluctuation=mf,
            )
            success &= better_than_basic

            if not success:
                break

        success &= test.propose_new_event(
            event=event,
            minimum_mass=self._mm,
            spatial_threshold=pc.spatial_threshold,  # type: ignore[union-attr]
            mass_fluctuation=pc.mass_fluctuation,  # type: ignore[union-attr]
        )

        if not success:
            logger.info("Pseudo particle did not accept all the new events")
            return False

        logger.info("Pseudo particle happy — rewriting assignments for prior events")

        new_ghost_particle = GhostParticle(
            ghost_event, overwrite_id=original_ghost_particle.ID
        )
        sanity_check = True

        for index, old_event in enumerate(old_events):
            expected_c = float(
                np.mean(
                    [
                        new_ghost_particle[i].c if new_ghost_particle[i] else 0  # type: ignore[union-attr]
                        for i in old_event.frame_range  # type: ignore[attr-defined]
                    ]
                )
            )
            recorrected_series = apply_correction(
                event=old_event, save=not self.parameters.save_memory
            )
            old_event.disconnect_all()

            recorrected_attempt = self.find_classification(
                old_event,  # type: ignore[arg-type]
                recorrected_series,
                particle_c=expected_c,
            )
            assert isinstance(recorrected_attempt, MultiFitResult)
            recorrected_attempt.set_particle(
                f"Reconstructed double fit {original_ghost_particle.ID}"
            )

            if not self.parameters.save_memory:
                old_event._corrections_attempts.append(recorrected_attempt)  # type: ignore[attr-defined]

            old_event.apply_3D_fit(recorrected_attempt)  # type: ignore[attr-defined]

            st = None if index == 0 else pc.spatial_threshold  # type: ignore[union-attr]
            mf = None if index == 0 else pc.mass_fluctuation  # type: ignore[union-attr]

            sanity_check &= new_ghost_particle.propose_new_event(
                event=old_event,
                minimum_mass=self._mm,
                spatial_threshold=st,
                mass_fluctuation=mf,
            )

        sanity_check &= new_ghost_particle.propose_new_event(
            event=event,
            minimum_mass=self._mm,
            spatial_threshold=pc.spatial_threshold,  # type: ignore[union-attr]
            mass_fluctuation=pc.mass_fluctuation,  # type: ignore[union-attr]
        )

        new_ghost_particle._last_used_series = new_attempt.series
        if not sanity_check:
            logger.warning(
                "Double ghost passed pseudo check but failed sanity check "
                "(should be identical process)"
            )

        logger.info(
            "Replacing original ghost particle ID = %s", original_ghost_particle.ID
        )
        self.particle_manager.remove(original_ghost_particle)
        new_ghost_particle.double_checked = True
        return new_ghost_particle


# Investigator


@dataclass
class _CandidateContext:
    """Bundles per-candidate scoring state during ``find_and_fit``."""

    expected_cs: list[float | None] = field(default_factory=list)
    series_args: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[Particle | None] = field(default_factory=list)


class Investigator(GhostInvestigator):
    """Top-level analysis class combining classification + ghost handling."""

    def __init__(  # noqa: PLR0913
        self,
        movie: npt.NDArray[np.floating],
        movie_params: Info,
        parameters: FullAnalysisParams,
        particle_manager: ParticleManager,
        peak_model: PsfInterface,
        open_lid_frames: list[int] | None = None,
        event_veto_frames: list[int] | None = None,
    ) -> None:
        super().__init__(
            movie=movie,
            movie_params=movie_params,
            parameters=parameters,
            particle_manager=particle_manager,
            peak_model=peak_model,
            open_lid_frames=open_lid_frames,
        )
        self.event_veto_frames: list[int] = (
            [] if not event_veto_frames else list(event_veto_frames)
        )

    # find_and_fit

    def find_and_fit(  # noqa: PLR0912, PLR0915
        self,
        event: RatiometricEvent,
    ) -> Decision:
        """Decide what to do with *event* — the main pipeline entry point.

        TODO: test matching against ``close_particles`` and ``all_particles``.
        ``TestFindAndFitNoParticles`` covers the no-particles-found branch.
        """
        self.doubles_checked = set()

        l_thum: int = event.peak.image.shape[0]
        psf_radius: float = self.parameters.tandemmp_analysis.psf_radius

        close_particles: list[list[Particle]] = (
            self.particle_manager.particles_in_thumbnail(
                event=event, l_thum=l_thum, psf_radius=psf_radius
            )
        )
        all_particles: list[Particle] = list(
            {particle for frame in close_particles for particle in frame}
        )

        maximum_start_frame: int = max(event.i - event.navg - 1, 0)

        if all_particles:
            particle_frames = [
                p.choose_background(
                    event.i - event.navg, event.peak.midpoint, l_thum, int(psf_radius)
                )
                for p in all_particles
            ]
            minimum_particle_frame: int = int(np.min(particle_frames))
            nc_start_frame: int = int(
                np.nanmin([minimum_particle_frame, maximum_start_frame])
            )
        else:
            nc_start_frame = maximum_start_frame

        no_correction: SubtractionSeries = self.generate_subtraction_series(
            event, nc_start_frame
        )
        event._first_attempt = no_correction

        particle_in_view: list[Particle | None] = [
            i for i in all_particles if i.on_slide
        ] + [None]

        ctx = _CandidateContext()

        # Frames adjacent to particle bindings often carry residual impressions.
        veto_set: set[int] = set()
        for p in all_particles:
            if not isinstance(p, GhostParticle):
                veto_set.add(p.binding_frame - 1)
                veto_set.add(p.binding_frame)
        veto_set.update(int(ev.i) for p in all_particles for ev in p.events)

        need_correcting: set[Particle] = set()

        for candidate in particle_in_view:
            if candidate is None:
                start_frame = maximum_start_frame
            else:
                start_frame = int(
                    min(
                        candidate.choose_background(
                            event.i - event.navg,
                            event.peak.midpoint,
                            l_thum,
                            int(psf_radius),
                        ),
                        maximum_start_frame,
                    )
                )

            if isinstance(candidate, GhostParticle):
                start_frame = candidate.preferred_background

            if start_frame in self.open_lid_frames:
                start_frame = self.open_lid_frames[0] - 1

            if start_frame in veto_set:
                for _ in range(100):
                    if start_frame < 0:
                        start_frame = 0
                        break
                    if start_frame in veto_set:
                        start_frame -= 1
                    else:
                        break

            for sf in set([start_frame, maximum_start_frame, nc_start_frame]):
                new_particles = set(self.background_check(event=event, frame=sf))
                need_correcting = new_particles.union(set(all_particles))

                if candidate is None:
                    expected_c: float | None = None
                else:
                    expected_c = float(
                        np.mean(
                            [
                                candidate[i].c if candidate[i] else 0  # type: ignore[union-attr]
                                for i in event.frame_range
                            ]
                        )
                    )

                ctx.expected_cs.append(expected_c)

                args = self.correct_series(
                    event=event,
                    series=self.generate_subtraction_series(
                        event=event, start_frame=sf, fit=False
                    ),
                    particles=list(need_correcting),
                    focus=candidate,
                    return_peak_args=True,
                )
                assert isinstance(args, list)
                ctx.series_args.extend(args)
                ctx.candidates.append(candidate)

        # Batch-fit all corrected peaks in parallel
        method = functools.partial(
            make_subtraction_event_from_peak, peak_model=self.peak_model
        )
        new_events = dispatch_kwargs(
            method, ctx.series_args, parallel=self.parameters.parallel
        )
        length = event.navg * 2 + 1
        particle_new_series = [
            SubtractionSeries(list(events))
            for events in np.reshape(new_events, (-1, length))
        ]

        classify_args: list[dict[str, Any]] = []
        ghost_unbinders: set[Particle] = set()

        for candidate, s, ec in zip(
            ctx.candidates, particle_new_series, ctx.expected_cs, strict=True
        ):
            partial_fit = (
                candidate is not None and candidate.binding_frame in event.frame_range
            )
            classify_args.append(
                {
                    "event": event,
                    "series": s,
                    "particle_c": ec,
                    "partial_fit": partial_fit,
                    "particle_desc": candidate,
                }
            )
            if candidate and candidate not in ghost_unbinders:
                ghost_unbinders.add(candidate)
                unbind_series: SubtractionSeries = self.correct_series(  # type: ignore[assignment]
                    event=event,
                    series=self.generate_subtraction_series(
                        event=event,
                        start_frame=event.i + event.navg + 1,
                        fit=False,
                        custom_frames=sorted(event.frame_range),
                    ),
                    particles=list(need_correcting),
                    focus=candidate,
                    gdb=True,
                )
                classify_args.append(
                    {
                        "event": event,
                        "series": unbind_series,
                        "particle_c": ec,
                        "partial_fit": partial_fit,
                        "ghost_unbind_series": True,
                        "particle_desc": candidate,
                    }
                )

        classify_args.append(
            {
                "event": event,
                "series": no_correction,
                "particle_c": np.nan,
                "particle_desc": "No correction",
            }
        )

        results: list[MultiFitResult] = []
        for arg in classify_args:
            out = self.find_classification(**arg)
            assert isinstance(out, MultiFitResult)
            results.append(out)

        if not self.parameters.save_memory:
            for r in results:
                event._corrections_attempts.append(r.slim())

        # Acceptance tolerance
        c_orig = abs(event.c)
        c_3d = np.array([abs(float(r.fit.x[0])) for r in results])
        tol = np.full_like(
            c_3d,
            fill_value=self.parameters.tandemmp_analysis.complex_fit_error_tolerance,
        )
        similar_mass = (
            1 + abs(abs(c_3d - c_orig) / c_orig)
        ) < self.parameters.tandemmp_analysis.mass_similarity_flexibility
        tol[similar_mass] = tol[similar_mass] * 2 - 1

        norm_func: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
        if len(results) > 1:
            norm_func = normalise
        else:
            norm_func = lambda i: i  # noqa: E731

        errors = np.array([r.error for r in results])
        norm_errors = norm_func(errors)
        norm_penalty = norm_func(np.array([r.penalty for r in results]))
        norm_image = norm_func(np.array([r.image_error for r in results]))
        norm_distance = norm_func(_distance_array(results))

        score = (
            (norm_errors * 2) + norm_penalty + (norm_image * 2) + (norm_distance / 2)
        )

        suggested_m: list[Any] = [
            (
                c_to_mass(r.c[r.classification.get_slice(event.navg)])
                if len(r.c) > 2
                else c_to_mass(np.asarray(r.c))
            )
            for r in results
        ]
        suggests_negative = np.array(
            [np.mean(np.asarray(m) < -self._mm) >= 0.5 for m in suggested_m]
        )

        under_tolerance = (errors < event.basic_fit_error) & ~suggests_negative

        if not under_tolerance[:-1].sum() > 1:
            under_tolerance = (
                errors < event.basic_fit_error * tol
            ) & ~suggests_negative

        logger.info(
            "%d solutions provided a better fit for this event",
            int(under_tolerance.sum()),
        )

        if not under_tolerance.any():
            if (
                no_correction.ghost_signal_ML
                and no_correction.ghost_signal_ML
                > self.parameters.tandemmp_analysis.ghost_signal_check
            ):
                decision = self.process_ghost(
                    event, all_particles, preferred_background=nc_start_frame
                )
                if decision.action == Action.ignore:
                    logger.info(
                        "Failed ghost fitting for event %s - ignoring", event._ID
                    )
                return decision

            logger.info("No ghost signal")
            event.self_classify()
            return Decision(action=Action.failed, particle=None)

        ascending_error: list[int] = [int(x) for x in np.argsort(score)]

        # Push "No correction" to the last position of the ranked list
        index = int(np.where([i.particle == "No correction" for i in results])[0][0])
        pos = int(np.where(index == np.array(ascending_error))[0][0])
        ascending_error = ascending_error[:pos] + ascending_error[pos + 1 :] + [index]
        ascending_error_arr = np.array(ascending_error)

        checked_ghost = False
        _attempts = 0

        for i in ascending_error_arr[under_tolerance[ascending_error_arr]]:
            _attempts += 1
            best_fit: MultiFitResult = results[int(i)]

            desc = (
                f"Particle {best_fit.particle.ID}"
                if isinstance(best_fit.particle, Particle)
                else str(best_fit.particle)
            )
            logger.info("Looking at complex fit number %d - %s", _attempts, desc)

            background_ahead = best_fit.series.bleed_time[0] < 0
            check_ghost = (
                best_fit.series.ghost_signal_ML is not None
                and best_fit.series.ghost_signal_ML
                > self.parameters.tandemmp_analysis.ghost_signal_check
            )

            if background_ahead and check_ghost:
                logger.info("Ghost signal but background frame ahead - skipping")
                continue

            is_ghost = isinstance(best_fit.particle, GhostParticle)
            is_single_checked = is_ghost and not best_fit.particle.double_checked  # type: ignore[union-attr]

            if check_ghost and (not checked_ghost) and not is_single_checked:
                logger.info("Signal seen - checking for ghost")
                decision = self.process_ghost(
                    event,
                    all_particles,
                    best_error=best_fit.error,
                    preferred_background=int(best_fit.series.subtracted_i[0]),
                )
                checked_ghost = True
                if decision.action == Action.create_new:
                    return decision
                logger.info("Failed ghost fitting for event %s - continuing", event._ID)

            if best_fit.series.unreliability(best_fit.classification) > 0.7:
                logger.info("Series has low SNR - ignoring completely")
                continue

            if best_fit.particle is None:
                event.apply_3D_fit(best_fit)
                logger.info(
                    "Success! %s event fitted but not added to a particle",
                    best_fit.classification,
                )
                return Decision(action=Action.create_new, particle=None)

            if best_fit.particle == "No correction":
                logger.info("No correction had best fit")
                if check_ghost:
                    logger.info("But ghost signal present")
                    continue
                s = best_fit.series[0]
                event.apply_3D_fit(best_fit)
                if best_fit.classification == EvClassification.Binding:
                    return Decision(action=Action.create_new, particle=None)
                matches = self.particle_manager.find_best_matches(s.x, s.y, s.i, s.c)
                for m in matches[:3]:
                    success = m.propose_new_event(
                        event=event,
                        minimum_mass=self._mm,
                        spatial_threshold=self.parameters.tandemmp_analysis.particle_continuation.spatial_threshold  # type: ignore[union-attr]
                        / 2,
                        mass_fluctuation=self.parameters.tandemmp_analysis.particle_continuation.mass_fluctuation,  # type: ignore[union-attr]
                    )
                    if success:
                        logger.info("Matched and accepted by particle %s", m.ID)
                        return Decision(action=Action.updated, particle=m)

                logger.info("Nearest current particles rejected the event")
                continue

            # Corrected + successful fit — attempt to assign
            confirmed = (
                True
                if not is_ghost
                else (best_fit.particle.double_checked and not check_ghost)  # type: ignore[union-attr]
            )
            confirmed |= best_fit.particle.is_possible_split  # type: ignore[union-attr]

            if (not is_ghost) or confirmed:
                event.apply_3D_fit(best_fit)
                success = best_fit.particle.propose_new_event(  # type: ignore[union-attr]
                    event=event,
                    minimum_mass=self._mm,
                    spatial_threshold=self.parameters.tandemmp_analysis.particle_continuation.spatial_threshold,  # type: ignore[union-attr]
                    mass_fluctuation=self.parameters.tandemmp_analysis.particle_continuation.mass_fluctuation,  # type: ignore[union-attr]
                )
                if success:
                    logger.info(
                        "Success! %s event added to particle %s",
                        best_fit.classification,
                        best_fit.particle.ID,  # type: ignore[union-attr]
                    )
                    return Decision(
                        action=Action.updated,
                        particle=best_fit.particle,  # type: ignore[arg-type]
                    )
                logger.info(
                    "Failed! %s event given to particle %s but rejected",
                    best_fit.classification,
                    best_fit.particle.ID,  # type: ignore[union-attr]
                )
                if (
                    best_fit.particle.possible_ghost  # type: ignore[union-attr]
                    and best_fit.distance is not None
                    and best_fit.distance < 7
                ):
                    logger.info("Particle is a ghost, attempting double check")
                    lock_to_unbinding = (
                        best_fit.classification == EvClassification.Unbinding
                    )
                    refitted_ghost = self.double_check_ghost(
                        event,
                        list(need_correcting),
                        best_fit.particle,  # type: ignore[arg-type]
                        best_fit.error,
                        lock_to_unbinding=lock_to_unbinding,
                    )
                    self.doubles_checked.add(best_fit.particle)  # type: ignore[arg-type]
                    if refitted_ghost:
                        return Decision(
                            action=Action.create_new,
                            particle=refitted_ghost,  # type: ignore[arg-type]
                        )
                else:
                    continue
            else:
                if not checked_ghost and (
                    best_fit.particle not in self.doubles_checked
                ):
                    refitted_ghost = self.double_check_ghost(
                        event,
                        list(need_correcting),
                        best_fit.particle,  # type: ignore[arg-type]
                        best_fit.error,
                    )
                    self.doubles_checked.add(best_fit.particle)  # type: ignore[arg-type]
                    if refitted_ghost:
                        return Decision(
                            action=Action.create_new,
                            particle=refitted_ghost,  # type: ignore[arg-type]
                        )
                else:
                    continue

        if not checked_ghost:
            if (
                no_correction.ghost_signal_ML
                and no_correction.ghost_signal_ML
                > self.parameters.tandemmp_analysis.ghost_signal_check
            ):
                decision = self.process_ghost(
                    event, all_particles, preferred_background=nc_start_frame
                )
                if decision.action == Action.ignore:
                    logger.info(
                        "Failed ghost fitting for event %s - ignoring", event._ID
                    )
                return decision

            logger.info("No ghost signal")
            event.disconnect_all()
            event.self_classify()
            return Decision(action=Action.failed, particle=None)

        logger.info("Run out of possible explanations")
        action = Action.ignore if no_correction.ghost_signal_ML else Action.failed
        return Decision(action=action, particle=None)

    # multi_continuation

    def multi_continuation(  # noqa: PLR0912, PLR0915
        self,
        latest_event: RatiometricEvent,
    ) -> None:
        """Continue subtraction-event tracking for particles already on slide.

        TODO: add direct test coverage for subtraction-event continuation.
        """
        logger.info("Beginning multi continuation")

        all_particles: list[Particle] = self.particle_manager.get_all()
        c_params = self.parameters.tandemmp_analysis.particle_continuation
        min_mass: float = self._mm

        if self.parameters.tandemmp_analysis.correct_with_images:
            correction: Callable[..., PeakImage] = correct_subtraction_image
        else:
            correction = functools.partial(
                correct_subtraction_fit, psf_model=self.peak_model
            )

        save: bool = not self.parameters.save_memory

        tracked_particles = [p for p in all_particles if p.on_slide]
        initial_images: list[SubtractionSeries] = []
        particles_to_continue: list[Particle] = []

        for particle in tracked_particles:
            last_ratiometric: RatiometricEvent = particle._event_map.get_ratiometric(  # type: ignore[assignment]
                latest_event.i + latest_event.navg
            )
            if particle._last_used_series is None:
                subtracted_i: int = particle.binding_frame - 3
            else:
                subtracted_i = int(particle._last_used_series.subtracted_i[0])

            if particle.fixed_background_events:
                beginning_frame = particle.last_continuation.i + 1
                subtracted_i = int(particle.last_continuation.subtracted_i)
            else:
                beginning_frame = last_ratiometric.i + 1

            if len(particle.rejected) != 0:
                beginning_frame = max(
                    beginning_frame, max(ev.i + 1 for ev in particle.rejected)
                )

            next_overlap = None
            if last_ratiometric == latest_event:
                if latest_event.thread_intersections:
                    next_overlap = latest_event.thread_intersections[0]
                    next_frame = next_overlap.frames
                else:
                    next_frame = len(self.movie)
            else:
                if last_ratiometric.thread_intersections:
                    previous_intersecting_events = np.array(
                        [ov.event for ov in last_ratiometric.thread_intersections]
                    )
                    if latest_event in previous_intersecting_events:
                        current_index = int(
                            np.where(previous_intersecting_events == latest_event)[0][0]
                        )
                        if (
                            latest_event
                            == last_ratiometric.thread_intersections[-1].event
                        ):
                            next_frame = len(self.movie)
                        else:
                            next_overlap = last_ratiometric.thread_intersections[
                                current_index + 1
                            ]
                            next_overlap.event.following_threads.append(particle.ID)
                            next_frame = next_overlap.frames
                    else:
                        next_overlap = last_ratiometric.thread_intersections[0]
                        next_overlap.event.following_threads.append(particle.ID)
                        next_frame = next_overlap.frames
                else:
                    next_frame = len(self.movie)

            if isinstance(next_frame, list):
                next_frame = next_frame[0]

            if beginning_frame >= next_frame:
                continue

            frames = np.arange(beginning_frame, next_frame)
            logger.info(
                "Continuation of particle %s for %d frames", particle.ID, len(frames)
            )

            dis_series = self.generate_subtraction_series(
                event=last_ratiometric,
                start_frame=subtracted_i,
                custom_frames=[int(f) for f in frames.tolist()],
                fit=False,
            )
            initial_images.append(dis_series)
            particles_to_continue.append(particle)

        if not particles_to_continue:
            return

        unique_frames = sorted({ev.i for series in initial_images for ev in series})
        still_tracked = bool(np.any([p.on_slide for p in particles_to_continue]))

        while still_tracked:
            for frame in tqdm.tqdm(
                unique_frames,
                desc=f"Continuation of {len(particles_to_continue)} particles",
                unit="frames",
            ):
                if frame in self.open_lid_frames:
                    continue

                still_bound = np.array([p.on_slide for p in particles_to_continue])
                still_tracked = bool(still_bound.any())

                init_events: list[SubtractionEvent | None] = [
                    series.get_by_frame(frame) for series in initial_images
                ]
                valid = np.array([i is not None for i in init_events]) & still_bound

                if np.any(valid):
                    images = list(itertools.compress(init_events, valid))
                    continuing_particles = list(
                        itertools.compress(particles_to_continue, valid)
                    )
                    backgrounds = [i.subtracted_i for i in images]  # type: ignore[union-attr]
                    nearby = [
                        self.particle_manager.get_close_particles(
                            frame=latest_event.i + latest_event.navg + 1,
                            x=p[frame].x,  # type: ignore[union-attr]
                            y=p[frame].y,  # type: ignore[union-attr]
                            distance=self.parameters.first_analysis.l_thum // 2
                            + self.parameters.tandemmp_analysis.psf_radius
                            - 2,
                        )
                        for p in continuing_particles
                    ]
                    veto: set[int] = set()
                    for collection in nearby:
                        for p in collection:
                            if not isinstance(p, GhostParticle):
                                veto.add(p.binding_frame - 1)
                    veto.update(i + 1 for i in list(veto))

                    if frame in veto:
                        continue

                    corrected_peaks = [
                        correction(ev, particles=n, focus=p, save=save)
                        for ev, p, n in zip(
                            images, continuing_particles, nearby, strict=True
                        )
                    ]

                    args = [
                        {"peak": p, "subtracted_i": si, "fit": True}
                        for p, si in zip(corrected_peaks, backgrounds, strict=True)
                    ]

                    method = functools.partial(
                        make_subtraction_event_from_peak, peak_model=self.peak_model
                    )
                    events = dispatch_kwargs(
                        method,
                        args,
                        parallel=(len(args) > 15) and self.parameters.parallel,
                    )

                    order = np.argsort([i.binding_frame for i in continuing_particles])

                    for new_ev, c_particle in zip(
                        np.array(events)[order],
                        np.array(continuing_particles)[order],
                        strict=True,
                    ):
                        c_particle.propose_new_event(
                            event=new_ev,
                            minimum_mass=min_mass,
                            spatial_threshold=c_params.spatial_threshold,  # type: ignore[union-attr]
                            mass_fluctuation=c_params.mass_fluctuation,  # type: ignore[union-attr]
                            max_strikes=c_params.max_strikes,  # type: ignore[union-attr]
                        )
                else:
                    continue
            return
