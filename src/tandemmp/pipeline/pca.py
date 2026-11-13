"""Top-level analysis orchestration for TandemMP.

:class:`Experiment` ties together :class:`~tandemmp.pipeline.loader.Loader` (movie
loading and first-pass event detection),
:class:`~tandemmp.pipeline.investigator.Investigator` (per-event classification
and fitting) and :class:`~tandemmp.tracking.manager.ParticleManager` (tracked
particles) into the entry point for a full particle-centric analysis of an iSCAT
movie.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

from tandemmp.core.classifications import EvClassification
from tandemmp.core.conversions import get_calibration
from tandemmp.core.parameters import FullAnalysisParams, get_parameters
from tandemmp.core.types import FloatArray, UInt16Array
from tandemmp.events import RatiometricEvent
from tandemmp.fitting.psf_interface import PsfInterface
from tandemmp.io.save import (
    ResultsFile,
    build_event_records,
    build_particle_records,
    save_results,
)
from tandemmp.pipeline.investigator import Action, Decision, Investigator
from tandemmp.pipeline.loader import Loader
from tandemmp.pipeline.pipeline_interface import PipelineRunner
from tandemmp.pipeline.simple_analysis import (
    Pair,
    build_simple_events,
    match_pairs_sigma,
)
from tandemmp.tracking.manager import ParticleManager
from tandemmp.tracking.particle import GhostParticle, Particle

logger = logging.getLogger(__name__)


class Experiment:
    """Orchestrates a single particle-centric TandemMP analysis.

    Loads a movie via :class:`~tandemmp.pipeline.loader.Loader`, then steps through
    detected events with :meth:`analyse`, classifying and fitting each with an
    :class:`~tandemmp.pipeline.investigator.Investigator` and accumulating tracked
    particles in a :class:`~tandemmp.tracking.manager.ParticleManager`.

    *pipeline_runner* selects the event-finding backend (see
    :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`; the Loader's
    default is :class:`~tandemmp.pipeline.open_iscat_pipeline.OpenIscatPipeline`).
    *peak_model* sets the PSF model (default: bundled Rust spline).
    """

    def __init__(
        self,
        mp_filepath: str | Path,
        *,
        just_movie: bool = False,
        special_bypass: UInt16Array | None = None,
        pipeline_runner: PipelineRunner | None = None,
        peak_model: PsfInterface | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        self.parameters: FullAnalysisParams = get_parameters(**kwargs)
        self.loader = Loader(
            mp_filepath,
            parameters=self.parameters,
            just_movie=just_movie,
            special_bypass=special_bypass,
            pipeline_runner=pipeline_runner,
            peak_model=peak_model,
        )
        self.folder = self.loader.folder
        self.movie = self.loader.movie
        self.movie_params = self.loader.params

        if just_movie:
            return

        if self.movie_params is None:
            msg = (
                "Full analysis requires movie metadata (Loader.params); "
                "pass a real mp_filepath, or use just_movie=True with "
                "special_bypass for metadata-free movie loading."
            )
            raise ValueError(msg)

        self._events: list[RatiometricEvent] = self.loader.events
        self.particles = ParticleManager()
        self.investigator = Investigator(
            movie=self.movie,
            movie_params=self.movie_params,
            parameters=self.parameters,
            particle_manager=self.particles,
            peak_model=self.loader.peak_model,
            open_lid_frames=list(self.loader.exclusion_range),
            event_veto_frames=[e.i for e in self._events],
        )

        self._index = 0
        self.interesting: list[tuple[RatiometricEvent, Particle | None]] = []

    # Event views

    @property
    def events(self) -> list[RatiometricEvent]:
        """All first-pass events, sorted by ID."""
        return sorted(self._events, key=lambda e: e._ID)  # noqa: SLF001

    @property
    def clean_events(self) -> list[RatiometricEvent]:
        """Events that have been investigated and were not discarded."""
        return [
            e
            for e in self.events
            if e.decision is not None and e.decision.action != Action.ignore
        ]

    # Simple pair matching

    def simple_analysis(
        self,
        *,
        sigma_multiplier: float = 3.0,
        max_pairs: int = 5,
        analyse_distance: float = 4.0,
    ) -> list[Pair]:
        """Match binding/unbinding events by spatial and temporal overlap.

        Returns the matched pairs; see :mod:`tandemmp.plotting.experiment`
        for visualising the result.
        """
        simple_events = build_simple_events(self.events)
        return match_pairs_sigma(
            simple_events,
            sigma_multiplier=sigma_multiplier,
            max_pairs=max_pairs,
            analyse_distance=analyse_distance,
        )

    # Analysis loop

    def reset(self) -> None:
        """Undo in-progress classification state so :meth:`analyse` can restart.

        Re-fits events up to (and just past) the current progress index, clears all
        tracked particles and resets :attr:`_index` to zero.
        """
        for event in self.events[: self._index + 2]:
            event.disconnect_all()
            event.basic_fit()
        self.particles = ParticleManager()
        self._index = 0

    def analyse(self, *, refresh: bool = False, stop_at: int = -1) -> None:
        """Step through :attr:`events`, classifying and tracking each one.

        Resumes from :attr:`_index` unless *refresh* is set (which calls :meth:`reset`
        first). Stops after event index *stop_at* if given.
        """
        if refresh:
            self.reset()

        continuation = self.parameters.tandemmp_analysis.particle_continuation

        for index, event in enumerate(self.events):
            if index == stop_at:
                return
            if index < self._index:
                continue

            if self._index > 0 and self.parameters.supervise:  # pragma: no cover
                # Interactive plotting, deliberately untested here.
                from tandemmp.plotting.events import report_and_draw

                report_and_draw(self.events[self._index - 1])

            logger.debug("Analysing event %d", index)
            if index % 500 == 0:
                gc.collect()

            decision = self.investigator.find_and_fit(event=event)
            event.decision = decision

            if decision.interesting:
                self.interesting.append((event, decision.particle))

            self._index += 1

            new_particle: Particle | None = None

            if decision.action == Action.create_new:
                if decision.particle is not None:
                    new_particle = decision.particle
                elif event.classification != EvClassification.Binding:
                    msg = (
                        "Investigator requested a new particle from a non-Binding event"
                    )
                    raise RuntimeError(msg)
                elif np.any(np.asarray(event.rel_m) < self.parameters.minimum_mass):
                    logger.info("Won't make a particle with such small mass")
                    event.decision = Decision(action=Action.ignore, particle=None)
                else:
                    new_particle = Particle(event)

                if new_particle is not None:
                    self.particles.append(new_particle)

            elif decision.action == Action.failed:
                if (
                    event.classification == EvClassification.Binding
                    and event.particle_m > self.parameters.minimum_mass
                    and np.any(np.abs(event.dx) < 2.5)
                    and np.any(np.abs(event.dy) < 2.5)
                ):
                    logger.info("Making new particle from failed classification")
                    new_particle = Particle(event)
                    self.particles.append(new_particle)
                else:
                    logger.info("Failed basic checks, no action taken.")

            elif decision.action == Action.ignore:
                event.disconnect_all()

            # Action.updated - Investigator has already enacted changes itself.

            if continuation:
                self.investigator.multi_continuation(latest_event=event)

                if new_particle is not None and not isinstance(
                    new_particle, GhostParticle
                ):
                    candidates = [
                        p for p in self.particles.get_all() if p.lost_tracking
                    ]
                    coincidental = [
                        p
                        for p in candidates
                        if p.unbinding_frame is not None
                        and abs(p.unbinding_frame - event.i) <= 1
                    ]
                    if coincidental:
                        logger.info("%d coincidence(s) found!", len(coincidental))
                        for trial_p in coincidental:
                            success = trial_p.propose_new_event(
                                event=event,
                                minimum_mass=self.parameters.minimum_mass,
                                spatial_threshold=30,
                                mass_fluctuation=continuation.mass_fluctuation  # type: ignore[union-attr]
                                / 2,
                                max_strikes=continuation.max_strikes + 1,  # type: ignore[union-attr]
                                coincidence_checking=True,
                            )
                            if success:
                                self.particles.remove(new_particle)
                                logger.info("Coincidence successfully accepted!")
                                trial_p.unbinding_frame = None
                                trial_p.lost_tracking = False
                            else:
                                logger.info(
                                    "Coincidence rejected by particle %s", trial_p.ID
                                )

    # Persistence

    def save(self, filepath: str | Path, *, save_movie: bool = True) -> Path:
        """Pickle this experiment's events and particles to *filepath*."""
        results = ResultsFile(
            source_filepath=str(self.loader.mp_filepath),
            parameters=self.parameters,
            drift_vector=self.loader.vector,
            events=build_event_records(
                self.events,
                old_masses=cast("list[float | FloatArray]", self.loader.old_mass),
            ),
            particles=build_particle_records(self.particles.get_all()),
            movie=self.movie if save_movie else None,
            calibration=get_calibration(),
        )
        return save_results(filepath, results)
