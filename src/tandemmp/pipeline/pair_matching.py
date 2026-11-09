"""Standalone pair-matching entry point.

Loads a movie via :class:`~tandemmp.pipeline.loader.Loader` (which
drift-corrects and finds/fits first-pass events in the usual way — see
:class:`~tandemmp.core.parameters.FullAnalysisParams`'s ``drift_correction``
field, on by default), converts the resulting first-pass
:class:`~tandemmp.events.RatiometricEvent`\\ s into
:class:`~tandemmp.pipeline.simple_analysis.SimpleEvent`\\ s via
:func:`~tandemmp.pipeline.simple_analysis.build_simple_events`, and matches
binding/unbinding pairs via
:func:`~tandemmp.pipeline.simple_analysis.match_pairs_sigma`.

Supports movie slicing via ``frame_start`` and ``frame_end`` without running
the PSF-fit-based tracking loop. :func:`run_pair_matching` returns matched
pairs and events in a :class:`PairMatchingResult`; callers handle plotting.
:attr:`PairMatchingResult.initial_masses`/:attr:`~PairMatchingResult.secondary_masses`
can be passed to :func:`~tandemmp.plotting.experiment.mass_tablecloth`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from tandemmp.core.conversions import set_calibration
from tandemmp.core.parameters import get_parameters
from tandemmp.core.types import FloatArray
from tandemmp.events import RatiometricEvent
from tandemmp.pipeline.calibration import CalibrationLike, resolve_calibration
from tandemmp.pipeline.loader import Loader
from tandemmp.pipeline.pipeline_interface import PipelineRunner
from tandemmp.pipeline.simple_analysis import (
    Pair,
    build_simple_events,
    match_pairs_sigma,
)

# Result container


@dataclasses.dataclass(frozen=True)
class PairMatchingResult:
    """The outcome of :func:`run_pair_matching` on a single movie.

    ``events`` holds every first-pass event the movie produced (not just
    those that ended up in a pair), in case a caller wants to inspect or
    re-match against the full set.
    """

    pairs: list[Pair]
    events: list[RatiometricEvent]
    source_filepath: str

    @property
    def initial_masses(self) -> FloatArray:
        """Binding-event mass for each matched pair (kDa)."""
        return np.array([p.binding.m for p in self.pairs], dtype=float)

    @property
    def secondary_masses(self) -> FloatArray:
        """Unbinding-event mass for each matched pair (kDa)."""
        return np.array([p.unbinding.m for p in self.pairs], dtype=float)


# Orchestration


def run_pair_matching(
    filepath: str | Path,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    seed: int | np.random.Generator | np.random.RandomState | None = None,
    pipeline_runner: PipelineRunner | None = None,
    calibration: CalibrationLike | None = None,
    sigma_multiplier: float = 3.0,
    max_pairs: int = 5,
    analyse_distance: float = 4.0,
    minimum_overlap: float = 0.01,
    match_binding: bool = False,
    parallel: bool = True,
    **parameter_overrides: Any,  # noqa: ANN401
) -> PairMatchingResult:
    """Load a movie, drift-correct it, and match binding/unbinding event pairs.

    The movie is loaded via :class:`~tandemmp.pipeline.loader.Loader` with
    its default full first-pass pipeline (event finding *and* drift
    correction, since ``FullAnalysisParams.drift_correction`` defaults to
    on) — pass e.g. ``drift_correction=False`` or ``frame_gap=50`` via
    *parameter_overrides* to tune or disable it (forwarded to
    :func:`~tandemmp.core.parameters.get_parameters`, which resolves
    keywords against the full, possibly-nested
    :class:`~tandemmp.core.parameters.FullAnalysisParams` tree).

    *calibration* (a :class:`~tandemmp.core.conversions.Calibration` or a
    calibration filepath) sets the process-wide contrast/mass conversion
    before events are loaded; left as ``None``, whatever calibration is
    already active is used unchanged.

    See :func:`~tandemmp.pipeline.simple_analysis.match_pairs_sigma` for the
    meaning of *sigma_multiplier*, *max_pairs*, *analyse_distance*,
    *minimum_overlap* and *match_binding*.
    """
    if calibration is not None:
        set_calibration(resolve_calibration(calibration))
    parameters = get_parameters(parallel=parallel, **parameter_overrides)

    loader = Loader(
        filepath,
        parameters=parameters,
        frame_start=frame_start,
        frame_end=frame_end,
        seed=seed,
        pipeline_runner=pipeline_runner,
    )

    simple_events = build_simple_events(loader.events)

    pairs = match_pairs_sigma(
        simple_events,
        sigma_multiplier=sigma_multiplier,
        max_pairs=max_pairs,
        analyse_distance=analyse_distance,
        minimum_overlap=minimum_overlap,
        match_binding=match_binding,
    )

    return PairMatchingResult(
        pairs=pairs,
        events=loader.events,
        source_filepath=str(filepath),
    )


if __name__ == "__main__":
    from tandemmp.core.functions import choose_file

    movie_filepath = choose_file()
    if not isinstance(movie_filepath, str):
        raise TypeError("Select a single movie file.")

    result = run_pair_matching(movie_filepath)
    print(
        f"{len(result.pairs)} pair(s) matched from {len(result.events)} "
        f"event(s) in {movie_filepath}"
    )
