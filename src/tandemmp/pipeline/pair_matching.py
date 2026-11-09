"""Standalone pair-matching entry point.

Thin wrapper around :class:`~tandemmp.pipeline.pca.Experiment` and
:meth:`~tandemmp.pipeline.pca.Experiment.simple_analysis` — movie
loading/slicing/drift-correction is all handled by
:class:`~tandemmp.pipeline.loader.Loader` (via ``Experiment``), without
running the PSF-fit-based tracking loop (:meth:`Experiment.analyse` is never
called). :func:`run_pair_matching` returns matched pairs and events in a
:class:`PairMatchingResult`; callers handle plotting.
:attr:`PairMatchingResult.initial_masses`/:attr:`~PairMatchingResult.secondary_masses`
can be passed to :func:`~tandemmp.plotting.experiment.mass_tablecloth`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from tandemmp.core.types import FloatArray
from tandemmp.events import RatiometricEvent
from tandemmp.pipeline.pca import Experiment
from tandemmp.pipeline.pipeline_interface import PipelineRunner
from tandemmp.pipeline.simple_analysis import Pair

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


def run_pair_matching(  # noqa: PLR0913
    filepath: str | Path,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    seed: int | np.random.Generator | np.random.RandomState | None = None,
    pipeline_runner: PipelineRunner | None = None,
    events_csv: str | Path | None = None,
    sigma_multiplier: float = 3.0,
    max_pairs: int = 5,
    analyse_distance: float = 4.0,
    minimum_overlap: float = 0.01,
    match_binding: bool = False,
    parallel: bool = True,
    **parameter_overrides: Any,  # noqa: ANN401
) -> PairMatchingResult:
    """Load a movie, drift-correct it, and match binding/unbinding event pairs.

    Thin wrapper around :class:`~tandemmp.pipeline.pca.Experiment` and
    :meth:`~tandemmp.pipeline.pca.Experiment.simple_analysis` — see
    :func:`~tandemmp.pipeline.simple_analysis.match_pairs_sigma` for the
    meaning of *sigma_multiplier*, *max_pairs*, *analyse_distance*,
    *minimum_overlap* and *match_binding*. ``**parameter_overrides`` is
    forwarded to :func:`~tandemmp.core.parameters.get_parameters` (e.g.
    ``drift_correction=False``).
    """
    experiment_kwargs: dict[str, Any] = {
        "pipeline_runner": pipeline_runner,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "seed": seed,
        "parallel": parallel,
        **parameter_overrides,
    }
    if events_csv is not None:
        experiment_kwargs["events_csv"] = events_csv

    experiment = Experiment(filepath, **experiment_kwargs)

    pairs = experiment.simple_analysis(
        sigma_multiplier=sigma_multiplier,
        max_pairs=max_pairs,
        analyse_distance=analyse_distance,
        minimum_overlap=minimum_overlap,
        match_binding=match_binding,
    )

    return PairMatchingResult(
        pairs=pairs,
        events=experiment.events,
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
