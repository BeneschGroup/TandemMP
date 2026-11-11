"""Batch orchestration for running :class:`~tandemmp.pipeline.pca.Experiment`."""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tandemmp.core.conversions import Calibration, set_calibration
from tandemmp.core.parallel import dispatch_kwargs
from tandemmp.fitting.gaussian_psf import GaussianPsfModel
from tandemmp.fitting.psf_interface import PsfInterface
from tandemmp.pipeline.pca import Experiment
from tandemmp.pipeline.pipeline_interface import PipelineRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchResult:
    """Outcome of running :func:`process_movie` on a single file."""

    filepath: Path
    duration_seconds: float
    n_events: int
    skipped: bool
    output_filepath: Path | None


def process_movie(
    filepath: str | Path,
    *,
    parameters: Mapping[str, Any] | None = None,
    save_filepath: str | Path | None = None,
    max_events: int = 3500,
    pipeline_runner: PipelineRunner | None = None,
    calibration: Calibration | None = None,
    save_movie: bool = False,
    peak_model: PsfInterface | None = None,
) -> BatchResult:
    """Run a full :class:`~tandemmp.pipeline.pca.Experiment` analysis on one movie.

    Movies with *max_events* or more first-pass events are skipped rather
    than analysed, since that many events usually indicates a bad movie
    (dust, drift, or a mis-set threshold) rather than genuine data.

    Results are pickled to *save_filepath*, defaulting to *filepath* with
    ``_tandemmp`` appended before its (``.pkl``) suffix.

    *pipeline_runner* selects the event-finding backend (see
    :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`); defaults
    to open_iscat.

    *calibration* overrides the process-wide contrast/mass conversion (see
    :mod:`tandemmp.core.conversions`) for the duration of this call — set
    explicitly (including resetting to the default when ``None``) so this is
    safe to call repeatedly against a persistent, possibly-reused worker
    process (see :func:`batch_process`).

    *peak_model* is the PSF model to use; defaults to the Bessel ``PsfModel``.
    """
    set_calibration(calibration)

    filepath = Path(filepath)
    start = time.monotonic()
    experiment = Experiment(
        filepath,
        pipeline_runner=pipeline_runner,
        peak_model=peak_model,
        **(parameters or {}),
    )
    n_events = len(experiment.events)

    if n_events >= max_events:
        logger.warning(
            "%s: %d events found, skipping (limit %d)", filepath, n_events, max_events
        )
        del experiment
        gc.collect()
        return BatchResult(
            filepath=filepath,
            duration_seconds=time.monotonic() - start,
            n_events=n_events,
            skipped=True,
            output_filepath=None,
        )

    logger.info("%s: %d events found", filepath, n_events)
    experiment.analyse()
    output_filepath = (
        Path(save_filepath)
        if save_filepath is not None
        else filepath.with_name(f"{filepath.stem}_tandemmp.pkl")
    )
    experiment.save(output_filepath, save_movie=save_movie)

    del experiment
    gc.collect()
    return BatchResult(
        filepath=filepath,
        duration_seconds=time.monotonic() - start,
        n_events=n_events,
        skipped=False,
        output_filepath=output_filepath,
    )


def batch_process(
    filepaths: Sequence[str | Path],
    *,
    parameters: Mapping[str, Any] | None = None,
    max_events: int = 3500,
    parallel: bool = True,
    n_jobs: int | None = None,
    pipeline_runner: PipelineRunner | None = None,
    calibration: Calibration | None = None,
    peak_model: PsfInterface | None = None,
) -> list[BatchResult]:
    """Run :func:`process_movie` over *filepaths*, in parallel by default.

    Each movie can use on the order of tens of GB of RAM to analyse, so
    *n_jobs* defaults to ``min(6, len(filepaths))`` to bound peak memory use
    rather than spawning one worker per file.

    The underlying worker pool (:func:`tandemmp.core.parallel.dispatch_kwargs`)
    persists for the process lifetime rather than being recreated per call,
    so expensive per-worker state each worker process lazily initialises
    (e.g. a TensorFlow ghost model, cached via :func:`functools.lru_cache`)
    is set up at most once per worker and reused across subsequent
    :func:`batch_process` calls in the same process — at the cost of that
    per-worker RAM staying resident (additive to the per-movie RAM noted
    above) for as long as the process runs. Calling with a different
    *n_jobs* than a previous call tears down and recreates the pool.

    Note: :func:`process_movie` (via :class:`~tandemmp.pipeline.pca.Experiment`)
    itself calls :func:`~tandemmp.core.parallel.dispatch_kwargs` internally
    for its own sub-tasks (e.g. via ``FullAnalysisParams(parallel=True)``).
    Pool workers are daemonic and cannot spawn child pools of their own, so
    combining ``batch_process(parallel=True)`` with a ``parallel=True``
    analysis-parameters object raises ``AssertionError: daemonic processes
    are not allowed to have children``. This is a pre-existing constraint of
    :class:`multiprocessing.Pool` itself, not something this persistent-pool
    change introduces or fixes — avoid enabling both parallelism layers at
    once.

    *pipeline_runner* selects the event-finding backend (see
    :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`); defaults
    to open_iscat. When *parallel* is ``True`` it must be picklable, since
    each :func:`process_movie` call (and the *pipeline_runner* passed to it)
    is sent across the :class:`multiprocessing.Pool` boundary —
    :class:`~tandemmp.pipeline.open_iscat_pipeline.OpenIscatPipeline` is
    stateless and satisfies this, as does the default ``None``.

    *calibration* overrides the process-wide contrast/mass conversion (see
    :mod:`tandemmp.core.conversions`) for every movie in this batch; it is
    forwarded to each :func:`process_movie` call the same way as
    *pipeline_runner*, so it is applied per-call and survives worker reuse.

    *peak_model* is forwarded to each :func:`process_movie` call; it must be
    picklable (a top-level class) when *parallel* is ``True``.
    """
    args_list: list[dict[str, Any]] = [
        {
            "filepath": fp,
            "parameters": parameters,
            "max_events": max_events,
            "pipeline_runner": pipeline_runner,
            "calibration": calibration,
            "peak_model": peak_model,
        }
        for fp in filepaths
    ]
    resolved_n_jobs = n_jobs if n_jobs is not None else min(6, len(filepaths))
    results = dispatch_kwargs(
        process_movie, args_list, parallel=parallel, n_jobs=resolved_n_jobs
    )

    for result in results:
        logger.info(
            f"{result.filepath.stem}: {result.n_events} events, "
            f"{result.duration_seconds:.1f}s, "
            f"{'skipped' if result.skipped else 'processed'}"
        )

    return results


if __name__ == "__main__":
    from tandemmp.core.functions import choose_file
    from tandemmp.pipeline.calibration import (
        load_calibration,
        run_calibration,
        save_calibration,
    )

    # PSF model for this run: any PsfInterface implementation, e.g.
    # ``GaussianPsfModel(sigma=1.5)`` or your own class. ``None`` = Bessel.
    PEAK_MODEL: PsfInterface | None = GaussianPsfModel()

    calibration_file = choose_file(mp_file=False)
    if not isinstance(calibration_file, str):
        raise TypeError("Select a single calibration movie or calibration file.")

    if calibration_file.endswith((".json", ".pkl", ".pickle")):
        calibration_result = load_calibration(calibration_file)
    else:
        calibration_result = run_calibration(calibration_file)
        calibration_path = Path(calibration_file).with_name(
            f"{Path(calibration_file).stem}_calibration.json"
        )
        save_calibration(calibration_path, calibration_result)

    chosen_movies = choose_file()
    movie_filepaths = (
        [chosen_movies] if isinstance(chosen_movies, str) else list(chosen_movies)
    )

    results = batch_process(
        movie_filepaths,
        calibration=calibration_result.calibration,
        parallel=False,
        peak_model=PEAK_MODEL,
    )
