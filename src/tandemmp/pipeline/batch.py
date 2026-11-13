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

    Movies with *max_events* or more first-pass events are skipped: that many usually
    means a bad movie (dust, drift, a mis-set threshold), not genuine data.

    Results are pickled to *save_filepath*, default *filepath* with ``_tandemmp``
    appended before its (``.pkl``) suffix.

    *pipeline_runner* selects the event-finding backend (see
    :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`); default
    open_iscat. *calibration* overrides the process-wide contrast/mass conversion (see
    :mod:`tandemmp.core.conversions`) for this call; it is always set explicitly
    (``None`` resets to the default), so repeated calls on a reused worker process
    (see :func:`batch_process`) are safe. *peak_model* is the PSF model; default the
    bundled Rust spline.
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

    Each movie can need tens of GB of RAM, so *n_jobs* defaults to
    ``min(6, len(filepaths))`` to bound peak memory.

    The worker pool (:func:`tandemmp.core.parallel.dispatch_kwargs`) persists for the
    process lifetime, so per-worker state (e.g. the cached TensorFlow ghost model) is
    initialised at most once per worker and reused across :func:`batch_process`
    calls, at the cost of that RAM staying resident. A different *n_jobs* recreates
    the pool.

    Do not combine ``batch_process(parallel=True)`` with ``parallel=True`` analysis
    parameters: pool workers are daemonic and cannot spawn child pools
    (``AssertionError: daemonic processes are not allowed to have children``). This
    is a :class:`multiprocessing.Pool` constraint.

    *pipeline_runner* selects the event-finding backend (see
    :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`; default
    open_iscat). With *parallel* it must be picklable, as it crosses the pool boundary
    with each :func:`process_movie` call; the default ``None`` and
    :class:`~tandemmp.pipeline.open_iscat_pipeline.OpenIscatPipeline` are.
    *calibration* overrides the contrast/mass conversion (see
    :mod:`tandemmp.core.conversions`) for every movie in the batch, applied per call
    so it survives worker reuse. *peak_model* is forwarded to each call and must be a
    picklable top-level class when *parallel* is ``True``.
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

    # PSF model for this run: any PsfInterface, e.g. ``GaussianPsfModel(sigma=1.5)``;
    # ``None`` = bundled Rust spline (``SplinePsfModel``).
    PEAK_MODEL: PsfInterface | None = None

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
