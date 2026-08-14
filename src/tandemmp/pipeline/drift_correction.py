"""Standalone drift-correction entry point.

Standalone from :mod:`tandemmp.pipeline.batch` so a movie's drift can be
estimated (:func:`run_drift_correction`) and persisted
(:func:`save_drift_vector` / :func:`save_correction_function`) without
running a full tracking/pair-matching analysis. The movie is loaded through
:class:`~tandemmp.pipeline.loader.Loader` with ``just_movie=True`` and the
loader's own drift correction disabled — this module drives the lower-level
:func:`tandemmp.drift.corrector.get_vector` directly instead, mirroring how
:func:`tandemmp.pipeline.simple_analysis.load_events` obtains a vector for
its CSV-only workflow.

Two on-disk representations are supported for the resulting
:class:`~tandemmp.drift.corrector.DriftVector`:

- :func:`save_drift_vector` / :func:`load_drift_vector` — plain JSON
  (``source_filepath``, ``frame_start``/``frame_end``, and per-frame
  ``x``/``y`` arrays), human-readable and hand-editable, mirroring
  :mod:`tandemmp.pipeline.calibration`'s save format.
- :func:`save_correction_function` / :func:`load_correction_function` —
  pickles a :class:`DriftCorrectionFunction`, a small wrapper exposing the
  same vector as ``correct_coordinates(x, y, i) -> (x_corrected,
  y_corrected)`` and ``correct_movie(movie, i) -> corrected_movie``, for
  callers that want to correct arbitrary positions or a whole movie
  directly (linearly interpolating between sampled frames) rather than
  re-deriving the subtraction themselves.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tandemmp.core.parameters import FullAnalysisParams
from tandemmp.core.types import FloatArray, IntArray
from tandemmp.drift.corrector import DriftVector, get_corrected_movie, get_vector
from tandemmp.pipeline.loader import Loader

# ---------------------------------------------------------------------------
# Correction function
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DriftCorrectionFunction:
    """A :class:`DriftVector` wrapped up as a callable position/movie corrector.

    ``correct_coordinates(x, y, i)`` linearly interpolates the vector at
    frame *i* (via :func:`numpy.interp`, which clamps rather than
    extrapolates outside the vector's frame range) and subtracts it from
    ``(x, y)`` — the same convention used elsewhere in the pipeline, e.g.
    :func:`tandemmp.pipeline.simple_analysis.load_events`. *x*, *y* and *i*
    may be scalars or equal-shaped arrays.

    ``correct_movie(movie, i)`` applies that same per-frame shift to every
    frame of a raw movie array instead of a set of point coordinates.

    Plain dataclass, so instances pickle without issue (see
    :func:`save_correction_function`).
    """

    vector: DriftVector

    def correct_coordinates(
        self,
        x: float | npt.NDArray[np.floating[Any]],
        y: float | npt.NDArray[np.floating[Any]],
        i: float | npt.NDArray[np.floating[Any]],
    ) -> tuple[Any, Any]:
        x_shape, y_shape, i_shape = np.shape(x), np.shape(y), np.shape(i)
        non_scalar_shapes = {s for s in (x_shape, y_shape, i_shape) if s != ()}
        if len(non_scalar_shapes) > 1:
            msg = (
                "x, y and i must be scalars or arrays of the same shape "
                f"(got shapes {x_shape}, {y_shape}, {i_shape})"
            )
            raise ValueError(msg)
        frames = np.arange(len(self.vector), dtype=float)
        dx = np.interp(i, frames, self.vector.x)
        dy = np.interp(i, frames, self.vector.y)
        return np.asarray(x) - dx, np.asarray(y) - dy

    def correct_movie(
        self,
        movie: FloatArray | IntArray,
        i: FloatArray | IntArray | None = None,
        *,
        parallel: bool = True,
        n_jobs: int | None = None,
    ) -> FloatArray:
        """Apply this drift correction to every frame of *movie*.

        *i* gives, for each frame of *movie*, the frame index into this
        object's vector; it defaults to ``range(len(movie))``, i.e. *movie*
        is assumed frame-aligned with the vector it was estimated from. Pass
        the original frame indices here to correct a *movie* that was
        loaded with a different ``frame_start``/``frame_end`` slice than the
        one the vector was estimated from — non-integer/out-of-range values
        are linearly interpolated/clamped exactly as in
        :meth:`correct_coordinates`.

        The actual per-frame shift, and cropping to the region that stays in
        view across every frame, is delegated to
        :func:`tandemmp.drift.corrector.get_corrected_movie`, called with a
        vector resampled onto *i*.
        """
        movie_arr: FloatArray = np.asarray(movie, dtype=float)
        frame_indices: FloatArray
        if i is None:
            if len(movie_arr) != len(self.vector):
                msg = (
                    f"movie has {len(movie_arr)} frames but this vector has "
                    f"{len(self.vector)}; pass i explicitly to correct a movie "
                    "that is a different slice than the one the vector was "
                    "estimated from"
                )
                raise ValueError(msg)
            frame_indices = np.arange(len(movie_arr), dtype=float)
        else:
            frame_indices = np.asarray(i, dtype=float)
            if len(frame_indices) != len(movie_arr):
                msg = (
                    f"i has length {len(frame_indices)}, "
                    f"movie has length {len(movie_arr)}"
                )
                raise ValueError(msg)

        frames = np.arange(len(self.vector), dtype=float)
        resampled_vector = DriftVector(
            np.interp(frame_indices, frames, self.vector.x),
            np.interp(frame_indices, frames, self.vector.y),
        )

        corrected_movie, _ = get_corrected_movie(  # type: ignore[misc]
            movie_arr, vector=resampled_vector, parallel=parallel, n_jobs=n_jobs
        )
        return corrected_movie


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DriftCorrectionResult:
    """A movie's estimated :class:`DriftVector` plus its source filepath.

    ``frame_start``/``frame_end`` record the slice of the *source* movie the
    vector was estimated over (i.e. the ``frame_start``/``frame_end`` passed
    to :func:`run_drift_correction`, ``None`` meaning "from the start" /
    "to the end"). The vector's own index ``0`` always corresponds to
    ``frame_start`` (or the source movie's first frame if ``None``), *not*
    necessarily raw frame ``0`` of the file on disk — persisting these two
    values lets a later caller recover that offset instead of having to
    remember it out-of-band.
    """

    vector: DriftVector
    source_filepath: str
    frame_start: int | None = None
    frame_end: int | None = None

    def as_function(self) -> DriftCorrectionFunction:
        """Wrap this result's vector as a :class:`DriftCorrectionFunction`."""
        return DriftCorrectionFunction(vector=self.vector)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_drift_correction(
    filepath: str | Path,
    *,
    frame_gap: int = 25,
    rolling_baseline: bool = False,
    threshold: float = 0.0,
    threshold_margin: int = 0,
    parallel: bool = True,
    n_jobs: int | None = None,
    seed: int | np.random.Generator | np.random.RandomState | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> DriftCorrectionResult:
    """Load a movie and estimate its :class:`~tandemmp.drift.corrector.DriftVector`.

    The movie is loaded via :class:`~tandemmp.pipeline.loader.Loader` with
    ``just_movie=True`` and the loader's own drift correction disabled — this
    function drives :func:`~tandemmp.drift.corrector.get_vector` itself so
    the raw vector is returned for inspection/persistence rather than being
    silently applied and discarded inside the loader. See
    :func:`~tandemmp.drift.corrector.get_vector` for the meaning of
    *frame_gap*, *rolling_baseline*, *threshold*, *threshold_margin*,
    *parallel*, *n_jobs* and *seed*.
    """
    movie = Loader(
        filepath,
        just_movie=True,
        parameters=FullAnalysisParams(drift_correction=False, parallel=parallel),
        frame_start=frame_start,
        frame_end=frame_end,
    ).movie

    vector = get_vector(
        movie=movie,
        frame_gap=frame_gap,
        rolling_baseline=rolling_baseline,
        threshold=threshold,
        threshold_margin=threshold_margin,
        parallel=parallel,
        n_jobs=n_jobs,
        seed=seed,
    )

    return DriftCorrectionResult(
        vector=vector,
        source_filepath=str(filepath),
        frame_start=frame_start,
        frame_end=frame_end,
    )


# ---------------------------------------------------------------------------
# Persistence — plain JSON vector
# ---------------------------------------------------------------------------
#
# On disk, a drift vector is plain JSON so it can be opened and hand-edited
# in a text editor, mirroring tandemmp.pipeline.calibration's save format.


def _drift_vector_to_mapping(result: DriftCorrectionResult) -> dict[str, Any]:
    return {
        "source_filepath": result.source_filepath,
        "frame_start": result.frame_start,
        "frame_end": result.frame_end,
        "x": result.vector.x.tolist(),
        "y": result.vector.y.tolist(),
    }


def _drift_vector_from_mapping(
    payload: Mapping[str, Any], *, default_source_filepath: str
) -> DriftCorrectionResult:
    try:
        x = payload["x"]
        y = payload["y"]
    except KeyError as exc:
        msg = f"drift vector data is missing required field {exc}"
        raise ValueError(msg) from exc
    return DriftCorrectionResult(
        vector=DriftVector(np.asarray(x, dtype=float), np.asarray(y, dtype=float)),
        source_filepath=str(payload.get("source_filepath", default_source_filepath)),
        frame_start=payload.get("frame_start"),
        frame_end=payload.get("frame_end"),
    )


def save_drift_vector(filepath: str | Path, result: DriftCorrectionResult) -> Path:
    """Write *result* to *filepath* as human-editable JSON.

    Creates parent directories as needed. See the module docstring for the
    on-disk field names.
    """
    if not isinstance(result, DriftCorrectionResult):
        msg = f"result must be a DriftCorrectionResult, got {type(result).__name__}"
        raise TypeError(msg)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_drift_vector_to_mapping(result), fh, indent=2)
        fh.write("\n")
    return path


def load_drift_vector(filepath: str | Path) -> DriftCorrectionResult:
    """Load a :class:`DriftCorrectionResult` written by :func:`save_drift_vector`."""
    path = Path(filepath)
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        msg = f"{path} does not contain a drift vector"
        raise TypeError(msg)
    return _drift_vector_from_mapping(payload, default_source_filepath=str(path))


# ---------------------------------------------------------------------------
# Persistence — correction function
# ---------------------------------------------------------------------------


def save_correction_function(
    filepath: str | Path, result: DriftCorrectionResult
) -> Path:
    """Pickle *result* as a :class:`DriftCorrectionFunction`.

    Unlike :func:`save_drift_vector`, this is a binary format — load it back
    with :func:`load_correction_function` and call
    ``result.correct_coordinates(x, y, i)`` or ``result.correct_movie(movie)``
    directly.
    """
    if not isinstance(result, DriftCorrectionResult):
        msg = f"result must be a DriftCorrectionResult, got {type(result).__name__}"
        raise TypeError(msg)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result.as_function(), fh)
    return path


def load_correction_function(filepath: str | Path) -> DriftCorrectionFunction:
    """Load a :class:`DriftCorrectionFunction` written by :func:`save_correction_function`."""  # noqa: E501
    path = Path(filepath)
    with path.open("rb") as fh:
        obj = pickle.load(fh)  # noqa: S301 — trusted, locally-produced pickle
    if not isinstance(obj, DriftCorrectionFunction):
        msg = f"{path} does not contain a DriftCorrectionFunction"
        raise TypeError(msg)
    return obj


if __name__ == "__main__":
    from tandemmp.core.functions import choose_file

    movie_filepath = choose_file()
    if not isinstance(movie_filepath, str):
        raise TypeError("Select a single movie file.")

    drift_result = run_drift_correction(movie_filepath)

    vector_path = Path(movie_filepath).with_name(
        f"{Path(movie_filepath).stem}_drift_vector.json"
    )
    save_drift_vector(vector_path, drift_result)
    print(f"Drift vector saved to {vector_path}")

    function_path = Path(movie_filepath).with_name(
        f"{Path(movie_filepath).stem}_drift_correction.pkl"
    )
    save_correction_function(function_path, drift_result)
    print(f"Drift correction function saved to {function_path}")
