"""Pure movie processing utilities for iSCAT data.

Provides background-removal algorithms (ratiometric, rolling mean, start
subtraction) and temporal binning helpers that operate on raw 3-D movie arrays
(frames × height × width).

All functions are stateless; no matplotlib or GUI code is present in this
module.  Animation and visualisation helpers live in
:mod:`tandemmp.plotting.animation`.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from tqdm import tqdm


def _maybe_tqdm(iterable: Any, *, silent: bool, desc: str) -> Any:  # noqa: ANN401
    """Wrap *iterable* in ``tqdm`` unless *silent* is ``True``."""
    if silent:
        return iterable
    return tqdm(iterable, desc=desc)


def ratiometric(
    movie: npt.NDArray[np.floating[Any]],
    navg: int = 5,
    *,
    silent: bool = False,
) -> npt.NDArray[np.floating[Any]]:
    """Ratiometric background-removal filter.

    For each frame *i* in ``[navg, len(movie) - navg)`` computes::

        (sum of frames i+1 .. i+navg) / (sum of frames i-navg .. i-1) - 1

    Edge frames (first and last *navg*) are set to NaN.

    Parameters
    ----------
    movie:
        Raw 3-D movie array (frames × height × width).
    navg:
        Number of frames to average in each direction.
    silent:
        If ``True``, suppress the ``tqdm`` progress bar.
    """
    new_movie = np.full_like(movie, np.nan, dtype=float)
    for index in _maybe_tqdm(
        np.arange(navg, len(movie) - navg),
        silent=silent,
        desc="Calculating ratiometric",
    ):
        new_movie[index] = (
            movie[index + 1 : index + navg + 1].sum(axis=0)
            / movie[index - navg : index].sum(axis=0)
        ) - 1
    return new_movie


def rolling_mean(
    movie: npt.NDArray[np.floating[Any]],
    navg: int = 50,
    *,
    silent: bool = False,
) -> npt.NDArray[np.floating[Any]]:
    """Rolling-mean background-removal filter.

    Uses a symmetric window of size ``2*navg + 1`` centred on each frame.
    Edge frames (indices ``0..navg-1`` and ``len-navg..end``) are NaN.

    Parameters
    ----------
    movie:
        Raw 3-D movie array (frames × height × width).
    navg:
        Half-window size; full window is ``2*navg + 1`` frames.
    silent:
        If ``True``, suppress the ``tqdm`` progress bar.
    """
    new_movie = np.full_like(movie, np.nan, dtype=float)
    for i, frame in _maybe_tqdm(
        enumerate(movie[navg:-navg]),
        silent=silent,
        desc="Calculating rolling mean",
    ):
        index = i + navg
        new_movie[index] = (
            frame / movie[index - navg : index + navg + 1].mean(axis=0)
        ) - 1
    return new_movie


def fast_rolling_mean(
    movie: npt.NDArray[np.floating[Any]],
    navg: int,
) -> npt.NDArray[np.floating[Any]]:
    """Sliding-window mean background subtraction using incremental updates.

    Faster than :func:`rolling_mean` for large *navg* because the running
    average is updated frame-by-frame rather than recomputed from scratch.

    Edge frames in the first and last *navg* positions are NaN.

    Parameters
    ----------
    movie:
        Raw 3-D movie array (frames × height × width).
    navg:
        Half-window size; full window is ``2*navg + 1`` frames.
    """
    window = navg * 2
    denom = window + 1
    # avg[j] is the mean of movie[j : j + 2*navg + 1] — the same window
    # centred on frame (j + navg) that :func:`rolling_mean` uses.
    avg = np.full(
        (movie.shape[0] - window, movie.shape[1], movie.shape[2]),
        np.nan,
        dtype=float,
    )
    avg[0] = movie[:denom].mean(axis=0)
    for j in range(1, len(avg)):
        avg[j] = avg[j - 1] + (movie[j + window] - movie[j - 1]) / denom
    new_movie = np.full_like(movie, np.nan, dtype=float)
    for i in range(navg, len(movie) - navg):
        new_movie[i] = (movie[i] / avg[i - navg]) - 1
    return new_movie


def fast_ratiometric(
    movie: npt.NDArray[np.floating[Any]],
    navg: int,
) -> npt.NDArray[np.floating[Any]]:
    """Faster ratiometric filter using pre-computed rolling sums.

    Stores partial frame sums in a single pass, then computes ratios in a
    second pass.  More cache-efficient than :func:`ratiometric` on large
    movies.

    Parameters
    ----------
    movie:
        Raw 3-D movie array (frames × height × width).
    navg:
        Number of frames to sum in each direction.
    """
    n_sum = np.full(
        (movie.shape[0] + 1 - navg, movie.shape[1], movie.shape[2]),
        np.nan,
        dtype=float,
    )
    for i in range(navg, len(movie) + 1):
        n_sum[i - navg] = movie[i - navg : i].sum(axis=0)

    new_movie = np.full_like(movie, np.nan, dtype=float)
    for i in range(navg, len(movie) - navg):
        new_movie[i] = (n_sum[i + 1] / n_sum[i - navg]) - 1
    return new_movie


def start_subtraction(
    movie: npt.NDArray[np.floating[Any]],
    navg: int = 0,
    start_frame: int = 0,
    *,
    silent: bool = False,
) -> npt.NDArray[np.floating[Any]]:
    """Background removal by normalising against the first few frames.

    Each frame is divided by the mean of
    ``movie[start_frame : start_frame + navg + 1]`` and shifted by −1.
    Frames before index *navg* are NaN.

    Parameters
    ----------
    movie:
        Raw 3-D movie array (frames × height × width).
    navg:
        Number of frames (from *start_frame*) to average as the reference.
    start_frame:
        First frame index of the reference window.
    silent:
        If ``True``, suppress the ``tqdm`` progress bar.
    """
    new_movie = np.full_like(movie, np.nan, dtype=float)
    ref = movie[start_frame : start_frame + navg + 1].mean(axis=0)
    for index, frame in enumerate(
        _maybe_tqdm(movie[navg:], silent=silent, desc="Calculating start subtraction")
    ):
        new_movie[index + navg] = frame / ref - 1
    return new_movie


def start_subtraction_fast(
    movie: npt.NDArray[np.floating[Any]],
    reference: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Vectorised start subtraction using a pre-computed reference frame.

    Parameters
    ----------
    movie:
        Raw 3-D movie array (frames × height × width).
    reference:
        Reference array (the image to divide by).  If 3-D (a stack of
        reference frames), the mean along ``axis=0`` is used.  Note this
        differs from :func:`start_subtraction`, where ``start_frame`` is an
        integer index into *movie*; here the reference is the actual array.
    """
    if reference.ndim == 3:
        reference = np.mean(reference, axis=0)
    return cast("npt.NDArray[np.floating[Any]]", (movie / reference) - 1)


def pre_bin(
    movie: npt.NDArray[np.floating[Any]],
    n: int,
) -> npt.NDArray[np.floating[Any]]:
    """Temporally bin *movie* by averaging every *n* consecutive frames.

    Partial bins at the end are dropped.  The output has
    ``len(np.arange(0, len(movie), n)) - 1`` frames.

    Parameters
    ----------
    movie:
        Raw 3-D movie array (frames × height × width).
    n:
        Number of frames per bin.
    """
    indexes = np.arange(0, len(movie), n)
    return np.array(
        [
            movie[indexes[i] : indexes[i + 1]].mean(axis=0)
            for i in range(len(indexes) - 1)
        ]
    )


def final_polish_fix(
    movie: npt.NDArray[np.floating[Any]],
    navg: int = 5,
    *,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Prepend duplicated frames so ratiometric can begin at frame 0.

    Repeats ``movie[0]`` a total of ``navg * 2`` times (with tiny random
    noise) and prepends them so the ratiometric filter has a valid backward
    window from the very start of the original movie.

    Parameters
    ----------
    movie:
        Raw 3-D movie array (frames × height × width).
    navg:
        Half-window size used by the subsequent ratiometric call; ``navg * 2``
        frames are prepended.
    rng:
        Optional NumPy random generator.  Pass a seeded generator
        (``np.random.default_rng(seed)``) for reproducible output; defaults to
        an unseeded generator.
    """
    first_frame_stack = np.array([movie[0]] * (navg * 2))
    if rng is None:
        rng = np.random.default_rng()
    added_noise = 1 - (rng.random(size=first_frame_stack.shape) + 1.5) * 0.0003
    return np.append(first_frame_stack * added_noise, movie, axis=0)


def duplicate_and_change_frames(
    frames: npt.NDArray[Any],
    filepath: str | Path,
    distinguisher: str = "edited",
) -> Path:
    """Copy an .mp HDF5 file and overwrite its frame dataset.

    Parameters
    ----------
    frames:
        New frame data to write into the copied file.
    filepath:
        Path to the source .mp file.
    distinguisher:
        Suffix appended (before the file extension) in the output filename.

    Returns
    -------
    Path
        Path to the newly created file.
    """
    import h5py

    filepath = Path(filepath)
    new_filepath = filepath.with_stem(filepath.stem + f"_{distinguisher}")
    copyfile(filepath, new_filepath)
    with h5py.File(new_filepath, "r+") as f:
        if "movie" in f.keys():
            del f["movie"]["frame"]
            f["movie"].create_dataset("frame", data=frames)
        elif "frame" in f.keys():
            del f["frame"]
            f.create_dataset("frame", data=frames)
        else:
            raise KeyError(
                f"{filepath} does not contain a 'movie/frame' group or a "
                "top-level 'frame' dataset; cannot replace frames."
            )
    return new_filepath
