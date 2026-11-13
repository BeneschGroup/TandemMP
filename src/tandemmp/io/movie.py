"""Stateless movie-processing utilities for iSCAT data.

Background-removal filters (ratiometric, rolling mean, start subtraction) and
binning helpers. Every ``movie`` is a raw array of shape (frames, height, width).
Visualisation lives in :mod:`tandemmp.plotting.animation`.
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

    The first and last *navg* frames are NaN.

    Args:
        navg: Frames averaged in each direction.
        silent: Suppress the progress bar.
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

    Uses a symmetric window of ``2*navg + 1`` frames centred on each frame; the first
    and last *navg* frames are NaN.

    Args:
        navg: Half-window size.
        silent: Suppress the progress bar.
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
    """Sliding-window mean subtraction using incremental updates.

    Faster than :func:`rolling_mean` for large *navg*: the running average is
    updated per frame, not recomputed. The first and last *navg* frames are NaN.

    Args:
        navg: Half-window size.
    """
    window = navg * 2
    denom = window + 1
    # avg[j] = mean of movie[j : j + 2*navg + 1], the window centred on frame
    # j + navg that :func:`rolling_mean` uses.
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
    """Ratiometric filter computed from pre-computed rolling sums.

    Stores partial frame sums in one pass, then computes ratios in a second; more
    cache-efficient than :func:`ratiometric` on large movies.

    Args:
        navg: Frames summed in each direction.
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

    Each frame is divided by the mean of ``movie[start_frame : start_frame + navg + 1]``
    and shifted by −1. Frames before index *navg* are NaN.

    Args:
        navg: Frames (from *start_frame*) averaged as the reference.
        start_frame: First frame of the reference window.
        silent: Suppress the progress bar.
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
    """Vectorised start subtraction against a pre-computed reference.

    Args:
        reference: Image to divide by; a 3-D stack is averaged over ``axis=0``. Unlike
            the integer ``start_frame`` of :func:`start_subtraction`, this is the
            array itself.
    """
    if reference.ndim == 3:
        reference = np.mean(reference, axis=0)
    return cast("npt.NDArray[np.floating[Any]]", (movie / reference) - 1)


def pre_bin(
    movie: npt.NDArray[np.floating[Any]],
    n: int,
) -> npt.NDArray[np.floating[Any]]:
    """Temporally bin *movie* by averaging every *n* consecutive frames.

    A partial final bin is dropped; the output has
    ``len(np.arange(0, len(movie), n)) - 1`` frames.
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
    """Prepend duplicated frames so ratiometric can start at frame 0.

    Repeats ``movie[0]`` ``navg * 2`` times (plus tiny random noise), giving the
    ratiometric filter a valid backward window.

    Args:
        navg: Half-window of the following ratiometric call.
        rng: Seeded generator (``np.random.default_rng(seed)``) for reproducible
            output; default is unseeded.
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

    Args:
        frames: New frame data.
        filepath: Source .mp file.
        distinguisher: Suffix added before the extension of the output name.

    Returns:
        Path of the new file.
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
