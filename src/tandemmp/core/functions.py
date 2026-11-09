"""General-purpose utility functions for TandemMP."""

from __future__ import annotations

import tkinter as tk
from statistics import NormalDist
from tkinter import filedialog
from typing import Any

import numpy as np
import numpy.typing as npt

# Array normalisation


def normalise(arr: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Min-max normalise *arr* to [0, 1]."""
    return np.asarray((arr - np.min(arr)) / (np.max(arr) - np.min(arr)))


def nan_normalise(arr: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Min-max normalise *arr* to [0, 1], ignoring NaNs."""
    return np.asarray((arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr)))


# Dictionary lookup


def closest_index(dictionary: dict[int, Any], frame: int) -> Any | None:  # noqa: ANN401
    """Return the *value* for the largest key ≤ *frame*.

    Returns ``None`` if no key is ≤ *frame*.
    """
    keys = [k for k in dictionary.keys() if k <= frame]
    if keys:
        return dictionary[max(keys)]
    return None


# Spatial probability


def get_overlap_probability(
    x1: float,
    y1: float,
    s1: float,
    x2: float,
    y2: float,
    s2: float,
) -> float:
    """Overlap coefficient of two 1-D Gaussians projected onto the line
    connecting *(x1,y1)* and *(x2,y2)*.

    Uses ``statistics.NormalDist.overlap``. Returns a value in [0, 1].
    """
    distance = float(np.hypot(x1 - x2, y1 - y2))
    return NormalDist(0, s1).overlap(NormalDist(distance, s2))


# 2-D Gaussian


def gaussian_2d(
    xy: npt.NDArray[np.floating],
    sigma: float,
) -> npt.NDArray[np.floating]:
    """Evaluate an isotropic 2-D Gaussian centred at the origin.

    *xy* has shape ``(2, N)`` where ``xy[0]`` is x and ``xy[1]`` is y.
    """
    x, y = xy[0], xy[1]
    return np.asarray(
        (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    )


# File browser


def choose_file(
    mp_file: bool = True,
) -> str | tuple[str, ...]:
    """Open a tkinter file dialog and return the selected path(s).

    Returns a single path string if one file is selected, or a tuple of
    paths if multiple files are selected.
    """
    root = tk.Tk()

    if mp_file:
        filetypes = [("MP", "*.mp"), ("All files", "*")]
    else:
        filetypes = [("All files", "*")]

    results = filedialog.askopenfilenames(
        parent=root, initialdir="/", initialfile="tmp", filetypes=filetypes
    )

    root.destroy()

    if len(results) == 1:
        return results[0]
    return results
