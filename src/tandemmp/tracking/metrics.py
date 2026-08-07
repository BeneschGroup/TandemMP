"""Trajectory metrics for tracked iSCAT particles.

Pure array-based helpers computing summary statistics over particle
trajectories (step sizes, mean-squared displacement, residency time,
etc.).  All functions operate on NumPy-compatible input sequences so
they can be reused without coupling to any particular particle type.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

from tandemmp.core.conversions import pixels2nm
from tandemmp.core.types import FloatArray, IntArray

__all__ = [
    "delta_distance",
    "delta_travel",
    "total_distance_moved",
    "total_travel",
    "residency_time",
    "num_jumps",
    "msd",
    "mean_squared_displacement",
]


def _as_float_array(values: Sequence[float] | npt.ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def delta_distance(
    xs: Sequence[float] | npt.ArrayLike,
    ys: Sequence[float] | npt.ArrayLike,
) -> FloatArray:
    """Return per-step displacements between consecutive points."""
    x = _as_float_array(xs)
    y = _as_float_array(ys)
    if x.shape != y.shape:
        raise ValueError("xs and ys must have matching shape")
    if x.size < 2:
        return np.zeros(0, dtype=np.float64)
    return cast(FloatArray, np.hypot(np.diff(x), np.diff(y)))


def delta_travel(
    xs: Sequence[float] | npt.ArrayLike,
    ys: Sequence[float] | npt.ArrayLike,
) -> FloatArray:
    """Return distances from the origin point at each subsequent step."""
    x = _as_float_array(xs)
    y = _as_float_array(ys)
    if x.shape != y.shape:
        raise ValueError("xs and ys must have matching shape")
    if x.size < 2:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(np.hypot(x[1:] - x[0], y[1:] - y[0]), dtype=np.float64)


def total_distance_moved(
    xs: Sequence[float] | npt.ArrayLike,
    ys: Sequence[float] | npt.ArrayLike,
    threshold: float = 0.0,
) -> float:
    """Return the sum of step distances greater than *threshold*."""
    steps = delta_distance(xs, ys)
    return float(np.sum(steps[steps > threshold]))


def total_travel(
    xs: Sequence[float] | npt.ArrayLike,
    ys: Sequence[float] | npt.ArrayLike,
) -> float:
    """Return the straight-line distance from first to last point."""
    x = _as_float_array(xs)
    y = _as_float_array(ys)
    if x.shape != y.shape:
        raise ValueError("xs and ys must have matching shape")
    if x.size < 2:
        return 0.0
    return float(np.hypot(x[-1] - x[0], y[-1] - y[0]))


def residency_time(frames: Sequence[int] | npt.ArrayLike) -> int:
    """Return the number of frames spanned by *frames* (``last - first``)."""
    f = np.asarray(frames, dtype=np.int64)
    if f.size == 0:
        return 1  # sentinel: avoids division-by-zero in callers
    return int(f[-1] - f[0])


def num_jumps(
    xs: Sequence[float] | npt.ArrayLike,
    ys: Sequence[float] | npt.ArrayLike,
    threshold: float,
) -> int:
    """Return the number of step displacements exceeding *threshold*."""
    return int(np.sum(delta_distance(xs, ys) > threshold))


def msd(
    xs: Sequence[float] | npt.ArrayLike,
    ys: Sequence[float] | npt.ArrayLike,
) -> tuple[IntArray, FloatArray]:
    """Return ``(lags, msd_values)`` computed over integer frame lags.

    The lag ``i`` averages ``(r(s+i) - r(s))**2`` across all valid
    starting points ``s``.  Lag 0 is always 0.
    """
    x = _as_float_array(xs)
    y = _as_float_array(ys)
    if x.shape != y.shape:
        raise ValueError("xs and ys must have matching shape")
    n = x.size
    lags = np.arange(n, dtype=np.int64)
    values = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return lags, values
    values[0] = 0.0
    for i in range(1, n):
        dx = x[i:] - x[: n - i]
        dy = y[i:] - y[: n - i]
        if dx.size:
            values[i] = float(np.mean(dx * dx + dy * dy))
    return lags, values


def mean_squared_displacement(
    xs: Sequence[float] | npt.ArrayLike,
    ys: Sequence[float] | npt.ArrayLike,
    frames: Sequence[int] | npt.ArrayLike,
    fps: float = 131.0,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(times, msd_values)`` in seconds and nm².

    Positions are converted from pixels to nm via
    :func:`tandemmp.core.conversions.pixels2nm`; *frames* are divided
    by *fps* to yield seconds.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    x: FloatArray = np.asarray(pixels2nm(_as_float_array(xs)), dtype=np.float64)
    y: FloatArray = np.asarray(pixels2nm(_as_float_array(ys)), dtype=np.float64)
    f = np.asarray(frames, dtype=np.float64)
    if x.shape != y.shape or x.shape != f.shape:
        raise ValueError("xs, ys and frames must have matching shape")
    n = x.size
    if n < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    t = f / fps
    times = np.empty(n - 1, dtype=np.float64)
    values = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        dx = x[i:] - x[: n - i]
        dy = y[i:] - y[: n - i]
        times[i - 1] = float(t[i] - t[0])
        values[i - 1] = float(np.mean(dx * dx + dy * dy))
    return times, values
