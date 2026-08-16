"""Drift correction for iSCAT movies.

Provides :class:`DriftVector` (a 2-D drift trajectory) and the functions used
to estimate it from a movie and apply it as a per-frame stabilising shift.

The original ``DriftCorrector.py`` mixed I/O, plotting, and a ``tkinter`` file
dialog into one module.  This refactor strips all of that out:

- File I/O (``MPFileReader``, ``h5py`` writes) and the ``run_drift_correction``
  / ``duplicate_and_change_frames`` helpers are gone — drift correction is now
  a pure transformation from movie to movie + vector.  Callers handle I/O.
- Plotting (``rmse_surface_plot`` and the ``show_vector`` matplotlib block in
  ``get_vector``) is dropped here and was not recovered by item 6k, which
  only relocated plotting code already present in ``src/`` at the time it
  ran (see CLAUDE.md) — reintroducing this remains a separate future task.
- Parallelism uses :func:`tandemmp.core.parallel.dispatch_kwargs`, a shared
  local :mod:`multiprocessing` dispatcher also used by
  :mod:`tandemmp.pipeline.investigator`, instead of
  ``open_iscat.utils.parallel``, so the module no longer depends on
  ``open_iscat`` at all.

Rust acceleration candidate — see refactor item 6 (drift correction shift +
RMSE inner loop is currently driven by :func:`scipy.optimize.differential_evolution`).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from typing import Any, overload

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, shift
from scipy.optimize import differential_evolution

from tandemmp.core.parallel import dispatch_kwargs
from tandemmp.core.types import FloatArray, IntArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drift vector container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DriftVector:
    """A 2-D drift trajectory: per-frame ``(x, y)`` shifts in pixels.

    Replaces the old ``Vector`` class.  ``x`` and ``y`` must be 1-D arrays of
    equal length.
    """

    x: FloatArray
    y: FloatArray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("DriftVector components must be 1-D arrays")
        if x.shape != y.shape:
            raise ValueError(f"DriftVector x/y shape mismatch: {x.shape} vs {y.shape}")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return ((float(xi), float(yi)) for xi, yi in zip(self.x, self.y, strict=True))

    @overload
    def __getitem__(self, key: int) -> tuple[np.floating[Any], np.floating[Any]]: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[FloatArray, FloatArray]: ...

    def __getitem__(self, key: int | slice) -> tuple[Any, Any]:
        return (self.x[key], self.y[key])

    def __neg__(self) -> DriftVector:
        return DriftVector(-self.x, -self.y)


# ---------------------------------------------------------------------------
# Low-level shift / compare helpers
# ---------------------------------------------------------------------------


def _rmse(a: FloatArray, b: FloatArray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _most_positive(arr: FloatArray) -> float:
    pos = arr[arr >= 0]
    return float(np.max(pos)) if pos.size else 0.0


def _most_negative(arr: FloatArray) -> float:
    neg = arr[arr < 0]
    return float(np.min(neg)) if neg.size else 0.0


def _always_visible_1d(arr: FloatArray) -> slice:
    return slice(-int(_most_negative(arr)) + 1, -int(_most_positive(arr)) - 1)


def _always_visible_region(vector: DriftVector) -> tuple[slice, slice]:
    return _always_visible_1d(vector.y), _always_visible_1d(vector.x)


def _float_to_slice(n: float) -> slice:
    if n >= 0:
        return slice(int(n) + 1, None)
    return slice(None, int(n) - 1)


def stabilise(
    frame: FloatArray, x: float, y: float, *, use_rust: bool = True
) -> FloatArray:
    """Shift *frame* by ``(-y, -x)`` to undo a drift of ``(x, y)``.

    ``use_rust=True`` (the default) prefers the compiled
    ``tandemmp._rust.drift_correction`` extension (see WORKLIST.md priority 6)
    when it has been built; otherwise (or if it isn't available) falls back
    transparently to the :func:`scipy.ndimage.shift` implementation below.
    """
    if use_rust:
        try:
            return _stabilise_rust(frame, x, y)
        except ImportError:
            logger.debug(
                "Rust drift-correction module unavailable, falling back to Python"
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Rust stabilise failed, falling back to Python", exc_info=True
            )
    return shift(frame, (-y, -x))  # type: ignore[no-any-return]


def _stabilise_rust(frame: FloatArray, x: float, y: float) -> FloatArray:
    """Rust-backed :func:`stabilise`, via ``tandemmp._rust.drift_correction``."""
    from tandemmp._rust import drift_correction as _dc

    out = _dc.stabilise(np.asarray(frame, dtype=np.float64), float(x), float(y))
    return np.asarray(out, dtype=float)


def _shift_compare(
    xy: tuple[float, float],
    input_frame: FloatArray,
    baseline_frame: FloatArray,
) -> float:
    """RMSE between *input_frame* and *baseline_frame* shifted by ``xy``.

    Both frames are sliced to discard edge artefacts introduced by the shift.
    """
    x, y = xy
    x_slice, y_slice = _float_to_slice(x), _float_to_slice(y)

    shifted_baseline = shift(baseline_frame, (y, x))[y_slice, x_slice]
    cropped_input = input_frame[y_slice, x_slice]

    return _rmse(shifted_baseline, cropped_input)


def _solve(
    input_frame: FloatArray,
    baseline_frame: FloatArray,
    bounds: tuple[float, float] = (-6.0, 6.0),
    seed: int | np.random.Generator | np.random.RandomState | None = None,
    *,
    use_rust: bool = True,
) -> FloatArray:
    """Find the ``(x, y)`` shift minimising RMSE between two frames.

    ``seed`` is forwarded to :func:`scipy.optimize.differential_evolution`.
    Left as ``None`` (the default) the search is genuinely stochastic, which
    is what real usage wants; pass an explicit seed for reproducible runs
    (e.g. in tests).

    ``use_rust=True`` (the default) prefers the fused Rust ``solve`` entry
    point (a single differential-evolution + shift + RMSE call, see
    WORKLIST.md priority 6) when the compiled
    ``tandemmp._rust.drift_correction`` extension is available; otherwise
    falls back transparently to :func:`scipy.optimize.differential_evolution`
    below. Only plain ``int`` (or ``None``) seeds are supported by the Rust
    path — an :class:`~numpy.random.Generator`/:class:`~numpy.random.RandomState`
    ``seed`` also falls back to Python.
    """
    if use_rust:
        try:
            return _solve_rust(input_frame, baseline_frame, bounds, seed)
        except ImportError:
            logger.debug(
                "Rust drift-correction module unavailable, falling back to Python"
            )
        except Exception:  # noqa: BLE001
            logger.warning("Rust _solve failed, falling back to Python", exc_info=True)

    res = differential_evolution(
        _shift_compare,
        bounds=[bounds, bounds],
        args=(input_frame, baseline_frame),
        polish=True,
        disp=False,
        seed=seed,
    )
    return np.asarray(res.x, dtype=float)


def _solve_rust(
    input_frame: FloatArray,
    baseline_frame: FloatArray,
    bounds: tuple[float, float],
    seed: int | np.random.Generator | np.random.RandomState | None,
) -> FloatArray:
    """Rust-backed :func:`_solve`, via ``tandemmp._rust.drift_correction``.

    Raises :class:`ImportError` if the extension hasn't been built, or
    :class:`TypeError` if *seed* isn't a plain ``int``/``None`` — both are
    caught by :func:`_solve`'s fallback.
    """
    from tandemmp._rust import drift_correction as _dc

    rust_seed = None if seed is None else int(seed)  # type: ignore[arg-type]
    x, y = _dc.solve(
        np.asarray(input_frame, dtype=np.float64),
        np.asarray(baseline_frame, dtype=np.float64),
        bounds,
        rust_seed,
    )
    return np.array([x, y], dtype=float)


# ---------------------------------------------------------------------------
# Vector estimation
# ---------------------------------------------------------------------------


def _get_initial_vector(
    movie: FloatArray,
    indexes: IntArray,
    rolling_baseline: bool,
    parallel: bool,
    n_jobs: int | None,
    seed: int | np.random.Generator | np.random.RandomState | None = None,
    *,
    use_rust: bool = True,
) -> DriftVector:
    """Estimate per-index ``(x, y)`` shifts for *movie* against a baseline.

    If ``rolling_baseline`` is *True*, the baseline for each frame is the
    previous sampled frame and per-step shifts are accumulated into the
    returned trajectory.

    ``seed`` is forwarded to :func:`_solve` for each frame — see its
    docstring for why it defaults to ``None``. ``use_rust`` is likewise
    forwarded to each :func:`_solve` call — see its docstring.
    """
    frames = np.asarray([movie[i] for i in indexes], dtype=float)

    baselines: list[FloatArray] = [frames[0]] * len(indexes)
    if rolling_baseline:
        baselines[2:] = [np.asarray(movie[i], dtype=float) for i in indexes[1:-1]]

    args = [
        {"input_frame": f, "baseline_frame": b, "seed": seed, "use_rust": use_rust}
        for f, b in zip(frames, baselines, strict=True)
    ]

    bests = dispatch_kwargs(_solve, args, parallel=parallel, n_jobs=n_jobs)

    x, y = np.transpose(np.asarray(bests, dtype=float))

    if rolling_baseline:
        x, y = np.cumsum(x), np.cumsum(y)

    return DriftVector(x, y)


def get_vector(
    movie: FloatArray,
    frame_gap: int,
    rolling_baseline: bool = False,
    threshold: float = 0.0,
    threshold_margin: int = 0,
    *,
    parallel: bool = True,
    n_jobs: int | None = None,
    seed: int | np.random.Generator | np.random.RandomState | None = None,
    use_rust: bool = True,
) -> DriftVector:
    """Estimate a drift vector for an entire movie.

    Args:
        movie: A 3-D array of shape ``(frames, y, x)``.
        frame_gap: Sample every *frame_gap*-th frame when building the initial vector.
        rolling_baseline: If *True*, use the previous sampled frame as the baseline
            (use this when z-drift is suspected).
        threshold: Per-frame intensity sum below which a frame is considered "lid
            open" and skipped from the initial sample. Set to ``0`` to disable.
        threshold_margin: Number of extra frames around each lid-open region to
            re-check at full sampling density.
        parallel: Run the per-frame minimisation across processes.
        n_jobs: Worker count for parallel mode. ``None`` lets ``multiprocessing``
            pick a default (typically ``os.cpu_count()``).
        seed: Forwarded to the underlying :func:`scipy.optimize.differential_evolution`
            call for each frame. Left as ``None`` (the default), the search is
            genuinely stochastic — appropriate for real usage. Pass an explicit
            seed for reproducible results, e.g. in tests.
        use_rust: Forwarded to :func:`_solve` for each frame — prefer the compiled
            Rust extension when available, falling back to Python transparently
            otherwise. Defaults to ``True``.
    """
    all_frames_to_check: IntArray = np.arange(0, len(movie), frame_gap, dtype=int)

    below_threshold = np.sum(movie[all_frames_to_check], axis=(1, 2)) < threshold
    if threshold > 0 and below_threshold.any():
        frames_to_check = all_frames_to_check[~below_threshold]
        if frames_to_check.size == 0:
            raise ValueError(
                "All sampled frames are below threshold — check the threshold value"
            )
        logger.info(
            "Lid open for approximately %d frames",
            int(below_threshold.sum() * frame_gap),
        )
        index = np.argwhere(below_threshold)
        first, last = int(index[0][0]), int(index[-1][0])
        gap = threshold_margin // frame_gap
        lo = max(0, first - gap)
        hi = min(len(all_frames_to_check) - 1, last + gap)
        refined_check = np.arange(all_frames_to_check[lo], all_frames_to_check[hi])
        below = np.sum(movie[refined_check], axis=(1, 2)) < threshold
        frames_to_add = refined_check[~below]
        frames_to_check = np.array(
            sorted(set(frames_to_check.tolist()).union(set(frames_to_add.tolist()))),
            dtype=int,
        )
    else:
        frames_to_check = all_frames_to_check

    initial_vector = _get_initial_vector(
        movie=movie,
        indexes=frames_to_check,
        rolling_baseline=rolling_baseline,
        parallel=parallel,
        n_jobs=n_jobs,
        seed=seed,
        use_rust=use_rust,
    )

    delta_x = initial_vector.x[1:] - initial_vector.x[:-1]
    delta_y = initial_vector.y[1:] - initial_vector.y[:-1]
    delta_t = frames_to_check[1:] - frames_to_check[:-1]
    delta_d = np.hypot(delta_x, delta_y)
    velocity = delta_d / delta_t

    vmean, vstd = float(np.mean(velocity)), float(np.std(velocity))

    # First step is baseline→first sampled frame (both measured against frames[0]),
    # so its velocity is meaningless noise; replace with the mean to avoid it
    # tripping the 3σ outlier filter below.
    velocity[0] = vmean

    too_fast = velocity > vmean + vstd * 3
    too_slow = velocity < vmean - vstd * 3
    bad = too_fast | too_slow

    bad_points = np.full(len(bad) + 1, False)
    bad_points[np.argwhere(bad)[:, 0]] = True
    bad_points[np.argwhere(bad)[:, 0] + 1] = True

    logger.info("Bad points = %d", int(np.sum(bad_points)))

    smooth_x, smooth_y = (
        np.interp(np.arange(len(movie)), frames_to_check[~bad_points], xy[~bad_points])
        for xy in (initial_vector.x, initial_vector.y)
    )

    return DriftVector(smooth_x, smooth_y)


def get_vector_segmented(
    movie: FloatArray,
    x_grid: int | None = None,
    y_grid: int | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> list[DriftVector]:
    """Estimate per-tile drift vectors for a spatially segmented *movie*.

    Splits the movie into a ``y_grid`` × ``x_grid`` grid of tiles and runs
    :func:`get_vector` on each tile.  When neither grid size is supplied,
    each tile is roughly 30 pixels per side (with a minimum of 2 rows).
    """
    _, y_dim, x_dim = movie.shape

    if x_grid is None and y_grid is None:
        y_grid = max(y_dim // 30, 2)
        x_grid = max(x_dim // 30, 1)
    elif x_grid is None or y_grid is None:
        raise ValueError("Specify both x_grid and y_grid, or neither")

    def _bounds(dim: int, grid: int) -> IntArray:
        return np.linspace(0, (dim - dim % grid), grid + 1, dtype=int)

    xs = _bounds(x_dim, x_grid)
    ys = _bounds(y_dim, y_grid)

    x_gaps = np.transpose([xs[:-1], xs[1:]])
    y_gaps = np.transpose([ys[:-1], ys[1:]])

    slices = [
        (slice(None, None), slice(*ygap), slice(*xgap))
        for xgap in x_gaps
        for ygap in y_gaps
    ]
    return [get_vector(movie[sl], **kwargs) for sl in slices]


# ---------------------------------------------------------------------------
# Movie correction
# ---------------------------------------------------------------------------


def get_corrected_movie(
    movie: FloatArray,
    vector: DriftVector | None = None,
    *,
    full_output: bool = False,
    parallel: bool = True,
    n_jobs: int | None = None,
    use_rust: bool = True,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[FloatArray, DriftVector] | tuple[FloatArray, DriftVector, tuple[slice, ...]]:
    """Apply drift correction to *movie*.

    If *vector* is *None*, one is estimated by calling :func:`get_vector` with
    any extra ``kwargs``.  The returned movie is cropped to the region that
    remains in view across every frame.

    ``use_rust`` is forwarded both to :func:`get_vector` (when *vector* is
    estimated here) and to each :func:`stabilise` call below — prefer the
    compiled Rust extension when available, falling back to Python
    transparently otherwise. Defaults to ``True``.
    """
    if vector is None:
        vector = get_vector(
            movie, parallel=parallel, n_jobs=n_jobs, use_rust=use_rust, **kwargs
        )

    if len(movie) != len(vector):
        raise ValueError(
            f"Vector length {len(vector)} does not match movie length {len(movie)}"
        )

    args = [
        {"frame": f, "x": float(x), "y": float(y), "use_rust": use_rust}
        for f, (x, y) in zip(movie, vector, strict=True)
    ]
    stabilised_movie = np.array(
        dispatch_kwargs(stabilise, args, parallel=parallel, n_jobs=n_jobs)
    )

    sel2d = _always_visible_region(vector)
    sel3d: tuple[slice, ...] = (slice(None, None),) + sel2d

    if full_output:
        return stabilised_movie[sel3d], vector, sel3d
    return stabilised_movie[sel3d], vector


# ---------------------------------------------------------------------------
# Illumination correction
# ---------------------------------------------------------------------------


def illumination_correction(
    movie: FloatArray,
    dc_movie: FloatArray,
    space_smoothing: int = 5,
    time_smoothing: int = 100,
) -> FloatArray:
    """Correct illumination non-uniformity and per-frame intensity drift.

    Estimates a spatial illumination profile from the median of per-frame
    log-Gaussian-smoothed images, normalises by a smoothed proxy for laser
    power derived from *dc_movie*, and divides each frame of *movie* through.

    Both *movie* and *dc_movie* must be strictly positive — the spatial profile
    is computed via ``log10``, which returns ``-inf``/``nan`` (with only a
    ``RuntimeWarning``) for non-positive pixels.  Raw iSCAT frames meet this
    precondition, but ratiometric or subtraction pre-processing can produce
    negative values and should not be passed in directly.
    """
    if np.any(movie <= 0) or np.any(dc_movie <= 0):
        raise ValueError(
            "illumination_correction requires strictly positive pixel values"
        )

    log_smoothed = np.array(
        [10 ** gaussian_filter(np.log10(frame), space_smoothing) for frame in movie]
    )

    totals = np.sum(dc_movie, axis=(1, 2))
    smooth_totals = median_filter(totals, time_smoothing)
    norm_totals = smooth_totals / np.mean(smooth_totals)

    log_smoothed_norm = np.array(
        [lp / n for lp, n in zip(log_smoothed, norm_totals, strict=True)]
    )

    illum_profile = np.median(log_smoothed_norm, axis=0)
    illum_profile /= illum_profile.mean()

    corrected = np.array(
        [
            frame / (illum_profile * nt)
            for frame, nt in zip(movie, norm_totals, strict=True)
        ]
    )
    return corrected
