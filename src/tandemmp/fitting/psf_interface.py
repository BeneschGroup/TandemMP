"""Generic PSF-fitting interface and algorithm.

Defines :class:`PsfInterface`, a :class:`typing.Protocol` describing the
minimal surface a PSF model must expose (PSF rendering + peak fitting), and
the backend-agnostic peak-fitting algorithm (:func:`fit_peak_generic`,
:func:`fit_peak_exclusion_generic`) that operates on any conforming model.

This module is intentionally self-contained (no imports from
:mod:`tandemmp.fitting.psf`) so that Bessel-specific
(:class:`~tandemmp.fitting.psf.PsfModel`) and non-Bessel (e.g.
:class:`~tandemmp.fitting.gaussian_psf.GaussianPsfModel`) implementations can
both depend on it without introducing an import cycle — :mod:`tandemmp.fitting.psf`
imports :data:`PeakFitResult`, :func:`square_mesh`, :func:`dynamic_mesh`, and
:func:`error_metric` from here (and re-exports them for backward
compatibility) rather than the other way around.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

from tandemmp.fitting.minimize import minimize_model
from tandemmp.fitting.radialcenter import radialcenter

logger = logging.getLogger(__name__)

# Data containers


@dataclasses.dataclass(frozen=True)
class PeakFitResult:
    """Result of a 2-D PSF peak fit."""

    contrast: float
    dx: float
    dy: float
    error: float
    converged: bool
    backend: Literal["python", "spline", "rust", "none"]


# Grid utilities (cached)


@functools.lru_cache(maxsize=128)
def square_mesh(
    width: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return *(X, Y)* coordinate grids centred on zero for a square image."""
    half = width // 2
    if width % 2 == 0:
        msg = "square_mesh requires an odd width"
        raise ValueError(msg)
    coords = np.arange(-half, half + 1)
    return np.meshgrid(coords, coords)  # type: ignore[return-value]


@functools.lru_cache(maxsize=128)
def dynamic_mesh(
    height: int,
    width: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return *(X, Y)* grids for potentially non-square or even-sized images."""
    wh, hh = width // 2, height // 2
    x = np.arange(-wh, wh + 1) if width % 2 == 1 else np.arange(-wh, wh)
    y = np.arange(-hh, hh + 1) if height % 2 == 1 else np.arange(-hh, hh)
    return np.meshgrid(x, y)  # type: ignore[return-value]


# Error metric


def error_metric(
    peak: npt.NDArray[np.floating],
    fit: npt.NDArray[np.floating],
    exclusion: npt.NDArray[np.bool_] | None = None,
) -> float:
    """Sum-of-squares error between *peak* and *fit*, optionally masked."""
    if exclusion is not None:
        peak = peak[~exclusion]
        fit = fit[~exclusion]
    return float(np.sum((fit - peak) ** 2))


# PSF interface


@runtime_checkable
class PsfInterface(Protocol):
    """Structural interface a PSF model must satisfy to use the generic
    peak-fitting algorithm (:func:`fit_peak_generic`,
    :func:`fit_peak_exclusion_generic`).

    Concrete implementations: :class:`tandemmp.fitting.psf.PsfModel` (Bessel
    PSF, spline/Rust backends) and
    :class:`tandemmp.fitting.gaussian_psf.GaussianPsfModel` (analytic 2-D
    Gaussian, no spline/Rust backends).
    """

    def __hash__(self) -> int:
        """Declared so implementations are usable as ``functools.lru_cache``
        keys (e.g. :func:`tandemmp.fitting.psf._psf_unit_cached`) — every
        concrete implementation gets this for free from :func:`object`'s
        default identity-based hash, since neither defines a custom
        ``__eq__``.
        """
        ...

    @property
    def psf_params(self) -> dict[str, float]:
        """Return the model's shape parameters as a plain dict."""
        ...

    def get_psf_function(self) -> Callable[..., npt.NDArray[np.floating]]:
        """Return a callable ``f(x, y, c=1.0) -> image``."""
        ...

    def psf_2d_offset(
        self,
        width: int,
        dx: float,
        dy: float,
        c: float,
    ) -> npt.NDArray[np.floating]:
        """Generate a PSF image offset by *(dx, dy)* with given contrast."""
        ...

    def render(
        self,
        dx: float,
        dy: float,
        c: float,
        size: int,
    ) -> npt.NDArray[np.floating]:
        """Render a PSF image at *(dx, dy)* with contrast *c*."""
        ...

    def fit_peak(
        self,
        peak: npt.NDArray[np.floating],
        **kwargs: Any,  # noqa: ANN401
    ) -> PeakFitResult:
        """Fit this PSF model to a 2-D peak thumbnail."""
        ...

    def fit_peak_exclusion(
        self,
        peak: npt.NDArray[np.floating],
        exclusion: npt.NDArray[np.bool_],
        **kwargs: Any,  # noqa: ANN401
    ) -> PeakFitResult:
        """Fit this PSF model to *peak*, ignoring pixels where *exclusion* is True."""
        ...


# Initial position guess


def _initial_position_guess(
    peak: npt.NDArray[np.floating],
) -> tuple[float, float]:
    """Estimate initial *(dx, dy)* from a peak image.

    Shared by every :class:`PsfInterface` implementation — the estimator
    (:func:`tandemmp.fitting.radialcenter.radialcenter`, with a
    weighted-centre-of-mass fallback for degenerate thumbnails) has no
    dependency on any particular PSF shape.
    """
    if np.isnan(np.asarray(peak)).any():
        return 0.0, 0.0
    try:
        dx, dy, _ = radialcenter(np.asarray(peak, dtype=float), mode="rel")
        if np.isfinite(dx) and np.isfinite(dy):
            return float(dx), float(dy)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        pass

    # Fallback: weighted centre-of-mass
    weights = np.abs(np.asarray(peak, dtype=float))
    total = np.sum(weights)
    if total == 0:
        return 0.0, 0.0
    width = peak.shape[0]
    x_coords, y_coords = square_mesh(width)
    return (
        float(np.sum(x_coords * weights) / total),
        float(np.sum(y_coords * weights) / total),
    )


# Generic peak fitting


def fit_peak_generic(  # noqa: PLR0913
    psf_model: PsfInterface,
    peak: npt.NDArray[np.floating],
    *,
    c_guess: float | None = None,
    dx_guess: float | None = None,
    dy_guess: float | None = None,
    method: str = "BFGS",
    gtol: float = 1e-6,
    allow_offset: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> PeakFitResult:
    """Fit any :class:`PsfInterface` model to a 2-D peak thumbnail.

    This is the backend-agnostic fitting algorithm — it only calls
    ``psf_model.get_psf_function()``, so it works identically for
    :class:`~tandemmp.fitting.psf.PsfModel` and
    :class:`~tandemmp.fitting.gaussian_psf.GaussianPsfModel` alike. Returned
    results always report ``backend="python"`` (or ``"none"`` for an
    all-zero image); callers that have a more specific backend to report
    (e.g. :class:`~tandemmp.fitting.psf.PsfModel`'s spline backend) should
    override ``.backend`` on the returned :class:`PeakFitResult` themselves.

    Args:
        psf_model: Any object satisfying :class:`PsfInterface`.
        peak: Square 2-D image (odd edge length expected).
        c_guess: Initial contrast guess. Computed from central pixels if *None*.
        dx_guess: Initial x position offset. Estimated automatically if *None*.
        dy_guess: Initial y position offset. Estimated automatically if *None*.
        method: Optimisation method for :func:`scipy.optimize.minimize`.
        gtol: Gradient tolerance (BFGS).
        allow_offset: Allow a constant-offset parameter in the fit. Forced to
            *True* regardless of this argument when the contrast guess is large.
        **kwargs: Ignored. Accepted so every :class:`PsfInterface` implementation
            that forwards its own ``fit_peak(**kwargs)`` here (e.g.
            :class:`~tandemmp.fitting.gaussian_psf.GaussianPsfModel`) tolerates
            backend-specific keywords like :class:`~tandemmp.fitting.psf.PsfModel`
            ``.fit_peak``'s ``rust=`` the same way that method does.
    """
    if peak.ndim != 2 or peak.shape[0] != peak.shape[1]:
        msg = "Peak image must be a square 2D array"
        raise ValueError(msg)

    if peak.dtype != np.float64:
        peak = peak.astype(np.float64)

    # All-zero image → trivial result
    if np.all(peak == 0):
        return PeakFitResult(
            contrast=0.0,
            dx=0.0,
            dy=0.0,
            error=1.0,
            converged=False,
            backend="none",
        )

    width = peak.shape[0]
    x_coords, y_coords = square_mesh(width)
    r = np.hypot(x_coords, y_coords)

    if dx_guess is None or dy_guess is None:
        dx_est, dy_est = _initial_position_guess(peak)
        if dx_guess is None:
            dx_guess = dx_est
        if dy_guess is None:
            dy_guess = dy_est

    if c_guess is None:
        c_guess = float(np.mean(peak[r <= 1]))

    # Large-contrast peaks → force offset
    if abs(c_guess) >= 0.5:
        allow_offset = True

    psf_fn = psf_model.get_psf_function()

    def _model(c: float, dx: float, dy: float) -> npt.NDArray[np.floating]:
        return psf_fn(x=x_coords - dx, y=y_coords - dy, c=c)

    if allow_offset:
        x0 = np.array([c_guess, dx_guess, dy_guess, 0.0])

        def err(v: npt.NDArray[np.floating]) -> float:
            return error_metric(peak + v[3], _model(v[0], v[1], v[2]))

    else:
        x0 = np.array([c_guess, dx_guess, dy_guess])

        def err(v: npt.NDArray[np.floating]) -> float:
            return error_metric(peak, _model(v[0], v[1], v[2]))

    options: dict[str, float] = {"gtol": gtol} if method == "BFGS" else {}
    fit = minimize_model(err, x0, method=method, options=options)

    return PeakFitResult(
        contrast=float(fit.params[0]),
        dx=float(fit.params[1]),
        dy=float(fit.params[2]),
        error=float(fit.fun),
        converged=bool(fit.success),
        backend="python",
    )


def fit_peak_exclusion_generic(
    psf_model: PsfInterface,
    peak: npt.NDArray[np.floating],
    exclusion: npt.NDArray[np.bool_],
    *,
    method: str = "BFGS",
    gtol: float = 1e-6,
    **kwargs: Any,  # noqa: ANN401
) -> PeakFitResult:
    """Fit any :class:`PsfInterface` model to *peak*, ignoring pixels where
    *exclusion* is True.

    Two initial guesses (radial-centre and argmin) are tried; the one
    with lower error is returned. Always reports ``backend="python"`` —
    see :func:`fit_peak_generic` for the override convention. ``**kwargs``
    is ignored, for the same reason documented on :func:`fit_peak_generic`.
    """
    if peak.shape != exclusion.shape:
        msg = "Exclusion mask must match peak shape"
        raise ValueError(msg)

    psf_fn = psf_model.get_psf_function()
    masked_peak = np.ma.array(peak, mask=exclusion)  # type: ignore[no-untyped-call]

    width = peak.shape[0]
    half_width = width // 2
    x_coords, y_coords = square_mesh(width)

    # Guess 1: position estimator
    dx_g1, dy_g1 = _initial_position_guess(masked_peak)

    # Guess 2: argmin position
    min_idx = np.unravel_index(
        int(np.argmin(masked_peak)),
        peak.shape,
    )
    dy_g2 = float(min_idx[0] - half_width)
    dx_g2 = float(min_idx[1] - half_width)

    c_guess = float(np.min(masked_peak))

    def _make(c: float, dx: float, dy: float) -> npt.NDArray[np.floating]:
        return psf_fn(x=x_coords - dx, y=y_coords - dy, c=c)

    def err(v: npt.NDArray[np.floating]) -> float:
        return error_metric(peak, _make(v[0], v[1], v[2]), exclusion)

    options: dict[str, float] = {"gtol": gtol} if method == "BFGS" else {}

    results = [
        minimize_model(
            err,
            x0=np.array([c_guess, dx, dy]),
            method=method,
            options=options,
        )
        for dx, dy in [(dx_g1, dy_g1), (dx_g2, dy_g2)]
    ]

    best = results[int(np.argmin([r.fun for r in results]))]
    c_fit, dx_fit, dy_fit = best.params

    # Report error on full (unmasked) peak for comparability
    full_error = error_metric(peak, _make(c_fit, dx_fit, dy_fit))

    return PeakFitResult(
        contrast=float(c_fit),
        dx=float(dx_fit),
        dy=float(dy_fit),
        error=float(full_error),
        converged=bool(best.success),
        backend="python",
    )
