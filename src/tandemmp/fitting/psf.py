"""PSF fitting for iSCAT peak analysis.

Provides :class:`PsfModel` (a scipy-spline PSF, e.g. the per-movie
experimental PSF), 3-D ratiometric fitting, and cached PSF generation
utilities. The default PSF (:func:`get_psf_model`) is the bundled Rust spline
(:class:`~tandemmp.fitting.spline_psf.SplinePsfModel`).
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.typing as npt
import scipy.optimize

if TYPE_CHECKING:
    from collections.abc import Callable

    from scipy.interpolate import RectBivariateSpline

    from tandemmp.fitting.gaussian_psf import GaussianPsfModel
    from tandemmp.fitting.spline_psf import SplinePsfModel

from tandemmp.core.classifications import Classification
from tandemmp.fitting.minimize import minimize_model
from tandemmp.fitting.psf_interface import (
    PeakFitResult,
    PsfInterface,
    _check_param_names,
    dynamic_mesh,
    error_metric,
    fit_peak_exclusion_generic,
    fit_peak_generic,
    square_mesh,
)

__all__ = [
    "PeakFitResult",
    "PsfModel",
    "correct_known_positions",
    "dynamic_mesh",
    "error_metric",
    "fit_3d_ratiometric",
    "get_psf_model",
    "make_3d_image",
    "native_impression",
    "psf_2d_cached",
    "square_mesh",
]

logger = logging.getLogger(__name__)

# Constants

_UPPER_CONTRAST_BOUND: float = -0.001652

# PSF Model


# TODO: make PsfModel immutable such that setting a new spline_model with
# `set_spline_model()` doesn't hit old cache from the previous spline model.
class PsfModel:
    """PSF backed by a :class:`~scipy.interpolate.RectBivariateSpline`.

    Holds the per-movie experimental PSF; has no shape parameters. Hashing is
    by identity, so caches keyed on it survive :meth:`set_spline_model`.
    """

    def __init__(self, spline: RectBivariateSpline | None = None) -> None:
        self._spline_model = spline

    @property
    def has_spline_model(self) -> bool:
        """True when a spline has been set."""
        return self._spline_model is not None

    @property
    def spline_model(self) -> RectBivariateSpline | None:
        """The spline, or ``None`` if not set."""
        return self._spline_model

    @property
    def psf_params(self) -> dict[str, float]:
        """No shape parameters."""
        return {}

    @property
    def param_bounds(self) -> dict[str, tuple[float | None, float | None]]:
        """No shape parameters."""
        return {}

    def with_params(self, **params: float) -> PsfModel:
        """Return a copy sharing the spline (no parameters to replace)."""
        _check_param_names(params, self.psf_params)
        return PsfModel(self._spline_model)

    def set_spline_model(self, spline: RectBivariateSpline) -> None:
        """Attach the spline."""
        self._spline_model = spline

    def get_psf_function(self) -> Callable[..., npt.NDArray[np.floating]]:
        """Return ``f(x, y, c=1.0) -> image``."""
        spline = self._spline_model
        if spline is None:
            msg = "PsfModel has no spline; call set_spline_model() first"
            raise ValueError(msg)

        def _psf_spline(
            x: npt.NDArray[np.floating],
            y: npt.NDArray[np.floating],
            c: float = 1.0,
        ) -> npt.NDArray[np.floating]:
            return spline(y, x, grid=False) * -c  # type: ignore[no-any-return]

        return _psf_spline

    def psf_2d_offset(
        self,
        width: int,
        dx: float,
        dy: float,
        c: float,
    ) -> npt.NDArray[np.floating]:
        """PSF image offset by *(dx, dy)* with contrast *c*."""
        x_coords, y_coords = square_mesh(width)
        return self.get_psf_function()(x_coords - dx, y_coords - dy, c=c)

    def render(
        self,
        dx: float,
        dy: float,
        c: float,
        size: int,
    ) -> npt.NDArray[np.floating]:
        """Same as :meth:`psf_2d_offset`."""
        return self.psf_2d_offset(size, dx, dy, c)

    def fit_peak(
        self,
        peak: npt.NDArray[np.floating],
        **kwargs: Any,  # noqa: ANN401
    ) -> PeakFitResult:
        """Fit via :func:`fit_peak_generic` (``backend="spline"``)."""
        result = fit_peak_generic(self, np.asarray(peak, dtype=np.float64), **kwargs)
        if result.backend == "none":
            return result
        return dataclasses.replace(result, backend="spline")

    def fit_peak_exclusion(
        self,
        peak: npt.NDArray[np.floating],
        exclusion: npt.NDArray[np.bool_],
        **kwargs: Any,  # noqa: ANN401
    ) -> PeakFitResult:
        """Fit via :func:`fit_peak_exclusion_generic` (``backend="spline"``)."""
        result = fit_peak_exclusion_generic(self, peak, exclusion, **kwargs)
        return dataclasses.replace(result, backend="spline")


# Module-level cached PSF helpers


@overload
def get_psf_model(*, kind: Literal["spline"] = "spline") -> SplinePsfModel: ...


@overload
def get_psf_model(*, kind: Literal["gaussian"]) -> GaussianPsfModel: ...


@functools.lru_cache(maxsize=4)
def get_psf_model(
    *,
    kind: Literal["spline", "gaussian"] = "spline",
) -> PsfInterface:
    """Return a cached default PSF: ``"spline"`` (bundled Rust spline) or
    ``"gaussian"``.
    """
    if kind == "gaussian":
        from tandemmp.fitting.gaussian_psf import GaussianPsfModel

        return GaussianPsfModel()
    from tandemmp.fitting.spline_psf import SplinePsfModel

    return SplinePsfModel()


@functools.lru_cache(maxsize=128)
def _psf_unit_cached(
    dx: float,
    dy: float,
    l_thum: int,
    psf_model: PsfInterface,
) -> npt.NDArray[np.floating]:
    """Cached unit-contrast PSF image at offset *(dx, dy)*."""
    return psf_model.render(dx, dy, 1.0, l_thum)


def psf_2d_cached(
    dx: float,
    dy: float,
    c: float,
    l_thum: int,
    psf_model: PsfInterface | None = None,
) -> npt.NDArray[np.floating]:
    """Return a PSF image at *(dx, dy)* with contrast *c*, using cache."""
    if psf_model is None:
        psf_model = get_psf_model()
    return _psf_unit_cached(dx, dy, l_thum, psf_model) * c


def correct_known_positions(
    peak: npt.NDArray[np.floating],
    dxs: Sequence[float],
    dys: Sequence[float],
    cs: Sequence[float],
    psf_model: PsfInterface | None = None,
) -> npt.NDArray[np.floating]:
    """Subtract known PSF impressions from *peak*."""
    l_thum = peak.shape[0]
    frames = [
        psf_2d_cached(dx, dy, c, l_thum, psf_model)
        for dx, dy, c in zip(dxs, dys, cs, strict=True)
    ]
    impression: npt.NDArray[np.floating] = np.sum(
        frames,
        axis=0,
    )
    return peak - impression


# 3-D ratiometric image construction


def _model_3d_psf(
    dx: Sequence[float],
    dy: Sequence[float],
    c: float | Sequence[float],
    l_thum: int = 15,
    psf_model: PsfInterface | None = None,
    use_cache: bool = True,
) -> npt.NDArray[np.floating]:
    """Model several frames of a PSF at single or multiple contrasts."""
    n = len(dx)

    # Expand contrast to a per-frame list
    if isinstance(c, (list, np.ndarray)):
        c_list = list(c)
        if len(c_list) == 1:
            c_list = c_list * n
        elif len(c_list) == 2:
            navg = n // 2
            c_list = [c_list[0]] * navg + [c_list[1]] * (n - navg)
        elif len(c_list) != n:
            msg = f"Contrast list length {len(c_list)} doesn't match frame count {n}"
            raise ValueError(msg)
    else:
        c_list = [float(c)] * n  # type: ignore[arg-type]

    if psf_model is None:
        psf_model = get_psf_model()

    if use_cache:
        return np.array(
            [
                psf_2d_cached(_dx, _dy, _c, l_thum, psf_model)
                for _dx, _dy, _c in zip(dx, dy, c_list, strict=True)
            ]
        )

    return np.array(
        [
            psf_model.render(_dx, _dy, _c, l_thum)
            for _dx, _dy, _c in zip(dx, dy, c_list, strict=True)
        ]
    )


def make_3d_image(
    dx: Sequence[float],
    dy: Sequence[float],
    c: float | Sequence[float],
    classification: Classification,
    navg: int,
    l_thum: int = 15,
    is_bad: Sequence[bool] | None = None,
    psf_model: PsfInterface | None = None,
    *,
    use_cache: bool = True,
) -> npt.NDArray[np.floating]:
    """Create a 2-D ratiometric image from a sequence of PSF frames.

    Combines *before* and *after* windows according to the classification's
    ``presence_before`` / ``presence_after`` flags.
    """
    psfs = _model_3d_psf(dx, dy, c, l_thum, psf_model, use_cache)

    if is_bad is not None and np.mean(is_bad) < 0.3:
        bad_arr = np.asarray(is_bad, dtype=bool)
        psfs[bad_arr] = 0.0

    if not classification.presence_before:
        # Binding: only "after" window
        return psfs[-navg:].mean(axis=0)  # type: ignore[no-any-return]

    if not classification.presence_after:
        # Unbinding: only "before" window (negated)
        return -psfs[:navg].mean(axis=0)  # type: ignore[no-any-return]

    # Movement / association / dissociation: after − before
    return psfs[-navg:].mean(axis=0) - psfs[:navg].mean(axis=0)  # type: ignore[no-any-return]


# 3-D ratiometric fitting


def fit_3d_ratiometric(  # noqa: PLR0913
    peak: npt.NDArray[np.floating],
    dx: Sequence[float],
    dy: Sequence[float],
    classification: Classification,
    *,
    c_guess: float | None = None,
    lock: bool = False,
    particle_c: float | None = None,
    is_bad: Sequence[bool] | None = None,
    psf_model: PsfInterface | None = None,
    method: str = "Powell",
    use_cache: bool = True,
) -> tuple[scipy.optimize.OptimizeResult, npt.NDArray[np.floating]]:
    """Fit contrast(s) to a ratiometric peak via a 3-D PSF model.

    Returns ``(result, best_fit_image)``.
    """
    l_thum = peak.shape[0]
    navg = len(dx) // 2

    def _fit_2c(c1: float, c2: float) -> npt.NDArray[np.floating]:
        return make_3d_image(
            dx,
            dy,
            [c1, c2],
            classification,
            navg,
            l_thum,
            is_bad,
            psf_model,
            use_cache=use_cache,
        )

    # Build error function depending on classification and lock mode
    n_params = 1

    if lock:
        if particle_c is None:
            msg = "particle_c is required when lock=True"
            raise ValueError(msg)

        def make_im(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return _fit_2c(particle_c, float(v[0]))

        bounds: list[list[float | None]] = [[None, _UPPER_CONTRAST_BOUND]]

    elif not classification.presence_before:

        def make_im(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return _fit_2c(0.0, float(v[0]))

        bounds = [[None, _UPPER_CONTRAST_BOUND]]

    elif not classification.presence_after:

        def make_im(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return _fit_2c(float(v[0]), 0.0)

        bounds = [[None, _UPPER_CONTRAST_BOUND]]

    else:
        n_params = 2

        def make_im(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            return _fit_2c(float(v[0]), float(v[1]))

        if particle_c is None or np.isnan(particle_c):
            bounds = [[None, _UPPER_CONTRAST_BOUND]] * n_params
        else:
            c_guess = particle_c
            margin = abs(particle_c * 0.2)
            lo, hi = particle_c - margin, particle_c + margin
            bounds = [[min(lo, hi), max(lo, hi)], [None, _UPPER_CONTRAST_BOUND]]

    if c_guess is None:
        c_guess = -float(np.max(np.abs(peak)))

    def err(v: npt.NDArray[np.floating]) -> float:
        return error_metric(peak, make_im(v))

    result = minimize_model(
        err,
        x0=[c_guess] * n_params,
        bounds=bounds,
        method=method,
    )

    return result.raw, make_im(result.raw.x)


def native_impression(
    dx: Sequence[float],
    dy: Sequence[float],
    c: Sequence[float],
    classification: Classification,
    l_thum: int = 15,
    psf_model: PsfInterface | None = None,
) -> npt.NDArray[np.floating]:
    """Generate a PSF impression for known positions and contrasts."""
    navg = len(dx) // 2
    return make_3d_image(
        dx,
        dy,
        c,
        classification=classification,
        navg=navg,
        l_thum=l_thum,
        psf_model=psf_model,
    )
