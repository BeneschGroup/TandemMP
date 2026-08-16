"""PSF fitting for iSCAT peak analysis.

Provides :class:`PsfModel` (replaces the old ``CustomPeakModel``), 2-D peak
fitting, 3-D ratiometric fitting, and cached PSF generation utilities.

``open_iscat`` functions (``psf_2d``, ``radialcenter``) are lazy-imported so
the module works without that package when a spline model is provided.
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

from tandemmp.core.classifications import Classification
from tandemmp.fitting.minimize import minimize_model
from tandemmp.fitting.psf_interface import (
    PeakFitResult,
    PsfInterface,
    _initial_position_guess,
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_A12: float = -5.184416672336133
_DEFAULT_W: float = 2.2614175995143952
_DEFAULT_S: float = 4.249591584927938
_UPPER_CONTRAST_BOUND: float = -0.001652

# ---------------------------------------------------------------------------
# Optional dependency loader
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_open_iscat_psf_2d() -> Any:  # noqa: ANN401
    """Load and cache ``open_iscat.utils.psf.psf_2d`` on first use."""
    from open_iscat.utils.psf import psf_2d

    return psf_2d


# ---------------------------------------------------------------------------
# PSF Model
# ---------------------------------------------------------------------------


# TODO: make PsfModel immutable such that setting a new spline_model with
# `set_spline_model()` doesn't hit old cache from the previous spline model.
class PsfModel:
    """PSF model for iSCAT peak fitting.

    Standalone replacement for the old ``CustomPeakModel``.  Does **not**
    inherit from ``open_iscat``'s ``PeakModel`` — all required functionality
    is implemented directly.  When a :class:`~scipy.interpolate.RectBivariateSpline`
    is set via :meth:`set_spline_model`, ``open_iscat`` is not required at all.
    """

    def __init__(
        self,
        a12: float = _DEFAULT_A12,
        w: float = _DEFAULT_W,
        s: float = _DEFAULT_S,
        *,
        use_rust: bool = False,
    ) -> None:
        self.a12 = a12
        self.w = w
        self.s = s
        self.use_rust = use_rust
        self._spline_model: RectBivariateSpline | None = None
        self._rust_model_cache: dict[int, Any] = {}

    # Identity-based hashing (inherited from ``object``) is used so that
    # caches keyed on :class:`PsfModel` instances remain correct when the
    # model is mutated via :meth:`set_spline_model`.

    # -- Properties ----------------------------------------------------------

    @property
    def has_spline_model(self) -> bool:
        """True when an empirical spline PSF has been set."""
        return self._spline_model is not None

    @property
    def spline_model(self) -> RectBivariateSpline | None:
        """The empirical spline PSF, or ``None`` if not set."""
        return self._spline_model

    @property
    def psf_params(self) -> dict[str, float]:
        """Return the three analytical PSF parameters."""
        return {"a12": self.a12, "w": self.w, "s": self.s}

    def set_spline_model(self, spline: RectBivariateSpline) -> None:
        """Attach an empirical spline-based PSF model."""
        self._spline_model = spline

    # -- PSF generation ------------------------------------------------------

    def get_psf_function(self) -> Callable[..., npt.NDArray[np.floating]]:
        """Return a callable ``f(x, y, c=1.0) -> image``.

        Uses the spline model if available, otherwise falls back to
        ``open_iscat.utils.psf.psf_2d`` (lazy-imported).
        """
        if self._spline_model is not None:
            spline = self._spline_model

            def _psf_spline(
                x: npt.NDArray[np.floating],
                y: npt.NDArray[np.floating],
                c: float = 1.0,
            ) -> npt.NDArray[np.floating]:
                return spline(y, x, grid=False) * -c  # type: ignore[no-any-return]

            return _psf_spline

        # Fallback — requires open_iscat
        a12, w, s = self.a12, self.w, self.s

        def _psf_oiscat(
            x: npt.NDArray[np.floating],
            y: npt.NDArray[np.floating],
            c: float = 1.0,
        ) -> npt.NDArray[np.floating]:
            psf_2d = _get_open_iscat_psf_2d()
            return psf_2d(x=x, y=y, a12=a12, w=w, s=s, c=c)  # type: ignore[no-any-return]

        return _psf_oiscat

    def psf_2d_offset(
        self,
        width: int,
        dx: float,
        dy: float,
        c: float,
    ) -> npt.NDArray[np.floating]:
        """Generate a PSF image offset by *(dx, dy)* with given contrast."""
        x_coords, y_coords = square_mesh(width)
        psf_fn = self.get_psf_function()
        return psf_fn(x_coords - dx, y_coords - dy, c=c)

    def _rust_model(self, thumb_size: int) -> Any:  # noqa: ANN202,ANN401
        """Create or retrieve the Rust ``BesselPSF`` model for *thumb_size*."""
        cached = self._rust_model_cache.get(thumb_size)
        if cached is not None:
            return cached
        try:
            # deprecated/rust is pre-refactor, unmaintained code kept only
            # until the Rust PSF hot path is re-implemented under _rust/.
            from deprecated.rust.psf_a12ws import (
                psf_a12ws,
            )
        except (ModuleNotFoundError, ImportError) as exc:
            msg = "Rust PSF module (psf_a12ws) is not available"
            raise ImportError(msg) from exc
        model = psf_a12ws.BesselPSF(self.a12, self.w, self.s, thumb_size)
        self._rust_model_cache[thumb_size] = model
        return model

    def render(
        self,
        dx: float,
        dy: float,
        c: float,
        size: int,
    ) -> npt.NDArray[np.floating]:
        """Render a PSF image at *(dx, dy)* with contrast *c*, using the
        best available backend (spline > Rust > open_iscat).
        """
        if self._spline_model is not None:
            return self.psf_2d_offset(size, dx, dy, c)
        try:
            rust_model = self._rust_model(size)
        except ImportError:
            return self.psf_2d_offset(size, dx, dy, c)
        return rust_model.psf_offset(dx, dy, c)  # type: ignore[no-any-return]

    # -- 2-D peak fitting ----------------------------------------------------

    def fit_peak(  # noqa: PLR0913
        self,
        peak: npt.NDArray[np.floating],
        *,
        c_guess: float | None = None,
        dx_guess: float | None = None,
        dy_guess: float | None = None,
        method: str = "BFGS",
        gtol: float = 1e-6,
        rust: bool = False,
        allow_offset: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> PeakFitResult:
        """Fit a PSF model to a 2-D peak thumbnail.

        Args:
            peak: Square 2-D image (odd edge length expected).
            c_guess: Initial contrast guess. Computed from central pixels if *None*.
            dx_guess: Initial x position offset. Estimated automatically if *None*.
            dy_guess: Initial y position offset. Estimated automatically if *None*.
            method: Optimisation method for :func:`scipy.optimize.minimize`.
            gtol: Gradient tolerance (BFGS).
            rust: Prefer the Rust backend if available.
            allow_offset: Allow a constant-offset parameter in the fit.
            **kwargs: Ignored. Accepted only so this method's signature satisfies
                :class:`~tandemmp.fitting.psf_interface.PsfInterface`'s
                ``fit_peak(self, peak, **kwargs: Any)`` structurally (mypy
                requires a concrete method to accept arbitrary keyword
                arguments to be considered a subtype of a ``**kwargs``-typed
                Protocol method, even though every keyword this class actually
                uses is already named explicitly above).
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

        # Resolve guesses here (rather than leaving them to
        # `fit_peak_generic`) because the Rust fast-path below needs
        # concrete values *before* deciding whether to attempt it.
        if dx_guess is None or dy_guess is None:
            dx_est, dy_est = _initial_position_guess(peak)
            if dx_guess is None:
                dx_guess = dx_est
            if dy_guess is None:
                dy_guess = dy_est

        if c_guess is None:
            width = peak.shape[0]
            x_coords, y_coords = square_mesh(width)
            r = np.hypot(x_coords, y_coords)
            c_guess = float(np.mean(peak[r <= 1]))

        c_under = abs(c_guess) < 0.5

        # Try Rust path
        if not allow_offset and (rust or self.use_rust) and c_under:
            try:
                return self._fit_peak_rust(peak, dx_guess, dy_guess, c_guess, gtol)
            except ImportError:
                logger.debug("Rust PSF module unavailable, falling back to Python")
            except Exception:  # noqa: BLE001
                logger.debug("Rust fit failed, falling back to Python", exc_info=True)

        result = fit_peak_generic(
            self,
            peak,
            c_guess=c_guess,
            dx_guess=dx_guess,
            dy_guess=dy_guess,
            method=method,
            gtol=gtol,
            allow_offset=allow_offset,
        )
        if self.has_spline_model:
            result = dataclasses.replace(result, backend="spline")
        return result

    def _fit_peak_rust(
        self,
        peak: npt.NDArray[np.floating],
        dx_guess: float,
        dy_guess: float,
        c_guess: float,
        gtol: float,
    ) -> PeakFitResult:
        """Fit using the Rust ``BesselPSF`` backend."""
        rust_model = self._rust_model(peak.shape[0])
        dx_fit, dy_fit, c_fit = rust_model.psf_fit(
            thumbnail=peak,
            init_param=np.array([dx_guess, dy_guess, c_guess]),
            g_tol=gtol,
        )
        rust_im = rust_model.psf_offset(dx_fit, dy_fit, contrast=c_fit)
        return PeakFitResult(
            contrast=float(c_fit),
            dx=float(dx_fit),
            dy=float(dy_fit),
            error=float(error_metric(peak, rust_im)),
            converged=True,
            backend="rust",
        )

    def fit_peak_exclusion(
        self,
        peak: npt.NDArray[np.floating],
        exclusion: npt.NDArray[np.bool_],
        *,
        method: str = "BFGS",
        gtol: float = 1e-6,
        **kwargs: Any,  # noqa: ANN401
    ) -> PeakFitResult:
        """Fit a PSF to *peak*, ignoring pixels where *exclusion* is True.

        Two initial guesses (radial-centre and argmin) are tried; the one
        with lower error is returned.

        ``**kwargs`` is accepted-but-ignored for the same
        :class:`~tandemmp.fitting.psf_interface.PsfInterface` structural-
        typing reason documented on :meth:`fit_peak`.
        """
        result = fit_peak_exclusion_generic(
            self, peak, exclusion, method=method, gtol=gtol
        )
        if self.has_spline_model:
            result = dataclasses.replace(result, backend="spline")
        return result


# ---------------------------------------------------------------------------
# Module-level cached PSF helpers
# ---------------------------------------------------------------------------


@overload
def get_psf_model(
    *, use_rust: bool = False, kind: Literal["bessel"] = "bessel"
) -> PsfModel: ...


@overload
def get_psf_model(*, kind: Literal["gaussian"]) -> GaussianPsfModel: ...


@functools.lru_cache(maxsize=4)
def get_psf_model(
    *,
    use_rust: bool = False,
    kind: Literal["bessel", "gaussian"] = "bessel",
) -> PsfInterface:
    """Return a cached default :class:`PsfInterface` implementation.

    ``kind="bessel"`` (the default) returns a :class:`PsfModel`, honouring
    ``use_rust``. ``kind="gaussian"`` returns a
    :class:`~tandemmp.fitting.gaussian_psf.GaussianPsfModel`, which has no
    Rust backend — the ``kind="gaussian"`` overload deliberately omits
    ``use_rust`` so passing it together with ``kind="gaussian"`` is a
    ``mypy`` error rather than a silent no-op.
    """
    if kind == "gaussian":
        from tandemmp.fitting.gaussian_psf import GaussianPsfModel

        return GaussianPsfModel()
    return PsfModel(use_rust=use_rust)


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


# ---------------------------------------------------------------------------
# 3-D ratiometric image construction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3-D ratiometric fitting
# ---------------------------------------------------------------------------


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
