"""Bundled experimental spline PSF, evaluated by ``tandemmp._rust.spline_psf``.

The PSF (20260423, ms1000, mfp2) is compiled into the extension. Shape
parameters ``a``, ``b`` scale the x and y axes: ``psf(x, y) = s(b*y, a*x)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from tandemmp.fitting.psf_interface import (
    PeakFitResult,
    _check_param_names,
    fit_peak_exclusion_generic,
    fit_peak_generic,
    square_mesh,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_SCALE_BOUNDS: tuple[float, float] = (0.5, 2.0)


def _load_eval() -> Callable[..., npt.NDArray[np.float64]]:
    try:
        from tandemmp._rust.spline_psf import eval as spline_eval
    except ImportError as exc:
        msg = (
            "tandemmp._rust.spline_psf is not built; run `maturin develop "
            "--release` in src/tandemmp/_rust/spline_psf"
        )
        raise ImportError(msg) from exc
    return spline_eval


class SplinePsfModel:
    """Bundled experimental spline PSF with axis scales ``a`` (x), ``b`` (y)."""

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self.a = a
        self.b = b

    @property
    def psf_params(self) -> dict[str, float]:
        """Axis scales."""
        return {"a": self.a, "b": self.b}

    @property
    def param_bounds(self) -> dict[str, tuple[float | None, float | None]]:
        """Keeps scaled coordinates within the spline's +/-12 px domain."""
        return {"a": _SCALE_BOUNDS, "b": _SCALE_BOUNDS}

    def with_params(self, **params: float) -> SplinePsfModel:
        """Return a copy with *params* replaced."""
        _check_param_names(params, self.psf_params)
        return SplinePsfModel(**{**self.psf_params, **params})

    def get_psf_function(self) -> Callable[..., npt.NDArray[np.floating]]:
        """Return ``f(x, y, c=1.0) -> image``; the peak value is ~``c``."""
        spline_eval = _load_eval()
        a, b = self.a, self.b

        def _psf(
            x: npt.NDArray[np.floating],
            y: npt.NDArray[np.floating],
            c: float = 1.0,
        ) -> npt.NDArray[np.floating]:
            u, v = np.broadcast_arrays(
                b * np.asarray(y, dtype=np.float64),
                a * np.asarray(x, dtype=np.float64),
            )
            return spline_eval(u, v) * -c

        return _psf

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
        """Fit via :func:`fit_peak_generic`."""
        return fit_peak_generic(self, peak, **kwargs)

    def fit_peak_exclusion(
        self,
        peak: npt.NDArray[np.floating],
        exclusion: npt.NDArray[np.bool_],
        **kwargs: Any,  # noqa: ANN401
    ) -> PeakFitResult:
        """Fit via :func:`fit_peak_exclusion_generic`."""
        return fit_peak_exclusion_generic(self, peak, exclusion, **kwargs)
