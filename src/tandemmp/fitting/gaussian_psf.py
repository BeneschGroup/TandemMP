"""A 2-D Gaussian PSF model.

The simplest concrete :class:`~tandemmp.fitting.psf_interface.PsfInterface`
implementation: an analytic, un-normalised 2-D Gaussian with a single
``sigma`` shape parameter, no spline/Rust backends. Exists primarily as a
worked example that the generic peak-fitting algorithm in
:mod:`tandemmp.fitting.psf_interface` is not coupled to the spline PSF shape
used by :class:`tandemmp.fitting.psf.PsfModel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from tandemmp.fitting.psf_interface import (
    PeakFitResult,
    fit_peak_exclusion_generic,
    fit_peak_generic,
    square_mesh,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ``1.5`` matches the PSF-width scale already used elsewhere in the codebase
# for a "typical" iSCAT point-spread function in pixel units (e.g.
# `pipeline/investigator.py`'s `fake_sigma = 1.5`) — there is no single
# canonical default sigma for a generic Gaussian PSF, so this value is chosen
# for consistency with that call site rather than derived from a fit.
_DEFAULT_SIGMA: float = 1.5


class GaussianPsfModel:
    """A 2-D Gaussian PSF model — the simplest :class:`PsfInterface`
    implementation, with no spline/Rust backends.
    """

    def __init__(self, sigma: float = _DEFAULT_SIGMA) -> None:
        self.sigma = sigma

    # Properties

    @property
    def psf_params(self) -> dict[str, float]:
        """Return the Gaussian's shape parameter."""
        return {"sigma": self.sigma}

    # PSF generation

    def get_psf_function(self) -> Callable[..., npt.NDArray[np.floating]]:
        """Return a callable ``f(x, y, c=1.0) -> image``.

        Uses the analytic 2-D Gaussian ``c * exp(-(x^2 + y^2) / (2 sigma^2))``
        — contrast is applied as a direct multiplicative factor, not negated
        (unlike :class:`tandemmp.fitting.psf.PsfModel`'s spline backend).
        """
        sigma = self.sigma

        def _psf(
            x: npt.NDArray[np.floating],
            y: npt.NDArray[np.floating],
            c: float = 1.0,
        ) -> npt.NDArray[np.floating]:
            return np.asarray(c * np.exp(-(x**2 + y**2) / (2.0 * sigma**2)))

        return _psf

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

    def render(
        self,
        dx: float,
        dy: float,
        c: float,
        size: int,
    ) -> npt.NDArray[np.floating]:
        """Render a PSF image at *(dx, dy)* with contrast *c*.

        No spline/Rust backends exist for this model — always uses
        :meth:`psf_2d_offset`.
        """
        return self.psf_2d_offset(size, dx, dy, c)

    # Peak fitting

    def fit_peak(
        self,
        peak: npt.NDArray[np.floating],
        **kwargs: Any,  # noqa: ANN401
    ) -> PeakFitResult:
        """Fit this Gaussian PSF to a 2-D peak thumbnail.

        Delegates entirely to
        :func:`tandemmp.fitting.psf_interface.fit_peak_generic` — there is
        no Rust fast-path for this model.
        """
        return fit_peak_generic(self, peak, **kwargs)

    def fit_peak_exclusion(
        self,
        peak: npt.NDArray[np.floating],
        exclusion: npt.NDArray[np.bool_],
        **kwargs: Any,  # noqa: ANN401
    ) -> PeakFitResult:
        """Fit this Gaussian PSF to *peak*, ignoring pixels where
        *exclusion* is True.
        """
        return fit_peak_exclusion_generic(self, peak, exclusion, **kwargs)
