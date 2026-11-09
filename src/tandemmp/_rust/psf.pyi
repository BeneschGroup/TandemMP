"""Type stub for the ``tandemmp._rust.psf`` submodule.

Source: ``src/tandemmp/_rust/psf/src/python.rs``, registered as a submodule
by ``tandemmp._rust._native``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

class SplinePsf:
    """Tensor-product B-spline PSF, ``RectBivariateSpline``-parity."""

    def __init__(
        self,
        tx: npt.NDArray[np.float64],
        ty: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        kx: int,
        ky: int,
    ) -> None: ...
    def render(
        self, dx: float, dy: float, c: float, l_thum: int
    ) -> npt.NDArray[np.float64]:
        """Render an ``l_thum x l_thum`` PSF image at offset ``(dx, dy)``."""

    def fit(
        self,
        thumb: npt.NDArray[np.float64],
        init: tuple[float, float, float],
        allow_offset: bool = ...,
        max_iter: int = ...,
        tol: float = ...,
    ) -> tuple[float, float, float, float, bool]:
        """Fit ``(c, dx, dy)`` to *thumb*; returns ``(c, dx, dy, error, converged)``."""

    def fit_batch(
        self,
        thumbs: npt.NDArray[np.float64],
        inits: npt.NDArray[np.float64],
        allow_offset: bool = ...,
        max_iter: int = ...,
        tol: float = ...,
    ) -> npt.NDArray[np.float64]:
        """Batch-fit ``thumbs[n, h, w]`` against ``inits[n, 3]``; returns ``[n, 5]``."""
