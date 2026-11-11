"""Type stub for ``tandemmp._rust.spline_psf`` (``spline_psf/src/python.rs``)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def eval(
    u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Bundled PSF at ``(u, v)`` = (row, col); equal-shape arrays."""

def tck() -> (
    tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int,
        int,
    ]
):
    """The bundled spline as ``(tx, ty, c, kx, ky)``."""
