"""Type stub for the compiled ``tandemmp._rust.drift_correction`` extension.

Source: ``src/tandemmp/_rust/drift_correction/src/python.rs``. It sits next to the
gitignored compiled module so mypy and IDEs see real signatures; the extension
itself carries no type information.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def solve(
    input_frame: npt.NDArray[np.float64],
    baseline_frame: npt.NDArray[np.float64],
    bounds: tuple[float, float] = ...,
    seed: int | None = ...,
) -> tuple[float, float]:
    """Find the ``(x, y)`` shift minimising RMSE between two 2-D frames."""

def stabilise(
    frame: npt.NDArray[np.float64], x: float, y: float
) -> npt.NDArray[np.float64]:
    """Shift *frame* by ``(-y, -x)`` to undo a drift of ``(x, y)``."""

def rmse(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Root-mean-square error between two equal-shaped arrays."""

def shift_compare(
    x: float,
    y: float,
    input_frame: npt.NDArray[np.float64],
    baseline_frame: npt.NDArray[np.float64],
) -> float:
    """RMSE between *input_frame* and *baseline_frame* shifted by ``(x, y)``."""
