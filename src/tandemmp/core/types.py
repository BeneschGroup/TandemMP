"""Shared NumPy array type aliases used across `src/tandemmp`.

`FloatArray` and `IntArray` accept any floating-point or integer bit width.
`UInt16Array` describes raw movie frames stored as `uint16`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.integer[Any]]
UInt16Array = npt.NDArray[np.uint16]
