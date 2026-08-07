"""Shared NumPy array type aliases used across `src/tandemmp`.

Consolidates `FloatArray`/`BoolArray`/`IntArray` aliases that were
previously redefined independently (and inconsistently) in
`pipeline/simple_analysis.py`, `pipeline/pca.py`, `pipeline/loader.py`,
`drift/corrector.py`, and `tracking/metrics.py`. `FloatArray`/`IntArray`
use an explicit `[Any]` type argument (rather than a bare
`npt.NDArray[np.floating]`/`npt.NDArray[np.integer]`) so annotations using
these aliases satisfy `mypy --strict`'s `type-arg` check; no code in
`src/` depends on a specific float/int bit width (e.g. `float32` is never
used), so this is a static-typing-only widening with no behaviour change.

Also consolidates `UInt16Array`, previously redefined identically in
`pipeline/pca.py` and `pipeline/loader.py` — deliberately kept as a
concrete `np.uint16` (not `[Any]`-parameterised) since it specifically
types raw movie frame data, which genuinely is stored as `uint16`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.integer[Any]]
UInt16Array = npt.NDArray[np.uint16]
