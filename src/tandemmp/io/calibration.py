"""Load contrast-to-mass calibrations from files.

Supported files, all giving ``mass = (contrast - offset) / slope``:

- ``.mc`` (DiscoverMP calibration, JSON): ``gradient`` and ``offset``.
- ``.json``: ``c2kDa`` and ``intercept``.
- ``.pkl`` / ``.pickle``: a dict with ``c2kDa`` and ``intercept``. Pickles can run
  code on load, so only open files you trust.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

from tandemmp.core.conversions import Calibration

__all__ = ["load_calibration", "resolve_calibration"]

_SLOPE_KEYS = ("gradient", "c2kDa")
_OFFSET_KEYS = ("offset", "intercept")


def load_calibration(path: str | Path) -> Calibration:
    """Read a :class:`~tandemmp.core.conversions.Calibration` from *path*.

    Raises:
        ValueError: If the suffix is not supported, the slope and offset are
            missing or not finite, or the slope is zero.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".mc", ".json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".pkl", ".pickle"}:
        payload = pickle.loads(path.read_bytes())  # noqa: S301
    else:
        msg = f"{path.name}: expected a .mc, .json, .pkl or .pickle calibration file"
        raise ValueError(msg)
    return _to_calibration(payload, path)


def resolve_calibration(
    calibration: Calibration | str | Path | None,
) -> Calibration | None:
    """Return *calibration* as is, loading it first when it is a file path."""
    if isinstance(calibration, (str, Path)):
        return load_calibration(calibration)
    return calibration


def _to_calibration(payload: Any, path: Path) -> Calibration:  # noqa: ANN401
    if not isinstance(payload, dict):
        msg = f"{path.name}: calibration file does not hold a mapping"
        raise ValueError(msg)  # noqa: TRY004
    try:
        slope = float(next(payload[k] for k in _SLOPE_KEYS if k in payload))
        offset = float(next(payload[k] for k in _OFFSET_KEYS if k in payload))
    except StopIteration:
        msg = (
            f"{path.name}: needs a slope ({' or '.join(_SLOPE_KEYS)}) and an "
            f"offset ({' or '.join(_OFFSET_KEYS)})"
        )
        raise ValueError(msg) from None
    if not (math.isfinite(slope) and math.isfinite(offset)) or slope == 0:
        msg = f"{path.name}: slope must be non-zero and finite, offset finite"
        raise ValueError(msg)
    return Calibration(slope=slope, offset=offset)
