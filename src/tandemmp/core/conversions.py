"""Unit conversions for TandemMP (spatial, mass/contrast, precision).

The active contrast/mass calibration (:data:`_active_calibration`) is
process-wide mutable state, read by :func:`c_to_mass`/:func:`mass_to_c` on
every call rather than being baked into them at import time. This is not
thread-safe — it assumes a single writer per process, matching the
persistent worker-pool precedent in :mod:`tandemmp.core.parallel`. It is
safe under :func:`tandemmp.pipeline.batch.batch_process`'s persistent pool
specifically because every dispatched call sets the calibration explicitly
before use (including resetting to the default when none is given), so a
reused worker process never carries over a stale calibration from a
previous call.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

# Type alias for values that can be scalar or array
Numeric = float | npt.NDArray[np.floating]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NM_PER_PIXEL: float = 72.6
"""Nanometres per pixel for the standard iSCAT camera."""


@dataclasses.dataclass(frozen=True)
class Calibration:
    """Linear contrast-to-mass calibration: ``mass = (c - offset) / slope``."""

    slope: float
    offset: float


DEFAULT_CALIBRATION = Calibration(slope=-0.000030, offset=-0.000070)
"""Open-iSCAT default calibration (linear contrast-to-mass)."""

_active_calibration: Calibration = DEFAULT_CALIBRATION


def set_calibration(calibration: Calibration | None) -> None:
    """Set the process-wide active calibration.

    Passing ``None`` resets to :data:`DEFAULT_CALIBRATION`.
    """
    global _active_calibration
    _active_calibration = (
        calibration if calibration is not None else DEFAULT_CALIBRATION
    )


def get_calibration() -> Calibration:
    """Return the process-wide active calibration."""
    return _active_calibration


# ---------------------------------------------------------------------------
# Spatial conversions
# ---------------------------------------------------------------------------


def nm2pixels(nm: Numeric) -> Numeric:
    """Convert nanometres to pixels."""
    return nm / NM_PER_PIXEL


def pixels2nm(px: Numeric) -> Numeric:
    """Convert pixels to nanometres."""
    return np.multiply(px, NM_PER_PIXEL)


# ---------------------------------------------------------------------------
# Mass / contrast conversions (open-iSCAT defaults)
# ---------------------------------------------------------------------------


def c_to_mass(c: Numeric) -> Numeric:
    """Convert iSCAT contrast to mass (kDa), using the active calibration."""
    cal = _active_calibration
    return np.divide(np.subtract(c, cal.offset), cal.slope)  # type: ignore[no-any-return]


def mass_to_c(m: Numeric) -> Numeric:
    """Convert mass (kDa) to iSCAT contrast, using the active calibration."""
    cal = _active_calibration
    return np.add(np.multiply(m, cal.slope), cal.offset)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Precision / size
# ---------------------------------------------------------------------------


def mass_to_sigma(mass: float) -> float:
    """Estimate positional precision (in nanometres) for a given mass (kDa).

    Precision is capped at 4 px for very light particles. If the physical
    radius exceeds the precision estimate, the radius is returned instead —
    with these coefficients this is always the case for heavy particles, so
    there is no separate lower clamp.
    """
    a, b = 5.6254382094058055, 0.00572794123917841
    sigma = a / (abs(mass) * b)  # in nm

    size = 0.6658267737663075 * abs(mass) ** (1 / 3)  # in nm

    max_sigma = pixels2nm(4.0)
    if sigma > max_sigma:
        sigma = float(max_sigma)

    return float(size) if size > sigma else sigma


def mass_to_nm_radius(m: float) -> float:
    """Estimate protein radius in nanometres (Erickson 2009)."""
    return float(0.066 * (m * 1000) ** (1 / 3))


def mass_to_px_radius(m: float) -> float:
    """Estimate protein radius in pixels."""
    return float(nm2pixels(mass_to_nm_radius(m)))
