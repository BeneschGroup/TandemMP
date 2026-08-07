"""Ghost event — inferred particle that was not directly observed."""

from __future__ import annotations

from typing import Any

from tandemmp.core.classifications import EvClassification, ProcessingMethod
from tandemmp.core.conversions import c_to_mass
from tandemmp.events._base import AbstractEvent, PeakImage, _allocate_event_id


class GhostEvent(AbstractEvent):
    """Event for an inferred (ghost) particle.

    Created from a known PSF fit rather than a direct observation.
    Uses the shared ratiometric ID counter.
    """

    def __init__(
        self,
        peak: PeakImage,
        peak_fit: tuple[float, float, float],
    ) -> None:
        if peak.method != ProcessingMethod.inferred:
            msg = f"Expected ProcessingMethod.inferred, got {peak.method!r}"
            raise ValueError(msg)

        super().__init__(peak=peak, fit=False)

        dx, dy, c = peak_fit
        self._ID: int = _allocate_event_id()
        self.classification = EvClassification.Binding
        self.dx: float = dx
        self.dy: float = dy
        self.x: float = self.peak.midpoint[0] + dx
        self.y: float = self.peak.midpoint[1] + dy
        self.c: float = c
        self.particle_c: float = c
        self.particle_i: int = self.i
        self.m: float = float(c_to_mass(c))
        self.particle_m: float = self.m
        self.basic_fit_error: float | None = None  # type: ignore[assignment]
        self._basic_fit: Any = None
