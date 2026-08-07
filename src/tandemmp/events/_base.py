"""Shared base classes and containers for the events subpackage.

Provides :class:`Correction`, :class:`PeakImage`, :class:`Connectable`,
and :class:`AbstractEvent` — the building blocks from which concrete event
types (ratiometric, subtraction, ghost) are derived.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import numpy.typing as npt

from tandemmp.core.classifications import Classification, ProcessingMethod
from tandemmp.core.conversions import c_to_mass
from tandemmp.fitting.psf import PeakFitResult, get_psf_model
from tandemmp.fitting.psf_interface import PsfInterface

# ---------------------------------------------------------------------------
# Shared event-ID counter
# ---------------------------------------------------------------------------

_next_event_id: int = 0


def _allocate_event_id() -> int:
    """Return the next sequential event ID."""
    global _next_event_id  # noqa: PLW0603
    id_ = _next_event_id
    _next_event_id += 1
    return id_


def _reset_event_ids() -> None:
    """Reset the event-ID counter (for tests only)."""
    global _next_event_id  # noqa: PLW0603
    _next_event_id = 0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Correction:
    """Details of an artificial PSF correction applied to a peak image."""

    sliced_image: npt.NDArray[np.floating] | None
    orig_image: npt.NDArray[np.floating]
    particle: int
    frame_requested: int
    background: bool
    data_from: int | None = None


@dataclasses.dataclass(frozen=True)
class PeakImage:
    """Container for a peak image (2-D array) and its metadata.

    ``midpoint`` corresponds to the image centre, not the peak-fit position.
    """

    image: npt.NDArray[np.floating]
    method: ProcessingMethod
    midpoint: tuple[int, int]
    cutout: list[slice]
    i: int
    original: npt.NDArray[np.floating] | None = None
    background: npt.NDArray[np.floating] | None = None
    corrections: list[Correction] | None = None

    def as_reduced_memory(self) -> PeakImage:
        """Return a copy with the image down-cast to float16."""
        return dataclasses.replace(
            self,
            image=np.asarray(self.image, dtype=np.float16),
        )


# ---------------------------------------------------------------------------
# Connectable mixin
# ---------------------------------------------------------------------------


class Connectable:
    """Mixin that links events to show they correspond to the same entity
    across different post-processing methods.
    """

    def __init__(self) -> None:
        self.connected_events: list[Any] = []
        self.has_connection: bool = False

    def connect(self, event: Connectable, *, do_back: bool = True) -> None:
        """Link *self* to *event* (bidirectionally by default)."""
        if not isinstance(event, Connectable):
            msg = "event must inherit from Connectable"
            raise TypeError(msg)
        if event in self.connected_events:
            msg = "Event already connected"
            raise ValueError(msg)
        self.connected_events.append(event)
        self.has_connection = True
        if do_back:
            event.connect(self, do_back=False)

    def disconnect_all(self) -> None:
        """Recursively disconnect all linked events."""
        self.has_connection = False
        for e in self.connected_events:
            if e.has_connection:
                e.disconnect_all()
        self.connected_events = []


# ---------------------------------------------------------------------------
# Abstract event
# ---------------------------------------------------------------------------


class AbstractEvent(Connectable):
    """Base class for all event types.

    Holds a peak image, optional PSF model, and basic 2-D fit results.
    Concrete event types (ratiometric, subtraction, ghost) inherit from this.
    """

    def __init__(
        self,
        peak: PeakImage,
        fit: bool,
        peak_model: PsfInterface | None = None,
    ) -> None:
        super().__init__()
        self.decision: Any | None = None
        self.navg: int = 0
        self.peak: PeakImage = peak
        self.classification: Classification | None = None
        self.fit_3D: bool = False

        self.peak_model: PsfInterface = (
            get_psf_model() if peak_model is None else peak_model
        )

        if fit:
            self.basic_fit()
        else:
            self.c: float = 0.0

    # -- Properties ----------------------------------------------------------

    @property
    def i(self) -> int:
        """Frame index of this event."""
        return self.peak.i

    def __repr__(self) -> str:
        try:
            return f"{self._basic_fit}, {self.classification}"
        except AttributeError:
            return "Not fitted yet"

    # -- Fitting -------------------------------------------------------------

    def _underlying_basic_fit(
        self,
        method: str = "BFGS",
        *,
        allow_offset: bool = False,
        gtol: float = 1e-6,
    ) -> None:
        """Run a basic 2-D PSF fit and store the results on *self*."""
        c_guess = float(np.min(self.peak.image))

        fit_result: PeakFitResult = self.peak_model.fit_peak(
            self.peak.image,
            method=method,
            c_guess=c_guess,
            allow_offset=allow_offset,
            gtol=gtol,
        )

        self.dx: float | npt.NDArray[np.floating] = fit_result.dx
        self.dy: float | npt.NDArray[np.floating] = fit_result.dy
        self.x: float | npt.NDArray[np.floating] = self.peak.midpoint[0] + fit_result.dx
        self.y: float | npt.NDArray[np.floating] = self.peak.midpoint[1] + fit_result.dy
        self.c = fit_result.contrast
        self.m: float = float(c_to_mass(self.c))
        self.basic_fit_error: float = fit_result.error
        self._basic_fit: PeakFitResult = fit_result
        self.rel_m: list[float] = [self.m]
        self.rel_c: list[float] = [self.c]

    def basic_fit(
        self,
        method: str = "BFGS",
        *,
        allow_offset: bool = False,
        gtol: float = 1e-6,
    ) -> None:
        """Perform (or redo) the basic 2-D PSF fit."""
        self._underlying_basic_fit(method, allow_offset=allow_offset, gtol=gtol)

    # -- SNR / noise ---------------------------------------------------------

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio of the peak fit."""
        if self.fit_3D:
            fit, residual = self._3D_replicate_fit()
        else:
            fit, residual = self._basic_replicate_fit()
        noise = float(np.sum(np.abs(residual - np.mean(residual))))
        signal = float(np.sum(np.abs(fit)))
        return signal / noise if noise != 0 else 0.0

    @property
    def noise(self) -> float:
        """Standard deviation of the residual from the peak fit."""
        if self.fit_3D:
            _, residual = self._3D_replicate_fit()
        else:
            _, residual = self._basic_replicate_fit()
        return float(np.std(np.abs(residual - np.mean(residual))))

    def _basic_replicate_fit(
        self,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Replicate the basic 2-D fit and return ``(fit_image, residual)``."""
        peak = self.peak.image
        width = peak.shape[0]
        replicate_fit_im = self.peak_model.psf_2d_offset(
            width,
            self._basic_fit.dx,
            self._basic_fit.dy,
            self._basic_fit.contrast,
        )
        residual = peak - replicate_fit_im
        return replicate_fit_im, residual

    def _3D_replicate_fit(  # noqa: N802
        self,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Return ``(fit_image, residual)`` for a 3-D ratiometric fit.

        Subclasses that support 3-D fitting should set ``self.fit_image``.
        """
        peak = self.peak.image
        fit3d = self.fit_image  # type: ignore[attr-defined]
        return fit3d, peak - fit3d
