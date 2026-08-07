"""Ratiometric event — produced by ratiometric iSCAT processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from tandemmp.core.classifications import (
    Classification,
    EvClassification,
    ProcessingMethod,
)
from tandemmp.core.conversions import c_to_mass
from tandemmp.events._base import AbstractEvent, PeakImage, _allocate_event_id
from tandemmp.events.subtraction import SubtractionSeries
from tandemmp.fitting.psf_interface import PsfInterface

if TYPE_CHECKING:
    from tandemmp.pipeline.investigator import MultiFitResult
    from tandemmp.pipeline.loader import OverlapArea


def _as_frame_list(frames: list[int] | int) -> list[int]:
    """Normalise an :class:`OverlapArea`'s ``frames`` field to a list."""
    return frames if isinstance(frames, list) else [frames]


class RatiometricEvent(AbstractEvent):
    """Event arising from ratiometric processing.

    Tracks overlaps with other ratiometric events, supports 3-D fitting
    via :meth:`apply_3D_fit`, and provides exclusion-zone computation.
    """

    def __init__(
        self,
        peak: PeakImage,
        *,
        fit: bool = True,
        navg: int = 5,
        peak_model: PsfInterface | None = None,
    ) -> None:
        if peak.method != ProcessingMethod.ratiometric:
            msg = f"Expected ProcessingMethod.ratiometric, got {peak.method!r}"
            raise ValueError(msg)

        super().__init__(peak, fit=fit, peak_model=peak_model)

        self.navg: int = navg
        self.overlaps: list[OverlapArea] = []
        self.classification: Classification | None = None
        self._ID: int = _allocate_event_id()
        self._first_attempt: SubtractionSeries | None = None
        self._corrections_attempts: list[MultiFitResult] = []
        self.following_threads: list[int | str] = []
        self.particle: int | str | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RatiometricEvent):
            return NotImplemented
        return self._ID == other._ID

    def __hash__(self) -> int:
        return hash(self._ID)

    # -- Classification ------------------------------------------------------

    def self_classify(self) -> None:
        """Set classification based on the sign of the contrast."""
        self.classification = (
            EvClassification.Binding
            if np.sign(self.c) == -1
            else EvClassification.Unbinding
        )
        self.c = -abs(self.c)
        self.m = float(c_to_mass(self.c))

    # -- 3-D fitting ---------------------------------------------------------

    def apply_3D_fit(self, fit_result: MultiFitResult) -> None:  # noqa: N802
        """Apply a 3-D ratiometric fit result to this event."""
        if self.has_connection:
            self.disconnect_all()

        self.fit_3D = True
        self.fit_3D_error: float = fit_result.error
        self.classification = fit_result.classification
        self.connect(fit_result.series)

        sel = fit_result.classification.get_slice(self.navg, remove=1)
        self.dx = fit_result.series.dx[sel]
        self.dy = fit_result.series.dy[sel]
        self.smooth_dx = fit_result.series.smooth_dx[sel]
        self.smooth_dy = fit_result.series.smooth_dy[sel]
        self.x = fit_result.series.x[sel]
        self.y = fit_result.series.y[sel]
        self.c = fit_result.c

        if len(fit_result.c) > 2:
            self.rel_c = np.array(fit_result.c)[sel]  # type: ignore[assignment]
        else:
            self.rel_c = np.array(fit_result.c)  # type: ignore[assignment]

        self.m = c_to_mass(self.c)  # type: ignore[assignment]
        self.rel_m = c_to_mass(np.asarray(self.rel_c))  # type: ignore[assignment]

        index = np.arange(len(fit_result.series.events))
        bad = set(
            index[fit_result.classification.get_slice(self.navg, remove=0)]
        ) ^ set(index)

        for ev in fit_result.series[list(bad)]:
            ev._is_bad = True

        # fit_result.fit_image is only None if MultiFitResult is constructed
        # without a resolved fit; find_and_fit (the only real constructor)
        # always supplies best_image, so this is never None in practice.
        self.fit_image: npt.NDArray[np.floating] = fit_result.fit_image  # type: ignore[assignment]

    # -- Disconnect ----------------------------------------------------------

    def disconnect_all(self) -> None:
        """Reset 3-D fit state and restore basic-fit attributes."""
        super().disconnect_all()
        self.fit_3D = False
        self.fit_3D_error = None  # type: ignore[assignment]
        self.classification = None
        self.c = self._basic_fit.contrast
        self.m = float(c_to_mass(self.c))
        self.dx = self._basic_fit.dx
        self.dy = self._basic_fit.dy
        self.smooth_dx = None  # type: ignore[assignment]
        self.smooth_dy = None  # type: ignore[assignment]
        self.x = self.peak.midpoint[0] + self.dx
        self.y = self.peak.midpoint[1] + self.dy

    # -- Properties ----------------------------------------------------------

    @property
    def frame_range(self) -> set[int]:
        """Set of frame indices covered by this event."""
        return set(range(self.i - self.navg, self.i + self.navg + 1))

    @property
    def particle_c(self) -> float:
        """Contrast value for particle tracking."""
        return self.c if isinstance(self.c, float) else self.c[-1]

    @property
    def particle_m(self) -> float:
        """Mass value for particle tracking."""
        return float(c_to_mass(self.particle_c))

    @property
    def particle_x(self) -> float:
        """X position for particle tracking."""
        return self.x if isinstance(self.x, float) else self.x[-1]

    @property
    def particle_y(self) -> float:
        """Y position for particle tracking."""
        return self.y if isinstance(self.y, float) else self.y[-1]

    @property
    def particle_i(self) -> int:
        """Frame index adjusted for classification direction."""
        if self.classification == EvClassification.Binding:
            return self.i + 1
        if self.classification == EvClassification.Unbinding:
            return self.i - 1
        return self.i

    @property
    def temporal_overlaps(self) -> list[OverlapArea]:
        """Overlaps with a temporal component."""
        return [i for i in self.overlaps if i.temporal]

    @property
    def exclusion_zone(self) -> npt.NDArray[np.bool_]:
        """Boolean exclusion mask shaped ``(n_frames, H, W)``."""
        ev_frames = np.arange(self.i - self.navg, self.i + self.navg + 1)
        overlaps = self.temporal_overlaps
        exclude = np.full((len(ev_frames), *self.peak.image.shape), False)
        mid = exclude.shape[1] // 2

        for overlap in overlaps:
            frames = _as_frame_list(overlap.frames)
            sel = np.array([i in frames for i in ev_frames])
            # Mixed ndarray/slice fancy indexing is valid at runtime but not
            # expressible in numpy's stubs as a single tuple type.
            exclude[(sel, *overlap.area)] = True  # type: ignore[index]

        exclude[:, mid - 2 : mid + 3, mid - 3 : mid + 4] = False
        exclude[:, mid - 3 : mid + 4, mid - 2 : mid + 3] = False
        return exclude

    def recalculate_exclusion_zone(
        self,
        overlaps: list[OverlapArea],
    ) -> npt.NDArray[np.bool_]:
        """Recompute the exclusion zone from a given list of overlaps."""
        ev_frames = np.arange(self.i - self.navg, self.i + self.navg + 1)
        exclude = np.full((len(ev_frames), *self.peak.image.shape), False)

        for overlap in overlaps:
            frames = _as_frame_list(overlap.frames)
            sel = np.array([i in frames for i in ev_frames])
            exclude[(sel, *overlap.area)] = True  # type: ignore[index]

        bad = np.sum(exclude, axis=(1, 2)) / np.prod(exclude.shape[1:])
        too_much = bad > 0.5

        if np.mean(too_much) > 0.5:
            exclude = np.full_like(exclude, fill_value=False, dtype=bool)

        return exclude

    @property
    def thread_intersections(self) -> list[OverlapArea]:
        """Overlaps sorted by frame number."""

        def key(e: OverlapArea) -> int:
            return e.frames if isinstance(e.frames, int) else e.frames[-1]

        return sorted(self.overlaps, key=key)
