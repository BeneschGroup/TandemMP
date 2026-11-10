"""Particle — tracks a single iSCAT particle across events.

Provides :class:`Particle` and :class:`GhostParticle` — the core domain
objects used by :mod:`tandemmp.tracking.manager` to group events into
persistent particles across a movie.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from tandemmp.core.classifications import EvClassification, ProcessingMethod
from tandemmp.core.conversions import mass_to_sigma
from tandemmp.core.functions import closest_index
from tandemmp.events import (
    AbstractEvent,
    GhostEvent,
    RatiometricEvent,
    SubtractionEvent,
)

logger = logging.getLogger(__name__)

# Module-level ID counter, shared across all Particle/GhostParticle instances.
_next_particle_id: int = 0


def _allocate_particle_id() -> int:
    global _next_particle_id  # noqa: PLW0603
    id_ = _next_particle_id
    _next_particle_id += 1
    return id_


def _reset_particle_ids() -> None:
    """Reset the module-level particle ID counter (tests only)."""
    global _next_particle_id  # noqa: PLW0603
    _next_particle_id = 0


class MassChangeWarning(Warning):
    """Raised via :func:`warnings.warn` on an unexplainable mass change."""


# HistoryEntry


@dataclasses.dataclass(frozen=True)
class HistoryEntry:
    """Snapshot of a particle at a single frame."""

    i: int
    x: float
    y: float
    m: float
    c: float
    sigma: float | None
    data_from: int


# EventMap


class EventMap:
    """Three frame-indexed dictionaries of events for a single particle.

    Stores ratiometric, connected (subtraction events attached to a
    ratiometric series), and disconnected (standalone subtraction) events
    separately so the most recent event preceding a given frame can be
    retrieved for each category independently.
    """

    __slots__ = ("ratiometric", "connected", "disconnected")

    def __init__(self) -> None:
        self.ratiometric: dict[int, AbstractEvent] = {}
        self.connected: dict[int, SubtractionEvent] = {}
        self.disconnected: dict[int, SubtractionEvent] = {}

    def __repr__(self) -> str:
        return (
            f"Ratiometric {len(self.ratiometric)}\n"
            f"Connected {len(self.connected)}\n"
            f"Disconnected {len(self.disconnected)}"
        )

    def get_ratiometric(self, frame: int) -> AbstractEvent | None:
        return closest_index(self.ratiometric, frame)

    def get_connected(self, frame: int) -> SubtractionEvent | None:
        return closest_index(self.connected, frame)

    def get_disconnected(self, frame: int) -> SubtractionEvent | None:
        return closest_index(self.disconnected, frame)


# Particle


class Particle:
    """A tracked iSCAT particle built from a binding ratiometric event.

    Events are added via :meth:`propose_new_event`; on-particle positions,
    masses and trajectories are derived from the accumulated event map.
    """

    def __init__(
        self,
        event: RatiometricEvent,
        *,
        overwrite_id: int | str | None = None,
    ) -> None:
        if event.classification != EvClassification.Binding:
            msg = "Particle must be initialised from a Binding-classified event"
            raise ValueError(msg)

        self.ID: int | str = (
            _allocate_particle_id() if overwrite_id is None else overwrite_id
        )
        self.possible_ghost: bool = False
        self._init_common_state(event)
        self.preferred_background: int = event.i - 3
        self.binding_event: RatiometricEvent = event
        self._last_used_series: Any | None = (
            event.connected_events[0] if event.has_connection else None
        )
        self._add_event(event)

    def _init_common_state(self, event: AbstractEvent) -> None:
        """Initialise attributes shared by Particle and GhostParticle."""
        self.is_possible_split: bool = False
        self.split_mass: tuple[int, float] | None = None
        self.lost_tracking: bool = False
        self.strikes: int = 0
        self._event_map: EventMap = EventMap()
        self.rejected: list[AbstractEvent] = []
        self.binding_frame: int = event.i
        self.unbinding_frame: int | None = None
        self._last_used_series = None

    # Identity

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Particle):
            return self.ID == other.ID
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.ID)

    def __repr__(self) -> str:
        return f"Particle(ID={self.ID})\n{self._event_map!r}"

    # Tracking state

    @property
    def on_slide(self) -> bool:
        """True while the particle is still being tracked."""
        return self.unbinding_frame is None

    def was_bound(self, frame: int) -> bool:
        bound_before = self.binding_frame < frame
        if self.on_slide:
            return bound_before
        if self.unbinding_frame is None:
            raise ValueError("Particle off slide but unbinding_frame is None")
        return bound_before and self.unbinding_frame >= frame

    @property
    def residency_time(self) -> float:
        if self.unbinding_frame is None:
            return float("nan")
        return float(self.unbinding_frame - self.binding_frame)

    # Indexing

    def __getitem__(self, frame: int) -> HistoryEntry | None:
        if not self.was_bound(frame):
            return None

        event = self.get_best_event(frame)
        if event is None:
            return None

        # If the chosen event returns non-scalar x at this frame (arrays from
        # a 3-D ratiometric fit), try stepping past the averaging window.
        if not isinstance(event.x, float):
            if isinstance(event, RatiometricEvent) and event.has_connection:
                later = self.get_best_event(frame + 2)
                if later is not None:
                    event = later

        if (
            isinstance(event, RatiometricEvent)
            and event.has_connection
            and event.classification is not None
        ):
            frame_range = np.arange(event.i - event.navg, event.i + event.navg + 1)[
                event.classification.get_slice(event.navg, remove=1)
            ]
            matches = np.where(frame_range == frame)[0]
            if len(matches) == 0:
                return None
            index = int(matches[0])
            return HistoryEntry(
                i=frame,
                x=float(event.x[index]),  # type: ignore[index]
                y=float(event.y[index]),  # type: ignore[index]
                m=float(event.m[index]),  # type: ignore[index]
                c=float(event.c[index]),  # type: ignore[index]
                sigma=mass_to_sigma(float(event.m[index])),  # type: ignore[index]
                data_from=event.i,
            )

        return HistoryEntry(
            i=frame,
            x=float(event.x),
            y=float(event.y),
            m=float(event.m),
            c=float(event.c),
            sigma=mass_to_sigma(float(event.particle_m)),  # type: ignore[attr-defined]
            data_from=event.i,
        )

    # Event bookkeeping

    def _add_event(self, event: AbstractEvent) -> None:
        method = event.peak.method
        if method in (ProcessingMethod.ratiometric, ProcessingMethod.inferred):
            self._event_map.ratiometric[event.i] = event
            if isinstance(event, RatiometricEvent):
                event.particle = self.ID

                if event.has_connection and event.classification is not None:
                    self._last_used_series = event.connected_events[0]
                    sel = event.classification.get_slice(event.navg, remove=0)
                    series = event.connected_events[0][sel]

                    ml_score = event.connected_events[0].ghost_signal_ML
                    if isinstance(ml_score, float) and ml_score > 0.5:
                        self.possible_ghost = True

                    for start_event in series:
                        if not start_event._is_bad:
                            self._event_map.connected[start_event.i] = start_event
        elif method == ProcessingMethod.subtraction:
            if not isinstance(event, SubtractionEvent):
                raise TypeError(
                    "Expected SubtractionEvent for subtraction method, "
                    f"got {type(event).__name__}"
                )
            event.reduce_memory()
            self._event_map.disconnected[event.i] = event
        else:
            msg = f"Unsupported ProcessingMethod: {method!r}"
            raise ValueError(msg)

    # Background choice

    def choose_background(
        self,
        frame: int,
        midpoint: tuple[int, int],
        l_thum: int,
        psf_radius: int,
    ) -> int:
        """Return the last frame where this particle was spatially clear of
        the thumbnail centred at *midpoint*.
        """
        (x, y), i = self.trajectory
        dx = x - midpoint[0]
        dy = y - midpoint[1]
        min_dist = (l_thum // 2) + psf_radius
        before = i < frame
        suitable = (dx > min_dist) & (dy > min_dist) & before
        if not suitable.any():
            return self.preferred_background
        # TODO: bug — should be argwhere(suitable) not argwhere(before); fix alongside
        # test_returns_latest_suitable_not_latest_before once confirmed safe.
        return int(i[np.argwhere(before).max()])

    # Event lookup

    def get_best_event(
        self,
        frame: int,
        *,
        disconnected_only: bool = False,
    ) -> AbstractEvent | None:
        """Return the most informative event available at *frame*."""
        if not self.was_bound(frame):
            return None

        closest_ratio = self.binding_event
        closest_connected = self._event_map.get_connected(frame)
        closest_disconnected = self._event_map.get_disconnected(frame)

        ratio_viable = False
        if isinstance(closest_ratio, GhostEvent):
            ratio_viable = frame >= closest_ratio.i
        elif isinstance(closest_ratio, RatiometricEvent):
            if closest_ratio.has_connection:
                ratio_viable = frame > closest_ratio.i + closest_ratio.navg
            else:
                ratio_viable = frame > closest_ratio.i

        c_viable = closest_connected is not None
        d_viable = closest_disconnected is not None

        if disconnected_only:
            c_viable = False
            ratio_viable = False

        if d_viable:
            event: AbstractEvent = closest_disconnected  # type: ignore[assignment]
            if c_viable and closest_connected.i >= event.particle_i:  # type: ignore[union-attr, attr-defined]
                event = closest_connected  # type: ignore[assignment]
            return event
        if c_viable:
            return closest_connected
        if ratio_viable:
            return closest_ratio
        if disconnected_only:
            return None
        return closest_ratio

    # Event proposal

    def _propose_ratiometric(
        self,
        event: RatiometricEvent,
        last_known: HistoryEntry,
        minimum_mass: float,
        spatial_threshold: float | None,
        mass_fluctuation: float | None,
    ) -> bool:
        classed_unbinding = event.classification == EvClassification.Unbinding

        proposed_mass = (
            last_known.m - event.particle_m if classed_unbinding else event.particle_m
        )
        indicates_unbound = proposed_mass < minimum_mass
        unbinding_too_large = proposed_mass < -abs(last_known.m * 0.1)

        accept = True

        if unbinding_too_large:
            warnings.warn(
                f"Unexplainable mass change! Last known = {last_known.m} "
                f"Proposed = {proposed_mass}",
                MassChangeWarning,
                stacklevel=2,
            )
            accept = False

        if accept and spatial_threshold is not None and mass_fluctuation is not None:
            if event.classification != EvClassification.Binding:
                updated = self[event.i - event.navg]
                if updated is not None:
                    last_known = updated

            if not isinstance(event.m, float):
                if len(event.m) > 3:
                    peak_index = int(event.m[: event.navg].argmax())
                    event_mass = float(event.m[: event.navg].max())
                    peak_frame = sorted(event.frame_range)[peak_index]
                    updated = self[peak_frame]
                    if updated is not None:
                        last_known = updated
                else:
                    event_mass = float(event.m[0])
            else:
                event_mass = float(event.m)

            mass_change = abs(last_known.m - event_mass)
            signal = 1 / (np.exp(2 - (mass_change / minimum_mass * 3)) + 1)
            relative_mass_change = abs(mass_change / last_known.m)
            mass_consistent = (relative_mass_change * signal) < mass_fluctuation

            if isinstance(event.x, float):
                eventx, eventy = float(event.x), float(event.y)
            else:
                # x and y are always set together (both float or both array)
                # from the same fit result, so y is array-typed here too.
                eventx = float(event.x[0])
                eventy = float(event.y[0])  # type: ignore[index]

            dist = float(np.hypot(last_known.x - eventx, last_known.y - eventy))
            space_consistent = dist < spatial_threshold

            latest_event = self.events[-1]
            if (
                not isinstance(latest_event, GhostEvent)
                and event.has_connection
                and event in [o.event for o in latest_event.temporal_overlaps]  # type: ignore[attr-defined]
            ):
                # A connected event has been through a 3-D fit, so x/y are
                # arrays here despite the wider float | array declared type.
                for oi in sorted(event.frame_range):
                    sample = self[int(oi)]
                    if sample is None:
                        continue
                    sample_dist = float(np.hypot(sample.x - eventx, sample.y - eventy))
                    space_consistent |= sample_dist < spatial_threshold

                    sample_mass_change = abs(sample.m - event_mass)
                    sample_signal = 1 / (
                        np.exp(2 - (sample_mass_change / minimum_mass * 3)) + 1
                    )
                    sample_rel_change = abs(sample_mass_change / sample.m)
                    mass_consistent |= (
                        sample_rel_change * sample_signal
                    ) < mass_fluctuation

            if not (space_consistent and mass_consistent):
                logger.info(
                    "Particle %s rejected ratiometric event %s: "
                    "dist=%.2f, mass_change=%.2fkDa",
                    self.ID,
                    event._ID,
                    dist,
                    mass_change,
                )
                accept = False

        if accept:
            self._add_event(event)
            if indicates_unbound or classed_unbinding:
                self.unbinding_frame = event.i
            return True
        return False

    def _propose_fixed(
        self,
        event: SubtractionEvent,
        last_known: HistoryEntry,
        minimum_mass: float,
        spatial_threshold: float = 2.0,
        mass_fluctuation: float = 0.15,
    ) -> bool:
        if event._is_bad:
            logger.info(
                "Particle %s rejected bad subtraction event (m=%.2fkDa)",
                self.ID,
                event.m,
            )
            self.strikes += 1
            return False

        mass_change = abs(last_known.m - event.m)
        over_minimum = event.m >= minimum_mass
        signal = 1 / (np.exp(2 - (mass_change / minimum_mass * 3)) + 1)
        relative_mass_change = abs(mass_change / last_known.m)
        mass_consistent = (
            relative_mass_change * signal
        ) < mass_fluctuation or mass_change <= minimum_mass
        dist = float(np.hypot(last_known.x - event.x, last_known.y - event.y))
        space_consistent = dist < spatial_threshold

        if self.split_mass is not None:
            split_frame, possible_mass = self.split_mass
            if abs(event.i - split_frame) < 5:
                spl_mass_change = abs((last_known.m - possible_mass) - event.m)
                spl_signal = 1 / (np.exp(2 - (spl_mass_change / minimum_mass * 3)) + 1)
                spl_rel_change = abs(spl_mass_change / (last_known.m - possible_mass))
                spl_consistent = (
                    spl_rel_change * spl_signal
                ) < mass_fluctuation or spl_mass_change <= minimum_mass
                mass_consistent |= spl_consistent

        if mass_consistent and space_consistent and over_minimum:
            self._add_event(event)
            self.strikes = max(self.strikes - 1, 0)
            return True

        self.strikes += 1
        logger.info(
            "Particle %s rejected subtraction event: m=%.2fkDa (particle=%.2fkDa), "
            "rel_change=%.2f%%, dist=%.2f",
            self.ID,
            event.m,
            last_known.m,
            relative_mass_change * 100,
            dist,
        )
        return False

    def propose_new_event(
        self,
        event: AbstractEvent,
        minimum_mass: float,
        spatial_threshold: float | None = None,
        mass_fluctuation: float | None = None,
        max_strikes: int = 6,
        *,
        coincidence_checking: bool = False,
    ) -> bool:
        """Propose *event* for addition to this particle.

        Returns ``True`` if the event was accepted.
        """
        if not self.on_slide and not coincidence_checking:
            return False

        if coincidence_checking:
            if self.unbinding_frame is None:
                raise ValueError(
                    "coincidence_checking requires unbinding_frame to be set"
                )
            last_known = self[self.unbinding_frame]
        else:
            last_known = self[event.i]

        if last_known is None:
            last_known = self[event.i + 1]

        if last_known is None:
            logger.warning(
                "Particle %s can't find event for frame %s (bound on %s)",
                self.ID,
                event.i,
                self.binding_frame,
            )
            return False

        if last_known.m < minimum_mass:
            logger.info(
                "Particle %s mass %.2f below minimum, forcing unbinding",
                self.ID,
                last_known.m,
            )
            self.unbinding_frame = event.i
            self.rejected.append(event)
            return False

        if isinstance(event, RatiometricEvent):
            success = self._propose_ratiometric(
                event, last_known, minimum_mass, spatial_threshold, mass_fluctuation
            )
        elif isinstance(event, SubtractionEvent):
            sub_spatial = spatial_threshold if spatial_threshold is not None else 2.0
            sub_mass = mass_fluctuation if mass_fluctuation is not None else 0.15
            success = self._propose_fixed(
                event, last_known, minimum_mass, sub_spatial, sub_mass
            )
            if self.strikes >= max_strikes:
                if len(self.fixed_background_events) == 0:
                    self.unbinding_frame = event.i
                else:
                    self.unbinding_frame = self.last_continuation.i + 1
                logger.info(
                    "Particle %s reached %d bad fits, removed",
                    self.ID,
                    max_strikes,
                )
                self.lost_tracking = True
                return False
        else:
            msg = f"Unsupported event type: {type(event).__name__}"
            raise TypeError(msg)

        if not success:
            self.rejected.append(event)
            return False
        return True

    # Derived views

    @property
    def fixed_background_events(self) -> list[SubtractionEvent]:
        connected = list(self._event_map.connected.values())
        disconnected = list(self._event_map.disconnected.values())
        return sorted(connected + disconnected, key=lambda e: e.i)

    @property
    def fixed_data(self) -> list[HistoryEntry]:
        return [
            HistoryEntry(
                i=e.i,
                x=float(e.x),
                y=float(e.y),
                m=float(e.m),
                c=float(e.c),
                sigma=mass_to_sigma(float(e.m)),
                data_from=e.i,
            )
            for e in self.fixed_background_events
        ]

    @property
    def last_continuation(self) -> SubtractionEvent:
        return self.fixed_background_events[-1]

    @property
    def trajectory(
        self,
    ) -> tuple[
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
        npt.NDArray[np.int_],
    ]:
        """Return ``((x, y), frames)`` from trusted fixed-background events.

        Falls back to the binding event when no trusted data is available.
        """
        evs = self.fixed_background_events
        if evs:
            trust = np.array([e.snr > 0.2 for e in evs])
            xs = np.array([e.x for e in evs], dtype=float)
            ys = np.array([e.y for e in evs], dtype=float)
            frames = np.array([e.i for e in evs], dtype=int)
            if trust.any():
                return (xs[trust], ys[trust]), frames[trust]

        # after a 3-D fit the binding event's x/y are per-frame arrays; keep
        # every position, each at the binding frame
        b = self.binding_event
        xs = np.atleast_1d(np.asarray(b.x, dtype=float)).ravel()
        ys = np.atleast_1d(np.asarray(b.y, dtype=float)).ravel()
        return (xs, ys), np.full(xs.shape, b.i, dtype=int)

    @property
    def events(self) -> list[AbstractEvent]:
        return list(self._event_map.ratiometric.values())

    # Spatial queries

    def is_close(self, event: AbstractEvent, threshold: float) -> bool:
        """Return whether *event* is within *threshold* pixels of this particle."""
        if not self.was_bound(event.i):
            return False
        entry = self[event.i]
        if entry is None:
            return False
        return bool(np.hypot(event.x - entry.x, event.y - entry.y) < threshold)

    def distance_to(self, x: float, y: float, frame: int) -> float | None:
        """Return the distance from ``(x, y)`` to this particle at *frame*.

        Returns ``None`` if the particle was not bound at *frame*.
        """
        if not self.was_bound(frame):
            return None
        entry = self[frame]
        if entry is None:
            return None
        return float(np.hypot(x - entry.x, y - entry.y))


# GhostParticle


class GhostParticle(Particle):
    """Particle built from an inferred (ghost) event rather than an observation."""

    def __init__(
        self,
        event: GhostEvent,
        *,
        overwrite_id: int | str | None = None,
    ) -> None:
        if event.classification != EvClassification.Binding:
            msg = "GhostParticle must be initialised from a Binding event"
            raise ValueError(msg)

        self.ID: int | str = (
            _allocate_particle_id() if overwrite_id is None else overwrite_id
        )
        self.possible_ghost = True
        self._bad_corrections: int = 0
        self.double_checked: bool = False
        self.self_destruct: bool = False
        self._init_common_state(event)
        self.preferred_background = event.i
        self.binding_event = event  # type: ignore[assignment]
        # (x, y, -c) arguments used to subtract the ghost's own background
        # impression when no prior subtraction event exists.
        self._impression_args: tuple[float, float, float] = (
            float(event.x),
            float(event.y),
            -float(event.c),
        )
        self._add_event(event)

    def was_bound(self, frame: int) -> bool:
        bound_before = self.binding_frame <= frame
        if self.on_slide:
            return bound_before
        if self.unbinding_frame is None:
            raise ValueError("GhostParticle off slide but unbinding_frame is None")
        return bound_before and self.unbinding_frame >= frame

    def _propose_fixed(
        self,
        event: SubtractionEvent,
        last_known: HistoryEntry,
        minimum_mass: float,
        spatial_threshold: float = 2.0,
        mass_fluctuation: float = 0.15,
    ) -> bool:
        # Unchecked first acceptance — only the ghost binding event exists.
        if len(self._event_map.ratiometric) < 2:
            self._add_event(event)
            return True

        mass_change = abs(last_known.m - event.m)
        over_minimum = event.m >= minimum_mass
        signal = 1 / (np.exp(2 - (mass_change / minimum_mass * 3)) + 1)
        relative_mass_change = abs(mass_change / last_known.m)
        mass_consistent = (
            relative_mass_change * signal
        ) < mass_fluctuation or mass_change <= minimum_mass
        dist = float(np.hypot(last_known.x - event.x, last_known.y - event.y))
        space_consistent = dist < spatial_threshold

        if self.split_mass is not None:
            split_frame, possible_mass = self.split_mass
            if abs(event.i - split_frame) < 5:
                spl_mass_change = abs((last_known.m - possible_mass) - event.m)
                spl_signal = 1 / (np.exp(2 - (spl_mass_change / minimum_mass * 3)) + 1)
                spl_rel_change = abs(spl_mass_change / (last_known.m - possible_mass))
                spl_consistent = (
                    spl_rel_change * spl_signal
                ) < mass_fluctuation or spl_mass_change <= minimum_mass
                mass_consistent |= spl_consistent

        background = self[event.subtracted_i]
        if background is not None:
            bx, by = background.x, background.y
        else:
            bx, by, _bc = self._impression_args

        dist_development = float(np.hypot(bx - event.x, by - event.y))
        d_similar = dist_development < (spatial_threshold / 1000)
        if d_similar:
            self.strikes += 1
            logger.info(
                "Ghost particle %s rejected event too close to impression",
                self.ID,
            )
            return False

        if mass_consistent and space_consistent and over_minimum:
            self._add_event(event)
            self.strikes = max(self.strikes - 1, 0)
            return True

        self.strikes += 1
        logger.info(
            "Ghost particle %s rejected subtraction event: "
            "m=%.2fkDa (particle=%.2fkDa), rel_change=%.2f%%, dist=%.2f",
            self.ID,
            event.m,
            last_known.m,
            relative_mass_change * 100,
            dist,
        )
        return False
