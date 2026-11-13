"""ParticleManager: a collection of tracked particles.

Aggregates :class:`Particle` and :class:`GhostParticle` instances and supports the
spatial/temporal queries needed by pair matching and ghost detection.

Rust acceleration candidate: :meth:`ParticleManager.find_best_matches` and the
per-frame spatial filters use O(n^2) Python loops.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import numpy as np

from tandemmp.events import AbstractEvent
from tandemmp.tracking.particle import GhostParticle, Particle

logger = logging.getLogger(__name__)


class ParticleManager:
    """Container for the :class:`Particle` instances tracked in one movie.

    It owns the particle list and serves the investigator's queries (present
    particles, close-particle search, thumbnail-based spatial filtering,
    nearest-neighbour ranking).

    Ghost particles are also accepted via :meth:`append`; adding one flags nearby
    particles via :attr:`Particle.split_mass` so they can accommodate mass that may
    have left as the ghost.
    """

    GHOST_SPLIT_RADIUS: float = 0.5

    def __init__(self, particles: Iterable[Particle] | None = None) -> None:
        self._particles: list[Particle] = list(particles) if particles else []

    # Container protocol

    def __iter__(self) -> Iterator[Particle]:
        return iter(self._particles)

    def __len__(self) -> int:
        return len(self._particles)

    def __contains__(self, particle: object) -> bool:
        return particle in self._particles

    def __repr__(self) -> str:
        return f"ParticleManager({len(self._particles)} particles)"

    def get_all(self) -> list[Particle]:
        """Return a shallow copy of the managed particle list."""
        return list(self._particles)

    # Mutation

    def append(self, particle: Particle) -> None:
        """Add *particle* to the manager.

        For a :class:`GhostParticle`, nearby particles at its binding frame are marked
        as possible mass-split sources, making their mass-consistency checks more
        permissive.
        """
        if not isinstance(particle, Particle):
            msg = f"Expected Particle, got {type(particle).__name__}"
            raise TypeError(msg)
        if particle in self._particles:
            msg = f"Particle {particle.ID} is already managed"
            raise ValueError(msg)

        if isinstance(particle, GhostParticle):
            self._mark_possible_split(particle)

        self._particles.append(particle)

    def _mark_possible_split(self, ghost: GhostParticle) -> None:
        ghost_event = ghost.binding_event
        nearby = self.get_close_particles(
            frame=ghost_event.i,
            x=float(ghost_event.x),
            y=float(ghost_event.y),
            distance=self.GHOST_SPLIT_RADIUS,
        )
        if not nearby:
            return

        logger.info(
            "Ghost particle %s may have split from particles %s",
            ghost.ID,
            [p.ID for p in nearby],
        )
        for p in nearby:
            p.split_mass = (int(ghost_event.i), float(ghost_event.m))
        ghost.is_possible_split = True

    def remove(self, particle: Particle) -> bool:
        """Remove *particle* from the manager.

        Returns ``True`` if removed, ``False`` if not found. Missing particles are
        logged, not raised, so repeat removal during coincidence handling is non-fatal.
        """
        try:
            self._particles.remove(particle)
        except ValueError:
            logger.warning("Particle %s not found in manager", particle.ID)
            return False
        return True

    # Frame queries

    def get_present_particles(self, frame: int) -> list[Particle]:
        """Return all particles that were bound at *frame*."""
        return [p for p in self._particles if p.was_bound(frame)]

    def get_close_particles(
        self,
        frame: int,
        x: float,
        y: float,
        distance: float,
    ) -> list[Particle]:
        """Return particles within *distance* pixels of ``(x, y)`` at *frame*."""
        close: list[Particle] = []
        for particle in self.get_present_particles(frame):
            entry = particle[frame]
            if entry is None:
                continue
            if float(np.hypot(entry.x - x, entry.y - y)) < distance:
                close.append(particle)
        return close

    # Thumbnail-based spatial filters

    def background_particle_check(
        self,
        event: AbstractEvent,
        frame: int,
        l_thum: int,
        psf_radius: float,
    ) -> list[Particle]:
        """Particles inside the thumbnail around *event*'s peak midpoint at *frame*.

        The thumbnail is expanded by *psf_radius*.
        """
        midpoint = event.peak.midpoint
        distance = (l_thum / 2) + psf_radius
        return self._particles_in_box(frame, midpoint, distance)

    def particles_in_thumbnail(
        self,
        event: AbstractEvent,
        l_thum: int,
        psf_radius: float = 7.0,
    ) -> list[list[Particle]]:
        """Per frame of *event*'s averaging window, the particles in its thumbnail."""
        cx, cy = event.peak.midpoint
        distance = (l_thum / 2) + psf_radius
        frame_range = range(event.i - event.navg, event.i + event.navg + 1)

        per_frame: list[list[Particle]] = [[] for _ in frame_range]
        for particle in self._particles:
            for idx, f in enumerate(frame_range):
                if not particle.was_bound(f):
                    continue
                entry = particle[f]
                if entry is None:
                    continue
                if abs(entry.x - cx) < distance and abs(entry.y - cy) < distance:
                    per_frame[idx].append(particle)
        return per_frame

    def _particles_in_box(
        self,
        frame: int,
        midpoint: tuple[int, int],
        distance: float,
    ) -> list[Particle]:
        cx, cy = midpoint
        suitable: list[Particle] = []
        for particle in self.get_present_particles(frame):
            entry = particle[frame]
            if entry is None:
                continue
            if abs(entry.x - cx) < distance and abs(entry.y - cy) < distance:
                suitable.append(particle)
        return suitable

    # Match ranking

    def find_best_matches(
        self,
        x: float,
        y: float,
        i: int,
        c: float,
    ) -> list[Particle]:
        """Particles at frame *i*, ranked by ``(x, y, c)`` proximity to the query.

        Particles with no entry at that frame are dropped; the returned list is aligned
        with those that remain.
        """
        present = self.get_present_particles(i)
        with_entry: list[tuple[Particle, float]] = []
        for particle in present:
            entry = particle[i]
            if entry is None:
                continue
            distance = float(
                ((entry.x - x) ** 2 + (entry.y - y) ** 2 + (entry.c - c) ** 2) ** 0.5
            )
            with_entry.append((particle, distance))

        with_entry.sort(key=lambda item: item[1])
        return [particle for particle, _ in with_entry]
