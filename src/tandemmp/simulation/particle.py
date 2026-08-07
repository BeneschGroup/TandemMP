"""Simulated particle and particle-list classes for the Monte Carlo engine.

Provides :class:`SimulatedParticle` (a single Monte Carlo particle carrying a
:class:`~tandemmp.simulation.meta.Structure` and a current
:class:`~tandemmp.simulation.meta.Stage`)
and :class:`SimulatedParticleList` (a mutable container with bulk accessors
used by the simulator to query per-particle rates and positions).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import overload

import numpy as np
import numpy.typing as npt

from tandemmp.simulation.meta import Reaction, Stage, Structure
from tandemmp.simulation.movement import MovementVector

__all__ = [
    "Command",
    "ParticleData",
    "SimulatedParticle",
    "SimulatedParticleList",
]


# ---------------------------------------------------------------------------
# ID allocation
# ---------------------------------------------------------------------------


_particle_id_counter: itertools.count[int] = itertools.count(0)


def _allocate_particle_id() -> int:
    return next(_particle_id_counter)


def _reset_particle_ids() -> None:
    """Reset the module-level particle ID counter (tests only)."""
    global _particle_id_counter
    _particle_id_counter = itertools.count(0)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Command:
    """Scheduled override that forces a particle's state on a given frame."""

    frame: int
    x: float
    y: float
    mass: float | None = None


@dataclass(frozen=True)
class ParticleData:
    """Recorded time-series for a single simulated particle."""

    frames: tuple[int, ...]
    masses: tuple[float, ...]
    xs: tuple[float, ...]
    ys: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.frames)
        if not (len(self.masses) == len(self.xs) == len(self.ys) == n):
            raise ValueError("frames, masses, xs, ys must all have the same length")

    def __len__(self) -> int:
        return len(self.frames)


# ---------------------------------------------------------------------------
# Single particle
# ---------------------------------------------------------------------------


class SimulatedParticle:
    """A single Monte Carlo simulated particle.

    Each particle carries a :class:`Structure` describing its mass and binding
    chemistry, a current :class:`Stage` (location), and optionally a
    2-D spatial position once it has bound to the surface.
    """

    def __init__(
        self,
        *,
        structure: Structure,
        stage: Stage = Stage.SOLUTION,
    ) -> None:
        if not isinstance(structure, Structure):
            raise TypeError(
                f"structure must be a Structure, got {type(structure).__name__}"
            )
        if not isinstance(stage, Stage):
            raise TypeError(f"stage must be a Stage, got {type(stage).__name__}")
        self.structure: Structure = structure
        self.stage: Stage = stage
        self.position: tuple[float, float] | None = None
        self.commands: dict[int, Command] = {}
        self._records: dict[int, tuple[float, float, float]] = {}
        self.id: int = _allocate_particle_id()

    def distance_to(self, x: float, y: float) -> float:
        """Return the Euclidean distance from this particle to ``(x, y)``."""
        if self.position is None:
            raise RuntimeError("particle has no position")
        px, py = self.position
        return float(np.hypot(x - px, y - py))

    def set_command(self, command: Command, *, remove_rates: bool = True) -> None:
        """Schedule *command* to fire on its frame.

        When *remove_rates* is true (the default) all kinetic rates on the
        particle's structure are zeroed so the command is the only action that
        will affect this particle going forward.
        """
        if not isinstance(command, Command):
            raise TypeError(f"command must be a Command, got {type(command).__name__}")
        self.commands[command.frame] = command
        if remove_rates:
            self.structure.binding_rate = 0.0
            self.structure.unbinding_rate = 0.0
            self.structure.diffusion_rate = 0.0
            self.structure.movement_vector = MovementVector(0.0, 0.0)

    def record(self, frame: int) -> None:
        """Record the particle's current (mass, x, y) for *frame*."""
        if self.position is None:
            raise RuntimeError("cannot record a particle without a position")
        x, y = self.position
        self._records[frame] = (float(self.structure.mass), float(x), float(y))

    @property
    def data(self) -> ParticleData:
        """Return the recorded time-series as an immutable :class:`ParticleData`."""
        frames = sorted(self._records)
        entries = [self._records[f] for f in frames]
        return ParticleData(
            frames=tuple(frames),
            masses=tuple(e[0] for e in entries),
            xs=tuple(e[1] for e in entries),
            ys=tuple(e[2] for e in entries),
        )

    def choose_reaction(
        self,
        chance: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> Reaction | None:
        """Pick a reaction whose rate is ≥ *chance*, or ``None``.

        If more than one reaction is eligible a uniform random choice is made
        using *rng* (a fresh :func:`numpy.random.default_rng` when omitted).
        """
        reactions = self.structure.actual_reactions
        eligible = [r for r in reactions if r.rate >= chance]
        if not eligible:
            return None
        if len(eligible) == 1:
            return eligible[0]
        if rng is None:
            rng = np.random.default_rng()
        return eligible[int(rng.integers(len(eligible)))]

    def __repr__(self) -> str:
        return (
            f"SimulatedParticle(id={self.id}, stage={self.stage!r}, "
            f"position={self.position!r}, structure={self.structure!r})"
        )


# ---------------------------------------------------------------------------
# Particle list
# ---------------------------------------------------------------------------


class SimulatedParticleList:
    """Mutable container of :class:`SimulatedParticle` with bulk accessors."""

    def __init__(self, particles: Iterable[SimulatedParticle] | None = None) -> None:
        self._particles: list[SimulatedParticle] = []
        if particles is not None:
            self.extend(particles)

    def __iter__(self) -> Iterator[SimulatedParticle]:
        return iter(self._particles)

    def __len__(self) -> int:
        return len(self._particles)

    @overload
    def __getitem__(self, index: int) -> SimulatedParticle: ...

    @overload
    def __getitem__(
        self,
        index: slice
        | npt.NDArray[np.bool_]
        | npt.NDArray[np.integer]
        | list[int]
        | list[bool],
    ) -> npt.NDArray[np.object_]: ...

    def __getitem__(
        self,
        index: int
        | slice
        | npt.NDArray[np.bool_]
        | npt.NDArray[np.integer]
        | list[int]
        | list[bool],
    ) -> SimulatedParticle | npt.NDArray[np.object_]:
        if isinstance(index, int):
            return self._particles[index]
        if isinstance(index, slice):
            return np.array(self._particles[index], dtype=object)
        return np.array(self._particles, dtype=object)[index]

    def extend(self, particles: Iterable[SimulatedParticle]) -> None:
        """Append each particle in *particles* to this list."""
        for particle in particles:
            if not isinstance(particle, SimulatedParticle):
                raise TypeError(
                    "particles must be SimulatedParticle instances, got "
                    f"{type(particle).__name__}"
                )
            self._particles.append(particle)

    def overlap(self, other: SimulatedParticleList) -> bool:
        """Return ``True`` if this list shares any particle with *other*."""
        return bool(set(self._particles) & set(other._particles))

    def set_stage(self, stage: Stage) -> None:
        """Set every particle's stage to *stage*."""
        if not isinstance(stage, Stage):
            raise TypeError(f"stage must be a Stage, got {type(stage).__name__}")
        for particle in self._particles:
            particle.stage = stage

    def get_stage(self, stage: Stage) -> npt.NDArray[np.bool_]:
        """Return a boolean mask selecting particles whose stage equals *stage*."""
        if not isinstance(stage, Stage):
            raise TypeError(f"stage must be a Stage, got {type(stage).__name__}")
        return np.array([p.stage == stage for p in self._particles], dtype=bool)

    @property
    def commands(self) -> list[dict[int, Command]]:
        return [p.commands for p in self._particles]

    @property
    def command_frames(self) -> list[list[int]]:
        return [list(p.commands.keys()) for p in self._particles]

    @property
    def bound_positions(self) -> npt.NDArray[np.floating]:
        sel = self.get_stage(Stage.GLASS)
        result = [p.position for p, s in zip(self._particles, sel, strict=True) if s]
        return np.array(result, dtype=float).reshape(-1, 2)

    @property
    def bound_masses(self) -> npt.NDArray[np.floating]:
        sel = self.get_stage(Stage.GLASS)
        return np.array(
            [p.structure.mass for p, s in zip(self._particles, sel, strict=True) if s],
            dtype=float,
        )

    def binding_rates(self) -> npt.NDArray[np.floating]:
        return np.array([p.structure.binding_rate for p in self._particles])

    def unbinding_rates(self) -> npt.NDArray[np.floating]:
        return np.array([p.structure.unbinding_rate for p in self._particles])

    def diffusion_rates(self) -> npt.NDArray[np.floating]:
        return np.array([p.structure.diffusion_rate for p in self._particles])

    def movement_rates(self) -> npt.NDArray[np.floating]:
        return np.array(
            [p.structure.movement_vector.chance_per_frame for p in self._particles]
        )

    def species_numbers(self) -> dict[Structure, int]:
        counts: dict[Structure, int] = {}
        for p in self._particles:
            counts[p.structure] = counts.get(p.structure, 0) + 1
        return counts

    def structures(self, stage: Stage | None = None) -> npt.NDArray[np.object_]:
        """Return every structure (optionally filtered to *stage*)."""
        structures = np.array([p.structure for p in self._particles], dtype=object)
        if stage is None:
            return structures
        return np.asarray(structures[self.get_stage(stage)], dtype=object)

    def available_reactions(
        self, stage: Stage | None = None
    ) -> npt.NDArray[np.object_]:
        """Return an object array of ``actual_reactions`` per structure."""
        structures = self.structures(stage)
        result = np.empty(len(structures), dtype=object)
        for i, s in enumerate(structures):
            result[i] = s.actual_reactions
        return result
