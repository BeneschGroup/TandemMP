"""Metadata definitions for the TandemMP simulation engine.

Provides the :class:`Structure` hierarchy used to describe simulated
particles and their binding reactions, along with the :class:`Reaction`,
:class:`Stage` and :class:`Behaviour` enums consumed by the simulator.
"""

from __future__ import annotations

import enum
import itertools
import math
from dataclasses import dataclass, field

from tandemmp.core.conversions import nm2pixels
from tandemmp.simulation.defaults import SimulationDefaults
from tandemmp.simulation.movement import MovementVector

__all__ = [
    "Behaviour",
    "Reaction",
    "Stage",
    "Structure",
    "create_oligomer",
]

_DEFAULTS = SimulationDefaults()


def _mass_to_area(mass: float) -> float:
    """Return the nominal cross-sectional area for a particle of *mass* (kDa)."""
    return float(nm2pixels(0.066 * mass ** (1 / 3)) * 2 * math.pi)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Stage(enum.IntEnum):
    """Axial location of a simulated particle."""

    GLASS = 0
    NEAR = 1
    SOLUTION = 2


class Behaviour(enum.IntEnum):
    """Possible simulation event types."""

    BINDING = 0
    UNBINDING = 1
    MOVEMENT = 2
    DISSOCIATION = 3
    AGGREGATION = 4
    MERGING = 5
    SPLITTING = 6


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------


_reaction_id_counter: itertools.count[int] = itertools.count(1)


def _allocate_reaction_id() -> int:
    return next(_reaction_id_counter)


def _reset_reaction_ids() -> None:
    """Reset the module-level reaction ID counter (tests only)."""
    global _reaction_id_counter
    _reaction_id_counter = itertools.count(1)


@dataclass(frozen=True)
class Reaction:
    """A forward or reverse binding reaction between structures."""

    reactants: Structure | list[Structure]
    products: Structure | list[Structure]
    rate: float
    id: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _allocate_reaction_id())

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return (
            f"k{self.id} Reaction: {self.reactants} -> "
            f"{self.products}. Rate: {self.rate}"
        )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class Structure:
    """Recursive simulation particle description.

    A :class:`Structure` is either a leaf (carrying *mass*) or a
    composite that *contains* other structures.  Composites sum the
    mass and subunit count of their children and declare their own
    forward and backward binding reactions.
    """

    mass: float
    n_subunits: int

    def __init__(
        self,
        contains: list[Structure] | None = None,
        *,
        mass: float | None = None,
        movement_vector: MovementVector | None = None,
        forwards_reaction: float | None = None,
        backwards_reaction: float | None = None,
        label: str = "",
        binding_rate: float = _DEFAULTS.binding_rate,
        unbinding_rate: float = _DEFAULTS.unbinding_rate,
        diffusion_rate: float = _DEFAULTS.diffusion_rate,
    ) -> None:
        self.contains: list[Structure] | None = list(contains) if contains else None

        if self.contains:
            self.mass = float(sum(s.mass for s in self.contains))
            self.n_subunits = int(sum(s.n_subunits for s in self.contains))
        else:
            if mass is None:
                raise ValueError("Structure requires mass when contains is empty")
            self.mass = float(mass)
            self.n_subunits = 1

        self.label = label
        self.size = _mass_to_area(self.mass)
        self.movement_vector = (
            movement_vector if movement_vector is not None else MovementVector(0.0, 0.0)
        )
        self.binding_rate = binding_rate
        self.unbinding_rate = unbinding_rate
        self.diffusion_rate = diffusion_rate
        self._available_reactions: list[Reaction] = []
        self.forwards_reaction: Reaction | None = None
        self.backwards_reaction: Reaction | None = None

        if self.contains:
            if forwards_reaction is None or backwards_reaction is None:
                raise ValueError(
                    "forwards_reaction and backwards_reaction are required "
                    "when contains is set"
                )
            self.forwards_reaction = Reaction(
                reactants=self.contains, products=self, rate=forwards_reaction
            )
            self.backwards_reaction = Reaction(
                reactants=self, products=self.contains, rate=backwards_reaction
            )
            self.add_reactions(self.forwards_reaction, self.backwards_reaction)
            for subunit in self.contains:
                subunit.add_reactions(self.forwards_reaction)

    def add_reactions(self, *reactions: Reaction) -> None:
        """Register *reactions* on this structure, ignoring duplicates."""
        for reaction in reactions:
            if not isinstance(reaction, Reaction):
                raise TypeError(f"expected Reaction, got {type(reaction).__name__}")
            if reaction not in self._available_reactions:
                self._available_reactions.append(reaction)

    @property
    def available_reactions(self) -> list[Reaction]:
        return list(self._available_reactions)

    @property
    def actual_reactions(self) -> list[Reaction]:
        """Reactions in which this structure actually participates as a reactant."""
        results: list[Reaction] = []
        for reaction in self._available_reactions:
            reactants = reaction.reactants
            if self is reactants:
                results.append(reaction)
            elif isinstance(reactants, (list, tuple, set)) and self in reactants:
                results.append(reaction)
        return results

    def _all_components(self) -> set[Structure]:
        found: set[Structure] = {self}
        if self.contains:
            for child in self.contains:
                found |= child._all_components()
        return found

    @property
    def monomers(self) -> list[Structure]:
        return [c for c in self._all_components() if c.n_subunits == 1]

    @property
    def unique_configurations(self) -> list[Structure]:
        return sorted(self._all_components(), key=lambda s: -s.n_subunits)

    _KNOWN_ATTRS: frozenset[str] = frozenset(
        {"binding_rate", "unbinding_rate", "diffusion_rate", "movement_vector"}
    )

    def set_global(self, **properties: object) -> None:
        """Set *properties* on every component in the hierarchy."""
        unknown = set(properties) - self._KNOWN_ATTRS
        if unknown:
            raise AttributeError(f"Unknown Structure attributes: {unknown}")
        for component in self._all_components():
            for key, value in properties.items():
                setattr(component, key, value)

    def __repr__(self) -> str:
        return f"{self.label} {self.n_subunits} subunits: {self.mass}kDa"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Structure):
            return NotImplemented
        return (
            self.mass == other.mass
            and self.n_subunits == other.n_subunits
            and self.label == other.label
        )

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.mass, self.n_subunits, self.label))


def create_oligomer(monomer: Structure, n: int) -> Structure:
    """Build an *n*-mer by recursively binding *monomer* to itself.

    ``n=1`` returns the monomer unchanged.  Higher values produce a
    nested composite with unit forward/backward reaction rates.

    Note: each intermediate assembly's forward reaction is registered on
    *monomer* via :meth:`Structure.add_reactions`.  After
    ``create_oligomer(m, n)``, ``m.available_reactions`` will contain
    ``n - 1`` forward reactions (one per intermediate level).  This is
    intentional — the simulator uses ``available_reactions`` to determine
    which reactions a monomer can participate in at each stage.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    structure = monomer
    for _ in range(n - 1):
        structure = Structure(
            [monomer, structure],
            label=monomer.label,
            forwards_reaction=1.0,
            backwards_reaction=1.0,
        )
    return structure
