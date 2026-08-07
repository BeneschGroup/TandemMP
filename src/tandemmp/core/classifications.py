"""Event and processing classifications for TandemMP."""

from __future__ import annotations

import enum
from typing import NamedTuple


class ProcessingMethod(enum.IntFlag):
    """How a peak image was produced."""

    ratiometric = enum.auto()
    subtraction = enum.auto()
    inferred = enum.auto()


class Classification(NamedTuple):
    """Characterises a binding-event type by what is observed before/after."""

    presence_before: bool
    presence_after: bool
    mass_change: int | None
    number_change: int | None
    rep: str

    def get_slice(self, navg: int, remove: int = 0) -> slice | None:
        """Return a slice selecting the relevant portion of an event series.

        Returns ``None`` when neither presence flag is set (edge case).
        """
        if self.presence_before and not self.presence_after:
            return slice(None, navg + 1 - remove)

        if self.presence_after and not self.presence_before:
            return slice(navg + remove, None)

        if self.presence_after and self.presence_before:
            return slice(None, None)

        return None  # TODO: should this raise an exception/error instead?

    def __repr__(self) -> str:
        return self.rep


class EventClassification(NamedTuple):
    """Namespace of canonical event classifications."""

    Binding: Classification = Classification(False, True, 1, 1, "Binding")
    Unbinding: Classification = Classification(True, False, -1, -1, "Unbinding")
    Movement: Classification = Classification(True, True, 0, 0, "Movement")
    Dissociation: Classification = Classification(True, True, -1, 0, "Dissociation")
    Association: Classification = Classification(True, True, 1, 0, "Association")
    Merge: Classification = Classification(True, True, 0, -1, "Merge")
    Split: Classification = Classification(True, True, 0, 1, "Split")
    Complex: Classification = Classification(True, True, None, None, "Complex")


# Module-level singleton so callers write ``EvClassification.Binding``.
EvClassification = EventClassification()


def classify_mass_change(
    mass: float,
    new_mass: float,
    minimum_change: float = 30.0,
) -> Classification:
    """Classify a mass transition between two states.

    Returns one of ``EvClassification.Movement``,
    ``EvClassification.Association``, or
    ``EvClassification.Dissociation``.
    """
    diff = new_mass - mass
    rel_diff = diff / abs(mass) if mass != 0.0 else 0.0

    if abs(rel_diff) <= 0.1 or abs(diff) <= minimum_change:
        return EvClassification.Movement
    elif rel_diff > 0:
        return EvClassification.Association
    else:
        return EvClassification.Dissociation
