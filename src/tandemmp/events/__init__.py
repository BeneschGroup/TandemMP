"""Events subpackage — ratiometric, subtraction, and ghost event types.

Public API
----------
.. autosummary::
    Correction
    PeakImage
    Connectable
    AbstractEvent
    RatiometricEvent
    SubtractionEvent
    SubtractionSeries
    GhostEvent
    get_ghost_model
"""

from tandemmp.events._base import (
    AbstractEvent,
    Connectable,
    Correction,
    PeakImage,
)
from tandemmp.events.ghost import GhostEvent
from tandemmp.events.ratiometric import RatiometricEvent
from tandemmp.events.subtraction import (
    SubtractionEvent,
    SubtractionSeries,
    get_ghost_model,
)

__all__ = [
    "AbstractEvent",
    "Connectable",
    "Correction",
    "GhostEvent",
    "PeakImage",
    "RatiometricEvent",
    "SubtractionEvent",
    "SubtractionSeries",
    "get_ghost_model",
]
