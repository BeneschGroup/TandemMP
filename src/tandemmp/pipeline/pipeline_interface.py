"""Generic event-finding pipeline interface.

Defines :class:`PipelineRunner`, a :class:`typing.Protocol` describing the
minimal surface an event-finding backend must expose, plus the shared result
types (:class:`EventsFound`, :class:`PipelineResults`) any backend produces.

This module does not import backend implementations, avoiding circular
imports. :mod:`tandemmp.pipeline.open_iscat_pipeline` re-exports the result
types for backward compatibility.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import pathlib

    import numpy.typing as npt

# Data containers


class EventsFound(Protocol):
    """Structural type for a backend's peak-finder output.

    Not a plain list — exposes per-axis accessors instead of per-event
    records.
    """

    def get_X(self) -> npt.NDArray[np.floating[Any]]: ...  # noqa: N802
    def get_Y(self) -> npt.NDArray[np.floating[Any]]: ...  # noqa: N802
    def get_I(self) -> npt.NDArray[np.integer[Any]]: ...  # noqa: N802
    def get_C(self) -> npt.NDArray[np.floating[Any]]: ...  # noqa: N802


@dataclasses.dataclass(frozen=True)
class PipelineResults:
    """Container for event-finding pipeline outputs."""

    frames: npt.NDArray[np.floating[Any]]
    rframes: npt.NDArray[np.floating[Any]]
    events: EventsFound


# Protocol


@runtime_checkable
class PipelineRunner(Protocol):
    """Structural interface an event-finding pipeline must satisfy.

    Only ``filepath``/``frame_bypass``/``parallel`` are treated as universal
    across backends — backend-specific tuning (e.g. open_iscat's ``navg``
    ratiometric-averaging window and ``l_thum`` thumbnail size) is forwarded
    via ``**kwargs`` rather than fixed in this signature, so a
    differently-parameterised backend isn't forced to accept knobs it has no
    use for.

    Concrete implementation:
    :class:`tandemmp.pipeline.open_iscat_pipeline.OpenIscatPipeline`.
    Other backends implement the same structural contract without subclassing.
    """

    def run(
        self,
        filepath: str | pathlib.Path,
        *,
        frame_bypass: npt.NDArray[np.floating[Any]] | None = None,
        parallel: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> PipelineResults: ...
