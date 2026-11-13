"""Generic event-finding pipeline interface.

:class:`PipelineRunner` is a :class:`typing.Protocol` for the minimal surface an
event-finding backend must expose; :class:`EventsFound` and
:class:`PipelineResults` are the shared result types any backend produces. No
backend implementations are imported, avoiding circular imports;
:mod:`tandemmp.pipeline.open_iscat_pipeline` re-exports the result types for
backward compatibility.
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

    Exposes per-axis accessors rather than per-event records, so it is not a plain
    list.
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

    Only ``filepath``/``frame_bypass``/``parallel`` are universal across backends.
    Backend-specific tuning (e.g. open_iscat's ``navg`` and ``l_thum``) goes through
    ``**kwargs``, so a backend isn't forced to accept knobs it doesn't use.

    Implemented by :class:`tandemmp.pipeline.open_iscat_pipeline.OpenIscatPipeline`;
    other backends satisfy the contract without subclassing.
    """

    def run(
        self,
        filepath: str | pathlib.Path,
        *,
        frame_bypass: npt.NDArray[np.floating[Any]] | None = None,
        parallel: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> PipelineResults: ...
