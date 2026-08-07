"""Contract tests for :mod:`tandemmp.pipeline.pipeline_interface`.

:class:`~tandemmp.pipeline.open_iscat_pipeline.OpenIscatPipeline` requires
``open_iscat`` to actually run, so only its structural conformance
(``isinstance``) is checked here. A minimal fake stands in as a second,
runnable implementation — proving the
:class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner` protocol is
satisfiable by more than one class, without needing ``open_iscat`` or the
proprietary Refeyn ``iscat`` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tandemmp.pipeline.open_iscat_pipeline import OpenIscatPipeline
from tandemmp.pipeline.pipeline_interface import PipelineResults, PipelineRunner


class _FakeEventsFound:
    def get_X(self) -> np.ndarray:  # noqa: N802
        return np.array([])

    def get_Y(self) -> np.ndarray:  # noqa: N802
        return np.array([])

    def get_I(self) -> np.ndarray:  # noqa: N802
        return np.array([], dtype=int)

    def get_C(self) -> np.ndarray:  # noqa: N802
        return np.array([])


class _FakePipelineRunner:
    """Minimal PipelineRunner double, unrelated to open_iscat."""

    def run(
        self,
        filepath: str | Path,
        *,
        frame_bypass: np.ndarray | None = None,
        parallel: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> PipelineResults:
        frames = frame_bypass if frame_bypass is not None else np.zeros((1, 4, 4))
        return PipelineResults(frames=frames, rframes=frames, events=_FakeEventsFound())


RUNNER_FACTORIES = [OpenIscatPipeline, _FakePipelineRunner]
RUNNER_IDS = ["OpenIscatPipeline", "_FakePipelineRunner"]


@pytest.fixture(params=RUNNER_FACTORIES, ids=RUNNER_IDS)
def pipeline_runner(request: pytest.FixtureRequest) -> PipelineRunner:
    return request.param()


class TestStructuralConformance:
    def test_isinstance_of_pipeline_runner(
        self, pipeline_runner: PipelineRunner
    ) -> None:
        assert isinstance(pipeline_runner, PipelineRunner)


class TestFakePipelineRunnerRun:
    def test_run_returns_pipeline_results(self) -> None:
        runner = _FakePipelineRunner()
        result = runner.run(
            "dummy.mp", frame_bypass=np.ones((2, 3, 3)), navg=5, l_thum=15
        )
        assert isinstance(result, PipelineResults)
        assert result.frames.shape == (2, 3, 3)

    def test_ignores_backend_specific_kwargs_meant_for_other_runners(self) -> None:
        # navg/l_thum are open_iscat-specific tuning, not part of the
        # PipelineRunner protocol itself — a generic caller may still pass
        # them, and any conforming implementation must tolerate that via
        # **kwargs rather than erroring.
        runner = _FakePipelineRunner()
        result = runner.run(
            "dummy.mp", navg=99, l_thum=1, some_other_backends_kwarg="x"
        )
        assert isinstance(result, PipelineResults)
