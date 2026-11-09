"""Contract tests for :mod:`tandemmp.pipeline.pipeline_interface`.

Checks structural conformance of a minimal fake against the ``PipelineRunner``
protocol, and the ``Info``/``Movie``/``PipelineResults`` data containers.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tandemmp.pipeline.pipeline_interface import (
    Info,
    Movie,
    PipelineResults,
    PipelineRunner,
)


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
    """Minimal PipelineRunner double."""

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


class TestStructuralConformance:
    def test_isinstance_of_pipeline_runner(self) -> None:
        assert isinstance(_FakePipelineRunner(), PipelineRunner)


class TestFakePipelineRunnerRun:
    def test_run_returns_pipeline_results(self) -> None:
        runner = _FakePipelineRunner()
        result = runner.run(
            "dummy.mp", frame_bypass=np.ones((2, 3, 3)), navg=5, l_thum=15
        )
        assert isinstance(result, PipelineResults)
        assert result.frames.shape == (2, 3, 3)

    def test_ignores_backend_specific_kwargs_meant_for_other_runners(self) -> None:
        # navg/l_thum are backend-specific tuning, not part of the
        # PipelineRunner protocol itself — a generic caller may still pass
        # them, and any conforming implementation must tolerate that via
        # **kwargs rather than erroring.
        runner = _FakePipelineRunner()
        result = runner.run(
            "dummy.mp", navg=99, l_thum=1, some_other_backends_kwarg="x"
        )
        assert isinstance(result, PipelineResults)


# Info


class TestInfo:
    def test_fields(self) -> None:
        info = Info(
            scatter_interference_destructive=True,
            raw_camera_frame_rate=1320.0,
            frame_binning=1,
            pixel_binning=1,
            full_well_capacity=None,
            grey_levels=None,
        )
        assert info.scatter_interference_destructive is True
        assert info.raw_camera_frame_rate == 1320.0
        assert info.frame_binning == 1
        assert info.pixel_binning == 1
        assert info.full_well_capacity is None
        assert info.grey_levels is None

    def test_frozen(self) -> None:
        info = Info(
            scatter_interference_destructive=True,
            raw_camera_frame_rate=1320.0,
            frame_binning=1,
            pixel_binning=1,
            full_well_capacity=None,
            grey_levels=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.frame_binning = 2  # type: ignore[misc]

    def test_optional_fields_none(self) -> None:
        info = Info(
            scatter_interference_destructive=None,
            raw_camera_frame_rate=1000.0,
            frame_binning=2,
            pixel_binning=4,
            full_well_capacity=None,
            grey_levels=None,
        )
        assert info.scatter_interference_destructive is None


# PipelineResults


class TestPipelineResults:
    def test_fields(self) -> None:
        frames = np.zeros((10, 64, 64))
        rframes = np.zeros((10, 64, 64))
        events: list[object] = []
        pr = PipelineResults(frames=frames, rframes=rframes, events=events)
        assert pr.frames is frames
        assert pr.rframes is rframes
        assert pr.events is events

    def test_frozen(self) -> None:
        pr = PipelineResults(
            frames=np.zeros((1, 2, 2)),
            rframes=np.zeros((1, 2, 2)),
            events=[],
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            pr.events = []  # type: ignore[misc]


# Movie


class TestMovie:
    def test_frames_and_info(self) -> None:
        frames = np.ones((5, 32, 32))
        info = Info(
            scatter_interference_destructive=False,
            raw_camera_frame_rate=1000.0,
            frame_binning=1,
            pixel_binning=1,
            full_well_capacity=None,
            grey_levels=None,
        )
        movie = Movie(frames=frames, info=info)
        assert movie.frames is frames
        assert movie.info is info

    def test_getattr_delegates_to_frames(self) -> None:
        frames = np.ones((5, 32, 32))
        info = Info(
            scatter_interference_destructive=False,
            raw_camera_frame_rate=1000.0,
            frame_binning=1,
            pixel_binning=1,
            full_well_capacity=None,
            grey_levels=None,
        )
        movie = Movie(frames=frames, info=info)
        # shape, dtype, ndim come from the frames array
        assert movie.shape == (5, 32, 32)
        assert movie.dtype == np.float64
        assert movie.ndim == 3

    def test_frozen(self) -> None:
        frames = np.ones((5, 32, 32))
        info = Info(
            scatter_interference_destructive=False,
            raw_camera_frame_rate=1000.0,
            frame_binning=1,
            pixel_binning=1,
            full_well_capacity=None,
            grey_levels=None,
        )
        movie = Movie(frames=frames, info=info)
        with pytest.raises(dataclasses.FrozenInstanceError):
            movie.frames = np.zeros((1, 1, 1))  # type: ignore[misc]

    def test_getattr_raises_for_missing(self) -> None:
        frames = np.ones((2, 4, 4))
        info = Info(
            scatter_interference_destructive=False,
            raw_camera_frame_rate=1000.0,
            frame_binning=1,
            pixel_binning=1,
            full_well_capacity=None,
            grey_levels=None,
        )
        movie = Movie(frames=frames, info=info)
        with pytest.raises(AttributeError):
            _ = movie.nonexistent_attribute
