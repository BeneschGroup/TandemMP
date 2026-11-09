"""Tests for tandemmp.pipeline.open_iscat — target API."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tandemmp.pipeline.open_iscat_pipeline import Info, Movie, PipelineResults

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
