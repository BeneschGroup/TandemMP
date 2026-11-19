"""Tests for the movie containers and metadata types."""

from __future__ import annotations

import copy
import dataclasses
import pickle

import h5py
import numpy as np
import pytest

from tandemmp.io.movie import Movie, load_mp_file
from tandemmp.pipeline.pipeline_interface import Info, PipelineResults

# Info


class TestInfo:
    def test_fields(self) -> None:
        info = Info(
            scatter_interference_destructive=True,
            raw_camera_frame_rate=1320.0,
            frame_binning=1,
            pixel_binning=1,
        )
        assert info.scatter_interference_destructive is True
        assert info.raw_camera_frame_rate == 1320.0
        assert info.frame_binning == 1
        assert info.pixel_binning == 1

    def test_frozen(self) -> None:
        info = Info(
            scatter_interference_destructive=True,
            raw_camera_frame_rate=1320.0,
            frame_binning=1,
            pixel_binning=1,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.frame_binning = 2  # type: ignore[misc]

    def test_optional_fields_none(self) -> None:
        info = Info(
            scatter_interference_destructive=None,
            raw_camera_frame_rate=1000.0,
            frame_binning=2,
            pixel_binning=4,
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
        )
        movie = Movie(frames=frames, info=info)
        # shape, dtype, ndim come from the frames array
        assert movie.shape == (5, 32, 32)
        assert movie.dtype == np.float64
        assert movie.ndim == 3

    def test_copy_deepcopy_and_pickle_do_not_recurse(self) -> None:
        info = Info(
            scatter_interference_destructive=False,
            raw_camera_frame_rate=1000.0,
            frame_binning=1,
            pixel_binning=1,
        )
        movie = Movie(frames=np.arange(8.0).reshape(2, 2, 2), info=info)
        for clone in (
            copy.copy(movie),
            copy.deepcopy(movie),
            pickle.loads(pickle.dumps(movie)),  # noqa: S301
        ):
            assert isinstance(clone, Movie)
            np.testing.assert_array_equal(clone.frames, movie.frames)
            assert clone.info == info
            assert clone.shape == (2, 2, 2)

    def test_missing_frames_attribute_raises_attribute_error(self) -> None:
        bare = object.__new__(Movie)
        with pytest.raises(AttributeError):
            bare.frames  # noqa: B018
        with pytest.raises(AttributeError):
            bare.shape  # noqa: B018

    def test_np_asarray_still_reaches_the_frames(self) -> None:
        frames = np.arange(8.0).reshape(2, 2, 2)
        info = Info(
            scatter_interference_destructive=False,
            raw_camera_frame_rate=1000.0,
            frame_binning=1,
            pixel_binning=1,
        )
        np.testing.assert_array_equal(
            np.asarray(Movie(frames=frames, info=info)), frames
        )

    def test_frozen(self) -> None:
        frames = np.ones((5, 32, 32))
        info = Info(
            scatter_interference_destructive=False,
            raw_camera_frame_rate=1000.0,
            frame_binning=1,
            pixel_binning=1,
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
        )
        movie = Movie(frames=frames, info=info)
        with pytest.raises(AttributeError):
            _ = movie.nonexistent_attribute


# load_mp_file


def _write_mp(path, *, schema="acq_camera", encoding=None, frames=None):
    if frames is None:
        frames = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("movie/frame", data=frames)
        if encoding is not None:
            ds.attrs["encoding"] = encoding
        cfg = f.create_group("movie/configuration")
        if schema == "acq_camera":
            cfg["scatter_interference_destructive"] = False
            cfg["acq_camera/frame_rate"] = 128.5
            cfg["acq_camera/frame_binning"] = 2
            cfg["acq_camera/pixel_binning"] = 6
        elif schema == "devices":
            acq = cfg.create_group("Devices/AcqCam")
            acq["ScatterInterferenceDestructive"] = True
            acq["FrameRate"] = 1320.0
            acq["SoftwareFrameBinning"] = 3
            acq["SoftwarePixelBinning"] = 4
    return path


class TestLoadMpFile:
    def test_acq_camera_schema(self, tmp_path) -> None:
        movie = load_mp_file(_write_mp(tmp_path / "a.mp"))
        assert movie.shape == (2, 3, 4)
        assert movie.info == Info(False, 128.5, 2, 6)

    def test_devices_schema(self, tmp_path) -> None:
        movie = load_mp_file(_write_mp(tmp_path / "a.mp", schema="devices"))
        assert movie.info == Info(True, 1320.0, 3, 4)

    @pytest.mark.parametrize("encoding", ["BASIC", b"BASIC", np.array([b"BASIC"])])
    def test_basic_encoding_loads(self, tmp_path, encoding) -> None:
        load_mp_file(_write_mp(tmp_path / "a.mp", encoding=encoding))

    def test_time_difference_raises(self, tmp_path) -> None:
        path = _write_mp(tmp_path / "a.mp", encoding="TIME_DIFFERENCE")
        with pytest.raises(ValueError, match="TIME_DIFFERENCE"):
            load_mp_file(path)

    def test_missing_frames_raises(self, tmp_path) -> None:
        with h5py.File(tmp_path / "a.mp", "w") as f:
            f.create_group("movie")
        with pytest.raises(ValueError, match="movie/frame"):
            load_mp_file(tmp_path / "a.mp")

    def test_non_3d_raises(self, tmp_path) -> None:
        path = _write_mp(tmp_path / "a.mp", frames=np.zeros((3, 4), dtype=np.uint16))
        with pytest.raises(ValueError, match="3-D"):
            load_mp_file(path)

    def test_unknown_configuration_raises(self, tmp_path) -> None:
        path = _write_mp(tmp_path / "a.mp", schema="none")
        with pytest.raises(ValueError, match="configuration"):
            load_mp_file(path)
