"""Tests for tandemmp.io.movie — pure movie processing functions."""

from __future__ import annotations

import numpy as np
import pytest
from tandemmp.io.movie import (
    MpMetadata,
    duplicate_and_change_frames,
    fast_ratiometric,
    fast_rolling_mean,
    final_polish_fix,
    load_mp,
    pre_bin,
    ratiometric,
    rolling_mean,
    start_subtraction,
    start_subtraction_fast,
)

# Helpers

RNG = np.random.default_rng(42)


def _movie(n_frames: int = 20, height: int = 8, width: int = 8) -> np.ndarray:
    """Synthetic movie filled with small positive values to avoid division-by-zero."""
    return (RNG.random((n_frames, height, width)) + 1.0).astype(float)


# ratiometric


class TestRatiometric:
    def test_returns_same_shape(self) -> None:
        movie = _movie()
        result = ratiometric(movie, navg=3)
        assert result.shape == movie.shape

    def test_edge_frames_are_nan(self) -> None:
        movie = _movie()
        navg = 3
        result = ratiometric(movie, navg=navg)
        assert np.all(np.isnan(result[:navg]))
        assert np.all(np.isnan(result[-navg:]))

    def test_inner_frames_are_finite(self) -> None:
        movie = _movie()
        navg = 3
        result = ratiometric(movie, navg=navg)
        inner = result[navg : len(movie) - navg]
        assert np.all(np.isfinite(inner))

    def test_default_navg_5(self) -> None:
        movie = _movie(n_frames=30)
        result = ratiometric(movie)
        assert result.shape == movie.shape
        assert np.all(np.isnan(result[:5]))
        assert np.all(np.isnan(result[-5:]))

    def test_constant_movie_gives_zero(self) -> None:
        """A constant movie should give exactly 0 everywhere (ratio=1 → ratio-1=0)."""
        movie = np.ones((20, 8, 8)) * 2.0
        navg = 3
        result = ratiometric(movie, navg=navg)
        inner = result[navg : len(movie) - navg]
        np.testing.assert_allclose(inner, 0.0, atol=1e-12)

    def test_output_dtype_is_float(self) -> None:
        movie = np.ones((20, 4, 4), dtype=np.uint16) * 100
        result = ratiometric(movie, navg=2)
        assert result.dtype == float


# rolling_mean


class TestRollingMean:
    def test_returns_same_shape(self) -> None:
        movie = _movie()
        result = rolling_mean(movie, navg=3)
        assert result.shape == movie.shape

    def test_edge_frames_are_nan(self) -> None:
        movie = _movie()
        navg = 3
        result = rolling_mean(movie, navg=navg)
        assert np.all(np.isnan(result[:navg]))
        assert np.all(np.isnan(result[-navg:]))

    def test_inner_frames_are_finite(self) -> None:
        movie = _movie()
        navg = 3
        result = rolling_mean(movie, navg=navg)
        inner = result[navg : len(movie) - navg]
        assert np.all(np.isfinite(inner))

    def test_constant_movie_gives_zero(self) -> None:
        movie = np.ones((20, 8, 8)) * 3.0
        navg = 3
        result = rolling_mean(movie, navg=navg)
        inner = result[navg : len(movie) - navg]
        np.testing.assert_allclose(inner, 0.0, atol=1e-12)


# fast_rolling_mean


class TestFastRollingMean:
    def test_returns_same_shape(self) -> None:
        movie = _movie()
        result = fast_rolling_mean(movie, navg=3)
        assert result.shape == movie.shape

    def test_edge_frames_are_nan(self) -> None:
        movie = _movie()
        navg = 3
        result = fast_rolling_mean(movie, navg=navg)
        assert np.all(np.isnan(result[:navg]))
        assert np.all(np.isnan(result[-navg:]))

    def test_inner_frames_are_finite(self) -> None:
        movie = _movie()
        navg = 3
        result = fast_rolling_mean(movie, navg=navg)
        inner = result[navg : len(movie) - navg]
        assert np.all(np.isfinite(inner))

    def test_constant_movie_gives_zero(self) -> None:
        movie = np.ones((20, 4, 4)) * 5.0
        navg = 2
        result = fast_rolling_mean(movie, navg=navg)
        inner = result[navg : len(movie) - navg]
        np.testing.assert_allclose(inner, 0.0, atol=1e-12)

    def test_agrees_with_rolling_mean_on_constant(self) -> None:
        """fast and slow should agree on a constant movie."""
        movie = np.ones((24, 4, 4)) * 2.0
        navg = 3
        slow = rolling_mean(movie, navg=navg)
        fast = fast_rolling_mean(movie, navg=navg)
        np.testing.assert_allclose(
            slow[navg : len(movie) - navg],
            fast[navg : len(movie) - navg],
            atol=1e-12,
        )

    def test_agrees_with_rolling_mean_on_random_input(self) -> None:
        """fast and slow must agree numerically on non-constant input.

        The real test of the incremental-update divisor — a constant movie
        would pass regardless of the divisor because both outputs are 0.
        """
        rng = np.random.default_rng(123)
        movie = (rng.random((30, 5, 5)) + 1.0).astype(float)
        navg = 4
        slow = rolling_mean(movie, navg=navg)
        fast = fast_rolling_mean(movie, navg=navg)
        np.testing.assert_allclose(
            slow[navg : len(movie) - navg],
            fast[navg : len(movie) - navg],
            atol=1e-10,
        )


# fast_ratiometric


class TestFastRatiometric:
    def test_returns_same_shape(self) -> None:
        movie = _movie()
        result = fast_ratiometric(movie, navg=3)
        assert result.shape == movie.shape

    def test_edge_frames_are_nan(self) -> None:
        movie = _movie()
        navg = 3
        result = fast_ratiometric(movie, navg=navg)
        assert np.all(np.isnan(result[:navg]))
        assert np.all(np.isnan(result[-navg:]))

    def test_inner_frames_are_finite(self) -> None:
        movie = _movie()
        navg = 3
        result = fast_ratiometric(movie, navg=navg)
        inner = result[navg : len(movie) - navg]
        assert np.all(np.isfinite(inner))

    def test_constant_movie_gives_zero(self) -> None:
        movie = np.ones((20, 4, 4)) * 2.0
        navg = 3
        result = fast_ratiometric(movie, navg=navg)
        inner = result[navg : len(movie) - navg]
        np.testing.assert_allclose(inner, 0.0, atol=1e-12)

    def test_agrees_with_ratiometric_on_constant(self) -> None:
        """fast_ratiometric should agree with ratiometric on a constant movie."""
        movie = np.ones((20, 4, 4)) * 3.0
        navg = 2
        slow = ratiometric(movie, navg=navg)
        fast = fast_ratiometric(movie, navg=navg)
        np.testing.assert_allclose(
            slow[navg : len(movie) - navg],
            fast[navg : len(movie) - navg],
            atol=1e-12,
        )

    def test_agrees_with_ratiometric_on_random_input(self) -> None:
        """fast_ratiometric must match ratiometric numerically on non-constant input.

        The constant-input test would pass even with an off-by-one indexing
        bug in the ``n_sum`` rolling sum, because both outputs are 0
        regardless of the divisor.
        """
        rng = np.random.default_rng(456)
        movie = (rng.random((30, 5, 5)) + 1.0).astype(float)
        navg = 4
        slow = ratiometric(movie, navg=navg)
        fast = fast_ratiometric(movie, navg=navg)
        np.testing.assert_allclose(
            slow[navg : len(movie) - navg],
            fast[navg : len(movie) - navg],
            atol=1e-10,
        )


# start_subtraction


class TestStartSubtraction:
    def test_returns_same_shape(self) -> None:
        movie = _movie()
        result = start_subtraction(movie, navg=2)
        assert result.shape == movie.shape

    def test_first_navg_frames_are_nan(self) -> None:
        movie = _movie()
        navg = 3
        result = start_subtraction(movie, navg=navg)
        assert np.all(np.isnan(result[:navg]))

    def test_remaining_frames_are_finite(self) -> None:
        movie = _movie()
        navg = 2
        result = start_subtraction(movie, navg=navg, silent=True)
        assert np.all(np.isfinite(result[navg:]))

    def test_silent_suppresses_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """silent=True should write no progress bar to stderr."""
        movie = _movie()
        start_subtraction(movie, navg=2, silent=True)
        err = capsys.readouterr().err
        # tqdm writes to stderr; silent=True must leave it clean.
        assert "Calculating start subtraction" not in err

    def test_non_silent_shows_progress_bar(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        movie = _movie()
        start_subtraction(movie, navg=2, silent=False)
        err = capsys.readouterr().err
        assert "Calculating start subtraction" in err

    def test_start_frame_shifts_reference(self) -> None:
        """Non-zero start_frame takes the reference from that window onward."""
        movie = _movie(n_frames=20)
        navg = 2
        start = 3
        result = start_subtraction(movie, navg=navg, start_frame=start, silent=True)
        expected_ref = movie[start : start + navg + 1].mean(axis=0)
        # Any finite frame after navg should equal frame/ref - 1.
        np.testing.assert_allclose(result[navg + 2], movie[navg + 2] / expected_ref - 1)

    def test_constant_movie_gives_zero(self) -> None:
        movie = np.ones((15, 4, 4)) * 4.0
        navg = 2
        result = start_subtraction(movie, navg=navg, silent=True)
        np.testing.assert_allclose(result[navg:], 0.0, atol=1e-12)

    def test_output_dtype_is_float(self) -> None:
        movie = np.ones((15, 4, 4), dtype=np.uint8) * 50
        result = start_subtraction(movie, navg=1, silent=True)
        assert result.dtype == float


# start_subtraction_fast


class TestStartSubtractionFast:
    def test_2d_reference(self) -> None:
        movie = _movie()
        ref = movie[0]
        result = start_subtraction_fast(movie, ref)
        assert result.shape == movie.shape
        assert np.all(np.isfinite(result))

    def test_3d_reference_averages_axis0(self) -> None:
        movie = _movie()
        ref_3d = movie[:3]  # use first 3 frames as reference
        ref_2d = ref_3d.mean(axis=0)
        result_3d = start_subtraction_fast(movie, ref_3d)
        result_2d = start_subtraction_fast(movie, ref_2d)
        np.testing.assert_allclose(result_3d, result_2d)

    def test_constant_movie_gives_zero(self) -> None:
        movie = np.ones((10, 4, 4)) * 3.0
        ref = np.ones((4, 4)) * 3.0
        result = start_subtraction_fast(movie, ref)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_does_not_modify_input(self) -> None:
        movie = _movie()
        ref = movie[0].copy()
        ref_copy = ref.copy()
        start_subtraction_fast(movie, ref)
        np.testing.assert_array_equal(ref, ref_copy)


# pre_bin


class TestPreBin:
    def test_reduces_frame_count(self) -> None:
        movie = _movie(n_frames=20)
        result = pre_bin(movie, n=4)
        expected_frames = len(np.arange(0, 20, 4)) - 1
        assert result.shape[0] == expected_frames

    def test_preserves_spatial_dims(self) -> None:
        movie = _movie(n_frames=20, height=6, width=10)
        result = pre_bin(movie, n=4)
        assert result.shape[1:] == (6, 10)

    def test_bin_1_returns_all_but_last(self) -> None:
        """n=1 means each frame is its own bin; result = movie[:-1] effectively."""
        movie = _movie(n_frames=10)
        result = pre_bin(movie, n=1)
        assert result.shape[0] == 9  # len(arange(0,10,1))-1 = 9

    def test_constant_movie_unchanged_per_bin(self) -> None:
        movie = np.ones((12, 4, 4)) * 7.0
        result = pre_bin(movie, n=3)
        np.testing.assert_allclose(result, 7.0)

    def test_mean_is_correct(self) -> None:
        movie = np.zeros((6, 2, 2))
        movie[0] = 1.0
        movie[1] = 3.0  # bin 0 mean = 2.0
        movie[2] = 5.0
        movie[3] = 7.0  # bin 1 mean = 6.0
        result = pre_bin(movie, n=2)
        np.testing.assert_allclose(result[0], 2.0)
        np.testing.assert_allclose(result[1], 6.0)


# final_polish_fix


class TestFinalPolishFix:
    def test_prepends_correct_number_of_frames(self) -> None:
        movie = _movie(n_frames=10)
        navg = 5
        result = final_polish_fix(movie, navg=navg)
        assert result.shape[0] == 10 + navg * 2

    def test_preserves_spatial_dims(self) -> None:
        movie = _movie(n_frames=10, height=6, width=8)
        result = final_polish_fix(movie)
        assert result.shape[1:] == (6, 8)

    def test_original_frames_preserved(self) -> None:
        """The original frames (after the prepended ones) should be unchanged."""
        movie = _movie(n_frames=10)
        navg = 3
        result = final_polish_fix(movie, navg=navg)
        np.testing.assert_array_equal(result[navg * 2 :], movie)

    def test_prepended_frames_close_to_first_frame(self) -> None:
        """Prepended frames are noisy versions of movie[0]; shape must match."""
        movie = _movie(n_frames=10)
        navg = 3
        result = final_polish_fix(movie, navg=navg)
        prepended = result[: navg * 2]
        assert prepended.shape == (navg * 2, *movie.shape[1:])

    def test_default_navg_5(self) -> None:
        movie = _movie(n_frames=8)
        result = final_polish_fix(movie)
        assert result.shape[0] == 8 + 10


# load_mp


def _write_lowercase_schema_mp(
    path, frames, *, encoding="BASIC", scatter_destructive=False
):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        frame_ds = f.create_dataset("movie/frame", data=frames)
        if encoding is not None:
            frame_ds.attrs["encoding"] = encoding
        acq = f.create_group("movie/configuration/acq_camera")
        acq.create_dataset("frame_rate", data=500.0)
        acq.create_dataset("frame_binning", data=2)
        acq.create_dataset("pixel_binning", data=4)
        f["movie/configuration"].create_dataset(
            "scatter_interference_destructive", data=scatter_destructive
        )
    return path


def _write_devices_schema_mp(path, frames, *, scatter_destructive=True):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        frame_ds = f.create_dataset("movie/frame", data=frames)
        frame_ds.attrs["encoding"] = np.array(["BASIC"], dtype=object)
        acq = f.create_group("movie/configuration/Devices/AcqCam")
        acq.create_dataset("FrameRate", data=1320.0)
        acq.create_dataset("SoftwareFrameBinning", data=1)
        acq.create_dataset("SoftwarePixelBinning", data=1)
        acq.create_dataset("ScatterInterferenceDestructive", data=scatter_destructive)
    return path


class TestLoadMp:
    def test_lowercase_schema_loads(self, tmp_path):
        frames = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4)
        path = _write_lowercase_schema_mp(tmp_path / "a.mp", frames)
        loaded_frames, metadata = load_mp(path)
        np.testing.assert_array_equal(loaded_frames, frames)
        assert metadata == MpMetadata(
            scatter_interference_destructive=False,
            frame_rate=500.0,
            frame_binning=2,
            pixel_binning=4,
        )

    def test_devices_schema_loads(self, tmp_path):
        frames = np.ones((2, 3, 3), dtype=np.uint16)
        path = _write_devices_schema_mp(tmp_path / "b.mp", frames)
        loaded_frames, metadata = load_mp(path)
        np.testing.assert_array_equal(loaded_frames, frames)
        assert metadata == MpMetadata(
            scatter_interference_destructive=True,
            frame_rate=1320.0,
            frame_binning=1,
            pixel_binning=1,
        )

    def test_basic_encoding_as_array_attr(self, tmp_path):
        frames = np.ones((2, 3, 3), dtype=np.uint16)
        path = _write_lowercase_schema_mp(
            tmp_path / "c.mp", frames, encoding=np.array(["BASIC"], dtype=object)
        )
        loaded_frames, _ = load_mp(path)
        np.testing.assert_array_equal(loaded_frames, frames)

    def test_time_difference_encoding_raises(self, tmp_path):
        frames = np.ones((2, 3, 3), dtype=np.uint16)
        path = _write_lowercase_schema_mp(
            tmp_path / "d.mp", frames, encoding="TIME_DIFFERENCE"
        )
        with pytest.raises(ValueError, match="TIME_DIFFERENCE"):
            load_mp(path)

    def test_missing_movie_frame_raises(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "e.mp"
        with h5py.File(path, "w") as f:
            f.create_group("movie")
        with pytest.raises(ValueError, match="movie/frame"):
            load_mp(path)

    def test_non_3d_frame_raises(self, tmp_path):
        frames = np.ones((4, 4), dtype=np.uint16)
        path = _write_lowercase_schema_mp(tmp_path / "f.mp", frames)
        with pytest.raises(ValueError, match="3-D"):
            load_mp(path)

    def test_unrecognised_configuration_schema_raises(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "g.mp"
        with h5py.File(path, "w") as f:
            frame_ds = f.create_dataset(
                "movie/frame", data=np.ones((2, 3, 3), dtype=np.uint16)
            )
            frame_ds.attrs["encoding"] = "BASIC"
            f.create_group("movie/configuration")
        with pytest.raises(ValueError, match="configuration schema"):
            load_mp(path)


# duplicate_and_change_frames


class TestDuplicateAndChangeFrames:
    def test_writes_new_file_and_replaces_frames(self, tmp_path) -> None:
        h5py = pytest.importorskip("h5py")

        src = tmp_path / "sample.mp"
        original = np.arange(4 * 3 * 3, dtype=float).reshape(4, 3, 3)
        with h5py.File(src, "w") as f:
            grp = f.create_group("movie")
            grp.create_dataset("frame", data=original)

        new_frames = np.zeros_like(original)
        new_frames[1] = 7.0
        out = duplicate_and_change_frames(new_frames, src, distinguisher="edited")

        assert out.exists()
        assert out.name == "sample_edited.mp"
        assert out != src
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(f["movie"]["frame"][...], new_frames)
        # Source is untouched.
        with h5py.File(src, "r") as f:
            np.testing.assert_array_equal(f["movie"]["frame"][...], original)

    def test_preserves_multi_character_extension(self, tmp_path) -> None:
        """``.with_stem`` must not truncate longer extensions like ``.mp4``."""
        h5py = pytest.importorskip("h5py")

        src = tmp_path / "sample.mp4"
        with h5py.File(src, "w") as f:
            f.create_dataset("frame", data=np.zeros((2, 2, 2)))

        out = duplicate_and_change_frames(np.ones((2, 2, 2)), src, distinguisher="x")
        assert out.name == "sample_x.mp4"
