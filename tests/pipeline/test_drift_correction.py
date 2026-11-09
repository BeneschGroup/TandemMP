"""Tests for tandemmp.pipeline.drift_correction — target API.

``run_drift_correction`` is exercised with ``Loader``/``get_vector`` monkeypatched,
avoiding a real ``.mp``/``open_iscat`` dependency.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import pytest

from tandemmp.drift.corrector import DriftVector
from tandemmp.pipeline import drift_correction as dc
from tandemmp.pipeline.drift_correction import (
    DriftCorrectionFunction,
    DriftCorrectionResult,
    load_correction_function,
    load_drift_vector,
    run_drift_correction,
    save_correction_function,
    save_drift_vector,
)


class _FakeLoader:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.movie = np.zeros((4, 8, 8))


def _fake_get_vector(*, movie: Any, **kwargs: Any) -> DriftVector:
    return DriftVector(
        np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, -1.0, -2.0, -3.0])
    )


# DriftCorrectionFunction


class TestDriftCorrectionFunction:
    def test_subtracts_vector_at_integer_frame(self):
        vector = DriftVector(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        correct = DriftCorrectionFunction(vector=vector)
        x, y = correct.correct_coordinates(10.0, 20.0, 1)
        assert x == pytest.approx(10.0 - 2.0)
        assert y == pytest.approx(20.0 - 5.0)

    def test_interpolates_between_frames(self):
        vector = DriftVector(np.array([0.0, 2.0]), np.array([0.0, -4.0]))
        correct = DriftCorrectionFunction(vector=vector)
        x, y = correct.correct_coordinates(0.0, 0.0, 0.5)
        assert x == pytest.approx(-1.0)
        assert y == pytest.approx(2.0)

    def test_clamps_out_of_range(self):
        vector = DriftVector(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        correct = DriftCorrectionFunction(vector=vector)
        x, y = correct.correct_coordinates(0.0, 0.0, 100)
        assert x == pytest.approx(-2.0)
        assert y == pytest.approx(-2.0)

    def test_accepts_array_inputs(self):
        vector = DriftVector(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
        correct = DriftCorrectionFunction(vector=vector)
        xs = np.array([10.0, 10.0, 10.0])
        ys = np.array([10.0, 10.0, 10.0])
        i = np.array([0, 1, 2])
        x, y = correct.correct_coordinates(xs, ys, i)
        np.testing.assert_allclose(x, [10.0, 9.0, 8.0])
        np.testing.assert_allclose(y, [10.0, 9.0, 8.0])

    def test_rejects_mismatched_shapes(self):
        vector = DriftVector(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
        correct = DriftCorrectionFunction(vector=vector)
        xs = np.array([10.0, 10.0, 10.0])
        ys = np.array([10.0, 10.0, 10.0])
        i = np.array([[0], [1], [2]])  # shape (3, 1), broadcastable but wrong
        with pytest.raises(ValueError, match="same shape"):
            correct.correct_coordinates(xs, ys, i)

    def test_picklable(self, tmp_path):
        vector = DriftVector(np.array([1.0]), np.array([2.0]))
        correct = DriftCorrectionFunction(vector=vector)
        path = tmp_path / "fn.pkl"
        with path.open("wb") as fh:
            pickle.dump(correct, fh)
        with path.open("rb") as fh:
            loaded = pickle.load(fh)
        x, y = loaded.correct_coordinates(5.0, 5.0, 0)
        assert x == pytest.approx(4.0)
        assert y == pytest.approx(3.0)


# DriftCorrectionFunction.correct_movie


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestCorrectMovie:
    def _make_drifting_movie(self, dxs, dys, size=21):
        """Build a movie whose frames are shifted copies of a fixed base image."""
        from scipy.ndimage import shift

        rng = np.random.default_rng(0)
        base = rng.normal(0.0, 1.0, size=(size, size))
        return np.array(
            [shift(base, (dy, dx)) for dx, dy in zip(dxs, dys, strict=True)]
        )

    def test_default_i_matches_full_vector(self):
        dxs, dys = [0.0, 1.0, 2.0], [0.0, -1.0, -2.0]
        movie = self._make_drifting_movie(dxs, dys)
        vector = DriftVector(np.array(dxs), np.array(dys))
        correct = DriftCorrectionFunction(vector=vector)

        corrected = correct.correct_movie(movie, parallel=False)

        assert corrected.shape[0] == len(movie)
        # Correction should shrink the frame (cropped to the always-visible
        # region) and make the frames agree with each other far better than
        # the raw (drifting) input did.
        assert (
            corrected.shape[1] < movie.shape[1] or corrected.shape[2] < movie.shape[2]
        )
        rmse_before = np.mean([_rmse(movie[0], movie[k]) for k in range(1, len(movie))])
        rmse_after = np.mean(
            [_rmse(corrected[0], corrected[k]) for k in range(1, len(corrected))]
        )
        assert rmse_after < rmse_before * 0.5

    def test_rejects_mismatched_i_length(self):
        vector = DriftVector(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]))
        correct = DriftCorrectionFunction(vector=vector)
        movie = np.zeros((3, 10, 10))
        with pytest.raises(ValueError, match="length"):
            correct.correct_movie(movie, i=np.array([0.0, 1.0]), parallel=False)

    def test_rejects_default_i_when_movie_length_differs_from_vector(self):
        # Vector has 5 frames; movie only has 3 and no explicit i is given —
        # this is very likely a caller mistake (movie is a different slice
        # than the one the vector was estimated from), so it should raise
        # rather than silently clamping via np.interp.
        vector = DriftVector(np.zeros(5), np.zeros(5))
        correct = DriftCorrectionFunction(vector=vector)
        movie = np.zeros((3, 10, 10))
        with pytest.raises(ValueError, match="frames"):
            correct.correct_movie(movie, parallel=False)

    def test_custom_i_resamples_vector(self):
        # Vector estimated over 5 frames; movie only covers the last 3 —
        # pass their original indices so the right sub-range of drift is used.
        dxs, dys = [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, -1.0, -2.0, -3.0, -4.0]
        full_movie = self._make_drifting_movie(dxs, dys)
        vector = DriftVector(np.array(dxs), np.array(dys))
        correct = DriftCorrectionFunction(vector=vector)

        sub_movie = full_movie[2:]
        corrected = correct.correct_movie(
            sub_movie, i=np.array([2.0, 3.0, 4.0]), parallel=False
        )

        assert corrected.shape[0] == len(sub_movie)
        rmse_before = np.mean(
            [_rmse(sub_movie[0], sub_movie[k]) for k in range(1, len(sub_movie))]
        )
        rmse_after = np.mean(
            [_rmse(corrected[0], corrected[k]) for k in range(1, len(corrected))]
        )
        assert rmse_after < rmse_before * 0.5

    def test_returns_ndarray(self):
        vector = DriftVector(np.zeros(3), np.zeros(3))
        correct = DriftCorrectionFunction(vector=vector)
        movie = np.random.default_rng(1).normal(size=(3, 12, 12))
        corrected = correct.correct_movie(movie, parallel=False)
        assert isinstance(corrected, np.ndarray)
        assert corrected.ndim == 3


# DriftCorrectionResult


class TestDriftCorrectionResult:
    def test_as_function_wraps_same_vector(self):
        vector = DriftVector(np.array([1.0]), np.array([2.0]))
        result = DriftCorrectionResult(vector=vector, source_filepath="foo.mp")
        fn = result.as_function()
        assert isinstance(fn, DriftCorrectionFunction)
        assert fn.vector is vector


# run_drift_correction


class TestRunDriftCorrection:
    def test_wires_loader_movie_into_get_vector(self, monkeypatch):
        calls: dict[str, Any] = {}

        def _spy_get_vector(*, movie: Any, **kwargs: Any) -> DriftVector:
            calls["movie_shape"] = movie.shape
            calls.update(kwargs)
            return _fake_get_vector(movie=movie)

        monkeypatch.setattr(dc, "Loader", _FakeLoader)
        monkeypatch.setattr(dc, "get_vector", _spy_get_vector)

        result = run_drift_correction(
            "movie.mp",
            frame_gap=10,
            rolling_baseline=True,
            threshold=1.5,
            threshold_margin=2,
            parallel=False,
            n_jobs=3,
            seed=42,
        )

        assert isinstance(result, DriftCorrectionResult)
        assert result.source_filepath == "movie.mp"
        assert calls["movie_shape"] == (4, 8, 8)
        assert calls["frame_gap"] == 10
        assert calls["rolling_baseline"] is True
        assert calls["threshold"] == 1.5
        assert calls["threshold_margin"] == 2
        assert calls["parallel"] is False
        assert calls["n_jobs"] == 3
        assert calls["seed"] == 42

    def test_forwards_frame_start_end_to_loader(self, monkeypatch):
        seen: dict[str, Any] = {}

        class _CapturingLoader(_FakeLoader):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["frame_start"] = kwargs["frame_start"]
                seen["frame_end"] = kwargs["frame_end"]
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(dc, "Loader", _CapturingLoader)
        monkeypatch.setattr(dc, "get_vector", _fake_get_vector)

        result = run_drift_correction("movie.mp", frame_start=5, frame_end=15)

        assert seen["frame_start"] == 5
        assert seen["frame_end"] == 15
        assert result.frame_start == 5
        assert result.frame_end == 15

    def test_returns_estimated_vector(self, monkeypatch):
        monkeypatch.setattr(dc, "Loader", _FakeLoader)
        monkeypatch.setattr(dc, "get_vector", _fake_get_vector)

        result = run_drift_correction("movie.mp")

        np.testing.assert_allclose(result.vector.x, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(result.vector.y, [0.0, -1.0, -2.0, -3.0])

    def test_disables_loader_drift_correction(self, monkeypatch):
        seen: dict[str, Any] = {}

        class _CapturingLoader(_FakeLoader):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["parameters"] = kwargs["parameters"]
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(dc, "Loader", _CapturingLoader)
        monkeypatch.setattr(dc, "get_vector", _fake_get_vector)

        run_drift_correction("movie.mp")

        assert seen["parameters"].drift_correction is False


# save_drift_vector / load_drift_vector


class TestSaveLoadDriftVector:
    def test_round_trip(self, tmp_path):
        vector = DriftVector(np.array([0.0, 1.0, 2.0]), np.array([0.0, -1.0, -2.0]))
        result = DriftCorrectionResult(vector=vector, source_filepath="foo.mp")
        path = save_drift_vector(tmp_path / "vec.json", result)

        loaded = load_drift_vector(path)
        np.testing.assert_allclose(loaded.vector.x, vector.x)
        np.testing.assert_allclose(loaded.vector.y, vector.y)
        assert loaded.source_filepath == "foo.mp"

    def test_saved_file_is_plain_editable_json(self, tmp_path):
        vector = DriftVector(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        result = DriftCorrectionResult(vector=vector, source_filepath="foo.mp")
        path = save_drift_vector(tmp_path / "vec.json", result)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == {
            "source_filepath": "foo.mp",
            "frame_start": None,
            "frame_end": None,
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
        }

    def test_round_trip_preserves_frame_start_end(self, tmp_path):
        vector = DriftVector(np.array([0.0, 1.0]), np.array([0.0, -1.0]))
        result = DriftCorrectionResult(
            vector=vector, source_filepath="foo.mp", frame_start=5, frame_end=15
        )
        path = save_drift_vector(tmp_path / "vec.json", result)

        loaded = load_drift_vector(path)
        assert loaded.frame_start == 5
        assert loaded.frame_end == 15

    def test_save_creates_parent_directories(self, tmp_path):
        vector = DriftVector(np.array([1.0]), np.array([2.0]))
        result = DriftCorrectionResult(vector=vector, source_filepath="foo.mp")
        path = tmp_path / "nested" / "dir" / "vec.json"
        save_drift_vector(path, result)
        assert path.exists()

    def test_save_rejects_wrong_type(self, tmp_path):
        with pytest.raises(TypeError):
            save_drift_vector(tmp_path / "bad.json", {"not": "a result"})  # type: ignore[arg-type]

    def test_load_rejects_missing_required_fields(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not": "a vector"}', encoding="utf-8")
        with pytest.raises(ValueError, match="x"):
            load_drift_vector(path)

    def test_hand_edited_json_loads(self, tmp_path):
        path = tmp_path / "hand_edited.json"
        path.write_text('{"x": [1.0, 2.0], "y": [3.0, 4.0]}', encoding="utf-8")
        loaded = load_drift_vector(path)
        np.testing.assert_allclose(loaded.vector.x, [1.0, 2.0])
        np.testing.assert_allclose(loaded.vector.y, [3.0, 4.0])
        assert loaded.source_filepath == str(path)


# save_correction_function / load_correction_function


class TestSaveLoadCorrectionFunction:
    def test_round_trip(self, tmp_path):
        vector = DriftVector(np.array([0.0, 1.0]), np.array([0.0, -1.0]))
        result = DriftCorrectionResult(vector=vector, source_filepath="foo.mp")
        path = save_correction_function(tmp_path / "fn.pkl", result)

        loaded = load_correction_function(path)
        assert isinstance(loaded, DriftCorrectionFunction)
        x, y = loaded.correct_coordinates(10.0, 10.0, 1)
        assert x == pytest.approx(9.0)
        assert y == pytest.approx(11.0)

    def test_save_creates_parent_directories(self, tmp_path):
        vector = DriftVector(np.array([1.0]), np.array([2.0]))
        result = DriftCorrectionResult(vector=vector, source_filepath="foo.mp")
        path = tmp_path / "nested" / "dir" / "fn.pkl"
        save_correction_function(path, result)
        assert path.exists()

    def test_save_rejects_wrong_type(self, tmp_path):
        with pytest.raises(TypeError):
            save_correction_function(tmp_path / "bad.pkl", {"not": "a result"})  # type: ignore[arg-type]

    def test_load_rejects_non_function_pickle(self, tmp_path):
        path = tmp_path / "bad.pkl"
        with path.open("wb") as fh:
            pickle.dump({"not": "a function"}, fh)
        with pytest.raises(TypeError):
            load_correction_function(path)
