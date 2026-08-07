"""Tests for tandemmp.tracking.metrics — target API."""

from __future__ import annotations

import numpy as np
import pytest

from tandemmp.core.conversions import NM_PER_PIXEL
from tandemmp.tracking.metrics import (
    delta_distance,
    delta_travel,
    mean_squared_displacement,
    msd,
    num_jumps,
    residency_time,
    total_distance_moved,
    total_travel,
)


class TestDeltaDistance:
    def test_basic_steps(self) -> None:
        xs = [0.0, 3.0, 3.0]
        ys = [0.0, 4.0, 4.0]
        result = delta_distance(xs, ys)
        np.testing.assert_allclose(result, [5.0, 0.0])

    def test_single_point_returns_empty(self) -> None:
        assert delta_distance([0.0], [0.0]).size == 0

    def test_empty_returns_empty(self) -> None:
        assert delta_distance([], []).size == 0

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            delta_distance([0.0, 1.0], [0.0])

    def test_accepts_numpy_arrays(self) -> None:
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(delta_distance(xs, ys), [1.0, 1.0])


class TestDeltaTravel:
    def test_from_origin(self) -> None:
        xs = [0.0, 1.0, 3.0, 3.0]
        ys = [0.0, 0.0, 4.0, 4.0]
        np.testing.assert_allclose(delta_travel(xs, ys), [1.0, 5.0, 5.0])

    def test_single_point_returns_empty(self) -> None:
        assert delta_travel([1.0], [2.0]).size == 0

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            delta_travel([0.0, 1.0], [0.0])


class TestTotalDistanceMoved:
    def test_no_threshold_sums_all_steps(self) -> None:
        xs = [0.0, 1.0, 3.0, 3.0]
        ys = [0.0, 0.0, 0.0, 4.0]
        assert total_distance_moved(xs, ys) == pytest.approx(1.0 + 2.0 + 4.0)

    def test_threshold_filters_small_steps(self) -> None:
        xs = [0.0, 0.1, 5.0]
        ys = [0.0, 0.0, 0.0]
        assert total_distance_moved(xs, ys, threshold=1.0) == pytest.approx(4.9)

    def test_returns_float(self) -> None:
        result = total_distance_moved([0.0, 1.0], [0.0, 0.0])
        assert isinstance(result, float)

    def test_empty_trajectory(self) -> None:
        assert total_distance_moved([], []) == 0.0


class TestTotalTravel:
    def test_straight_line(self) -> None:
        assert total_travel([0.0, 1.0, 3.0], [0.0, 0.0, 4.0]) == pytest.approx(5.0)

    def test_single_point(self) -> None:
        assert total_travel([1.0], [2.0]) == 0.0

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            total_travel([0.0], [0.0, 1.0])


class TestResidencyTime:
    def test_span(self) -> None:
        assert residency_time([10, 15, 42]) == 32

    def test_empty_returns_one(self) -> None:
        assert residency_time([]) == 1

    def test_single_frame(self) -> None:
        assert residency_time([5]) == 0


class TestNumJumps:
    def test_counts_steps_above_threshold(self) -> None:
        xs = [0.0, 0.5, 5.0, 5.1]
        ys = [0.0, 0.0, 0.0, 0.0]
        assert num_jumps(xs, ys, threshold=1.0) == 1

    def test_zero_when_none_exceed(self) -> None:
        assert num_jumps([0.0, 0.1], [0.0, 0.0], threshold=1.0) == 0


class TestMSD:
    def test_stationary_is_zero(self) -> None:
        xs = [1.0, 1.0, 1.0, 1.0]
        ys = [2.0, 2.0, 2.0, 2.0]
        lags, values = msd(xs, ys)
        np.testing.assert_array_equal(lags, [0, 1, 2, 3])
        np.testing.assert_allclose(values, [0.0, 0.0, 0.0, 0.0])

    def test_uniform_motion(self) -> None:
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        ys = np.zeros(4)
        lags, values = msd(xs, ys)
        np.testing.assert_allclose(values, [0.0, 1.0, 4.0, 9.0])

    def test_empty(self) -> None:
        lags, values = msd([], [])
        assert lags.size == 0
        assert values.size == 0

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            msd([0.0, 1.0], [0.0])


class TestMeanSquaredDisplacement:
    def test_matches_pixel_conversion(self) -> None:
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 0.0, 0.0])
        frames = np.array([0, 1, 2])
        times, values = mean_squared_displacement(xs, ys, frames, fps=1.0)
        # pixels2nm multiplies by NM_PER_PIXEL → step = NM_PER_PIXEL nm
        np.testing.assert_allclose(times, [1.0, 2.0])
        np.testing.assert_allclose(values, [NM_PER_PIXEL**2, (2 * NM_PER_PIXEL) ** 2])

    def test_fps_scales_time(self) -> None:
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 0.0, 0.0])
        frames = np.array([0, 10, 20])
        times, _ = mean_squared_displacement(xs, ys, frames, fps=10.0)
        np.testing.assert_allclose(times, [1.0, 2.0])

    def test_invalid_fps(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            mean_squared_displacement([0.0, 1.0], [0.0, 0.0], [0, 1], fps=0.0)

    def test_too_short_returns_empty(self) -> None:
        times, values = mean_squared_displacement([1.0], [2.0], [0], fps=1.0)
        assert times.size == 0
        assert values.size == 0

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            mean_squared_displacement([0.0, 1.0], [0.0, 1.0], [0], fps=1.0)

    def test_non_zero_starting_frame(self) -> None:
        # Lag times must be relative to the first frame, not absolute.
        # frames=[100, 110, 120] at fps=10 → lag 1 = 1.0 s, lag 2 = 2.0 s
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 0.0, 0.0])
        frames = np.array([100, 110, 120])
        times, _ = mean_squared_displacement(xs, ys, frames, fps=10.0)
        np.testing.assert_allclose(times, [1.0, 2.0])
