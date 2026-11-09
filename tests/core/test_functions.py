"""Tests for tandemmp.core.functions — target API."""

import numpy as np
import pytest

from tandemmp.core.functions import (
    closest_index,
    gaussian_2d,
    get_overlap_probability,
    nan_normalise,
    normalise,
)

# normalise / nan_normalise


class TestNormalise:
    def test_basic(self):
        arr = np.array([2.0, 4.0, 6.0])
        result = normalise(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_constant_array(self):
        arr = np.array([5.0, 5.0, 5.0])
        # Division by zero → nan is acceptable
        result = normalise(arr)
        assert np.all(np.isnan(result)) or np.all(result == 0.0)

    def test_preserves_shape(self):
        arr = np.ones((3, 4))
        assert normalise(arr).shape == (3, 4)


class TestNanNormalise:
    def test_basic(self):
        arr = np.array([2.0, np.nan, 6.0])
        result = nan_normalise(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)
        assert np.isnan(result[1])

    def test_no_nans(self):
        arr = np.array([1.0, 3.0, 5.0])
        result = nan_normalise(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)


# closest_index


class TestClosestIndex:
    def test_exact_match(self):
        d = {10: "a", 20: "b", 30: "c"}
        assert closest_index(d, 20) == "b"

    def test_returns_value_at_largest_key_leq_frame(self):
        d = {10: "a", 20: "b", 30: "c"}
        # frame=25 → largest key ≤ 25 is 20
        assert closest_index(d, 25) == "b"

    def test_frame_below_all_keys(self):
        d = {10: "a", 20: "b"}
        assert closest_index(d, 5) is None

    def test_single_key(self):
        d = {5: "x"}
        assert closest_index(d, 5) == "x"
        assert closest_index(d, 100) == "x"


# get_overlap_probability


class TestGetOverlapProbability:
    def test_identical_positions_same_sigma(self):
        p = get_overlap_probability(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        # Two identical Gaussians overlap completely
        assert p == pytest.approx(1.0)

    def test_far_apart(self):
        p = get_overlap_probability(0.0, 0.0, 1.0, 100.0, 100.0, 1.0)
        assert p == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        p1 = get_overlap_probability(1.0, 2.0, 1.0, 3.0, 4.0, 1.5)
        p2 = get_overlap_probability(3.0, 4.0, 1.5, 1.0, 2.0, 1.0)
        assert p1 == pytest.approx(p2)

    def test_returns_between_0_and_1(self):
        p = get_overlap_probability(0.0, 0.0, 1.0, 2.0, 0.0, 1.0)
        assert 0.0 <= p <= 1.0

    def test_uses_normal_dist_overlap(self):
        """Verify we match the statistics.NormalDist.overlap approach."""
        from statistics import NormalDist

        x1, y1, s1 = 1.0, 0.0, 1.5
        x2, y2, s2 = 4.0, 0.0, 2.0
        dist = np.hypot(x1 - x2, y1 - y2)
        expected = NormalDist(0, s1).overlap(NormalDist(dist, s2))
        assert get_overlap_probability(x1, y1, s1, x2, y2, s2) == pytest.approx(
            expected
        )


# gaussian_2d


class TestGaussian2D:
    def test_peak_at_center(self):
        xy = np.array([[0.0], [0.0]])
        val = gaussian_2d(xy, sigma=1.0)
        assert val[0] == pytest.approx(1.0 / (2 * np.pi))

    def test_symmetry(self):
        xy = np.array([[1.0, -1.0], [0.0, 0.0]])
        vals = gaussian_2d(xy, sigma=1.0)
        assert vals[0] == pytest.approx(vals[1])

    def test_output_length(self):
        xy = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        vals = gaussian_2d(xy, sigma=1.0)
        assert len(vals) == 3
