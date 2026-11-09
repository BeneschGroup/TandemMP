"""Tests for tandemmp.drift.corrector — drift correction."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.ndimage import shift

from tandemmp.drift.corrector import (
    DriftVector,
    _always_visible_1d,
    _always_visible_region,
    _float_to_slice,
    _get_initial_vector,
    _most_negative,
    _most_positive,
    _rmse,
    _shift_compare,
    _solve,
    get_corrected_movie,
    get_vector,
    get_vector_segmented,
    illumination_correction,
    stabilise,
)

# `differential_evolution` is stochastic in real usage (see `_solve`'s `seed`
# parameter), so every call in this file pins it to a fixed seed to keep the
# test suite deterministic.
_DE_SEED = 42

# Helpers


def _make_drifting_movie(
    n_frames: int = 12,
    size: int = 41,
    drift_per_frame: tuple[float, float] = (0.4, -0.3),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic movie with a known linear drift.

    Returns ``(movie, true_x, true_y)`` where ``true_x``/``true_y`` are the
    per-frame drift offsets used to generate the movie.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(size, size)).astype(float)
    # Add a few sharp features so the RMSE landscape has a clear minimum.
    yy, xx = np.mgrid[:size, :size]
    for cx, cy, amp in [
        (12, 14, 8.0),
        (28, 22, -6.0),
        (20, 30, 5.0),
    ]:
        base += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 4.0)

    dxs = np.arange(n_frames) * drift_per_frame[0]
    dys = np.arange(n_frames) * drift_per_frame[1]

    frames = np.array([shift(base, (dy, dx)) for dx, dy in zip(dxs, dys, strict=True)])
    return frames, dxs, dys


# DriftVector


class TestDriftVector:
    def test_basic_construction(self):
        v = DriftVector(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        np.testing.assert_array_equal(v.x, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(v.y, [4.0, 5.0, 6.0])

    def test_len(self):
        v = DriftVector(np.zeros(7), np.zeros(7))
        assert len(v) == 7

    def test_iter_yields_xy_pairs(self):
        v = DriftVector(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert list(v) == [(1.0, 3.0), (2.0, 4.0)]

    def test_getitem_returns_tuple(self):
        v = DriftVector(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        x, y = v[1]
        assert (float(x), float(y)) == (2.0, 5.0)

    def test_getitem_slice(self):
        v = DriftVector(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        xs, ys = v[:2]
        np.testing.assert_array_equal(xs, [1.0, 2.0])
        np.testing.assert_array_equal(ys, [4.0, 5.0])

    def test_negation(self):
        v = DriftVector(np.array([1.0, -2.0]), np.array([3.0, -4.0]))
        neg = -v
        np.testing.assert_array_equal(neg.x, [-1.0, 2.0])
        np.testing.assert_array_equal(neg.y, [-3.0, 4.0])

    def test_coerces_to_float(self):
        v = DriftVector(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert v.x.dtype == np.float64
        assert v.y.dtype == np.float64

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            DriftVector(np.zeros(3), np.zeros(4))

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            DriftVector(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_frozen(self):
        v = DriftVector(np.zeros(2), np.zeros(2))
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.x = np.zeros(3)  # type: ignore[misc]


# Internal scalar helpers


class TestScalarHelpers:
    def test_rmse_zero_for_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _rmse(a, a) == 0.0

    def test_rmse_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        # mean((9, 16)) = 12.5  → sqrt = 3.5355...
        assert _rmse(a, b) == pytest.approx(np.sqrt(12.5))

    def test_most_positive_with_positives(self):
        assert _most_positive(np.array([-2.0, 3.0, 1.0])) == 3.0

    def test_most_positive_all_negative(self):
        assert _most_positive(np.array([-2.0, -3.0])) == 0.0

    def test_most_negative_with_negatives(self):
        assert _most_negative(np.array([-2.0, 3.0, 1.0])) == -2.0

    def test_most_negative_all_positive(self):
        assert _most_negative(np.array([2.0, 3.0])) == 0.0

    def test_float_to_slice_positive(self):
        s = _float_to_slice(2.5)
        assert s == slice(3, None)

    def test_float_to_slice_negative(self):
        s = _float_to_slice(-2.5)
        assert s == slice(None, -3)

    def test_float_to_slice_zero(self):
        s = _float_to_slice(0.0)
        assert s == slice(1, None)

    def test_always_visible_1d(self):
        # Drift values from -2.3 to 1.7 → crop slice(3, -2)
        arr = np.array([-2.3, 0.0, 1.7])
        s = _always_visible_1d(arr)
        assert s == slice(3, -2)

    def test_always_visible_region(self):
        v = DriftVector(np.array([-1.0, 1.0]), np.array([-2.0, 2.0]))
        ys, xs = _always_visible_region(v)
        # x slice: from y of vector? No — region is (y_slice_for_y, y_slice_for_x)
        assert isinstance(ys, slice)
        assert isinstance(xs, slice)


# stabilise / shift_compare


class TestStabilise:
    def test_zero_shift_is_identity(self):
        frame = np.arange(25, dtype=float).reshape(5, 5)
        out = stabilise(frame, 0.0, 0.0)
        # scipy cubic-spline shift introduces ~1e-15 floating-point noise.
        np.testing.assert_allclose(out, frame, atol=1e-10)

    def test_undoes_known_shift(self):
        rng = np.random.default_rng(1)
        base = rng.normal(0, 1, size=(15, 15))
        shifted = shift(base, (2, -1))  # shift by (dy=2, dx=-1)
        out = stabilise(shifted, x=-1.0, y=2.0)
        # Centre region should match the original closely
        np.testing.assert_allclose(out[3:-3, 3:-3], base[3:-3, 3:-3], atol=1e-6)


class TestShiftCompare:
    def test_zero_shift_identical_frames(self):
        rng = np.random.default_rng(2)
        f = rng.normal(0, 1, size=(20, 20))
        assert _shift_compare((0.0, 0.0), f, f) == pytest.approx(0.0, abs=1e-9)

    def test_correct_shift_minimises(self):
        """Shifting baseline by the true offset gives lower RMSE than not."""
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1, size=(25, 25))
        true_dx, true_dy = 1.0, -2.0
        shifted = shift(base, (true_dy, true_dx))
        rmse_at_truth = _shift_compare((true_dx, true_dy), shifted, base)
        rmse_at_zero = _shift_compare((0.0, 0.0), shifted, base)
        assert rmse_at_truth < rmse_at_zero


# _solve — single-frame minimisation


class TestSolve:
    def test_recovers_known_integer_shift(self):
        rng = np.random.default_rng(4)
        base = rng.normal(0, 1, size=(31, 31))
        true_dx, true_dy = 2.0, -1.0
        shifted = shift(base, (true_dy, true_dx))
        x, y = _solve(input_frame=shifted, baseline_frame=base, seed=_DE_SEED)
        assert x == pytest.approx(true_dx, abs=0.2)
        assert y == pytest.approx(true_dy, abs=0.2)

    def test_recovers_subpixel_shift(self):
        rng = np.random.default_rng(5)
        base = rng.normal(0, 1, size=(31, 31))
        true_dx, true_dy = 0.7, -1.3
        shifted = shift(base, (true_dy, true_dx))
        x, y = _solve(input_frame=shifted, baseline_frame=base, seed=_DE_SEED)
        assert x == pytest.approx(true_dx, abs=0.3)
        assert y == pytest.approx(true_dy, abs=0.3)

    def test_zero_shift_for_identical_frames(self):
        rng = np.random.default_rng(6)
        f = rng.normal(0, 1, size=(31, 31))
        x, y = _solve(input_frame=f, baseline_frame=f, seed=_DE_SEED)
        assert x == pytest.approx(0.0, abs=0.3)
        assert y == pytest.approx(0.0, abs=0.3)

    def test_seed_gives_reproducible_result(self):
        """Same seed -> identical result; this is what keeps the rest of the
        suite deterministic despite `differential_evolution` being stochastic
        by default (`seed=None`, the production default)."""
        rng = np.random.default_rng(7)
        base = rng.normal(0, 1, size=(31, 31))
        shifted = shift(base, (-1.0, 2.0))
        first = _solve(input_frame=shifted, baseline_frame=base, seed=123)
        second = _solve(input_frame=shifted, baseline_frame=base, seed=123)
        np.testing.assert_array_equal(first, second)


# _get_initial_vector


class TestInitialVector:
    def test_serial_recovers_linear_drift(self):
        movie, dxs, dys = _make_drifting_movie(n_frames=6, size=41)
        indexes = np.arange(len(movie), dtype=int)
        v = _get_initial_vector(
            movie=movie,
            indexes=indexes,
            rolling_baseline=False,
            parallel=False,
            n_jobs=None,
            seed=_DE_SEED,
        )
        assert isinstance(v, DriftVector)
        assert len(v) == len(movie)
        np.testing.assert_allclose(v.x, dxs, atol=0.3)
        np.testing.assert_allclose(v.y, dys, atol=0.3)

    def test_rolling_baseline_recovers_linear_drift(self):
        movie, dxs, dys = _make_drifting_movie(n_frames=6, size=41)
        indexes = np.arange(len(movie), dtype=int)
        v = _get_initial_vector(
            movie=movie,
            indexes=indexes,
            rolling_baseline=True,
            parallel=False,
            n_jobs=None,
            seed=_DE_SEED,
        )
        # Rolling baseline cumulates per-step shifts; with linear drift the
        # endpoint should still match the true total.
        assert v.x[-1] == pytest.approx(dxs[-1], abs=0.5)
        assert v.y[-1] == pytest.approx(dys[-1], abs=0.5)

    def test_first_frame_has_zero_shift(self):
        movie, _, _ = _make_drifting_movie(n_frames=4, size=41)
        v = _get_initial_vector(
            movie=movie,
            indexes=np.arange(len(movie), dtype=int),
            rolling_baseline=False,
            parallel=False,
            n_jobs=None,
            seed=_DE_SEED,
        )
        assert v.x[0] == pytest.approx(0.0, abs=0.3)
        assert v.y[0] == pytest.approx(0.0, abs=0.3)


# get_vector


class TestGetVector:
    def test_returns_drift_vector_with_movie_length(self):
        movie, _, _ = _make_drifting_movie(n_frames=8, size=41)
        v = get_vector(movie, frame_gap=2, parallel=False, seed=_DE_SEED)
        assert isinstance(v, DriftVector)
        assert len(v) == len(movie)

    def test_recovers_linear_drift(self):
        movie, dxs, dys = _make_drifting_movie(n_frames=10, size=41)
        v = get_vector(movie, frame_gap=2, parallel=False, seed=_DE_SEED)
        # Interpolated vector should track the true linear drift.
        np.testing.assert_allclose(v.x, dxs, atol=0.5)
        np.testing.assert_allclose(v.y, dys, atol=0.5)

    def test_rolling_baseline(self):
        movie, dxs, dys = _make_drifting_movie(n_frames=10, size=41)
        v = get_vector(
            movie, frame_gap=2, rolling_baseline=True, parallel=False, seed=_DE_SEED
        )
        assert len(v) == len(movie)
        # End-to-end drift should still match.
        assert v.x[-1] == pytest.approx(dxs[-1], abs=0.7)
        assert v.y[-1] == pytest.approx(dys[-1], abs=0.7)

    def test_frame_gap_one(self):
        movie, dxs, dys = _make_drifting_movie(n_frames=6, size=41)
        v = get_vector(movie, frame_gap=1, parallel=False, seed=_DE_SEED)
        np.testing.assert_allclose(v.x, dxs, atol=0.4)
        np.testing.assert_allclose(v.y, dys, atol=0.4)

    def test_threshold_lid_open_branch(self):
        """Sampled frames below threshold should be skipped (lid-open path)."""
        movie, _, _ = _make_drifting_movie(n_frames=12, size=41)
        # Zero out a contiguous block of frames to simulate lid-open.
        movie = movie.copy()
        movie[4:7] = 1e-6
        # Threshold chosen so only the lid-open frames fall below.
        threshold = float(np.sum(movie[0])) * 0.5
        v = get_vector(
            movie,
            frame_gap=2,
            threshold=threshold,
            threshold_margin=2,
            parallel=False,
            seed=_DE_SEED,
        )
        assert isinstance(v, DriftVector)
        assert len(v) == len(movie)

    def test_threshold_clamps_at_edges(self):
        """Lid-open region at frame 0 must not trigger negative-index wrap."""
        movie, _, _ = _make_drifting_movie(n_frames=10, size=41)
        movie = movie.copy()
        movie[0:2] = 1e-6
        threshold = float(np.sum(movie[2])) * 0.5
        v = get_vector(
            movie,
            frame_gap=2,
            threshold=threshold,
            threshold_margin=6,
            parallel=False,
            seed=_DE_SEED,
        )
        assert len(v) == len(movie)

    def test_velocity_outlier_rejection(self, caplog):
        """A single spurious jump should be caught by the 3σ velocity cut."""
        import logging

        movie, dxs, dys = _make_drifting_movie(n_frames=20, size=41)
        # Inject a spurious jump (within solver bounds) at one frame.
        movie = movie.copy()
        movie[10] = shift(movie[0], (5, 5))
        with caplog.at_level(logging.INFO, logger="tandemmp.drift.corrector"):
            v = get_vector(movie, frame_gap=1, parallel=False, seed=_DE_SEED)
        # The branch should have reported at least one bad point.
        bad_messages = [r for r in caplog.records if "Bad points" in r.message]
        assert bad_messages, "expected a 'Bad points' log message"
        assert len(v) == len(movie)

    def test_parallel_smoke(self):
        """Smoke test for parallel path — verifies picklability of the dispatch call."""
        movie, _, _ = _make_drifting_movie(n_frames=4, size=31)
        v = get_vector(movie, frame_gap=1, parallel=True, n_jobs=2, seed=_DE_SEED)
        assert isinstance(v, DriftVector)
        assert len(v) == len(movie)


# get_corrected_movie


class TestGetCorrectedMovie:
    def test_returns_movie_and_vector(self):
        movie, _, _ = _make_drifting_movie(n_frames=6, size=41)
        out, vector = get_corrected_movie(
            movie, frame_gap=2, parallel=False, seed=_DE_SEED
        )
        assert isinstance(vector, DriftVector)
        assert out.ndim == 3
        assert out.shape[0] == len(movie)

    def test_full_output_returns_slice(self):
        movie, _, _ = _make_drifting_movie(n_frames=6, size=41)
        out, vector, sel3d = get_corrected_movie(
            movie, frame_gap=2, parallel=False, full_output=True, seed=_DE_SEED
        )
        assert isinstance(sel3d, tuple)
        assert len(sel3d) == 3
        assert sel3d[0] == slice(None, None)

    def test_corrected_frames_match_each_other(self):
        movie, _, _ = _make_drifting_movie(n_frames=6, size=41)
        out, _ = get_corrected_movie(movie, frame_gap=2, parallel=False, seed=_DE_SEED)
        # After correction, the inter-frame RMSE should be much smaller
        # than the original.
        before = float(
            np.mean([_rmse(movie[0], movie[i]) for i in range(1, len(movie))])
        )
        after = float(np.mean([_rmse(out[0], out[i]) for i in range(1, len(out))]))
        assert after < before * 0.5

    def test_uses_provided_vector(self):
        movie, dxs, dys = _make_drifting_movie(n_frames=4, size=41)
        v = DriftVector(dxs.astype(float), dys.astype(float))
        out, returned = get_corrected_movie(movie, vector=v, parallel=False)
        # Identity comparison: returned vector is the one we passed in.
        assert returned is v
        assert out.shape[0] == len(movie)

    def test_length_mismatch_raises(self):
        movie, _, _ = _make_drifting_movie(n_frames=4, size=21)
        bad_vector = DriftVector(np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError, match="does not match"):
            get_corrected_movie(movie, vector=bad_vector, parallel=False)

    def test_cropped_to_visible_region(self):
        movie, dxs, dys = _make_drifting_movie(n_frames=5, size=41)
        v = DriftVector(dxs.astype(float), dys.astype(float))
        out, _ = get_corrected_movie(movie, vector=v, parallel=False)
        # Some cropping must have occurred (drift is nonzero).
        assert out.shape[1] < movie.shape[1] or out.shape[2] < movie.shape[2]


# get_vector_segmented


class TestGetVectorSegmented:
    def test_returns_one_vector_per_tile(self):
        movie, _, _ = _make_drifting_movie(n_frames=4, size=60)
        vectors = get_vector_segmented(
            movie, x_grid=2, y_grid=2, frame_gap=1, parallel=False, seed=_DE_SEED
        )
        assert len(vectors) == 4
        assert all(isinstance(v, DriftVector) for v in vectors)

    def test_each_tile_recovers_drift(self):
        # Tiles are size//2 per side; smaller frames give the cross-correlation
        # too few pixels to converge reliably and the test becomes flaky.
        # 100 px frames → 50 px tiles is the empirically stable size. `seed`
        # pins `differential_evolution` so this doesn't also flake on the
        # optimiser occasionally landing on the wrong local minimum.
        movie, dxs, dys = _make_drifting_movie(n_frames=4, size=100)
        vectors = get_vector_segmented(
            movie, x_grid=2, y_grid=2, frame_gap=1, parallel=False, seed=_DE_SEED
        )
        for v in vectors:
            assert len(v) == len(movie)
            assert v.x[-1] == pytest.approx(dxs[-1], abs=0.6)
            assert v.y[-1] == pytest.approx(dys[-1], abs=0.6)

    def test_default_grid(self):
        movie, _, _ = _make_drifting_movie(n_frames=3, size=60)
        vectors = get_vector_segmented(
            movie, frame_gap=1, parallel=False, seed=_DE_SEED
        )
        # Default: y_grid = max(60 // 30, 2) = 2, x_grid = max(60 // 30, 1) = 2
        assert len(vectors) == 4

    def test_partial_grid_specification_raises(self):
        movie, _, _ = _make_drifting_movie(n_frames=3, size=30)
        with pytest.raises(ValueError, match="both"):
            get_vector_segmented(movie, x_grid=2, frame_gap=1, parallel=False)


# illumination_correction


class TestIlluminationCorrection:
    def test_output_shape_matches_input(self):
        rng = np.random.default_rng(11)
        movie = rng.uniform(0.5, 1.5, size=(20, 30, 30))
        dc_movie = rng.uniform(0.5, 1.5, size=(20, 30, 30))
        out = illumination_correction(
            movie, dc_movie, space_smoothing=2, time_smoothing=5
        )
        assert out.shape == movie.shape

    def test_flattens_radial_illumination(self):
        """A movie with a radial illumination falloff should be flattened."""
        size = 41
        yy, xx = np.mgrid[:size, :size]
        cy, cx = size // 2, size // 2
        radius2 = (yy - cy) ** 2 + (xx - cx) ** 2
        # Smooth radial profile, all positive, peaked at the centre.
        profile = 1.0 + 0.5 * np.exp(-radius2 / (2 * 8.0**2))

        n_frames = 30
        rng = np.random.default_rng(12)
        # Each frame is the profile times a near-constant scalar.
        scalars = 1.0 + 0.01 * rng.normal(size=n_frames)
        movie = np.array([profile * s for s in scalars])
        dc_movie = movie.copy()

        out = illumination_correction(
            movie, dc_movie, space_smoothing=3, time_smoothing=5
        )
        # After correction, the spatial std should be much smaller than before.
        before_std = np.std(movie[0])
        after_std = np.std(out[0])
        assert after_std < before_std * 0.5

    def test_returns_ndarray(self):
        rng = np.random.default_rng(13)
        movie = rng.uniform(0.5, 1.5, size=(10, 20, 20))
        dc_movie = rng.uniform(0.5, 1.5, size=(10, 20, 20))
        out = illumination_correction(
            movie, dc_movie, space_smoothing=2, time_smoothing=3
        )
        assert isinstance(out, np.ndarray)

    def test_rejects_non_positive_pixels(self):
        movie = np.ones((5, 10, 10))
        dc_movie = np.ones((5, 10, 10))
        movie[2, 3, 3] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            illumination_correction(movie, dc_movie)
