"""Tests for the Rust-backed (``use_rust=True``) path of
``tandemmp.drift.corrector``.

Mirrors the relevant cases from ``TestSolve``/``TestStabilise`` in
``tests/drift/test_corrector.py`` (which must keep passing unmodified — see
that file), plus direct tests of the PyO3 binding layer
(``tandemmp._rust.drift_correction``) and of the ``use_rust=True`` ->
Python-fallback degradation path.

The whole module is skipped (not failed) when the compiled extension
hasn't been built (``maturin develop`` from
``src/tandemmp/_rust/drift_correction/``) — this keeps CI green for anyone
running a plain ``pip install ".[dev]"`` with no Rust toolchain/build step,
per WORKLIST.md's Rust-acceleration convention.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.ndimage import shift

pytest.importorskip("tandemmp._rust.drift_correction")

import tandemmp._rust.drift_correction as _dc  # noqa: E402
from tandemmp.drift.corrector import (  # noqa: E402
    DriftVector,
    _rmse,
    _shift_compare,
    _solve,
    _solve_rust,
    get_corrected_movie,
    get_vector,
    stabilise,
)

_DE_SEED = 42


def _make_drifting_movie(
    n_frames: int = 12,
    size: int = 41,
    drift_per_frame: tuple[float, float] = (0.4, -0.3),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same synthetic-movie generator as ``test_corrector.py``."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(size, size)).astype(float)
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


# ---------------------------------------------------------------------------
# Direct binding-level tests (tandemmp._rust.drift_correction)
# ---------------------------------------------------------------------------


class TestBindingRmse:
    def test_zero_for_identical(self):
        import sys

        print("DEBUG _dc.__file__ =", getattr(_dc, "__file__", None), file=sys.stderr)
        print("DEBUG _dc.__spec__ =", getattr(_dc, "__spec__", None), file=sys.stderr)
        print("DEBUG _dc.__loader__ =", getattr(_dc, "__loader__", None), file=sys.stderr)
        print("DEBUG _dc.__path__ =", getattr(_dc, "__path__", "NO __path__"), file=sys.stderr)
        print("DEBUG dir(_dc) =", sorted(dir(_dc)), file=sys.stderr)
        import tandemmp._rust as _rust_pkg

        print("DEBUG _rust_pkg.__path__ =", list(_rust_pkg.__path__), file=sys.stderr)
        import importlib

        fresh = importlib.import_module("tandemmp._rust.drift_correction")
        print("DEBUG fresh is _dc:", fresh is _dc, file=sys.stderr)
        print("DEBUG fresh.__file__ =", getattr(fresh, "__file__", None), file=sys.stderr)
        print("DEBUG has rmse on fresh:", hasattr(fresh, "rmse"), file=sys.stderr)

        a = np.array([1.0, 2.0, 3.0])
        assert _dc.rmse(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert _dc.rmse(a, b) == pytest.approx(np.sqrt(12.5))

    def test_matches_python_rmse(self):
        rng = np.random.default_rng(21)
        a = rng.normal(size=(10, 10))
        b = rng.normal(size=(10, 10))
        assert _dc.rmse(a, b) == pytest.approx(_rmse(a, b), abs=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            _dc.rmse(np.zeros(3), np.zeros(4))


class TestBindingShiftCompare:
    def test_zero_shift_identical_frames(self):
        rng = np.random.default_rng(22)
        f = rng.normal(0, 1, size=(20, 20))
        assert _dc.shift_compare(0.0, 0.0, f, f) == pytest.approx(0.0, abs=1e-9)

    def test_matches_python_shift_compare(self):
        rng = np.random.default_rng(23)
        base = rng.normal(0, 1, size=(25, 25))
        true_dx, true_dy = 1.0, -2.0
        shifted = shift(base, (true_dy, true_dx))

        rust_val = _dc.shift_compare(true_dx, true_dy, shifted, base)
        python_val = _shift_compare((true_dx, true_dy), shifted, base)
        assert rust_val == pytest.approx(python_val, abs=1e-6)


class TestBindingStabilise:
    def test_zero_shift_is_identity(self):
        frame = np.arange(25, dtype=float).reshape(5, 5)
        out = _dc.stabilise(frame, 0.0, 0.0)
        np.testing.assert_allclose(out, frame, atol=1e-10)

    def test_matches_python_stabilise(self):
        rng = np.random.default_rng(24)
        base = rng.normal(0, 1, size=(15, 15))
        shifted = shift(base, (2, -1))
        rust_out = _dc.stabilise(shifted, -1.0, 2.0)
        python_out = stabilise(shifted, x=-1.0, y=2.0)
        np.testing.assert_allclose(rust_out, python_out, atol=1e-9)


class TestBindingSolve:
    def test_recovers_known_integer_shift(self):
        rng = np.random.default_rng(25)
        base = rng.normal(0, 1, size=(31, 31))
        true_dx, true_dy = 2.0, -1.0
        shifted = shift(base, (true_dy, true_dx))
        x, y = _dc.solve(shifted, base, (-6.0, 6.0), _DE_SEED)
        assert x == pytest.approx(true_dx, abs=0.2)
        assert y == pytest.approx(true_dy, abs=0.2)

    def test_same_seed_gives_reproducible_result(self):
        rng = np.random.default_rng(26)
        base = rng.normal(0, 1, size=(31, 31))
        shifted = shift(base, (-1.0, 2.0))
        first = _dc.solve(shifted, base, (-6.0, 6.0), 123)
        second = _dc.solve(shifted, base, (-6.0, 6.0), 123)
        assert first == second


# ---------------------------------------------------------------------------
# corrector.py's use_rust=True path
# ---------------------------------------------------------------------------


class TestStabiliseUseRust:
    def test_zero_shift_is_identity(self):
        frame = np.arange(25, dtype=float).reshape(5, 5)
        out = stabilise(frame, 0.0, 0.0, use_rust=True)
        np.testing.assert_allclose(out, frame, atol=1e-10)

    def test_undoes_known_shift(self):
        rng = np.random.default_rng(1)
        base = rng.normal(0, 1, size=(15, 15))
        shifted = shift(base, (2, -1))
        out = stabilise(shifted, x=-1.0, y=2.0, use_rust=True)
        np.testing.assert_allclose(out[3:-3, 3:-3], base[3:-3, 3:-3], atol=1e-6)

    def test_matches_python_path(self):
        """Rust and Python backends should agree closely on the same input."""
        rng = np.random.default_rng(2)
        base = rng.normal(0, 1, size=(20, 20))
        shifted = shift(base, (1.5, -0.7))
        rust_out = stabilise(shifted, x=-0.7, y=1.5, use_rust=True)
        python_out = stabilise(shifted, x=-0.7, y=1.5, use_rust=False)
        np.testing.assert_allclose(rust_out, python_out, atol=1e-6)


class TestSolveUseRust:
    def test_recovers_known_integer_shift(self):
        rng = np.random.default_rng(4)
        base = rng.normal(0, 1, size=(31, 31))
        true_dx, true_dy = 2.0, -1.0
        shifted = shift(base, (true_dy, true_dx))
        x, y = _solve(
            input_frame=shifted, baseline_frame=base, seed=_DE_SEED, use_rust=True
        )
        assert x == pytest.approx(true_dx, abs=0.2)
        assert y == pytest.approx(true_dy, abs=0.2)

    def test_recovers_subpixel_shift(self):
        rng = np.random.default_rng(5)
        base = rng.normal(0, 1, size=(31, 31))
        true_dx, true_dy = 0.7, -1.3
        shifted = shift(base, (true_dy, true_dx))
        x, y = _solve(
            input_frame=shifted, baseline_frame=base, seed=_DE_SEED, use_rust=True
        )
        assert x == pytest.approx(true_dx, abs=0.3)
        assert y == pytest.approx(true_dy, abs=0.3)

    def test_zero_shift_for_identical_frames(self):
        rng = np.random.default_rng(6)
        f = rng.normal(0, 1, size=(31, 31))
        x, y = _solve(input_frame=f, baseline_frame=f, seed=_DE_SEED, use_rust=True)
        assert x == pytest.approx(0.0, abs=0.3)
        assert y == pytest.approx(0.0, abs=0.3)

    def test_seed_gives_reproducible_result(self):
        rng = np.random.default_rng(7)
        base = rng.normal(0, 1, size=(31, 31))
        shifted = shift(base, (-1.0, 2.0))
        first = _solve(
            input_frame=shifted, baseline_frame=base, seed=123, use_rust=True
        )
        second = _solve(
            input_frame=shifted, baseline_frame=base, seed=123, use_rust=True
        )
        np.testing.assert_array_equal(first, second)

    def test_generator_seed_falls_back_to_python(self, caplog):
        """A `np.random.Generator` seed isn't supported by the Rust path
        (it only accepts a plain int/None) — `_solve` should silently fall
        back to the Python `differential_evolution` implementation rather
        than erroring."""
        rng = np.random.default_rng(8)
        base = rng.normal(0, 1, size=(31, 31))
        shifted = shift(base, (1.0, -1.0))
        with caplog.at_level(logging.DEBUG, logger="tandemmp.drift.corrector"):
            result = _solve(
                input_frame=shifted,
                baseline_frame=base,
                seed=np.random.default_rng(_DE_SEED),
                use_rust=True,
            )
        assert result.shape == (2,)
        assert any("falling back to Python" in r.message for r in caplog.records)


class TestUseRustFallback:
    """Verify the ImportError -> Python-fallback degradation path itself,
    by simulating the extension being unavailable (rather than relying on
    it actually being absent, which `importorskip` above already rules out
    for this test run)."""

    def test_solve_falls_back_when_rust_unavailable(self, monkeypatch, caplog):
        import tandemmp.drift.corrector as corrector

        def _raise_import_error(*args, **kwargs):
            raise ImportError("simulated: extension not built")

        monkeypatch.setattr(corrector, "_solve_rust", _raise_import_error)

        rng = np.random.default_rng(9)
        base = rng.normal(0, 1, size=(20, 20))
        shifted = shift(base, (1.0, -1.0))
        with caplog.at_level(logging.DEBUG, logger="tandemmp.drift.corrector"):
            result = corrector._solve(
                input_frame=shifted, baseline_frame=base, seed=_DE_SEED, use_rust=True
            )
        assert result.shape == (2,)
        assert any("unavailable" in r.message for r in caplog.records)

    def test_stabilise_falls_back_when_rust_unavailable(self, monkeypatch, caplog):
        import tandemmp.drift.corrector as corrector

        def _raise_import_error(*args, **kwargs):
            raise ImportError("simulated: extension not built")

        monkeypatch.setattr(corrector, "_stabilise_rust", _raise_import_error)

        frame = np.arange(25, dtype=float).reshape(5, 5)
        with caplog.at_level(logging.DEBUG, logger="tandemmp.drift.corrector"):
            out = corrector.stabilise(frame, 0.0, 0.0, use_rust=True)
        np.testing.assert_allclose(out, frame, atol=1e-10)
        assert any("unavailable" in r.message for r in caplog.records)


class TestSolveRustHelperTypeError:
    def test_non_int_seed_raises(self):
        """`_solve_rust` itself should reject non-int/None seeds (caught by
        `_solve`'s broad `except Exception` — see `TestSolveUseRust`)."""
        rng = np.random.default_rng(10)
        base = rng.normal(0, 1, size=(10, 10))
        with pytest.raises(TypeError):
            _solve_rust(base, base, (-6.0, 6.0), np.random.default_rng(1))


# ---------------------------------------------------------------------------
# End-to-end smoke tests through get_vector / get_corrected_movie
# ---------------------------------------------------------------------------


class TestGetVectorUseRust:
    def test_recovers_linear_drift(self):
        movie, dxs, dys = _make_drifting_movie(n_frames=8, size=41)
        v = get_vector(movie, frame_gap=2, parallel=False, seed=_DE_SEED, use_rust=True)
        assert isinstance(v, DriftVector)
        assert len(v) == len(movie)
        np.testing.assert_allclose(v.x, dxs, atol=0.5)
        np.testing.assert_allclose(v.y, dys, atol=0.5)


class TestGetCorrectedMovieUseRust:
    def test_returns_movie_and_vector(self):
        movie, _, _ = _make_drifting_movie(n_frames=6, size=41)
        out, vector = get_corrected_movie(
            movie, frame_gap=2, parallel=False, seed=_DE_SEED, use_rust=True
        )
        assert isinstance(vector, DriftVector)
        assert out.ndim == 3
        assert out.shape[0] == len(movie)

    def test_corrected_frames_match_each_other(self):
        movie, _, _ = _make_drifting_movie(n_frames=6, size=41)
        out, _ = get_corrected_movie(
            movie, frame_gap=2, parallel=False, seed=_DE_SEED, use_rust=True
        )
        before = float(
            np.mean([_rmse(movie[0], movie[i]) for i in range(1, len(movie))])
        )
        after = float(np.mean([_rmse(out[0], out[i]) for i in range(1, len(out))]))
        assert after < before * 0.5
