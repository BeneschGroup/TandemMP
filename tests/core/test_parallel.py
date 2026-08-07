"""Tests for tandemmp.core.parallel — shared picklable-kwargs dispatch.

Pool cleanup between tests is handled by the autouse
``_shutdown_persistent_pool`` fixture in ``tests/conftest.py`` (applies
suite-wide, not just to this module).
"""

from __future__ import annotations

from tandemmp.core.conversions import NM_PER_PIXEL, nm2pixels
from tandemmp.core.parallel import dispatch_kwargs, shutdown_pool


def _double(x: int) -> int:
    return x * 2


class TestDispatchKwargs:
    """Parallel/serial dispatch helper — shared by drift/corrector and pipeline/investigator."""

    def test_serial_empty(self):
        assert dispatch_kwargs(_double, [], parallel=False) == []

    def test_serial_runs(self):
        out = dispatch_kwargs(_double, [{"x": 1}, {"x": 2}], parallel=False)
        assert out == [2, 4]

    def test_parallel_empty_still_serial(self):
        # Empty list → skip pool, avoids spawn overhead in tests.
        assert dispatch_kwargs(_double, [], parallel=True) == []

    def test_parallel_runs_via_pool(self):
        # Uses a real src-defined function (not one local to this test module) —
        # a worker spawned via Pool.map must be able to re-import its target,
        # which a pytest-collected test function is not guaranteed to support.
        args = [{"nm": NM_PER_PIXEL}, {"nm": NM_PER_PIXEL * 2}]
        out = dispatch_kwargs(nm2pixels, args, parallel=True, n_jobs=2)
        assert out == [1.0, 2.0]


class TestPersistentPool:
    """The worker pool persists across calls instead of being torn down."""

    def test_pool_identity_persists_across_calls(self):
        args = [{"nm": NM_PER_PIXEL}]
        dispatch_kwargs(nm2pixels, args, parallel=True, n_jobs=2)
        from tandemmp.core.parallel import _pool as pool_after_first

        dispatch_kwargs(nm2pixels, args, parallel=True, n_jobs=2)
        from tandemmp.core.parallel import _pool as pool_after_second

        assert pool_after_first is not None
        assert pool_after_first is pool_after_second

    def test_pool_recreated_when_n_jobs_changes(self):
        args = [{"nm": NM_PER_PIXEL}]
        dispatch_kwargs(nm2pixels, args, parallel=True, n_jobs=2)
        from tandemmp.core.parallel import _pool as pool_with_two_jobs

        dispatch_kwargs(nm2pixels, args, parallel=True, n_jobs=3)
        from tandemmp.core.parallel import _pool as pool_with_three_jobs

        assert pool_with_two_jobs is not pool_with_three_jobs

    def test_shutdown_pool_is_idempotent(self):
        shutdown_pool()
        shutdown_pool()

    def test_shutdown_pool_clears_pool(self):
        args = [{"nm": NM_PER_PIXEL}]
        dispatch_kwargs(nm2pixels, args, parallel=True, n_jobs=2)
        shutdown_pool()
        from tandemmp.core.parallel import _pool

        assert _pool is None
