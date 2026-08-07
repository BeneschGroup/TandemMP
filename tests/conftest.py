"""Shared pytest fixtures for the whole test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _shutdown_persistent_pool() -> Iterator[None]:
    """Terminate `core.parallel`'s persistent worker pool after every test.

    `core.parallel.dispatch_kwargs` now reuses a module-level
    `multiprocessing.Pool` singleton across calls (see 6l) instead of
    tearing one down per call. Any test that exercises real parallel
    dispatch (`parallel=True` without monkeypatching `dispatch_kwargs`,
    e.g. `tests/drift/test_corrector.py::test_parallel_smoke`) would
    otherwise leave live worker subprocesses running for the rest of the
    pytest session, reaped only by the `atexit` hook when the whole test
    process exits.
    """
    yield
    from tandemmp.core.parallel import shutdown_pool

    shutdown_pool()


@pytest.fixture(autouse=True)
def _reset_active_calibration() -> Iterator[None]:
    """Reset `core.conversions`'s active calibration after every test.

    `set_calibration`/`get_calibration` (see the calibration feature) are
    process-wide mutable state read by `c_to_mass`/`mass_to_c` on every
    call. Any test that overrides it (directly, or indirectly via
    `pipeline.batch.process_movie`) would otherwise leak that override into
    every subsequent test in the session.
    """
    yield
    from tandemmp.core.conversions import set_calibration

    set_calibration(None)
