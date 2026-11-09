"""Shared pytest fixtures for the whole test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _shutdown_persistent_pool() -> Iterator[None]:
    """Terminate `core.parallel`'s persistent worker pool after every test.

    Parallel dispatch reuses a module-level pool. Shut it down so worker
    processes do not outlive the test that created them.
    """
    yield
    from tandemmp.core.parallel import shutdown_pool

    shutdown_pool()


@pytest.fixture(autouse=True)
def _reset_active_calibration() -> Iterator[None]:
    """Reset `core.conversions`'s active calibration after every test.

    Calibration is process-wide state; overrides must not leak between tests.
    """
    yield
    from tandemmp.core.conversions import set_calibration

    set_calibration(None)
