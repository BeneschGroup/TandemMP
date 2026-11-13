"""Shared pytest fixtures for the whole test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _shutdown_persistent_pool() -> Iterator[None]:
    """Shut down `core.parallel`'s persistent pool after every test.

    Parallel dispatch reuses a module-level pool; this stops workers outliving the
    test that created them.
    """
    yield
    from tandemmp.core.parallel import shutdown_pool

    shutdown_pool()


@pytest.fixture(autouse=True)
def _reset_active_calibration() -> Iterator[None]:
    """Reset `core.conversions`'s active calibration after every test.

    It is process-wide state, so overrides must not leak between tests.
    """
    yield
    from tandemmp.core.conversions import set_calibration

    set_calibration(None)
