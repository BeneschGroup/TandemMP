"""Generic picklable-kwargs dispatch shared by parallel-processing modules.

Applies a module-level callable to a list of keyword-argument dicts, either
serially or via a persistent :class:`multiprocessing.Pool`.

The pool is a module-level singleton that persists for the lifetime of the
process rather than being torn down and recreated on every
:func:`dispatch_kwargs` call. This means expensive per-worker runtime state
(for example a lazy-loaded TensorFlow ghost model, initialised once per
worker via :func:`functools.lru_cache`) is set up at most once per worker
process and then reused across every subsequent call, instead of being
reloaded every time a fresh pool spins up. The trade-off is that each
worker's in-memory caches/model instances stay resident for the process
lifetime, so RAM use scales with worker count rather than being reclaimed
between calls.

Changing *n_jobs* between calls tears down the existing pool and starts a
new one with the requested worker count.

Not thread-safe: the module-level pool state (``_pool``, ``_pool_n_jobs``)
is mutated without a lock, so calling :func:`dispatch_kwargs` with
``parallel=True`` concurrently from multiple threads in the same process
can race two pools into existence, silently leaking the first (its
reference is overwritten before it can be terminated). All current call
sites are single-threaded, so this is a documented assumption rather than
an active bug; add a lock in :func:`_get_pool`/:func:`shutdown_pool` if a
concurrent caller is ever introduced.
"""

from __future__ import annotations

import atexit
import multiprocessing
import multiprocessing.pool
from collections.abc import Callable
from typing import Any

_pool: multiprocessing.pool.Pool | None = None
_pool_n_jobs: int | None = None
_atexit_registered = False


def _dispatch_kwargs_call(args: tuple[Callable[..., Any], dict[str, Any]]) -> Any:  # noqa: ANN401
    """Picklable helper — apply *method* to a single kwargs dict."""
    method, kwargs = args
    return method(**kwargs)


def _get_pool(n_jobs: int | None) -> multiprocessing.pool.Pool:
    """Return the persistent worker pool, sized to *n_jobs*.

    Recreates the pool if it doesn't exist yet, or if *n_jobs* differs from
    the pool's current worker count.
    """
    global _pool, _pool_n_jobs, _atexit_registered
    if _pool is not None and _pool_n_jobs != n_jobs:
        _pool.terminate()
        _pool.join()
        _pool = None
    if _pool is None:
        _pool = multiprocessing.Pool(processes=n_jobs)
        _pool_n_jobs = n_jobs
        if not _atexit_registered:
            atexit.register(shutdown_pool)
            _atexit_registered = True
    return _pool


def shutdown_pool() -> None:
    """Terminate and clear the persistent worker pool, if one exists.

    Safe to call multiple times. The first time a pool is ever created,
    this function is registered once via :func:`atexit.register` so the
    pool doesn't outlive the process — the same registration is reused for
    every pool recreated afterwards (e.g. after an *n_jobs* change), since
    it always reads the current ``_pool`` at exit time rather than closing
    over a specific pool instance. Also useful to call directly in tests,
    to avoid leaking worker processes across the test session.
    """
    global _pool, _pool_n_jobs
    if _pool is not None:
        _pool.terminate()
        _pool.join()
        _pool = None
    _pool_n_jobs = None


def dispatch_kwargs(
    method: Callable[..., Any],
    args_list: list[dict[str, Any]],
    *,
    parallel: bool,
    n_jobs: int | None = 4,
) -> list[Any]:
    """Apply *method* to each kwargs dict in *args_list*.

    Runs via the persistent :class:`multiprocessing.Pool` (see module
    docstring) when *parallel* is ``True`` and *args_list* is non-empty;
    otherwise falls back to a serial list comprehension. *method* must be a
    module-level (picklable) callable.
    """
    if not args_list or not parallel:
        return [method(**kwargs) for kwargs in args_list]

    paired = [(method, kw) for kw in args_list]
    pool = _get_pool(n_jobs)
    return list(pool.map(_dispatch_kwargs_call, paired))
