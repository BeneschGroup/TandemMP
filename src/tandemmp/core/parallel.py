"""Generic picklable-kwargs dispatch shared by the parallel-processing modules.

Applies a module-level callable to a list of kwargs dicts, serially or via a
persistent :class:`multiprocessing.Pool`.

The pool is a module-level singleton kept for the life of the process, so costly
per-worker state (e.g. a lazy-loaded TensorFlow ghost model, set up once per
worker via :func:`functools.lru_cache`) is built at most once per worker. The
trade-off is that worker memory stays resident, so RAM scales with worker count.
Changing *n_jobs* between calls replaces the pool.

Not thread-safe: ``_pool`` and ``_pool_n_jobs`` are mutated without a lock, so
concurrent ``parallel=True`` calls from several threads can create two pools and
leak the first. All current callers are single-threaded; add a lock in
:func:`_get_pool` / :func:`shutdown_pool` if that changes.
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


def _dispatch_kwargs_call(
    args: tuple[Callable[..., Any], dict[str, Any]],
) -> Any:  # noqa: ANN401
    """Picklable helper: apply *method* to a single kwargs dict."""
    method, kwargs = args
    return method(**kwargs)


def _get_pool(n_jobs: int | None) -> multiprocessing.pool.Pool:
    """Return the persistent worker pool, recreating it if missing or resized."""
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

    Safe to call repeatedly. Registered once via :func:`atexit.register` when the
    first pool is created, so the pool doesn't outlive the process; it reads the
    current ``_pool`` at exit, so recreated pools are covered. Also useful in tests
    to avoid leaking workers.
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

    Uses the persistent pool (see module docstring) when *parallel* is ``True`` and
    *args_list* is non-empty, else a serial loop. *method* must be a module-level
    (picklable) callable.
    """
    if not args_list or not parallel:
        return [method(**kwargs) for kwargs in args_list]

    paired = [(method, kw) for kw in args_list]
    pool = _get_pool(n_jobs)
    return list(pool.map(_dispatch_kwargs_call, paired))
