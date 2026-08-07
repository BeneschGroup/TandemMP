# TandemMP Worklist

## Rust acceleration targets

Listed in priority order. Each should be implemented as a PyO3 crate under `src/tandemmp/_rust/`. Acceleration should only be attempted after the full refactor is complete.

Before any Rustification begins, Python-side PSF interface should be complete (described in 6m of the refactor guide). A
stable, generalisable PSF abstraction should make it easier to move hot paths into Rust without
coupling the Rust layer to a single PSF implementation.

> **Remove dependence on old `deprecated.rust` crates** — `src/tandemmp/fitting/psf.py` currently
> lazy-imports `deprecated.rust.psf_a12ws`. The pre-refactor `tandemmp/rust/` PyO3 crate source
> (`fit_psf`, `psf_a12ws`, `psf_traits`) was moved to `deprecated/rust/` once the rest of the old
> flat `tandemmp/` package was deleted (Layer 6), since `psf.py`'s `use_rust=True` backend still
> depends on it and it isn't part of the installed `tandemmp` distribution — same treatment as
> `deprecated/postprocessing.py` (4d). This is the only place in `src/` that reaches into
> `deprecated/`, and it forces a mypy override (`ignore_errors = true` for `deprecated.rust.*` in
> `pyproject.toml`) to suppress type errors from untyped pre-refactor code. When the PSF fitting
> hot path is Rustified (priority 1 above), the new PyO3 crate under `src/tandemmp/_rust/` should
> replace this import entirely, allowing the mypy override and `deprecated/rust/` to be removed.

| Priority | Target | Current location | Bottleneck |
|---|---|---|---|
| 1 | Batch PSF fitting | `utils/fit.py` `_fit_peak_core()` | Per-peak `scipy.optimize.minimize` |
| 2 | 3D ratiometric fitting | `utils/fit.py` `fit_3d_ratiometric()` | Per-event multi-hypothesis optimisation |
| 3 | Distance matrix for pair matching | `manager.py` `find_best_matches()` | O(n²) Python list comprehension |
| 4 | Ghost overlap probability | `investigator.py` `match_ghost()` | Per-event Gaussian overlap loop |
| 5 | Batch Kalman updates | `utils/kalman.py` | Per-particle per-frame matrix ops |
| 6 | Drift correction shift+RMSE | `DriftCorrector.py` | `differential_evolution` inner loop |

## Conventions

- **Python >=3.10**, type-annotated public APIs (`mypy --strict` on `src/`).
- **ruff** for linting and formatting (line length 88).
- **pytest** for all tests; tests live in `tests/` mirroring the `src/` structure.
- **No `tkinter` dialogs in library code** — file paths are passed as arguments, not selected
  interactively. Interactive file selection belongs in CLI entry points or notebooks only.
- **No wildcard imports** (`from module import *`).
- **Dataclasses or NamedTuples** over ad-hoc dicts for structured data.
- **Snake_case** for all module filenames (no `PCA.py`, `DriftCorrector.py`).

## CI strategy

CI workflows are **enabled** and scoped to refactored code only:

- **Triggers** — `push` to `main`/`refactor` and all PRs, filtered by `paths: src/**, tests/**,
  pyproject.toml`. Changes to old `tandemmp/` directory do not trigger CI.
- **ruff** — `ruff check src/ tests/` and `ruff format --check src/ tests/`.
- **mypy** — `mypy src/`.
- **pytest** — `testpaths = ["tests"]` in `pyproject.toml`.
- **pip-audit** — security scan on declared dependencies.
- **Claude Code Review** — runs on PRs touching `src/` or `tests/`, scoped to those paths.

## Running tests

```bash
pip install ".[dev]"
pytest
```

## Running lints

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
```

## open_iscat dependency

`open_iscat` is a private GitHub repo and is **not listed in `pyproject.toml`**. It is not on
PyPI and its codebase has not been updated for Python >=3.10, so including it as a dependency
would break CI.

**Current workaround:** `pipeline/open_iscat.py` uses lazy imports — `open_iscat`, `h5py`, and
`nptdms` are imported inside function bodies (`load_mp_file`, `run_pipeline`,
`_read_mp_metadata`, `_read_tdms_metadata`), not at module level. The data containers (`Info`,
`Movie`, `PipelineResults`) and their tests work without any of these packages installed.

**Revisit after refactor is complete:**
- [ ] Refactor `open_iscat` to support Python >=3.10.
- [ ] Publish `open_iscat` to PyPI (or a private index) or use a git URL dependency.
- [ ] Add `open_iscat` to `pyproject.toml` dependencies (or an optional extra).
- [ ] Move `h5py` and `nptdms` from lazy imports to declared dependencies.
- [ ] Add integration tests for `load_mp_file` and `run_pipeline` that exercise the full
  `open_iscat` pipeline.
