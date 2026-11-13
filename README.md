# TandemMP

Pair matching and particle tracking analysis for Mass Photometry data.

## Installation

```bash
pip install -e /path_to_repo/.
```

TandemMP includes two native extensions written in Rust (`drift_correction`
and `spline_psf`, under `src/tandemmp/_rust/`). They are compiled when you
install from source, so you need a [Rust toolchain](https://rustup.rs). Release
wheels (Windows and Linux; Python >= 3.10; no macOS wheels) ship them prebuilt, so
`pip install <wheel>` needs no compiler. Compiled binaries are never committed
to git; CI builds them (see `.github/workflows/wheels.yml`).

Release builds embed the builder's absolute source paths. If you build a wheel
on your own machine to share, strip them and check the result:

```bash
RUSTFLAGS="--remap-path-prefix=$HOME=/home/user" pip wheel . --no-deps -w dist
python scripts/check_wheel_paths.py dist/*.whl
```

## Requirements
`open_iscat`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `tqdm`, `opencv-python`, `tensorflow`.
`open_iscat` is an internal dependency; we are working on an alternative.

## Entry points

- **`src/tandemmp/pipeline/batch.py`** — main entry point for particle
  tracking analysis. Runs the full event-finding, tracking, and pair-matching
  pipeline over one or many movies (`process_movie` / `batch_process`).
- **`src/tandemmp/pipeline/calibration.py`** — generates a contrast-to-mass
  calibration from a calibration movie (`run_calibration`), which can then be
  passed into a batch run or loaded from a saved calibration file
  (`save_calibration` / `load_calibration`).
- **`src/tandemmp/pipeline/simple_analysis.py`** — pair matching analysis for
  CSV-exported point-events, without re-running PSF fitting.
- **`src/tandemmp/pipeline/drift_correction.py`** — estimates a movie's drift
  vector without running a full analysis (`run_drift_correction`), saved as
  JSON or as a pickled `DriftCorrectionFunction` (`save_drift_vector` /
  `save_correction_function`) that can correct coordinates or a whole movie.
- **`src/tandemmp/pipeline/pair_matching.py`** — loads a movie, drift-corrects
  it, and matches binding/unbinding event pairs (`run_pair_matching`) without
  running the full PSF-fit-based tracking pipeline. Returns a
  `PairMatchingResult`; plotting is left for the caller to wire up.

## Tests

```bash
pytest
```
