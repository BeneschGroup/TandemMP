# TandemMP

Pair matching and particle tracking analysis for Mass Photometry data.

## Installation

```bash
pip install -e /path_to_repo/.
```

TandemMP includes two native extensions written in Rust (`drift_correction`
and `spline_psf`, under `src/tandemmp/_rust/`). Compiled binaries are never
committed to git; CI builds them (see `.github/workflows/wheels.yml`).

**From a release wheel** (Windows x86_64, Linux x86_64/aarch64; Python >= 3.10):
`pip install <wheel>` needs no compiler.

**From source** (any platform, including macOS, which has no prebuilt wheels):
you need a [Rust toolchain](https://rustup.rs), **version 1.88 or newer**
(`rustup update`). The extensions are compiled during `pip install`, which takes
a few minutes the first time because they are built with full optimisation.

Release builds embed the builder's absolute source paths. If you build a wheel
on your own machine to share, strip them and check the result:

```bash
# Linux / macOS / Git Bash
RUSTFLAGS="--remap-path-prefix=$HOME=/home/builder" pip wheel . --no-deps -w dist
python scripts/check_wheel_paths.py dist/*.whl
```

```powershell
# Windows PowerShell (if CARGO_HOME or RUSTUP_HOME live outside your home
# directory, add a --remap-path-prefix for those too)
$env:RUSTFLAGS = "--remap-path-prefix=$env:USERPROFILE=/home/builder"
pip wheel . --no-deps -w dist
python scripts/check_wheel_paths.py (Get-ChildItem dist\*.whl)
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
