# TandemMP

Pair matching and particle tracking analysis for Mass Photometry data.

## Installation

One-time prerequisite: a Rust toolchain ([rustup](https://rustup.rs)), plus,
on Windows, the MSVC Build Tools (the "Desktop development with C++"
workload). `pip install` then compiles the native extension automatically.

```bash
pip install -e /path_to_repo/.
```

## Requirements
`h5py`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `tqdm`, `opencv-python`, `tensorflow`.

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
