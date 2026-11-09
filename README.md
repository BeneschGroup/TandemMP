# TandemMP

Pair matching and particle tracking analysis for Mass Photometry data.

## Installation

```bash
pip install -e /path_to_repo/.
```

## Requirements
`open_iscat`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `tqdm`, `opencv-python`, `tensorflow`.
`open_iscat` is a private codebase.

## Entry points

- **`src/tandemmp/pipeline/batch.py`** — main entry point for particle
  tracking analysis. Runs the full event-finding, tracking, and pair-matching
  pipeline over one or many movies (`process_movie` / `batch_process`).
- **`src/tandemmp/pipeline/calibration.py`** — loads a contrast-to-mass
  calibration from a Refeyn `.mc` file (`load_mc`) or TandemMP's own JSON
  format (`load_calibration`, which dispatches on suffix; `save_calibration`
  to write one), to pass into a batch run or other entry point.
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
