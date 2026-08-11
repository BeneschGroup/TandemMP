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
- **`src/tandemmp/pipeline/calibration.py`** — generates a contrast-to-mass
  calibration from a calibration movie (`run_calibration`), which can then be
  passed into a batch run or loaded from a saved calibration file
  (`save_calibration` / `load_calibration`).
- **`src/tandemmp/pipeline/simple_analysis.py`** — pair matching analysis for
  CSV-exported point-events, without re-running PSF fitting.
- **`src/tandemmp/pipeline/drift_correction.py`** — estimates a movie's drift
  vector without running a full analysis (`run_drift_correction`). Saves
  either as plain JSON (`save_drift_vector` / `load_drift_vector`) or as a
  pickled callable that corrects arbitrary `(x, y, i)` positions directly
  (`save_correction_function` / `load_correction_function`).

## Tests

```bash
pytest
```
