# TandemMP

Pair matching and particle tracking analysis for Mass Photometry data.

## Installation

```bash
pip install -e /path_to_repo/.
```

**From a release wheel** (Windows, Linux; Python >= 3.10):
`pip install <wheel>` 

## Requirements
`numpy`, `scipy`, `matplotlib`, `scikit-learn`, `tqdm`, `opencv-python`, `h5py`, `tensorflow`.

## Workflow

TandemMP does not find events itself.

1. Find events in the raw `.mp` movie with DiscoverMP and export them as a `.csv`.
2. Run an entry point below and select the `.mp` and its `.csv`. The two file
   names must match (ie. `sample1.mp` and `sample1.csv`).

Drift is estimated from the raw movie frames, and the same drift vector corrects
the event positions.

**Calibration.** Masses use a linear contrast-to-mass calibration. Pass a
DiscoverMP `.mc` file (or a `.json` / `.pkl` with `c2kDa` and `intercept`) as
`calibration=`; see `tandemmp.io.calibration.load_calibration`. Without one, a
default calibration applies.

## Entry points

- **`src/tandemmp/pipeline/batch.py`** — main entry point for particle
  tracking analysis. Runs the full tracking and pair-matching pipeline over one
  or many movies with their events CSVs (`process_movie` / `batch_process`).
- **`src/tandemmp/pipeline/drift_correction.py`** — estimates a movie's drift
  vector without running a full analysis (`run_drift_correction`), saved as
  JSON or as a pickled `DriftCorrectionFunction` (`save_drift_vector` /
  `save_correction_function`) that can correct coordinates or a whole movie.
- **`src/tandemmp/pipeline/pair_matching.py`** — loads a movie and its events
  CSV, drift-corrects them, and matches binding/unbinding event pairs
  (`run_pair_matching`) without running the full PSF-fit-based tracking
  pipeline. Returns a `PairMatchingResult`; plotting is left for the caller to
  wire up.

## Tests

```bash
pytest
```
