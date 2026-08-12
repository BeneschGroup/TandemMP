# Drift-correction benchmark

Baseline (and, once the Rust port lands, comparison) timings for
`tandemmp.drift.corrector` / `tandemmp.pipeline.drift_correction`, the
WORKLIST.md priority-6 Rust-acceleration target (the
`scipy.optimize.differential_evolution` shift/RMSE inner loop).

## Method

`drift_correction_baseline.py` calls
`tandemmp.pipeline.drift_correction.run_drift_correction` on a real `.mp`
movie with `frame_gap=25` and a fixed `seed=42` (for reproducibility — see
`corrector._solve`'s docstring), once with `parallel=False` (single core —
the fairer comparison against a future Rust rewrite, since it isolates
algorithmic speed from multiprocessing overhead) and once with the
production default `parallel=True`.

```bash
python benchmarks/drift_correction_baseline.py /path/to/movie.mp
```

## Baseline (Python, pre-Rust-port)

Benchmark file: `012_20250717_beta_gal_6nM.mp` — 3000 frames, 59x150 px,
120 frames sampled at `frame_gap=25`. Measured on a 20-logical-core machine.

| Mode | Elapsed |
|---|---|
| Serial (`parallel=False`) | **72.37 s** |
| Parallel (`parallel=True`, default `n_jobs`) | **21.69 s** |

That's ~0.60 s per sampled frame serial (matches a standalone single-frame
`_solve` timing of ~0.71 s including call overhead). See `results.json` for
the machine-readable record; `rust_port` there is filled in once the Rust
port is benchmarked on the same file with the same parameters.
