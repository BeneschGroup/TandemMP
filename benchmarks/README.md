# Drift-correction benchmark

Baseline and Rust-port comparison timings for `tandemmp.drift.corrector` /
`tandemmp.pipeline.drift_correction`, the WORKLIST.md priority-6
Rust-acceleration target (the `scipy.optimize.differential_evolution`
shift/RMSE inner loop, replaced by the `tandemmp._rust.drift_correction`
PyO3 extension — crate at `src/tandemmp/_rust/drift_correction/`).

## Method

`drift_correction_baseline.py` calls
`tandemmp.pipeline.drift_correction.run_drift_correction` on a real `.mp`
movie with `frame_gap=25` and a fixed `seed=42` (for reproducibility — see
`corrector._solve`'s docstring), for all four combinations of
`parallel={False, True}` x `use_rust={False, True}`:

- `parallel=False` isolates algorithmic speed (Python DE loop vs. the fused
  Rust `solve()`) from multiprocessing, so it's the fairer per-frame
  comparison.
- `parallel=True` (the production default) additionally dispatches frames
  across a `multiprocessing.Pool` via `tandemmp.core.parallel.dispatch_kwargs`
  — this is what a real pipeline run actually pays.

```bash
python benchmarks/drift_correction_baseline.py /path/to/movie.mp
```

Requires the `tandemmp._rust.drift_correction` extension to be built first
(`maturin develop --release` from `src/tandemmp/_rust/drift_correction/`)
for the `use_rust=True` runs to exercise the compiled path rather than
falling back to Python.

## Results

Benchmark file: `012_20250717_beta_gal_6nM.mp` — 3000 frames, 59x150 px,
120 frames sampled at `frame_gap=25`. Measured on a 20-logical-core machine,
Python and Rust runs in the same benchmark invocation (so the Python numbers
below supersede the original pre-Rust-port record as the same-run baseline;
see "Original baseline" below for that earlier measurement).

| Mode | Python | Rust | Speedup |
|---|---|---|---|
| Serial (`parallel=False`) | 72.64 s | **11.01 s** | **6.6x** |
| Parallel (`parallel=True`, default `n_jobs`) | 24.33 s | **2.70 s** | **9.0x** |

Rust serial is even faster than the *Python parallel* run (11.01 s vs.
24.33 s on 20 cores) — i.e. the single-threaded Rust port outruns Python
fanned out across the whole machine. Rust parallel combines both: the
per-frame algorithmic speedup *and* the existing
`tandemmp.core.parallel.dispatch_kwargs` multiprocessing fan-out (Rust's
`solve()`/`stabilise()` release the GIL via PyO3's `py.detach`, so they
parallelise across the pool exactly like the Python path did).

See `results.json` for the full machine-readable record (`python_baseline`
and `rust_port`, each with `serial`/`parallel` sub-entries).

### Original baseline (Python, pre-Rust-port)

The first baseline measurement, recorded before the Rust port existed:

| Mode | Elapsed |
|---|---|
| Serial (`parallel=False`) | 72.37 s |
| Parallel (`parallel=True`, default `n_jobs`) | 21.69 s |

That's ~0.60 s per sampled frame serial (matches a standalone single-frame
`_solve` timing of ~0.71 s including call overhead). The ~2-3 s difference
from the same-run Python numbers above is ordinary run-to-run variance
(machine load, OS scheduling), not a regression.

## Rust-side micro-optimisation notes

Profiling `solve()` (see
`src/tandemmp/_rust/drift_correction/examples/bench_solve.rs`, an ad-hoc
instrumented harness — not part of `cargo test`) found the dominant cost was
`SplineCoefficients2D::sample`, called once per pixel per DE candidate.
Three changes there took a single `shift_compare` call on a 59x150 frame
from ~600 us to ~135 us (about 4.4x):

1. **Skip `mirror_index`'s integer division for in-bounds taps.** The
   boundary-mirroring index fold used `i64::rem_euclid` (a genuine integer
   division) for all 16 taps of every pixel, even though the common case
   (any pixel not within 1-2 taps of a frame edge) never needs mirroring.
   Added a fast path that direct-indexes when the whole 4x4 tap window is
   provably in-bounds.
2. **Hoist the B-spline weight computation out of the pixel loop.** For a
   *constant* per-call shift, the weights returned by `cubic_weights`
   depend only on the shift's fractional part, which is identical for
   every integer row/column — only the integer tap-window position
   advances by 1 per pixel. Previously recomputed (a `floor` + a
   short polynomial) on every pixel; now computed once per
   `shift_compare`/`shift2d` call.
3. **`get_unchecked` in the (already bounds-proven) fast path**, with the
   invariant documented inline, plus `[profile.release] lto = "fat",
   codegen-units = 1` in `Cargo.toml` (the crate had no `[profile.release]`
   section at all before).

All 40 `cargo test --release` cases (including the 1e-9-tolerance
`scipy_parity` tests) and all 78 `pytest tests/drift` cases pass after these
changes — see each crate/module's doc comments for the full reasoning and
safety invariants.
