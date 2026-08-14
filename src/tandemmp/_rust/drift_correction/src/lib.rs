//! Native shift + RMSE hot path for `tandemmp.drift.corrector`.
//!
//! This crate's core modules (everything except [`python`]) started out as
//! pure Rust with no `pyo3` dependency and no Python bindings; the [`python`]
//! module now wraps them for `tandemmp._rust.drift_correction`, but the core
//! modules themselves remain plain `&[f64]`-based Rust, independently
//! testable via `cargo test`. It is the WORKLIST.md priority-6 target
//! ("Drift correction shift+RMSE" / `differential_evolution` inner loop):
//! a from-scratch port of `_rmse`, `_most_positive`, `_most_negative`,
//! `_float_to_slice`, `_shift_compare`, `_solve`, and `stabilise` from
//! `src/tandemmp/drift/corrector.py`.
//!
//! # Why the whole `_solve`, not just a `scipy.ndimage.shift` wrapper
//!
//! `scipy.optimize.differential_evolution` is pure Python, so its
//! per-candidate objective-evaluation loop carries real, uncompiled
//! overhead. On top of that, the Python `_shift_compare` recomputes the
//! baseline frame's cubic-spline prefilter from scratch on *every single
//! candidate evaluation*, even though the baseline is constant within one
//! `_solve` call. Wrapping only the compiled `scipy.ndimage.shift` routine
//! from Rust would still pay the per-candidate FFI/Python round-trip and
//! the redundant prefilter. Fusing the whole DE loop + shift + RMSE into
//! one native call means a single FFI crossing per movie frame, with the
//! baseline prefilter hoisted out of the inner loop — see
//! [`solve::BaselineFrame`].
//!
//! # Binding split
//!
//! Per `WORKLIST.md`'s "each should be implemented as a PyO3 crate under
//! `src/tandemmp/_rust/`" convention, the PyO3/numpy bindings live in their
//! own module ([`python`]) rather than being mixed into the core modules
//! below — those keep taking plain `&[f64]` slices plus explicit
//! `rows`/`cols` dimensions (row-major layout) rather than anything
//! Python- or NumPy-specific. [`python`] is thin: it validates shapes,
//! copies numpy arrays into row-major `Vec<f64>`s (via
//! `ndarray::ArrayView::as_standard_layout`, so non-contiguous/Fortran-order
//! input is handled correctly), and calls straight through to the functions
//! below. See `Cargo.toml`/`pyproject.toml` in this directory for how the
//! crate is packaged as the `tandemmp._rust.drift_correction` extension
//! module via maturin.
//!
//! # Modules
//!
//! - [`scalar`] — `_rmse`, `_most_positive`, `_most_negative`,
//!   `_float_to_slice`.
//! - [`spline`] — the cubic B-spline (order=3) prefilter + shift
//!   primitive shared by `stabilise` and the DE objective
//!   (`scipy.ndimage.shift` equivalent, `mode='constant', cval=0.0`).
//! - [`optimize`] — a from-scratch differential-evolution global
//!   optimiser (`strategy='best1bin'`) with a Nelder-Mead local "polish"
//!   pass, replacing `scipy.optimize.differential_evolution`.
//! - [`solve`] — [`solve::BaselineFrame`] (prefilter-once, compare-many)
//!   and [`solve::solve`], the fused hot-loop entry point matching
//!   `_solve`.
//! - [`python`] — the PyO3 `#[pymodule]` wrapping the above for
//!   `tandemmp._rust.drift_correction`.

pub mod optimize;
pub mod python;
pub mod scalar;
pub mod solve;
pub mod spline;

pub use scalar::{float_to_slice, most_negative, most_positive, rmse, FloatSlice};
pub use solve::{shift_compare, solve, BaselineFrame, SolveBounds};
pub use spline::{shift2d, stabilise, SplineCoefficients2D};
