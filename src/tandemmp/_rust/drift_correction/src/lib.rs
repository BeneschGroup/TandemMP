//! Native shift + RMSE hot path for `tandemmp.drift.corrector`.
//!
//! This is a pure Rust library crate — **no `pyo3` dependency and no
//! Python bindings live here**. It is the WORKLIST.md priority-6 target
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
//! This crate is meant to be wrapped by a separate PyO3 crate (per
//! `WORKLIST.md`'s "each should be implemented as a PyO3 crate under
//! `src/tandemmp/_rust/`" convention) — that binding/maturin packaging
//! work is a distinct follow-up step, not part of this crate. The public
//! functions here deliberately take plain `&[f64]` slices plus explicit
//! `rows`/`cols` dimensions (row-major layout) rather than anything
//! Python- or NumPy-specific, so wrapping them with PyO3 (e.g. via
//! `numpy::PyReadonlyArray2`) should be mechanical.
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

pub mod optimize;
pub mod scalar;
pub mod solve;
pub mod spline;

pub use scalar::{float_to_slice, most_negative, most_positive, rmse, FloatSlice};
pub use solve::{shift_compare, solve, BaselineFrame, SolveBounds};
pub use spline::{shift2d, stabilise, SplineCoefficients2D};
