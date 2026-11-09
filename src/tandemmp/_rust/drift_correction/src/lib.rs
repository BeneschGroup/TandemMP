//! Native shift + RMSE hot path for `tandemmp.drift.corrector`.
//!
//! The numerical modules operate on `&[f64]` slices and can be tested with
//! `cargo test`. The [`python`] module exposes them as
//! `tandemmp._rust.drift_correction`.
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
//! The PyO3/numpy bindings live in [`python`]. The numerical modules take
//! `&[f64]` slices with explicit `rows`/`cols` dimensions in row-major order.
//! The binding module validates shapes,
//! copies numpy arrays into row-major `Vec<f64>`s (via
//! `ndarray::ArrayView::as_standard_layout`, so non-contiguous/Fortran-order
//! input is handled correctly), and calls straight through to the functions
//! below. This crate is an `rlib` registered as the `tandemmp._rust.drift_correction`
//! submodule by `../_native`, which is the crate maturin actually builds.
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
