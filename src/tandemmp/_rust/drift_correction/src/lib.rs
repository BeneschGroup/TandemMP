//! Native shift + RMSE hot path for `tandemmp.drift.corrector`.
//!
//! The numerical modules operate on `&[f64]` slices and can be tested with
//! `cargo test`. The [`python`] module exposes them as
//! `tandemmp._rust.drift_correction`.
//!
//! # Why the whole `_solve`, not just a `scipy.ndimage.shift` wrapper
//!
//! `scipy.optimize.differential_evolution` is pure Python, so its per-candidate
//! objective loop carries uncompiled overhead. On top of that, the Python
//! `_shift_compare` recomputes the baseline frame's cubic-spline prefilter from
//! scratch on *every candidate evaluation*, though the baseline is constant within
//! one `_solve` call. Wrapping only the compiled `scipy.ndimage.shift` from Rust
//! would still pay the per-candidate FFI round-trip and the redundant prefilter.
//! Fusing the whole DE loop + shift + RMSE into one native call means a single FFI
//! crossing per movie frame, with the prefilter hoisted out of the inner loop; see
//! [`solve::BaselineFrame`].
//!
//! # Binding split
//!
//! The PyO3/numpy bindings live in [`python`]. The numerical modules take `&[f64]`
//! slices with explicit `rows`/`cols` in row-major order. The binding module
//! validates shapes, copies numpy arrays into row-major `Vec<f64>`s (via
//! `ndarray::ArrayView::as_standard_layout`, so non-contiguous/Fortran-order input is
//! handled) and calls straight through. `Cargo.toml` and the `pyproject.toml` in this
//! directory package the crate as the `tandemmp._rust.drift_correction` extension via
//! maturin; the main package's `setuptools-rust` build uses the same crate.
//!
//! # Modules
//!
//! - [`scalar`]: `_rmse`, `_most_positive`, `_most_negative`, `_float_to_slice`.
//! - [`spline`]: the cubic B-spline (order 3) prefilter + shift primitive shared by
//!   `stabilise` and the DE objective (`scipy.ndimage.shift` equivalent,
//!   `mode='constant', cval=0.0`).
//! - [`optimize`]: a from-scratch differential-evolution optimiser
//!   (`strategy='best1bin'`) with a Nelder-Mead polish, replacing
//!   `scipy.optimize.differential_evolution`.
//! - [`solve`]: [`solve::BaselineFrame`] (prefilter once, compare many) and
//!   [`solve::solve`], the fused hot-loop entry point matching `_solve`.
//! - [`python`]: the PyO3 `#[pymodule]` wrapping the above.

pub mod optimize;
pub mod python;
pub mod scalar;
pub mod solve;
pub mod spline;

pub use scalar::{float_to_slice, most_negative, most_positive, rmse, FloatSlice};
pub use solve::{shift_compare, solve, BaselineFrame, SolveBounds};
pub use spline::{shift2d, stabilise, SplineCoefficients2D};
