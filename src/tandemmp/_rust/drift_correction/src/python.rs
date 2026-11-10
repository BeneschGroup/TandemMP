//! PyO3 bindings exposing this crate to Python as
//! `tandemmp._rust.drift_correction`.
//!
//! Kept in its own module (rather than scattered across `scalar.rs`/
//! `spline.rs`/`solve.rs`) so the pure-Rust core stays easy to read and
//! `cargo test` independently of Python — the functions here are thin
//! numpy <-> `&[f64]` adapters around the crate's public API, plus
//! `PyResult`-flavoured error handling for the `assert!`/panics that the
//! core uses internally (see the module docs on [`crate::scalar::rmse`]
//! etc. for why: those panics are meant to become `ValueError`s at the
//! Python boundary, which is exactly what happens here since PyO3 turns
//! any panic that unwinds out of a `#[pyfunction]` into a Python
//! `PanicException` — acceptable for now since we additionally pre-check
//! shapes explicitly below and return proper `ValueError`s before ever
//! reaching the panicking code paths).
//!
//! `solve` and `stabilise` are the two call sites `tandemmp.drift.corrector`
//! actually needs (`_solve`'s hot loop and the per-frame `stabilise` call in
//! `get_corrected_movie`); `rmse` and `shift_compare` are exposed too, both
//! because they're useful for testing this binding layer directly and
//! because they were cheap to add on top of the existing public API.

use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::scalar;
use crate::solve as solve_native;
use crate::spline;

/// Copy `view` into a row-major (C-contiguous) `Vec<f64>`, returning it
/// alongside `(rows, cols)`. Handles non-contiguous / Fortran-ordered numpy
/// input transparently (e.g. views produced by slicing or transposing on
/// the Python side) by falling back to a copy — the crate's core functions
/// assume row-major layout throughout.
fn to_row_major(view: &ArrayView2<'_, f64>) -> (Vec<f64>, usize, usize) {
    let rows = view.shape()[0];
    let cols = view.shape()[1];
    let standard = view.as_standard_layout();
    (standard.iter().copied().collect(), rows, cols)
}

fn shape_mismatch_err(a: &[usize], b: &[usize]) -> PyErr {
    PyValueError::new_err(format!(
        "input_frame and baseline_frame must have the same shape, got {a:?} and {b:?}"
    ))
}

/// Find the `(x, y)` shift minimising RMSE between `input_frame` and
/// `baseline_frame`, via differential evolution + Nelder-Mead polish.
///
/// Direct binding for `crate::solve::solve` / `tandemmp.drift.corrector._solve`.
/// `bounds` defaults to `(-6.0, 6.0)`, matching `_solve`'s default. `seed`
/// left as `None` seeds from OS entropy, matching `_solve(seed=None)`'s
/// genuinely-stochastic default.
#[pyfunction]
#[pyo3(signature = (input_frame, baseline_frame, bounds=(-6.0, 6.0), seed=None))]
fn solve(
    py: Python<'_>,
    input_frame: PyReadonlyArray2<'_, f64>,
    baseline_frame: PyReadonlyArray2<'_, f64>,
    bounds: (f64, f64),
    seed: Option<u64>,
) -> PyResult<(f64, f64)> {
    let input_view = input_frame.as_array();
    let baseline_view = baseline_frame.as_array();
    if input_view.shape() != baseline_view.shape() {
        return Err(shape_mismatch_err(
            input_view.shape(),
            baseline_view.shape(),
        ));
    }

    let (input_vec, rows, cols) = to_row_major(&input_view);
    let (baseline_vec, _, _) = to_row_major(&baseline_view);
    let solve_bounds = solve_native::SolveBounds {
        low: bounds.0,
        high: bounds.1,
    };

    // The DE + Nelder-Mead loop is pure CPU work over plain `Vec<f64>`s —
    // no Python objects are touched, so releasing the GIL here (`detach`,
    // PyO3 0.29's rename of the old `allow_threads`) lets other Python
    // threads (e.g. the `dispatch_kwargs` process pool's parent, or a
    // future threaded caller) run concurrently while this frame solves.
    let (x, y) = py
        .detach(|| solve_native::solve(&input_vec, &baseline_vec, rows, cols, solve_bounds, seed));
    Ok((x, y))
}

/// Shift `frame` by `(-y, -x)` to undo a drift of `(x, y)`.
///
/// Direct binding for `crate::spline::stabilise` / `tandemmp.drift.corrector.stabilise`.
#[pyfunction]
fn stabilise<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray2<'_, f64>,
    x: f64,
    y: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = frame.as_array();
    let (vec, rows, cols) = to_row_major(&view);

    let out = py.detach(|| spline::stabilise(&vec, rows, cols, x, y));
    let arr = Array2::from_shape_vec((rows, cols), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Root-mean-square error between two equal-shaped arrays.
///
/// Direct binding for `crate::scalar::rmse` / `tandemmp.drift.corrector._rmse`.
/// Accepts arrays of any dimensionality/shape (as `_rmse` does via NumPy
/// broadcasting-free elementwise comparison) as long as both inputs match.
#[pyfunction]
fn rmse(a: PyReadonlyArrayDyn<'_, f64>, b: PyReadonlyArrayDyn<'_, f64>) -> PyResult<f64> {
    let a_view = a.as_array();
    let b_view = b.as_array();
    if a_view.shape() != b_view.shape() {
        return Err(shape_mismatch_err(a_view.shape(), b_view.shape()));
    }
    let a_flat: Vec<f64> = a_view.iter().copied().collect();
    let b_flat: Vec<f64> = b_view.iter().copied().collect();
    Ok(scalar::rmse(&a_flat, &b_flat))
}

/// RMSE between `input_frame` and `baseline_frame` shifted by `(x, y)`,
/// both cropped to discard edge artefacts introduced by the shift.
///
/// Direct binding for `crate::solve::shift_compare` /
/// `tandemmp.drift.corrector._shift_compare`. Prefer [`solve`] when solving
/// a full frame (it reuses the baseline's spline prefilter across every DE
/// candidate); this is exposed mainly to unit-test the binding layer
/// against individual `(x, y)` candidates directly.
#[pyfunction]
fn shift_compare(
    x: f64,
    y: f64,
    input_frame: PyReadonlyArray2<'_, f64>,
    baseline_frame: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let input_view = input_frame.as_array();
    let baseline_view = baseline_frame.as_array();
    if input_view.shape() != baseline_view.shape() {
        return Err(shape_mismatch_err(
            input_view.shape(),
            baseline_view.shape(),
        ));
    }

    let (input_vec, rows, cols) = to_row_major(&input_view);
    let (baseline_vec, _, _) = to_row_major(&baseline_view);
    Ok(solve_native::shift_compare(
        x,
        y,
        &input_vec,
        &baseline_vec,
        rows,
        cols,
    ))
}

/// The `tandemmp._rust.drift_correction` submodule, registered by `_native`.
#[pymodule]
pub fn drift_correction(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(stabilise, m)?)?;
    m.add_function(wrap_pyfunction!(rmse, m)?)?;
    m.add_function(wrap_pyfunction!(shift_compare, m)?)?;
    Ok(())
}
