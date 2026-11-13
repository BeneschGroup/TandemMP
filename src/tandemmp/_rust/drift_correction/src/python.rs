//! PyO3 bindings exposing this crate to Python as `tandemmp._rust.drift_correction`.
//!
//! Kept in its own module so the pure-Rust core stays easy to read and `cargo test`
//! independently of Python. The functions are thin numpy <-> `&[f64]` adapters around
//! the crate's public API. Shapes are checked explicitly and returned as
//! `ValueError`s before reaching the core's panicking `assert!`s (PyO3 would
//! otherwise surface a panic unwinding out of a `#[pyfunction]` as a Python
//! `PanicException`).
//!
//! `solve` and `stabilise` are what `tandemmp.drift.corrector` needs (`_solve`'s hot
//! loop and the per-frame call in `get_corrected_movie`); `rmse` and `shift_compare`
//! are exposed too, for testing this layer directly and because they were cheap to
//! add.

use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::scalar;
use crate::solve as solve_native;
use crate::spline;

/// Copy `view` into a row-major (C-contiguous) `Vec<f64>`, returning it with
/// `(rows, cols)`. Non-contiguous / Fortran-ordered numpy input (e.g. sliced or
/// transposed views) falls back to a copy, as the core assumes row-major layout.
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
/// Binding for `crate::solve::solve` / `tandemmp.drift.corrector._solve`. `bounds`
/// defaults to `(-6.0, 6.0)` as in `_solve`; `seed=None` seeds from OS entropy
/// (stochastic), as in `_solve(seed=None)`.
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

    // The DE + Nelder-Mead loop is pure CPU work on plain `Vec<f64>`s, touching no
    // Python objects, so the GIL is released (`detach`, PyO3 0.29's name for
    // `allow_threads`) to let other Python threads (e.g. the `dispatch_kwargs` pool's
    // parent) run meanwhile.
    let (x, y) = py
        .detach(|| solve_native::solve(&input_vec, &baseline_vec, rows, cols, solve_bounds, seed));
    Ok((x, y))
}

/// Shift `frame` by `(-y, -x)` to undo a drift of `(x, y)`.
///
/// Binding for `crate::spline::stabilise` / `tandemmp.drift.corrector.stabilise`.
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
/// Binding for `crate::scalar::rmse` / `tandemmp.drift.corrector._rmse`. Accepts any
/// dimensionality as long as both inputs match.
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

/// RMSE between `input_frame` and `baseline_frame` shifted by `(x, y)`, both cropped
/// to discard edge artefacts from the shift.
///
/// Binding for `crate::solve::shift_compare` /
/// `tandemmp.drift.corrector._shift_compare`. Prefer [`solve`] for a full frame (it
/// reuses the baseline's spline prefilter across every DE candidate); this exists
/// mainly to unit-test the binding layer on individual `(x, y)` candidates.
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

/// The `tandemmp._rust.drift_correction` extension module.
#[pymodule]
fn drift_correction(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(stabilise, m)?)?;
    m.add_function(wrap_pyfunction!(rmse, m)?)?;
    m.add_function(wrap_pyfunction!(shift_compare, m)?)?;
    Ok(())
}
