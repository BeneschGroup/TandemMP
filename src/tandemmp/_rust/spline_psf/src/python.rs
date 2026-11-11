//! PyO3 bindings: `tandemmp._rust.spline_psf`.

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bundled;

/// Evaluate the bundled PSF at `(u, v)` (row, col); equal-shape arrays.
#[pyfunction]
fn eval<'py>(
    py: Python<'py>,
    u: PyReadonlyArrayDyn<'py, f64>,
    v: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let (u, v) = (u.as_array(), v.as_array());
    if u.shape() != v.shape() {
        return Err(PyValueError::new_err(format!(
            "u and v must have the same shape, got {:?} and {:?}",
            u.shape(),
            v.shape()
        )));
    }
    let shape = u.shape().to_vec();
    let (u, v) = (u.as_standard_layout(), v.as_standard_layout());
    let (us, vs) = (u.as_slice().unwrap(), v.as_slice().unwrap());
    let s = bundled();
    let out: Vec<f64> = py.detach(|| us.iter().zip(vs).map(|(&a, &b)| s.eval(a, b)).collect());
    let out = ArrayD::from_shape_vec(IxDyn(&shape), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

/// The bundled spline as `(tx, ty, c, kx, ky)`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn tck(
    py: Python<'_>,
) -> (
    Bound<'_, PyArray1<f64>>,
    Bound<'_, PyArray1<f64>>,
    Bound<'_, PyArray1<f64>>,
    usize,
    usize,
) {
    let s = bundled();
    let (tu, tv) = s.knots();
    let (ku, kv) = s.degrees();
    (
        PyArray1::from_slice(py, tu),
        PyArray1::from_slice(py, tv),
        PyArray1::from_slice(py, s.coefficients()),
        ku,
        kv,
    )
}

#[pymodule]
fn spline_psf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval, m)?)?;
    m.add_function(wrap_pyfunction!(tck, m)?)?;
    Ok(())
}
