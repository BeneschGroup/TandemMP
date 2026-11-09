//! PyO3 bindings exposing this crate to Python as `tandemmp._rust.psf`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::bspline::BSpline2D;
use crate::fit::{fit_peak, FitResult};
use crate::model::{render, SplinePsfCore};

fn err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

fn to_row_major(view: &numpy::ndarray::ArrayView2<'_, f64>) -> Vec<f64> {
    view.as_standard_layout().iter().copied().collect()
}

/// `tandemmp._rust.psf.SplinePsf` — a tensor-product B-spline PSF matching
/// `scipy.interpolate.RectBivariateSpline`, with a Levenberg-Marquardt
/// pixel-residual fitter over `[c, dx, dy]` (+ optional offset).
#[pyclass]
pub struct SplinePsf {
    core: SplinePsfCore,
}

#[pymethods]
impl SplinePsf {
    #[new]
    fn new(
        tx: PyReadonlyArray1<'_, f64>,
        ty: PyReadonlyArray1<'_, f64>,
        c: PyReadonlyArray1<'_, f64>,
        kx: usize,
        ky: usize,
    ) -> PyResult<Self> {
        let spline = BSpline2D::new(
            tx.as_slice()?.to_vec(),
            ty.as_slice()?.to_vec(),
            c.as_slice()?.to_vec(),
            kx,
            ky,
        )
        .map_err(err)?;
        Ok(SplinePsf {
            core: SplinePsfCore::new(spline),
        })
    }

    /// Render an `l_thum x l_thum` PSF image at offset `(dx, dy)` with
    /// contrast `c`, matching `PsfModel.psf_2d_offset`'s spline path
    /// (`spline(y, x, grid=False) * -c` over a centred square mesh).
    fn render<'py>(
        &self,
        py: Python<'py>,
        dx: f64,
        dy: f64,
        c: f64,
        l_thum: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let flat = render(&self.core, dx, dy, c, l_thum).map_err(err)?;
        let arr = Array2::from_shape_vec((l_thum, l_thum), flat).map_err(|e| err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Fit `[c, dx, dy]` (+ optional offset) to `thumb` via
    /// Levenberg-Marquardt, starting from `init = (c0, dx0, dy0)`. Returns
    /// `(c, dx, dy, error, converged)`; `error` is the sum-of-squares
    /// residual, matching `tandemmp.fitting.psf_interface.error_metric`.
    #[pyo3(signature = (thumb, init, allow_offset=false, max_iter=50, tol=1e-10))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        thumb: PyReadonlyArray2<'_, f64>,
        init: (f64, f64, f64),
        allow_offset: bool,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<(f64, f64, f64, f64, bool)> {
        let view = thumb.as_array();
        let shape = view.shape();
        if shape[0] != shape[1] {
            return Err(err("thumb must be square"));
        }
        let l_thum = shape[0];
        let flat = to_row_major(&view);
        let core = &self.core;

        let result = py.detach(|| fit_peak(core, &flat, l_thum, init, allow_offset, max_iter, tol));
        let r = result.map_err(err)?;
        Ok((r.c, r.dx, r.dy, r.error, r.converged))
    }

    /// Batch-fit `thumbs[n, h, w]` against `inits[n, 3]` in parallel (rayon,
    /// GIL released). Returns an `[n, 5]` array of
    /// `(c, dx, dy, error, converged)`.
    #[pyo3(signature = (thumbs, inits, allow_offset=false, max_iter=50, tol=1e-10))]
    #[allow(clippy::too_many_arguments)]
    fn fit_batch<'py>(
        &self,
        py: Python<'py>,
        thumbs: PyReadonlyArray3<'_, f64>,
        inits: PyReadonlyArray2<'_, f64>,
        allow_offset: bool,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let thumbs_view = thumbs.as_array();
        let inits_view = inits.as_array();
        let shape = thumbs_view.shape();
        let (n, h, w) = (shape[0], shape[1], shape[2]);
        if h != w {
            return Err(err("each thumbnail must be square"));
        }
        if inits_view.shape() != [n, 3] {
            return Err(err(format!(
                "inits must have shape ({n}, 3), got {:?}",
                inits_view.shape()
            )));
        }

        let thumbs_owned = thumbs_view.as_standard_layout().into_owned();
        let inits_owned: Vec<(f64, f64, f64)> = (0..n)
            .map(|i| (inits_view[[i, 0]], inits_view[[i, 1]], inits_view[[i, 2]]))
            .collect();
        let core = &self.core;

        let results: Vec<Result<FitResult, String>> = py.detach(|| {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let thumb = thumbs_owned.index_axis(numpy::ndarray::Axis(0), i);
                    let flat: Vec<f64> = thumb.iter().copied().collect();
                    fit_peak(core, &flat, h, inits_owned[i], allow_offset, max_iter, tol)
                })
                .collect()
        });

        let mut out = Array2::<f64>::zeros((n, 5));
        for (i, res) in results.into_iter().enumerate() {
            let r = res.map_err(err)?;
            out[[i, 0]] = r.c;
            out[[i, 1]] = r.dx;
            out[[i, 2]] = r.dy;
            out[[i, 3]] = r.error;
            out[[i, 4]] = if r.converged { 1.0 } else { 0.0 };
        }
        Ok(out.into_pyarray(py))
    }
}

/// The `tandemmp._rust.psf` submodule, registered by `_native`.
#[pymodule]
pub fn psf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SplinePsf>()?;
    Ok(())
}
