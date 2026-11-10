//! TandemMP's PSF-image convention layered on top of [`BSpline2D`].
//!
//! `tandemmp.fitting.psf.PsfModel.get_psf_function`'s spline branch renders
//! `spline(y, x, grid=False) * -c` over `tandemmp.fitting.psf_interface.square_mesh`
//! coordinates evaluated at `(x_coords - dx, y_coords - dy)`. For a pixel at
//! grid position `(x, y)` (centred square mesh, `x` varying along columns,
//! `y` along rows) that is:
//!
//! ```text
//! model(x, y) = -c * spline.eval(y - dy, x - dx)
//! ```
//!
//! i.e. the spline's first argument is `y - dy`, its second is `x - dx`.
//! This module keeps that convention in one place so [`render`] and the
//! fitter's Jacobian ([`crate::fit`]) can't drift apart.

use crate::bspline::BSpline2D;

/// A spline PSF plus its two partial-derivative splines, precomputed once
/// so a fit's per-iteration Jacobian is just two extra spline evaluations.
pub struct SplinePsfCore {
    pub spline: BSpline2D,
    /// Derivative with respect to the spline's first argument (`y - dy`).
    pub dspline_da: BSpline2D,
    /// Derivative with respect to the spline's second argument (`x - dx`).
    pub dspline_db: BSpline2D,
}

impl SplinePsfCore {
    pub fn new(spline: BSpline2D) -> Self {
        let dspline_da = spline.derivative_x();
        let dspline_db = spline.derivative_y();
        SplinePsfCore {
            spline,
            dspline_da,
            dspline_db,
        }
    }
}

/// Centred 1-D pixel coordinates for a square thumbnail of odd side
/// `l_thum`, matching `tandemmp.fitting.psf_interface.square_mesh`.
pub fn thumb_coords(l_thum: usize) -> Result<Vec<f64>, String> {
    if l_thum.is_multiple_of(2) {
        return Err("l_thum must be odd".to_string());
    }
    let half = (l_thum / 2) as f64;
    Ok((0..l_thum).map(|k| k as f64 - half).collect())
}

/// Renders an `l_thum x l_thum` PSF image at offset `(dx, dy)` with contrast
/// `c`, row-major, matching `PsfModel.psf_2d_offset`/`render`'s spline path.
pub fn render(
    core: &SplinePsfCore,
    dx: f64,
    dy: f64,
    c: f64,
    l_thum: usize,
) -> Result<Vec<f64>, String> {
    let coords = thumb_coords(l_thum)?;
    let mut out = vec![0.0f64; l_thum * l_thum];
    for (row, &y) in coords.iter().enumerate() {
        let a = y - dy;
        for (col, &x) in coords.iter().enumerate() {
            let b = x - dx;
            out[row * l_thum + col] = -c * core.spline.eval(a, b);
        }
    }
    Ok(out)
}
