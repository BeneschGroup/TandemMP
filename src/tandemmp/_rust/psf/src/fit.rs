//! Hand-rolled Levenberg-Marquardt fit of `[c, dx, dy]` (+ optional constant
//! offset) to a pixel thumbnail, using the analytic spline derivatives in
//! [`crate::model`] for the Jacobian. Small, fixed parameter count (3 or 4)
//! so a plain Gaussian-elimination normal-equations solve is simpler and
//! just as fast as pulling in a general-purpose least-squares crate.

use crate::model::{thumb_coords, SplinePsfCore};

const MAX_PARAMS: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct FitResult {
    pub c: f64,
    pub dx: f64,
    pub dy: f64,
    pub offset: f64,
    /// Sum-of-squares error between the fitted model and the thumbnail,
    /// matching `tandemmp.fitting.psf_interface.error_metric`.
    pub error: f64,
    pub converged: bool,
}

/// Solves the `n x n` linear system `a x = b` in place via Gaussian
/// elimination with partial pivoting. Returns `None` if `a` is singular.
#[allow(clippy::needless_range_loop)] // indices double as row/col offsets into `a`
fn solve_linear(
    a: &mut [[f64; MAX_PARAMS]; MAX_PARAMS],
    b: &mut [f64; MAX_PARAMS],
    n: usize,
) -> Option<[f64; MAX_PARAMS]> {
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > best {
                best = a[row][col].abs();
                pivot = row;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }

    let mut x = [0.0f64; MAX_PARAMS];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in (row + 1)..n {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

/// Residual + Jacobian for one pixel, written into `jac_row[0..n_params]`.
/// Returns the residual `model - thumb_pixel`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn pixel_residual_and_jac(
    core: &SplinePsfCore,
    y: f64,
    x: f64,
    c: f64,
    dx: f64,
    dy: f64,
    offset: f64,
    allow_offset: bool,
    thumb_pixel: f64,
    jac_row: &mut [f64; MAX_PARAMS],
) -> f64 {
    let a = y - dy;
    let b = x - dx;
    let s = core.spline.eval(a, b);
    let dsda = core.dspline_da.eval(a, b);
    let dsdb = core.dspline_db.eval(a, b);

    let model = -c * s + if allow_offset { offset } else { 0.0 };

    jac_row[0] = -s; // d model / d c
    jac_row[1] = c * dsdb; // d model / d dx
    jac_row[2] = c * dsda; // d model / d dy
    if allow_offset {
        jac_row[3] = 1.0; // d model / d offset
    }

    model - thumb_pixel
}

/// Fits `[c, dx, dy]` (+ optional `offset`) to `thumb` (row-major,
/// `l_thum x l_thum`) via Levenberg-Marquardt, starting from `init`.
pub fn fit_peak(
    core: &SplinePsfCore,
    thumb: &[f64],
    l_thum: usize,
    init: (f64, f64, f64),
    allow_offset: bool,
    max_iter: usize,
    tol: f64,
) -> Result<FitResult, String> {
    if thumb.len() != l_thum * l_thum {
        return Err(format!(
            "thumb length {} does not match l_thum*l_thum = {}",
            thumb.len(),
            l_thum * l_thum
        ));
    }
    let coords = thumb_coords(l_thum)?;
    let n_params = if allow_offset { 4 } else { 3 };

    let mut params = [init.0, init.1, init.2, 0.0];
    let mut lambda = 1e-3;
    let nu = 10.0;

    let cost = |p: &[f64; MAX_PARAMS]| -> f64 {
        let mut jac_row = [0.0f64; MAX_PARAMS];
        let mut sum = 0.0;
        for (row, &y) in coords.iter().enumerate() {
            for (col, &x) in coords.iter().enumerate() {
                let r = pixel_residual_and_jac(
                    core,
                    y,
                    x,
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    allow_offset,
                    thumb[row * l_thum + col],
                    &mut jac_row,
                );
                sum += r * r;
            }
        }
        sum
    };

    let mut current_cost = cost(&params);
    let mut converged = false;

    for _ in 0..max_iter {
        let mut jtj = [[0.0f64; MAX_PARAMS]; MAX_PARAMS];
        let mut jtr = [0.0f64; MAX_PARAMS];
        let mut jac_row = [0.0f64; MAX_PARAMS];

        for (row, &y) in coords.iter().enumerate() {
            for (col, &x) in coords.iter().enumerate() {
                let r = pixel_residual_and_jac(
                    core,
                    y,
                    x,
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    allow_offset,
                    thumb[row * l_thum + col],
                    &mut jac_row,
                );
                for i in 0..n_params {
                    jtr[i] += jac_row[i] * r;
                    for j in 0..n_params {
                        jtj[i][j] += jac_row[i] * jac_row[j];
                    }
                }
            }
        }

        let mut step_accepted = false;
        for _ in 0..30 {
            let mut a = jtj;
            for i in 0..n_params {
                a[i][i] += lambda * jtj[i][i].max(1e-12);
            }
            let mut b = [-jtr[0], -jtr[1], -jtr[2], -jtr[3]];
            let Some(delta) = solve_linear(&mut a, &mut b, n_params) else {
                lambda *= nu;
                continue;
            };

            let mut trial = params;
            for i in 0..n_params {
                trial[i] += delta[i];
            }
            let trial_cost = cost(&trial);

            if trial_cost.is_finite() && trial_cost < current_cost {
                let step_norm: f64 = delta[..n_params].iter().map(|d| d * d).sum::<f64>().sqrt();
                let rel_improve = (current_cost - trial_cost) / current_cost.max(1e-300);
                params = trial;
                current_cost = trial_cost;
                lambda = (lambda / nu).max(1e-12);
                step_accepted = true;
                if step_norm < tol || rel_improve < tol {
                    converged = true;
                }
                break;
            }
            lambda *= nu;
            if lambda > 1e12 {
                break;
            }
        }

        if converged || !step_accepted {
            if step_accepted {
                converged = true;
            }
            break;
        }
    }

    Ok(FitResult {
        c: params[0],
        dx: params[1],
        dy: params[2],
        offset: params[3],
        error: current_cost,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bspline::BSpline2D;
    use crate::model::render;

    fn simple_core() -> SplinePsfCore {
        // Degree-3 spline on a uniform-ish knot vector fitting a smooth bump;
        // coefficients chosen so the spline itself (not just its shape) is
        // exercised by both eval and derivative code paths.
        let tx: Vec<f64> = vec![-6.0, -6.0, -6.0, -6.0, -3.0, 0.0, 3.0, 6.0, 6.0, 6.0, 6.0];
        let ty = tx.clone();
        let ncoef = tx.len() - 3 - 1;
        let mut c = vec![0.0f64; ncoef * ncoef];
        for i in 0..ncoef {
            for j in 0..ncoef {
                let xi = i as f64 - (ncoef as f64 - 1.0) / 2.0;
                let yj = j as f64 - (ncoef as f64 - 1.0) / 2.0;
                c[i * ncoef + j] = -((-(xi * xi + yj * yj) / 6.0).exp());
            }
        }
        SplinePsfCore::new(BSpline2D::new(tx, ty, c, 3, 3).unwrap())
    }

    #[test]
    fn recovers_known_params_noise_free() {
        let core = simple_core();
        let l_thum = 15usize;
        let truth = (-0.05f64, 0.7f64, -1.3f64);
        let thumb = render(&core, truth.1, truth.2, truth.0, l_thum).unwrap();

        let fit = fit_peak(&core, &thumb, l_thum, (-0.04, 0.0, 0.0), false, 100, 1e-12).unwrap();
        assert!(fit.converged);
        assert!((fit.c - truth.0).abs() < 1e-6, "c={}", fit.c);
        assert!((fit.dx - truth.1).abs() < 1e-6, "dx={}", fit.dx);
        assert!((fit.dy - truth.2).abs() < 1e-6, "dy={}", fit.dy);
        assert!(fit.error < 1e-16, "error={}", fit.error);
    }

    #[test]
    fn recovers_with_offset() {
        let core = simple_core();
        let l_thum = 15usize;
        let truth = (-0.08f64, -0.3f64, 0.4f64);
        let mut thumb = render(&core, truth.1, truth.2, truth.0, l_thum).unwrap();
        for v in thumb.iter_mut() {
            *v += 0.01;
        }

        let fit = fit_peak(&core, &thumb, l_thum, (-0.05, 0.0, 0.0), true, 200, 1e-12).unwrap();
        assert!(fit.converged);
        assert!((fit.c - truth.0).abs() < 1e-5, "c={}", fit.c);
        assert!((fit.dx - truth.1).abs() < 1e-5, "dx={}", fit.dx);
        assert!((fit.dy - truth.2).abs() < 1e-5, "dy={}", fit.dy);
        assert!((fit.offset - 0.01).abs() < 1e-5, "offset={}", fit.offset);
    }
}
