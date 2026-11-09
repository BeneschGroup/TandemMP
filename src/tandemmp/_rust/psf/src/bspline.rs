//! Tensor-product B-spline evaluation, numerically matching
//! `scipy.interpolate.RectBivariateSpline.__call__(x, y, grid=False)` for the
//! default, interpolating (not smoothing) spline.
//!
//! Ported (ownership transferred, no runtime dependency) from
//! `rust_mp_sim/src/psf_model_spline.rs`'s `SplinePSF`. The evaluation
//! algorithm is a 0-indexed port of the algorithm FITPACK uses internally
//! for scattered-point bivariate spline evaluation (`bispeu`, built on
//! `fpbspl`'s de Boor-Cox recurrence), not an approximation of it.

/// Maximum supported spline degree per axis. `RectBivariateSpline` defaults
/// to (and TandemMP's bundled experimental PSF uses) bicubic splines
/// (kx=ky=3); this constant is generous so `eval()` can use fixed-size stack
/// arrays without hard-coding degree 3.
pub const MAX_DEGREE: usize = 7;

/// A tensor-product B-spline, numerically matching
/// `scipy.interpolate.RectBivariateSpline`.
///
/// # Fields
/// * `tx` - Knot vector along the first axis (`tck[0]`).
/// * `ty` - Knot vector along the second axis (`tck[1]`).
/// * `c` - Flattened coefficients (`tck[2]`), row-major with the first-axis
///   basis index varying slower: `c[ix * ncoef_y + iy]`.
/// * `kx`, `ky` - Spline degree per axis (3 for bicubic).
///
/// # Extrapolation
/// `RectBivariateSpline.__call__` clamps each query coordinate to the fitted
/// domain before evaluating, rather than extrapolating FITPACK's boundary
/// polynomial piece. `eval` reproduces this clamping.
#[derive(Clone, Debug)]
pub struct BSpline2D {
    pub tx: Vec<f64>,
    pub ty: Vec<f64>,
    pub c: Vec<f64>,
    pub kx: usize,
    pub ky: usize,
}

impl BSpline2D {
    pub fn new(
        tx: Vec<f64>,
        ty: Vec<f64>,
        c: Vec<f64>,
        kx: usize,
        ky: usize,
    ) -> Result<Self, String> {
        if kx > MAX_DEGREE || ky > MAX_DEGREE {
            return Err(format!(
                "spline degree {kx}/{ky} exceeds MAX_DEGREE ({MAX_DEGREE})"
            ));
        }
        if tx.len() < kx + 2 || ty.len() < ky + 2 {
            return Err("knot vectors too short for the given degree".to_string());
        }
        let ncoef_x = tx.len() - kx - 1;
        let ncoef_y = ty.len() - ky - 1;
        if c.len() != ncoef_x * ncoef_y {
            return Err(format!(
                "coefficient array length {} does not match (len(tx)-kx-1)*(len(ty)-ky-1) = {}",
                c.len(),
                ncoef_x * ncoef_y
            ));
        }
        Ok(BSpline2D { tx, ty, c, kx, ky })
    }

    pub fn ncoef_x(&self) -> usize {
        self.tx.len() - self.kx - 1
    }

    pub fn ncoef_y(&self) -> usize {
        self.ty.len() - self.ky - 1
    }

    /// Finds the 0-indexed knot span `l` such that `t[l] <= x < t[l+1]`,
    /// restricted to `l` in `[k, ncoef-1]`.
    #[inline]
    fn find_span(t: &[f64], k: usize, ncoef: usize, x: f64) -> usize {
        let mut l = k;
        while l + 1 < ncoef && x >= t[l + 1] {
            l += 1;
        }
        l
    }

    /// Evaluates the `k+1` non-zero degree-`k` B-spline basis functions at
    /// `x`, given the knot span `l`. Writes results into `out[0..=k]`. A
    /// 0-indexed port of FITPACK's `fpbspl` (de Boor-Cox recurrence).
    #[inline]
    fn bspline_basis(t: &[f64], k: usize, l: usize, x: f64, out: &mut [f64; MAX_DEGREE + 1]) {
        let mut hh = [0.0f64; MAX_DEGREE + 1];
        out[0] = 1.0;

        for j in 1..=k {
            hh[..j].copy_from_slice(&out[..j]);
            out[0] = 0.0;
            for i in 0..j {
                let li = l + i + 1;
                let lj = li - j;
                if t[li] == t[lj] {
                    out[i + 1] = 0.0;
                } else {
                    let f = hh[i] / (t[li] - t[lj]);
                    out[i] += f * (t[li] - x);
                    out[i + 1] = f * (x - t[lj]);
                }
            }
        }
    }

    /// Evaluates the spline at `(x, y)`, matching
    /// `RectBivariateSpline.__call__(x, y, grid=False)`.
    #[allow(clippy::needless_range_loop)] // `i`/`j` double as basis & row/col offsets
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let ncoef_x = self.ncoef_x();
        let ncoef_y = self.ncoef_y();

        let x = x.clamp(self.tx[self.kx], self.tx[self.tx.len() - self.kx - 1]);
        let y = y.clamp(self.ty[self.ky], self.ty[self.ty.len() - self.ky - 1]);

        let lx = Self::find_span(&self.tx, self.kx, ncoef_x, x);
        let ly = Self::find_span(&self.ty, self.ky, ncoef_y, y);

        let mut hx = [0.0f64; MAX_DEGREE + 1];
        let mut hy = [0.0f64; MAX_DEGREE + 1];
        Self::bspline_basis(&self.tx, self.kx, lx, x, &mut hx);
        Self::bspline_basis(&self.ty, self.ky, ly, y, &mut hy);

        let row0 = lx - self.kx;
        let col0 = ly - self.ky;

        let mut result = 0.0f64;
        for i in 0..=self.kx {
            let row_base = (row0 + i) * ncoef_y;
            let hxi = hx[i];
            for j in 0..=self.ky {
                result += hxi * hy[j] * self.c[row_base + col0 + j];
            }
        }
        result
    }

    /// Returns the spline's derivative with respect to its first argument
    /// (`x`), as an exact B-spline of degree `kx - 1` on knots `tx[1..-1]`.
    /// Matches `scipy`'s `splder`/`BSpline.derivative` coefficient formula.
    ///
    /// # Panics
    /// If `kx == 0` (no derivative spline representable in this basis).
    pub fn derivative_x(&self) -> BSpline2D {
        assert!(self.kx >= 1, "cannot differentiate a degree-0 spline");
        let kx = self.kx;
        let ncoef_x = self.ncoef_x();
        let ncoef_y = self.ncoef_y();
        let new_tx = self.tx[1..self.tx.len() - 1].to_vec();
        let new_ncoef_x = ncoef_x - 1;

        let mut d = vec![0.0f64; new_ncoef_x * ncoef_y];
        for i in 0..new_ncoef_x {
            let denom = self.tx[i + kx + 1] - self.tx[i + 1];
            let scale = if denom == 0.0 { 0.0 } else { kx as f64 / denom };
            for j in 0..ncoef_y {
                let c_next = self.c[(i + 1) * ncoef_y + j];
                let c_here = self.c[i * ncoef_y + j];
                d[i * ncoef_y + j] = scale * (c_next - c_here);
            }
        }

        BSpline2D {
            tx: new_tx,
            ty: self.ty.clone(),
            c: d,
            kx: kx - 1,
            ky: self.ky,
        }
    }

    /// Returns the spline's derivative with respect to its second argument
    /// (`y`); see [`derivative_x`](Self::derivative_x).
    pub fn derivative_y(&self) -> BSpline2D {
        assert!(self.ky >= 1, "cannot differentiate a degree-0 spline");
        let ky = self.ky;
        let ncoef_x = self.ncoef_x();
        let ncoef_y = self.ncoef_y();
        let new_ty = self.ty[1..self.ty.len() - 1].to_vec();
        let new_ncoef_y = ncoef_y - 1;

        let mut d = vec![0.0f64; ncoef_x * new_ncoef_y];
        for i in 0..ncoef_x {
            for j in 0..new_ncoef_y {
                let denom = self.ty[j + ky + 1] - self.ty[j + 1];
                let scale = if denom == 0.0 { 0.0 } else { ky as f64 / denom };
                let c_next = self.c[i * ncoef_y + j + 1];
                let c_here = self.c[i * ncoef_y + j];
                d[i * new_ncoef_y + j] = scale * (c_next - c_here);
            }
        }

        BSpline2D {
            tx: self.tx.clone(),
            ty: new_ty,
            c: d,
            kx: self.kx,
            ky: ky - 1,
        }
    }
}
