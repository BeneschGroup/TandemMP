//! Tensor-product B-spline evaluation matching
//! `scipy.interpolate.RectBivariateSpline(u, v, grid=False)`.
//!
//! Ported from rust_mp_sim's `psf_model_spline.rs` (FITPACK `fpbspl`
//! recurrence). Queries are clamped to the fitted domain, as scipy does.

/// Maximum supported degree per axis (fixed-size stack buffers).
pub const MAX_DEGREE: usize = 7;

/// Bivariate B-spline with knots `tu`, `tv` and coefficients `c[iu * ncoef_v + iv]`.
pub struct Spline {
    tu: Vec<f64>,
    tv: Vec<f64>,
    c: Vec<f64>,
    ku: usize,
    kv: usize,
}

impl Spline {
    pub fn new(
        tu: Vec<f64>,
        tv: Vec<f64>,
        c: Vec<f64>,
        ku: usize,
        kv: usize,
    ) -> Result<Self, String> {
        if ku > MAX_DEGREE || kv > MAX_DEGREE {
            return Err(format!("spline degree {ku}/{kv} exceeds {MAX_DEGREE}"));
        }
        if tu.len() < 2 * (ku + 1) || tv.len() < 2 * (kv + 1) {
            return Err("too few knots for the spline degree".into());
        }
        let n = (tu.len() - ku - 1) * (tv.len() - kv - 1);
        if c.len() != n {
            return Err(format!("expected {n} coefficients, got {}", c.len()));
        }
        Ok(Spline { tu, tv, c, ku, kv })
    }

    pub fn knots(&self) -> (&[f64], &[f64]) {
        (&self.tu, &self.tv)
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.c
    }

    pub fn degrees(&self) -> (usize, usize) {
        (self.ku, self.kv)
    }

    /// Knot span `l` in `[k, ncoef - 1]` with `t[l] <= x < t[l + 1]`.
    #[inline]
    fn find_span(t: &[f64], k: usize, ncoef: usize, x: f64) -> usize {
        let p = t.partition_point(|&ti| ti <= x);
        p.saturating_sub(1).clamp(k, ncoef - 1)
    }

    /// The `k + 1` non-zero basis functions at `x` (de Boor-Cox recurrence).
    #[inline]
    fn basis(t: &[f64], k: usize, l: usize, x: f64, out: &mut [f64; MAX_DEGREE + 1]) {
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

    /// Evaluate at `(u, v)`.
    pub fn eval(&self, u: f64, v: f64) -> f64 {
        let (ku, kv) = (self.ku, self.kv);
        let nu = self.tu.len() - ku - 1;
        let nv = self.tv.len() - kv - 1;
        let u = u.clamp(self.tu[ku], self.tu[nu]);
        let v = v.clamp(self.tv[kv], self.tv[nv]);

        let lu = Self::find_span(&self.tu, ku, nu, u);
        let lv = Self::find_span(&self.tv, kv, nv, v);
        let mut hu = [0.0f64; MAX_DEGREE + 1];
        let mut hv = [0.0f64; MAX_DEGREE + 1];
        Self::basis(&self.tu, ku, lu, u, &mut hu);
        Self::basis(&self.tv, kv, lv, v, &mut hv);

        let (row0, col0) = (lu - ku, lv - kv);
        let mut out = 0.0;
        for (i, &hui) in hu.iter().enumerate().take(ku + 1) {
            let base = (row0 + i) * nv + col0;
            for (j, &hvj) in hv.iter().enumerate().take(kv + 1) {
                out += hui * hvj * self.c[base + j];
            }
        }
        out
    }
}
