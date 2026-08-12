//! Cubic B-spline (order=3) prefiltering and evaluation, equivalent to
//! `scipy.ndimage.shift(frame, (dy, dx))` with its defaults
//! (`order=3, mode='constant', cval=0.0, prefilter=True`).
//!
//! This is *not* naive cubic convolution — `scipy.ndimage.shift` first
//! deconvolves the samples into B-spline coefficients (a causal +
//! anticausal IIR recursion with a single pole at `z = sqrt(3) - 2`) and
//! only then evaluates the interpolant at shifted coordinates. Skipping
//! that prefilter step gives visibly different (smoother, lower-amplitude)
//! results, so it's implemented in full here, ported directly from
//! scipy's `ni_splines.c` (`_init_causal_mirror` / `_init_anticausal_mirror`
//! / `_apply_filter` / `_apply_filter_gain` — scipy dispatches
//! `mode='constant'` to the same causal/anticausal boundary formulas as
//! `mode='mirror'` for the *coefficient* recursion; the `constant`/`cval`
//! behaviour only shows up afterwards, at evaluation time, for spline
//! support taps that fall outside the array).
//!
//! Numerically cross-checked against `scipy.ndimage.shift` — see the
//! `scipy_parity` tests at the bottom of this file, whose expected arrays
//! were generated directly by SciPy (`scipy.ndimage.shift`) via the
//! `tandemmp-refactor` conda environment.

/// The single pole of the cubic (order=3) B-spline prefilter: `sqrt(3) - 2`.
const POLE: f64 = -0.267_949_192_431_122_7;

#[inline]
fn gain() -> f64 {
    (1.0 - POLE) * (1.0 - 1.0 / POLE)
}

/// In-place cubic B-spline coefficient recursion along a single 1-D line,
/// port of scipy's `_apply_filter` (with the causal/anticausal
/// initialisation scipy uses for `mode='constant'`, i.e.
/// `_init_causal_mirror` / `_init_anticausal_mirror`).
///
/// Rows/columns of length 0 or 1 are left as-is (after gain scaling) —
/// there's no meaningful two-sided recursion with fewer than 2 samples,
/// and real movie frames are always far larger than that.
fn apply_filter_1d(c: &mut [f64]) {
    let n = c.len();
    let z = POLE;

    if n < 2 {
        return;
    }

    // --- causal initialisation (mirror boundary) ---
    let z_n_1 = z.powi((n - 1) as i32);
    let mut z_i = z;
    let mut c0 = c[0] + z_n_1 * c[n - 1];
    for i in 1..(n - 1) {
        c0 += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
        z_i *= z;
    }
    c0 /= 1.0 - z_n_1 * z_n_1;
    c[0] = c0;

    // --- forward (causal) recursion ---
    for i in 1..n {
        c[i] += z * c[i - 1];
    }

    // --- anticausal initialisation (mirror boundary) ---
    c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0);

    // --- backward (anticausal) recursion ---
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// Scale-then-filter a single line: mirrors scipy applying `_apply_filter_gain`
/// once (before the pole recursion) since cubic splines have exactly one pole.
fn prefilter_1d(line: &mut [f64]) {
    let g = gain();
    for v in line.iter_mut() {
        *v *= g;
    }
    apply_filter_1d(line);
}

/// A 2-D frame's cubic B-spline coefficients, precomputed once and reusable
/// across many [`SplineCoefficients2D::sample`] calls at different
/// coordinates — this is the piece that lets `_solve`'s hot loop hoist the
/// baseline frame's prefilter out of the per-candidate objective
/// evaluation (see the crate root docs).
pub struct SplineCoefficients2D {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl SplineCoefficients2D {
    /// Compute cubic B-spline coefficients for `frame` (row-major,
    /// `rows * cols` values), matching
    /// `scipy.ndimage.spline_filter(frame, order=3, mode='constant')`
    /// (which is what `scipy.ndimage.shift` calls internally before
    /// evaluating at shifted coordinates).
    pub fn new(frame: &[f64], rows: usize, cols: usize) -> Self {
        assert_eq!(
            frame.len(),
            rows * cols,
            "frame length must equal rows * cols"
        );
        let mut data = frame.to_vec();

        // Axis 0: filter each column (a line running down the rows).
        // Mirrors `spline_filter1d(input, order, axis=0, ...)`.
        if rows >= 2 {
            let mut buf = vec![0.0; rows];
            for c in 0..cols {
                for r in 0..rows {
                    buf[r] = data[r * cols + c];
                }
                prefilter_1d(&mut buf);
                for r in 0..rows {
                    data[r * cols + c] = buf[r];
                }
            }
        } else {
            let g = gain();
            for v in data.iter_mut() {
                *v *= g;
            }
        }

        // Axis 1: filter each row (a line running across the columns).
        // Mirrors `spline_filter1d(input, order, axis=1, ...)`.
        if cols >= 2 {
            let mut buf = vec![0.0; cols];
            for r in 0..rows {
                let start = r * cols;
                buf.copy_from_slice(&data[start..start + cols]);
                prefilter_1d(&mut buf);
                data[start..start + cols].copy_from_slice(&buf);
            }
        } else {
            let g = gain();
            for v in data.iter_mut() {
                *v *= g;
            }
        }

        Self { data, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Evaluate the reconstructed spline surface at the real-valued
    /// coordinate `(y, x)` (row, column), matching
    /// `scipy.ndimage.map_coordinates(..., order=3, mode='constant',
    /// cval=0.0)`'s actual (empirically verified — see `spline.rs`'s
    /// module docs) two-tier boundary rule:
    ///
    /// - If `y` or `x` itself falls outside `[0, rows-1]` / `[0, cols-1]`,
    ///   the whole output is `cval = 0.0` — no partial/blended
    ///   contribution, however close to the edge.
    /// - Otherwise, the 4x4 tap window is evaluated normally, and any
    ///   individual tap that falls outside the array is resolved via
    ///   whole-sample mirror reflection (`[..., c2, c1, c0 | c0, c1,
    ///   c2, ...]`, not repeating the edge sample) rather than being
    ///   dropped — consistent with the coefficient array itself having
    ///   been prefiltered with mirror boundary conditions.
    pub fn sample(&self, y: f64, x: f64) -> f64 {
        if !in_domain(y, self.rows) || !in_domain(x, self.cols) {
            return 0.0;
        }
        let (wy, base_y) = cubic_weights(y);
        let (wx, base_x) = cubic_weights(x);
        self.sample_with_weights(&wy, base_y, &wx, base_x)
    }

    /// Evaluate at an explicit tap window: `wy`/`wx` are the 4 uniform
    /// cubic B-spline basis weights for the y/x axes (as returned by
    /// [`cubic_weights`]), `base_y`/`base_x` the corresponding leftmost tap
    /// indices.
    ///
    /// Exists so [`shift2d`] and `solve::BaselineFrame::shift_compare`'s
    /// pixel loops can pass in weights computed *once per call* rather than
    /// recomputing them per pixel: for a constant real-valued shift `s`,
    /// `cubic_weights(idx - s)`'s weights depend only on `frac(idx - s)`,
    /// which is identical for every integer `idx` (adding an integer never
    /// changes a fractional part) — only the integer `base` tap index
    /// advances by exactly 1 per step. Recomputing the weight polynomial
    /// (a handful of multiplies plus a `floor`) on every one of a frame's
    /// pixels was, after the `mirror_index` fast path above, the next
    /// largest cost in the DE inner loop; callers now compute `wy`/`wx`
    /// once per axis per call (via [`cubic_weights`] or [`weights_for_shift`])
    /// and only pay a cheap `floor` (via [`base_index`]) per row/column to
    /// track `base_y`/`base_x`.
    #[inline]
    pub(crate) fn sample_with_weights(
        &self,
        wy: &[f64; 4],
        base_y: i64,
        wx: &[f64; 4],
        base_x: i64,
    ) -> f64 {
        // Fast path: the overwhelming majority of samples (any pixel not
        // within 1-2 taps of an edge) have their whole 4x4 support window
        // in-bounds, so no tap needs mirror reflection. `mirror_index`'s
        // `rem_euclid` is a genuine integer division — one of the most
        // expensive scalar ALU ops — so paying it unconditionally for all
        // 16 taps of every sample, even when every tap is already in
        // range, was the dominant cost of the whole DE inner loop (each
        // `shift_compare` call does this for every pixel in the frame).
        if base_y >= 0
            && base_y + 3 < self.rows as i64
            && base_x >= 0
            && base_x + 3 < self.cols as i64
        {
            let by = base_y as usize;
            let bx = base_x as usize;
            let mut acc = 0.0;
            for (ky, &wyk) in wy.iter().enumerate() {
                let row_start = (by + ky) * self.cols + bx;
                // SAFETY: the branch above proves `0 <= by`, `by + 3 <
                // self.rows`, `0 <= bx`, `bx + 3 < self.cols`, so for every
                // `ky` in `0..4`, `row_start + 3 == (by + ky) * self.cols +
                // bx + 3 < self.rows * self.cols == self.data.len()` (the
                // invariant `SplineCoefficients2D::new` establishes and
                // never mutates afterwards) — i.e. `row_start..row_start+4`
                // is always a valid range into `self.data`.
                let row = unsafe { self.data.get_unchecked(row_start..row_start + 4) };
                acc += wyk * (wx[0] * row[0] + wx[1] * row[1] + wx[2] * row[2] + wx[3] * row[3]);
            }
            return acc;
        }

        let mut acc = 0.0;
        for (ky, &wyk) in wy.iter().enumerate() {
            if wyk == 0.0 {
                continue;
            }
            let ry = mirror_index(base_y + ky as i64, self.rows);
            let row_start = ry * self.cols;
            for (kx, &wxk) in wx.iter().enumerate() {
                if wxk == 0.0 {
                    continue;
                }
                let rx = mirror_index(base_x + kx as i64, self.cols);
                acc += wyk * wxk * self.data[row_start + rx];
            }
        }
        acc
    }
}

/// Weights-only variant of [`cubic_weights`] for a constant shift `s`,
/// evaluated at the reference coordinate `0 - s`: since `frac(idx - s)` is
/// identical for every integer `idx`, this is the one weight array a whole
/// `shift2d`/`shift_compare` call needs for that axis — see
/// [`SplineCoefficients2D::sample_with_weights`]'s docs. Pair with
/// [`base_index`] per row/column to track the (still `idx`-dependent) tap
/// window position.
#[inline]
pub(crate) fn weights_for_shift(shift: f64) -> [f64; 4] {
    cubic_weights(-shift).0
}

/// The leftmost of the 4 support taps covering coordinate `u`
/// (`floor(u) - 1`), i.e. just the `base` half of [`cubic_weights`]'s
/// return value, for callers that already have the weight array from
/// [`weights_for_shift`] and only need the (cheap, per-pixel) integer part
/// tracked.
#[inline]
pub(crate) fn base_index(u: f64) -> i64 {
    u.floor() as i64 - 1
}

/// Is `u` within the closed interval `[0, len-1]` — the range scipy treats
/// as "close enough to interpolate" for `mode='constant'` before falling
/// back to a hard `cval`? An empty axis (`len == 0`) is never in-domain.
#[inline]
fn in_domain(u: f64, len: usize) -> bool {
    len > 0 && u >= 0.0 && u <= (len - 1) as f64
}

/// Fold an out-of-range tap index back into `[0, len)` via whole-sample
/// mirror reflection (period `2 * (len - 1)`, no repeated edge sample),
/// matching scipy's `NI_EXTEND_MIRROR` index mapping. `len <= 1` always
/// folds to `0`.
#[inline]
fn mirror_index(idx: i64, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len as i64 - 1);
    let mut m = idx.rem_euclid(period);
    if m >= len as i64 {
        m = period - m;
    }
    m as usize
}

/// The four uniform cubic B-spline basis weights covering coordinate `u`,
/// plus the index of the first (leftmost) support tap
/// (`floor(u) - 1 .. floor(u) + 2` inclusive, i.e. 4 consecutive taps).
fn cubic_weights(u: f64) -> ([f64; 4], i64) {
    let floor_u = u.floor();
    let t = u - floor_u;
    let base = floor_u as i64 - 1;

    let one_minus_t = 1.0 - t;
    let w0 = one_minus_t * one_minus_t * one_minus_t / 6.0;
    let w1 = 2.0 / 3.0 - t * t + t * t * t / 2.0;
    let w2 = 2.0 / 3.0 - one_minus_t * one_minus_t + one_minus_t * one_minus_t * one_minus_t / 2.0;
    let w3 = t * t * t / 6.0;

    ([w0, w1, w2, w3], base)
}

/// Full-frame equivalent of `scipy.ndimage.shift(frame, (dy, dx))`
/// (`order=3, mode='constant', cval=0.0, prefilter=True`).
///
/// `frame` is row-major with `rows * cols` elements; the result has the
/// same shape. Output pixel `(r, c)` is the spline evaluated at input
/// coordinate `(r - dy, c - dx)`.
pub fn shift2d(frame: &[f64], rows: usize, cols: usize, dy: f64, dx: f64) -> Vec<f64> {
    let coeffs = SplineCoefficients2D::new(frame, rows, cols);
    let mut out = vec![0.0; rows * cols];

    // Weights depend only on the shift's fractional part, identical for
    // every row/column — computed once for the whole frame instead of
    // once per pixel (see `SplineCoefficients2D::sample_with_weights`).
    let wy = weights_for_shift(dy);
    let wx = weights_for_shift(dx);

    for r in 0..rows {
        let y = r as f64 - dy;
        // Out-of-domain rows stay all-zero (`out` was zero-initialised
        // above), matching `cval = 0.0`.
        if !in_domain(y, rows) {
            continue;
        }
        let base_y = base_index(y);
        let out_row = &mut out[r * cols..r * cols + cols];
        for (c, out_val) in out_row.iter_mut().enumerate() {
            let x = c as f64 - dx;
            if !in_domain(x, cols) {
                continue;
            }
            let base_x = base_index(x);
            *out_val = coeffs.sample_with_weights(&wy, base_y, &wx, base_x);
        }
    }
    out
}

/// Shift `frame` by `(-y, -x)` to undo a drift of `(x, y)`.
///
/// Direct port of `stabilise(frame, x, y) -> shift(frame, (-y, -x))`.
pub fn stabilise(frame: &[f64], rows: usize, cols: usize, x: f64, y: f64) -> Vec<f64> {
    shift2d(frame, rows, cols, -y, -x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_allclose(actual: &[f64], expected: &[f64], atol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= atol,
                "index {i}: actual={a} expected={e} diff={}",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn zero_shift_is_near_identity() {
        let frame = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ];
        let out = shift2d(&frame, 5, 5, 0.0, 0.0);
        assert_allclose(&out, &frame, 1e-9);
    }

    // --- SciPy parity: expected arrays generated via
    // `scipy.ndimage.shift` on the `tandemmp-refactor` conda env. ---

    #[test]
    fn scipy_parity_zero_shift_6x5() {
        let input: [f64; 30] = [
            -0.989121, -0.367787, 1.287925, 0.193974, 0.920231, 0.577104, -0.636464, 0.541952,
            -0.316595, -0.322389, 0.097167, -1.52593, 1.192166, -0.67109, 1.000269, 0.136321,
            1.532033, -0.659969, -0.311795, 0.337769, -2.207471, 0.827921, 1.54163, 1.126807,
            0.75477, -0.145978, 1.281902, 1.074031, 0.392621, 0.005114,
        ];
        let expected: [f64; 30] = [
            -0.989121,
            -0.36778700000000014,
            1.2879250000000009,
            0.19397400000000004,
            0.9202310000000007,
            0.5771040000000005,
            -0.6364639999999997,
            0.5419519999999999,
            -0.3165950000000002,
            -0.3223889999999997,
            0.09716699999999996,
            -1.5259299999999996,
            1.1921660000000005,
            -0.6710900000000002,
            1.0002690000000005,
            0.1363209999999998,
            1.5320330000000006,
            -0.6599690000000001,
            -0.31179500000000027,
            0.3377689999999999,
            -2.2074710000000017,
            0.8279210000000001,
            1.54163,
            1.126807,
            0.7547700000000004,
            -0.1459780000000003,
            1.2819020000000003,
            1.0740309999999995,
            0.39262099999999994,
            0.0051140000000000655,
        ];
        let out = shift2d(&input, 6, 5, 0.0, 0.0);
        assert_allclose(&out, &expected, 1e-9);
    }

    #[test]
    fn scipy_parity_integer_shift_6x5() {
        let input: [f64; 30] = [
            -0.989121, -0.367787, 1.287925, 0.193974, 0.920231, 0.577104, -0.636464, 0.541952,
            -0.316595, -0.322389, 0.097167, -1.52593, 1.192166, -0.67109, 1.000269, 0.136321,
            1.532033, -0.659969, -0.311795, 0.337769, -2.207471, 0.827921, 1.54163, 1.126807,
            0.75477, -0.145978, 1.281902, 1.074031, 0.392621, 0.005114,
        ];
        let expected: [f64; 30] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.36778700000000014,
            1.2879250000000009,
            0.19397400000000004,
            0.9202310000000007,
            0.0,
            -0.6364639999999997,
            0.5419519999999999,
            -0.3165950000000002,
            -0.3223889999999997,
            0.0,
            -1.5259299999999996,
            1.1921660000000005,
            -0.6710900000000002,
            1.0002690000000005,
            0.0,
            1.5320330000000006,
            -0.6599690000000001,
            -0.31179500000000027,
            0.3377689999999999,
            0.0,
        ];
        let out = shift2d(&input, 6, 5, 2.0, -1.0);
        assert_allclose(&out, &expected, 1e-9);
    }

    #[test]
    fn scipy_parity_subpixel_shift_6x5() {
        let input: [f64; 30] = [
            -0.989121, -0.367787, 1.287925, 0.193974, 0.920231, 0.577104, -0.636464, 0.541952,
            -0.316595, -0.322389, 0.097167, -1.52593, 1.192166, -0.67109, 1.000269, 0.136321,
            1.532033, -0.659969, -0.311795, 0.337769, -2.207471, 0.827921, 1.54163, 1.126807,
            0.75477, -0.145978, 1.281902, 1.074031, 0.392621, 0.005114,
        ];
        let expected: [f64; 30] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.15897235219283712,
            0.9404551427658344,
            0.1828298227549935,
            0.0,
            0.0,
            -0.7564130187889658,
            0.6570539410857164,
            -0.5076850554706253,
            0.0,
            0.0,
            -0.37499713439534776,
            0.3058788596544277,
            -0.3036465968766191,
            0.0,
            0.0,
            1.4257137545411087,
            -0.4125359934006752,
            0.35409766973434853,
            0.0,
            0.0,
            1.4171494681007906,
            1.4802300710156442,
            0.800211435886849,
            0.0,
            0.0,
        ];
        let out = shift2d(&input, 6, 5, 0.7, -1.3);
        assert_allclose(&out, &expected, 1e-9);
    }

    #[test]
    fn scipy_parity_negative_subpixel_shift_6x5() {
        let input: [f64; 30] = [
            -0.989121, -0.367787, 1.287925, 0.193974, 0.920231, 0.577104, -0.636464, 0.541952,
            -0.316595, -0.322389, 0.097167, -1.52593, 1.192166, -0.67109, 1.000269, 0.136321,
            1.532033, -0.659969, -0.311795, 0.337769, -2.207471, 0.827921, 1.54163, 1.126807,
            0.75477, -0.145978, 1.281902, 1.074031, 0.392621, 0.005114,
        ];
        let expected: [f64; 30] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.023698583198578527,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.17954609138679442,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.8464524124902582,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let out = shift2d(&input, 6, 5, -2.4, 3.6);
        assert_allclose(&out, &expected, 1e-9);
    }

    #[test]
    fn scipy_parity_square_5x5_shift() {
        let input: [f64; 25] = [
            0.00123, 0.298746, -0.274138, -0.890592, -0.454671, -0.991647, 0.060144, 1.340215,
            -0.492207, -0.620475, 0.489842, 0.356887, 0.105414, -0.930468, -0.029252, 0.695303,
            -1.344215, -0.457616, -1.901223, -1.289538, -1.841735, -0.235091, -1.267446, 0.271264,
            0.156751,
        ];
        let expected: [f64; 25] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.2972844964055428,
            1.0078833162009566,
            -0.22419791104616454,
            -0.7059782617336197,
            0.0,
            0.3144602037480414,
            0.6731560418049559,
            -0.4856274350373405,
            -0.1064646158427808,
            0.0,
            -0.6300066712651929,
            -0.4598862783077792,
            -1.6713086405525217,
            -1.3251060168734963,
            0.0,
        ];
        let out = shift2d(&input, 5, 5, 1.25, -0.75);
        assert_allclose(&out, &expected, 1e-9);
    }

    #[test]
    fn stabilise_zero_shift_is_identity() {
        let frame: Vec<f64> = (0..25).map(|v| v as f64).collect();
        let out = stabilise(&frame, 5, 5, 0.0, 0.0);
        assert_allclose(&out, &frame, 1e-9);
    }

    #[test]
    fn stabilise_undoes_shift_in_centre_region() {
        // Build a frame, shift it, then stabilise should approximately
        // recover the original away from the boundary (mirrors
        // TestStabilise.test_undoes_known_shift).
        let rows = 15;
        let cols = 15;
        let mut frame = vec![0.0; rows * cols];
        // Deterministic pseudo-random-ish pattern (no external RNG needed
        // for this smoke test).
        for (i, v) in frame.iter_mut().enumerate() {
            *v = ((i as f64) * 0.173).sin() * 2.0 + ((i as f64) * 0.081).cos();
        }
        let shifted = shift2d(&frame, rows, cols, 2.0, -1.0);
        let out = stabilise(&shifted, rows, cols, -1.0, 2.0);

        // Compare the centre region only (edges are affected by
        // constant-mode boundary artefacts, same caveat the Python test
        // documents).
        for r in 3..rows - 3 {
            for c in 3..cols - 3 {
                let a = out[r * cols + c];
                let b = frame[r * cols + c];
                assert!((a - b).abs() < 1e-6, "r={r} c={c} a={a} b={b}");
            }
        }
    }
}
