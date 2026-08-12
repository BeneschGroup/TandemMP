//! The drift-correction hot loop: fused shift + RMSE + differential
//! evolution, replacing `tandemmp.drift.corrector._shift_compare` and
//! `._solve`.
//!
//! The whole point of porting this instead of just wrapping
//! `scipy.ndimage.shift` is that `_solve`'s Python version recomputes the
//! baseline frame's cubic-spline prefilter *from scratch on every DE
//! candidate evaluation*, even though the baseline is constant for the
//! whole `_solve` call. [`BaselineFrame`] computes it exactly once and
//! reuses it for every candidate `(x, y)` the optimiser tries.

use crate::optimize::{differential_evolution, Bound, DeConfig};
use crate::scalar::float_to_slice;
use crate::spline::SplineCoefficients2D;

/// A baseline frame with its cubic B-spline coefficients precomputed once,
/// ready to be repeatedly shifted and compared against candidate `(x, y)`
/// offsets — the piece that makes the differential-evolution inner loop
/// cheap.
pub struct BaselineFrame {
    coeffs: SplineCoefficients2D,
}

impl BaselineFrame {
    /// `baseline_frame` is row-major with `rows * cols` elements.
    pub fn new(baseline_frame: &[f64], rows: usize, cols: usize) -> Self {
        Self {
            coeffs: SplineCoefficients2D::new(baseline_frame, rows, cols),
        }
    }

    pub fn rows(&self) -> usize {
        self.coeffs.rows()
    }

    pub fn cols(&self) -> usize {
        self.coeffs.cols()
    }

    /// RMSE between `input_frame` and this baseline shifted by `(x, y)`,
    /// both cropped via `_float_to_slice(x)` / `_float_to_slice(y)` to
    /// discard edge artefacts introduced by the shift.
    ///
    /// Direct port of `_shift_compare`, except the baseline's spline
    /// coefficients are reused from construction time instead of being
    /// recomputed here.
    pub fn shift_compare(&self, x: f64, y: f64, input_frame: &[f64]) -> f64 {
        let rows = self.coeffs.rows();
        let cols = self.coeffs.cols();
        assert_eq!(
            input_frame.len(),
            rows * cols,
            "input_frame must have the same shape as the baseline frame"
        );

        let (row_start, row_stop) = float_to_slice(y).resolve(rows);
        let (col_start, col_stop) = float_to_slice(x).resolve(cols);

        let count = (row_stop.saturating_sub(row_start)) * (col_stop.saturating_sub(col_start));
        if count == 0 {
            // Mirrors NumPy's `mean` of an empty array: NaN.
            return f64::NAN;
        }

        let mut sum_sq = 0.0;
        for r in row_start..row_stop {
            let y_coord = r as f64 - y;
            let input_row = &input_frame[r * cols..r * cols + cols];
            for (c, &actual) in input_row.iter().enumerate().take(col_stop).skip(col_start) {
                let x_coord = c as f64 - x;
                let shifted = self.coeffs.sample(y_coord, x_coord);
                let diff = shifted - actual;
                sum_sq += diff * diff;
            }
        }
        (sum_sq / count as f64).sqrt()
    }
}

/// Standalone equivalent of `_shift_compare((x, y), input_frame,
/// baseline_frame)`. Prefer [`BaselineFrame`] directly when comparing the
/// same baseline against many candidate shifts (e.g. inside an
/// optimiser), since this convenience wrapper reprefilters the baseline
/// on every call.
pub fn shift_compare(
    x: f64,
    y: f64,
    input_frame: &[f64],
    baseline_frame: &[f64],
    rows: usize,
    cols: usize,
) -> f64 {
    BaselineFrame::new(baseline_frame, rows, cols).shift_compare(x, y, input_frame)
}

/// Search-space bounds (shared by both the x and y axes), mirroring
/// `_solve`'s `bounds: tuple[float, float] = (-6.0, 6.0)` default.
#[derive(Debug, Clone, Copy)]
pub struct SolveBounds {
    pub low: f64,
    pub high: f64,
}

impl Default for SolveBounds {
    fn default() -> Self {
        Self {
            low: -6.0,
            high: 6.0,
        }
    }
}

/// Find the `(x, y)` shift minimising RMSE between `input_frame` and
/// `baseline_frame`, via differential evolution + Nelder-Mead polish.
///
/// Direct port of `_solve`: `baseline_frame`'s spline prefilter is
/// computed exactly once (via [`BaselineFrame::new`]) and reused for
/// every candidate the optimiser evaluates — the fix for the Python
/// version's per-candidate `scipy.ndimage.shift` prefilter recomputation.
///
/// `seed` drives the optimiser's internal RNG for reproducibility; pass
/// `None` to seed from OS entropy (genuinely stochastic, matching `_solve`
/// left at its `seed=None` default).
pub fn solve(
    input_frame: &[f64],
    baseline_frame: &[f64],
    rows: usize,
    cols: usize,
    bounds: SolveBounds,
    seed: Option<u64>,
) -> (f64, f64) {
    let baseline = BaselineFrame::new(baseline_frame, rows, cols);
    assert_eq!(
        input_frame.len(),
        rows * cols,
        "input_frame must have the same shape as the baseline frame"
    );

    let de_bounds = [
        Bound::new(bounds.low, bounds.high),
        Bound::new(bounds.low, bounds.high),
    ];
    let objective = |v: &[f64]| baseline.shift_compare(v[0], v[1], input_frame);
    let config = DeConfig::default();
    let seed_value = seed.unwrap_or_else(random_seed);

    let result = differential_evolution(objective, &de_bounds, &config, seed_value);
    (result.x[0], result.x[1])
}

fn random_seed() -> u64 {
    use rand::RngCore;
    rand::thread_rng().next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spline::shift2d;

    /// A synthetic frame with a few sharp Gaussian-ish bumps, giving the
    /// RMSE landscape a clear minimum — mirrors
    /// `tests/drift/test_corrector.py::_make_drifting_movie`'s base frame.
    fn textured_frame(size: usize) -> Vec<f64> {
        let mut base = vec![0.0; size * size];
        for r in 0..size {
            for c in 0..size {
                let v = ((r as f64) * 0.31 + (c as f64) * 0.17).sin()
                    + ((r as f64) * 0.11 - (c as f64) * 0.23).cos() * 0.5;
                base[r * size + c] = v;
            }
        }
        let bumps: [(f64, f64, f64); 3] =
            [(12.0, 14.0, 8.0), (28.0, 22.0, -6.0), (20.0, 30.0, 5.0)];
        for r in 0..size {
            for c in 0..size {
                let mut extra = 0.0;
                for &(cx, cy, amp) in &bumps {
                    let dx = c as f64 - cx;
                    let dy = r as f64 - cy;
                    extra += amp * (-(dx * dx + dy * dy) / 4.0).exp();
                }
                base[r * size + c] += extra;
            }
        }
        base
    }

    #[test]
    fn shift_compare_zero_shift_identical_frames_is_zero() {
        let frame = textured_frame(20);
        let rmse = shift_compare(0.0, 0.0, &frame, &frame, 20, 20);
        assert!(rmse.abs() < 1e-9, "rmse={rmse}");
    }

    #[test]
    fn shift_compare_correct_shift_minimises() {
        let base = textured_frame(25);
        let (true_dx, true_dy) = (1.0, -2.0);
        let shifted = shift2d(&base, 25, 25, true_dy, true_dx);

        let rmse_at_truth = shift_compare(true_dx, true_dy, &shifted, &base, 25, 25);
        let rmse_at_zero = shift_compare(0.0, 0.0, &shifted, &base, 25, 25);
        assert!(
            rmse_at_truth < rmse_at_zero,
            "rmse_at_truth={rmse_at_truth} rmse_at_zero={rmse_at_zero}"
        );
    }

    #[test]
    fn solve_recovers_known_integer_shift() {
        let base = textured_frame(31);
        let (true_dx, true_dy) = (2.0, -1.0);
        let shifted = shift2d(&base, 31, 31, true_dy, true_dx);

        let (x, y) = solve(&shifted, &base, 31, 31, SolveBounds::default(), Some(42));
        assert!((x - true_dx).abs() < 0.2, "x={x} true_dx={true_dx}");
        assert!((y - true_dy).abs() < 0.2, "y={y} true_dy={true_dy}");
    }

    #[test]
    fn solve_recovers_subpixel_shift() {
        let base = textured_frame(31);
        let (true_dx, true_dy) = (0.7, -1.3);
        let shifted = shift2d(&base, 31, 31, true_dy, true_dx);

        let (x, y) = solve(&shifted, &base, 31, 31, SolveBounds::default(), Some(42));
        assert!((x - true_dx).abs() < 0.3, "x={x} true_dx={true_dx}");
        assert!((y - true_dy).abs() < 0.3, "y={y} true_dy={true_dy}");
    }

    #[test]
    fn solve_zero_shift_for_identical_frames() {
        let frame = textured_frame(31);
        let (x, y) = solve(&frame, &frame, 31, 31, SolveBounds::default(), Some(42));
        assert!(x.abs() < 0.3, "x={x}");
        assert!(y.abs() < 0.3, "y={y}");
    }

    #[test]
    fn solve_same_seed_gives_reproducible_result() {
        let base = textured_frame(31);
        let shifted = shift2d(&base, 31, 31, -1.0, 2.0);

        let first = solve(&shifted, &base, 31, 31, SolveBounds::default(), Some(123));
        let second = solve(&shifted, &base, 31, 31, SolveBounds::default(), Some(123));
        assert_eq!(first, second);
    }
}
