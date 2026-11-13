//! Bundled experimental spline PSF (20260423, ms1000, mfp2) for TandemMP.
//!
//! The PSF's `RectBivariateSpline` knots/coefficients are compiled in from
//! `data/*.bin` (written by `scripts/export_spline_psf.py`). Axis order is
//! TandemMP's: `eval(row, col)`, peak -1 at the origin.

use std::sync::OnceLock;

pub mod bspline;
pub mod python;

pub use bspline::Spline;

const DEGREE: usize = 3;

fn decode(bytes: &[u8]) -> Vec<f64> {
    // Any trailing partial chunk is ignored, as with `chunks_exact`.
    let (chunks, _remainder) = bytes.as_chunks::<8>();
    chunks.iter().map(|b| f64::from_le_bytes(*b)).collect()
}

/// The bundled PSF spline, decoded on first use.
pub fn bundled() -> &'static Spline {
    static PSF: OnceLock<Spline> = OnceLock::new();
    PSF.get_or_init(|| {
        Spline::new(
            decode(include_bytes!("../data/tx.bin")),
            decode(include_bytes!("../data/ty.bin")),
            decode(include_bytes!("../data/c.bin")),
            DEGREE,
            DEGREE,
        )
        .expect("bundled PSF data is valid")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ones_spline() -> Spline {
        let (tu, tv) = bundled().knots();
        let c = vec![1.0; bundled().coefficients().len()];
        Spline::new(tu.to_vec(), tv.to_vec(), c, DEGREE, DEGREE).unwrap()
    }

    #[test]
    fn partition_of_unity() {
        let s = ones_spline();
        for &(u, v) in &[(0.0, 0.0), (-11.93, 3.21), (5.55, -7.7), (12.0, -12.0)] {
            assert!((s.eval(u, v) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn clamps_outside_domain() {
        let s = bundled();
        assert_eq!(s.eval(100.0, 2.5), s.eval(12.0, 2.5));
        assert_eq!(s.eval(-1.0, -50.0), s.eval(-1.0, -12.0));
    }

    #[test]
    fn bundled_peak_is_minus_one() {
        assert!((bundled().eval(0.0, 0.0) + 1.0).abs() < 2e-3);
    }

    #[test]
    fn rejects_bad_coefficient_count() {
        let (tu, tv) = bundled().knots();
        assert!(Spline::new(tu.to_vec(), tv.to_vec(), vec![0.0; 3], 3, 3).is_err());
    }
}
