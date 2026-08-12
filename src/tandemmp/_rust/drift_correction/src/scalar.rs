//! Small scalar helpers ported from `tandemmp.drift.corrector`.
//!
//! These mirror `_rmse`, `_most_positive`, `_most_negative`, and
//! `_float_to_slice` closely enough that a PyO3 binding layer can expose
//! them 1:1 if ever useful for testing parity from the Python side.

/// Root-mean-square error between two equal-length slices.
///
/// Mirrors `_rmse(a, b) = sqrt(mean((a - b) ** 2))`. Panics if the slices
/// have different lengths (mirrors NumPy's broadcast error for mismatched
/// shapes, just surfaced as a Rust panic instead of a Python exception —
/// the PyO3 layer should turn this into a `ValueError`).
pub fn rmse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "rmse: slices must have equal length");
    if a.is_empty() {
        return f64::NAN;
    }
    let sum_sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
    (sum_sq / a.len() as f64).sqrt()
}

/// The largest non-negative value in `values`, or `0.0` if none exists.
///
/// Mirrors `_most_positive`.
pub fn most_positive(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .filter(|v| *v >= 0.0)
        .fold(None, |acc: Option<f64>, v| match acc {
            Some(m) => Some(m.max(v)),
            None => Some(v),
        })
        .unwrap_or(0.0)
}

/// The smallest negative value in `values`, or `0.0` if none exists.
///
/// Mirrors `_most_negative`.
pub fn most_negative(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .filter(|v| *v < 0.0)
        .fold(None, |acc: Option<f64>, v| match acc {
            Some(m) => Some(m.min(v)),
            None => Some(v),
        })
        .unwrap_or(0.0)
}

/// A Python-slice-like `(start, stop)` pair, mirroring the object returned
/// by `_float_to_slice`, before it is resolved against a concrete axis
/// length (Python slices are lazy about the array they'll index into —
/// this type keeps that same two-step shape: build a [`FloatSlice`], then
/// [`FloatSlice::resolve`] it against a length).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatSlice {
    /// `None` mirrors Python's `slice(None, stop)` (start defaults to 0).
    pub start: Option<i64>,
    /// `None` mirrors Python's `slice(start, None)` (stop defaults to the
    /// axis length).
    pub stop: Option<i64>,
}

/// Mirrors `_float_to_slice(n)`:
///
/// ```python
/// def _float_to_slice(n: float) -> slice:
///     if n >= 0:
///         return slice(int(n) + 1, None)
///     return slice(None, int(n) - 1)
/// ```
///
/// `n as i64` truncates toward zero, exactly like Python's `int()` on a
/// float, so the two implementations agree bit-for-bit on the integer part.
pub fn float_to_slice(n: f64) -> FloatSlice {
    if n >= 0.0 {
        FloatSlice {
            start: Some(n as i64 + 1),
            stop: None,
        }
    } else {
        FloatSlice {
            start: None,
            stop: Some(n as i64 - 1),
        }
    }
}

impl FloatSlice {
    /// Resolve this slice against an axis of length `len`, returning
    /// `(start, stop)` indices such that `start <= stop <= len`, matching
    /// NumPy/Python slice-indexing semantics (negative `stop` counts back
    /// from the end; out-of-range bounds clamp rather than panic).
    pub fn resolve(&self, len: usize) -> (usize, usize) {
        let len_i = len as i64;
        let start = match self.start {
            Some(s) => s.clamp(0, len_i),
            None => 0,
        };
        let stop = match self.stop {
            Some(s) => {
                let resolved = if s < 0 { len_i + s } else { s };
                resolved.clamp(0, len_i)
            }
            None => len_i,
        };
        (start as usize, stop.max(start) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmse_zero_for_identical() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(rmse(&a, &a), 0.0);
    }

    #[test]
    fn rmse_known_value() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        // mean((9, 16)) = 12.5 -> sqrt = 3.5355...
        assert!((rmse(&a, &b) - 12.5f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn most_positive_with_positives() {
        assert_eq!(most_positive(&[-2.0, 3.0, 1.0]), 3.0);
    }

    #[test]
    fn most_positive_all_negative() {
        assert_eq!(most_positive(&[-2.0, -3.0]), 0.0);
    }

    #[test]
    fn most_negative_with_negatives() {
        assert_eq!(most_negative(&[-2.0, 3.0, 1.0]), -2.0);
    }

    #[test]
    fn most_negative_all_positive() {
        assert_eq!(most_negative(&[2.0, 3.0]), 0.0);
    }

    #[test]
    fn float_to_slice_positive() {
        let s = float_to_slice(2.5);
        assert_eq!(
            s,
            FloatSlice {
                start: Some(3),
                stop: None
            }
        );
    }

    #[test]
    fn float_to_slice_negative() {
        let s = float_to_slice(-2.5);
        assert_eq!(
            s,
            FloatSlice {
                start: None,
                stop: Some(-3)
            }
        );
    }

    #[test]
    fn float_to_slice_zero() {
        let s = float_to_slice(0.0);
        assert_eq!(
            s,
            FloatSlice {
                start: Some(1),
                stop: None
            }
        );
    }

    #[test]
    fn resolve_positive_slice() {
        // slice(3, None) on length 10 -> (3, 10)
        let s = float_to_slice(2.5);
        assert_eq!(s.resolve(10), (3, 10));
    }

    #[test]
    fn resolve_negative_slice() {
        // slice(None, -3) on length 10 -> (0, 7)
        let s = float_to_slice(-2.5);
        assert_eq!(s.resolve(10), (0, 7));
    }

    #[test]
    fn resolve_clamps_out_of_range() {
        // A start far beyond len clamps to len, giving an empty (but valid) range.
        let s = FloatSlice {
            start: Some(1000),
            stop: None,
        };
        assert_eq!(s.resolve(10), (10, 10));
    }
}
