//! Experimental spline PSF for TandemMP: `RectBivariateSpline`-parity
//! tensor-product B-spline evaluation ([`bspline`]), TandemMP's PSF-image
//! rendering convention ([`model`]), a Levenberg-Marquardt pixel-residual
//! fitter ([`fit`]), and PyO3 bindings ([`python`], exposed as
//! `tandemmp._rust.psf`).

pub mod bspline;
pub mod fit;
pub mod model;
pub mod python;
