"""Bundled experimental spline PSF (ml_mp_movements-derived, RectBivariateSpline).

Loads ``tandemmp/resources/exp_psf_20260423_mfp2.npz`` (converted once by
``scripts/convert_psf.py``, no ml_mp_movements/open_iscat dependency at
runtime) into a :class:`~tandemmp.fitting.psf.PsfModel` with the Rust
``tandemmp._rust.psf.SplinePsf`` backend.
"""

from __future__ import annotations

import functools
import importlib.resources

import numpy as np
from scipy.interpolate import BivariateSpline

from tandemmp.fitting.psf import PsfModel

_RESOURCE_NAME = "exp_psf_20260423_mfp2.npz"


@functools.lru_cache(maxsize=1)
def _load_bundled_spline() -> BivariateSpline:
    """Load and cache the bundled experimental PSF as a SciPy spline."""
    ref = importlib.resources.files("tandemmp.resources").joinpath(_RESOURCE_NAME)
    with importlib.resources.as_file(ref) as path:
        data = np.load(path)
        tx, ty, c = data["tx"], data["ty"], data["c"]
        kx, ky = int(data["kx"]), int(data["ky"])
    return BivariateSpline._from_tck((tx, ty, c, kx, ky))  # noqa: SLF001


@functools.lru_cache(maxsize=4)
def get_experimental_psf_model(*, use_rust: bool = True) -> PsfModel:
    """Return a cached :class:`PsfModel` backed by the bundled spline PSF."""
    model = PsfModel(use_rust=use_rust)
    model.set_spline_model(_load_bundled_spline())
    return model
