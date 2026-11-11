"""Tests for :mod:`tandemmp.fitting.psf_refit`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from tandemmp.fitting.gaussian_psf import GaussianPsfModel
from tandemmp.fitting.psf import PsfModel
from tandemmp.fitting.psf_interface import PsfInterface
from tandemmp.fitting.psf_refit import refit_psf_parameters

GRID = np.arange(-6.0, 6.0 + 0.5, 0.5)


def _render(model: PsfInterface, c: float) -> npt.NDArray[np.floating]:
    xx, yy = np.meshgrid(GRID, GRID)
    return model.get_psf_function()(xx, yy, c=c)


class _PolyGaussian:
    """Dummy model with N params: Gaussian whose sigma is the sum of params."""

    def __init__(self, **params: float) -> None:
        self._params = params

    @property
    def psf_params(self) -> dict[str, float]:
        return dict(self._params)

    @property
    def param_bounds(self) -> dict[str, tuple[float | None, float | None]]:
        return {k: (0.1, None) for k in self._params}

    def with_params(self, **params: float) -> _PolyGaussian:
        return _PolyGaussian(**{**self._params, **params})

    def get_psf_function(self) -> Callable[..., npt.NDArray[np.floating]]:
        sigma = sum(self._params.values()) or 1.0
        return lambda x, y, c=1.0: c * np.exp(-(x**2 + y**2) / (2 * sigma**2))


class TestRefitPsfParameters:
    def test_recovers_gaussian_sigma(self) -> None:
        target = _render(GaussianPsfModel(sigma=2.5), c=-1.0)
        model, image, sse = refit_psf_parameters(
            GaussianPsfModel(sigma=1.2), target, GRID
        )
        assert model.psf_params["sigma"] == pytest.approx(2.5, rel=1e-3)
        assert sse == pytest.approx(0.0, abs=1e-8)
        assert image.shape == target.shape

    def test_recovers_bessel_params(self) -> None:
        pytest.importorskip("open_iscat")
        truth = PsfModel(a12=-5.0, w=2.3, s=4.0)
        target = _render(truth, c=-1.0)
        start = PsfModel(a12=-4.0, w=2.0, s=4.5)
        model, _, sse = refit_psf_parameters(start, target, GRID)
        start_sse = float(np.sum((_render(start, -1.0) - target) ** 2))
        assert sse < 1e-3 * start_sse
        assert isinstance(model, PsfModel)
        assert not model.has_spline_model

    @pytest.mark.parametrize("n_params", [0, 1, 5])
    def test_arbitrary_param_count(self, n_params: int) -> None:
        target = _render(GaussianPsfModel(sigma=2.0), c=-1.0)
        start = _PolyGaussian(**{f"p{i}": 0.3 for i in range(n_params)})
        start_sse = float(np.sum((_render(start, -1.0) - target) ** 2))
        model, _, sse = refit_psf_parameters(start, target, GRID)
        assert set(model.psf_params) == set(start.psf_params)
        assert sse <= start_sse
        if n_params:
            assert sse == pytest.approx(0.0, abs=1e-6)
            assert all(v >= 0.1 for v in model.psf_params.values())

    def test_matches_pre_refactor_fit_psf0(self) -> None:
        oi_psf = pytest.importorskip("open_iscat.utils.psf")
        rng = np.random.default_rng(0)
        target = _render(PsfModel(a12=-5.0, w=2.3, s=4.0), c=-1.0)
        target = target + rng.normal(0, 0.01, target.shape)
        # Same start for both: c=-1 and PsfModel defaults; a2 = c/(1+a12), a1 = a12*a2.
        start = PsfModel()
        a2_guess = -1.0 / (1 + start.a12)
        _, old_fit = oi_psf.fit_psf0(
            target,
            start.a12 * a2_guess,
            a2_guess,
            start.w,
            start.s,
            full_output=True,
            x=GRID,
        )
        old_sse = float(np.sum((old_fit - target) ** 2))
        _, _, new_sse = refit_psf_parameters(start, target, GRID, c_guess=-1.0)
        assert new_sse == pytest.approx(old_sse, rel=0.05)
