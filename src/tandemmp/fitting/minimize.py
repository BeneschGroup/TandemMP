"""Generic minimisation helpers shared by every curve/model fit in TandemMP.

Provides :func:`minimize_model` and :func:`curve_fit_model`, thin wrappers
around :func:`scipy.optimize.minimize` and :func:`scipy.optimize.curve_fit`
that return uniform, typed result dataclasses (:class:`FitResult` and
:class:`CurveFitResult`).  Consolidating on these two entry points removes
the repeated ``curve_fit``-plus-manual-R²-and-parameter-error boilerplate
that was previously duplicated across :mod:`tandemmp.core.histogram`,
:mod:`tandemmp.fitting.gaussian`, and :mod:`tandemmp.plotting.tools`, and the
repeated ``minimize``-plus-``OptimizeResult``-unpacking boilerplate in
:mod:`tandemmp.fitting.psf` and :mod:`tandemmp.pipeline.investigator`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import scipy.optimize

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclasses.dataclass(frozen=True)
class FitResult:
    """Uniform result of a :func:`scipy.optimize.minimize` call."""

    params: npt.NDArray[np.floating]
    fun: float
    success: bool
    method: str
    raw: scipy.optimize.OptimizeResult


def minimize_model(
    objective: Callable[[npt.NDArray[np.floating]], float],
    x0: Sequence[float] | npt.NDArray[np.floating],
    *,
    method: str = "BFGS",
    bounds: Sequence[Sequence[float | None]] | None = None,
    options: dict[str, Any] | None = None,
) -> FitResult:
    """Minimise *objective* from *x0*, returning a uniform :class:`FitResult`.

    Thin wrapper around :func:`scipy.optimize.minimize`.  ``raw`` carries the
    underlying :class:`~scipy.optimize.OptimizeResult` for callers whose
    public API contract exposes it directly.
    """
    res = scipy.optimize.minimize(
        objective, x0, method=method, bounds=bounds, options=options
    )
    return FitResult(
        params=np.asarray(res.x),
        fun=float(res.fun),
        success=bool(res.success),
        method=method,
        raw=res,
    )


@dataclasses.dataclass(frozen=True)
class CurveFitResult:
    """Uniform result of a :func:`scipy.optimize.curve_fit` call."""

    params: npt.NDArray[np.floating]
    covariance: npt.NDArray[np.floating]
    param_errors: npt.NDArray[np.floating]
    residual_sum_of_squares: float
    r_squared: float


def curve_fit_model(
    model: Callable[..., npt.NDArray[np.floating]],
    x_data: npt.NDArray[np.floating],
    y_data: npt.NDArray[np.floating],
    p0: Sequence[float] | None = None,
    *,
    sigma: npt.NDArray[np.floating] | None = None,
    absolute_sigma: bool = False,
) -> CurveFitResult:
    """Fit *model* to *(x_data, y_data)*, computing R² and parameter errors once.

    Thin wrapper around :func:`scipy.optimize.curve_fit`.
    """
    popt, pcov = scipy.optimize.curve_fit(
        model, x_data, y_data, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma
    )
    param_errors = np.sqrt(np.diag(pcov))

    residuals = y_data - model(x_data, *popt)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_data - np.mean(y_data)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return CurveFitResult(
        params=popt,
        covariance=pcov,
        param_errors=param_errors,
        residual_sum_of_squares=ss_res,
        r_squared=r_squared,
    )
