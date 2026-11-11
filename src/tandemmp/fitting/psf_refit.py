"""Refit a PSF model's shape parameters to a centred target image."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tandemmp.fitting.minimize import minimize_model
from tandemmp.fitting.psf_interface import PsfInterface, error_metric


def refit_psf_parameters(
    model: PsfInterface,
    target: npt.NDArray[np.floating],
    x: npt.NDArray[np.floating],
    *,
    c_guess: float | None = None,
    method: str = "BFGS",
) -> tuple[PsfInterface, npt.NDArray[np.floating], float]:
    """Fit *model*'s shape parameters and a contrast to *target* on grid ``(x, x)``.

    Seeds from the model's current parameters. Returns
    ``(refit_model, rendered_image, sse)``.
    """
    names = sorted(model.psf_params)
    if c_guess is None:
        c_guess = float(target.ravel()[np.abs(target).argmax()])
    x0 = [c_guess, *(model.psf_params[n] for n in names)]

    xx, yy = np.meshgrid(x, x)

    def render(
        v: npt.NDArray[np.floating],
    ) -> tuple[PsfInterface, npt.NDArray[np.floating]]:
        m = model.with_params(**dict(zip(names, map(float, v[1:]), strict=True)))
        return m, m.get_psf_function()(xx, yy, c=float(v[0]))

    def err(v: npt.NDArray[np.floating]) -> float:
        return error_metric(target, render(v)[1])

    bounds_map = model.param_bounds
    bounds = None
    if bounds_map:
        bounds = [(None, None), *(bounds_map.get(n, (None, None)) for n in names)]
        if method == "BFGS":  # BFGS ignores bounds
            method = "L-BFGS-B"
    fit = minimize_model(err, x0, method=method, bounds=bounds)
    refit_model, image = render(fit.params)
    return refit_model, image, error_metric(target, image)
