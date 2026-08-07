"""1-D Gaussian fitting utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tandemmp.fitting.minimize import curve_fit_model


def gaussian(
    x: npt.NDArray[np.floating],
    a0: float,
    x0: float,
    sigma: float,
) -> npt.NDArray[np.floating]:
    """Evaluate a 1-D Gaussian.

    Uses ``abs(A0)`` so the amplitude is always positive.
    """
    return np.asarray(abs(a0) * np.exp((-((x - x0) ** 2)) / (2.0 * sigma**2)))


def fit_gaussian(
    x_data: npt.NDArray[np.floating],
    y_data: npt.NDArray[np.floating],
    sigma_percentage_guess: float = 0.08,
) -> npt.NDArray[np.floating]:
    """Fit a 1-D Gaussian to *(x_data, y_data)*.

    Returns the optimised parameters ``(A0, x0, sigma)``.
    """
    a_index_guess = int(np.argmax(y_data))
    a_guess = float(y_data[a_index_guess])
    mean_guess = float(x_data[a_index_guess])
    sigma_guess = mean_guess * sigma_percentage_guess

    fit = curve_fit_model(
        gaussian, x_data, y_data, p0=[a_guess, mean_guess, sigma_guess]
    )
    return np.asarray(fit.params)


def gaussian_fit_histogram(
    data: npt.NDArray[np.floating],
    bins: int = 100,
) -> tuple[float, float, float]:
    """Histogram *data*, then fit a Gaussian to the histogram.

    Returns ``(amplitude, centre, sigma)``.
    """
    counts, bin_edges = np.histogram(data, bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    popt = fit_gaussian(bin_centres, counts.astype(float))
    return float(popt[0]), float(popt[1]), float(popt[2])
