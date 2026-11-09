"""Histogram plotting — single-Gaussian and GMM overlays.

Fitting logic (bin selection, curve fitting, GMM selection) lives in
:mod:`tandemmp.core.histogram`; this module owns the matplotlib calls that
render those fits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm

from tandemmp.core.histogram import (
    _line,
    _prepare_means_for_line_fit,
    duplicate_inliers,
    fit_gmm,
    fit_means,
    scott_bin_count,
)
from tandemmp.fitting.minimize import curve_fit_model

if TYPE_CHECKING:
    import matplotlib.axes
    import numpy.typing as npt
    from sklearn.mixture import GaussianMixture


def plot_histogram(
    data: npt.NDArray[np.floating[Any]],
    bins: int | npt.NDArray[np.floating[Any]] | None = None,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot a histogram of *data*.

    Returns the :class:`~matplotlib.axes.Axes` used.
    """
    import matplotlib.pyplot as plt

    if bins is None:
        bins = scott_bin_count(data)
    if ax is None:
        _, ax = plt.subplots(dpi=300, figsize=(7, 7))

    ax.hist(data, bins=bins, color="gray", alpha=1)  # type: ignore[arg-type]
    ax.set(ylabel="Counts")
    return ax


def plot_gmm(
    data: npt.NDArray[np.floating[Any]],
    max_components: int,
    criterion: str = "bic",
    bins: int | npt.NDArray[np.floating[Any]] | None = None,
    *,
    duplicate_clusters: int | None = None,
    cut: bool = False,
    fit_means_flag: bool = False,
    axes: list[matplotlib.axes.Axes] | None = None,
    dpi: float = 400,
    adjust_unit: bool = False,
    crop: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    plot: bool = True,
) -> GaussianMixture | tuple[GaussianMixture, float | None, float | None]:
    """Fit a GMM and optionally plot the result.

    When *fit_means_flag* is ``True``, returns
    ``(gmm, r_squared, slope)``; otherwise returns *gmm*.
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data, dtype=float)

    if crop is not None:
        data = data[(data > crop[0]) & (data < crop[1])]

    if duplicate_clusters is not None:
        data = duplicate_inliers(data, duplicates=duplicate_clusters, cut=cut)

    if bins is None:
        bins = scott_bin_count(data)
    if isinstance(bins, (int, np.integer)):
        bins = np.linspace(float(np.min(data)), float(np.max(data)), int(bins))

    bin_width = float(bins[1] - bins[0])

    gmm = fit_gmm(data, max_components, criterion, xlim=xlim)
    x = np.linspace(float(np.min(bins)), float(np.max(bins)), 1000).reshape(-1, 1)
    density = np.exp(gmm.score_samples(x))
    total_count = len(data)

    if plot and axes is None:
        cols = 2 if fit_means_flag else 1
        _, axes = plt.subplots(1, cols, dpi=dpi)
        if cols == 1:
            axes = [axes]  # type: ignore[list-item]

    if plot:
        assert axes is not None
        plot_histogram(data, bins, ax=axes[0])

        for i, (mean, covar, weight) in enumerate(
            zip(gmm.means_, gmm.covariances_, gmm.weights_, strict=True)
        ):
            std_dev = float(np.sqrt(covar[0][0]))
            p = norm.pdf(x, mean, std_dev)
            scaled = p * total_count * bin_width * weight
            axes[0].plot(
                x,
                scaled,
                label=f"Gaussian {i + 1}",
                linewidth=0.5,
            )

        gmm_curve = density * total_count * bin_width
        axes[0].plot(x, gmm_curve, "b", linewidth=2, label="GMM")
        axes[0].set(
            title="Gaussian Mixture Model",
            xlabel="Mass (kDa)",
            ylabel="Counts",
        )

    if fit_means_flag:
        r2, slope = fit_means(gmm, adjust_unit=adjust_unit)

        if plot and r2 is not None:
            assert axes is not None
            _plot_means_on_ax(gmm, axes[1], adjust_unit=adjust_unit)

        return gmm, r2, slope

    return gmm


def _plot_means_on_ax(
    gmm: GaussianMixture,
    ax: matplotlib.axes.Axes,
    sigma_limits: tuple[float, float] = (2.0, 80.0),
    adjust_unit: bool = False,
) -> None:
    """Plot GMM component means with a line of best fit on *ax*."""
    prepared = _prepare_means_for_line_fit(gmm, sigma_limits, adjust_unit)
    if prepared is None:
        return
    steps, means, sigmas = prepared

    fit = curve_fit_model(_line, steps, means, sigma=sigmas, absolute_sigma=True)
    popt = fit.params
    perr = fit.param_errors
    r2 = fit.r_squared

    x_fit = np.linspace(float(np.min(steps)), float(np.max(steps)), 100)
    y_fit = _line(x_fit, *popt)

    ax.scatter(steps, means, label="Means")
    ax.plot(x_fit, y_fit, "r")
    ax.text(
        float(np.min(steps)),
        float(np.max(means)),
        f"y = ({popt[0]:.2f} ± {perr[0]:.2f})x, r²={r2:.5f}",
        size=14,
    )
    ax.set(xlabel="Arbitrary units", ylabel="Mass (kDa)")
