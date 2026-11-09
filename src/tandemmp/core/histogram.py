"""Histogram analysis utilities — bin selection, Gaussian fitting, GMM.

Plotting for these fits lives in :mod:`tandemmp.plotting.histogram`; this
module is pure data/fitting with no matplotlib dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from tandemmp.fitting.gaussian import gaussian
from tandemmp.fitting.minimize import curve_fit_model

if TYPE_CHECKING:
    import numpy.typing as npt


# Private curve-fit targets


def _line(
    x: npt.NDArray[np.floating[Any]],
    m: float,
    c: float,
) -> npt.NDArray[np.floating[Any]]:
    return m * x + c


# Bin selection


def scott_bin_count(data: npt.NDArray[np.floating[Any]]) -> int:
    """Return the number of histogram bins using Scott's rule.

    Returns 1 when the standard deviation is zero.
    """
    std_dev = float(np.std(data))
    if std_dev == 0:
        return 1
    bin_width = 0.65 * std_dev / (len(data) ** (1 / 3))
    return int(np.ceil((np.max(data) - np.min(data)) / bin_width))


# Single-Gaussian fitting


def fit_gaussian_to_histogram(
    data: npt.NDArray[np.floating[Any]],
    bins: int | npt.NDArray[np.floating[Any]] | None = None,
) -> tuple[float, float, float]:
    """Fit a single Gaussian to the histogram of *data*.

    Returns ``(amplitude, mean, sigma)``.

    Raises :class:`ValueError` if *data* contains NaN values.
    """
    if np.isnan(data).any():
        raise ValueError("NaN values found in data")

    if bins is None:
        bins = scott_bin_count(data)
    if isinstance(bins, (int, np.integer)):
        bins = np.linspace(float(np.min(data)), float(np.max(data)), int(bins))

    counts, bin_edges = np.histogram(data, bins=bins)
    bin_mid = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Exclude zero counts
    sel = counts > 0
    counts = counts[sel]
    bin_mid = bin_mid[sel]

    fit = curve_fit_model(
        gaussian,
        bin_mid,
        counts.astype(float),
        p0=(float(np.max(counts)), float(np.mean(data)), float(np.std(data))),
    )
    popt = fit.params
    return float(popt[0]), float(popt[1]), float(popt[2])


# Gaussian Mixture Model


def fit_gmm(
    data: npt.NDArray[np.floating[Any]],
    max_components: int,
    criterion: str = "bic",
    xlim: tuple[float, float] | None = None,
) -> GaussianMixture:
    """Fit a :class:`~sklearn.mixture.GaussianMixture` with BIC/AIC model selection.

    Args:
        data: 1-D array of observations.
        max_components: Maximum number of Gaussian components to try.
        criterion: ``'bic'`` or ``'aic'``.
        xlim: If provided, only data within ``[xlim[0], xlim[1]]`` is used.
    """
    if criterion not in ("bic", "aic"):
        raise ValueError(f"criterion must be 'bic' or 'aic', got {criterion!r}")

    data = np.asarray(data, dtype=float)

    if xlim is not None:
        data = data[(data >= xlim[0]) & (data <= xlim[1])]

    data_2d = data.reshape(-1, 1)

    ic = []
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(data_2d)
        ic.append(gmm.bic(data_2d) if criterion == "bic" else gmm.aic(data_2d))

    optimal = int(np.argmin(ic)) + 1
    gmm = GaussianMixture(n_components=optimal, random_state=42)
    gmm.fit(data_2d)
    return gmm


def duplicate_inliers(
    data: npt.NDArray[np.floating[Any]] | list[float],
    eps: float = 0.1,
    min_samples: int = 5,
    duplicates: int = 2,
    cut: bool = False,
) -> npt.NDArray[np.floating[Any]]:
    """Use DBSCAN to identify inliers, then duplicate them.

    Args:
        data: 1-D array of observations.
        eps: DBSCAN neighbourhood radius.
        min_samples: DBSCAN minimum samples per cluster.
        duplicates: Number of times to repeat inlier points.
        cut: If ``True``, return only inlier points (no duplication).
    """
    data = np.asarray(data, dtype=float)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data.reshape(-1, 1))
    inliers = labels != -1

    if cut:
        return np.asarray(data[inliers])

    return np.asarray(np.concatenate([data, np.repeat(data[inliers], duplicates)]))


# Fit line to GMM component means


def _prepare_means_for_line_fit(
    gmm: GaussianMixture,
    sigma_limits: tuple[float, float],
    adjust_unit: bool,
) -> (
    tuple[
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
    ]
    | None
):
    """Filter and sort GMM component means ahead of a line fit.

    Shared by :func:`fit_means` and
    :func:`tandemmp.plotting.histogram._plot_means_on_ax`.

    Components whose standard deviation falls outside *sigma_limits* are
    dropped, the remainder are sorted by ascending mean, and per-component
    "steps" are computed (unit spacing, or inferred from median spacing
    between sorted means when *adjust_unit* is ``True``).

    Returns ``(steps, means, sigmas)`` or ``None`` if fewer than two
    components survive the sigma filter.
    """
    means = gmm.means_[:, 0]
    sigmas = np.array([float(np.sqrt(c[0][0])) for c in gmm.covariances_])

    keep = (sigmas > sigma_limits[0]) & (sigmas < sigma_limits[1])
    means = means[keep]
    sigmas = sigmas[keep]

    if len(means) < 2:
        return None

    order = np.argsort(means)
    means = means[order]
    sigmas = sigmas[order]

    if adjust_unit:
        diffs = np.diff(means)
        unit_size = float(np.median(diffs))
        steps = np.concatenate([[0.0], np.cumsum(np.round(diffs / unit_size))])
    else:
        steps = np.arange(1, len(means) + 1, dtype=float)

    return steps, means, sigmas


def fit_means(
    gmm: GaussianMixture,
    sigma_limits: tuple[float, float] = (2.0, 80.0),
    adjust_unit: bool = False,
) -> tuple[float | None, float | None]:
    """Fit a weighted line to the means of *gmm* components.

    Args:
        gmm: A fitted :class:`~sklearn.mixture.GaussianMixture`.
        sigma_limits: ``(min, max)`` standard-deviation range; components
            outside are excluded.
        adjust_unit: If ``True``, infer step sizes from median spacing
            between sorted means rather than assuming unit spacing.

    Returns:
        ``(r_squared, slope)``, or ``(None, None)`` if fewer than two
        components survive the sigma filter.
    """
    prepared = _prepare_means_for_line_fit(gmm, sigma_limits, adjust_unit)
    if prepared is None:
        return None, None
    steps, means, sigmas = prepared

    fit = curve_fit_model(_line, steps, means, sigma=sigmas, absolute_sigma=True)

    return fit.r_squared, float(fit.params[0])
