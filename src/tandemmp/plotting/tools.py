"""Base plotting utilities for TandemMP."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

import numpy as np

from tandemmp.fitting.gaussian import gaussian
from tandemmp.fitting.minimize import curve_fit_model

if TYPE_CHECKING:
    import matplotlib.axes
    import numpy.typing as npt


# ---------------------------------------------------------------------------
# Decorator for auto-creating axes
# ---------------------------------------------------------------------------


def easy_plot(
    nx: int = 1,
    ny: int = 1,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that creates ``(fig, axes)`` if *axes* is not supplied.

    Decorated functions must accept an ``axes`` keyword argument.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if "axes" not in kwargs or kwargs["axes"] is None:
                import matplotlib.pyplot as plt

                _, axes = plt.subplots(ny, nx, dpi=300)
                kwargs["axes"] = axes
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Scatter + marginal histograms layout
# ---------------------------------------------------------------------------


def scatter_hist(
    margin: float = 0.1,
    scatter_size: float = 0.65,
    spacing: float = 0.015,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[matplotlib.axes.Axes, matplotlib.axes.Axes, matplotlib.axes.Axes]:
    """Create a figure with a central scatter plot and marginal histograms.

    Returns ``(ax_scatter, ax_hist_x, ax_hist_y)``.

    Extra *kwargs* are forwarded to :func:`matplotlib.pyplot.figure`.
    """
    import matplotlib.pyplot as plt

    rect_scatter = (margin, margin, scatter_size, scatter_size)
    rect_histx = (
        margin,
        margin + scatter_size + spacing,
        scatter_size,
        1 - (2 * margin) - scatter_size - spacing,
    )
    rect_histy = (
        margin + scatter_size + spacing,
        margin,
        1 - (2 * margin) - scatter_size - spacing,
        scatter_size,
    )

    fig = plt.figure(**kwargs)
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx)
    ax_histy = fig.add_axes(rect_histy)

    return ax, ax_histx, ax_histy


# ---------------------------------------------------------------------------
# Gaussian overlay on histogram
# ---------------------------------------------------------------------------


def add_gaussian_fit(
    axes: matplotlib.axes.Axes,
    data: npt.NDArray[np.floating[Any]],
    bins: npt.NDArray[np.floating[Any]],
    *,
    mean_guess: float | None = None,
    amplitude_guess: float | None = None,
    sigma_guess: float | None = None,
    color: str = "black",
    annotate: bool = False,
) -> None:
    """Fit a single Gaussian to *data* and overlay the curve on *axes*.

    Args:
        axes: Target axes (should already contain a histogram of *data*).
        data: Raw 1-D observations.
        bins: Bin edges used when histogramming *data*.
        mean_guess: Initial guess for the mean. Inferred from data when
            ``None``.
        amplitude_guess: Initial guess for the amplitude. Inferred from
            data when ``None``.
        sigma_guess: Initial guess for the sigma. Inferred from data when
            ``None``.
        color: Line colour for the fitted curve.
        annotate: If ``True``, add a text annotation with fitted parameters.
    """
    counts, bin_edges = np.histogram(data, bins)
    bin_mid = (bin_edges[1:] + bin_edges[:-1]) / 2

    if mean_guess is None:
        mean_guess = float(np.mean(data))
    if amplitude_guess is None:
        amplitude_guess = float(np.max(counts))
    if sigma_guess is None:
        sigma_guess = mean_guess / 10.0

    fit = curve_fit_model(
        gaussian,
        bin_mid,
        counts.astype(float),
        p0=(amplitude_guess, mean_guess, sigma_guess),
    )
    popt = fit.params

    lims = axes.get_xlim()
    x_space = np.linspace(lims[0], lims[1], len(bins) ** 2)
    axes.plot(x_space, gaussian(x_space, *popt), color=color)

    if annotate:
        axes.annotate(
            f"Counts = {int(np.sum(counts))}\n"
            f"Mass = {popt[1]:.2f}\n"
            f"Sigma = {popt[2]:.2f}",
            xy=(popt[1], popt[0] * 1.1),
            horizontalalignment="center",
        )
