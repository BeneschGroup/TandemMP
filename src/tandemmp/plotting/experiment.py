"""Experiment-level plotting functions.

Public functions take plain numeric arrays, not event or experiment objects;
callers extract the per-event / per-particle quantities first (e.g. with the
:mod:`tandemmp.io.save` record builders).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from tandemmp.plotting.tools import scatter_hist

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# Mass histogram


def mass_histogram(
    masses: npt.ArrayLike,
    *,
    xlim: tuple[float, float] | None = None,
    bins: int | npt.ArrayLike = 400,
    ax: Axes | None = None,
    title: str = "Mass histogram",
    saveas: str | None = None,
) -> Axes:
    """Plot a 1-D mass histogram.

    Args:
        masses: Per-event masses in kDa.
        xlim: ``(low, high)`` mass-axis bounds; defaults to ``(min, max)``.
        bins: Number of bins or explicit bin edges.
        ax: Existing axes to draw on.
        title: Plot title.
        saveas: Filepath to save the figure to.
    """
    masses_arr = np.asarray(masses, dtype=float)
    if masses_arr.size == 0:
        msg = "masses must be non-empty"
        raise ValueError(msg)

    if xlim is None:
        xlim = (float(masses_arr.min()), float(masses_arr.max()))

    # matplotlib's `Axes.hist` stub accepts `Sequence[float]` for `bins`, not ndarray,
    # so keep this a list.
    bin_edges: list[float] | int
    if isinstance(bins, int):
        bin_edges = list(np.linspace(xlim[0], xlim[1], bins))
    else:
        bin_edges = list(np.asarray(bins, dtype=float))

    if ax is None:
        _, ax = plt.subplots(dpi=300)

    ax.hist(masses_arr, bins=bin_edges)
    ax.set(xlabel="Mass (kDa)", ylabel="Counts", title=title, xlim=xlim)

    if saveas:
        ax.figure.savefig(saveas)  # type: ignore[union-attr]
    return ax


# Initial / secondary mass scatter


def mass_tablecloth(
    initial_mass: npt.ArrayLike,
    secondary_mass: npt.ArrayLike,
    *,
    xlim: tuple[float, float] | None = None,
    bins: int = 400,
    s: float = 4.0,
    alpha: float = 1.0,
    color: npt.ArrayLike | None = None,
    saveas: str | None = None,
) -> tuple[Figure, Axes, Axes, Axes]:
    """Plot a 2-D "tablecloth" of initial vs. secondary event masses.

    Returns ``(figure, scatter_ax, x_hist_ax, y_hist_ax)``.
    """
    initial = np.asarray(initial_mass, dtype=float)
    secondary = np.asarray(secondary_mass, dtype=float)
    if initial.shape != secondary.shape:
        msg = "initial_mass and secondary_mass must have the same shape"
        raise ValueError(msg)
    if initial.size == 0:
        msg = "mass arrays must be non-empty"
        raise ValueError(msg)

    if xlim is None:
        lo = float(min(initial.min(), secondary.min()))
        hi = float(max(initial.max(), secondary.max()))
        xlim = (lo, hi)

    # See `mass_histogram` for why this stays a list (mpl stub requirement).
    bin_edges = list(np.linspace(xlim[0], xlim[1], bins))

    ax, ax_histx, ax_histy = scatter_hist(figsize=(8, 8), dpi=300)
    fig: Figure = ax.figure  # type: ignore[assignment]

    if color is not None:
        ax.scatter(initial, secondary, s=s, alpha=alpha, c=np.asarray(color))
    else:
        ax.scatter(initial, secondary, s=s, alpha=alpha)

    ax.plot(xlim, xlim, color="black", linewidth=0.4)
    ax_histx.hist(initial, bins=bin_edges)
    ax_histx.set(xlim=xlim)
    ax_histy.hist(secondary, bins=bin_edges, orientation="horizontal")
    ax_histy.set(ylim=xlim)
    ax.set(
        xlabel="Initial mass (kDa)",
        ylabel="Secondary mass (kDa)",
        xlim=xlim,
        ylim=xlim,
    )
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    if saveas:
        fig.savefig(saveas)
    return fig, ax, ax_histx, ax_histy


# Mass vs error


def error_vs_mass(
    masses: npt.ArrayLike,
    errors: npt.ArrayLike,
    *,
    xlim: tuple[float, float] | None = None,
    bins: int = 1000,
    hist2d: bool = False,
    alpha: float = 1.0,
    log_error: bool = True,
    title: str | None = None,
    saveas: str | None = None,
) -> tuple[Figure, Axes, Axes, Axes]:
    """Scatter (or 2-D histogram) of fit error against mass, with side hists.

    Returns ``(figure, scatter_ax, x_hist_ax, y_hist_ax)``.
    """
    masses_arr = np.asarray(masses, dtype=float)
    errors_arr = np.asarray(errors, dtype=float)
    if masses_arr.shape != errors_arr.shape:
        msg = "masses and errors must have the same shape"
        raise ValueError(msg)
    if masses_arr.size == 0:
        msg = "input arrays must be non-empty"
        raise ValueError(msg)
    if errors_arr.min() <= 0:
        msg = "errors must all be positive"
        raise ValueError(msg)

    if xlim is None:
        xlim = (float(masses_arr.min()), float(masses_arr.max()))

    # See `mass_histogram` for why these stay lists (mpl stub requirement).
    xbins = list(np.linspace(xlim[0], xlim[1], bins))
    ybins = list(
        np.logspace(
            np.log10(float(errors_arr.min())),
            np.log10(float(errors_arr.max())),
            bins,
        )
    )

    ax, ax_histx, ax_histy = scatter_hist(figsize=(8, 8), dpi=300)
    fig: Figure = ax.figure  # type: ignore[assignment]

    if hist2d:
        ax.hist2d(
            masses_arr, errors_arr, [xbins, ybins], cmap="BuPu", norm=mcolors.LogNorm()
        )
    else:
        ax.scatter(masses_arr, errors_arr, s=1, alpha=alpha, zorder=2)

    ax_histx.hist(masses_arr, xbins, zorder=2)
    ax_histy.hist(errors_arr, ybins, orientation="horizontal", zorder=2)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    yscale = "log" if log_error else "linear"
    ax.set(
        xlabel="Mass (kDa)",
        ylabel="Fit error",
        yscale=yscale,
        ylim=(float(errors_arr.min()), float(errors_arr.max())),
        xlim=xlim,
    )
    if title is not None:
        ax.set_title(title)

    if saveas:
        fig.savefig(saveas)

    return fig, ax, ax_histx, ax_histy


# Events cube


#: Unit cube as 6 quad faces, each a (4, 3) array of corner coordinates. Scaled and
#: translated per event in ``_cuboid``; built once here, not per call.
_CUBOID_TEMPLATE: npt.NDArray[np.floating] = np.array(
    [
        [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
        [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
        [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
    ],
    dtype=float,
)


def events_cube(
    positions: npt.ArrayLike,
    *,
    widths: npt.ArrayLike,
    is_bad: npt.ArrayLike | None = None,
    movie_shape: tuple[int, int, int],
    frame_limit: int | None = None,
    figsize: tuple[float, float] = (10.0, 10.0),
    alpha: float = 0.1,
    view: tuple[float | None, float | None] = (None, None),
) -> Axes:
    """Render a 3-D cube plot of event positions (x, y, frame).

    Args:
        positions: ``(n_events, 3)`` array of ``(x, y, frame)``.
        widths: ``(n_events, 3)`` array of each cube's ``(dx, dy, dt)``.
        is_bad: Boolean array of length ``n_events``; ``True`` marks a bad event
            (red). Defaults to all-good (green).
        movie_shape: Movie shape ``(i_dim, y_dim, x_dim)``.
        frame_limit: Upper frame bound used to filter events.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    pos = np.asarray(positions, dtype=float)
    w = np.asarray(widths, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        msg = "positions must have shape (n, 3)"
        raise ValueError(msg)
    if w.shape != pos.shape:
        msg = "widths must have the same shape as positions"
        raise ValueError(msg)

    i_dim, y_dim, x_dim = movie_shape
    if y_dim <= 0:
        msg = f"movie_shape y-dimension must be positive, got {y_dim}"
        raise ValueError(msg)
    if frame_limit is None:
        frame_limit = i_dim

    sel = pos[:, 2] < frame_limit
    pos = pos[sel]
    w = w[sel]
    if is_bad is None:
        colors = np.array(["green"] * len(pos))
    else:
        bad = np.asarray(is_bad, dtype=bool)[sel]
        colors = np.where(bad, "red", "green")

    def _cuboid(
        origin: npt.NDArray[np.floating], size: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        scaled = _CUBOID_TEMPLATE.copy()
        for axis in range(3):
            scaled[:, :, axis] *= size[axis]
        return np.asarray(scaled + origin, dtype=float)

    polys = [_cuboid(pos[i], w[i]) for i in range(len(pos))]
    if polys:
        collection = Poly3DCollection(
            np.concatenate(polys),
            facecolors=np.repeat(colors, 6),
            alpha=alpha,
            edgecolor="k",
        )
    else:
        collection = Poly3DCollection([], facecolors=[], alpha=alpha)

    fig = plt.figure(dpi=300, figsize=figsize)
    ax: Any = fig.add_subplot(projection="3d")
    ax.view_init(*view)
    ax.add_collection3d(collection)
    ax.set_box_aspect([x_dim / y_dim, 1, 2])
    ax.set(
        xlim=(0, x_dim),
        ylim=(0, y_dim),
        zlim=(0, frame_limit),
        xlabel="X (pixels)",
        ylabel="Y (pixels)",
        zlabel="Time (frame)",
    )
    return ax  # type: ignore[no-any-return]
