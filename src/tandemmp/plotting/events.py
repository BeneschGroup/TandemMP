"""Plotting for the events subpackage — peak images, fit diagnostics, reports.

Event classes in :mod:`tandemmp.events` hold fit state (peak images,
positions, corrections, ...) but perform no matplotlib calls themselves;
this module renders that state. Functions here take the event/series object
directly (duck-typed on their public and semi-private attributes) rather
than pre-extracted plain arrays, matching the precedent set in
:mod:`tandemmp.plotting.animation` — the state involved (connections,
correction attempts, smoothed trajectories) is too interlinked across
objects to flatten into array arguments without a larger redesign than the
Layer 6k move warrants.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import numpy.typing as npt

from tandemmp.core.functions import nan_normalise
from tandemmp.events._base import AbstractEvent
from tandemmp.events.ratiometric import RatiometricEvent
from tandemmp.events.subtraction import SubtractionEvent, SubtractionSeries
from tandemmp.plotting.tools import easy_plot

# ---------------------------------------------------------------------------
# AbstractEvent — peak image and basic fit
# ---------------------------------------------------------------------------


def show(
    event: AbstractEvent,
    *,
    axes: Any = None,  # noqa: ANN401
    vmin: float = -0.01,
    vmax: float = 0.01,
    saveas: str | None = None,
    title: str | None | bool = None,
) -> None:
    """Quick image show of *event*'s peak.

    If *axes* is ``None`` a new figure is created.
    """
    import matplotlib.pyplot as plt

    @easy_plot()
    def _show(
        event_: AbstractEvent,
        axes: Any,  # noqa: ANN401  # matplotlib axes
        vmin: float,
        vmax: float,
        saveas: str | None,
        title: str | None | bool,
    ) -> None:
        width: int = event_.peak.image.shape[0] // 2

        axes.imshow(
            event_.peak.image,
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )

        if hasattr(event_, "exclusion") and event_.exclusion is not None:
            red, green = (254, 0, 0), (0, 254, 0)
            coloured = np.zeros(
                (*event_.exclusion.shape, 3),
                dtype=int,
            )
            if event_.exclusion.any():
                coloured[event_.exclusion] = red
            coloured[~event_.exclusion] = green
            axes.imshow(coloured, origin="lower", cmap="RdYlGn", alpha=0.1)

        if event_.has_connection:
            axes.plot(
                width + event_.smooth_dx,  # type: ignore[attr-defined]
                width + event_.smooth_dy,  # type: ignore[attr-defined]
                alpha=0.5,
            )
        elif hasattr(event_, "_is_bad") and event_._is_bad:
            axes.scatter(
                width + event_.interp_dx,  # type: ignore[attr-defined]
                width + event_.interp_dy,  # type: ignore[attr-defined]
                c="tab:purple",
                alpha=0.3,
            )
        else:
            axes.scatter(
                width + event_.dx,
                width + event_.dy,
                c="tab:blue",
                alpha=0.3,
            )

        if title is None:
            axes.set_title(repr(event_))
        elif isinstance(title, str):
            axes.set_title(title)

        if saveas:
            plt.savefig(f"{saveas}")

    _show(event, axes=axes, vmin=vmin, vmax=vmax, saveas=saveas, title=title)


def show_basic_fit(
    event: AbstractEvent,
    *,
    axes: Any = None,  # noqa: ANN401
    vmin: float = -0.01,
    vmax: float = 0.01,
) -> None:
    """Show *event*'s basic fit and residual side by side."""

    @easy_plot(2)
    def _plot(
        event_: AbstractEvent,
        axes: Any,  # noqa: ANN401
        vmin: float,
        vmax: float,
    ) -> None:
        replicate_fit, residual = event_._basic_replicate_fit()
        axes[0].imshow(replicate_fit, vmin=vmin, vmax=vmax, cmap="gray", origin="lower")
        axes[0].set_title("Basic fit")
        axes[1].imshow(residual, vmin=vmin, vmax=vmax, cmap="gray", origin="lower")
        axes[1].set_title("Basic residual")

    _plot(event, axes=axes, vmin=vmin, vmax=vmax)


def _show_fits_basic(
    event: AbstractEvent,
    *,
    axes: Any = None,  # noqa: ANN401
    vmin: float = -0.01,
    vmax: float = 0.01,
    saveas: str | None = None,
) -> None:
    """Show raw data, basic fit, and residual for *event*."""
    import matplotlib.pyplot as plt

    @easy_plot(3)
    def _plot(
        event_: AbstractEvent,
        axes: Any,  # noqa: ANN401
        vmin: float,
        vmax: float,
        saveas: str | None,
    ) -> None:
        show(event_, axes=axes[0], vmin=vmin, vmax=vmax, title="Raw data")
        show_basic_fit(event_, axes=axes[1:], vmin=vmin, vmax=vmax)
        if saveas:
            plt.savefig(f"{saveas}")

    _plot(event, axes=axes, vmin=vmin, vmax=vmax, saveas=saveas)


def show_fits(
    event: AbstractEvent,
    *,
    axes: Any = None,  # noqa: ANN401
    vmin: float = -0.01,
    vmax: float = 0.01,
    saveas: str | None = None,
) -> None:
    """Show *event*'s fit diagnostics.

    Dispatches to :func:`show_fits_improved` (raw data, 3-D fit, basic fit)
    when *event* is a :class:`~tandemmp.events.ratiometric.RatiometricEvent`
    with an applied 3-D fit; otherwise renders the basic 2-D fit (raw data,
    fit, residual). This replaces the pre-6k pattern of monkey-patching
    ``show_fits`` onto the instance in ``apply_3D_fit``/``disconnect_all``.
    """
    if isinstance(event, RatiometricEvent) and event.fit_3D:
        show_fits_improved(event, axes=axes, vmin=vmin, vmax=vmax, saveas=saveas)
    else:
        _show_fits_basic(event, axes=axes, vmin=vmin, vmax=vmax, saveas=saveas)


# ---------------------------------------------------------------------------
# SubtractionEvent / SubtractionSeries
# ---------------------------------------------------------------------------


def show_subtraction_corrections(
    event: SubtractionEvent,
    *,
    axes: Any = None,  # noqa: ANN401
    vmin: float = -0.01,
    vmax: float = 0.01,
) -> None:
    """Show the corrections applied to *event*."""
    import matplotlib.pyplot as plt

    if axes is None:
        n = len(event.peak.corrections) + 3 if event.peak.corrections else 3
        _, axes = plt.subplots(n, 1, dpi=300)

    def im(i: int, image: npt.NDArray[np.floating]) -> None:
        axes[i].imshow(image, vmin=vmin, vmax=vmax, cmap="gray", origin="lower")

    im(0, event.peak.image)
    axes[0].set_title("Corrected image")
    axes[1].set_title("Correction applied")

    if isinstance(event.peak.background, np.ndarray):
        im(1, event.peak.background)

    if event.peak.corrections:
        for j, c in enumerate(event.peak.corrections):
            peak_img = c.sliced_image
            if peak_img is not None:
                peak_img = peak_img.copy()
                peak_img[peak_img == 0] = np.nan
                im(j + 2, peak_img)
                axes[j + 2].set_title(f"Particle {c.particle}")

    if event.peak.original is not None:
        im(-1, event.peak.original)
    axes[-1].set_title("Original image")


def show_series(
    series: SubtractionSeries, vmin: float = -0.01, vmax: float = 0.01
) -> None:
    """Show every event image in *series*."""
    for event in series.events:
        show(event, vmin=vmin, vmax=vmax)


def show_series_impression(series: SubtractionSeries) -> None:
    """Show the PSF impression of *series*."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(dpi=300, figsize=(5, 5))
    ax.imshow(series.impression(), vmin=-0.01, vmax=0.01, cmap="gray", origin="lower")
    plt.draw()


def show_series_big(
    series: SubtractionSeries,
    *,
    axes: Any = None,  # noqa: ANN401
    vmin: float = -0.01,
    vmax: float = 0.01,
    saveas: str | None = None,
) -> None:
    """Show all events in *series* side by side with the ML ghost score."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, len(series.events), dpi=300)

    for ax, e in zip(axes, series.events, strict=True):
        show(e, axes=ax, title=False, vmin=vmin, vmax=vmax)
        ax.axis("off")

    ml_score = series.ghost_signal_ML
    axes[-1].set_title(
        f"ML Ghost score = {ml_score:.3f}"
        if ml_score is not None
        else "ML Ghost score = unavailable",
        loc="right",
    )
    if saveas:
        plt.savefig(f"{saveas}")


def show_series_corrections(
    series: SubtractionSeries, vmin: float = -0.01, vmax: float = 0.01
) -> None:
    """Show corrections for every event in *series*."""
    import matplotlib.pyplot as plt

    size = max(
        [0] + [len(e.peak.corrections) for e in series.events if e.peak.corrections]
    )
    _, axes = plt.subplots(len(series.events), size + 3)

    for j, e in enumerate(series.events):
        show_subtraction_corrections(e, axes=axes[j, :], vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# RatiometricEvent — 3-D fit and reports
# ---------------------------------------------------------------------------


def show_3d_fit(  # noqa: N802
    event: RatiometricEvent,
    *,
    axes: Any = None,  # noqa: ANN401
    vmin: float = -0.01,
    vmax: float = 0.01,
) -> None:
    """Show *event*'s 3-D fit and residual."""

    @easy_plot(2)
    def _plot(
        event_: RatiometricEvent,
        axes: Any,  # noqa: ANN401
        vmin: float,
        vmax: float,
    ) -> None:
        replicate_fit, residual = event_._3D_replicate_fit()
        axes[0].imshow(replicate_fit, vmin=vmin, vmax=vmax, cmap="gray", origin="lower")
        axes[0].set_title("3D fit")
        axes[1].imshow(residual, vmin=vmin, vmax=vmax, cmap="gray", origin="lower")
        axes[1].set_title("3D residual")

    _plot(event, axes=axes, vmin=vmin, vmax=vmax)


def show_fits_improved(
    event: RatiometricEvent,
    *,
    axes: Any = None,  # noqa: ANN401
    vmin: float = -0.01,
    vmax: float = 0.01,
    saveas: str | None = None,
) -> None:
    """Show raw data, 3-D fit, and basic fit together for *event*."""
    import matplotlib.pyplot as plt

    @easy_plot(5)
    def _plot(
        event_: RatiometricEvent,
        axes: Any,  # noqa: ANN401
        vmin: float,
        vmax: float,
        saveas: str | None,
    ) -> None:
        show(event_, axes=axes[0], vmin=vmin, vmax=vmax, title="Ratiometric peak")
        show_3d_fit(event_, axes=axes[1:3], vmin=vmin, vmax=vmax)
        show_basic_fit(event_, axes=axes[3:], vmin=vmin, vmax=vmax)
        if saveas:
            plt.savefig(f"{saveas}")

    _plot(event, axes=axes, vmin=vmin, vmax=vmax, saveas=saveas)


def report(
    event: RatiometricEvent,
    vmin: float = -0.01,
    vmax: float = 0.01,
    saveas: str | None = None,
) -> None:
    """Generate a detailed report figure for *event*."""
    import matplotlib.pyplot as plt

    grid_size = (7, event.navg * 2 + 1) if event.has_connection else (6, 12)
    fig = plt.figure(dpi=300, figsize=grid_size[::-1])
    if event.decision is not None:
        plt.suptitle(
            f"Event {event._ID}, frame {event.i}"
            + " " * 5
            + f"Classification = {event.classification}"
            + " " * 5
            + f"Action: {event.decision.action.name}"
            + " " * 5
            + f"Mass = {np.round(event.rel_m, 1)}kDa"
            + " " * 5
            + f"SNR = {np.round(event.snr, 3)}"
        )
    else:
        plt.suptitle(f"Event {event._ID}")

    get_grid = functools.partial(plt.subplot2grid, shape=grid_size, fig=fig)

    def place(x: int, y: int, w: int = 1, h: int = 1) -> Any:  # noqa: ANN401
        return get_grid(loc=(y, x), rowspan=h, colspan=w)

    show(
        event,
        axes=place(0, 0, 4, 4),
        title="Ratiometric peak",
        vmin=vmin,
        vmax=vmax,
    )
    show_series_big(
        event._first_attempt,  # type: ignore[arg-type]
        axes=[place(i, 5) for i in range(event.navg * 2 + 1)],
        vmin=vmin,
        vmax=vmax,
    )

    if event.has_connection:
        show_basic_fit(
            event,
            axes=[place(4, 0, 2, 2), place(6, 0, 2, 2)],
            vmin=vmin,
            vmax=vmax,
        )
        show_3d_fit(
            event,
            axes=[place(4, 2, 2, 2), place(6, 2, 2, 2)],
            vmin=vmin,
            vmax=vmax,
        )
        show_series_big(
            event.connected_events[0],
            axes=[place(i, 6) for i in range(event.navg * 2 + 1)],
            vmin=vmin,
            vmax=vmax,
        )
    else:
        show_basic_fit(
            event,
            axes=[place(4, 0, 4, 4), place(8, 0, 4, 4)],
            vmin=vmin,
            vmax=vmax,
        )

    if saveas:
        plt.savefig(saveas)


def report_and_draw(event: RatiometricEvent) -> None:
    """Generate *event*'s report and force a redraw.

    Convenience wrapper for interactive (``supervise=True``) analysis loops
    — see :meth:`tandemmp.pipeline.pca.Experiment.analyse`.
    """
    import matplotlib.pyplot as plt

    report(event)
    plt.draw()


def fit_reports(
    event: RatiometricEvent,
    vmin: float = -0.01,
    vmax: float = 0.01,
    saveas: str | None = None,
    *,
    sort: bool = True,
) -> None:
    """Show all correction attempts for *event*."""
    import matplotlib.pyplot as plt

    fits = event._corrections_attempts
    if sort:
        errors = np.array([r.error for r in fits])
        norm_errors = nan_normalise(errors)
        norm_penalty = nan_normalise(np.array([r.penalty for r in fits]))
        norm_image = nan_normalise(np.array([r.image_error for r in fits]))
        norm_distance = nan_normalise(np.array([r.distance for r in fits]))
        score = (
            (norm_errors * 2) + norm_penalty + (norm_image * 2) + (norm_distance / 2)
        )
        order = np.argsort(score)
    else:
        order = np.arange(len(fits))

    first = event._first_attempt
    fig, axes = plt.subplots(
        len(fits),
        len(first.events) + 2,  # type: ignore[union-attr]
        dpi=300,
        figsize=((len(first.events) + 4) * 2, len(fits) * 2),  # type: ignore[union-attr]
    )
    plt.suptitle(f"Event {event._ID} attempts, best to worst")

    for fit, index in zip([fits[i] for i in order], range(len(fits)), strict=True):
        ax = axes[index, :]
        show_series_big(fit.series, axes=ax[:-2], vmin=vmin, vmax=vmax)
        ax[0].set_title(
            f"View of particle {fit.particle}, "
            f"background behind by {fit.series[0].bleed_time} frames",
            loc="left",
        )
        ax[-2].imshow(
            fit.fit_image,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            cmap="gray",
        )
        ax[-2].set_title(f"3D fit {fit.fit_type}")
        ax[-2].axis("off")
        ax[-1].imshow(
            event.peak.image - fit.fit_image,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            cmap="gray",
        )
        ax[-1].set_title("Residual")
        ax[-1].axis("off")

    if saveas:
        plt.savefig(saveas)


def assignment_check(event: RatiometricEvent, saveas: str | None = None) -> None:
    """Bar chart comparing *event*'s correction attempts by error component."""
    import matplotlib.pyplot as plt

    @easy_plot(1)
    def _plot(
        event_: RatiometricEvent,
        axes: Any,  # noqa: ANN401
        saveas: str | None,
    ) -> None:
        fits = event_._corrections_attempts
        x = np.arange(len(fits))
        labels = [
            f.particle if f.particle is not None else "New particle" for f in fits
        ]
        y1 = [f.error for f in fits]
        y2 = [f.penalty for f in fits]
        y3 = [f.image_error for f in fits]
        axes.bar(x, y1, label="Fit error")
        axes.bar(x, y2, bottom=y1, label="Mass penalty")
        axes.bar(x, y3, bottom=np.add(y1, y2), label="Avg residual noise")
        axes.set_xticks(x, labels, rotation=45)
        axes.set(
            yscale="log",
            ylim=(np.min(y1) * 0.9, None),
            title=f"Event {event_._ID}",
        )
        axes.axhline(
            event_.basic_fit_error,
            linestyle="--",
            color="black",
            alpha=0.8,
        )
        plt.legend()
        if saveas:
            plt.savefig(saveas)

    _plot(event, saveas=saveas)


def ranking_plot(event: RatiometricEvent, saveas: str | None = None) -> None:
    """Stacked bar chart ranking *event*'s correction attempts."""
    import matplotlib.pyplot as plt

    @easy_plot(1)
    def _plot(
        event_: RatiometricEvent,
        axes: Any,  # noqa: ANN401
        saveas: str | None,
    ) -> None:
        fits = event_._corrections_attempts
        x = np.arange(len(fits))
        labels = [
            f.particle if f.particle is not None else "New particle" for f in fits
        ]
        errors = np.array([r.error for r in fits])
        y1 = nan_normalise(errors)
        y2 = nan_normalise(np.array([r.penalty for r in fits]))
        y3 = nan_normalise(np.array([r.image_error for r in fits]))
        y4 = nan_normalise(np.array([r.distance for r in fits]))
        axes.bar(x, y1, label="Fit error")
        axes.bar(x, y2, bottom=y1, label="Mass penalty")
        axes.bar(x, y3, bottom=np.add(y1, y2), label="Avg residual noise")
        axes.bar(x, y4, bottom=np.add(y1, y2, y3), label="Distance")
        axes.set_xticks(x, labels, rotation=45)
        axes.set(title=f"Event {event_._ID}")
        plt.legend()
        if saveas:
            plt.savefig(saveas)

    _plot(event, saveas=saveas)
