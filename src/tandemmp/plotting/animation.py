"""Animation and movie-visualisation utilities.

Generates matplotlib animations (saved as .mp4 via ffmpeg) and composite
mosaic movies built from PSF-fitted particle data.

.. note::
   ffmpeg must be available on PATH, or its location configured via
   ``matplotlib.pyplot.rcParams['animation.ffmpeg_path']`` before calling
   any function that saves video.

Pure movie processing functions (ratiometric, start_subtraction, etc.) live
in :mod:`tandemmp.io.movie`.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from tandemmp.fitting.psf import get_psf_model
from tandemmp.io.movie import (
    ratiometric as _ratiometric,
)
from tandemmp.io.movie import (
    rolling_mean as _rolling_mean,
)
from tandemmp.io.movie import (
    start_subtraction as _start_subtraction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mosaic builders (operate on arrays; no rendering)
# ---------------------------------------------------------------------------


def reconstruct_particle_movie(
    experiment: Any,  # noqa: ANN401
) -> npt.NDArray[np.floating[Any]]:
    """Reconstruct a synthetic movie from PSF fits stored in particle data.

    For each frame, renders the PSF for every particle that has fit data
    present, then sums the contributions into a single image per frame.

    Parameters
    ----------
    experiment:
        Experiment object exposing ``.movie`` (shape: frames × H × W) and
        ``.particles`` (iterable, each indexable by frame).
    """
    model = get_psf_model()
    psf_fn = model.get_psf_function()
    shape = experiment.movie.shape
    new_movie: list[npt.NDArray[np.floating[Any]]] = []
    grid_x, grid_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]))
    for frame in range(shape[0]):
        logger.debug("Frame: %d", frame)
        images = []
        for p in experiment.particles:
            data = p[frame]
            if data is not None:
                images.append(psf_fn(grid_x - data.x, grid_y - data.y, data.c))
        new_movie.append(
            np.sum(images, axis=0) if images else np.zeros((shape[1], shape[2]))
        )
    return np.array(new_movie)


def create_raw_mosaic(
    experiment: Any,  # noqa: ANN401
    new_movie: npt.NDArray[np.floating[Any]] | None = None,
    *,
    cumulative: bool = False,
    silent: bool = False,
) -> npt.NDArray[np.floating[Any]]:
    """Build a mosaic movie by stamping event peak images onto frames.

    Parameters
    ----------
    experiment:
        Experiment object exposing ``.movie`` and ``.particles``.
    new_movie:
        Pre-allocated output array.  If ``None``, filled with NaN.
    cumulative:
        If ``True``, existing pixel values are zeroed before each new event
        and subsequent contributions are summed rather than replaced.
    silent:
        If ``True``, suppress the ``tqdm`` progress bar.
    """
    if new_movie is None:
        new_movie = np.full_like(experiment.movie, fill_value=np.nan)

    iterator = experiment.particles if silent else tqdm(experiment.particles)
    for p in iterator:
        events = p.fixed_background_events
        for index in range(len(events)):
            next_ev = None if index + 1 == len(events) else events[index + 1].i + 1
            image = events[index].peak.image
            position = events[index].peak.cutout
            if p.unbinding_frame and next_ev is None:
                next_ev = p.unbinding_frame
            if cumulative:
                new_movie[events[index].i : next_ev, position[0], position[1]] = 0
                new_movie[events[index].i : next_ev, position[0], position[1]] += image
            else:
                new_movie[events[index].i : next_ev, position[0], position[1]] = image

    return new_movie


def create_mosaic_movie(
    experiment: Any,  # noqa: ANN401
    *,
    corrections: bool = True,
    new_movie: npt.NDArray[np.floating[Any]] | None = None,
    add: bool = False,
) -> npt.NDArray[np.floating[Any]]:
    """Build a mosaic movie from particle trajectory history.

    Parameters
    ----------
    experiment:
        Experiment object with ``.movie``, ``.events``, and ``.particles``.
    corrections:
        If ``True``, use the corrected event map from each particle's history.
        If ``False``, use ``_first_attempt`` sub-events.
    new_movie:
        Pre-allocated output array.  Defaults to a NaN-filled copy of
        ``experiment.movie``.
    add:
        If ``True``, add images to existing values instead of overwriting.
    """
    if new_movie is None:
        new_movie = np.full_like(experiment.movie, fill_value=np.nan)

    if not corrections:
        for event in experiment.events:
            if event._first_attempt is None:
                continue
            for sub_event in event._first_attempt:
                image = sub_event.peak.image
                position = sub_event.peak.cutout
                new_movie[sub_event.i :, position[0], position[1]] = image
    else:
        for particle in experiment.particles:
            items = list(particle.history.event_map.connected.items())
            if not items:
                continue
            frame_keys: list[int | None] = [int(item[0]) for item in items] + [None]
            ev_vals = [item[1] for item in items]
            for index in range(len(ev_vals)):
                start, end = frame_keys[index], frame_keys[index + 1]
                image = ev_vals[index].peak.image
                position = ev_vals[index].peak.cutout
                if add:
                    new_movie[start:end, position[0], position[1]] += image
                else:
                    new_movie[start:end, position[0], position[1]] = image

    return new_movie


# ---------------------------------------------------------------------------
# Animation rendering helpers
# ---------------------------------------------------------------------------


def quick_output_movie(
    movie: npt.NDArray[np.floating[Any]],
    filename: str,
    fps: int,
    title: str,
    *,
    autoscale: bool = False,
    vmin: float = -0.01,
    vmax: float = 0.01,
    cmap: str = "viridis",
) -> None:
    """Render a single-panel animation and save to *filename*.mp4.

    Parameters
    ----------
    movie:
        3-D array (frames × H × W) to animate.
    filename:
        Output path without the ``.mp4`` extension.
    fps:
        Frames per second.
    title:
        Axes title for the animation.
    autoscale:
        If ``True``, auto-scale the colour limits every frame.
    vmin, vmax:
        Colour-scale limits (ignored when *autoscale* is ``True``).
    cmap:
        Matplotlib colourmap name.
    """
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(
        movie[0], animated=True, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap
    )
    plt.colorbar(im)

    def init() -> list[matplotlib.artist.Artist]:
        ax.set_title(title)
        return [im]

    def update(frame: int) -> list[matplotlib.artist.Artist]:
        im.set_array(movie[frame])
        if autoscale:
            im.autoscale()
        return [im]

    ani = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=tqdm(range(len(movie)), desc="Compiling"),
        init_func=init,
        blit=True,
    )
    writer = mpl_animation.writers["ffmpeg"](bitrate=4000, fps=fps)
    ani.save(filename + ".mp4", writer=writer, dpi=300)
    plt.close(fig)


def create_comparison_animation(
    movie: npt.NDArray[np.floating[Any]],
    filename: str,
    frame_range: list[int],
    *,
    ratiometric_navg: int = 5,
    median_navg: int = 30,
    start_navg: int = 5,
    fps: int = 25,
    vmin: float = -0.01,
    vmax: float = 0.01,
    dpi: int = 400,
) -> None:
    """Four-panel animation: ratiometric, start-subtraction, median, and native.

    Parameters
    ----------
    movie:
        Raw 3-D movie (frames × H × W).
    filename:
        Output path without the ``.mp4`` extension.
    frame_range:
        Sequence of frame indices to animate.
    ratiometric_navg, median_navg, start_navg:
        Window sizes for each processing method.
    fps:
        Frames per second.
    vmin, vmax:
        Shared colour-scale limits for the processed panels.
    dpi:
        Figure resolution.
    """
    fig, ((ax_ratio, ax_start_sub), (ax_median, ax_native)) = plt.subplots(
        2, 2, dpi=dpi
    )

    ratio = -_ratiometric(movie, ratiometric_navg)
    med = _rolling_mean(movie, median_navg)
    start = -_start_subtraction(movie, start_navg)

    native_im = ax_native.imshow(movie[0], animated=True, origin="lower", cmap="gray")
    ratio_im = ax_ratio.imshow(
        ratio[0], animated=True, origin="lower", vmin=vmin, vmax=vmax, cmap="gray"
    )
    median_im = ax_median.imshow(
        med[0], animated=True, origin="lower", vmin=vmin, vmax=vmax, cmap="gray"
    )
    start_im = ax_start_sub.imshow(
        start[0], animated=True, origin="lower", vmin=vmin, vmax=vmax, cmap="gray"
    )

    def init() -> tuple[matplotlib.artist.Artist, ...]:
        ax_native.set_title("Native")
        ax_ratio.set_title("Ratiometric")
        ax_median.set_title("Rolling mean")
        ax_start_sub.set_title("Start subtraction")
        ax_native.axis("off")
        ax_ratio.axis("off")
        ax_median.axis("off")
        ax_start_sub.axis("off")
        return native_im, ratio_im, median_im, start_im

    def update(frame: int) -> tuple[matplotlib.artist.Artist, ...]:
        native_im.set_array(movie[frame])
        ratio_im.set_array(ratio[frame])
        median_im.set_array(med[frame])
        start_im.set_array(start[frame])
        return native_im, ratio_im, median_im, start_im

    ani = mpl_animation.FuncAnimation(
        fig, update, frames=frame_range, init_func=init, blit=True
    )
    writer = mpl_animation.writers["ffmpeg"](bitrate=4000, fps=fps)
    ani.save(filename + ".mp4", writer=writer, dpi=dpi)
    plt.close(fig)


def create_navg_scaling_animation(
    movie: npt.NDArray[np.floating[Any]],
    dc_movie: npt.NDArray[np.floating[Any]],
    movie_fps: float,
    filename: str = "navg_comparison",
    *,
    frame: int | None = None,
    seconds: float = 10,
    num_points: int = 501,
    max_window_seconds: float = 15,
    vmin: float = -0.01,
    vmax: float = 0.01,
    cmap: str = "gray",
) -> None:
    """Animate how image quality and noise scale with *navg*.

    Shows ratiometric images for both a raw and drift-corrected movie as
    *navg* sweeps from 1 up to the maximum available window, alongside a
    log-log noise-vs-integration-time curve.

    The animation is saved as ``navg comparison.mp4`` in the working directory.

    Parameters
    ----------
    movie, dc_movie:
        Raw and drift-corrected 3-D movies of the same shape.
    movie_fps:
        Acquisition frame rate (used to convert *navg* to seconds).
    frame:
        Central frame to use for the ratiometric computation.  Defaults to
        ``len(movie) // 2``.
    seconds:
        Target duration of the saved animation.
    num_points:
        Number of *navg* values to sample (log-spaced).
    max_window_seconds:
        Cap on the largest averaging window (in seconds) to avoid running off
        the end of the movie or producing an animation dominated by extreme
        navg values.
    vmin, vmax:
        Shared colour-scale limits.
    cmap:
        Matplotlib colourmap name.
    """
    if frame is None:
        frame = len(movie) // 2

    max_navg = min(len(movie) - frame, frame)
    max_window_frames = int(movie_fps * max_window_seconds)
    max_navg = min(max_navg, max_window_frames)

    navgs = np.array(
        sorted(set(np.logspace(0, np.log10(max_navg), num_points).astype(int)))
    )

    def get_frame(
        array: npt.NDArray[np.floating[Any]], n: int
    ) -> npt.NDArray[np.floating[Any]]:
        # fmt: off
        return (  # type: ignore[no-any-return]
            array[frame + 1 : frame + n + 1].sum(axis=0)
            / array[frame - n : frame].sum(axis=0)
        ) - 1
        # fmt: on

    norm_noises: list[float] = [float(get_frame(movie, 1).std())]
    dc_noises: list[float] = [float(get_frame(dc_movie, 1).std())]
    max_val = float(np.percentile(abs(get_frame(movie, 1)), 50))

    fig, (ax_norm, ax_dc, ax_hist, ax_curve) = plt.subplots(
        1,
        4,
        dpi=300,
        figsize=(20, 3),
        width_ratios=[0.35, 0.35, 0.15, 0.15],
        layout="constrained",
    )

    im_norm = ax_norm.imshow(
        get_frame(movie, 1), vmin=vmin, vmax=vmax, cmap=cmap, animated=True
    )
    im_dc = ax_dc.imshow(
        get_frame(dc_movie, 1), vmin=vmin, vmax=vmax, cmap=cmap, animated=True
    )
    (line_norm,) = ax_curve.plot(
        [1 / movie_fps], norm_noises, label="Normal", linewidth=0.3
    )
    (line_dc,) = ax_curve.plot(
        [1 / movie_fps], dc_noises, label="Drift corrected", linewidth=0.3
    )
    plt.colorbar(im_dc, label="Contrast")

    def init() -> tuple[matplotlib.artist.Artist, ...]:
        ax_norm.set_title("Normal")
        ax_dc.set_title("Drift corrected")
        ax_hist.set(title="Pixel values", xlabel="Contrast", ylabel="Counts")
        ax_curve.set(
            title="Standard deviation",
            xlabel="Integration time (s)",
            ylabel="Contrast",
            xscale="log",
            yscale="log",
            xlim=(
                (navgs[0] / movie_fps) * 0.5,
                (navgs[-1] / movie_fps) * 2,
            ),
        )
        ax_curve.legend(loc="upper right")
        return im_norm, im_dc

    def update(index: int) -> tuple[matplotlib.artist.Artist, ...]:
        n = navgs[index]
        norm_image = get_frame(movie, n)
        dc_image = get_frame(dc_movie, n)

        norm_noises.append(float(norm_image.std()))
        dc_noises.append(float(dc_image.std()))

        im_norm.set_array(norm_image)
        im_dc.set_array(dc_image)

        line_norm.set_xdata(navgs[: index + 1] / movie_fps)
        line_norm.set_ydata(norm_noises)
        line_dc.set_xdata(navgs[: index + 1] / movie_fps)
        line_dc.set_ydata(dc_noises)
        ax_curve.set(ylim=(min(dc_noises) * 0.5, norm_noises[0] * 2))

        bins = np.linspace(-max_val, max_val, 800)
        ax_hist.clear()
        ax_hist.set(title="Pixel values", xlabel="Contrast", ylabel="Counts")
        ax_hist.hist(np.ravel(norm_image), bins=bins, label="Normal", alpha=0.5)
        ax_hist.hist(np.ravel(dc_image), bins=bins, label="Drift correction", alpha=0.5)
        plt.suptitle(f"Navg = {n} frames ({n / movie_fps}s)")
        return im_norm, im_dc

    ani = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=tqdm(np.arange(1, len(navgs)), desc="Making animation"),
        init_func=init,
        blit=True,
    )

    logger.info(
        "Number of data points: %d, effective fps: %g",
        len(navgs),
        len(navgs) / seconds,
    )
    # Floor to 1 fps — short movies can produce len(navgs) < seconds, which
    # would otherwise pass fps=0 to ffmpeg and crash.
    effective_fps = max(1, int(len(navgs) / seconds))
    writer = mpl_animation.writers["ffmpeg"](bitrate=3000, fps=effective_fps)
    ani.save(filename + ".mp4", writer=writer, dpi=300)
    plt.close(fig)


def create_method_comparison(
    movie: npt.NDArray[np.floating[Any]],
    frame: int,
    *,
    ratiometric_navg: int = 5,
    median_navg: int = 100,
    start_navg: int = 5,
    start_frame: int = 6,
    dpi: int = 300,
    cmap: str = "gray",
    limits: tuple[float, float] = (-0.01, 0.01),
) -> matplotlib.figure.Figure:
    """Create a static four-panel figure comparing background-removal methods.

    Returns the :class:`matplotlib.figure.Figure`; does **not** save to disk.

    Parameters
    ----------
    movie:
        Raw 3-D movie (frames × H × W).
    frame:
        Frame index to display.
    ratiometric_navg, median_navg, start_navg:
        Window sizes for each method.
    start_frame:
        Reference frame index for start subtraction.
    dpi:
        Figure resolution.
    cmap:
        Matplotlib colourmap name.
    limits:
        ``(vmin, vmax)`` colour-scale limits for the processed panels.
    """
    fig, (native_ax, ratio_ax, median_ax, start_ax) = plt.subplots(4, 1, dpi=dpi)

    ratio = _ratiometric(movie, ratiometric_navg)
    med = _rolling_mean(movie, median_navg)
    start = _start_subtraction(movie, start_navg, start_frame=start_frame)

    native_ax.imshow(movie[frame], cmap=cmap)
    ratio_ax.imshow(ratio[frame], cmap=cmap, vmin=limits[0], vmax=limits[1])
    median_ax.imshow(med[frame], cmap=cmap, vmin=limits[0], vmax=limits[1])
    start_ax.imshow(start[frame], cmap=cmap, vmin=limits[0], vmax=limits[1])

    return fig


def create_drift_correction_animation(
    uncorrected: npt.NDArray[np.floating[Any]],
    corrected: npt.NDArray[np.floating[Any]],
    filename: str,
    *,
    ratiometric_navg: int = 5,
    start_navg: int = 5,
    fps: int = 25,
    vmin: float = -0.01,
    vmax: float = 0.01,
) -> None:
    """Four-panel animation comparing uncorrected vs drift-corrected movies.

    Shows ratiometric and start-subtraction views for both inputs side-by-side.

    Parameters
    ----------
    uncorrected, corrected:
        3-D movies of the same shape (frames × H × W).
    filename:
        Output path without the ``.mp4`` extension.
    ratiometric_navg, start_navg:
        Window sizes for the respective methods.
    fps:
        Frames per second.
    vmin, vmax:
        Shared colour-scale limits for all four panels.
    """
    fig, ((ax_un_ratio, ax_c_ratio), (ax_un_start, ax_c_start)) = plt.subplots(
        2, 2, dpi=300
    )

    un_ratio = _ratiometric(uncorrected, ratiometric_navg)
    c_ratio = _ratiometric(corrected, ratiometric_navg)
    un_start = _start_subtraction(uncorrected, start_navg)
    c_start = _start_subtraction(corrected, start_navg)

    un_ratio_im = ax_un_ratio.imshow(
        un_ratio[0], animated=True, origin="lower", vmin=vmin, vmax=vmax
    )
    c_ratio_im = ax_c_ratio.imshow(
        c_ratio[0], animated=True, origin="lower", vmin=vmin, vmax=vmax
    )
    un_start_im = ax_un_start.imshow(
        un_start[0], animated=True, origin="lower", vmin=vmin, vmax=vmax
    )
    c_start_im = ax_c_start.imshow(
        c_start[0], animated=True, origin="lower", vmin=vmin, vmax=vmax
    )

    def init() -> tuple[matplotlib.artist.Artist, ...]:
        ax_un_ratio.set_title("Uncorrected ratiometric")
        ax_c_ratio.set_title("Corrected ratiometric")
        ax_un_start.set_title("Uncorrected start subtraction")
        ax_c_start.set_title("Corrected start subtraction")
        return un_ratio_im, c_ratio_im, un_start_im, c_start_im

    def update(frame: int) -> tuple[matplotlib.artist.Artist, ...]:
        un_ratio_im.set_array(un_ratio[frame])
        c_ratio_im.set_array(c_ratio[frame])
        un_start_im.set_array(un_start[frame])
        c_start_im.set_array(c_start[frame])
        return un_ratio_im, c_ratio_im, un_start_im, c_start_im

    ani = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=tqdm(np.arange(len(corrected)), desc="Making animation"),
        init_func=init,
        blit=True,
    )
    writer = mpl_animation.writers["ffmpeg"](bitrate=3000, fps=fps)
    ani.save(filename + ".mp4", writer=writer, dpi=300)
    plt.close(fig)


def mosaic_comparison(
    experiment: Any,  # noqa: ANN401
    filename: str,
    fps: int,
    *,
    vmin: float = -0.01,
    vmax: float = 0.01,
) -> None:
    """Three-panel animation: original, uncorrected mosaic, corrected mosaic.

    Parameters
    ----------
    experiment:
        Experiment object with ``.movie`` and ``.particles``.
    filename:
        Output path without the ``.mp4`` extension.
    fps:
        Frames per second.
    vmin, vmax:
        Shared colour-scale limits.
    """
    initial = -_start_subtraction(experiment.movie)
    without_corrections = create_mosaic_movie(experiment, corrections=False)
    with_corrections = create_mosaic_movie(experiment, corrections=True)

    fig, (start_ax, no_c_ax, with_c_ax) = plt.subplots(3, 1, dpi=300, figsize=(10, 10))

    start_im = start_ax.imshow(
        initial[0], animated=True, origin="lower", vmin=vmin, vmax=vmax
    )
    no_c_im = no_c_ax.imshow(
        without_corrections[0], animated=True, origin="lower", vmin=vmin, vmax=vmax
    )
    with_c_im = with_c_ax.imshow(
        with_corrections[0], animated=True, origin="lower", vmin=vmin, vmax=vmax
    )

    def init() -> tuple[matplotlib.artist.Artist, ...]:
        start_ax.set_title("Fixed background")
        no_c_ax.set_title("Dynamic background")
        with_c_ax.set_title("Dynamic background + corrections")
        return start_im, no_c_im, with_c_im

    def update(frame: int) -> tuple[matplotlib.artist.Artist, ...]:
        start_im.set_array(initial[frame])
        no_c_im.set_array(without_corrections[frame])
        with_c_im.set_array(with_corrections[frame])
        return start_im, no_c_im, with_c_im

    ani = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=tqdm(np.arange(len(initial)), desc="Making animation"),
        init_func=init,
        blit=True,
    )
    writer = mpl_animation.writers["ffmpeg"](bitrate=3000, fps=fps)
    ani.save(filename + ".mp4", writer=writer, dpi=300)
    plt.close(fig)


def background_cleanup_video(
    experiment: Any,  # noqa: ANN401
    filename: str | None = None,
    fps: float = 30,
    *,
    vmin: float = -0.01,
    vmax: float = 0.01,
    frame_limit: int | None = None,
    frame_start: int | None = None,
    dpi: int = 300,
    trajectory_color: str | None = None,
    cmap: str = "gray",
    cumulative: bool = True,
) -> None:
    """Two-panel animation showing original vs background-cleaned movie.

    Overlays particle trajectories that fade out after unbinding.

    Parameters
    ----------
    experiment:
        Experiment object with ``.movie``, ``.particles``, and
        ``.loader.mp_filepath``.
    filename:
        Output path without ``.mp4``.  Defaults to the experiment file name
        with " movie background cleanup" appended.
    fps:
        Frames per second.
    vmin, vmax:
        Colour-scale limits.
    frame_limit:
        Last frame to render.  Defaults to the full movie length.
    frame_start:
        First frame to render.  Defaults to ``0``.
    dpi:
        Figure resolution.
    trajectory_color:
        Fixed trajectory colour.  ``None`` uses matplotlib defaults.
    cmap:
        Matplotlib colourmap name.
    cumulative:
        Passed to :func:`create_raw_mosaic`.
    """
    if filename is None:
        filename = experiment.loader.mp_filepath.stem + " movie background cleanup"

    start = -_start_subtraction(experiment.movie, start_frame=5)
    zeros = np.zeros_like(start, dtype=float)
    with_corrections = create_raw_mosaic(
        experiment, new_movie=zeros, cumulative=cumulative
    )
    mask = with_corrections == 0
    with_corrections[mask] = start[mask]

    fig, (before_ax, after_ax) = plt.subplots(1, 2, dpi=dpi, figsize=(10, 5))
    before_ax.set_axis_off()
    after_ax.set_axis_off()

    trajectories = [p.trajectory for p in experiment.particles]
    unbinding_frames = [p.unbinding_frame for p in experiment.particles]

    def format_traj(
        t: Any,  # noqa: ANN401
        f: int,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        coords, frames_arr = t
        mask_f = frames_arr <= f
        return coords[0][mask_f], coords[1][mask_f]

    if trajectory_color is None:
        blines = [before_ax.plot(*format_traj(t, 0))[0] for t in trajectories]
        alines = [after_ax.plot(*format_traj(t, 0))[0] for t in trajectories]
    else:
        blines = [
            before_ax.plot(*format_traj(t, 0), color=trajectory_color)[0]
            for t in trajectories
        ]
        alines = [
            after_ax.plot(*format_traj(t, 0), color=trajectory_color)[0]
            for t in trajectories
        ]

    sim = before_ax.imshow(
        start[0], animated=True, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap
    )
    cim = after_ax.imshow(
        with_corrections[0],
        animated=True,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    def init() -> tuple[matplotlib.artist.Artist, ...]:
        before_ax.set_title("Original")
        after_ax.set_title("With background recovery")
        return cim, sim

    slope = 1000.0

    def update(frame: int) -> tuple[matplotlib.artist.Artist, ...]:
        unbound_for = np.array(
            [frame - u if u is not None else -1 for u in unbinding_frames], dtype=float
        )
        unbound_for[unbound_for < 0] = 0
        unbound_for[unbound_for > slope] = (1 - 0.05) * slope
        alpha = 1 - (unbound_for / slope)

        before_ax.set_title(f"Frame {frame}\n\nOriginal")
        sim.set_array(start[frame])
        cim.set_array(with_corrections[frame])
        for rline, sline, traj, a in zip(
            blines, alines, trajectories, alpha, strict=True
        ):
            new_t = format_traj(traj, frame)
            rline.set_xdata(new_t[0])
            rline.set_ydata(new_t[1])
            rline.set_alpha(float(a))
            sline.set_xdata(new_t[0])
            sline.set_ydata(new_t[1])
            sline.set_alpha(float(a))
        return cim, sim

    if frame_start is None:
        frame_start = 0
    if frame_limit is None:
        frame_limit = len(start)
    else:
        frame_limit = min(frame_limit, len(start))

    ani = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=tqdm(np.arange(frame_start, frame_limit), desc="Making animation"),
        init_func=init,
        blit=True,
    )
    writer = mpl_animation.writers["ffmpeg"](bitrate=3000, fps=int(fps))
    ani.save(filename + ".mp4", writer=writer)
    plt.close(fig)


def mosaic_comparison_neat_opencv(
    experiment: Any,  # noqa: ANN401
    filename: str,
    fps: float,
    *,
    vmin: float = -0.01,
    vmax: float = 0.01,
    frame_limit: int | None = None,
    frame_start: int | None = None,
) -> None:
    """Three-panel comparison animation rendered via OpenCV (no ffmpeg dependency).

    Uses ``cv2.VideoWriter`` rather than matplotlib's ffmpeg writer.

    Parameters
    ----------
    experiment:
        Experiment object with ``.loader.ratiometric``, ``.movie``,
        and ``.particles``.
    filename:
        Output path without the ``.mp4`` extension.
    fps:
        Frames per second.
    vmin, vmax:
        Shared colour-scale limits.
    frame_limit, frame_start:
        Frame range to render.
    """
    import cv2  # lazy import — optional dependency

    ratiometric = experiment.loader.ratiometric
    start = -_start_subtraction(experiment.movie, start_frame=5)
    with_corrections = create_raw_mosaic(experiment)
    trajectories = [p.trajectory for p in experiment.particles]
    unbinding_frames = [p.unbinding_frame for p in experiment.particles]

    def format_traj(t: Any, f: int) -> tuple[Any, Any]:  # noqa: ANN401
        coords, frames_arr = t
        mask_f = frames_arr <= f
        return coords[0][mask_f], coords[1][mask_f]

    fig, (rax, sax, cax) = plt.subplots(3, 1, dpi=300, figsize=(10, 10))

    slines = [sax.plot(*format_traj(t, 0))[0] for t in trajectories]
    rlines = [rax.plot(*format_traj(t, 0))[0] for t in trajectories]

    rax.set_axis_off()
    sax.set_axis_off()
    cax.set_axis_off()

    rim = rax.imshow(
        ratiometric[0], animated=True, origin="lower", vmin=vmin, vmax=vmax, cmap="gray"
    )
    sim = sax.imshow(
        start[0], animated=True, origin="lower", vmin=vmin, vmax=vmax, cmap="gray"
    )
    cim = cax.imshow(
        with_corrections[0],
        animated=True,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="gray",
    )

    def init() -> tuple[matplotlib.artist.Artist, ...]:
        rax.set_title("Ratiometric")
        sax.set_title("Fixed background")
        cax.set_title("Dynamic background and corrections")
        return rim, cim, sim

    slope = 1000.0

    def update(frame: int) -> tuple[matplotlib.artist.Artist, ...]:
        unbound_for = np.array(
            [frame - u if u is not None else -1 for u in unbinding_frames], dtype=float
        )
        unbound_for[unbound_for < 0] = 0
        unbound_for[unbound_for > slope] = (1 - 0.05) * slope
        alpha = 1 - (unbound_for / slope)

        rax.set_title(f"Frame {frame}\n\nRatiometric")
        rim.set_array(ratiometric[frame])
        sim.set_array(start[frame])
        cim.set_array(with_corrections[frame])
        for rline, sline, traj, a in zip(
            rlines, slines, trajectories, alpha, strict=True
        ):
            new_t = format_traj(traj, frame)
            rline.set_xdata(new_t[0])
            rline.set_ydata(new_t[1])
            rline.set_alpha(float(a))
            sline.set_xdata(new_t[0])
            sline.set_ydata(new_t[1])
            sline.set_alpha(float(a))
        return rim, cim, sim

    if frame_start is None:
        frame_start = 0
    if frame_limit is None:
        frame_limit = len(ratiometric)
    else:
        frame_limit = min(frame_limit, len(ratiometric))

    fig.canvas.draw()

    frame_data = np.asarray(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # type: ignore[attr-defined]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer_cv = cv2.VideoWriter(
        filename + ".mp4", fourcc, fps, (frame_data.shape[1], frame_data.shape[0])
    )

    for frame in tqdm(np.arange(frame_start, frame_limit), desc="Making animation"):
        update(frame)
        fig.canvas.draw()
        frame_data = np.asarray(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # type: ignore[attr-defined]
        writer_cv.write(frame_data)

    writer_cv.release()
    plt.close(fig)


def mosaic_comparison_neat(
    experiment: Any,  # noqa: ANN401
    filename: str,
    fps: float,
    *,
    vmin: float = -0.01,
    vmax: float = 0.01,
    frame_limit: int | None = None,
    frame_start: int | None = None,
) -> None:
    """Three-panel comparison animation (ratiometric, fixed bg, dynamic bg).

    Overlays particle trajectories that fade out after unbinding.

    Parameters
    ----------
    experiment:
        Experiment object with ``.loader.ratiometric``, ``.movie``,
        and ``.particles``.
    filename:
        Output path without the ``.mp4`` extension.
    fps:
        Frames per second.
    vmin, vmax:
        Shared colour-scale limits.
    frame_limit, frame_start:
        Frame range to render.
    """
    ratiometric = experiment.loader.ratiometric
    start = -_start_subtraction(experiment.movie, start_frame=5)
    with_corrections = create_raw_mosaic(experiment)

    trajectories = [p.trajectory for p in experiment.particles]
    unbinding_frames = [p.unbinding_frame for p in experiment.particles]

    def format_traj(t: Any, f: int) -> tuple[Any, Any]:  # noqa: ANN401
        coords, frames_arr = t
        mask_f = frames_arr <= f
        return coords[0][mask_f], coords[1][mask_f]

    fig, (rax, sax, cax) = plt.subplots(3, 1, dpi=300, figsize=(8, 8))

    slines = [sax.plot(*format_traj(t, 0))[0] for t in trajectories]
    rlines = [rax.plot(*format_traj(t, 0))[0] for t in trajectories]

    rax.set_axis_off()
    sax.set_axis_off()
    cax.set_axis_off()

    rim = rax.imshow(
        ratiometric[0], animated=True, origin="lower", vmin=vmin, vmax=vmax, cmap="gray"
    )
    sim = sax.imshow(
        start[0], animated=True, origin="lower", vmin=vmin, vmax=vmax, cmap="gray"
    )
    cim = cax.imshow(
        with_corrections[0],
        animated=True,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="gray",
    )

    def init() -> tuple[matplotlib.artist.Artist, ...]:
        rax.set_title("Ratiometric")
        sax.set_title("Fixed background")
        cax.set_title("Dynamic background and corrections")
        return (rim, cim, sim, *rlines, *slines)

    slope = 1000.0

    def update(frame: int) -> tuple[matplotlib.artist.Artist, ...]:
        unbound_for = np.array(
            [frame - u if u is not None else -1 for u in unbinding_frames], dtype=float
        )
        unbound_for[unbound_for < 0] = 0
        unbound_for[unbound_for > slope] = (1 - 0.05) * slope
        alpha = 1 - (unbound_for / slope)

        rax.set_title(f"Frame {frame}\n\nRatiometric")
        rim.set_array(ratiometric[frame])
        sim.set_array(start[frame])
        cim.set_array(with_corrections[frame])
        for rline, sline, traj, a in zip(
            rlines, slines, trajectories, alpha, strict=True
        ):
            new_t = format_traj(traj, frame)
            rline.set_xdata(new_t[0])
            rline.set_ydata(new_t[1])
            rline.set_alpha(float(a))
            sline.set_xdata(new_t[0])
            sline.set_ydata(new_t[1])
            sline.set_alpha(float(a))
        return (rim, cim, sim, *rlines, *slines)

    if frame_start is None:
        frame_start = 0
    if frame_limit is None:
        frame_limit = len(ratiometric)
    else:
        frame_limit = min(frame_limit, len(ratiometric))

    ani = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=tqdm(np.arange(frame_start, frame_limit), desc="Making animation"),
        init_func=init,
        blit=True,
    )
    writer = mpl_animation.writers["ffmpeg"](bitrate=4000, fps=int(fps))
    ani.save(filename + ".mp4", writer=writer)
    plt.close(fig)
