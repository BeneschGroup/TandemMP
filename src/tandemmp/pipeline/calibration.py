"""Interactive contrast-to-mass calibration.

Standalone from :mod:`tandemmp.pipeline.batch` so a calibration can be
generated (:func:`run_calibration`) and persisted (:func:`save_calibration`)
without running a full analysis. Event-finding goes directly through the
pluggable :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`
interface (open_iscat by default) — raw peak-finding contrasts, with no
mass filtering, since the interactive histogram step below is itself the
peak-selection filter.

The workflow is interactive: a histogram of calibration-movie contrasts is
shown, the user drag-selects each peak (fitted live via
:func:`~tandemmp.core.histogram.fit_gaussian_to_histogram`), then enters the
known mass for each peak in a small tkinter window, and a final straight-line
fit (:func:`~tandemmp.fitting.minimize.curve_fit_model`) of contrast vs. mass
produces the :class:`~tandemmp.core.conversions.Calibration`. Apply the result
process-wide via :func:`tandemmp.core.conversions.set_calibration`.

:func:`save_calibration` writes plain JSON so a calibration file can be
opened and hand-edited in a text editor; the field names (``c2kDa``,
``intercept``, ``raw_error``, ``calibrant_masses``, ``fitted_contrast_peaks``)
mirror the pre-refactor ``alex_area.calibration`` pickled-dict format.
:func:`load_calibration` reads that new JSON format and also transparently
falls back to unpickling an old legacy pickled dict, so old calibration
files keep working.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import tkinter as tk
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tkinter import ttk
from typing import Any

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Rectangle

from tandemmp.core.conversions import Calibration
from tandemmp.core.histogram import fit_gaussian_to_histogram
from tandemmp.fitting.gaussian import gaussian
from tandemmp.fitting.minimize import curve_fit_model
from tandemmp.pipeline.open_iscat_pipeline import OpenIscatPipeline
from tandemmp.pipeline.pipeline_interface import PipelineRunner

# Result container


@dataclasses.dataclass(frozen=True)
class CalibrationResult:
    """A fitted :class:`Calibration` plus the data/fit that produced it."""

    calibration: Calibration
    source_filepath: str
    calibrant_masses: list[float]
    fitted_contrast_peaks: list[float]
    residual_sum_of_squares: float


# Matplotlib drag-select-and-fit widget


class _ClickAndDrag:
    """Click-and-drag horizontal region selection on a matplotlib axes."""

    def __init__(self, ax: matplotlib.axes.Axes) -> None:
        self.ax = ax
        self.rect: Rectangle | None = None
        self.selection_start: float | None = None
        self.selection_end: float | None = None
        self.cid_press = ax.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cid_motion = ax.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

    def on_press(self, event: Any) -> None:  # noqa: ANN401
        if event.inaxes != self.ax:
            return
        self.selection_start = event.xdata
        self.rect = Rectangle((self.selection_start, 0), 0, 1, alpha=0.5, color="gray")
        self.ax.add_patch(self.rect)

    def on_motion(self, event: Any) -> None:  # noqa: ANN401
        if self.selection_start is None or event.inaxes != self.ax or self.rect is None:
            return
        self.selection_end = event.xdata
        self.rect.set_width(self.selection_end - self.selection_start)
        self.rect.set_height(self.ax.get_ylim()[1])
        self.rect.set_xy((self.selection_start, self.ax.get_ylim()[0]))
        self.ax.figure.canvas.draw()


class _DragAndFitPeaks(_ClickAndDrag):
    """Drag-select a histogram region; fit a Gaussian to the data in range.

    Fits the underlying data filtered to the selected range.
    """

    def __init__(
        self,
        ax: matplotlib.axes.Axes,
        data: npt.NDArray[np.floating],
        bins: npt.NDArray[np.floating],
    ) -> None:
        super().__init__(ax)
        self.cid_release = ax.figure.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        ax.set_ylim(ax.get_ylim()[0], 1.1 * ax.get_ylim()[1])
        self.data = data
        self.bins = bins
        self.peak_means: list[float] = []

    def on_release(self, event: Any) -> None:  # noqa: ANN401
        if self.selection_start is None or self.selection_end is None:
            return
        lower = min(self.selection_start, self.selection_end)
        upper = max(self.selection_start, self.selection_end)

        if self.rect is not None:
            self.rect.remove()
            self.rect = None
        self.selection_start = None
        self.selection_end = None

        in_range = self.data[(self.data >= lower) & (self.data <= upper)]
        bins_in_range = self.bins[(self.bins >= lower) & (self.bins <= upper)]
        if in_range.size < 3 or bins_in_range.size < 3:
            self.ax.figure.canvas.draw()
            return

        amplitude, mean, sigma = fit_gaussian_to_histogram(in_range, bins=bins_in_range)
        self.peak_means.append(mean)

        x_space = np.linspace(mean - 4 * abs(sigma), mean + 4 * abs(sigma), 500)
        self.ax.plot(x_space, gaussian(x_space, amplitude, mean, sigma), color="black")
        self.ax.annotate(
            f"{mean:.6f}\n$\\sigma$ = {abs(sigma):.6f}",
            xy=(mean, amplitude * 1.1),
            horizontalalignment="center",
            size=7,
        )
        self.ax.figure.canvas.draw()


def _hist_bins(
    data: npt.NDArray[np.floating], binwidth: float
) -> npt.NDArray[np.float64]:
    """Histogram bin edges symmetric around 0."""
    if np.nanmin(data) < 0:
        pos_bins = np.arange(0, np.nanmax(data) + binwidth, step=binwidth)
        neg_bins = (
            -1
            * np.arange(binwidth, np.abs(np.nanmin(data) - binwidth), step=binwidth)[
                ::-1
            ]
        )
        return np.concatenate((neg_bins, pos_bins))
    return np.arange(0, np.nanmax(data) + binwidth, step=binwidth)


# tkinter windows


class _BaseTkWindow:
    def __init__(
        self,
        title: str,
        position: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        pos = f"+{position[0]}+{position[1]}" if position is not None else ""
        sz = f"{size[0]}x{size[1]}" if size is not None else ""
        self.root.geometry(sz + pos)


class _YesNoWindow(_BaseTkWindow):
    def __init__(
        self,
        title: str,
        message: str,
        yes_action: Callable[[], None],
        no_action: Callable[[], None],
    ) -> None:
        super().__init__(title=title, position=(500, 500))
        tk.Label(self.root, text=message).pack()

        def _yes() -> None:
            yes_action()
            self.root.destroy()

        def _no() -> None:
            no_action()
            self.root.destroy()

        tk.Button(self.root, text="Yes", command=_yes).pack(side=tk.LEFT, padx=10)
        tk.Button(self.root, text="No", command=_no).pack(side=tk.RIGHT, padx=10)
        self.root.mainloop()


class _NumericEntry(ttk.Entry):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self.var = tk.StringVar()
        super().__init__(*args, textvariable=self.var, **kwargs)
        self.var.trace_add("write", self._validate)

    def _validate(self, *_args: Any) -> None:  # noqa: ANN401
        value = self.var.get()
        if not value:
            return
        try:
            float(value)
        except ValueError:
            self.var.set("".join(c for c in value if c.isdigit() or c == "."))

    def value(self) -> float:
        return float(self.var.get())


class _MassEntryWindow(_BaseTkWindow):
    """Collects a known calibrant mass for each fitted contrast peak."""

    def __init__(self, contrast_peaks: Sequence[float]) -> None:
        self.peak_vals = [round(p, 5) for p in contrast_peaks]
        super().__init__(
            title="Calibration",
            position=(500, 500),
            size=(250, len(self.peak_vals) * 33 + 70),
        )
        self.entries: list[_NumericEntry] = []
        self.calibrant_masses: list[float] = []
        self._build()
        self.root.mainloop()

    def _build(self) -> None:
        ttk.Label(self.root, text="Contrast", font=("Arial", 11, "bold")).grid(
            row=0, column=0, padx=5, pady=5
        )
        ttk.Label(self.root, text="Calibrant Mass", font=("Arial", 11, "bold")).grid(
            row=0, column=1, padx=5, pady=5
        )
        for idx, value in enumerate(self.peak_vals, start=1):
            ttk.Label(self.root, text=str(value)).grid(
                row=idx, column=0, padx=5, pady=5
            )
            entry = _NumericEntry(self.root)
            entry.grid(row=idx, column=1, padx=5, pady=5)
            self.entries.append(entry)
        ttk.Button(self.root, text="Calibrate", command=self._finish).grid(
            row=len(self.peak_vals) + 1, column=0, columnspan=2, pady=10
        )

    def _finish(self) -> None:
        self.calibrant_masses = [entry.value() for entry in self.entries]
        self.root.destroy()


# Orchestration


@dataclasses.dataclass
class _LoopFlags:
    """Mutable loop-continuation state shared by tkinter callbacks."""

    reattempt_fit: bool = True
    repeat_calibration: bool = True


def _mass_to_contrast_line(
    mass: npt.NDArray[np.floating], slope: float, offset: float
) -> npt.NDArray[np.floating]:
    return slope * mass + offset


def calibrate(
    contrasts: npt.NDArray[np.floating],
    source_filepath: str | Path,
    *,
    lower_c_bound: float = -0.05,
    higher_c_bound: float = 0.0,
) -> CalibrationResult:
    """Run the interactive peak-selection / mass-entry calibration workflow.

    *contrasts* is filtered to ``(lower_c_bound, higher_c_bound)`` before
    display. Blocks on matplotlib/tkinter windows until the user finishes.
    """
    contrasts = np.asarray(contrasts, dtype=float)
    sel = (contrasts > lower_c_bound) & (contrasts < higher_c_bound)
    filtered = contrasts[sel]
    title = Path(source_filepath).stem

    flags = _LoopFlags()
    peaks: npt.NDArray[np.floating] = np.array([])
    calibrant_masses: list[float] = []
    slope = 0.0
    offset = 0.0
    residual = 0.0

    while flags.repeat_calibration:
        flags.reattempt_fit = True
        drag: _DragAndFitPeaks | None = None

        while flags.reattempt_fit:
            _, ax = plt.subplots(dpi=150)
            bins = _hist_bins(filtered, binwidth=0.00020)
            ax.hist(filtered, bins=bins.tolist())
            ax.set_xlim(lower_c_bound, higher_c_bound)
            ax.invert_xaxis()
            ax.set_xlabel("Contrast")
            ax.set_ylabel("Counts")
            ax.set_title(title)

            drag = _DragAndFitPeaks(ax=ax, data=filtered, bins=bins)
            plt.show()

            def _no_reattempt() -> None:
                flags.reattempt_fit = False

            _YesNoWindow(
                title="Repeat?",
                message="Reattempt fit?",
                yes_action=lambda: None,
                no_action=_no_reattempt,
            )

        assert drag is not None
        peaks = np.sort(np.array(drag.peak_means))[::-1]

        mass_window = _MassEntryWindow(contrast_peaks=peaks.tolist())
        calibrant_masses = mass_window.calibrant_masses

        fit = curve_fit_model(
            _mass_to_contrast_line,
            np.array(calibrant_masses, dtype=float),
            peaks,
            p0=[-0.00003, 0.0],
        )
        slope, offset = (float(p) for p in fit.params)
        residual = fit.residual_sum_of_squares

        _, ax = plt.subplots(dpi=150)
        ax.scatter(calibrant_masses, peaks)
        x_line = np.array(ax.get_xlim())
        ax.plot(x_line, _mass_to_contrast_line(x_line, slope, offset), color="black")
        ax.set_xlabel("Mass (kDa)")
        ax.set_ylabel("Contrast")
        error_pct = abs(residual / slope) * 100 if slope else float("nan")
        ax.set_title(
            f"{title}\nSlope: {slope:.8f}, Offset: {offset:.8f}, "
            f"Error: {error_pct:.2f}%"
        )
        plt.show()

        def _no_repeat() -> None:
            flags.repeat_calibration = False

        _YesNoWindow(
            title="Repeat?",
            message="Repeat calibration?",
            yes_action=lambda: None,
            no_action=_no_repeat,
        )

    return CalibrationResult(
        calibration=Calibration(slope=slope, offset=offset),
        source_filepath=str(source_filepath),
        calibrant_masses=[float(m) for m in calibrant_masses],
        fitted_contrast_peaks=[float(p) for p in peaks],
        residual_sum_of_squares=float(residual),
    )


def run_calibration(
    filepath: str | Path,
    *,
    pipeline_runner: PipelineRunner | None = None,
    parallel: bool = True,
    lower_c_bound: float = -0.05,
    higher_c_bound: float = 0.0,
    **pipeline_kwargs: Any,  # noqa: ANN401
) -> CalibrationResult:
    """Find events in a calibration movie, then run the interactive fit.

    Event-finding goes through *pipeline_runner* (open_iscat by default; see
    :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`) — raw
    peak-finding contrasts, unfiltered by mass, since the interactive
    histogram/drag-select step is itself the peak-selection filter.
    **pipeline_kwargs* is forwarded to the runner (e.g. ``navg``, ``l_thum``
    for the open_iscat backend).
    """
    pipeline = pipeline_runner if pipeline_runner is not None else OpenIscatPipeline()

    pipeline_results = pipeline.run(filepath, parallel=parallel, **pipeline_kwargs)

    events = pipeline_results.events
    contrasts = np.array(events.get_C(), dtype=float)

    return calibrate(
        contrasts,
        filepath,
        lower_c_bound=lower_c_bound,
        higher_c_bound=higher_c_bound,
    )


# Persistence
#
# On disk, a calibration is plain JSON so it can be opened and hand-edited in
# a text editor. Field names mirror the pre-refactor
# ``alex_area.calibration`` pickled-dict format (``c2kDa``, ``intercept``,
# ``raw_error``, ``calibrant_masses``, ``fitted_contrast_peaks``), plus
# ``source_filepath`` for provenance. load_calibration() also accepts that
# legacy pickled dict directly, since the field names line up.


def _calibration_to_mapping(result: CalibrationResult) -> dict[str, Any]:
    return {
        "c2kDa": result.calibration.slope,
        "intercept": result.calibration.offset,
        "raw_error": result.residual_sum_of_squares,
        "calibrant_masses": result.calibrant_masses,
        "fitted_contrast_peaks": result.fitted_contrast_peaks,
        "source_filepath": result.source_filepath,
    }


def _calibration_from_mapping(
    payload: Mapping[str, Any], *, default_source_filepath: str
) -> CalibrationResult:
    try:
        slope = float(payload["c2kDa"])
        offset = float(payload["intercept"])
    except KeyError as exc:
        msg = f"calibration data is missing required field {exc}"
        raise ValueError(msg) from exc
    return CalibrationResult(
        calibration=Calibration(slope=slope, offset=offset),
        source_filepath=str(payload.get("source_filepath", default_source_filepath)),
        calibrant_masses=[float(m) for m in payload.get("calibrant_masses", [])],
        fitted_contrast_peaks=[
            float(p) for p in payload.get("fitted_contrast_peaks", [])
        ],
        residual_sum_of_squares=float(payload.get("raw_error", 0.0)),
    )


def save_calibration(filepath: str | Path, result: CalibrationResult) -> Path:
    """Write *result* to *filepath* as human-editable JSON.

    Creates parent directories as needed. See the module docstring for the
    on-disk field names.
    """
    if not isinstance(result, CalibrationResult):
        msg = f"result must be a CalibrationResult, got {type(result).__name__}"
        raise TypeError(msg)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_calibration_to_mapping(result), fh, indent=2)
        fh.write("\n")
    return path


def load_calibration(filepath: str | Path) -> CalibrationResult:
    """Load a :class:`CalibrationResult` from *filepath*.

    Reads the JSON format written by :func:`save_calibration`. Also accepts
    a legacy pickled dict (the pre-refactor ``alex_area.calibration`` save
    format) for backward compatibility.
    """
    path = Path(filepath)
    raw = path.read_bytes()
    payload: Any
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = pickle.loads(raw)  # noqa: S301 — legacy pickled-dict format

    if not isinstance(payload, Mapping):
        msg = f"{path} does not contain a calibration"
        raise TypeError(msg)
    return _calibration_from_mapping(payload, default_source_filepath=str(path))


if __name__ == "__main__":
    from tandemmp.core.functions import choose_file

    movie_filepath = choose_file()
    if not isinstance(movie_filepath, str):
        raise TypeError("Select a single calibration movie file.")
    calibration_result = run_calibration(movie_filepath)

    calibration_path = Path(movie_filepath).with_name(
        f"{Path(movie_filepath).stem}_calibration.json"
    )
    save_calibration(calibration_path, calibration_result)
    print(f"Calibration saved to {calibration_path}")
