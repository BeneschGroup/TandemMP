"""Subtraction event and series — produced by frame-subtraction processing."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from tandemmp.core.classifications import (
    Classification,
    EvClassification,
    ProcessingMethod,
)
from tandemmp.core.conversions import c_to_mass
from tandemmp.events._base import AbstractEvent, Connectable, PeakImage
from tandemmp.fitting.psf import (
    error_metric,
    native_impression,
)
from tandemmp.fitting.psf_interface import PsfInterface

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Ghost-prediction model (lazy TensorFlow import)


@functools.lru_cache(maxsize=1)
def get_ghost_model() -> Any:  # noqa: ANN401
    """Load and cache the ghost-prediction CNN (``GP_model.h5``).

    TensorFlow is imported inside this function so the package imports without it;
    the model is loaded once per process.
    """
    import importlib.resources
    import os

    # Hide TF/tf_keras start-up and deprecation noise unless already configured.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf_logger = logging.getLogger("tensorflow")
    if tf_logger.level == logging.NOTSET:
        tf_logger.setLevel(logging.ERROR)
    import tensorflow as tf

    ref = importlib.resources.files("tandemmp.resources").joinpath("GP_model.h5")
    with importlib.resources.as_file(ref) as model_path:
        return tf.keras.models.load_model(str(model_path))


# SubtractionEvent


class SubtractionEvent(AbstractEvent):
    """Event created by dividing / subtracting a reference frame.

    Has an associated bleed time and subtracted frame index.
    """

    def __init__(
        self,
        peak: PeakImage,
        subtracted_i: int,
        *,
        fit: bool = True,
        exclusion: npt.NDArray[np.bool_] | None = None,
        peak_model: PsfInterface | None = None,
    ) -> None:
        self.exclusion = exclusion

        if peak.method != ProcessingMethod.subtraction:
            msg = f"Expected ProcessingMethod.subtraction, got {peak.method!r}"
            raise ValueError(msg)

        super().__init__(peak, fit=False, peak_model=peak_model)

        if fit:
            if exclusion is not None and float(np.mean(exclusion)) < 0.5:
                self.basic_fit_exclusion(exclusion)
                self.exc_fit: bool = True
            else:
                self.basic_fit()
                self.exc_fit = False

            # No 3-D fit here: dx/dy are plain floats (AbstractEvent's type is wider).
            self.interp_dx: float | None = float(self.dx)
            self.interp_dy: float | None = float(self.dy)
            self.test_bad()
        else:
            self._is_bad: bool = False
            self.interp_dx = None
            self.interp_dy = None

        self.particle_i: int = self.i
        self.mean_c: float = float(np.mean(self.peak.image))
        self.mean_m: float = float(c_to_mass(self.mean_c))
        self.subtracted_i: int = subtracted_i
        self.bleed_time: int = self.i - subtracted_i

    def reduce_memory(self) -> None:
        """Down-cast the peak image to float16 in-place."""
        self.peak = self.peak.as_reduced_memory()

    def basic_fit(
        self,
        method: str = "BFGS",
        *,
        allow_offset: bool = False,
        gtol: float = 1e-6,
    ) -> None:
        """Perform the basic 2-D PSF fit and update bad-fit flag."""
        self._underlying_basic_fit(method, allow_offset=allow_offset, gtol=gtol)
        self.test_bad()

    def test_bad(self) -> None:
        """Flag the event if signatures of a bad fit are detected."""
        negative_mass = self.c >= 0
        low_snr = self.snr <= 0.1
        bounds = (self.peak.image.shape[0] // 2) + 4
        # No 3-D fit here: dx/dy are plain floats despite AbstractEvent's wider type.
        out_of_bounds = (abs(float(self.dx)) > bounds) or (abs(float(self.dy)) > bounds)
        self._is_bad = negative_mass or low_snr or out_of_bounds

    @property
    def particle_m(self) -> float:
        """Mass value for particle tracking."""
        return self.m

    def basic_fit_exclusion(
        self,
        exclusion: npt.NDArray[np.bool_],
        method: str = "BFGS",
    ) -> None:
        """Fit with spatial exclusion mask and update bad-fit flag."""
        fit_result = self.peak_model.fit_peak_exclusion(
            self.peak.image, exclusion, method=method
        )
        self.dx = fit_result.dx
        self.dy = fit_result.dy
        self.x = self.peak.midpoint[0] + fit_result.dx
        self.y = self.peak.midpoint[1] + fit_result.dy
        self.c = fit_result.contrast
        self.m = float(c_to_mass(self.c))
        self.rel_m = [self.m]
        self.rel_c = [self.c]
        self.basic_fit_error = fit_result.error
        self._basic_fit = fit_result
        self.test_bad()


# SubtractionSeries


class SubtractionSeries(Connectable):
    """Ordered collection of :class:`SubtractionEvent` instances.

    Aggregates per-event attributes into arrays and provides smoothed position
    interpolation, ghost-signal scoring and impression generation.

    The ``_AGGREGATE_ATTRS`` attributes are injected in :meth:`__init__` (one element
    per event); they are declared here so mypy sees them and callers need no
    ``attr-defined`` suppression.
    """

    dx: npt.NDArray[np.floating]
    dy: npt.NDArray[np.floating]
    x: npt.NDArray[np.floating]
    y: npt.NDArray[np.floating]
    c: npt.NDArray[np.floating]
    m: npt.NDArray[np.floating]
    mean_c: npt.NDArray[np.floating]
    mean_m: npt.NDArray[np.floating]
    basic_fit_error: npt.NDArray[np.floating]
    _is_bad: npt.NDArray[np.bool_]
    exc_fit: npt.NDArray[np.bool_]
    interp_dx: npt.NDArray[np.floating]
    interp_dy: npt.NDArray[np.floating]
    particle_i: npt.NDArray[np.integer]
    subtracted_i: npt.NDArray[np.integer]
    bleed_time: npt.NDArray[np.integer]
    navg: npt.NDArray[np.integer]

    _AGGREGATE_ATTRS: tuple[str, ...] = (
        "dx",
        "dy",
        "x",
        "y",
        "c",
        "m",
        "mean_c",
        "mean_m",
        "basic_fit_error",
        "_is_bad",
        "exc_fit",
        "interp_dx",
        "interp_dy",
        "particle_i",
        "subtracted_i",
        "bleed_time",
        "navg",
    )

    def __init__(self, events: list[SubtractionEvent]) -> None:
        if len(events) == 0:
            msg = "SubtractionSeries requires at least one event"
            raise ValueError(msg)

        first = events[0]
        for attr in self._AGGREGATE_ATTRS:
            if attr not in first.__dict__:
                continue
            try:
                self.__dict__[attr] = np.asarray(
                    [e.__dict__[attr] for e in events],
                )
            except (KeyError, ValueError):
                pass

        self.events: npt.NDArray[np.object_] = np.asarray(events)
        self._frame_map: dict[int, SubtractionEvent] = {ev.i: ev for ev in self.events}
        self.original: SubtractionSeries | None = None
        self._impression_cache: dict[
            tuple[Classification, PsfInterface | None],
            npt.NDArray[np.floating],
        ] = {}

        super().__init__()

        # Mark interpolated positions from the smoothed trajectory
        if hasattr(first, "_basic_fit"):
            for ev, sx, sy in zip(
                self.events, self.smooth_dx, self.smooth_dy, strict=True
            ):
                ev.interp_dx = sx
                ev.interp_dy = sy

    # Sequence protocol

    def __iter__(self) -> Any:  # noqa: ANN401
        return self.events.__iter__()

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: Any) -> Any:  # noqa: ANN401
        return self.events[index]

    # Lookup

    def get_by_frame(self, frame: int) -> SubtractionEvent | None:
        """Return the event at *frame*, or ``None``."""
        return self._frame_map.get(frame)

    # Memory management

    def reduce_memory(self) -> None:
        """Down-cast all event images to float16."""
        for ev in self:
            ev.reduce_memory()

    # Impression / fitting

    def impression(
        self,
        classification: Classification = EvClassification.Complex,
        peak_model: PsfInterface | None = None,
    ) -> npt.NDArray[np.floating]:
        """Generate a PSF impression for the series trajectory."""
        key = (classification, peak_model)
        if key not in self._impression_cache:
            l_thum = self.events[0].peak.image.shape[0]
            self._impression_cache[key] = native_impression(
                self.smooth_dx,  # type: ignore[arg-type]
                self.smooth_dy,  # type: ignore[arg-type]
                self.c,  # type: ignore[arg-type]
                classification,
                l_thum,
                psf_model=peak_model,
            )
        return self._impression_cache[key]

    def fit_to(
        self,
        peak: npt.NDArray[np.floating],
        classification: Classification,
        peak_model: PsfInterface,
    ) -> OptimizeResult:
        """Return the fit error of the series impression against *peak*."""
        return OptimizeResult(
            fun=error_metric(peak, self.impression(classification, peak_model)),
            x=self.c,
        )

    # Aggregated flags

    @property
    def is_bad(self) -> npt.NDArray[np.bool_]:
        """Per-event mask marking events flagged as unreliable."""
        return self._is_bad

    # Smoothed positions

    @property
    def smooth_dx(self) -> npt.NDArray[np.floating]:
        """Interpolated dx, skipping bad events."""
        navg = len(self.events)
        _is_bad = self._is_bad
        dx = self.dx
        # All events after the first are bad: too few points to interpolate.
        if _is_bad[-navg + 1 :].all():
            return dx
        i = np.arange(len(self))
        return np.interp(i, i[~_is_bad], dx[~_is_bad])  # type: ignore[no-any-return]

    @property
    def smooth_dy(self) -> npt.NDArray[np.floating]:
        """Interpolated dy, skipping bad events."""
        navg = len(self.events)
        _is_bad = self._is_bad
        dy = self.dy
        # All events after the first are bad: too few points to interpolate.
        if _is_bad[-navg + 1 :].all():
            return dy
        i = np.arange(len(self))
        return np.interp(i, i[~_is_bad], dy[~_is_bad])  # type: ignore[no-any-return]

    def unreliability(
        self,
        classification: Classification = EvClassification.Complex,
    ) -> float:
        """Fraction of bad events in the classification-relevant window."""
        navg = len(self) // 2
        _is_bad = self._is_bad
        return float(np.mean(_is_bad[classification.get_slice(navg, remove=0)]))

    @property
    def avg_residual_noise(self) -> float:
        """Mean residual std across the latter half of the series."""
        navg = len(self) // 2
        return float(
            np.mean([np.std(i._basic_replicate_fit()[-1]) for i in self[-navg:]])
        )

    # Ghost signals

    @property
    def ghost_signal_ML(self) -> float | None:  # noqa: N802
        """CNN ghost-prediction score in [0, 1], or ``None`` on failure.

        TensorFlow is lazy-imported via :func:`get_ghost_model`.
        """
        try:
            images = np.array([i.peak.image for i in self])
            width = images.shape[1]

            # Selective cropping for larger thumbnails
            if width > 17:
                flat = np.sum(np.abs(images), axis=0)
                xs, ys = np.where(flat > np.nanpercentile(flat, 80))
                vals = flat[ys, xs]

                sel = ~np.isinf(vals) & ~np.isnan(vals)
                y = round(float(np.average(ys[sel], weights=vals[sel])))
                x = round(float(np.average(xs[sel], weights=vals[sel])))

                y_bounds = [y - 8, y + 9]
                x_bounds = [x - 8, x + 9]

                if x_bounds[0] < 0:
                    x_bounds = [0, 17]
                if x_bounds[1] > width:
                    x_bounds = [width - 17, width]
                if y_bounds[0] < 0:
                    y_bounds = [0, 17]
                if y_bounds[1] > width:
                    y_bounds = [width - 17, width]

                new_slice = (
                    slice(*y_bounds),
                    slice(*x_bounds),
                )
                images = np.array([i.peak.image[new_slice] for i in self])

            images = images / np.abs(images).max()
            model = get_ghost_model()

            images_input = images.reshape((*images.shape, 1)).tolist()

            if hasattr(model, "predict"):
                return float(
                    np.round(model.predict([images_input], verbose=0)[0][0], 2)
                )
            return float(np.round(model([images_input]).numpy()[0][0], 2))
        except Exception:
            logger.debug("Ghost ML scoring failed", exc_info=True)
            return None

    def ghost_signal(
        self,
        min_mass: float,
        exclusion_zone: npt.NDArray[np.bool_] | bool = False,
    ) -> float:
        """Heuristic ghost signal from averaged residual images, in [0, 1].

        Higher means more likely a ghost.
        """
        masked = np.ma.array(  # type: ignore[no-untyped-call]
            [i._basic_replicate_fit()[-1] for i in self],
            mask=exclusion_zone,
        )
        mean_im = masked.mean(axis=0)
        most_pos = abs(float(c_to_mass(np.max(mean_im))))
        signal = 1.0 / (np.exp(2 - (most_pos / min_mass)) + 1)
        return float(signal)

    # Serialisation

    def append_original(self, original_series: SubtractionSeries) -> None:
        """Attach the un-corrected original series."""
        self.original = original_series
