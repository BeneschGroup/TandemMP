"""Wrapper around the open_iscat pipeline for TandemMP.

The pipeline functions (:func:`load_mp_file`, :class:`OpenIscatPipeline`)
require ``open_iscat``, ``h5py``, and ``nptdms`` to be installed.  The data
containers (:class:`Info`, :class:`Movie`) have no such dependency and can be
used standalone.

:class:`OpenIscatPipeline` implements the
:class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner` structural
interface — see that module for the generic event-finding pipeline contract
and rationale. :class:`EventsFound` and :class:`PipelineResults` are
re-exported from there for backward compatibility.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tandemmp.pipeline.pipeline_interface import EventsFound, PipelineResults

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "EventsFound",
    "Info",
    "Movie",
    "OpenIscatPipeline",
    "PipelineResults",
    "load_mp_file",
]

logger = logging.getLogger(__name__)


# Data containers


@dataclass(frozen=True)
class Info:
    """Microscope acquisition metadata."""

    scatter_interference_destructive: bool | None
    raw_camera_frame_rate: float
    frame_binning: int
    pixel_binning: int
    full_well_capacity: float | None
    grey_levels: int | None


@dataclass(frozen=True)
class Movie:
    """Movie frames with acquisition metadata.

    Attribute access falls through to the underlying *frames* array,
    so ``movie.shape`` is equivalent to ``movie.frames.shape``.
    """

    frames: npt.NDArray[np.floating[Any]]
    info: Info

    def __getattr__(self, attr: str) -> Any:  # noqa: ANN401
        return getattr(self.frames, attr)


# File loading


def _read_mp_metadata(
    filepath: str,
) -> tuple[bool | None, float, int, int, float | None, int | None]:
    """Extract metadata from an ``.mp`` (HDF5) file.

    Returns ``(scatter_interference_destructive, raw_camera_frame_rate,
    frame_binning, pixel_binning, full_well_capacity, grey_levels)``.
    """
    import h5py
    from open_iscat.preprocess_read.FrameReader import FrameReader

    frame_reader = FrameReader(filename=filepath, frame_range=None)

    with h5py.File(filepath, "r") as f:
        version = int(f["format_version_number"][()])
        if version >= 3 and frame_reader.mp_version_number == version:
            cfg = f["movie"]["configuration"]
            return (
                bool(cfg["scatter_interference_destructive"][()]),
                float(cfg["acq_camera"]["frame_rate"][()]),
                int(cfg["acq_camera"]["frame_binning"][()]),
                int(cfg["acq_camera"]["pixel_binning"][()]),
                None,
                None,
            )
        if version >= 3:
            dev = f["movie"]["configuration"]["Devices"]["AcqCam"]
            return (
                bool(dev["ScatterInterferenceDestructive"][()]),
                float(dev["FrameRate"][()]),
                int(dev["SoftwareFrameBinning"][()]),
                int(dev["SoftwarePixelBinning"][()]),
                None,
                None,
            )
        eng = f["configuration"]["Engines"]["AcqMovieEngine"]
        return (
            None,
            1320.0,
            int(eng["FrameBinning"][()]),
            int(eng["PixelBinning"][()]),
            None,
            None,
        )


def _read_tdms_metadata(
    filepath: str,
) -> tuple[None, float, int, int, None, None]:
    """Extract metadata from a ``.tdms`` file."""
    import nptdms

    with nptdms.TdmsFile(filepath) as f:
        attrs = f.object("img", "cam1").properties
        return (
            None,
            float(attrs["Effective frame rate"]),
            int(attrs["Bin frame size"]),
            int(attrs["Pixel Binning"]) ** 2,
            None,
            None,
        )


def load_mp_file(filepath: str | pathlib.Path) -> Movie:
    """Load a movie from an ``.mp`` or ``.tdms`` file.

    Requires ``open_iscat``, ``h5py``, and/or ``nptdms``.
    """
    from open_iscat.preprocess_read.FrameReader import FrameReader

    filepath_str = str(filepath)
    logger.info("Loading file: %s", filepath_str)

    frame_reader = FrameReader(filename=filepath_str, frame_range=None)
    frames = frame_reader.work()

    if filepath_str.endswith(".mp"):
        meta = _read_mp_metadata(filepath_str)
    elif filepath_str.endswith(".tdms"):
        meta = _read_tdms_metadata(filepath_str)
    else:
        msg = f"Unsupported file extension: {filepath_str}"
        raise ValueError(msg)

    return Movie(
        frames=frames,
        info=Info(*meta),
    )


# Pipeline execution


class OpenIscatPipeline:
    """Event-finding pipeline backed by the open_iscat preprocessing and
    peak-finding stack (``FrameReader`` -> ``Preprocessor`` ->
    ``MaskFilterPeakFinder``/``Analyser``).

    Implements the :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`
    structural interface. Stateless, so instances are trivially picklable
    (needed when a :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`
    crosses a :class:`multiprocessing.Pool` boundary, e.g. via
    :func:`tandemmp.pipeline.batch.batch_process`).
    """

    def run(
        self,
        filepath: str | pathlib.Path,
        *,
        frame_bypass: npt.NDArray[np.floating[Any]] | None = None,
        parallel: bool = True,
        navg: int = 5,
        **kwargs: Any,  # noqa: ANN401
    ) -> PipelineResults:
        """Run the open_iscat preprocessing and peak-finding pipeline.

        Requires ``open_iscat``.

        Args:
            filepath: Path to the movie file.
            frame_bypass: If provided, splice these (e.g. drift-corrected)
                frames into the pipeline instead of reading from disk.
            parallel: Use parallel processing where available.
            navg: Number of averaged frames for ratiometric event detection.
            **kwargs: Ignored — absorbs keywords meant for a different
                :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`
                implementation when called generically through the interface.

        Returns:
            :class:`~tandemmp.pipeline.pipeline_interface.PipelineResults`
        """
        from open_iscat.analyse_find.maskfilter import MaskFilterPeakFinder
        from open_iscat.analyser import Analyser
        from open_iscat.preprocess_correct.adutoe import AduToE
        from open_iscat.preprocess_precompute.iprecompute import IPrecomputer
        from open_iscat.preprocess_read.FrameReader import FrameReader
        from open_iscat.preprocessor import Preprocessor

        filepath_str = str(filepath)
        logger.info("Loading file: %s", filepath_str)

        reader_kwargs: dict[str, Any] = {
            "filename": filepath_str,
            "frame_range": None,
            "do_check_saturation": False,
        }
        if frame_bypass is not None:
            logger.info("Splicing in drift-corrected movie")
            reader_kwargs["frames"] = frame_bypass

        frame_reader = FrameReader(**reader_kwargs)

        adutoe_corrector = AduToE(
            binning_time=frame_reader.binning_time,
            binning_pixel=frame_reader.binning_pixel,
        )

        rframe_precomputers = [
            IPrecomputer(
                navg=navg,
                medfilt_kernel=15,
                medfilt_bin=2,
                reflectivity_correction=False,
            ),
        ]

        preprocessor = Preprocessor(
            frame_reader, [adutoe_corrector], rframe_precomputers
        )
        frames, rframes = preprocessor.preprocess(parallel=parallel)

        peak_finder = MaskFilterPeakFinder(
            navg=navg,
            sigma_iimg=1.5,
            filter1="ttest",
            filter1_scale=0.9,
            filter1_sigma_filter=1.5,
            filter1_navg=navg,
            filter1_medfilt_kernel=15,
            filter1_medfilt_bin=2,
            filter2="frsym",
            filter2_scale=0.35,
            filter2_sigma_filter=0.5,
            filter2_navg=navg,
        )

        analyser = Analyser(peak_finder, peak_fitter=None, preprocessor=preprocessor)
        events_found = analyser.analyse_events_only(frames, rframes, parallel=parallel)

        return PipelineResults(frames, rframes[navg], events_found)
