"""Wrapper around the open_iscat pipeline for TandemMP.

:class:`OpenIscatPipeline` needs ``open_iscat``. It implements
:class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`. :class:`Info`,
:class:`Movie`, :class:`PipelineResults` and :func:`load_mp_file` are re-exported
from their new homes for backward compatibility.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

import numpy as np

from tandemmp.io.movie import Movie, load_mp_file
from tandemmp.pipeline.pipeline_interface import EventsFound, Info, PipelineResults

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


# Pipeline execution


class OpenIscatPipeline:
    """Event-finding pipeline backed by the open_iscat stack.

    Chains ``FrameReader`` -> ``Preprocessor`` -> ``MaskFilterPeakFinder``/
    ``Analyser`` and implements
    :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`. Stateless, so it
    pickles across a :class:`multiprocessing.Pool` (e.g. via
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

        Needs ``open_iscat``.

        Args:
            filepath: Movie file.
            frame_bypass: Splice these (e.g. drift-corrected) frames in instead of
                reading from disk.
            parallel: Use parallel processing where available.
            navg: Averaged frames for ratiometric event detection.
            **kwargs: Ignored; absorbs keywords meant for other
                :class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`
                implementations.

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
