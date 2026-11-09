"""CSV-exported (DiscoverMP) event seeds as a :class:`PipelineRunner` backend.

:func:`load_event_csv` parses a DiscoverMP CSV export into typed arrays.
:class:`CsvEventsRunner` wraps that as a
:class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`: frames/rframes
come from TandemMP's own :mod:`tandemmp.io.movie` (no re-fitting — TandemMP
already does all PSF fitting itself downstream), events are the CSV seeds.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tandemmp.core.types import FloatArray, IntArray
from tandemmp.drift.corrector import DriftVector, get_corrected_movie
from tandemmp.io.movie import load_mp, ratiometric
from tandemmp.pipeline.drift_correction import (
    DriftCorrectionFunction,
    drift_vector_for_movie,
)
from tandemmp.pipeline.pipeline_interface import PipelineResults

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

# Column mapping

DISCOVERMP_COLUMNS: dict[str, str] = {
    "frame": "frame_indices",
    "x": "x_coords",
    "y": "y_coords",
    "contrast": "contrasts",
    "mass": "calibrated_values",
    "fit_error": "fit_errors",
    "diffblur_error": "diffblur_errors",
    "residual_error": "residual_errors",
    "nearest_neighbour_distance": "nearest_neighbour_distances",
    "selection": "selections",
}

_REQUIRED_ROLES = ("frame", "x", "y", "contrast", "mass")
_OPTIONAL_ROLES = (
    "fit_error",
    "diffblur_error",
    "residual_error",
    "nearest_neighbour_distance",
    "selection",
)

# Known column-name typos seen in the wild, keyed by the canonical name.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "nearest_neighbour_distances": ("neareast_neighbour_distances",),
}


def _find_column(name: str, header: Sequence[str]) -> str | None:
    if name in header:
        return name
    for alias in _COLUMN_ALIASES.get(name, ()):
        if alias in header:
            return alias
    return None


@dataclasses.dataclass(frozen=True)
class CsvEventRows:
    """Validated arrays loaded from a DiscoverMP events CSV."""

    frame: IntArray
    x: FloatArray
    y: FloatArray
    contrast: FloatArray
    mass: FloatArray
    fit_error: FloatArray | None = None
    diffblur_error: FloatArray | None = None
    residual_error: FloatArray | None = None
    nearest_neighbour_distance: FloatArray | None = None
    selection: FloatArray | None = None

    def __len__(self) -> int:
        return len(self.frame)


def load_event_csv(
    path: str | Path,
    column_map: Mapping[str, str] = DISCOVERMP_COLUMNS,
    *,
    selected_only: bool = False,
) -> CsvEventRows:
    """Load and validate a DiscoverMP-style events CSV.

    Values are parsed as Python ``float`` (tolerating scientific notation,
    ``nan`` and ``inf``); *frame* is cast to ``int`` afterwards. Columns not
    present in *column_map* (or missing from the file for an optional role)
    are left ``None``. ``selected_only=True`` filters rows to
    ``selections == 1`` (requires a ``selection`` column).
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        # DiscoverMP exports trail the data with a blank line and a block of
        # "# ..." provenance comment lines — skip both.
        rows = [r for r in reader if r and not r[0].startswith("#")]

    if header is None:
        msg = f"{path}: empty file"
        raise ValueError(msg)
    if not rows:
        msg = f"{path}: no data rows"
        raise ValueError(msg)

    resolved: dict[str, str] = {}
    for role, name in column_map.items():
        col = _find_column(name, header)
        if col is None:
            if role in _REQUIRED_ROLES:
                msg = f"{path}: missing required column {name!r} (role {role!r})"
                raise ValueError(msg)
            continue
        resolved[role] = col

    column_index = {col: header.index(col) for col in resolved.values()}

    def _column(role: str) -> FloatArray:
        idx = column_index[resolved[role]]
        return np.array([float(row[idx]) for row in rows], dtype=float)

    values: dict[str, FloatArray | None] = {
        role: (_column(role) if role in resolved else None) for role in _OPTIONAL_ROLES
    }

    result = CsvEventRows(
        frame=_column("frame").astype(int),
        x=_column("x"),
        y=_column("y"),
        contrast=_column("contrast"),
        mass=_column("mass"),
        **values,
    )

    if selected_only:
        if result.selection is None:
            msg = f"{path}: selected_only=True requires a selection column"
            raise ValueError(msg)
        sel = result.selection == 1
        result = dataclasses.replace(
            result,
            frame=result.frame[sel],
            x=result.x[sel],
            y=result.y[sel],
            contrast=result.contrast[sel],
            mass=result.mass[sel],
            fit_error=None if result.fit_error is None else result.fit_error[sel],
            diffblur_error=(
                None if result.diffblur_error is None else result.diffblur_error[sel]
            ),
            residual_error=(
                None if result.residual_error is None else result.residual_error[sel]
            ),
            nearest_neighbour_distance=(
                None
                if result.nearest_neighbour_distance is None
                else result.nearest_neighbour_distance[sel]
            ),
            selection=result.selection[sel],
        )

    return result


# Filename convention


def check_csv_matches_movie(
    csv_path: str | Path, movie_path: str | Path, *, strict: bool = True
) -> bool:
    """Check a CSV's stem (minus an optional trailing ``_events``) matches a movie's.

    E.g. ``018_..._9_dc_events.csv`` <-> ``018_..._9_dc.mp``;
    ``058_..._125x.csv`` <-> ``058_..._125x.mp``. Raises :class:`ValueError`
    on mismatch, or just logs a warning when *strict* is ``False``.
    """
    csv_stem = Path(csv_path).stem
    if csv_stem.endswith("_events"):
        csv_stem = csv_stem[: -len("_events")]
    movie_stem = Path(movie_path).stem

    matches = csv_stem == movie_stem
    if not matches:
        msg = (
            f"events CSV {str(csv_path)!r} (stem {csv_stem!r}) does not match "
            f"movie {str(movie_path)!r} (stem {movie_stem!r})"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    return matches


def find_matching_csv(movie_path: str | Path, folder: str | Path) -> Path | None:
    """Find a CSV in *folder* whose stem (see :func:`check_csv_matches_movie`)
    matches *movie_path*'s stem, or ``None`` if there isn't one."""
    for candidate in sorted(Path(folder).glob("*.csv")):
        if check_csv_matches_movie(candidate, movie_path, strict=False):
            return candidate
    return None


# EventsFound implementation


@dataclasses.dataclass(frozen=True)
class CsvEvents:
    """:class:`~tandemmp.pipeline.pipeline_interface.EventsFound` over CSV seeds."""

    x: FloatArray
    y: FloatArray
    i: IntArray
    c: FloatArray

    def get_X(self) -> FloatArray:  # noqa: N802
        return self.x

    def get_Y(self) -> FloatArray:  # noqa: N802
        return self.y

    def get_I(self) -> IntArray:  # noqa: N802
        return self.i

    def get_C(self) -> FloatArray:  # noqa: N802
        return self.c


# PipelineRunner


class CsvEventsRunner:
    """:class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner` backed by a CSV.

    Frames/ratiometric come from :mod:`tandemmp.io.movie`; events are the CSV
    seeds, drift-corrected to match whatever movie coordinate frame ``.run()``
    ends up returning (see ``events_drift_corrected``).

    Args:
        csv_path: Path to the DiscoverMP events CSV.
        events_drift_corrected: Set when the CSV was exported from an
            already drift-corrected movie (e.g. a ``..._dc.mp``/``..._dc_events.csv``
            pair) — no further drift correction is applied to the events.
            Default (``False``): CSV seeds are in raw-movie coordinates and
            get drift-corrected the same way the movie is.
        column_map: Forwarded to :func:`load_event_csv`.
        selected_only: Forwarded to :func:`load_event_csv`.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        events_drift_corrected: bool = False,
        column_map: Mapping[str, str] = DISCOVERMP_COLUMNS,
        selected_only: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.events_drift_corrected = events_drift_corrected
        self.column_map = column_map
        self.selected_only = selected_only

    def run(  # noqa: PLR0913
        self,
        filepath: str | Path,
        *,
        frame_bypass: FloatArray | None = None,
        parallel: bool = True,
        navg: int = 5,
        drift_vector: DriftVector | None = None,
        drift_crop: tuple[slice, ...] | None = None,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> PipelineResults:
        """Load CSV events, align them to *frame_bypass* (or a freshly-loaded movie).

        ``drift_vector``/``drift_crop`` are the vector/crop already applied
        to *frame_bypass* by a caller (e.g.
        :class:`~tandemmp.pipeline.loader.Loader`, which polish-fixes the
        movie — prepending ``navg * 2`` frames — before calling here, hence
        the matching shift of CSV frame indices below). When absent and
        ``events_drift_corrected`` is ``False``, drift correction is
        estimated and applied here instead, for standalone use.
        """
        filepath = Path(filepath)
        check_csv_matches_movie(self.csv_path, filepath)
        rows = load_event_csv(
            self.csv_path, column_map=self.column_map, selected_only=self.selected_only
        )

        index_shift = 0
        if frame_bypass is not None:
            frames: FloatArray = np.asarray(frame_bypass, dtype=float)
            index_shift = navg * 2
        else:
            raw_frames, _ = load_mp(filepath)
            frames = np.asarray(raw_frames, dtype=float)

        xs, ys, frame_idx, cs = rows.x, rows.y, rows.frame, rows.contrast

        if not self.events_drift_corrected:
            if drift_vector is not None:
                xs, ys = _apply_drift(xs, ys, frame_idx, drift_vector, drift_crop)
                xs, ys, frame_idx, cs = _drop_out_of_bounds(
                    xs, ys, frame_idx, cs, frames.shape[2], frames.shape[1]
                )
            elif frame_bypass is None:
                vector = drift_vector_for_movie(filepath, parallel=parallel)
                frames, _, crop = get_corrected_movie(  # type: ignore[misc]
                    frames, vector=vector, full_output=True, parallel=parallel
                )
                xs, ys = _apply_drift(xs, ys, frame_idx, vector, crop)
                xs, ys, frame_idx, cs = _drop_out_of_bounds(
                    xs, ys, frame_idx, cs, frames.shape[2], frames.shape[1]
                )

        shifted_idx = frame_idx + index_shift
        valid = (shifted_idx >= 0) & (shifted_idx < len(frames))
        n_dropped = int((~valid).sum())
        if n_dropped:
            logger.info(
                "Dropping %d/%d CSV event(s) with frame index outside the movie",
                n_dropped,
                len(valid),
            )
        xs, ys, shifted_idx, cs = (
            xs[valid],
            ys[valid],
            shifted_idx[valid],
            cs[valid],
        )

        rframes = ratiometric(frames, navg, silent=True)
        events = CsvEvents(x=xs, y=ys, i=shifted_idx.astype(int), c=cs)
        return PipelineResults(frames=frames, rframes=rframes, events=events)


def _apply_drift(
    xs: FloatArray,
    ys: FloatArray,
    frame_idx: IntArray,
    vector: DriftVector,
    crop: tuple[slice, ...] | None,
) -> tuple[FloatArray, FloatArray]:
    correct = DriftCorrectionFunction(vector=vector)
    corrected_xs, corrected_ys = correct.correct_coordinates(
        xs, ys, frame_idx.astype(float)
    )
    if crop is not None:
        y_slice, x_slice = crop[-2], crop[-1]
        corrected_xs = corrected_xs - (x_slice.start or 0)
        corrected_ys = corrected_ys - (y_slice.start or 0)
    return np.asarray(corrected_xs, dtype=float), np.asarray(corrected_ys, dtype=float)


def _drop_out_of_bounds(
    xs: FloatArray,
    ys: FloatArray,
    frame_idx: IntArray,
    cs: FloatArray,
    width: int,
    height: int,
) -> tuple[FloatArray, FloatArray, IntArray, FloatArray]:
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    n_dropped = int((~inside).sum())
    if n_dropped:
        logger.info(
            "Dropping %d/%d CSV event(s) outside the drift-corrected crop",
            n_dropped,
            len(inside),
        )
    return xs[inside], ys[inside], frame_idx[inside], cs[inside]
