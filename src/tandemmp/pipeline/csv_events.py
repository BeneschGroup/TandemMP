"""Event-finding runner that reads events from a DiscoverMP CSV.

The CSV holds events found in the raw movie by commercial software.
:class:`CsvEventsRunner` maps them into the :class:`~tandemmp.pipeline.loader.Loader`
working movie (frame slice and binning, drift, crop, ``final_polish_fix`` offset) and
builds the ratiometric movie, so it plugs in as a
:class:`~tandemmp.pipeline.pipeline_interface.PipelineRunner`.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tandemmp.io.movie import ratiometric
from tandemmp.pipeline.pipeline_interface import PipelineResults

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy.typing as npt

    from tandemmp.drift.corrector import DriftVector

__all__ = [
    "DISCOVERMP_COLUMNS",
    "CsvEvents",
    "CsvEventsFound",
    "CsvEventsRunner",
    "check_csv_matches_movie",
    "load_event_csv",
    "map_events_to_movie",
    "pair_movies_with_csvs",
]

# Role -> CSV column. The first four are required.
DISCOVERMP_COLUMNS: dict[str, str] = {
    "frame": "frame_indices",
    "x": "x_coords",
    "y": "y_coords",
    "contrast": "contrasts",
    "mass": "calibrated_values",
    "selected": "selections",
}
_REQUIRED_ROLES = ("frame", "x", "y", "contrast")


@dataclasses.dataclass(frozen=True)
class CsvEvents:
    """Events read from a CSV, one array entry per event.

    *mass* is NaN where the CSV has no mass column or no value.
    """

    frame: npt.NDArray[np.int64]
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    contrast: npt.NDArray[np.float64]
    mass: npt.NDArray[np.float64]


def load_event_csv(
    path: str | Path,
    column_map: Mapping[str, str] = DISCOVERMP_COLUMNS,
    *,
    selected_only: bool = False,
) -> CsvEvents:
    """Read a DiscoverMP events CSV, skipping its ``#`` metadata lines.

    Args:
        path: CSV file with a header row.
        column_map: Role (``frame``, ``x``, ``y``, ``contrast`` and optionally
            ``mass`` and ``selected``) to column name.
        selected_only: Keep only rows with a non-zero ``selected`` value.

    Raises:
        ValueError: If a required column is missing, or a frame, x, y or contrast
            value is non-finite, or a frame is negative or not a whole number.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        # DiscoverMP appends "# ..." metadata lines; skip those and blank lines.
        reader = (r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#"))
        header = [h.strip() for h in next(reader, [])]
        rows = list(reader)

    def column(role: str) -> npt.NDArray[np.float64] | None:
        name = column_map.get(role)
        if name not in header:
            return None
        idx = header.index(name)
        return np.array([float(r[idx]) for r in rows], dtype=float)

    frame, x, y, contrast = (column(r) for r in _REQUIRED_ROLES)
    if frame is None or x is None or y is None or contrast is None:
        missing = [
            column_map[r]
            for r, a in zip(_REQUIRED_ROLES, (frame, x, y, contrast), strict=True)
            if a is None
        ]
        msg = f"{path}: missing column(s) {missing}; found {header}"
        raise ValueError(msg)
    mass = column("mass")
    selected = column("selected")
    if selected_only and selected is None:
        msg = f"{path}: selected_only needs a {column_map.get('selected')!r} column"
        raise ValueError(msg)

    if not all(np.isfinite(a).all() for a in (frame, x, y, contrast)):
        msg = f"{path}: frame, x, y and contrast must be finite"
        raise ValueError(msg)
    if (frame < 0).any() or not np.array_equal(frame, np.round(frame)):
        msg = f"{path}: frame values must be non-negative whole numbers"
        raise ValueError(msg)

    keep: npt.NDArray[np.bool_] = np.ones(len(frame), dtype=bool)
    if selected_only and selected is not None:
        keep = selected != 0
    nan = np.full(len(frame), np.nan)
    return CsvEvents(
        frame=frame[keep].astype(np.int64),
        x=x[keep],
        y=y[keep],
        contrast=contrast[keep],
        mass=(nan if mass is None else mass)[keep],
    )


def check_csv_matches_movie(csv_path: str | Path, movie_path: str | Path) -> None:
    """Raise :class:`ValueError` unless the CSV and movie share a file stem."""
    if Path(csv_path).stem != Path(movie_path).stem:
        msg = (
            f"Events CSV {Path(csv_path).name!r} does not match movie "
            f"{Path(movie_path).name!r}; the file names must be the same"
        )
        raise ValueError(msg)


def pair_movies_with_csvs(
    paths: str | Path | Iterable[str | Path],
) -> list[tuple[Path, Path]]:
    """Pair each ``.mp`` movie with the ``.csv`` of the same file stem.

    Raises:
        ValueError: If a movie or CSV has no partner, a stem is repeated, or a file
            is neither ``.mp`` nor ``.csv``.
    """
    items = (
        [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    )
    movies: dict[str, Path] = {}
    csvs: dict[str, Path] = {}
    for path in items:
        kind = {".mp": movies, ".csv": csvs}.get(path.suffix.lower())
        if kind is None:
            msg = f"Expected .mp or .csv files, got {path.name!r}"
            raise ValueError(msg)
        if path.stem in kind:
            msg = f"More than one {path.suffix} file named {path.stem!r}"
            raise ValueError(msg)
        kind[path.stem] = path
    unmatched = sorted(movies.keys() ^ csvs.keys())
    if unmatched:
        msg = f"No matching movie/CSV pair for: {unmatched}"
        raise ValueError(msg)
    return [(movies[stem], csvs[stem]) for stem in movies]


def map_events_to_movie(  # noqa: PLR0913
    events: CsvEvents,
    *,
    n_frames: int,
    frame_shape: tuple[int, int],
    navg: int,
    frame_start: int = 0,
    frame_binning: int = 1,
    drift_vector: DriftVector | None = None,
    drift_crop: tuple[slice, slice] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
]:
    """Map raw-movie events into the Loader's working-movie coordinates.

    In order: frame slice and temporal binning, drift subtraction, crop offset,
    then the ``2 * navg`` frames :func:`~tandemmp.io.movie.final_polish_fix`
    prepends. Events outside the working movie, or in its last *navg* frames
    (which have no ratiometric frame), are dropped.

    Args:
        n_frames: Length of the polished movie (including the prepended frames).
        frame_shape: ``(height, width)`` of the working movie.
        frame_start: First raw frame kept by the Loader.
        frame_binning: Raw frames per working frame.
        drift_vector: Per-working-frame drift the movie was corrected by, if any.
        drift_crop: ``(y, x)`` slices the corrected movie was cropped to, if any.

    Returns:
        ``(x, y, i, contrast)`` of the kept events.
    """
    from tandemmp.pipeline.drift_correction import DriftCorrectionFunction

    prefix = 2 * navg
    i = (events.frame - frame_start) // frame_binning
    x, y, c = events.x, events.y, events.contrast
    # The last navg working frames have no ratiometric frame (it is NaN there).
    keep = (events.frame >= frame_start) & (i < n_frames - prefix - navg)
    x, y, i, c = x[keep], y[keep], i[keep], c[keep]

    if drift_vector is not None:
        x, y = DriftCorrectionFunction(drift_vector).correct_coordinates(
            x, y, i.astype(float)
        )
    if drift_crop is not None:
        y = y - (drift_crop[0].start or 0)
        x = x - (drift_crop[1].start or 0)

    height, width = frame_shape
    keep = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[keep], y[keep], i[keep] + prefix, c[keep]


@dataclasses.dataclass(frozen=True)
class CsvEventsFound:
    """Events as per-axis arrays (see :class:`EventsFound`)."""

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    i: npt.NDArray[np.int64]
    c: npt.NDArray[np.float64]

    def get_X(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return self.x

    def get_Y(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return self.y

    def get_I(self) -> npt.NDArray[np.int64]:  # noqa: N802
        return self.i

    def get_C(self) -> npt.NDArray[np.float64]:  # noqa: N802
        return self.c


class CsvEventsRunner:
    """Pipeline runner whose events come from a DiscoverMP CSV.

    Stateless and picklable. The CSV is *events_csv* (here or per call), else the
    ``<movie stem>.csv`` next to the movie; its stem must match the movie's.
    """

    def __init__(
        self,
        events_csv: str | Path | None = None,
        *,
        column_map: Mapping[str, str] = DISCOVERMP_COLUMNS,
        selected_only: bool = False,
    ) -> None:
        self.events_csv = events_csv
        self.column_map = dict(column_map)
        self.selected_only = selected_only

    def run(  # noqa: PLR0913
        self,
        filepath: str | Path,
        *,
        frame_bypass: npt.NDArray[np.floating[Any]] | None = None,
        parallel: bool = True,  # noqa: ARG002
        navg: int = 5,
        events_csv: str | Path | None = None,
        drift_vector: DriftVector | None = None,
        drift_crop: tuple[slice, slice] | None = None,
        frame_start: int = 0,
        frame_binning: int = 1,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> PipelineResults:
        """Map the CSV events into *frame_bypass* and compute its ratiometric movie.

        Args:
            frame_bypass: The Loader's polished working movie (required).
            navg: Ratiometric half-window, also the polish offset (``2 * navg``).
            events_csv: Overrides the runner's CSV.
            drift_vector, drift_crop, frame_start, frame_binning: How the Loader
                turned the raw movie into *frame_bypass*; see
                :func:`map_events_to_movie`.
            **kwargs: Ignored; absorbs keywords meant for other runners.
        """
        if frame_bypass is None:
            msg = "CsvEventsRunner needs frame_bypass, the Loader's working movie"
            raise ValueError(msg)

        csv_path = events_csv or self.events_csv or Path(filepath).with_suffix(".csv")
        if not Path(csv_path).exists():
            msg = f"Events CSV not found: {csv_path}"
            raise FileNotFoundError(msg)
        check_csv_matches_movie(csv_path, filepath)

        events = load_event_csv(
            csv_path, self.column_map, selected_only=self.selected_only
        )
        frames = np.asarray(frame_bypass)
        x, y, i, c = map_events_to_movie(
            events,
            n_frames=len(frames),
            frame_shape=(frames.shape[1], frames.shape[2]),
            navg=navg,
            frame_start=frame_start,
            frame_binning=frame_binning,
            drift_vector=drift_vector,
            drift_crop=drift_crop,
        )
        return PipelineResults(
            frames=frames,
            # DiscoverMP contrasts are before/after - 1; ratiometric() is after/before.
            rframes=-ratiometric(frames, navg, silent=True),
            events=CsvEventsFound(x=x, y=y, i=i, c=c),
        )
