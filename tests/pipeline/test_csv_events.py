"""Tests for tandemmp.pipeline.csv_events."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from tandemmp.drift.corrector import DriftVector
from tandemmp.pipeline.csv_events import (
    CsvEvents,
    CsvEventsRunner,
    check_csv_matches_movie,
    load_event_csv,
    map_events_to_movie,
)
from tandemmp.pipeline.pipeline_interface import PipelineResults, PipelineRunner

HEADER = "frame_indices,x_coords,y_coords,contrasts,calibrated_values,selections"


def _write_csv(path, rows, header=HEADER):
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def _events(frame, x, y, c=-0.01):
    n = len(frame)
    return CsvEvents(
        frame=np.asarray(frame, dtype=np.int64),
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        contrast=np.full(n, c),
        mass=np.full(n, np.nan),
    )


# load_event_csv


class TestLoadEventCsv:
    def test_reads_columns(self, tmp_path) -> None:
        path = _write_csv(
            tmp_path / "a.csv",
            ["5.0e+00,20.5,10.25,-0.01,nan,1", "7,3,4,0.02,60,0"],
        )
        events = load_event_csv(path)
        assert events.frame.tolist() == [5, 7]
        assert events.x.tolist() == [20.5, 3.0]
        assert events.y.tolist() == [10.25, 4.0]
        assert events.contrast.tolist() == [-0.01, 0.02]
        assert np.isnan(events.mass[0])
        assert events.mass[1] == 60.0

    def test_skips_comment_and_blank_lines(self, tmp_path) -> None:
        path = tmp_path / "a.csv"
        lines = [
            "# leading note",
            HEADER,
            "5,1,2,-0.1,nan,1",
            "",
            "# Refeyn DiscoverMP; 2026.1.1; email: x",
            "# Time created: now",
        ]
        path.write_text("\n".join(lines) + "\n")
        events = load_event_csv(path)
        assert events.frame.tolist() == [5]

    def test_selected_only(self, tmp_path) -> None:
        path = _write_csv(tmp_path / "a.csv", ["5,1,1,-0.1,nan,1", "6,2,2,-0.1,nan,0"])
        assert load_event_csv(path, selected_only=True).frame.tolist() == [5]
        assert load_event_csv(path).frame.tolist() == [5, 6]

    def test_mass_optional(self, tmp_path) -> None:
        path = _write_csv(
            tmp_path / "a.csv",
            ["5,1,1,-0.1"],
            header="frame_indices,x_coords,y_coords,contrasts",
        )
        assert np.isnan(load_event_csv(path).mass).all()

    def test_custom_column_map(self, tmp_path) -> None:
        path = _write_csv(tmp_path / "a.csv", ["5,1,2,-0.1"], header="f,px,py,c")
        events = load_event_csv(
            path, {"frame": "f", "x": "px", "y": "py", "contrast": "c"}
        )
        assert events.y.tolist() == [2.0]

    def test_missing_column_raises(self, tmp_path) -> None:
        path = _write_csv(
            tmp_path / "a.csv", ["5,1,1"], header="frame_indices,x_coords,y_coords"
        )
        with pytest.raises(ValueError, match="contrasts"):
            load_event_csv(path)

    def test_selected_only_without_column_raises(self, tmp_path) -> None:
        path = _write_csv(
            tmp_path / "a.csv",
            ["5,1,1,-0.1"],
            header="frame_indices,x_coords,y_coords,contrasts",
        )
        with pytest.raises(ValueError, match="selected_only"):
            load_event_csv(path, selected_only=True)

    @pytest.mark.parametrize(
        "row", ["nan,1,1,-0.1,nan,1", "5,nan,1,-0.1,nan,1", "5,1,1,inf,nan,1"]
    )
    def test_non_finite_raises(self, tmp_path, row) -> None:
        with pytest.raises(ValueError, match="finite"):
            load_event_csv(_write_csv(tmp_path / "a.csv", [row]))

    @pytest.mark.parametrize("frame", ["-1", "2.5"])
    def test_bad_frame_raises(self, tmp_path, frame) -> None:
        with pytest.raises(ValueError, match="whole numbers"):
            load_event_csv(_write_csv(tmp_path / "a.csv", [f"{frame},1,1,-0.1,nan,1"]))

    def test_empty(self, tmp_path) -> None:
        assert len(load_event_csv(_write_csv(tmp_path / "a.csv", [])).frame) == 0


# check_csv_matches_movie


class TestCheckCsvMatchesMovie:
    def test_same_stem_passes(self) -> None:
        check_csv_matches_movie("dir/run1.csv", "other/run1.mp")

    def test_different_stem_raises(self) -> None:
        with pytest.raises(ValueError, match="same"):
            check_csv_matches_movie("run1.csv", "run2.mp")


# map_events_to_movie


def _map(events, **kwargs):
    defaults = {"n_frames": 30, "frame_shape": (20, 40), "navg": 5}
    return map_events_to_movie(events, **{**defaults, **kwargs})


class TestMapEventsToMovie:
    def test_adds_polish_offset(self) -> None:
        x, y, i, c = _map(_events([0, 4], [1.0, 2.0], [3.0, 4.0]))
        assert i.tolist() == [10, 14]
        assert x.tolist() == [1.0, 2.0]
        assert y.tolist() == [3.0, 4.0]
        assert c.tolist() == [-0.01, -0.01]

    def test_frame_start_and_binning(self) -> None:
        # raw frame 14 -> (14 - 4) // 5 = 2
        _, _, i, _ = _map(
            _events([3, 14], [1, 1], [1, 1]), frame_start=4, frame_binning=5
        )
        assert i.tolist() == [2 + 10]

    def test_drops_frames_without_ratiometric(self) -> None:
        # 30 - 10 = 20 working frames; the last navg = 5 have no ratiometric frame
        _, _, i, _ = _map(_events([14, 15, 25], [1, 1, 1], [1, 1, 1]))
        assert i.tolist() == [24]

    def test_drift_is_subtracted_per_frame(self) -> None:
        vector = DriftVector(np.arange(20, dtype=float), np.full(20, 2.0))
        x, y, _, _ = _map(_events([3, 6], [10, 10], [10, 10]), drift_vector=vector)
        assert x.tolist() == [7.0, 4.0]
        assert y.tolist() == [8.0, 8.0]

    def test_drift_interpolates_fractional_frames(self) -> None:
        vector = DriftVector(np.arange(20, dtype=float), np.zeros(20))
        x, _, _, _ = _map(_events([4], [10], [5]), frame_binning=2, drift_vector=vector)
        assert x.tolist() == [8.0]  # raw 4 -> working 2 -> drift 2

    def test_crop_offset_and_bounds(self) -> None:
        crop = (slice(2, -2), slice(5, None))
        x, y, i, _ = _map(
            _events([0, 0, 0], [5.0, 4.0, 44.0], [2.0, 6.0, 6.0]),
            drift_crop=crop,
            frame_shape=(16, 36),
        )
        # (5, 2) -> (0, 0) kept; x=4 -> -1 dropped; x=44 -> 39 >= 36 dropped
        assert (x.tolist(), y.tolist(), i.tolist()) == ([0.0], [0.0], [10])

    def test_crop_slice_with_no_start(self) -> None:
        crop = (slice(None, -3), slice(None, -3))
        x, y, _, _ = _map(_events([0], [4.0], [5.0]), drift_crop=crop)
        assert (x.tolist(), y.tolist()) == ([4.0], [5.0])

    def test_edge_events_dropped(self) -> None:
        x, _, _, _ = _map(_events([0, 0], [-0.5, 39.9], [5, 5]))
        assert x.tolist() == [39.9]


# CsvEventsRunner


@pytest.fixture
def movie_and_csv(tmp_path):
    movie = tmp_path / "run1.mp"
    movie.touch()
    csv_path = _write_csv(
        tmp_path / "run1.csv", ["2,12.0,7.0,-0.02,nan,1", "3,30.0,15.0,0.03,nan,1"]
    )
    return movie, csv_path


class TestCsvEventsRunner:
    def _frames(self, n=30):
        rng = np.random.default_rng(0)
        frames = 1000.0 + rng.random((n, 20, 40))
        frames[13:, 7, 12] *= 1.1  # a bright particle appears at frame 13
        return frames

    def test_satisfies_protocol(self) -> None:
        assert isinstance(CsvEventsRunner(), PipelineRunner)

    def test_picklable(self) -> None:
        assert isinstance(
            pickle.loads(pickle.dumps(CsvEventsRunner())), CsvEventsRunner
        )  # noqa: S301

    def test_run_returns_results(self, movie_and_csv) -> None:
        movie, _ = movie_and_csv
        frames = self._frames()
        results = CsvEventsRunner().run(movie, frame_bypass=frames, navg=5)
        assert isinstance(results, PipelineResults)
        assert results.frames.shape == frames.shape
        assert results.rframes.shape == frames.shape
        assert np.isnan(results.rframes[:5]).all()
        assert np.isfinite(results.rframes[5:-5]).all()
        # A frame brightening after the event is a negative (binding) contrast.
        assert results.rframes[12, 7, 12] == pytest.approx(-0.1, abs=0.02)
        assert results.events.get_I().tolist() == [12, 13]
        assert results.events.get_X().tolist() == [12.0, 30.0]
        assert results.events.get_Y().tolist() == [7.0, 15.0]
        assert results.events.get_C().tolist() == [-0.02, 0.03]

    def test_explicit_csv_overrides_sibling(self, movie_and_csv, tmp_path) -> None:
        movie, _ = movie_and_csv
        other = tmp_path / "sub"
        other.mkdir()
        csv_path = _write_csv(other / "run1.csv", ["0,1,1,-0.1,nan,1"])
        results = CsvEventsRunner(csv_path).run(
            movie, frame_bypass=self._frames(), navg=5
        )
        assert results.events.get_I().tolist() == [10]
        results = CsvEventsRunner().run(
            movie, frame_bypass=self._frames(), navg=5, events_csv=csv_path
        )
        assert results.events.get_I().tolist() == [10]

    def test_requires_frame_bypass(self, movie_and_csv) -> None:
        with pytest.raises(ValueError, match="frame_bypass"):
            CsvEventsRunner().run(movie_and_csv[0])

    def test_missing_csv_raises(self, tmp_path) -> None:
        movie = tmp_path / "run1.mp"
        movie.touch()
        with pytest.raises(FileNotFoundError):
            CsvEventsRunner().run(movie, frame_bypass=self._frames())

    def test_mismatched_csv_raises(self, movie_and_csv, tmp_path) -> None:
        movie, _ = movie_and_csv
        wrong = _write_csv(tmp_path / "run2.csv", ["0,1,1,-0.1,nan,1"])
        with pytest.raises(ValueError, match="same"):
            CsvEventsRunner(wrong).run(movie, frame_bypass=self._frames())

    def test_ignores_unknown_kwargs(self, movie_and_csv) -> None:
        CsvEventsRunner().run(
            movie_and_csv[0], frame_bypass=self._frames(), l_thum=13, navg=5
        )
