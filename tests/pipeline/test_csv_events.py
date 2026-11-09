"""Tests for tandemmp.pipeline.csv_events and its wiring into every entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tandemmp.drift.corrector import DriftVector, get_corrected_movie
from tandemmp.fitting.gaussian_psf import GaussianPsfModel
from tandemmp.io.movie import final_polish_fix
from tandemmp.pipeline.csv_events import (
    CsvEventRows,
    CsvEvents,
    CsvEventsRunner,
    check_csv_matches_movie,
    find_matching_csv,
    load_event_csv,
)
from tandemmp.pipeline.pipeline_interface import PipelineResults

# Fixtures / helpers


def _write_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def _write_mp(path: Path, frames: np.ndarray, *, navg: int = 5) -> Path:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        frame_ds = f.create_dataset("movie/frame", data=frames.astype(np.uint16))
        frame_ds.attrs["encoding"] = "BASIC"
        acq = f.create_group("movie/configuration/acq_camera")
        acq.create_dataset("frame_rate", data=1000.0)
        acq.create_dataset("frame_binning", data=1)
        acq.create_dataset("pixel_binning", data=1)
        f["movie/configuration"].create_dataset(
            "scatter_interference_destructive", data=True
        )
    return path


def _make_binding_movie(
    n_frames: int = 30,
    size: int = 24,
    event_frame: int = 10,
    xy: tuple[int, int] = (12, 12),
    c: float = -0.05,
    width: int = 9,
    baseline: float = 1000.0,
) -> np.ndarray:
    """A movie with a Gaussian-shaped intensity drop at *xy* starting after *event_frame*."""
    bump = GaussianPsfModel().psf_2d_offset(width, 0.0, 0.0, c)
    movie = np.full((n_frames, size, size), baseline, dtype=float)
    half = width // 2
    x, y = xy
    region = (slice(y - half, y + half + 1), slice(x - half, x + half + 1))
    for f in range(event_frame + 1, n_frames):
        movie[f][region] = baseline * (1 + bump)
    return movie


# load_event_csv — real DiscoverMP column variants (fixtures cut from real exports)

_AAV_HEADER = (
    "frame_indices,x_coords,y_coords,contrasts,calibrated_values,fit_errors,"
    "diffblur_errors,residual_errors,neareast_neighbour_distances,selections"
)
_AAV_ROWS = [
    "5.000000000000000000e+00,2.037105339915977140e+02,7.015815150576175085e+01,"
    "-7.139297675652776093e-03,nan,2.293647440462389664e-01,nan,"
    "1.601412573045667696e-04,2.251557703448198211e+01,1.000000000000000000e+00",
    "5.000000000000000000e+00,1.711347472594304975e+02,1.045962581770996991e+02,"
    "-1.049558963714337587e-02,nan,2.121093436024682477e-01,nan,"
    "3.108549187792116766e-04,4.740427272225672084e+01,1.000000000000000000e+00",
]

_PAIR_MATCHING_HEADER = (
    "frame_indices,x_coords,y_coords,contrasts,calibrated_values,fit_errors,"
    "diffblur_errors,residual_errors,nearest_neighbour_distances,selections"
)
_PAIR_MATCHING_ROWS = [
    "5.000000000000000000e+00,1.109019107184958415e+02,5.111274842018919173e+01,"
    "2.694101734970400040e-03,-8.218887724772054071e+01,1.023254690567504954e-01,"
    "nan,8.472519605670096715e-06,inf,1.000000000000000000e+00",
    "1.100000000000000000e+01,9.087802878247569538e+01,4.839483560339024137e+01,"
    "2.421905990288400132e-03,-7.426854803092309965e+01,1.336867407653763995e-01,"
    "nan,9.621620284011376306e-06,2.918220772921523931e+01,1.000000000000000000e+00",
]


class TestLoadEventCsv:
    def test_raw_movie_variant_with_typo_alias(self, tmp_path: Path) -> None:
        csv_path = _write_csv(tmp_path / "aav.csv", _AAV_HEADER, _AAV_ROWS)
        rows = load_event_csv(csv_path)
        assert isinstance(rows, CsvEventRows)
        assert len(rows) == 2
        assert rows.frame.dtype.kind == "i"
        np.testing.assert_array_equal(rows.frame, [5, 5])
        assert rows.x[0] == pytest.approx(203.7105339915977)
        assert rows.y[1] == pytest.approx(104.5962581770997)
        assert np.all(np.isnan(rows.mass))
        assert rows.diffblur_error is not None
        assert np.all(np.isnan(rows.diffblur_error))
        assert rows.nearest_neighbour_distance is not None
        assert rows.nearest_neighbour_distance[0] == pytest.approx(22.51557703448198)

    def test_dc_pair_matching_variant_with_inf(self, tmp_path: Path) -> None:
        csv_path = _write_csv(
            tmp_path / "pair.csv", _PAIR_MATCHING_HEADER, _PAIR_MATCHING_ROWS
        )
        rows = load_event_csv(csv_path)
        np.testing.assert_array_equal(rows.frame, [5, 11])
        assert rows.mass[0] == pytest.approx(-82.18887724772054)
        assert rows.nearest_neighbour_distance is not None
        assert np.isinf(rows.nearest_neighbour_distance[0])
        assert rows.selection is not None
        np.testing.assert_array_equal(rows.selection, [1.0, 1.0])

    def test_selected_only_filters_rows(self, tmp_path: Path) -> None:
        header = (
            "frame_indices,x_coords,y_coords,contrasts,calibrated_values,selections"
        )
        rows_text = [
            "1,10.0,10.0,-0.01,100.0,1.0",
            "2,20.0,20.0,-0.01,100.0,0.0",
        ]
        csv_path = _write_csv(tmp_path / "sel.csv", header, rows_text)
        rows = load_event_csv(csv_path, selected_only=True)
        assert len(rows) == 1
        assert rows.x[0] == pytest.approx(10.0)

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        header = "x_coords,y_coords,contrasts,calibrated_values"
        csv_path = _write_csv(tmp_path / "bad.csv", header, ["10.0,10.0,-0.01,100.0"])
        with pytest.raises(ValueError, match="frame_indices"):
            load_event_csv(csv_path)

    def test_missing_optional_column_is_none(self, tmp_path: Path) -> None:
        header = "frame_indices,x_coords,y_coords,contrasts,calibrated_values"
        csv_path = _write_csv(
            tmp_path / "noopt.csv", header, ["1,10.0,10.0,-0.01,100.0"]
        )
        rows = load_event_csv(csv_path)
        assert rows.selection is None
        assert rows.fit_error is None

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_event_csv(csv_path)

    def test_header_only_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "header_only.csv"
        csv_path.write_text(
            "frame_indices,x_coords,y_coords,contrasts,calibrated_values\n"
        )
        with pytest.raises(ValueError, match="no data rows"):
            load_event_csv(csv_path)

    def test_trailing_provenance_comment_block_is_ignored(self, tmp_path: Path) -> None:
        # Real DiscoverMP exports trail the data with a blank line and a
        # block of "# ..." provenance comment lines.
        header = "frame_indices,x_coords,y_coords,contrasts,calibrated_values"
        csv_path = tmp_path / "with_footer.csv"
        csv_path.write_text(
            header + "\r\n"
            "1,10.0,10.0,-0.01,nan\r\n"
            "\r\n"
            "# Refeyn DiscoverMP; 2026.1.1; Refeyn Ltd; www.refeyn.com\r\n"
            "# Time created: 2026-11-09 17:12:07\r\n"
        )
        rows = load_event_csv(csv_path)
        assert len(rows) == 1
        assert rows.x[0] == pytest.approx(10.0)


# check_csv_matches_movie / find_matching_csv


class TestCheckCsvMatchesMovie:
    def test_plain_stem_match(self) -> None:
        assert check_csv_matches_movie(
            "058_20260418_APTES_aav5_125x.csv", "058_20260418_APTES_aav5_125x.mp"
        )

    def test_events_suffix_stripped(self) -> None:
        assert check_csv_matches_movie(
            "018_d_pimensis_25nM_9_dc_events.csv", "018_d_pimensis_25nM_9_dc.mp"
        )

    def test_mismatch_raises_by_default(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            check_csv_matches_movie("a.csv", "b.mp")

    def test_mismatch_strict_false_warns_and_returns_false(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            result = check_csv_matches_movie("a.csv", "b.mp", strict=False)
        assert result is False
        assert "does not match" in caplog.text


class TestFindMatchingCsv:
    def test_finds_match_by_stem(self, tmp_path: Path) -> None:
        (tmp_path / "018_movie_events.csv").write_text("x")
        (tmp_path / "other_events.csv").write_text("x")
        match = find_matching_csv(tmp_path / "018_movie.mp", tmp_path)
        assert match == tmp_path / "018_movie_events.csv"

    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        (tmp_path / "other_events.csv").write_text("x")
        assert find_matching_csv(tmp_path / "018_movie.mp", tmp_path) is None


# CsvEvents


class TestCsvEvents:
    def test_accessors(self) -> None:
        events = CsvEvents(
            x=np.array([1.0]), y=np.array([2.0]), i=np.array([3]), c=np.array([-0.1])
        )
        assert events.get_X() is events.x
        assert events.get_Y() is events.y
        assert events.get_I() is events.i
        assert events.get_C() is events.c


# CsvEventsRunner — deterministic drift/crop math


class TestCsvEventsRunnerDeterministicDrift:
    def _csv_path(self, tmp_path: Path, stem: str, rows: list[str]) -> Path:
        header = "frame_indices,x_coords,y_coords,contrasts,calibrated_values"
        return _write_csv(tmp_path / f"{stem}.csv", header, rows)

    def test_filename_mismatch_raises(self, tmp_path: Path) -> None:
        csv_path = self._csv_path(tmp_path, "movie_a", ["1,10.0,10.0,-0.01,nan"])
        runner = CsvEventsRunner(csv_path)
        with pytest.raises(ValueError, match="does not match"):
            runner.run(tmp_path / "movie_b.mp", frame_bypass=np.zeros((10, 20, 20)))

    def test_driven_by_loader_applies_vector_crop_and_frame_shift(
        self, tmp_path: Path
    ) -> None:
        navg = 5
        raw_x, raw_y, raw_frame = 10.0, 10.0, 8
        csv_path = self._csv_path(
            tmp_path, "movie", [f"{raw_frame},{raw_x},{raw_y},-0.05,nan"]
        )
        raw_movie = np.zeros((20, 30, 30))
        vector = DriftVector(np.linspace(0.0, 2.0, 20), np.linspace(0.0, -1.0, 20))
        corrected, _, crop = get_corrected_movie(
            raw_movie, vector=vector, full_output=True, parallel=False, use_rust=False
        )
        polished = final_polish_fix(corrected, navg)

        runner = CsvEventsRunner(csv_path)
        result = runner.run(
            tmp_path / "movie.mp",
            frame_bypass=polished,
            navg=navg,
            drift_vector=vector,
            drift_crop=crop,
            parallel=False,
        )

        assert isinstance(result, PipelineResults)
        y_slice, x_slice = crop[-2], crop[-1]
        dx = float(np.interp(raw_frame, np.arange(20), vector.x))
        dy = float(np.interp(raw_frame, np.arange(20), vector.y))
        expected_x = raw_x - dx - (x_slice.start or 0)
        expected_y = raw_y - dy - (y_slice.start or 0)

        assert result.events.get_X()[0] == pytest.approx(expected_x, abs=0.1)
        assert result.events.get_Y()[0] == pytest.approx(expected_y, abs=0.1)
        assert result.events.get_I()[0] == raw_frame + navg * 2
        assert result.frames is polished

    def test_events_drift_corrected_skips_correction(self, tmp_path: Path) -> None:
        navg = 5
        csv_path = self._csv_path(tmp_path, "movie_dc", ["8,10.0,10.0,-0.05,nan"])
        movie = np.zeros((20, 30, 30))
        polished = final_polish_fix(movie, navg)
        vector = DriftVector(np.full(20, 5.0), np.full(20, -5.0))

        runner = CsvEventsRunner(csv_path, events_drift_corrected=True)
        result = runner.run(
            tmp_path / "movie_dc.mp",
            frame_bypass=polished,
            navg=navg,
            drift_vector=vector,
            parallel=False,
        )

        assert result.events.get_X()[0] == pytest.approx(10.0)
        assert result.events.get_Y()[0] == pytest.approx(10.0)
        assert result.events.get_I()[0] == 8 + navg * 2

    def test_out_of_bounds_crop_event_is_dropped(self, tmp_path: Path) -> None:
        navg = 5
        csv_path = self._csv_path(
            tmp_path,
            "movie_crop",
            ["8,10.0,10.0,-0.05,nan", "8,1.0,1.0,-0.05,nan"],
        )
        raw_movie = np.zeros((20, 30, 30))
        vector = DriftVector(np.full(20, 5.0), np.full(20, 0.0))
        corrected, _, crop = get_corrected_movie(
            raw_movie, vector=vector, full_output=True, parallel=False, use_rust=False
        )
        polished = final_polish_fix(corrected, navg)

        runner = CsvEventsRunner(csv_path)
        result = runner.run(
            tmp_path / "movie_crop.mp",
            frame_bypass=polished,
            navg=navg,
            drift_vector=vector,
            drift_crop=crop,
            parallel=False,
        )
        # second event (1.0, 1.0) drifts further out of the crop than the first.
        assert len(result.events.get_X()) == 1

    def test_frame_index_beyond_movie_length_dropped(self, tmp_path: Path) -> None:
        navg = 5
        csv_path = self._csv_path(tmp_path, "movie_oob", ["1000,10.0,10.0,-0.05,nan"])
        polished = final_polish_fix(np.zeros((20, 30, 30)), navg)

        runner = CsvEventsRunner(csv_path)
        result = runner.run(
            tmp_path / "movie_oob.mp", frame_bypass=polished, navg=navg, parallel=False
        )
        assert len(result.events.get_X()) == 0

    def test_standalone_without_frame_bypass_loads_and_corrects_movie(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        navg = 5
        csv_path = self._csv_path(
            tmp_path, "movie_standalone", ["8,10.0,10.0,-0.05,nan"]
        )
        raw_movie = np.zeros((20, 30, 30))

        import tandemmp.pipeline.csv_events as ce

        monkeypatch.setattr(ce, "load_mp", lambda fp: (raw_movie, object()))
        vector = DriftVector(np.full(20, 2.0), np.full(20, -1.0))
        monkeypatch.setattr(ce, "drift_vector_for_movie", lambda *a, **k: vector)

        runner = CsvEventsRunner(csv_path)
        result = runner.run(tmp_path / "movie_standalone.mp", navg=navg, parallel=False)

        _, _, crop = get_corrected_movie(
            raw_movie, vector=vector, full_output=True, parallel=False
        )
        y_slice, x_slice = crop[-2], crop[-1]
        expected_x = 10.0 - 2.0 - (x_slice.start or 0)
        expected_y = 10.0 - -1.0 - (y_slice.start or 0)

        assert result.events.get_X()[0] == pytest.approx(expected_x, abs=0.1)
        assert result.events.get_Y()[0] == pytest.approx(expected_y, abs=0.1)
        # standalone path returns the (uncropped-index-shift) movie directly —
        # no polish-fix applied, so frame index is unshifted.
        assert result.events.get_I()[0] == 8


# End-to-end entry points


@pytest.fixture()
def binding_fixture(tmp_path: Path) -> dict[str, Any]:
    movie = _make_binding_movie()
    movie_path = _write_mp(tmp_path / "007_fixture.mp", movie)
    csv_path = _write_csv(
        tmp_path / "007_fixture.csv",
        "frame_indices,x_coords,y_coords,contrasts,calibrated_values",
        ["10,12.0,12.0,-0.05,nan"],
    )
    return {"movie_path": movie_path, "csv_path": csv_path}


def _small_params(**overrides: Any) -> Any:
    from tandemmp.core.parameters import get_parameters

    defaults: dict[str, Any] = {
        "drift_correction": False,
        "lid_open_threshold": 0.0,
        "auto_bin": False,
        "minimum_mass": 0.001,
        "snr_limit": 0.0,
        "parallel": False,
    }
    defaults.update(overrides)
    return get_parameters(**defaults)


class TestLoaderEntryPoint:
    def test_events_csv_runs_end_to_end(
        self, binding_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tandemmp.pipeline.loader as loader_module

        monkeypatch.setattr(
            loader_module, "get_psf_model", lambda **k: GaussianPsfModel()
        )

        loader = loader_module.Loader(
            binding_fixture["movie_path"],
            parameters=_small_params(),
            events_csv=binding_fixture["csv_path"],
            skip_exp_model=True,
        )
        assert isinstance(loader.events, list)
        assert len(loader.eventsFound.get_X()) == 1

    def test_events_csv_with_adjacent_events_exercises_merge_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two close, adjacent-frame events trigger Loader.correct_movements's
        # merge path, which builds a *new* RatiometricEvent with the default
        # (module-cached) peak_model — tandemmp.events._base's own
        # get_psf_model reference, not tandemmp.pipeline.loader's, so both
        # must be patched for a real end-to-end run with no bundled PSF.
        import tandemmp.events._base as events_base_module
        import tandemmp.pipeline.loader as loader_module

        movie = _make_binding_movie()
        movie_path = _write_mp(tmp_path / "008_fixture.mp", movie)
        csv_path = _write_csv(
            tmp_path / "008_fixture.csv",
            "frame_indices,x_coords,y_coords,contrasts,calibrated_values",
            ["10,12.0,12.0,-0.05,nan", "11,12.3,11.8,-0.05,nan"],
        )

        monkeypatch.setattr(
            loader_module, "get_psf_model", lambda **k: GaussianPsfModel()
        )
        monkeypatch.setattr(
            events_base_module, "get_psf_model", lambda **k: GaussianPsfModel()
        )

        loader = loader_module.Loader(
            movie_path,
            parameters=_small_params(),
            events_csv=csv_path,
            skip_exp_model=True,
        )
        assert len(loader.eventsFound.get_X()) == 2
        # The merge collapses the two close/adjacent-frame seeds into one.
        assert len(loader.events) == 1

    def test_events_csv_and_pipeline_runner_both_set_raises(
        self, binding_fixture: dict[str, Any]
    ) -> None:
        from tandemmp.pipeline.csv_events import CsvEventsRunner
        from tandemmp.pipeline.loader import Loader

        with pytest.raises(ValueError, match="one of"):
            Loader(
                binding_fixture["movie_path"],
                events_csv=binding_fixture["csv_path"],
                pipeline_runner=CsvEventsRunner(binding_fixture["csv_path"]),
                just_movie=True,
            )


class TestExperimentEntryPoint:
    def test_events_csv_runs_end_to_end(
        self, binding_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tandemmp.pipeline.loader as loader_module
        from tandemmp.pipeline.pca import Experiment

        monkeypatch.setattr(
            loader_module, "get_psf_model", lambda **k: GaussianPsfModel()
        )

        experiment = Experiment(
            binding_fixture["movie_path"],
            events_csv=binding_fixture["csv_path"],
            skip_exp_model=True,
            **{
                "drift_correction": False,
                "lid_open_threshold": 0.0,
                "auto_bin": False,
                "minimum_mass": 0.001,
                "snr_limit": 0.0,
                "parallel": False,
            },
        )
        assert len(experiment.events) >= 0
        assert isinstance(experiment.loader.eventsFound.get_X(), np.ndarray)


class TestBatchEntryPoints:
    def test_process_movie_events_csv_runs_end_to_end(
        self, binding_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tandemmp.pipeline.loader as loader_module
        from tandemmp.pipeline import batch

        monkeypatch.setattr(
            loader_module, "get_psf_model", lambda **k: GaussianPsfModel()
        )

        result = batch.process_movie(
            binding_fixture["movie_path"],
            events_csv=binding_fixture["csv_path"],
            parameters={
                "drift_correction": False,
                "lid_open_threshold": 0.0,
                "auto_bin": False,
                "minimum_mass": 0.001,
                "snr_limit": 0.0,
                "parallel": False,
            },
            save_filepath=binding_fixture["movie_path"].with_suffix(".pkl"),
        )
        assert result.skipped is False

    def test_batch_process_auto_pairs_folder(
        self, binding_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tandemmp.pipeline.loader as loader_module
        from tandemmp.pipeline import batch

        monkeypatch.setattr(
            loader_module, "get_psf_model", lambda **k: GaussianPsfModel()
        )

        folder = binding_fixture["movie_path"].parent
        results = batch.batch_process(
            [binding_fixture["movie_path"]],
            events_csv=folder,
            parameters={
                "drift_correction": False,
                "lid_open_threshold": 0.0,
                "auto_bin": False,
                "minimum_mass": 0.001,
                "snr_limit": 0.0,
            },
            parallel=False,
        )
        assert len(results) == 1
        assert results[0].skipped is False

    def test_batch_process_raises_when_no_match_in_folder(self, tmp_path: Path) -> None:
        from tandemmp.pipeline import batch

        movie_path = tmp_path / "no_csv_here.mp"
        movie_path.write_text("x")
        with pytest.raises(ValueError, match="no matching events CSV"):
            batch.batch_process([movie_path], events_csv=tmp_path, parallel=False)


class TestPairMatchingEntryPoint:
    def test_events_csv_runs_end_to_end(
        self, binding_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tandemmp.pipeline.loader as loader_module
        from tandemmp.pipeline.pair_matching import run_pair_matching

        monkeypatch.setattr(
            loader_module, "get_psf_model", lambda **k: GaussianPsfModel()
        )

        result = run_pair_matching(
            binding_fixture["movie_path"],
            events_csv=binding_fixture["csv_path"],
            drift_correction=False,
            lid_open_threshold=0.0,
            auto_bin=False,
            minimum_mass=0.001,
            snr_limit=0.0,
            parallel=False,
        )
        assert isinstance(result.events, list)
