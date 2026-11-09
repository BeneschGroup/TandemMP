"""Tests for tandemmp.pipeline.batch — target API.

``process_movie``/``batch_process`` are tested against a lightweight fake
``Experiment`` monkeypatched into the module, since real construction is expensive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tandemmp.core.conversions import Calibration, get_calibration
from tandemmp.pipeline import batch

# Fakes / helpers


class _FakeExperiment:
    """Stand-in for ``pipeline.pca.Experiment``.

    ``_events_by_path`` maps a filepath (as ``str``) to a fake event count;
    unmapped paths default to a single event.
    """

    _events_by_path: dict[str, int] = {}
    instances: list[_FakeExperiment] = []

    def __init__(self, mp_filepath: str | Path, **kwargs: Any) -> None:
        self.mp_filepath = Path(mp_filepath)
        self.kwargs = kwargs
        self.events = [object()] * self._events_by_path.get(str(self.mp_filepath), 1)
        self.analysed = False
        self.saved_to: Path | None = None
        self.saved_with_movie: bool | None = None
        self.calibration_at_construction = get_calibration()
        type(self).instances.append(self)

    def analyse(self) -> None:
        self.analysed = True

    def save(self, filepath: str | Path, *, save_movie: bool = True) -> Path:
        self.saved_to = Path(filepath)
        self.saved_with_movie = save_movie
        return self.saved_to


@pytest.fixture(autouse=True)
def _reset_fake_experiment() -> None:
    _FakeExperiment._events_by_path = {}
    _FakeExperiment.instances = []


@pytest.fixture()
def fake_experiment(monkeypatch: pytest.MonkeyPatch) -> type[_FakeExperiment]:
    monkeypatch.setattr(batch, "Experiment", _FakeExperiment)
    return _FakeExperiment


# process_movie


class TestProcessMovie:
    def test_constructs_experiment_with_flattened_parameters(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        movie = tmp_path / "movie.mp"
        batch.process_movie(movie, parameters={"minimum_mass": 30})

        assert len(fake_experiment.instances) == 1
        exp = fake_experiment.instances[0]
        assert exp.mp_filepath == movie
        assert exp.kwargs == {"minimum_mass": 30, "pipeline_runner": None}

    def test_no_parameters_passes_no_kwargs(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        batch.process_movie(tmp_path / "movie.mp")
        assert fake_experiment.instances[0].kwargs == {"pipeline_runner": None}

    def test_passes_pipeline_runner_through_to_experiment(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        sentinel_runner = object()
        batch.process_movie(tmp_path / "movie.mp", pipeline_runner=sentinel_runner)
        assert fake_experiment.instances[0].kwargs["pipeline_runner"] is sentinel_runner

    def test_no_calibration_uses_default(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        from tandemmp.core.conversions import DEFAULT_CALIBRATION

        batch.process_movie(tmp_path / "movie.mp")
        assert fake_experiment.instances[0].calibration_at_construction == (
            DEFAULT_CALIBRATION
        )

    def test_calibration_is_active_during_experiment_construction(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        custom = Calibration(slope=-0.00005, offset=-0.0001)
        batch.process_movie(tmp_path / "movie.mp", calibration=custom)
        assert fake_experiment.instances[0].calibration_at_construction == custom

    def test_analyses_and_saves_below_threshold(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        movie = tmp_path / "movie.mp"
        fake_experiment._events_by_path = {str(movie): 10}

        result = batch.process_movie(movie, max_events=3500)

        expected_output = movie.with_name("movie_tandemmp.pkl")
        exp = fake_experiment.instances[0]
        assert exp.analysed
        assert exp.saved_to == expected_output
        assert exp.saved_with_movie is False
        assert result.skipped is False
        assert result.n_events == 10
        assert result.output_filepath == expected_output
        assert result.filepath == movie
        assert result.duration_seconds >= 0

    def test_explicit_save_filepath_overrides_default(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        movie = tmp_path / "movie.mp"
        out = tmp_path / "custom_output.pkl"

        result = batch.process_movie(movie, save_filepath=out)

        assert fake_experiment.instances[0].saved_to == out
        assert result.output_filepath == out

    def test_skips_when_event_count_at_or_above_max(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        movie = tmp_path / "movie.mp"
        fake_experiment._events_by_path = {str(movie): 3500}

        result = batch.process_movie(movie, max_events=3500)

        exp = fake_experiment.instances[0]
        assert not exp.analysed
        assert exp.saved_to is None
        assert result.skipped is True
        assert result.n_events == 3500
        assert result.output_filepath is None

    def test_below_threshold_is_not_skipped(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        movie = tmp_path / "movie.mp"
        fake_experiment._events_by_path = {str(movie): 3499}

        result = batch.process_movie(movie, max_events=3500)

        assert result.skipped is False
        assert fake_experiment.instances[0].analysed

    def test_accepts_str_filepath(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        movie = tmp_path / "movie.mp"
        result = batch.process_movie(str(movie))
        assert result.filepath == movie


# batch_process


class TestBatchProcess:
    def test_serial_processes_every_file(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        movies = [tmp_path / "a.mp", tmp_path / "b.mp", tmp_path / "c.mp"]
        fake_experiment._events_by_path = {str(m): 5 for m in movies}

        results = batch.batch_process(movies, parallel=False)

        assert [r.filepath for r in results] == movies
        assert all(r.skipped is False for r in results)
        assert len(fake_experiment.instances) == 3

    def test_serial_passes_shared_parameters_and_max_events(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        movies = [tmp_path / "a.mp", tmp_path / "b.mp"]
        fake_experiment._events_by_path = {str(m): 4000 for m in movies}

        results = batch.batch_process(
            movies, parameters={"minimum_mass": 25}, max_events=5000, parallel=False
        )

        assert all(
            exp.kwargs == {"minimum_mass": 25, "pipeline_runner": None}
            for exp in fake_experiment.instances
        )
        assert all(r.skipped is False for r in results)

    def test_passes_pipeline_runner_to_each_process_movie_call(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        sentinel_runner = object()
        movies = [tmp_path / "a.mp", tmp_path / "b.mp"]

        batch.batch_process(movies, pipeline_runner=sentinel_runner, parallel=False)

        assert all(
            exp.kwargs["pipeline_runner"] is sentinel_runner
            for exp in fake_experiment.instances
        )

    def test_passes_calibration_to_each_process_movie_call(
        self, fake_experiment: type[_FakeExperiment], tmp_path: Path
    ) -> None:
        custom = Calibration(slope=-0.00005, offset=-0.0001)
        movies = [tmp_path / "a.mp", tmp_path / "b.mp"]

        batch.batch_process(movies, calibration=custom, parallel=False)

        assert all(
            exp.calibration_at_construction == custom
            for exp in fake_experiment.instances
        )

    def test_empty_filepaths_returns_empty(
        self, fake_experiment: type[_FakeExperiment]
    ) -> None:
        assert batch.batch_process([], parallel=False) == []
        assert batch.batch_process([], parallel=True) == []

    def test_dispatches_via_dispatch_kwargs_with_default_n_jobs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}
        movies = [tmp_path / f"{i}.mp" for i in range(4)]
        sentinel_results = [
            batch.BatchResult(
                filepath=m,
                duration_seconds=0.0,
                n_events=0,
                skipped=False,
                output_filepath=None,
            )
            for m in movies
        ]

        def _fake_dispatch(
            method: Any, args_list: Any, *, parallel: bool, n_jobs: Any
        ) -> list[Any]:
            captured["method"] = method
            captured["args_list"] = args_list
            captured["parallel"] = parallel
            captured["n_jobs"] = n_jobs
            return sentinel_results

        monkeypatch.setattr(batch, "dispatch_kwargs", _fake_dispatch)

        results = batch.batch_process(movies, parallel=True)

        assert captured["method"] is batch.process_movie
        assert captured["parallel"] is True
        assert captured["n_jobs"] == 4
        assert [a["filepath"] for a in captured["args_list"]] == movies
        assert results == sentinel_results

    def test_n_jobs_caps_at_six_by_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            batch,
            "dispatch_kwargs",
            lambda method, args_list, *, parallel, n_jobs: (
                captured.update(n_jobs=n_jobs) or []
            ),
        )

        movies = [tmp_path / f"{i}.mp" for i in range(10)]
        batch.batch_process(movies, parallel=True)

        assert captured["n_jobs"] == 6

    def test_explicit_n_jobs_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            batch,
            "dispatch_kwargs",
            lambda method, args_list, *, parallel, n_jobs: (
                captured.update(n_jobs=n_jobs) or []
            ),
        )

        movies = [tmp_path / f"{i}.mp" for i in range(10)]
        batch.batch_process(movies, parallel=True, n_jobs=2)

        assert captured["n_jobs"] == 2
