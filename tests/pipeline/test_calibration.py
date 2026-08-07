"""Tests for tandemmp.pipeline.calibration — target API."""

from __future__ import annotations

import pickle
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tandemmp.pipeline.calibration as calibration_module
from tandemmp.core.conversions import Calibration
from tandemmp.pipeline.calibration import (
    CalibrationResult,
    _DragAndFitPeaks,
    _hist_bins,
    _mass_to_contrast_line,
    calibrate,
    load_calibration,
    run_calibration,
    save_calibration,
)
from tandemmp.pipeline.pipeline_interface import PipelineResults


class _FakeMplEvent:
    def __init__(self, inaxes: object, xdata: float | None) -> None:
        self.inaxes = inaxes
        self.xdata = xdata


# ---------------------------------------------------------------------------
# _hist_bins
# ---------------------------------------------------------------------------


class TestHistBins:
    def test_symmetric_around_zero(self):
        data = np.array([-0.01, -0.005, 0.0, 0.005])
        bins = _hist_bins(data, binwidth=0.002)
        assert bins.min() <= data.min()
        assert bins.max() >= data.max()
        assert np.isclose(bins, 0.0).any()

    def test_all_positive_data_starts_at_zero(self):
        data = np.array([1.0, 2.0, 3.0])
        bins = _hist_bins(data, binwidth=1.0)
        assert bins.min() == 0.0


# ---------------------------------------------------------------------------
# _mass_to_contrast_line
# ---------------------------------------------------------------------------


class TestMassToContrastLine:
    def test_linear(self):
        mass = np.array([0.0, 100.0])
        result = _mass_to_contrast_line(mass, slope=-0.00003, offset=-0.00007)
        np.testing.assert_allclose(result, [-0.00007, -0.00007 - 0.003])


# ---------------------------------------------------------------------------
# _DragAndFitPeaks
# ---------------------------------------------------------------------------


class TestDragAndFitPeaks:
    def test_drag_select_fits_gaussian_and_records_peak(self):
        rng = np.random.default_rng(0)
        data = np.concatenate(
            [
                rng.normal(loc=-0.01, scale=0.0005, size=500),
                rng.normal(loc=-0.03, scale=0.0005, size=500),
            ]
        )
        bins = _hist_bins(data, binwidth=0.0002)
        _, ax = plt.subplots()
        ax.hist(data, bins=bins)

        drag = _DragAndFitPeaks(ax=ax, data=data, bins=bins)
        drag.on_press(_FakeMplEvent(inaxes=ax, xdata=-0.013))
        drag.on_motion(_FakeMplEvent(inaxes=ax, xdata=-0.007))
        drag.on_release(_FakeMplEvent(inaxes=ax, xdata=-0.007))

        assert len(drag.peak_means) == 1
        assert drag.peak_means[0] == pytest.approx(-0.01, abs=0.001)
        assert drag.rect is None
        assert drag.selection_start is None
        plt.close("all")

    def test_ignores_events_outside_axes(self):
        data = np.array([0.0, 1.0, 2.0])
        bins = np.array([0.0, 1.0, 2.0, 3.0])
        _, ax = plt.subplots()
        drag = _DragAndFitPeaks(ax=ax, data=data, bins=bins)

        drag.on_press(_FakeMplEvent(inaxes=None, xdata=1.0))
        assert drag.selection_start is None
        plt.close("all")

    def test_too_few_points_in_range_is_a_noop(self):
        data = np.array([0.0, 1.0, 2.0])
        bins = np.array([0.0, 1.0, 2.0, 3.0])
        _, ax = plt.subplots()
        drag = _DragAndFitPeaks(ax=ax, data=data, bins=bins)

        drag.on_press(_FakeMplEvent(inaxes=ax, xdata=0.4))
        drag.on_motion(_FakeMplEvent(inaxes=ax, xdata=0.6))
        drag.on_release(_FakeMplEvent(inaxes=ax, xdata=0.6))

        assert drag.peak_means == []
        plt.close("all")


# ---------------------------------------------------------------------------
# calibrate() orchestration — interactive pieces stubbed out
# ---------------------------------------------------------------------------


class _FakeDrag:
    """Stands in for _DragAndFitPeaks with canned peak means.

    The widget's own drag-select-and-fit mechanics are covered directly by
    TestDragAndFitPeaks; this lets calibrate()'s orchestration (the
    reattempt/repeat loop, the final line fit) be tested without simulating
    real mouse events.
    """

    def __init__(self, ax: object, data: object, bins: object) -> None:
        self.peak_means = [-0.01, -0.02, -0.03]


class _FakeYesNoWindow:
    """Always answers "no" — ends the reattempt/repeat loop after one pass."""

    def __init__(
        self, title: str, message: str, yes_action: object, no_action: object
    ) -> None:
        no_action()


class _FakeMassEntryWindow:
    def __init__(self, contrast_peaks: object) -> None:
        self.calibrant_masses = [100.0, 200.0, 300.0]


@pytest.fixture
def _stub_interactive_pieces(monkeypatch):
    monkeypatch.setattr(calibration_module, "_DragAndFitPeaks", _FakeDrag)
    monkeypatch.setattr(calibration_module, "_YesNoWindow", _FakeYesNoWindow)
    monkeypatch.setattr(calibration_module, "_MassEntryWindow", _FakeMassEntryWindow)


class TestCalibrate:
    def test_fits_line_and_builds_result(self, _stub_interactive_pieces):
        contrasts = np.array([-0.01, -0.02, -0.03, -0.5])
        result = calibrate(contrasts, source_filepath="foo.mp")

        assert isinstance(result, CalibrationResult)
        # peaks=[-0.01,-0.02,-0.03], masses=[100,200,300] -> slope*mass+0 == peak
        assert result.calibration.slope == pytest.approx(-0.0001)
        assert result.calibration.offset == pytest.approx(0.0, abs=1e-8)
        assert result.calibrant_masses == [100.0, 200.0, 300.0]
        assert result.fitted_contrast_peaks == pytest.approx([-0.01, -0.02, -0.03])
        assert result.source_filepath == "foo.mp"


# ---------------------------------------------------------------------------
# run_calibration()
# ---------------------------------------------------------------------------


class _FakeEventsFound:
    def __init__(self, contrasts: Sequence[float]) -> None:
        self._contrasts = np.asarray(contrasts, dtype=float)

    def get_X(self):  # noqa: N802
        return np.zeros_like(self._contrasts)

    def get_Y(self):  # noqa: N802
        return np.zeros_like(self._contrasts)

    def get_I(self):  # noqa: N802
        return np.zeros_like(self._contrasts, dtype=int)

    def get_C(self):  # noqa: N802
        return self._contrasts


class _FakePipelineRunner:
    """Stands in for OpenIscatPipeline / PipelineRunner without open_iscat."""

    def __init__(self, contrasts: Sequence[float] = (-0.01, -0.02)) -> None:
        self._contrasts = contrasts
        self.calls: list[dict[str, object]] = []

    def run(self, filepath, *, frame_bypass=None, parallel=True, **kwargs):
        self.calls.append(
            {"filepath": filepath, "parallel": parallel, "kwargs": kwargs}
        )
        return PipelineResults(
            frames=np.zeros((1, 1, 1)),
            rframes=np.zeros((1, 1, 1)),
            events=_FakeEventsFound(self._contrasts),
        )


class TestRunCalibration:
    def test_wires_pipeline_events_into_calibrate(self, monkeypatch):
        calls: dict[str, object] = {}

        def fake_calibrate(contrasts, source_filepath, **kwargs):
            calls["contrasts"] = np.asarray(contrasts)
            calls["source_filepath"] = source_filepath
            return CalibrationResult(
                calibration=Calibration(slope=-0.0001, offset=0.0),
                source_filepath=str(source_filepath),
                calibrant_masses=[],
                fitted_contrast_peaks=[],
                residual_sum_of_squares=0.0,
            )

        monkeypatch.setattr(calibration_module, "calibrate", fake_calibrate)
        fake_runner = _FakePipelineRunner(contrasts=(-0.01, -0.02))

        result = run_calibration("foo.mp", pipeline_runner=fake_runner)

        np.testing.assert_allclose(calls["contrasts"], [-0.01, -0.02])
        assert calls["source_filepath"] == "foo.mp"
        assert isinstance(result, CalibrationResult)

    def test_defaults_to_open_iscat_pipeline_when_no_runner_given(self, monkeypatch):
        fake_runner = _FakePipelineRunner(contrasts=(-0.01, -0.02))
        monkeypatch.setattr(
            calibration_module, "OpenIscatPipeline", lambda: fake_runner
        )
        monkeypatch.setattr(
            calibration_module,
            "calibrate",
            lambda contrasts, source_filepath, **kwargs: CalibrationResult(
                calibration=Calibration(slope=-0.0001, offset=0.0),
                source_filepath=str(source_filepath),
                calibrant_masses=[],
                fitted_contrast_peaks=[],
                residual_sum_of_squares=0.0,
            ),
        )

        run_calibration("foo.mp")

        assert len(fake_runner.calls) == 1

    def test_parallel_and_pipeline_kwargs_forwarded_to_runner(self, monkeypatch):
        fake_runner = _FakePipelineRunner(contrasts=(-0.01, -0.02))
        monkeypatch.setattr(
            calibration_module,
            "calibrate",
            lambda contrasts, source_filepath, **kwargs: CalibrationResult(
                calibration=Calibration(slope=-0.0001, offset=0.0),
                source_filepath=str(source_filepath),
                calibrant_masses=[],
                fitted_contrast_peaks=[],
                residual_sum_of_squares=0.0,
            ),
        )

        run_calibration(
            "foo.mp",
            pipeline_runner=fake_runner,
            parallel=False,
            navg=7,
        )

        assert fake_runner.calls == [
            {"filepath": "foo.mp", "parallel": False, "kwargs": {"navg": 7}}
        ]


# ---------------------------------------------------------------------------
# save_calibration / load_calibration
# ---------------------------------------------------------------------------


class TestSaveLoadCalibration:
    def test_roundtrip(self, tmp_path):
        result = CalibrationResult(
            calibration=Calibration(slope=-0.0001, offset=0.0),
            source_filepath="foo.mp",
            calibrant_masses=[100.0, 200.0],
            fitted_contrast_peaks=[-0.01, -0.02],
            residual_sum_of_squares=1e-10,
        )
        path = save_calibration(tmp_path / "foo_calibration.json", result)
        loaded = load_calibration(path)
        assert loaded == result

    def test_saved_file_is_plain_editable_json(self, tmp_path):
        import json

        result = CalibrationResult(
            calibration=Calibration(slope=-0.0001, offset=0.00002),
            source_filepath="foo.mp",
            calibrant_masses=[100.0, 200.0],
            fitted_contrast_peaks=[-0.01, -0.02],
            residual_sum_of_squares=1e-10,
        )
        path = save_calibration(tmp_path / "foo_calibration.json", result)
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload == {
            "c2kDa": -0.0001,
            "intercept": 0.00002,
            "raw_error": 1e-10,
            "calibrant_masses": [100.0, 200.0],
            "fitted_contrast_peaks": [-0.01, -0.02],
            "source_filepath": "foo.mp",
        }

    def test_hand_edited_json_loads(self, tmp_path):
        path = tmp_path / "hand_edited.json"
        path.write_text(
            '{"c2kDa": -0.00005, "intercept": 0.0001, '
            '"calibrant_masses": [66.0], "fitted_contrast_peaks": [-0.003]}',
            encoding="utf-8",
        )
        loaded = load_calibration(path)
        assert loaded.calibration == Calibration(slope=-0.00005, offset=0.0001)
        assert loaded.calibrant_masses == [66.0]
        assert loaded.fitted_contrast_peaks == [-0.003]
        assert loaded.source_filepath == str(path)

    def test_loads_legacy_pickled_dict(self, tmp_path):
        legacy = {
            "c2kDa": np.float64(-0.00003),
            "intercept": np.float64(-0.00007),
            "raw_error": np.float64(2.5e-9),
            "calibrant_masses": np.array([66.0, 146.0, 480.0]),
            "fitted_contrast_peaks": np.array([-0.0021, -0.0046, -0.0151]),
        }
        path = tmp_path / "legacy.pickle"
        with path.open("wb") as fh:
            pickle.dump(legacy, fh)

        loaded = load_calibration(path)

        assert loaded.calibration == Calibration(slope=-0.00003, offset=-0.00007)
        assert loaded.residual_sum_of_squares == pytest.approx(2.5e-9)
        assert loaded.calibrant_masses == pytest.approx([66.0, 146.0, 480.0])
        assert loaded.fitted_contrast_peaks == pytest.approx(
            [-0.0021, -0.0046, -0.0151]
        )
        assert loaded.source_filepath == str(path)

    def test_save_creates_parent_directories(self, tmp_path):
        result = CalibrationResult(
            calibration=Calibration(slope=-0.0001, offset=0.0),
            source_filepath="foo.mp",
            calibrant_masses=[],
            fitted_contrast_peaks=[],
            residual_sum_of_squares=0.0,
        )
        path = tmp_path / "nested" / "dir" / "foo_calibration.json"
        save_calibration(path, result)
        assert path.exists()

    def test_save_rejects_wrong_type(self, tmp_path):
        with pytest.raises(TypeError):
            save_calibration(tmp_path / "bad.json", {"not": "a CalibrationResult"})

    def test_load_rejects_missing_required_fields(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"not": "a calibration"}', encoding="utf-8")
        with pytest.raises(ValueError, match="c2kDa"):
            load_calibration(path)
