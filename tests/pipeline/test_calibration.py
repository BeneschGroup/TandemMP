"""Tests for tandemmp.pipeline.calibration — target API."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from tandemmp.core.conversions import Calibration
from tandemmp.pipeline.calibration import (
    CalibrationResult,
    load_calibration,
    load_mc,
    resolve_calibration,
    save_calibration,
)

FIXTURES = Path(__file__).parent / "fixtures"
MC_FIXTURE = FIXTURES / "20260329_mfp1.mc"


# load_mc


class TestLoadMc:
    def test_parses_gradient_and_offset_as_slope_and_offset(self) -> None:
        result = load_mc(MC_FIXTURE)
        payload = json.loads(MC_FIXTURE.read_text(encoding="utf-8"))

        assert result.calibration == Calibration(
            slope=payload["gradient"], offset=payload["offset"]
        )

    def test_keeps_name_contrasts_calibrated_values_r2_as_metadata(self) -> None:
        result = load_mc(MC_FIXTURE)
        payload = json.loads(MC_FIXTURE.read_text(encoding="utf-8"))

        assert result.name == payload["name"]
        assert result.fitted_contrast_peaks == payload["contrasts"]
        assert result.calibrant_masses == payload["calibrated_values"]
        assert result.r2 == pytest.approx(payload["r2"])
        assert result.source_filepath == str(MC_FIXTURE)

    def test_masses_reproduce_calibrated_values_within_max_calibrated_error(
        self,
    ) -> None:
        result = load_mc(MC_FIXTURE)
        payload = json.loads(MC_FIXTURE.read_text(encoding="utf-8"))
        max_error_pct = payload["max_calibrated_error"]
        cal = result.calibration

        for contrast, expected_mass in zip(
            payload["contrasts"], payload["calibrated_values"], strict=True
        ):
            mass = (contrast - cal.offset) / cal.slope
            error_pct = abs(mass - expected_mass) / expected_mass * 100
            assert error_pct <= max_error_pct

    def test_missing_gradient_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.mc"
        path.write_text(json.dumps({"offset": -1.5e-5}), encoding="utf-8")
        with pytest.raises(ValueError, match="gradient"):
            load_mc(path)


# load_calibration dispatch


class TestLoadCalibrationDispatch:
    def test_mc_suffix_goes_through_load_mc(self) -> None:
        result = load_calibration(MC_FIXTURE)
        assert result == load_mc(MC_FIXTURE)

    def test_mc_suffix_is_case_insensitive(self, tmp_path: Path) -> None:
        upper = tmp_path / "upper.MC"
        upper.write_text(MC_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
        result = load_calibration(upper)
        assert result.calibration == load_mc(MC_FIXTURE).calibration

    def test_json_suffix_goes_through_legacy_loader(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.json"
        path.write_text(
            '{"c2kDa": -0.00005, "intercept": 0.0001}', encoding="utf-8"
        )
        result = load_calibration(path)
        assert result.calibration == Calibration(slope=-0.00005, offset=0.0001)
        assert result.name is None
        assert result.r2 is None


# save_calibration / load_calibration (TandemMP's own JSON format)


class TestSaveLoadCalibration:
    def test_roundtrip(self, tmp_path: Path) -> None:
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

    def test_saved_file_is_plain_editable_json(self, tmp_path: Path) -> None:
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

    def test_hand_edited_json_loads(self, tmp_path: Path) -> None:
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

    def test_loads_legacy_pickled_dict(self, tmp_path: Path) -> None:
        legacy = {
            "c2kDa": -0.00003,
            "intercept": -0.00007,
            "raw_error": 2.5e-9,
            "calibrant_masses": [66.0, 146.0, 480.0],
            "fitted_contrast_peaks": [-0.0021, -0.0046, -0.0151],
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

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
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

    def test_save_rejects_wrong_type(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            save_calibration(tmp_path / "bad.json", {"not": "a CalibrationResult"})

    def test_load_rejects_missing_required_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"not": "a calibration"}', encoding="utf-8")
        with pytest.raises(ValueError, match="c2kDa"):
            load_calibration(path)


# resolve_calibration


class TestResolveCalibration:
    def test_none_passes_through(self) -> None:
        assert resolve_calibration(None) is None

    def test_calibration_passes_through_unchanged(self) -> None:
        cal = Calibration(slope=-0.0001, offset=0.0)
        assert resolve_calibration(cal) is cal

    def test_path_loads_mc_file(self) -> None:
        assert resolve_calibration(MC_FIXTURE) == load_mc(MC_FIXTURE).calibration

    def test_str_path_loads_mc_file(self) -> None:
        assert resolve_calibration(str(MC_FIXTURE)) == load_mc(MC_FIXTURE).calibration

    def test_path_loads_legacy_json_file(self, tmp_path: Path) -> None:
        result = CalibrationResult(
            calibration=Calibration(slope=-0.0001, offset=0.0),
            source_filepath="foo.mp",
            calibrant_masses=[],
            fitted_contrast_peaks=[],
            residual_sum_of_squares=0.0,
        )
        path = save_calibration(tmp_path / "foo_calibration.json", result)
        assert resolve_calibration(path) == result.calibration
