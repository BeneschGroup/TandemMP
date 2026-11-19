"""Tests for tandemmp.io.calibration."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from tandemmp.core.conversions import Calibration
from tandemmp.io.calibration import load_calibration, resolve_calibration

EXPECTED = Calibration(slope=-3.3e-5, offset=-9.1e-5)


class TestLoadCalibration:
    def test_mc_file(self, tmp_path) -> None:
        path = tmp_path / "run.mc"
        # A DiscoverMP .mc holds much more than the fit; only these two are read.
        path.write_text(
            json.dumps({"gradient": -3.3e-5, "offset": -9.1e-5, "r2": 0.99, "unit": 1})
        )
        assert load_calibration(path) == EXPECTED

    def test_json_file(self, tmp_path) -> None:
        path = tmp_path / "cal.json"
        path.write_text(json.dumps({"c2kDa": -3.3e-5, "intercept": -9.1e-5}))
        assert load_calibration(path) == EXPECTED

    @pytest.mark.parametrize("suffix", [".pkl", ".pickle"])
    def test_pickle_file_with_numpy_scalars(self, tmp_path, suffix) -> None:
        path = tmp_path / f"cal{suffix}"
        payload = {"c2kDa": np.float64(-3.3e-5), "intercept": np.float64(-9.1e-5)}
        path.write_bytes(pickle.dumps(payload))
        assert load_calibration(str(path)) == EXPECTED

    def test_unsupported_suffix_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="expected a .mc"):
            load_calibration(tmp_path / "cal.txt")

    def test_missing_keys_raise(self, tmp_path) -> None:
        path = tmp_path / "cal.json"
        path.write_text(json.dumps({"gradient": -3.3e-5}))
        with pytest.raises(ValueError, match="offset"):
            load_calibration(path)

    @pytest.mark.parametrize("slope", [0, float("nan")])
    def test_bad_slope_raises(self, tmp_path, slope) -> None:
        path = tmp_path / "cal.mc"
        path.write_text(json.dumps({"gradient": slope, "offset": 0.0}))
        with pytest.raises(ValueError, match="non-zero"):
            load_calibration(path)

    def test_non_mapping_raises(self, tmp_path) -> None:
        path = tmp_path / "cal.json"
        path.write_text("[1, 2]")
        with pytest.raises(ValueError, match="mapping"):
            load_calibration(path)


class TestResolveCalibration:
    def test_passes_through_none_and_calibration(self) -> None:
        assert resolve_calibration(None) is None
        assert resolve_calibration(EXPECTED) is EXPECTED

    def test_loads_a_path(self, tmp_path) -> None:
        path = tmp_path / "run.mc"
        path.write_text(json.dumps({"gradient": -3.3e-5, "offset": -9.1e-5}))
        assert resolve_calibration(path) == EXPECTED
        assert resolve_calibration(str(path)) == EXPECTED
