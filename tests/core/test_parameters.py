"""Tests for tandemmp.core.parameters — target API."""

from __future__ import annotations

import dataclasses

import pytest

from tandemmp.core.parameters import (
    ContinuationParams,
    DriftCorrectionParams,
    ExperimentalPsfParams,
    FirstAnalysisParams,
    FullAnalysisParams,
    TandemMPAnalysisParams,
    get_parameters,
)

# Defaults


class TestDefaults:
    def test_full_params_has_defaults(self) -> None:
        p = FullAnalysisParams()
        assert isinstance(p, FullAnalysisParams)

    def test_nested_defaults(self) -> None:
        p = FullAnalysisParams()
        assert isinstance(p.drift_correction, DriftCorrectionParams)
        assert isinstance(p.first_analysis, FirstAnalysisParams)
        assert isinstance(p.tandemmp_analysis, TandemMPAnalysisParams)
        assert isinstance(
            p.first_analysis.experimental_psf_model,
            ExperimentalPsfParams,
        )
        assert isinstance(
            p.tandemmp_analysis.particle_continuation,
            ContinuationParams,
        )

    def test_drift_correction_defaults(self) -> None:
        dc = DriftCorrectionParams()
        assert dc.frame_gap == 25
        assert dc.z_drift is False

    def test_first_analysis_defaults(self) -> None:
        fa = FirstAnalysisParams()
        assert fa.navg == 5
        assert fa.l_thum == 13


# Frozen (immutable)


class TestFrozen:
    def test_cannot_mutate(self) -> None:
        p = FullAnalysisParams()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.minimum_mass = 999  # type: ignore[misc]

    def test_nested_frozen(self) -> None:
        dc = DriftCorrectionParams()
        with pytest.raises(dataclasses.FrozenInstanceError):
            dc.frame_gap = 999  # type: ignore[misc]


# Validation


class TestValidation:
    def test_probability_in_range(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            TandemMPAnalysisParams(minimum_matching_probability=1.5)

    def test_probability_negative(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            TandemMPAnalysisParams(minimum_matching_probability=-0.1)

    def test_probability_boundary_zero(self) -> None:
        p = TandemMPAnalysisParams(minimum_matching_probability=0.0)
        assert p.minimum_matching_probability == 0.0

    def test_probability_boundary_one(self) -> None:
        p = TandemMPAnalysisParams(minimum_matching_probability=1.0)
        assert p.minimum_matching_probability == 1.0

    def test_percentile_in_range(self) -> None:
        with pytest.raises(ValueError, match="percentile"):
            ExperimentalPsfParams(selection_percentile=101)

    def test_positive_int_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            FirstAnalysisParams(navg=-1)

    def test_positive_float_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ContinuationParams(spatial_threshold=-1.0)


# get_parameters


class TestGetParameters:
    def test_no_overrides(self) -> None:
        p = get_parameters()
        assert p == FullAnalysisParams()

    def test_top_level_override(self) -> None:
        p = get_parameters(minimum_mass=100.0)
        assert p.minimum_mass == 100.0

    def test_nested_override(self) -> None:
        p = get_parameters(navg=10)
        assert p.first_analysis.navg == 10

    def test_deeply_nested_override(self) -> None:
        p = get_parameters(sampling_points=0.5)
        assert p.first_analysis.experimental_psf_model.sampling_points == 0.5

    def test_multiple_overrides(self) -> None:
        p = get_parameters(minimum_mass=200.0, navg=7, frame_gap=50)
        assert p.minimum_mass == 200.0
        assert p.first_analysis.navg == 7
        assert p.drift_correction.frame_gap == 50

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="not_a_real_param"):
            get_parameters(not_a_real_param=42)


# dataclasses.replace


class TestReplace:
    def test_replace_top_level(self) -> None:
        p = FullAnalysisParams()
        p2 = dataclasses.replace(p, minimum_mass=200.0)
        assert p2.minimum_mass == 200.0
        assert p.minimum_mass == 40.0  # original unchanged

    def test_replace_nested(self) -> None:
        p = FullAnalysisParams()
        new_dc = dataclasses.replace(p.drift_correction, frame_gap=50)
        p2 = dataclasses.replace(p, drift_correction=new_dc)
        assert p2.drift_correction.frame_gap == 50


# Bool-or-params pattern


class TestBoolOrParams:
    def test_drift_correction_disabled(self) -> None:
        p = FullAnalysisParams(drift_correction=False)
        assert p.drift_correction is False

    def test_drift_correction_custom(self) -> None:
        dc = DriftCorrectionParams(frame_gap=50)
        p = FullAnalysisParams(drift_correction=dc)
        assert p.drift_correction.frame_gap == 50

    def test_experimental_psf_disabled(self) -> None:
        fa = FirstAnalysisParams(experimental_psf_model=False)
        assert fa.experimental_psf_model is False

    def test_particle_continuation_disabled(self) -> None:
        ta = TandemMPAnalysisParams(particle_continuation=False)
        assert ta.particle_continuation is False
