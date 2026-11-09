"""Tests for tandemmp.core.conversions — target API."""

import numpy as np
import pytest

from tandemmp.core.conversions import (
    DEFAULT_CALIBRATION,
    NM_PER_PIXEL,
    Calibration,
    c_to_mass,
    get_calibration,
    mass_to_c,
    mass_to_nm_radius,
    mass_to_px_radius,
    mass_to_sigma,
    nm2pixels,
    pixels2nm,
    set_calibration,
)

# Spatial conversions


class TestSpatialConversions:
    def test_nm2pixels(self):
        assert nm2pixels(72.6) == pytest.approx(1.0)
        assert nm2pixels(0.0) == pytest.approx(0.0)

    def test_pixels2nm(self):
        assert pixels2nm(1.0) == pytest.approx(72.6)
        assert pixels2nm(0.0) == pytest.approx(0.0)

    def test_roundtrip(self):
        val = 42.0
        assert pixels2nm(nm2pixels(val)) == pytest.approx(val)

    def test_pixels2nm_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = pixels2nm(arr)
        np.testing.assert_allclose(result, [72.6, 145.2, 217.8])

    def test_nm_per_pixel_constant(self):
        assert NM_PER_PIXEL == pytest.approx(72.6)


# Mass / contrast conversions


class TestMassContrastConversions:
    def test_c_to_mass_zero_contrast(self):
        # c=0 → (0 - (-0.000070)) / (-0.000030) = -2.333...
        result = c_to_mass(0.0)
        assert isinstance(result, float)

    def test_mass_to_c_zero_mass(self):
        result = mass_to_c(0.0)
        assert isinstance(result, float)

    def test_roundtrip(self):
        mass = 500.0
        assert c_to_mass(mass_to_c(mass)) == pytest.approx(mass)

    def test_c_to_mass_array(self):
        arr = np.array([0.001, 0.002])
        result = c_to_mass(arr)
        assert result.shape == (2,)

    def test_mass_to_c_array(self):
        arr = np.array([100.0, 200.0])
        result = mass_to_c(arr)
        assert result.shape == (2,)


# Calibration


class TestCalibration:
    def test_default_calibration_matches_hardcoded_values(self):
        assert get_calibration() == DEFAULT_CALIBRATION
        assert DEFAULT_CALIBRATION == Calibration(slope=-0.000030, offset=-0.000070)

    def test_c_to_mass_uses_default_calibration(self):
        result = c_to_mass(0.0)
        expected = (0.0 - DEFAULT_CALIBRATION.offset) / DEFAULT_CALIBRATION.slope
        assert result == pytest.approx(expected)

    def test_set_calibration_overrides_conversions(self):
        custom = Calibration(slope=-0.00005, offset=-0.0001)
        set_calibration(custom)
        try:
            assert get_calibration() == custom
            assert c_to_mass(0.0) == pytest.approx((0.0 - custom.offset) / custom.slope)
            assert mass_to_c(100.0) == pytest.approx(
                100.0 * custom.slope + custom.offset
            )
        finally:
            set_calibration(None)

    def test_set_calibration_none_resets_default(self):
        set_calibration(Calibration(slope=-0.00005, offset=-0.0001))
        set_calibration(None)
        assert get_calibration() == DEFAULT_CALIBRATION

    def test_roundtrip_with_custom_calibration(self):
        set_calibration(Calibration(slope=-0.00005, offset=-0.0001))
        try:
            mass = 500.0
            assert c_to_mass(mass_to_c(mass)) == pytest.approx(mass)
        finally:
            set_calibration(None)


# mass_to_sigma


class TestMassToSigma:
    def test_returns_float(self):
        result = mass_to_sigma(500.0)
        assert isinstance(result, (int, float))

    def test_positive_mass(self):
        assert mass_to_sigma(500.0) > 0

    def test_midrange_precision_decreases_with_mass(self):
        """Between the two clamps ``sigma = a / (mass * b)``: heavier is tighter."""
        assert mass_to_sigma(500.0) < mass_to_sigma(100.0)
        assert mass_to_sigma(100.0) == pytest.approx(9.821047344076268)

    def test_heavy_mass_hits_lower_floor(self):
        """Very heavy particle: the precision estimate is floored at 1.0 nm.

        ``sigma = a / (mass * b)`` drops below 1 nm above ~982 kDa, so every
        heavier particle returns the 1.0 nm floor rather than continuing to
        shrink (the physical-radius branch is currently disabled).
        """
        assert mass_to_sigma(50_000.0) == 1.0
        assert mass_to_sigma(100_000.0) == 1.0

    def test_clamp_upper_bound(self):
        """Very light particle: precision capped at 4 px (expressed in nm)."""
        result = mass_to_sigma(1.0)
        assert result <= pixels2nm(4.0)

    def test_negative_mass_uses_abs(self):
        assert mass_to_sigma(-500.0) == mass_to_sigma(500.0)


# Radius conversions


class TestRadiusConversions:
    def test_mass_to_nm_radius(self):
        # Erickson 2009: r = 0.066 * (m*1000)^(1/3)
        result = mass_to_nm_radius(100.0)
        expected = 0.066 * (100_000.0) ** (1.0 / 3.0)
        assert result == pytest.approx(expected)

    def test_mass_to_px_radius(self):
        nm = mass_to_nm_radius(100.0)
        px = mass_to_px_radius(100.0)
        assert px == pytest.approx(nm2pixels(nm))
