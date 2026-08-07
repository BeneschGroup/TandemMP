"""Tests for tandemmp.simulation.defaults — target API."""

from __future__ import annotations

import dataclasses

import pytest

from tandemmp.simulation.defaults import SimulationDefaults


class TestSimulationDefaults:
    def test_has_expected_fields(self) -> None:
        d = SimulationDefaults()
        assert hasattr(d, "binding_rate")
        assert hasattr(d, "unbinding_rate")
        assert hasattr(d, "diffusion_rate")

    def test_default_values(self) -> None:
        d = SimulationDefaults()
        assert d.binding_rate == 0.005
        assert d.unbinding_rate == 0.0001
        assert d.diffusion_rate == 0.001

    def test_frozen(self) -> None:
        d = SimulationDefaults()
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.binding_rate = 0.1  # type: ignore[misc]

    def test_custom_values(self) -> None:
        d = SimulationDefaults(
            binding_rate=0.01,
            unbinding_rate=0.002,
            diffusion_rate=0.05,
        )
        assert d.binding_rate == 0.01
        assert d.unbinding_rate == 0.002
        assert d.diffusion_rate == 0.05

    def test_rates_are_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SimulationDefaults(binding_rate=-0.01)

    def test_replace(self) -> None:
        d = SimulationDefaults()
        d2 = dataclasses.replace(d, binding_rate=0.1)
        assert d2.binding_rate == 0.1
        assert d.binding_rate == 0.005  # original unchanged
