"""Tests for tandemmp.simulation.movement — target API."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tandemmp.simulation.movement import MovementVector


class TestMovementVector:
    def test_fields(self) -> None:
        mv = MovementVector(chance_per_frame=0.01, max_distance=5.0)
        assert mv.chance_per_frame == 0.01
        assert mv.max_distance == 5.0
        assert mv.bias is False

    def test_frozen(self) -> None:
        mv = MovementVector(chance_per_frame=0.01, max_distance=5.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            mv.max_distance = 10.0  # type: ignore[misc]

    def test_custom_bias(self) -> None:
        mv = MovementVector(chance_per_frame=0.01, max_distance=5.0, bias=True)
        assert mv.bias is True

    def test_validation_chance_per_frame(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MovementVector(chance_per_frame=-0.1, max_distance=1.0)

    def test_validation_max_distance(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MovementVector(chance_per_frame=0.01, max_distance=-1.0)

    def test_replace(self) -> None:
        mv = MovementVector(chance_per_frame=0.01, max_distance=5.0)
        mv2 = dataclasses.replace(mv, max_distance=10.0)
        assert mv2.max_distance == 10.0
        assert mv.max_distance == 5.0


class TestGenerate:
    def test_returns_two_floats(self) -> None:
        mv = MovementVector(chance_per_frame=0.01, max_distance=5.0)
        dx, dy = mv.generate()
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_within_max_distance(self) -> None:
        mv = MovementVector(chance_per_frame=0.01, max_distance=3.0)
        rng = np.random.default_rng(42)
        for _ in range(200):
            dx, dy = mv.generate(rng=rng)
            distance = np.sqrt(dx**2 + dy**2)
            assert distance <= mv.max_distance + 1e-10

    def test_biased_uses_centre_angle(self) -> None:
        mv = MovementVector(chance_per_frame=0.01, max_distance=5.0, bias=True)
        rng = np.random.default_rng(0)
        angles = []
        for _ in range(500):
            dx, dy = mv.generate(
                rng=rng,
                centre_angle_bias=np.pi / 2,
                standard_deviation=0.1,
            )
            angles.append(np.arctan2(dx, dy))
        # Mean angle should be near pi/2
        mean_angle = float(np.mean(angles))
        assert abs(mean_angle - np.pi / 2) < 0.3

    def test_unbiased_is_uniform(self) -> None:
        mv = MovementVector(chance_per_frame=0.01, max_distance=5.0)
        rng = np.random.default_rng(42)
        angles = []
        for _ in range(1000):
            dx, dy = mv.generate(rng=rng)
            angles.append(np.arctan2(dx, dy))
        # Mean angle should be near 0 for uniform distribution
        mean_angle = float(np.mean(angles))
        assert abs(mean_angle) < 0.3

    def test_deterministic_with_rng(self) -> None:
        mv = MovementVector(chance_per_frame=0.01, max_distance=5.0)
        dx1, dy1 = mv.generate(rng=np.random.default_rng(99))
        dx2, dy2 = mv.generate(rng=np.random.default_rng(99))
        assert dx1 == dx2
        assert dy1 == dy2
