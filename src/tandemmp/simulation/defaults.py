"""Default simulation rates for the TandemMP Monte Carlo engine."""

from __future__ import annotations

from dataclasses import dataclass


def _validate_positive(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class SimulationDefaults:
    """Per-frame event probabilities for the simulation engine."""

    #: Probability of a new particle binding per frame.
    binding_rate: float = 0.005
    #: Probability of a bound particle unbinding per frame.
    unbinding_rate: float = 0.0001
    #: Probability of a bound particle diffusing per frame.
    diffusion_rate: float = 0.001

    def __post_init__(self) -> None:
        _validate_positive(self.binding_rate, "binding_rate")
        _validate_positive(self.unbinding_rate, "unbinding_rate")
        _validate_positive(self.diffusion_rate, "diffusion_rate")
