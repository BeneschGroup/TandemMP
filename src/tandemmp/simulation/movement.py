"""Movement model definitions for the TandemMP simulation engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_positive(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class MovementVector:
    """Random displacement generator for simulated particle motion.

    Parameters
    ----------
    chance_per_frame:
        Probability of this movement occurring on a given frame.
    max_distance:
        Maximum displacement magnitude (pixels).
    bias:
        If ``True``, :meth:`generate` uses a normally distributed angle
        centred on *centre_angle_bias* instead of a uniform random angle.
    """

    chance_per_frame: float
    max_distance: float
    bias: bool = False

    def __post_init__(self) -> None:
        _validate_positive(self.chance_per_frame, "chance_per_frame")
        _validate_positive(self.max_distance, "max_distance")

    def generate(
        self,
        *,
        centre_angle_bias: float | None = None,
        standard_deviation: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, float]:
        """Return a random ``(dx, dy)`` displacement.

        Parameters
        ----------
        centre_angle_bias:
            Centre angle (radians) for biased movement.  Required when
            *bias* is ``True``.
        standard_deviation:
            Standard deviation of the angle distribution (radians).
            Required when *bias* is ``True``.
        rng:
            NumPy random generator.  Uses a module-level default when
            ``None``.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.bias:
            if centre_angle_bias is None or standard_deviation is None:
                msg = (
                    "centre_angle_bias and standard_deviation are required "
                    "when bias is True"
                )
                raise ValueError(msg)
            angle = rng.normal(loc=centre_angle_bias, scale=standard_deviation)
        else:
            angle = rng.random() * 2 * np.pi

        distance = rng.random() * self.max_distance
        dx = float(distance * np.sin(angle))
        dy = float(distance * np.cos(angle))
        return dx, dy
