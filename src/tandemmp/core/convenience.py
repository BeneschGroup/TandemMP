"""Small convenience utilities for TandemMP."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

# Object repr helper


def nice_repr(
    obj: object,
    *,
    remove_underscores: bool = True,
    exclude: list[str] | None = None,
    decimals: int = 4,
) -> str:
    """Return a human-readable multi-line summary of *obj*'s attributes.

    Args:
        obj: Object whose ``__dict__`` is formatted.
        remove_underscores: Replace underscores with spaces.
        exclude: Attribute names to omit (default none).
        decimals: Decimal places for floats.
    """
    if exclude is None:
        exclude = []

    formatted = {
        key: round(value, decimals) if isinstance(value, float) else value
        for key, value in obj.__dict__.items()
    }

    text = "".join(
        f"{key} = {value}\n" for key, value in formatted.items() if key not in exclude
    )

    if remove_underscores:
        text = text.replace("_", " ")

    return text


# Random background crop generator


def movie_random_generator(
    movie: npt.NDArray[np.floating[Any]],
) -> Callable[[int], npt.NDArray[np.floating[Any]]]:
    """Return a closure that extracts a random crop from *movie*.

    *movie* has shape ``(frames, height, width)``. The callable takes an integer
    *l_thum* (thumbnail side length) and returns a ``2*(l_thum//2)+1``-square crop
    from a random frame and position, clamped to the movie boundaries.
    """
    n_frames, height, width = movie.shape

    def get_random_background(l_thum: int) -> npt.NDArray[np.floating[Any]]:
        half = l_thum // 2
        side = 2 * half + 1

        i = int(np.random.random() * n_frames)
        y = int(np.random.random() * height)
        x = int(np.random.random() * width)

        x_min = x - half
        x_max = x + half + 1
        y_min = y - half
        y_max = y + half + 1

        if x_min < 0:
            x_min, x_max = 0, side
        if y_min < 0:
            y_min, y_max = 0, side
        if x_max > width:
            x_min, x_max = width - side, width
        if y_max > height:
            y_min, y_max = height - side, height

        return movie[i, y_min:y_max, x_min:x_max]

    return get_random_background
