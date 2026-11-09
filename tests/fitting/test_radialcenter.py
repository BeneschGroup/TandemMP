"""Tests for tandemmp.fitting.radialcenter."""

from __future__ import annotations

import numpy as np
import pytest
from tandemmp.fitting.radialcenter import radialcenter


def _gaussian_peak(
    size: int, cx: float, cy: float, sigma: float = 3.0
) -> np.ndarray:
    """A symmetric 2-D Gaussian on a *size*-by-*size* grid, peak at (cx, cy)."""
    y, x = np.mgrid[0:size, 0:size]
    return np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)))


class TestRadialcenter:
    def test_non_square_raises(self) -> None:
        image = np.zeros((11, 13))
        with pytest.raises(ValueError, match="square"):
            radialcenter(image)

    def test_even_size_raises(self) -> None:
        image = np.zeros((10, 10))
        with pytest.raises(ValueError, match="odd"):
            radialcenter(image)

    def test_centred_peak_abs_mode(self) -> None:
        """A peak centred on the grid midpoint recovers pixel #(N+1)/2."""
        size = 21
        image = _gaussian_peak(size, cx=10.0, cy=10.0)
        xc, yc, _ = radialcenter(image, mode="abs")
        assert xc == pytest.approx((size + 1) / 2, abs=0.02)
        assert yc == pytest.approx((size + 1) / 2, abs=0.02)

    def test_centred_peak_rel_mode(self) -> None:
        size = 21
        image = _gaussian_peak(size, cx=10.0, cy=10.0)
        xc, yc, _ = radialcenter(image, mode="rel")
        assert xc == pytest.approx(0.0, abs=0.02)
        assert yc == pytest.approx(0.0, abs=0.02)

    def test_recovers_known_subpixel_offset(self) -> None:
        """A peak offset by a known sub-pixel amount is recovered in rel mode."""
        size = 21
        dx_true, dy_true = 0.3, -0.3
        image = _gaussian_peak(size, cx=10.0 + dx_true, cy=10.0 + dy_true)
        dx, dy, _ = radialcenter(image, mode="rel")
        assert dx == pytest.approx(dx_true, abs=0.02)
        assert dy == pytest.approx(dy_true, abs=0.02)

    def test_sigma_is_positive(self) -> None:
        image = _gaussian_peak(21, cx=10.0, cy=10.0, sigma=3.0)
        _, _, sigma = radialcenter(image, mode="rel")
        assert sigma > 0

    def test_abs_and_rel_differ_by_image_centre(self) -> None:
        size = 15
        image = _gaussian_peak(size, cx=7.5, cy=6.5)
        xc_abs, yc_abs, _ = radialcenter(image, mode="abs")
        xc_rel, yc_rel, _ = radialcenter(image, mode="rel")
        assert xc_abs == pytest.approx(xc_rel + (size + 1) / 2, abs=1e-9)
        assert yc_abs == pytest.approx(yc_rel + (size + 1) / 2, abs=1e-9)
