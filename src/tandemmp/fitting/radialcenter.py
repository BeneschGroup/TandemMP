# Adapted from: radialcenter.m
#
# Copyright 2011-2012, Raghuveer Parthasarathy, The University of Oregon
#
##
# Disclaimer / License
#   This program is free software: you can redistribute it and/or
#     modify it under the terms of the GNU General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#   This set of programs is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#   General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   (gpl.txt) along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""Radial-symmetry centre estimation for a 2-D intensity peak.

Adapted from ``radialcenter.m``. See:
Parthasarathy, R. "Rapid, accurate particle tracking by calculation of
radial symmetry centers", Nature Methods 9, 724-726 (2012).

Considers lines through each half-pixel grid point with slope parallel to
the local intensity gradient, and finds the point minimising the weighted
sum of squared distances from those lines — the radial symmetry centre.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.signal


def _ls_radial_center_fit(
    m: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
    w: npt.NDArray[np.floating],
) -> tuple[float, float]:
    """Least-squares radial symmetry centre from gradient lines ``y = mx + b``."""
    wm2p1 = w / (m**2 + 1)
    sw = wm2p1.sum()
    smmw = (m**2 * wm2p1).sum()
    smw = (m * wm2p1).sum()
    smbw = (m * b * wm2p1).sum()
    sbw = (b * wm2p1).sum()
    det = smw**2 - smmw * sw
    xc = (smbw * sw - smw * sbw) / det
    yc = (smbw * smw - smmw * sbw) / det
    return xc, yc


def radialcenter(
    image: npt.NDArray[np.floating], mode: Literal["abs", "rel"] = "abs"
) -> tuple[float, float, float]:
    """Return ``(xc, yc, sigma)``, the radial symmetry centre of *image*.

    *image* must be square with an odd edge length. In ``mode="abs"``, the
    centre is relative to the upper-left pixel (pixel #1); in ``mode="rel"``,
    it is relative to the image centre. *sigma* is a rough width estimate
    (square root of the second moment), not used in the centre fit itself.
    """
    n_y, n_x = image.shape
    if n_y % 2 == 0 or n_x % 2 == 0 or n_x != n_y:
        msg = "radialcenter requires a square image with an odd edge length"
        raise ValueError(msg)
    n = (n_x - 1) // 2

    # Grid midpoint coordinates, offset by half a pixel from the grid points.
    mid = np.linspace(-(n - 0.5), n - 0.5, 2 * n)
    x_mid, y_mid = np.meshgrid(mid, mid)

    # Derivatives along 45-degree-shifted (u, v) coordinates. y increases
    # "downward" (row number); this is accounted for in the slope below.
    d_i_du = image[:-1, 1:] - image[1:, :-1]
    d_i_dv = image[:-1, :-1] - image[1:, 1:]

    # 3x3 smoothing.
    kernel = np.ones((3, 3)) / 9.0
    f_du = scipy.signal.convolve2d(d_i_du, kernel, mode="same")
    f_dv = scipy.signal.convolve2d(d_i_dv, kernel, mode="same")
    grad_mag_sq = f_du**2 + f_dv**2

    # Gradient slope in x-y coordinates (45-degree rotation of u, v; the
    # negative sign accounts for y increasing "downward").
    slope = -(f_dv + f_du) / (f_du - f_dv)

    nan_pixels = np.isnan(slope)
    if nan_pixels.any():
        unsmoothed = (d_i_dv + d_i_du) / (d_i_du - d_i_dv)
        slope[nan_pixels] = unsmoothed[nan_pixels]
        nan_pixels = np.isnan(slope)
        slope[nan_pixels] = 0.0

    # Replace infinite slopes (equal smoothed u/v derivatives) with a large
    # finite value; sign doesn't matter.
    inf_pixels = np.isinf(slope)
    if inf_pixels.any():
        if inf_pixels.sum() < slope.size:
            slope[inf_pixels] = 10 * slope[~inf_pixels].max()
        else:
            slope = (d_i_dv + d_i_du) / (d_i_du - d_i_dv)

    # y-intercept of the gradient line through each grid midpoint.
    intercept = y_mid - slope * x_mid

    # Weight by squared gradient magnitude and inverse distance to the
    # gradient-magnitude centroid.
    total_grad = grad_mag_sq.sum()
    x_centroid = (grad_mag_sq * x_mid).sum() / total_grad
    y_centroid = (grad_mag_sq * y_mid).sum() / total_grad
    weight = grad_mag_sq / np.sqrt(
        (x_mid - x_centroid) ** 2 + (y_mid - y_centroid) ** 2
    )

    xc, yc = _ls_radial_center_fit(slope, intercept, weight)

    if mode == "abs":
        xc += (n_x + 1) / 2.0
        yc += (n_y + 1) / 2.0

    # Rough width estimate; not used by the centre fit.
    sub = image - image.min()
    px, py = np.meshgrid(np.arange(n_x), np.arange(n_y))
    r_sq = (px - xc) ** 2 + (py - yc) ** 2
    sigma = float(np.sqrt((sub * r_sq).sum() / sub.sum()) / 2.0)

    return float(xc), float(yc), sigma
