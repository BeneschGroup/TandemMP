#!/usr/bin/env python3
"""Smoke test an *installed* tandemmp wheel: import and call both extensions.

Importing is not enough to catch ABI or linking problems (a mismatched stable-ABI
build can import fine and crash on the first call), so this calls into each
extension. Run it from outside the source checkout so the installed wheel, not
``src/``, is what gets imported. Needs only numpy.
"""

from __future__ import annotations

import sys

import numpy as np

import tandemmp
from tandemmp._rust import drift_correction as dc
from tandemmp._rust import spline_psf as sp


def main() -> None:
    assert "site-packages" in tandemmp.__file__, (
        f"tandemmp imported from {tandemmp.__file__}, not an installed wheel"
    )

    # spline_psf: bundled spline data decodes and evaluates.
    tx, ty, c, kx, ky = sp.tck()
    assert tx.size and ty.size and c.size and (kx, ky) == (3, 3), (kx, ky)
    u = np.linspace(-2.0, 2.0, 9)
    out = sp.eval(u, u.copy())
    assert out.shape == u.shape and np.isfinite(out).all(), out

    # drift_correction: every entry point runs on a small frame.
    frame = np.random.default_rng(0).random((16, 16))
    assert dc.rmse(frame, frame) == 0.0
    assert dc.shift_compare(0.0, 0.0, frame, frame) < 1e-9
    assert dc.stabilise(frame, 0.0, 0.0).shape == frame.shape
    x, y = dc.solve(frame, frame, seed=0)
    assert abs(x) < 0.1 and abs(y) < 0.1, (x, y)

    print(f"tandemmp {tandemmp.__version__}: both extensions OK ({sys.platform})")


if __name__ == "__main__":
    main()
