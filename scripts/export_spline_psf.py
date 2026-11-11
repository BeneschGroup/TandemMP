"""Export a pickled ``RectBivariateSpline`` PSF to the Rust crate's data files.

Usage: ``python scripts/export_spline_psf.py <psf.pickle>``

Writes little-endian f64 ``tx.bin``, ``ty.bin`` and ``c.bin`` to
``src/tandemmp/_rust/spline_psf/data/``. Only unpickle trusted files.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BivariateSpline

OUT = Path(__file__).resolve().parents[1] / "src/tandemmp/_rust/spline_psf/data"


def main(path: str) -> None:
    with open(path, "rb") as f:
        spline = pickle.load(f)  # noqa: S301
    if tuple(spline.degrees) != (3, 3):
        msg = f"expected a bicubic spline, got degrees {spline.degrees}"
        raise ValueError(msg)
    tx, ty, c = (np.ascontiguousarray(a, dtype="<f8") for a in spline.tck)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, arr in (("tx", tx), ("ty", ty), ("c", c)):
        arr.tofile(OUT / f"{name}.bin")

    # Round-trip check
    back = [np.fromfile(OUT / f"{n}.bin", dtype="<f8") for n in ("tx", "ty", "c")]
    rebuilt = BivariateSpline._from_tck((*back, 3, 3))  # noqa: SLF001
    g = np.linspace(tx[0], tx[-1], 101)
    err = np.abs(rebuilt(g, g) - spline(g, g)).max()
    print(f"wrote {OUT} (tx {tx.size}, ty {ty.size}, c {c.size}); max err {err:.1e}")


if __name__ == "__main__":
    main(sys.argv[1])
