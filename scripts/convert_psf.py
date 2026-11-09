"""One-time conversion: ml_mp_movements' pickled experimental PSF -> npz.

Not run by TandemMP at runtime (no ml_mp_movements/rust_mp_simulator import
anywhere in the package) — kept for provenance of
``src/tandemmp/resources/exp_psf_20260423_mfp2.npz``. Requires
ml_mp_movements and scipy installed locally to unpickle the source object.

Usage: python scripts/convert_psf.py <source.pickle> <dest.npz>
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
from scipy.interpolate import BivariateSpline


def convert(src: str, dst: str) -> None:
    with open(src, "rb") as f:  # noqa: PTH123
        obj = pickle.load(f)  # noqa: S301

    tx, ty, c = obj.tck
    kx, ky = obj.degrees if hasattr(obj, "degrees") else (3, 3)

    np.savez(
        dst,
        tx=tx,
        ty=ty,
        c=c,
        kx=kx,
        ky=ky,
        source=f"{src} (ml_mp_movements RectBivariateSpline)",
    )

    # Round-trip check against the original pickled spline.
    x = np.linspace(tx[0], tx[-1], 7)
    ref = obj(x, x)
    reloaded = np.load(dst)
    s = BivariateSpline._from_tck(  # noqa: SLF001
        (
            reloaded["tx"],
            reloaded["ty"],
            reloaded["c"],
            int(reloaded["kx"]),
            int(reloaded["ky"]),
        )
    )
    print(  # noqa: T201
        "max eval diff after round trip:",
        np.abs(s(x, x) - ref).max(),
        "value range:",
        ref.min(),
        ref.max(),
    )


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
