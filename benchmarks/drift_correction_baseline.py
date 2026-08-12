"""Benchmark for tandemmp.pipeline.drift_correction.run_drift_correction.

Times drift-vector estimation against a real ``.mp`` movie. Used to establish
the pre-Rust-port Python baseline (WORKLIST.md priority-6 target: the
``differential_evolution`` shift/RMSE inner loop in
:mod:`tandemmp.drift.corrector`) and to re-measure after the Rust port lands,
for a like-for-like comparison.

Usage::

    python benchmarks/drift_correction_baseline.py <path-to-mp-file>

Requires ``open_iscat`` (movie loading) in addition to the project's normal
dev dependencies — not part of the pytest suite or CI.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from tandemmp.pipeline.drift_correction import run_drift_correction

SEED = 42
FRAME_GAP = 25


def bench(mp_file: str, *, parallel: bool, n_jobs: int | None = None) -> dict:
    t0 = time.perf_counter()
    result = run_drift_correction(
        mp_file,
        frame_gap=FRAME_GAP,
        parallel=parallel,
        n_jobs=n_jobs,
        seed=SEED,
    )
    elapsed = time.perf_counter() - t0
    return {
        "parallel": parallel,
        "n_jobs": n_jobs,
        "elapsed_seconds": elapsed,
        "vector_len": len(result.vector),
    }


def main() -> None:
    mp_file = sys.argv[1] if len(sys.argv) > 1 else None
    if not mp_file:
        raise SystemExit("usage: drift_correction_baseline.py <path-to-mp-file>")

    out = {
        "mp_file": mp_file,
        "frame_gap": FRAME_GAP,
        "seed": SEED,
        "serial": bench(mp_file, parallel=False),
        "parallel": bench(mp_file, parallel=True),
    }
    print(json.dumps(out, indent=2))

    dest = Path(__file__).with_name("results.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"Saved to {dest}")


if __name__ == "__main__":
    main()
