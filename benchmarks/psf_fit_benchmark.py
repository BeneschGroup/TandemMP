"""Speed benchmark: Rust spline PSF fit (scalar + batch) vs Python/SciPy BFGS.

Run: python benchmarks/psf_fit_benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

from tandemmp.fitting.psf import PsfModel, get_psf_model

N_PEAKS = 500
L_THUM = 21


def make_peaks(model: PsfModel, n: int, l_thum: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    peaks = []
    for _ in range(n):
        dx, dy = rng.uniform(-1, 1, size=2)
        c = rng.uniform(-0.1, -0.02)
        peak = model.psf_2d_offset(l_thum, dx, dy, c)
        peak = peak + rng.normal(scale=0.002, size=peak.shape)
        peaks.append(peak)
    return peaks


def bench_python(model: PsfModel, peaks: list[np.ndarray]) -> float:
    start = time.perf_counter()
    for peak in peaks:
        model.fit_peak(peak, rust=False)
    return time.perf_counter() - start


def bench_rust_scalar(model: PsfModel, peaks: list[np.ndarray]) -> float:
    start = time.perf_counter()
    for peak in peaks:
        model.fit_peak(peak, rust=True)
    return time.perf_counter() - start


def bench_rust_batch(model: PsfModel, peaks: list[np.ndarray]) -> float:
    start = time.perf_counter()
    model.fit_peaks_batch(peaks)
    return time.perf_counter() - start


def main() -> None:
    model = get_psf_model()
    print(f"l_thum={L_THUM}, n_peaks={N_PEAKS}")  # noqa: T201
    peaks = make_peaks(model, N_PEAKS, L_THUM)

    def report(label: str, t: float) -> None:
        print(f"{label:<20} {t:.4f} s  ({t / N_PEAKS * 1e3:.4f} ms/peak)")  # noqa: T201

    t_batch = bench_rust_batch(model, peaks)
    report("Rust batch (rayon):", t_batch)

    t_rust = bench_rust_scalar(model, peaks)
    report("Rust scalar:", t_rust)

    t_python = bench_python(model, peaks)
    report("Python/SciPy BFGS:", t_python)

    print(f"speedup (batch vs python): {t_python / t_batch:.1f}x")  # noqa: T201
    print(f"speedup (rust scalar vs python): {t_python / t_rust:.1f}x")  # noqa: T201


if __name__ == "__main__":
    main()
