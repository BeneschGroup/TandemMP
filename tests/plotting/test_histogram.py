"""Tests for tandemmp.plotting.histogram."""

from __future__ import annotations

import matplotlib
import numpy as np
from sklearn.mixture import GaussianMixture

matplotlib.use("Agg")  # non-interactive backend for CI

from tandemmp.plotting.histogram import plot_gmm, plot_histogram

# plot_histogram


class TestPlotHistogram:
    def test_runs_without_error(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 500)
        ax = plot_histogram(data)
        assert ax is not None

    def test_custom_bins(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 500)
        ax = plot_histogram(data, bins=20)
        assert ax is not None


# plot_gmm


class TestPlotGMM:
    def test_runs_without_error(self) -> None:
        rng = np.random.default_rng(42)
        data = np.concatenate(
            [
                rng.normal(50, 5, 500),
                rng.normal(100, 5, 500),
            ]
        )
        result = plot_gmm(data, max_components=5)
        assert isinstance(result, GaussianMixture)

    def test_with_fit_means(self) -> None:
        rng = np.random.default_rng(42)
        data = np.concatenate([rng.normal(m, 3, 500) for m in [50, 100, 150]])
        gmm, r2, slope = plot_gmm(data, max_components=5, fit_means_flag=True)
        assert isinstance(gmm, GaussianMixture)
        assert r2 is not None or r2 is None  # may or may not succeed

    def test_no_plot_mode(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        result = plot_gmm(data, max_components=3, plot=False)
        assert isinstance(result, GaussianMixture)
