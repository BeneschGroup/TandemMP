"""Tests for tandemmp.core.histogram — target API."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from tandemmp.core.histogram import (
    duplicate_inliers,
    fit_gaussian_to_histogram,
    fit_gmm,
    fit_means,
    scott_bin_count,
)

# scott_bin_count


class TestScottBinCount:
    def test_returns_positive_int(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 1000)
        n = scott_bin_count(data)
        assert isinstance(n, int)
        assert n > 0

    def test_constant_data_returns_one(self) -> None:
        data = np.full(100, 5.0)
        assert scott_bin_count(data) == 1

    def test_more_data_gives_more_bins(self) -> None:
        rng = np.random.default_rng(42)
        small = rng.normal(0, 1, 100)
        large = rng.normal(0, 1, 10_000)
        # More data → more bins (Scott's rule: n^(1/3) in denominator)
        assert scott_bin_count(large) > scott_bin_count(small)


# fit_gaussian_to_histogram


class TestFitGaussianToHistogram:
    def test_recovers_known_parameters(self) -> None:
        rng = np.random.default_rng(42)
        true_mean, true_sigma = 50.0, 5.0
        data = rng.normal(true_mean, true_sigma, 10_000)

        amplitude, mean, sigma = fit_gaussian_to_histogram(data)
        assert mean == pytest.approx(true_mean, abs=1.0)
        assert abs(sigma) == pytest.approx(true_sigma, abs=1.0)

    def test_raises_on_nan(self) -> None:
        data = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="NaN"):
            fit_gaussian_to_histogram(data)

    def test_custom_bins(self) -> None:
        rng = np.random.default_rng(7)
        data = rng.normal(0, 1, 5000)
        amp, mean, sigma = fit_gaussian_to_histogram(data, bins=50)
        assert mean == pytest.approx(0.0, abs=0.5)

    def test_recovers_positive_amplitude(self) -> None:
        """Histogram counts are non-negative, so the fitted amplitude should
        land in the positive basin of the ``abs(a0)``-symmetric model even
        though a negative-amplitude optimum is now also mathematically valid.
        """
        rng = np.random.default_rng(3)
        data = rng.normal(20.0, 4.0, 10_000)

        amplitude, _, _ = fit_gaussian_to_histogram(data)
        assert amplitude > 0


# fit_gmm


class TestFitGMM:
    def test_single_cluster(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(100, 5, 1000)
        gmm = fit_gmm(data, max_components=5)
        assert isinstance(gmm, GaussianMixture)
        assert gmm.n_components == 1

    def test_two_clusters(self) -> None:
        rng = np.random.default_rng(42)
        data = np.concatenate(
            [
                rng.normal(50, 3, 1000),
                rng.normal(150, 3, 1000),
            ]
        )
        gmm = fit_gmm(data, max_components=5)
        assert gmm.n_components == 2

    def test_criterion_aic(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 500)
        gmm = fit_gmm(data, max_components=3, criterion="aic")
        assert isinstance(gmm, GaussianMixture)

    def test_invalid_criterion_raises(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="criterion"):
            fit_gmm(data, max_components=3, criterion="invalid")

    def test_xlim_filters_data(self) -> None:
        rng = np.random.default_rng(0)
        data = np.concatenate(
            [
                rng.normal(0, 1, 500),
                rng.normal(100, 1, 500),
            ]
        )
        gmm = fit_gmm(data, max_components=5, xlim=(90, 110))
        # With xlim filtering to one cluster, should find 1 component
        assert gmm.n_components == 1


# duplicate_inliers


class TestDuplicateInliers:
    def test_output_longer_than_input(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 200)
        result = duplicate_inliers(data, duplicates=2)
        assert len(result) > len(data)

    def test_cut_mode_returns_inliers_only(self) -> None:
        rng = np.random.default_rng(0)
        # Tight cluster + far outlier
        data = np.concatenate([rng.normal(0, 0.01, 100), [100.0]])
        result = duplicate_inliers(data, cut=True)
        assert len(result) <= len(data)
        assert 100.0 not in result

    def test_accepts_list_input(self) -> None:
        data = [1.0, 1.1, 1.2, 1.0, 0.9, 1.05, 0.95, 1.1, 50.0]
        result = duplicate_inliers(data)
        assert isinstance(result, np.ndarray)


# fit_means


class TestFitMeans:
    def test_linear_means(self) -> None:
        """GMM with evenly spaced means → high r² and correct slope."""
        rng = np.random.default_rng(42)
        data = np.concatenate([rng.normal(m, 3, 500) for m in [50, 100, 150, 200]])
        gmm = fit_gmm(data, max_components=6)
        r2, slope = fit_means(gmm)
        assert r2 is not None
        assert r2 > 0.99
        assert slope == pytest.approx(50.0, abs=5.0)

    def test_single_component_returns_none(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(100, 5, 1000)
        gmm = fit_gmm(data, max_components=3)
        r2, slope = fit_means(gmm)
        assert r2 is None
        assert slope is None
