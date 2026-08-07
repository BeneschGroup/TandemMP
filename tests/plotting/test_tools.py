"""Tests for tandemmp.plotting.tools — target API."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for CI

from tandemmp.plotting.tools import (
    add_gaussian_fit,
    easy_plot,
    scatter_hist,
)

# ---------------------------------------------------------------------------
# easy_plot
# ---------------------------------------------------------------------------


class TestEasyPlot:
    def test_creates_axes_when_not_provided(self) -> None:
        @easy_plot()
        def my_plot(*, axes=None):  # type: ignore[no-untyped-def]
            return axes

        ax = my_plot()
        assert ax is not None

    def test_uses_provided_axes(self) -> None:
        _, expected_ax = plt.subplots()

        @easy_plot()
        def my_plot(*, axes=None):  # type: ignore[no-untyped-def]
            return axes

        ax = my_plot(axes=expected_ax)
        assert ax is expected_ax

    def test_creates_grid(self) -> None:
        @easy_plot(nx=2, ny=3)
        def my_plot(*, axes=None):  # type: ignore[no-untyped-def]
            return axes

        axes = my_plot()
        assert axes.shape == (3, 2)

    def test_passes_args_through(self) -> None:
        @easy_plot()
        def my_plot(x, y, *, axes=None):  # type: ignore[no-untyped-def]
            return x + y

        assert my_plot(3, 4) == 7

    def test_passes_kwargs_through(self) -> None:
        @easy_plot()
        def my_plot(*, value=10, axes=None):  # type: ignore[no-untyped-def]
            return value

        assert my_plot(value=42) == 42

    def test_preserves_function_name(self) -> None:
        @easy_plot()
        def my_special_plot(*, axes=None):  # type: ignore[no-untyped-def]
            pass

        assert my_special_plot.__name__ == "my_special_plot"


# ---------------------------------------------------------------------------
# scatter_hist
# ---------------------------------------------------------------------------


class TestScatterHist:
    def test_returns_three_axes(self) -> None:
        ax, ax_histx, ax_histy = scatter_hist()
        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(ax_histx, matplotlib.axes.Axes)
        assert isinstance(ax_histy, matplotlib.axes.Axes)

    def test_passes_kwargs_to_figure(self) -> None:
        ax, _, _ = scatter_hist(figsize=(10, 10), dpi=100)
        fig = ax.get_figure()
        assert fig is not None
        assert fig.get_dpi() == 100

    def test_custom_margins(self) -> None:
        ax, ax_histx, ax_histy = scatter_hist(margin=0.12)
        assert ax is not None
        assert ax_histx is not None
        assert ax_histy is not None

    def test_custom_spacing(self) -> None:
        ax, _, _ = scatter_hist(spacing=0.05, scatter_size=0.5)
        assert ax is not None


# ---------------------------------------------------------------------------
# add_gaussian_fit
# ---------------------------------------------------------------------------


class TestAddGaussianFit:
    def test_adds_line_to_axes(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 1000)
        bins = np.linspace(30, 70, 50)

        _, ax = plt.subplots()
        ax.hist(data, bins=bins)
        n_lines_before = len(ax.get_lines())

        add_gaussian_fit(ax, data, bins)
        assert len(ax.get_lines()) > n_lines_before

    def test_with_annotation(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 1000)
        bins = np.linspace(30, 70, 50)

        _, ax = plt.subplots()
        ax.hist(data, bins=bins)

        add_gaussian_fit(ax, data, bins, annotate=True)
        assert len(ax.texts) > 0

    def test_custom_guesses(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, 2000)
        bins = np.linspace(60, 140, 80)

        _, ax = plt.subplots()
        ax.hist(data, bins=bins)

        add_gaussian_fit(
            ax,
            data,
            bins,
            mean_guess=100.0,
            amplitude_guess=200.0,
            sigma_guess=10.0,
        )

    def test_custom_color(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        bins = np.linspace(-4, 4, 30)

        _, ax = plt.subplots()
        ax.hist(data, bins=bins)

        add_gaussian_fit(ax, data, bins, color="red")
        line = ax.get_lines()[-1]
        assert line.get_color() == "red"

    def test_recovers_positive_amplitude(self, monkeypatch) -> None:
        """Histogram counts are non-negative, so the raw fitted amplitude
        should land in the positive basin of the ``abs(a0)``-symmetric
        Gaussian model even though a negative-amplitude optimum is now also
        mathematically valid. Asserts on the raw fit result rather than the
        plotted curve, since ``gaussian()``'s ``abs(a0)`` would make the
        plotted curve non-negative regardless of the fit's actual sign.
        """
        import tandemmp.plotting.tools as tools_module

        rng = np.random.default_rng(1)
        data = rng.normal(50, 5, 1000)
        bins = np.linspace(30, 70, 50)

        _, ax = plt.subplots()
        ax.hist(data, bins=bins)

        original_curve_fit_model = tools_module.curve_fit_model
        captured = {}

        def _capturing_curve_fit_model(*args, **kwargs):
            result = original_curve_fit_model(*args, **kwargs)
            captured["params"] = result.params
            return result

        monkeypatch.setattr(tools_module, "curve_fit_model", _capturing_curve_fit_model)

        add_gaussian_fit(ax, data, bins)
        assert captured["params"][0] > 0

    def test_annotation_contains_mass(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 1000)
        bins = np.linspace(30, 70, 50)

        _, ax = plt.subplots()
        ax.hist(data, bins=bins)

        add_gaussian_fit(ax, data, bins, annotate=True)
        text = ax.texts[0].get_text()
        assert "Mass" in text
        assert "Sigma" in text
        assert "Counts" in text
