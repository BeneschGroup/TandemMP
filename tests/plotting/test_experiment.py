"""Tests for :mod:`tandemmp.plotting.experiment`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tandemmp.plotting.experiment import (  # noqa: E402
    error_vs_mass,
    events_cube,
    mass_histogram,
    mass_tablecloth,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# mass_histogram
# ---------------------------------------------------------------------------


class TestMassHistogram:
    def test_returns_axes(self) -> None:
        ax = mass_histogram(np.array([100.0, 200.0, 300.0]))
        assert ax is not None
        assert ax.get_xlabel() == "Mass (kDa)"
        assert ax.get_ylabel() == "Counts"

    def test_default_xlim_spans_data(self) -> None:
        ax = mass_histogram(np.array([50.0, 150.0, 250.0]))
        lo, hi = ax.get_xlim()
        assert lo == pytest.approx(50.0)
        assert hi == pytest.approx(250.0)

    def test_custom_xlim_respected(self) -> None:
        ax = mass_histogram(np.array([100.0, 200.0]), xlim=(0.0, 500.0))
        lo, hi = ax.get_xlim()
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(500.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            mass_histogram(np.array([]))

    def test_title_applied(self) -> None:
        ax = mass_histogram(np.array([1.0, 2.0]), title="custom")
        assert ax.get_title() == "custom"

    def test_saveas_writes_file(self, tmp_path) -> None:
        path = tmp_path / "hist.png"
        mass_histogram(np.array([1.0, 2.0]), saveas=str(path))
        assert path.exists()

    def test_reuses_given_axes(self) -> None:
        fig, ax = plt.subplots()
        out = mass_histogram(np.array([1.0, 2.0]), ax=ax)
        assert out is ax


# ---------------------------------------------------------------------------
# mass_tablecloth
# ---------------------------------------------------------------------------


class TestMassTablecloth:
    def test_returns_fig_and_three_axes(self) -> None:
        initial = np.array([100.0, 200.0])
        secondary = np.array([110.0, 210.0])
        fig, ax, axx, axy = mass_tablecloth(initial, secondary)
        assert fig is not None
        assert ax.get_xlabel() == "Initial mass (kDa)"
        assert ax.get_ylabel() == "Secondary mass (kDa)"

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            mass_tablecloth(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            mass_tablecloth(np.array([]), np.array([]))

    def test_default_xlim_encompasses_both_series(self) -> None:
        _, ax, _, _ = mass_tablecloth(np.array([50.0, 200.0]), np.array([100.0, 300.0]))
        lo, hi = ax.get_xlim()
        assert lo == pytest.approx(50.0)
        assert hi == pytest.approx(300.0)

    def test_color_array_accepted(self) -> None:
        mass_tablecloth(
            np.array([1.0, 2.0]),
            np.array([1.1, 2.1]),
            color=np.array([0.1, 0.2]),
        )


# ---------------------------------------------------------------------------
# error_vs_mass
# ---------------------------------------------------------------------------


class TestErrorVsMass:
    def test_returns_log_axes_by_default(self) -> None:
        masses = np.array([100.0, 200.0, 300.0])
        errors = np.array([1e-4, 2e-4, 3e-4])
        _, ax, _, _ = error_vs_mass(masses, errors)
        assert ax.get_yscale() == "log"

    def test_linear_yscale(self) -> None:
        masses = np.array([100.0, 200.0])
        errors = np.array([1e-4, 2e-4])
        _, ax, _, _ = error_vs_mass(masses, errors, log_error=False)
        assert ax.get_yscale() == "linear"

    def test_hist2d_mode(self) -> None:
        masses = np.linspace(100.0, 500.0, 50)
        errors = np.logspace(-4, -3, 50)
        _, ax, _, _ = error_vs_mass(masses, errors, hist2d=True, bins=10)
        assert ax.get_xlabel() == "Mass (kDa)"

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            error_vs_mass(np.array([1.0, 2.0]), np.array([1e-4]))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            error_vs_mass(np.array([]), np.array([]))

    def test_zero_error_raises(self) -> None:
        with pytest.raises(ValueError):
            error_vs_mass(np.array([1.0, 2.0]), np.array([1e-4, 0.0]))

    def test_negative_error_raises(self) -> None:
        with pytest.raises(ValueError):
            error_vs_mass(np.array([1.0, 2.0]), np.array([1e-4, -1e-4]))

    def test_title_applied(self) -> None:
        _, ax, _, _ = error_vs_mass(
            np.array([1.0, 2.0]),
            np.array([1e-4, 2e-4]),
            title="err vs mass",
        )
        assert ax.get_title() == "err vs mass"


# ---------------------------------------------------------------------------
# events_cube
# ---------------------------------------------------------------------------


class TestEventsCube:
    def test_basic_render(self) -> None:
        pos = np.array([[5.0, 6.0, 10.0], [15.0, 16.0, 20.0]])
        widths = np.array([[11.0, 11.0, 5.0], [11.0, 11.0, 5.0]])
        ax = events_cube(pos, widths=widths, movie_shape=(100, 50, 50))
        assert ax.name == "3d"

    def test_is_bad_colors(self) -> None:
        pos = np.array([[5.0, 6.0, 10.0]])
        widths = np.array([[11.0, 11.0, 5.0]])
        ax = events_cube(
            pos,
            widths=widths,
            is_bad=np.array([True]),
            movie_shape=(100, 50, 50),
        )
        # Rendering passes; we don't inspect specific colours here.
        assert ax is not None

    def test_frame_limit_filters(self) -> None:
        pos = np.array([[1.0, 1.0, 5.0], [1.0, 1.0, 50.0]])
        widths = np.array([[11.0, 11.0, 5.0], [11.0, 11.0, 5.0]])
        ax = events_cube(
            pos,
            widths=widths,
            movie_shape=(100, 40, 40),
            frame_limit=20,
        )
        _, zhi = ax.get_zlim()
        assert zhi == pytest.approx(20)

    def test_positions_shape_validation(self) -> None:
        with pytest.raises(ValueError):
            events_cube(
                np.array([1.0, 2.0]),
                widths=np.array([1.0, 2.0]),
                movie_shape=(10, 10, 10),
            )

    def test_widths_shape_validation(self) -> None:
        with pytest.raises(ValueError):
            events_cube(
                np.array([[1.0, 2.0, 3.0]]),
                widths=np.array([[1.0, 2.0]]),
                movie_shape=(10, 10, 10),
            )

    def test_no_events_still_draws(self) -> None:
        ax = events_cube(
            np.zeros((0, 3)),
            widths=np.zeros((0, 3)),
            movie_shape=(10, 10, 10),
        )
        assert ax is not None

    def test_zero_y_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            events_cube(
                np.zeros((0, 3)),
                widths=np.zeros((0, 3)),
                movie_shape=(10, 0, 10),
            )
