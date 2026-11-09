"""Tests for tandemmp.core.convenience — target API."""

import numpy as np

from tandemmp.core.convenience import (
    movie_random_generator,
    nice_repr,
)

# nice_repr


class _FakeObj:
    """Minimal object with a __dict__ for testing nice_repr."""

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestNiceRepr:
    def test_basic(self) -> None:
        obj = _FakeObj(name="alice", score=3.14159)
        text = nice_repr(obj)
        assert "name = alice" in text
        assert "score = 3.1416" in text  # rounded to 4 decimals

    def test_removes_underscores_by_default(self) -> None:
        obj = _FakeObj(my_attr=1)
        text = nice_repr(obj)
        assert "my attr" in text
        assert "my_attr" not in text

    def test_keeps_underscores_when_disabled(self) -> None:
        obj = _FakeObj(my_attr=1)
        text = nice_repr(obj, remove_underscores=False)
        assert "my_attr" in text

    def test_exclude_fields(self) -> None:
        obj = _FakeObj(a=1, b=2, c=3)
        text = nice_repr(obj, exclude=["b"])
        assert "a = 1" in text
        assert "c = 3" in text
        assert "b" not in text

    def test_custom_decimals(self) -> None:
        obj = _FakeObj(val=1.23456789)
        text = nice_repr(obj, decimals=2)
        assert "val = 1.23" in text

    def test_empty_object(self) -> None:
        obj = _FakeObj()
        text = nice_repr(obj)
        assert text == ""

    def test_non_float_values_not_rounded(self) -> None:
        obj = _FakeObj(count=42, name="test")
        text = nice_repr(obj)
        assert "count = 42" in text
        assert "name = test" in text

    def test_default_exclude_is_empty(self) -> None:
        """Ensure no mutable-default-argument bug: calling with no exclude
        should not carry state between calls."""
        obj = _FakeObj(x=1)
        text1 = nice_repr(obj)
        text2 = nice_repr(obj)
        assert text1 == text2


# movie_random_generator


class TestMovieRandomGenerator:
    def test_returns_callable(self) -> None:
        movie = np.zeros((10, 50, 50))
        gen = movie_random_generator(movie)
        assert callable(gen)

    def test_crop_shape(self) -> None:
        movie = np.random.default_rng(42).random((10, 50, 50))
        gen = movie_random_generator(movie)
        crop = gen(l_thum=11)
        assert crop.shape == (11, 11)

    def test_crop_shape_even(self) -> None:
        """Even l_thum: width = l_thum // 2, side = 2*width + 1."""
        movie = np.random.default_rng(42).random((10, 50, 50))
        gen = movie_random_generator(movie)
        crop = gen(l_thum=10)
        expected_side = 2 * (10 // 2) + 1  # 11
        assert crop.shape == (expected_side, expected_side)

    def test_crop_values_come_from_movie(self) -> None:
        rng = np.random.default_rng(99)
        movie = rng.random((5, 40, 40))
        gen = movie_random_generator(movie)
        crop = gen(l_thum=7)
        # Every value in the crop must exist in the movie
        assert crop.min() >= movie.min()
        assert crop.max() <= movie.max()

    def test_small_movie_boundary(self) -> None:
        """Crop should still work when the movie is barely large enough."""
        movie = np.ones((1, 7, 7))
        gen = movie_random_generator(movie)
        crop = gen(l_thum=7)
        assert crop.shape == (7, 7)

    def test_different_calls_may_differ(self) -> None:
        """Two calls should (with high probability) return different crops."""
        rng = np.random.default_rng(0)
        movie = rng.random((100, 100, 100))
        gen = movie_random_generator(movie)
        c1 = gen(l_thum=5)
        c2 = gen(l_thum=5)
        # Very unlikely to be identical from a 100-frame random movie
        assert not np.array_equal(c1, c2)
