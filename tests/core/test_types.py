"""Tests for tandemmp.core.types — shared NumPy array type aliases."""

from __future__ import annotations

import numpy as np

from tandemmp.core.types import BoolArray, FloatArray, IntArray, UInt16Array


class TestFloatArray:
    def test_accepts_float64_array(self) -> None:
        arr: FloatArray = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert arr.dtype == np.float64

    def test_accepts_float32_array(self) -> None:
        arr: FloatArray = np.array([1.0, 2.0], dtype=np.float32)
        assert arr.dtype == np.float32


class TestBoolArray:
    def test_accepts_bool_array(self) -> None:
        arr: BoolArray = np.array([True, False, True])
        assert arr.dtype == np.bool_


class TestIntArray:
    def test_accepts_int64_array(self) -> None:
        arr: IntArray = np.array([1, 2, 3], dtype=np.int64)
        assert arr.dtype == np.int64

    def test_accepts_int32_array(self) -> None:
        arr: IntArray = np.array([1, 2], dtype=np.int32)
        assert arr.dtype == np.int32


class TestUInt16Array:
    def test_accepts_uint16_array(self) -> None:
        arr: UInt16Array = np.array([1, 2, 3], dtype=np.uint16)
        assert arr.dtype == np.uint16
