"""Tests for tandemmp.core.classifications — target API."""

import enum

from tandemmp.core.classifications import (
    Classification,
    EvClassification,
    ProcessingMethod,
    classify_mass_change,
)

# ---------------------------------------------------------------------------
# ProcessingMethod
# ---------------------------------------------------------------------------


class TestProcessingMethod:
    def test_is_int_flag(self):
        assert issubclass(ProcessingMethod, enum.IntFlag)

    def test_members(self):
        assert hasattr(ProcessingMethod, "ratiometric")
        assert hasattr(ProcessingMethod, "subtraction")
        assert hasattr(ProcessingMethod, "inferred")

    def test_composable(self):
        combo = ProcessingMethod.ratiometric | ProcessingMethod.subtraction
        assert ProcessingMethod.ratiometric in combo
        assert ProcessingMethod.subtraction in combo
        assert ProcessingMethod.inferred not in combo


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_fields(self):
        c = Classification(
            presence_before=True,
            presence_after=False,
            mass_change=-1,
            number_change=-1,
            rep="Unbinding",
        )
        assert c.presence_before is True
        assert c.presence_after is False
        assert c.mass_change == -1
        assert c.number_change == -1
        assert c.rep == "Unbinding"

    def test_repr(self):
        c = Classification(False, True, 1, 1, "Binding")
        assert repr(c) == "Binding"

    def test_get_slice_binding(self):
        """Binding: not present before, present after → second half."""
        c = Classification(False, True, 1, 1, "Binding")
        navg = 5
        s = c.get_slice(navg, remove=0)
        assert s == slice(5, None)

    def test_get_slice_binding_with_remove(self):
        c = Classification(False, True, 1, 1, "Binding")
        s = c.get_slice(5, remove=1)
        assert s == slice(6, None)

    def test_get_slice_unbinding(self):
        """Unbinding: present before, not after → first half."""
        c = Classification(True, False, -1, -1, "Unbinding")
        s = c.get_slice(5, remove=0)
        assert s == slice(None, 6)

    def test_get_slice_unbinding_with_remove(self):
        c = Classification(True, False, -1, -1, "Unbinding")
        s = c.get_slice(5, remove=1)
        assert s == slice(None, 5)

    def test_get_slice_movement(self):
        """Movement: present both sides → full slice."""
        c = Classification(True, True, 0, 0, "Movement")
        s = c.get_slice(5)
        assert s == slice(None, None)

    def test_get_slice_neither(self):
        """Neither present → returns None (edge case)."""
        c = Classification(False, False, 0, 0, "None")
        assert c.get_slice(5) is None


# ---------------------------------------------------------------------------
# EventClassification
# ---------------------------------------------------------------------------


class TestEventClassification:
    def test_binding(self):
        b = EvClassification.Binding
        assert b.presence_before is False
        assert b.presence_after is True
        assert repr(b) == "Binding"

    def test_unbinding(self):
        u = EvClassification.Unbinding
        assert u.presence_before is True
        assert u.presence_after is False
        assert repr(u) == "Unbinding"

    def test_all_members_exist(self):
        names = [
            "Binding",
            "Unbinding",
            "Movement",
            "Dissociation",
            "Association",
            "Merge",
            "Split",
            "Complex",
        ]
        for name in names:
            c = getattr(EvClassification, name)
            assert isinstance(c, Classification)
            assert repr(c) == name

    def test_complex_has_none_changes(self):
        c = EvClassification.Complex
        assert c.mass_change is None
        assert c.number_change is None


# ---------------------------------------------------------------------------
# classify_mass_change
# ---------------------------------------------------------------------------


class TestClassifyMassChange:
    def test_no_change_returns_movement(self):
        assert classify_mass_change(100.0, 100.0) == EvClassification.Movement

    def test_small_absolute_change_returns_movement(self):
        # 10% relative but < 30 absolute
        assert classify_mass_change(200.0, 220.0) == EvClassification.Movement

    def test_small_relative_change_returns_movement(self):
        # < 10% relative even if > 30 absolute
        assert classify_mass_change(1000.0, 1050.0) == EvClassification.Movement

    def test_large_positive_returns_association(self):
        assert classify_mass_change(500.0, 600.0) == EvClassification.Association

    def test_large_negative_returns_dissociation(self):
        assert classify_mass_change(500.0, 400.0) == EvClassification.Dissociation

    def test_zero_mass_returns_movement(self):
        assert classify_mass_change(0.0, 10.0) == EvClassification.Movement

    def test_custom_minimum_change(self):
        # 100→150: 50% relative, 50 absolute → Association with default (30).
        # With a high threshold (60), absolute is below it → Movement.
        assert classify_mass_change(100.0, 150.0) == EvClassification.Association
        assert (
            classify_mass_change(100.0, 150.0, minimum_change=60.0)
            == EvClassification.Movement
        )
