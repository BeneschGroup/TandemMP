"""Tests for tandemmp.simulation.meta — target API."""

from __future__ import annotations

import pytest

from tandemmp.simulation.defaults import SimulationDefaults
from tandemmp.simulation.meta import (
    Behaviour,
    Reaction,
    Stage,
    Structure,
    _reset_reaction_ids,
    create_oligomer,
)
from tandemmp.simulation.movement import MovementVector


@pytest.fixture(autouse=True)
def _reset_ids() -> None:
    _reset_reaction_ids()


class TestStage:
    def test_values(self) -> None:
        assert Stage.GLASS.value == 0
        assert Stage.NEAR.value == 1
        assert Stage.SOLUTION.value == 2

    def test_ordered_as_integers(self) -> None:
        assert Stage.GLASS < Stage.NEAR < Stage.SOLUTION


class TestBehaviour:
    def test_has_all_event_types(self) -> None:
        for name in (
            "BINDING",
            "UNBINDING",
            "MOVEMENT",
            "DISSOCIATION",
            "AGGREGATION",
            "MERGING",
            "SPLITTING",
        ):
            assert hasattr(Behaviour, name)


class TestReaction:
    def test_auto_id(self) -> None:
        r1 = Reaction(reactants="a", products="b", rate=1.0)
        r2 = Reaction(reactants="b", products="a", rate=0.5)
        assert r1.id == 1
        assert r2.id == 2

    def test_repr_includes_id_and_rate(self) -> None:
        r = Reaction(reactants="x", products="y", rate=2.5)
        text = repr(r)
        assert "k1" in text
        assert "2.5" in text

    def test_id_reset_between_tests(self) -> None:
        r = Reaction(reactants="a", products="b", rate=1.0)
        assert r.id == 1


class TestStructureLeaf:
    def test_requires_mass_when_no_contents(self) -> None:
        with pytest.raises(ValueError, match="mass"):
            Structure()

    def test_leaf_properties(self) -> None:
        s = Structure(mass=60.0, label="M")
        assert s.mass == 60.0
        assert s.n_subunits == 1
        assert s.label == "M"
        assert s.size > 0

    def test_default_movement_vector(self) -> None:
        s = Structure(mass=1.0)
        assert isinstance(s.movement_vector, MovementVector)

    def test_default_rates_match_defaults(self) -> None:
        d = SimulationDefaults()
        s = Structure(mass=1.0)
        assert s.binding_rate == d.binding_rate
        assert s.unbinding_rate == d.unbinding_rate
        assert s.diffusion_rate == d.diffusion_rate

    def test_override_rates(self) -> None:
        s = Structure(mass=1.0, binding_rate=0.1, diffusion_rate=0.02)
        assert s.binding_rate == 0.1
        assert s.diffusion_rate == 0.02

    def test_leaf_has_no_reactions(self) -> None:
        s = Structure(mass=1.0)
        assert s.available_reactions == []
        assert s.forwards_reaction is None


class TestStructureComposite:
    def test_composite_sums_mass_and_subunits(self) -> None:
        a = Structure(mass=60.0, label="A")
        b = Structure(mass=40.0, label="B")
        dimer = Structure(
            [a, b], label="AB", forwards_reaction=1.0, backwards_reaction=0.5
        )
        assert dimer.mass == 100.0
        assert dimer.n_subunits == 2

    def test_composite_requires_reactions(self) -> None:
        a = Structure(mass=60.0)
        b = Structure(mass=40.0)
        with pytest.raises(ValueError, match="reaction"):
            Structure([a, b])

    def test_composite_creates_forward_and_backward(self) -> None:
        a = Structure(mass=60.0)
        b = Structure(mass=40.0)
        dimer = Structure([a, b], forwards_reaction=1.0, backwards_reaction=0.25)
        assert isinstance(dimer.forwards_reaction, Reaction)
        assert isinstance(dimer.backwards_reaction, Reaction)
        assert dimer.forwards_reaction.rate == 1.0
        assert dimer.backwards_reaction.rate == 0.25

    def test_subunits_see_forward_reaction(self) -> None:
        a = Structure(mass=60.0)
        b = Structure(mass=40.0)
        Structure([a, b], forwards_reaction=1.0, backwards_reaction=0.5)
        assert len(a.available_reactions) == 1
        assert len(b.available_reactions) == 1

    def test_actual_reactions_matches_participation(self) -> None:
        a = Structure(mass=60.0)
        b = Structure(mass=40.0)
        dimer = Structure([a, b], forwards_reaction=1.0, backwards_reaction=0.5)
        # Subunits participate in the forward reaction (reactants is a list)
        assert dimer.forwards_reaction in a.actual_reactions
        # The dimer itself is the reactant of the backward reaction
        assert dimer.backwards_reaction in dimer.actual_reactions


class TestStructureHierarchy:
    def test_monomers(self) -> None:
        a = Structure(mass=60.0, label="A")
        b = Structure(mass=40.0, label="B")
        dimer = Structure([a, b], forwards_reaction=1.0, backwards_reaction=1.0)
        monomers = dimer.monomers
        assert a in monomers
        assert b in monomers
        assert dimer not in monomers

    def test_unique_configurations_sorted_by_subunits(self) -> None:
        a = Structure(mass=60.0, label="A")
        b = Structure(mass=40.0, label="B")
        dimer = Structure([a, b], forwards_reaction=1.0, backwards_reaction=1.0)
        configs = dimer.unique_configurations
        assert configs[0].n_subunits >= configs[-1].n_subunits

    def test_set_global_propagates(self) -> None:
        a = Structure(mass=60.0)
        b = Structure(mass=40.0)
        dimer = Structure([a, b], forwards_reaction=1.0, backwards_reaction=1.0)
        dimer.set_global(binding_rate=0.9)
        assert a.binding_rate == 0.9
        assert b.binding_rate == 0.9
        assert dimer.binding_rate == 0.9


class TestStructureEquality:
    def test_equal_leaves(self) -> None:
        a = Structure(mass=60.0, label="A")
        b = Structure(mass=60.0, label="A")
        assert a == b
        assert hash(a) == hash(b)

    def test_unequal_on_mass(self) -> None:
        a = Structure(mass=60.0, label="A")
        b = Structure(mass=61.0, label="A")
        assert a != b

    def test_not_equal_to_non_structure(self) -> None:
        s = Structure(mass=60.0)
        assert s != "structure"


class TestAddReactions:
    def test_rejects_non_reaction(self) -> None:
        s = Structure(mass=1.0)
        with pytest.raises(TypeError):
            s.add_reactions("not a reaction")  # type: ignore[arg-type]

    def test_ignores_duplicates(self) -> None:
        s = Structure(mass=1.0)
        r = Reaction(reactants="a", products="b", rate=1.0)
        s.add_reactions(r)
        s.add_reactions(r)
        assert len(s.available_reactions) == 1


class TestCreateOligomer:
    def test_monomer_returns_self(self) -> None:
        m = Structure(mass=60.0, label="M")
        assert create_oligomer(m, 1) is m

    def test_dimer(self) -> None:
        m = Structure(mass=60.0, label="M")
        dimer = create_oligomer(m, 2)
        assert dimer.mass == 120.0
        assert dimer.n_subunits == 2

    def test_trimer(self) -> None:
        m = Structure(mass=60.0, label="M")
        trimer = create_oligomer(m, 3)
        assert trimer.mass == pytest.approx(180.0)
        assert trimer.n_subunits == 3

    def test_invalid_n(self) -> None:
        m = Structure(mass=60.0)
        with pytest.raises(ValueError):
            create_oligomer(m, 0)

    def test_negative_n(self) -> None:
        m = Structure(mass=60.0)
        with pytest.raises(ValueError):
            create_oligomer(m, -5)

    def test_monomer_accumulates_reactions(self) -> None:
        # create_oligomer registers each intermediate's forward reaction on the
        # monomer. A trimer (n=3) creates 2 intermediates → 2 forward reactions.
        m = Structure(mass=60.0, label="M")
        create_oligomer(m, 3)
        assert len(m.available_reactions) == 2
