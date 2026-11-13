"""Tests for tandemmp.simulation.particle — target API."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tandemmp.simulation.meta import (
    Reaction,
    Stage,
    Structure,
    _reset_reaction_ids,
)
from tandemmp.simulation.movement import MovementVector
from tandemmp.simulation.particle import (
    Command,
    ParticleData,
    SimulatedParticle,
    SimulatedParticleList,
    _reset_particle_ids,
)


@pytest.fixture(autouse=True)
def _reset_ids() -> None:
    _reset_particle_ids()
    _reset_reaction_ids()


def _structure(mass: float = 60.0, label: str = "M") -> Structure:
    return Structure(mass=mass, label=label)


def _dimer(mass_a: float = 60.0, mass_b: float = 40.0) -> Structure:
    a = Structure(mass=mass_a, label="A")
    b = Structure(mass=mass_b, label="B")
    return Structure(
        [a, b],
        label="AB",
        forwards_reaction=1.0,
        backwards_reaction=0.5,
    )


def _multi_reaction_structure() -> Structure:
    """A monomer that is a reactant in two separate reactions.

    Two composites (AB at rate 1.0, AC at 0.7) are built from monomer A, registering
    two forward reactions on it. At chance=0.6 both are eligible, exercising the RNG
    branch in ``choose_reaction``.
    """
    a = Structure(mass=60.0, label="A")
    b = Structure(mass=40.0, label="B")
    c = Structure(mass=30.0, label="C")
    Structure([a, b], label="AB", forwards_reaction=1.0, backwards_reaction=0.5)
    Structure([a, c], label="AC", forwards_reaction=0.7, backwards_reaction=0.3)
    return a


# Command


class TestCommand:
    def test_fields(self) -> None:
        cmd = Command(frame=10, x=1.5, y=2.5)
        assert cmd.frame == 10
        assert cmd.x == 1.5
        assert cmd.y == 2.5
        assert cmd.mass is None

    def test_mass_override(self) -> None:
        cmd = Command(frame=1, x=0.0, y=0.0, mass=120.0)
        assert cmd.mass == 120.0

    def test_frozen(self) -> None:
        cmd = Command(frame=0, x=0.0, y=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cmd.frame = 99  # type: ignore[misc]


# ParticleData


class TestParticleData:
    def test_fields_and_length(self) -> None:
        data = ParticleData(
            frames=(0, 1, 2),
            masses=(60.0, 61.0, 62.0),
            xs=(1.0, 1.1, 1.2),
            ys=(2.0, 2.1, 2.2),
        )
        assert len(data) == 3
        assert data.frames == (0, 1, 2)
        assert data.masses == (60.0, 61.0, 62.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            ParticleData(frames=(0, 1), masses=(1.0,), xs=(0.0, 0.0), ys=(0.0, 0.0))

    def test_frozen(self) -> None:
        data = ParticleData(frames=(), masses=(), xs=(), ys=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            data.frames = (1,)  # type: ignore[misc]


# SimulatedParticle — construction & ID allocation


class TestSimulatedParticleInit:
    def test_defaults(self) -> None:
        p = SimulatedParticle(structure=_structure())
        assert p.stage == Stage.SOLUTION
        assert p.position is None
        assert p.commands == {}

    def test_stage_override(self) -> None:
        p = SimulatedParticle(structure=_structure(), stage=Stage.GLASS)
        assert p.stage == Stage.GLASS

    def test_id_allocation_increments(self) -> None:
        p1 = SimulatedParticle(structure=_structure())
        p2 = SimulatedParticle(structure=_structure())
        assert p2.id == p1.id + 1

    def test_id_reset_between_tests(self) -> None:
        p = SimulatedParticle(structure=_structure())
        assert p.id == 0

    def test_rejects_non_structure(self) -> None:
        with pytest.raises(TypeError):
            SimulatedParticle(structure="not a structure")  # type: ignore[arg-type]

    def test_rejects_non_stage(self) -> None:
        with pytest.raises(TypeError):
            SimulatedParticle(structure=_structure(), stage="glass")  # type: ignore[arg-type]


# SimulatedParticle — distance


class TestDistanceTo:
    def test_distance(self) -> None:
        p = SimulatedParticle(structure=_structure())
        p.position = (3.0, 4.0)
        assert p.distance_to(0.0, 0.0) == pytest.approx(5.0)

    def test_requires_position(self) -> None:
        p = SimulatedParticle(structure=_structure())
        with pytest.raises(RuntimeError, match="position"):
            p.distance_to(0.0, 0.0)


# SimulatedParticle — commands


class TestSetCommand:
    def test_records_command_by_frame(self) -> None:
        p = SimulatedParticle(structure=_structure())
        p.set_command(Command(frame=5, x=1.0, y=2.0), remove_rates=False)
        assert 5 in p.commands
        assert p.commands[5].x == 1.0

    def test_remove_rates_zeroes_structure_rates(self) -> None:
        s = Structure(
            mass=60.0,
            movement_vector=MovementVector(chance_per_frame=0.5, max_distance=2.0),
            binding_rate=0.2,
            unbinding_rate=0.3,
            diffusion_rate=0.4,
        )
        p = SimulatedParticle(structure=s)
        p.set_command(Command(frame=1, x=0.0, y=0.0))
        assert s.binding_rate == 0.0
        assert s.unbinding_rate == 0.0
        assert s.diffusion_rate == 0.0
        assert s.movement_vector.chance_per_frame == 0.0

    def test_keep_rates(self) -> None:
        s = Structure(mass=60.0, binding_rate=0.2)
        p = SimulatedParticle(structure=s)
        p.set_command(Command(frame=1, x=0.0, y=0.0), remove_rates=False)
        assert s.binding_rate == 0.2

    def test_rejects_non_command(self) -> None:
        p = SimulatedParticle(structure=_structure())
        with pytest.raises(TypeError):
            p.set_command("not a command")  # type: ignore[arg-type]


# SimulatedParticle — record / data


class TestRecord:
    def test_record_then_data(self) -> None:
        p = SimulatedParticle(structure=_structure(mass=60.0))
        p.position = (1.0, 2.0)
        p.record(0)
        p.position = (1.5, 2.5)
        p.record(1)
        data = p.data
        assert data.frames == (0, 1)
        assert data.masses == (60.0, 60.0)
        assert data.xs == (1.0, 1.5)
        assert data.ys == (2.0, 2.5)

    def test_records_sorted_by_frame(self) -> None:
        p = SimulatedParticle(structure=_structure())
        p.position = (0.0, 0.0)
        p.record(5)
        p.record(2)
        p.record(8)
        assert p.data.frames == (2, 5, 8)

    def test_record_requires_position(self) -> None:
        p = SimulatedParticle(structure=_structure())
        with pytest.raises(RuntimeError, match="position"):
            p.record(0)

    def test_data_empty_when_nothing_recorded(self) -> None:
        p = SimulatedParticle(structure=_structure())
        data = p.data
        assert len(data) == 0
        assert data.frames == ()


# SimulatedParticle — choose_reaction


class TestChooseReaction:
    def test_returns_none_when_no_reactions(self) -> None:
        p = SimulatedParticle(structure=_structure())
        assert p.choose_reaction(0.1) is None

    def test_picks_only_eligible(self) -> None:
        dimer = _dimer()
        p = SimulatedParticle(structure=dimer)
        # backwards has rate 0.5, chance just above that → not eligible
        assert p.choose_reaction(0.9) is None

    def test_picks_eligible_reaction(self) -> None:
        dimer = _dimer()
        p = SimulatedParticle(structure=dimer)
        reaction = p.choose_reaction(0.4)  # backwards (rate 0.5) is eligible
        assert isinstance(reaction, Reaction)
        assert reaction.rate >= 0.4

    def test_deterministic_with_rng(self) -> None:
        # _multi_reaction_structure() gives monomer A two eligible reactions at chance=0.6
        # (AB at 1.0, AC at 0.7), so the RNG branch in choose_reaction is exercised.
        a = _multi_reaction_structure()
        p = SimulatedParticle(structure=a)
        r1 = p.choose_reaction(0.6, rng=np.random.default_rng(0))
        r2 = p.choose_reaction(0.6, rng=np.random.default_rng(0))
        assert r1 is not None
        assert r1 is r2

    def test_multi_eligible_uses_rng(self) -> None:
        # Verify that different seeds can produce different results,
        # confirming the RNG branch is live.
        a = _multi_reaction_structure()
        p = SimulatedParticle(structure=a)
        reactions = {
            p.choose_reaction(0.6, rng=np.random.default_rng(seed))
            for seed in range(20)
        }
        # With two eligible reactions and 20 seeds, both should appear.
        assert len(reactions) == 2


# SimulatedParticleList


class TestParticleList:
    def test_empty_default(self) -> None:
        lst = SimulatedParticleList()
        assert len(lst) == 0
        assert list(lst) == []

    def test_extend_from_iterable(self) -> None:
        lst = SimulatedParticleList()
        particles = [SimulatedParticle(structure=_structure()) for _ in range(3)]
        lst.extend(particles)
        assert len(lst) == 3

    def test_init_with_particles(self) -> None:
        particles = [SimulatedParticle(structure=_structure()) for _ in range(2)]
        lst = SimulatedParticleList(particles)
        assert len(lst) == 2

    def test_extend_rejects_non_particle(self) -> None:
        lst = SimulatedParticleList()
        with pytest.raises(TypeError):
            lst.extend(["not a particle"])  # type: ignore[list-item]

    def test_getitem_int(self) -> None:
        p = SimulatedParticle(structure=_structure())
        lst = SimulatedParticleList([p])
        assert lst[0] is p

    def test_getitem_boolean_array(self) -> None:
        particles = [SimulatedParticle(structure=_structure()) for _ in range(3)]
        lst = SimulatedParticleList(particles)
        mask = np.array([True, False, True])
        selected = lst[mask]
        assert len(selected) == 2
        assert particles[0] in list(selected)
        assert particles[2] in list(selected)

    def test_iteration(self) -> None:
        particles = [SimulatedParticle(structure=_structure()) for _ in range(2)]
        lst = SimulatedParticleList(particles)
        assert list(lst) == particles


class TestOverlap:
    def test_disjoint_lists(self) -> None:
        a = SimulatedParticleList([SimulatedParticle(structure=_structure())])
        b = SimulatedParticleList([SimulatedParticle(structure=_structure())])
        assert a.overlap(b) is False

    def test_shared_particle(self) -> None:
        shared = SimulatedParticle(structure=_structure())
        a = SimulatedParticleList([shared])
        b = SimulatedParticleList([shared])
        assert a.overlap(b) is True


class TestStages:
    def test_set_stage_updates_all(self) -> None:
        particles = [SimulatedParticle(structure=_structure()) for _ in range(3)]
        lst = SimulatedParticleList(particles)
        lst.set_stage(Stage.GLASS)
        assert all(p.stage == Stage.GLASS for p in particles)

    def test_get_stage_returns_boolean_mask(self) -> None:
        p1 = SimulatedParticle(structure=_structure(), stage=Stage.GLASS)
        p2 = SimulatedParticle(structure=_structure(), stage=Stage.SOLUTION)
        lst = SimulatedParticleList([p1, p2])
        mask = lst.get_stage(Stage.GLASS)
        assert mask.dtype == bool
        assert mask.tolist() == [True, False]

    def test_set_stage_rejects_non_stage(self) -> None:
        lst = SimulatedParticleList()
        with pytest.raises(TypeError):
            lst.set_stage("glass")  # type: ignore[arg-type]


class TestBoundProperties:
    def test_bound_positions(self) -> None:
        p1 = SimulatedParticle(structure=_structure(), stage=Stage.GLASS)
        p1.position = (1.0, 2.0)
        p2 = SimulatedParticle(structure=_structure(), stage=Stage.SOLUTION)
        p2.position = (9.0, 9.0)
        lst = SimulatedParticleList([p1, p2])
        positions = lst.bound_positions
        assert positions.shape == (1, 2)
        assert positions[0].tolist() == [1.0, 2.0]

    def test_bound_masses(self) -> None:
        p1 = SimulatedParticle(structure=_structure(mass=60.0), stage=Stage.GLASS)
        p1.position = (0.0, 0.0)
        p2 = SimulatedParticle(structure=_structure(mass=120.0), stage=Stage.SOLUTION)
        lst = SimulatedParticleList([p1, p2])
        assert lst.bound_masses.tolist() == [60.0]

    def test_bound_positions_empty(self) -> None:
        p = SimulatedParticle(structure=_structure(), stage=Stage.SOLUTION)
        lst = SimulatedParticleList([p])
        positions = lst.bound_positions
        assert positions.shape == (0, 2)

    def test_bound_masses_empty(self) -> None:
        p = SimulatedParticle(structure=_structure(), stage=Stage.SOLUTION)
        lst = SimulatedParticleList([p])
        assert lst.bound_masses.shape == (0,)


class TestRates:
    def test_binding_rates(self) -> None:
        s1 = Structure(mass=1.0, binding_rate=0.1)
        s2 = Structure(mass=1.0, binding_rate=0.2)
        lst = SimulatedParticleList(
            [
                SimulatedParticle(structure=s1),
                SimulatedParticle(structure=s2),
            ]
        )
        np.testing.assert_allclose(lst.binding_rates(), [0.1, 0.2])

    def test_unbinding_rates(self) -> None:
        s = Structure(mass=1.0, unbinding_rate=0.3)
        lst = SimulatedParticleList([SimulatedParticle(structure=s)])
        np.testing.assert_allclose(lst.unbinding_rates(), [0.3])

    def test_diffusion_rates(self) -> None:
        s = Structure(mass=1.0, diffusion_rate=0.05)
        lst = SimulatedParticleList([SimulatedParticle(structure=s)])
        np.testing.assert_allclose(lst.diffusion_rates(), [0.05])

    def test_movement_rates(self) -> None:
        mv = MovementVector(chance_per_frame=0.7, max_distance=1.0)
        s = Structure(mass=1.0, movement_vector=mv)
        lst = SimulatedParticleList([SimulatedParticle(structure=s)])
        np.testing.assert_allclose(lst.movement_rates(), [0.7])


class TestSpeciesAndStructures:
    def test_species_numbers_counts_by_structure(self) -> None:
        s1 = _structure(mass=60.0, label="A")
        s2 = _structure(mass=60.0, label="A")  # hashes/equals s1
        s3 = _structure(mass=120.0, label="B")
        lst = SimulatedParticleList(
            [
                SimulatedParticle(structure=s1),
                SimulatedParticle(structure=s2),
                SimulatedParticle(structure=s3),
            ]
        )
        counts = lst.species_numbers()
        assert counts[s1] == 2
        assert counts[s3] == 1

    def test_structures_all(self) -> None:
        s1 = _structure()
        s2 = _structure(mass=120.0, label="B")
        lst = SimulatedParticleList(
            [SimulatedParticle(structure=s1), SimulatedParticle(structure=s2)]
        )
        structures = lst.structures()
        assert list(structures) == [s1, s2]

    def test_structures_filtered_by_stage(self) -> None:
        s1 = _structure(label="A")
        s2 = _structure(mass=120.0, label="B")
        p1 = SimulatedParticle(structure=s1, stage=Stage.GLASS)
        p2 = SimulatedParticle(structure=s2, stage=Stage.SOLUTION)
        lst = SimulatedParticleList([p1, p2])
        structures = lst.structures(Stage.GLASS)
        assert list(structures) == [s1]

    def test_available_reactions_shape(self) -> None:
        dimer = _dimer()
        lst = SimulatedParticleList([SimulatedParticle(structure=dimer)])
        reactions = lst.available_reactions()
        assert reactions.shape == (1,)


class TestCommandsAccessors:
    def test_commands_and_command_frames(self) -> None:
        p1 = SimulatedParticle(structure=_structure())
        p1.set_command(Command(frame=3, x=0.0, y=0.0), remove_rates=False)
        p2 = SimulatedParticle(structure=_structure())
        lst = SimulatedParticleList([p1, p2])
        assert lst.commands == [{3: Command(frame=3, x=0.0, y=0.0)}, {}]
        assert lst.command_frames == [[3], []]
