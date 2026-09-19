"""Independent boundary checks for sensing, collision resolution and imports."""

import copy
import math

import numpy as np
import pytest

from neuroterrarium.interface import MotorReadout, SensoryEncoder
from neuroterrarium.world import Body, Food, Obstacle, RADIUS, Stimulus, World


def test_obstacle_rollback_cannot_let_a_following_body_overlap():
    world = World(0, 2)
    world.bodies = [Body(20, 20, 0, speed=20), Body(18.89, 20, 0, speed=20)]
    world.obstacles = [Obstacle(22.6, 20, 2)]
    world.foods = []
    world.advance(np.array([[1, 0, 1, 0]] * 2))
    first, second = world.bodies
    assert math.hypot(first.x - second.x, first.y - second.y) >= 2 * RADIUS


def test_reading_panel_before_the_same_edit_does_not_change_sensing():
    panel, unopened = World(0, 1), World(0, 1)
    panel.observe()
    panel.stimuli = [Stimulus(15, 25, 5)]
    unopened.stimuli = [Stimulus(15, 25, 5)]
    np.testing.assert_array_equal(panel.observe(), unopened.observe())


@pytest.mark.parametrize("field,value", [("amount", -1), ("radius", -1), ("x", float("nan"))])
def test_world_import_rejects_invalid_resource_physics(field, value):
    state = World(0, 1).snapshot()
    state["foods"][0][field] = value
    with pytest.raises(ValueError):
        World.restore(state)


def test_world_import_rejects_forged_observation_cache():
    world = World(0, 1)
    world.observe()
    state = world.snapshot()
    state["observations"] = [[7.0] * 59]
    with pytest.raises(ValueError):
        World.restore(state)


def test_world_import_rejects_wrong_projection_shape():
    state = World(0, 1).snapshot()
    state["previous_projection"] = [[0.0] * 17]
    with pytest.raises(ValueError):
        World.restore(state)


def motor_sides():
    return {f"{group}_{side}": np.array([index], dtype=np.int64)
            for index, (group, side) in enumerate(
                (group, side) for group in ("mn9", "gf", "dna01", "dna02")
                for side in ("left", "right"))}


def test_negative_readout_index_cannot_silently_select_last_neuron():
    sides = motor_sides()
    sides["gf_left"] = np.array([-1], dtype=np.int64)
    with pytest.raises(ValueError):
        readout = MotorReadout(sides)
        readout.action(np.arange(8, dtype=np.int64))


def test_body_permutation_preserves_matched_actions_and_competition():
    first = World(4, 3)
    first.bodies = [Body(9.4, 10, 0), Body(10.6, 10, 0), Body(20, 20, 1)]
    first.foods = [Food(10, 10, 0.001)]
    other = World.restore(first.snapshot())
    order = [2, 0, 1]
    other.bodies = [other.bodies[index] for index in order]
    actions = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0.5, 0.2, 0.1, 0]])
    first.advance(actions)
    other.advance(actions[order])
    for index, original in enumerate(order):
        assert vars(other.bodies[index]) == vars(first.bodies[original])
    assert other.foods == first.foods


def test_encoder_snapshot_continues_its_named_random_streams():
    groups = {"sugar": np.array([0]), "lplc2": np.array([1, 2]), "lc4": np.array([3]),
              "dna02": np.array([4, 5])}
    first = SensoryEncoder(groups, {}, 53)
    restored = SensoryEncoder(groups, {}, 999)
    observation = np.zeros(59)
    observation[52] = 1
    observation[32:48] = 1
    first.encode(observation)
    restored.restore(first.snapshot())
    a, b = first.encode(observation), restored.encode(observation)
    assert sum(len(row) for row in a) > 0
    for actual, expected in zip(a, b, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_readout_snapshot_preserves_filter_tail():
    first, restored = MotorReadout(motor_sides()), MotorReadout(motor_sides())
    first.action(np.ones(8, dtype=np.int64))
    restored.restore(copy.deepcopy(first.snapshot()))
    actual, expected = first.action(np.zeros(8, dtype=np.int64)), restored.action(np.zeros(8, dtype=np.int64))
    assert actual.max() > 0
    np.testing.assert_array_equal(actual, expected)
