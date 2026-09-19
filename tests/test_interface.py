"""Interface causal checks using real registered identities, without graph loading."""

from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

from neuroterrarium.interface import MotorReadout, SensoryEncoder
from neuroterrarium.registry import Registry
from neuroterrarium.world import Body, Food, RAYS, Stimulus, World


@pytest.fixture
def mapping():
    registry = Registry.load(Path(__file__).resolve().parents[1] / "configs/interface-v783.json")
    roots = np.array([
        int(node["root_id"])
        for group in registry.groups.values()
        for node in group["neurons"]
    ], dtype=np.int64)
    return registry.resolve(roots), registry.resolve_sides(roots), len(roots)


def active_observation():
    observation = np.zeros(59)
    observation[2 * RAYS] = 0.8
    observation[52] = 1.0
    observation[53] = 1.0
    return observation


def same_tape(left, right):
    return len(left) == len(right) and all(
        np.array_equal(a, b) for a, b in zip(left, right, strict=True)
    )


def only_group(tape, indices):
    return [event[np.isin(event, indices)] for event in tape]


def test_readout_depends_only_on_registered_counts_and_filter_state(mapping):
    groups, sides, size = mapping
    counts = np.zeros(size + 11, dtype=np.int64)
    counts[sides["gf_left"]] = 2
    counts[sides["mn9_right"]] = 1
    counts[sides["dna02_left"]] = 3
    changed = counts.copy()
    readout_indices = np.concatenate([groups[name] for name in ("gf", "mn9", "dna01", "dna02")])
    unrelated = np.setdiff1d(np.arange(len(counts)), readout_indices)
    changed[unrelated] = 1_000_000
    expected = MotorReadout(sides).action(counts)
    actual = MotorReadout(sides).action(changed)
    np.testing.assert_array_equal(actual, expected)
    assert np.any(actual > 0)


def test_registered_motor_group_wiring_is_separate(mapping):
    _, sides, size = mapping
    counts = np.zeros(size, dtype=np.int64)
    counts[sides["mn9_right"]] = 1
    action = MotorReadout(sides).action(counts)
    assert action[3] > 0
    np.testing.assert_array_equal(action[:3], np.zeros(3))
    counts[:] = 0
    counts[sides["gf_left"]] = 1
    action = MotorReadout(sides).action(counts)
    assert action[2] > 0
    np.testing.assert_array_equal(action[[0, 1, 3]], np.zeros(3))
    counts[:] = 0
    counts[sides["dna02_left"]] = 1
    action = MotorReadout(sides).action(counts)
    assert action[1] > 0
    np.testing.assert_array_equal(action[2:], np.zeros(2))


def test_continuous_all_readout_clamp_overrides_history_and_new_counts(mapping):
    _, sides, size = mapping
    readout = MotorReadout(sides)
    counts = np.full(size, 4, dtype=np.int64)
    assert np.any(readout.action(counts) > 0)
    readout.clamped = True
    for index in range(30):
        np.testing.assert_array_equal(readout.action(counts * index), np.zeros(4))
    restored = MotorReadout(sides)
    restored.restore(readout.snapshot())
    np.testing.assert_array_equal(restored.action(counts), np.zeros(4))


def test_channel_disable_changes_actual_spike_input(mapping):
    groups, sides, _ = mapping
    encoder = SensoryEncoder(groups, sides, 73, tonic_hz=0)
    observation = active_observation()
    active = encoder.encode(observation)
    assert sum(map(len, active)) > 0
    encoder.channels = {"taste": False, "vision": False}
    disabled = encoder.encode(observation)
    assert all(len(row) == 0 for row in disabled)
    assert encoder.last_rates_hz == {"sugar": 0.0, "lplc2": 0.0, "lc4": 0.0, "dna02": 0.0}


def test_channel_random_streams_are_independent(mapping):
    groups, sides, _ = mapping
    left = SensoryEncoder(groups, sides, 79)
    right = SensoryEncoder(groups, sides, 79)
    right.channels["taste"] = False
    for _ in range(3):
        a, b = left.encode(active_observation()), right.encode(active_observation())
        assert same_tape(only_group(a, groups["lplc2"]), only_group(b, groups["lplc2"]))
        assert same_tape(only_group(a, groups["lc4"]), only_group(b, groups["lc4"]))
        assert all(len(row) == 0 for row in only_group(b, groups["sugar"]))


def test_encoder_random_stream_snapshot_restores_future_nonzero_tape(mapping):
    groups, sides, _ = mapping
    encoder = SensoryEncoder(groups, sides, 83)
    encoder.encode(active_observation())
    snapshot = encoder.snapshot()
    expected = encoder.encode(active_observation())
    restored = SensoryEncoder(groups, sides, 97)
    restored.restore(snapshot)
    actual = restored.encode(active_observation())
    assert sum(map(len, expected)) > 0
    assert same_tape(expected, actual)
    assert encoder.snapshot() == restored.snapshot()


def test_nonzero_sham_configuration_has_identical_input_and_readout(mapping):
    groups, sides, size = mapping
    normal = SensoryEncoder(groups, sides, 89)
    sham = SensoryEncoder(groups, sides, 89)
    sham.channels.update({"taste": True, "vision": True})
    a, b = normal.encode(active_observation()), sham.encode(active_observation())
    assert sum(map(len, a)) > 0
    assert same_tape(a, b)
    counts = np.arange(size, dtype=np.int64) % 3
    normal_readout = MotorReadout(sides)
    sham_readout = MotorReadout(sides)
    normal_readout.action(counts)
    sham_readout.restore(normal_readout.snapshot())
    actual = normal_readout.action(counts)
    assert np.any(actual > 0)
    np.testing.assert_array_equal(sham_readout.action(counts), actual)


def test_stimulus_removal_changes_world_observation_and_leaves_no_input(mapping):
    groups, sides, _ = mapping
    encoder = SensoryEncoder(groups, sides, 101, tonic_hz=0)
    world = World(7, 1)
    world.bodies = [Body(20, 28, 0)]
    world.foods = [Food(20, 28)]
    world.stimuli = [Stimulus(28, 28, 0.4, growth=8)]
    world.observe()
    world.advance(np.zeros((1, 4)))
    present = world.observe()[0]
    assert present[52] == 1
    assert np.max(present[2 * RAYS:3 * RAYS]) > 0
    assert sum(map(len, encoder.encode(present))) > 0
    world.foods.clear()
    world.stimuli.clear()
    world.advance(np.zeros((1, 4)))
    removed = world.observe()[0]
    assert removed[52] == 0
    assert np.max(removed[2 * RAYS:3 * RAYS]) == 0
    assert all(len(row) == 0 for row in encoder.encode(removed))
    world.advance(np.zeros((1, 4)))
    assert all(len(row) == 0 for row in encoder.encode(world.observe()[0]))


def test_encoder_uses_only_registered_observation_features(mapping):
    groups, sides, _ = mapping
    original = active_observation()
    changed = original.copy()
    changed[:2 * RAYS] = 0.7
    changed[48:52] = 0.9
    changed[53:] = 0.25
    a = SensoryEncoder(groups, sides, 107).encode(original)
    b = SensoryEncoder(groups, sides, 107).encode(changed)
    assert same_tape(a, b)


@pytest.mark.parametrize("bad", [np.zeros(58), np.full(59, np.nan), np.full(59, np.inf), ["0"] * 59])
def test_encoder_rejects_invalid_observation_shape_and_nonfinite_values(mapping, bad):
    groups, sides, _ = mapping
    with pytest.raises(ValueError, match="schema mismatch"):
        SensoryEncoder(groups, sides, 109).encode(bad)


def test_tonic_drive_is_independent_of_observation_and_sensory_channels(mapping):
    groups, sides, _ = mapping
    normal = SensoryEncoder(groups, sides, 113, tonic_hz=50)
    disabled = SensoryEncoder(groups, sides, 113, tonic_hz=50)
    disabled.channels = {"taste": False, "vision": False}
    a = normal.encode(active_observation(), steps=1000)
    b = disabled.encode(np.zeros(59), steps=1000)
    assert sum(map(len, b)) > 0
    assert same_tape(only_group(a, groups["dna02"]), b)
    assert disabled.last_rates_hz == {"sugar": 0.0, "lplc2": 0.0, "lc4": 0.0, "dna02": 50.0}
    disabled.tonic_hz = 0
    assert all(len(row) == 0 for row in disabled.encode(active_observation()))


@pytest.mark.parametrize("index,value", [(0, -0.01), (16, 1.01), (32, -1.01), (47, 1.01),
                                         (48, -0.01), (52, 2), (53, -0.01), (54, 2), (56, -1.01)])
def test_out_of_range_observations_are_rejected_before_rng_advances(mapping, index, value):
    groups, sides, _ = mapping
    encoder = SensoryEncoder(groups, sides, 127)
    before = encoder.snapshot()
    observation = np.zeros(59)
    observation[index] = value
    with pytest.raises(ValueError, match="schema range"):
        encoder.encode(observation)
    assert encoder.snapshot() == before


@pytest.mark.parametrize("channels", [{}, {"taste": True}, {"taste": True, "vision": "false"},
                                     {"taste": 1, "vision": False},
                                     {"taste": True, "vision": True, "hidden": True}])
def test_invalid_channels_are_rejected(mapping, channels):
    groups, sides, _ = mapping
    encoder = SensoryEncoder(groups, sides, 131)
    encoder.channels = channels
    with pytest.raises(ValueError, match="channel"):
        encoder.encode(np.zeros(59))


@pytest.mark.parametrize("tonic", [-1, 51, np.inf, np.nan, True, "15"])
def test_invalid_tonic_rate_is_rejected_at_creation_and_encoding(mapping, tonic):
    groups, sides, _ = mapping
    with pytest.raises(ValueError, match="tonic"):
        SensoryEncoder(groups, sides, 137, tonic_hz=tonic)
    encoder = SensoryEncoder(groups, sides, 137)
    encoder.tonic_hz = tonic
    with pytest.raises(ValueError, match="tonic"):
        encoder.encode(np.zeros(59))


@pytest.mark.parametrize("steps", [0, 10001, 2.5, True])
def test_invalid_neural_window_is_rejected(mapping, steps):
    groups, sides, _ = mapping
    with pytest.raises(ValueError, match="neural window"):
        SensoryEncoder(groups, sides, 139).encode(np.zeros(59), steps=steps)


@pytest.mark.parametrize("mutation", ["missing_stream", "wrong_algorithm", "float_state", "bad_increment",
                                     "bad_channels", "bad_tonic", "bad_rates", "extra_top_level"])
def test_malformed_encoder_snapshot_is_rejected_atomically(mapping, mutation):
    groups, sides, _ = mapping
    encoder = SensoryEncoder(groups, sides, 149)
    encoder.encode(active_observation())
    before = encoder.snapshot()
    bad = deepcopy(before)
    if mutation == "missing_stream":
        del bad["random"]["dna02"]
    elif mutation == "wrong_algorithm":
        bad["random"]["sugar"]["bit_generator"] = "MT19937"
    elif mutation == "float_state":
        bad["random"]["lplc2"]["state"]["state"] = 1.0
    elif mutation == "bad_increment":
        bad["random"]["lc4"]["state"]["inc"] = 2
    elif mutation == "bad_channels":
        bad["channels"]["vision"] = "false"
    elif mutation == "bad_tonic":
        bad["tonic_hz"] = -1
    elif mutation == "bad_rates":
        bad["last_rates_hz"]["sugar"] = float("inf")
    else:
        bad["extra"] = 1
    with pytest.raises(ValueError):
        encoder.restore(bad)
    assert encoder.snapshot() == before


def test_tonic_rate_is_preserved_by_snapshot(mapping):
    groups, sides, _ = mapping
    original = SensoryEncoder(groups, sides, 151, tonic_hz=37)
    snapshot = original.snapshot()
    restored = SensoryEncoder(groups, sides, 157, tonic_hz=0)
    restored.restore(snapshot)
    assert restored.tonic_hz == 37
    assert same_tape(original.encode(np.zeros(59), 1000), restored.encode(np.zeros(59), 1000))


def test_readout_snapshot_boolean_validation_is_atomic(mapping):
    _, sides, size = mapping
    readout = MotorReadout(sides)
    readout.action(np.ones(size, dtype=np.int64))
    before = readout.snapshot()
    bad = deepcopy(before)
    bad["clamped"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        readout.restore(bad)
    assert readout.snapshot() == before
