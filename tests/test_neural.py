"""Active sparse dynamics compared with a separate Brian2 implementation."""

import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from neuroterrarium.neural import NeuralLimits, NeuralResourceError, SparseBrain
from neuroterrarium.reference import run_reference


PRE = [1, 0, 4, 0, 2, 0, 3, 2, 5, 1, 6, 5]
POST = [3, 2, 3, 3, 4, 2, 5, 6, 2, 6, 1, 0]
COUNTS = [100, 240, -200, 140, 500, 130, 200, 320, 160, 350, -180, 120]


def input_tape(steps=600):
    rng = np.random.default_rng(98472)
    draws = rng.random((steps, 2)) < np.array([0.09, 0.045])
    return [np.flatnonzero(row).tolist() for row in draws]


def brain(counts=COUNTS):
    return SparseBrain(7, PRE, POST, counts, input_neurons=[0, 1])


def reference(events, **kwargs):
    return run_reference(7, PRE, POST, COUNTS, events, input_neurons=[0, 1],
                         record_indices=list(range(7)), **kwargs)


def assert_reference(actual, expected):
    np.testing.assert_array_equal(actual.spike_steps, expected.spike_steps)
    np.testing.assert_array_equal(actual.spike_neurons, expected.spike_neurons)
    np.testing.assert_allclose(actual.v_mV, expected.v_mV, rtol=0, atol=1e-9)
    np.testing.assert_allclose(actual.g_mV, expected.g_mV, rtol=0, atol=1e-9)


def test_active_recurrent_signed_graph_matches_each_reference_spike():
    events = input_tape()
    expected = reference(events)
    actual = brain().advance(events, list(range(7)), record_spikes=True)
    assert len(expected.spike_steps) > 50
    assert len(set(expected.spike_neurons.tolist()) - {0, 1}) >= 3
    assert_reference(actual, expected)
    np.testing.assert_array_equal(actual.spike_counts,
                                  np.bincount(actual.spike_neurons, minlength=7))


@pytest.mark.parametrize("changes", [
    {"disconnect_output": [0, 3]},
    {"disconnect_input": [2, 4]},
    {"clamp_spikes": [3, 5]},
])
def test_intervention_masks_match_reference_and_restore(changes):
    events = input_tape(420)
    restore = {key: [] for key in changes}
    schedule = {37: changes, 187: restore, 210: changes, 350: restore}
    expected = reference(events, interventions=schedule)
    actual = brain().advance(events, list(range(7)), True, interventions=schedule)
    assert len(expected.spike_steps) > 20
    assert_reference(actual, expected)


def test_simultaneous_delivery_and_external_reset_match_reference():
    events = [[0, 1] for _ in range(120)]
    expected = reference(events)
    actual = brain().advance(events, list(range(7)), True)
    assert_reference(actual, expected)
    assert actual.spike_steps[actual.spike_neurons == 0][:5].tolist() == [1, 3, 5, 7, 9]


def test_cross_window_queue_and_refractory_state_equal_continuous_run():
    events = input_tape(370)
    continuous = brain().advance(events, list(range(7)), True)
    persistent = brain()
    cuts = [0, 1, 2, 17, 18, 19, 20, 21, 22, 23, 57, 188, 189, 370]
    parts = [persistent.advance(events[left:right], list(range(7)), True)
             for left, right in zip(cuts[:-1], cuts[1:])]
    for key in ("spike_steps", "spike_neurons", "v_mV", "g_mV"):
        np.testing.assert_array_equal(np.concatenate([getattr(p, key) for p in parts]),
                                      getattr(continuous, key))
    assert parts[-1].end_step == 370


def test_structured_snapshot_restores_pending_events_and_nonresting_state():
    events = input_tape(320)
    original = brain()
    original.advance(events[:123], [0, 2, 4], True,
                     interventions={99: {"disconnect_output": [3]}})
    state = json.loads(json.dumps(original.snapshot(), allow_nan=False))
    assert sum(map(len, state["queue"])) > 0
    assert any(value != -52 for value in state["v_mV"])
    expected = original.advance(events[123:], list(range(7)), True)
    restored = brain()
    restored.restore(state)
    actual = restored.advance(events[123:], list(range(7)), True)
    for key in ("spike_steps", "spike_neurons", "v_mV", "g_mV"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    assert restored.snapshot() == original.snapshot()
    assert state["step"] == 123


def test_sham_and_recording_switch_do_not_change_hidden_state():
    events = input_tape(280)
    normal, sham, count_only = brain(), brain(), brain()
    recorded = normal.advance(events, list(range(7)), True)
    sham.advance(events, list(range(7)), True, interventions={0: {}, 51: {}, 166: {}})
    counts = count_only.advance(events)
    assert len(recorded.spike_steps) > 20
    assert counts.spike_steps.size == 0
    assert counts.v_mV.shape == (280, 0)
    np.testing.assert_array_equal(counts.spike_counts, recorded.spike_counts)
    assert normal.snapshot() == sham.snapshot() == count_only.snapshot()


def test_graph_csr_retains_all_edges_and_stable_duplicate_order():
    model = brain()
    assert len(model.weights_mV) == len(COUNTS)
    left, right = model.indptr[:2]
    assert model.post_indices[left:right].tolist() == [2, 3, 2]
    np.testing.assert_allclose(model.weights_mV[left:right], np.array([240, 140, 130])*.275)
    for array in (model.indptr, model.post_indices, model.weights_mV, model.input_mask):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        model.weights_mV[0] = 0


def test_unconnected_neuron_is_still_integrated_in_full_state():
    model = SparseBrain(3, [0], [1], [8000], input_neurons=[0],
                        initial_v_mV=[-52, -52, -51], initial_g_mV=[0, 0, 2])
    events = [[] for _ in range(80)]
    events[0] = [0]
    actual = model.advance(events, [2], True)
    assert 1 in actual.spike_neurons
    elapsed_ms = (np.arange(80) + 1) * 0.1
    a, b = np.exp(-elapsed_ms / 20), np.exp(-elapsed_ms / 5)
    np.testing.assert_allclose(actual.v_mV[:, 0], -52 + a + (5/15)*(a-b)*2,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(actual.g_mV[:, 0], 2*b, rtol=0, atol=1e-9)


def test_fork_shares_graph_but_preserves_independent_state():
    original = brain()
    events = input_tape(300)
    original.advance(events[:121])
    other = original.fork()
    for name in ("indptr", "post_indices", "weights_mV", "input_mask"):
        assert getattr(other, name) is getattr(original, name)
    for name in ("v_mV", "g_mV", "last_spike_steps", "_queue", "_queue_counts",
                 "output_disconnected", "input_disconnected", "spike_clamped"):
        assert not np.shares_memory(getattr(other, name), getattr(original, name))
    saved = other.snapshot()
    left = original.advance(events[121:], list(range(7)), True)
    assert other.snapshot() == saved
    right = other.advance(events[121:], list(range(7)), True)
    np.testing.assert_array_equal(left.spike_steps, right.spike_steps)
    np.testing.assert_array_equal(left.spike_neurons, right.spike_neurons)
    np.testing.assert_array_equal(left.v_mV, right.v_mV)
    np.testing.assert_array_equal(left.g_mV, right.g_mV)


def test_restore_rejects_corruption_before_changing_any_state():
    model = brain()
    model.advance(input_tape(90))
    good = model.snapshot()
    corruptions = [
        {"graph_digest": "0" * 64}, {"dt_ms": 1.0}, {"step": 2.5},
        {"v_mV": [float("nan")] * 7}, {"last_spike_steps": [1.0] * 7},
        {"queue": [[]]}, {"disconnect_output": [7]},
    ]
    for change in corruptions:
        invalid = {**copy.deepcopy(good), **change}
        with pytest.raises(ValueError):
            model.restore(invalid)
        assert model.snapshot() == good
    with pytest.raises(ValueError, match="mismatch"):
        brain(counts=[value * 2 for value in COUNTS]).restore(good)


def test_invalid_tape_or_intervention_does_not_partially_advance():
    model = brain()
    before = model.snapshot()
    for events, schedule in [([[0], [0, 0]], None), ([[0], [1.0]], None),
                             ([[0], [2]], None), ([[0], []], {1: {"bad": [0]}})]:
        with pytest.raises(ValueError):
            model.advance(events, interventions=schedule)
        assert model.snapshot() == before


def test_delay_and_refractory_edge_boundary_match_reference():
    for second_input in (22, 23):
        events = [[] for _ in range(70)]
        events[0] = [0]
        events[second_input] = [0]
        actual = SparseBrain(2, [0], [1], [8000], input_neurons=[0]).advance(events, [0, 1], True)
        expected = run_reference(2, [0], [1], [8000], events,
                                 input_neurons=[0], record_indices=[0, 1])
        assert_reference(actual, expected)


def test_output_cut_and_restore_at_event_delivery_match_reference():
    events = [[] for _ in range(70)]
    events[0] = [0]
    for restore_step in (19, 20):
        schedule = {18: {"disconnect_output": [0]}, restore_step: {"disconnect_output": []}}
        actual = SparseBrain(2, [0], [1], [8000], input_neurons=[0]).advance(
            events, [0, 1], True, interventions=schedule)
        expected = run_reference(2, [0], [1], [8000], events, input_neurons=[0],
                                 record_indices=[0, 1], interventions=schedule)
        assert_reference(actual, expected)


def test_spike_clamp_preserves_events_already_queued_before_clamp():
    events = [[] for _ in range(70)]
    events[0] = [0]
    events[20] = [0]
    schedule = {2: {"clamp_spikes": [0]}}
    actual = SparseBrain(2, [0], [1], [8000], input_neurons=[0]).advance(
        events, [0, 1], True, interventions=schedule)
    expected = run_reference(2, [0], [1], [8000], events, input_neurons=[0],
                             record_indices=[0, 1], interventions=schedule)
    assert actual.spike_steps[actual.spike_neurons == 0].tolist() == [1]
    assert actual.spike_steps[actual.spike_neurons == 1].tolist() == [20]
    assert_reference(actual, expected)


def test_input_disconnect_is_evaluated_at_delivery_and_does_not_erase_source_spikes():
    events = [[] for _ in range(70)]
    events[0] = [0]
    for restore_step in (19, 20):
        schedule = {18: {"disconnect_input": [1]}, restore_step: {"disconnect_input": []}}
        actual = SparseBrain(2, [0], [1], [8000], input_neurons=[0]).advance(
            events, [0, 1], True, interventions=schedule)
        expected = run_reference(2, [0], [1], [8000], events, input_neurons=[0],
                                 record_indices=[0, 1], interventions=schedule)
        assert actual.spike_steps[actual.spike_neurons == 0].tolist() == [1]
        assert_reference(actual, expected)


class UnreadableSized:
    """A size-only fixture that fails if a rejected request is materialized."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        raise AssertionError("oversized input was materialized")

    def __array__(self, *args, **kwargs):
        raise AssertionError("oversized input was converted to an array")


def test_resource_limits_reject_oversized_requests_before_allocation():
    model = brain()
    before = model.snapshot()
    with pytest.raises(NeuralResourceError, match="step budget"):
        model.advance(UnreadableSized(20_001))
    with pytest.raises(NeuralResourceError, match="memory budget"):
        SparseBrain(50_000_000, [], [], [], input_neurons=[])
    with pytest.raises(NeuralResourceError, match="memory budget"):
        SparseBrain(2, UnreadableSized(100_000_000), UnreadableSized(100_000_000),
                    UnreadableSized(100_000_000), input_neurons=[])
    with pytest.raises(NeuralResourceError, match="recorded neuron"):
        model.advance([[]], UnreadableSized(8))
    bad = {**before, "v_mV": UnreadableSized(8)}
    with pytest.raises(ValueError, match="one finite value"):
        model.restore(bad)
    assert model.snapshot() == before


@pytest.mark.parametrize("budget, events, recorded, schedule, message", [
    ({"max_trace_bytes": 100}, [[]]*10, [0, 1], None, "byte budget"),
    ({"max_input_events": 3}, [[0, 1], [0, 1]], [], None, "event budget"),
    ({"max_neuron_updates": 10}, [[], []], [], None, "update budget"),
    ({"max_interventions": 1}, [[], []], [], {0: {}, 1: {}}, "interventions"),
])
def test_each_resource_budget_is_independent_and_transactional(budget, events, recorded, schedule, message):
    model = SparseBrain(7, PRE, POST, COUNTS, input_neurons=[0, 1],
                        limits=replace(NeuralLimits(), **budget))
    before = model.snapshot()
    with pytest.raises(NeuralResourceError, match=message):
        model.advance(events, recorded, interventions=schedule)
    assert model.snapshot() == before


def test_recording_overflow_across_segments_rolls_back_every_state_component():
    model = SparseBrain(7, PRE, POST, COUNTS, input_neurons=[0, 1],
                        limits=replace(NeuralLimits(), max_recorded_spikes=3))
    model.advance(input_tape(121))
    before = model.snapshot()
    assert sum(map(len, before["queue"])) > 0
    events = [[0, 1] for _ in range(20)]
    schedule = {0: {"disconnect_output": [3]}, **{i: {} for i in range(1, 20)}}
    with pytest.raises(NeuralResourceError, match="state unchanged"):
        model.advance(events, list(range(7)), True, interventions=schedule)
    assert model.snapshot() == before
    comparison = model.fork()
    # A failed request is resumable without a hidden partial neural step.
    model.advance(events)
    comparison.advance(events)
    assert model.snapshot() == comparison.snapshot()


def test_interrupt_after_kernel_mutation_restores_the_pending_queue(monkeypatch):
    import neuroterrarium.neural as module

    model = brain()
    model.advance(input_tape(121))
    before = model.snapshot()
    kernel = module._advance_kernel

    def interrupted(*args):
        kernel(*args)
        raise KeyboardInterrupt("simulated cancellation after an actual kernel step")

    monkeypatch.setattr(module, "_advance_kernel", interrupted)
    with pytest.raises(KeyboardInterrupt):
        model.advance([[0, 1]], interventions={0: {"clamp_spikes": [3]}})
    assert model.snapshot() == before


def test_nested_import_arrays_and_non_boolean_recording_are_rejected():
    model = brain()
    before = model.snapshot()
    for key in ("v_mV", "g_mV", "last_spike_steps"):
        bad = {**before, key: [[0]] * 7}
        with pytest.raises(ValueError):
            model.restore(bad)
    with pytest.raises(ValueError, match="flat integer"):
        model.advance([[[0]]])
    with pytest.raises(ValueError, match="boolean"):
        model.advance([[]], record_spikes="false")
    assert model.snapshot() == before


def test_snapshot_rejects_missing_reordered_and_impossible_delayed_events():
    model = SparseBrain(3, [0, 1], [2, 2], [8000, 8000], input_neurons=[0, 1])
    model.advance([[0, 1], []])
    before = model.snapshot()
    # Both source neurons fire at step 1, with delivery at step 19 (slot 0).
    assert before["queue"][0] == [0, 1]
    missing = copy.deepcopy(before)
    missing["queue"][0] = []
    reordered = copy.deepcopy(before)
    reordered["queue"][0] = [1, 0]
    impossible = copy.deepcopy(before)
    impossible["queue"][18] = [0]  # Would require the same source to fire at steps 0 and 1.
    for invalid, message in ((missing, "omits"), (reordered, "threshold order"),
                              (impossible, "consecutive")):
        with pytest.raises(ValueError, match=message):
            model.restore(invalid)
        assert model.snapshot() == before


def test_snapshot_refractory_voltage_and_synaptic_state_are_not_fabricated():
    model = SparseBrain(2, [0], [1], [8000], input_neurons=[0])
    model.advance([[0]] + [[] for _ in range(20)])
    before = model.snapshot()
    assert before["last_spike_steps"][1] == 20
    for key, value in (("v_mV", -51.0), ("g_mV", 1.0)):
        invalid = copy.deepcopy(before)
        invalid[key][1] = value
        with pytest.raises(ValueError, match="refractory state"):
            model.restore(invalid)
        assert model.snapshot() == before
