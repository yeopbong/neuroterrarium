"""Independent analytic and upstream-input checks for the numerical reference."""

import numpy as np
import pytest

from neuroterrarium.reference import DT_MS, UPSTREAM_RESET, run_reference


def tape(steps, events):
    result = [[] for _ in range(steps)]
    for step, indices in events.items():
        result[step] = indices
    return result


def two_cell(*, events=None, count=400, steps=150, **kwargs):
    return run_reference(
        2,
        [0],
        [1],
        [count],
        tape(steps, {0: [0]} if events is None else events),
        input_neurons=[0],
        record_indices=[0, 1],
        **kwargs,
    )


def test_linear_solution_has_two_exponentials_and_voltage_units():
    result = run_reference(
        1, [], [], [], tape(50, {}), input_neurons=[], record_indices=[0],
        initial_v_mV=[-51], initial_g_mV=[2],
    )
    elapsed_ms = (np.arange(50) + 1) * DT_MS
    a, b = np.exp(-elapsed_ms / 20), np.exp(-elapsed_ms / 5)
    expected_v = -52 + a + (5 / 15) * (a - b) * 2
    np.testing.assert_allclose(result.v_mV[:, 0], expected_v, atol=1e-11, rtol=0)
    np.testing.assert_allclose(result.g_mV[:, 0], 2 * b, atol=1e-11, rtol=0)
    assert result.spike_steps.size == 0


def test_delay_and_input_are_distinct_scheduling_stages():
    result = two_cell()
    assert result.spike_steps[result.spike_neurons == 0].tolist() == [1]
    np.testing.assert_array_equal(result.g_mV[:19, 1], 0)
    assert result.g_mV[19, 1] == pytest.approx(110)
    assert result.v_mV[19, 1] == pytest.approx(-52)
    assert result.v_mV[20, 1] > -52
    assert np.count_nonzero(result.spike_neurons == 1) > 0


def test_inhibitory_sign_is_not_removed():
    result = two_cell(count=-400)
    assert result.g_mV[19, 1] == pytest.approx(-110)
    assert result.v_mV[20, 1] < -52
    assert np.count_nonzero(result.spike_neurons == 1) == 0


def test_output_input_and_spike_interventions_have_different_objects():
    normal = two_cell()
    output = two_cell(disconnect_output=[0])
    incoming = two_cell(disconnect_input=[1])
    firing = two_cell(clamp_spikes=[1])
    assert np.count_nonzero(normal.spike_neurons == 1) > 0
    for result in (output, incoming):
        assert result.spike_steps.tolist() == [1]
        np.testing.assert_array_equal(result.g_mV[:, 1], 0)
    assert firing.spike_steps.tolist() == [1]
    assert np.max(firing.v_mV[:, 1]) > -45
    assert np.max(firing.g_mV[:, 1]) > 0
    # Disconnecting network input of the external input cell leaves external
    # drive in place. Clamping its threshold prevents its outgoing events.
    external_intact = two_cell(disconnect_input=[0])
    source_clamped = two_cell(clamp_spikes=[0])
    np.testing.assert_array_equal(external_intact.spike_steps, normal.spike_steps)
    assert source_clamped.spike_steps.size == 0


def test_delayed_events_use_weights_at_delivery_and_restore_is_explicit():
    cut = two_cell(interventions={18: {"disconnect_output": [0]}})
    restored = two_cell(interventions={18: {"disconnect_output": [0]},
                                       19: {"disconnect_output": []}})
    late_restore = two_cell(interventions={18: {"disconnect_output": [0]},
                                           20: {"disconnect_output": []}})
    assert cut.g_mV[19, 1] == 0
    assert late_restore.g_mV[19, 1] == 0
    assert np.max(late_restore.g_mV[:, 1]) == 0
    assert restored.g_mV[19, 1] == pytest.approx(110)


def test_sham_leaves_active_spikes_and_trajectories_identical():
    normal = two_cell()
    sham = two_cell(interventions={0: {}, 10: {}, 80: {}})
    assert len(normal.spike_steps) > 1
    for attribute in ("spike_steps", "spike_neurons", "v_mV", "g_mV"):
        np.testing.assert_array_equal(getattr(normal, attribute), getattr(sham, attribute))


def test_refractory_boundary_drops_earlier_arrivals_but_accepts_step_22():
    early = two_cell(events={0: [0], 22: [0]}, count=8000, steps=70)
    boundary = two_cell(events={0: [0], 23: [0]}, count=8000, steps=70)
    assert early.spike_steps[early.spike_neurons == 1].tolist() == [20]
    assert boundary.spike_steps[boundary.spike_neurons == 1].tolist() == [20, 43]
    np.testing.assert_array_equal(early.g_mV[20:, 1], 0)
    assert boundary.g_mV[42, 1] == pytest.approx(2200)


def test_original_reset_and_saturated_poisson_input_match_tape():
    import brian2 as b2

    previous_target = b2.prefs.codegen.target
    try:
        b2.prefs.codegen.target = "numpy"
        group = b2.NeuronGroup(
            1,
            """dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
               dg/dt = -g / tau : volt (unless refractory)
               rfc : second""",
            method="linear", threshold="v > v_th", reset=UPSTREAM_RESET,
            refractory="rfc", dt=0.1*b2.ms,
            namespace={"v_0": -52*b2.mV, "v_rst": -52*b2.mV,
                       "v_th": -45*b2.mV, "t_mbr": 20*b2.ms, "tau": 5*b2.ms},
        )
        group.v = -52*b2.mV
        group.g = 0*b2.mV
        group.rfc = 0*b2.ms
        stimulus = b2.PoissonInput(group, "v", N=1, rate=10000*b2.Hz,
                                   weight=68.75*b2.mV)
        spikes = b2.SpikeMonitor(group)
        states = b2.StateMonitor(group, ["v", "g"], record=[0], when="end")
        b2.Network(group, stimulus, spikes, states).run(4*b2.ms)
        result = run_reference(1, [], [], [], [[0] for _ in range(40)],
                               input_neurons=[0], record_indices=[0])
        assert len(spikes.i) > 0
        np.testing.assert_array_equal(result.spike_steps,
                                      np.rint(np.asarray(spikes.t/b2.ms)/DT_MS))
        np.testing.assert_allclose(result.v_mV, np.asarray(states.v/b2.mV).T,
                                   atol=1e-11, rtol=0)
        np.testing.assert_array_equal(result.g_mV, np.asarray(states.g/b2.mV).T)
    finally:
        b2.prefs.codegen.target = previous_target


@pytest.mark.parametrize("broken", [1.0, -1, 2])
def test_bad_index_cannot_be_silently_coerced(broken):
    with pytest.raises(ValueError):
        two_cell(events={0: [broken]})


def test_duplicate_events_and_unregistered_inputs_are_rejected():
    with pytest.raises(ValueError, match="repeated"):
        two_cell(events={0: [0, 0]})
    with pytest.raises(ValueError, match="unregistered"):
        two_cell(events={0: [1]})


def test_corrupt_weights_and_schedule_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        two_cell(count=np.nan)
    with pytest.raises(ValueError, match="integer synapse"):
        two_cell(count=2.5)
    with pytest.raises(ValueError, match="unknown"):
        two_cell(interventions={0: {"pretend_blind": [1]}})
    with pytest.raises(ValueError, match="outside"):
        two_cell(interventions={150: {}})
