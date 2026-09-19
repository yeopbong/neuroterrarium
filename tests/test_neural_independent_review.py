"""Independent event-order, encoding and cached-graph probes on small test graphs."""
import copy

import numpy as np
import pytest

from neuroterrarium.interface import MotorReadout, SensoryEncoder
from neuroterrarium.neural import SparseBrain
from neuroterrarium.reference import run_reference
from neuroterrarium.registry import Registry


@pytest.mark.parametrize('seed', [7421, 7423, 7427])
def test_active_signed_csr_and_restored_windows_match_brian_per_spike(seed):
    rng = np.random.default_rng(seed)
    n = 17
    pairs = rng.choice(n*n, 90, replace=False)
    pre, post = pairs//n, pairs % n
    counts = rng.integers(50, 900, 90)*rng.choice([-1,1],90,p=[.3,.7])
    tape = [np.flatnonzero(row).tolist() for row in rng.random((220,4)) < .08]
    initial_g = rng.uniform(-2,2,n)
    schedule = {18:{'disconnect_output':[0]}, 41:{'disconnect_output':[]},
                65:{'clamp_spikes':[5]}, 87:{'clamp_spikes':[]},
                121:{'disconnect_input':[8]}, 149:{'disconnect_input':[]}}
    normal = SparseBrain(n,pre,post,counts,input_neurons=list(range(4)),initial_g_mV=initial_g)
    cached = SparseBrain.from_csr(normal.indptr, normal.post_indices, normal.weights_mV,
                                  normal.input_mask, expected_digest=normal.graph_digest)
    cached.restore(normal.snapshot())
    first = cached.advance(tape[:100], list(range(n)), True,
                           interventions={s:c for s,c in schedule.items() if s<100})
    restored = normal.fork()
    restored.restore(copy.deepcopy(cached.snapshot()))
    second = restored.advance(tape[100:],list(range(n)),True,
                              interventions={s-100:c for s,c in schedule.items() if s>=100})
    expected = run_reference(n,pre,post,counts,tape,input_neurons=list(range(4)),record_indices=list(range(n)),
                             initial_g_mV=initial_g,interventions=schedule)
    assert len(expected.spike_steps)>20
    assert np.any(expected.spike_neurons>=4)
    for field in ('spike_steps','spike_neurons'):
        np.testing.assert_array_equal(np.r_[getattr(first,field),getattr(second,field)],getattr(expected,field))
    for field in ('v_mV','g_mV'):
        np.testing.assert_allclose(np.concatenate([getattr(first,field),getattr(second,field)]),
                                   getattr(expected,field),rtol=0,atol=1e-9)


def test_delivery_time_output_disconnect_differs_from_post_emission_spike_clamp():
    tape = [[0]] + [[] for _ in range(70)]
    def run(changes):
        return SparseBrain(2,[0],[1],[3000],input_neurons=[0]).advance(tape,[0,1],True,interventions=changes)
    before_delivery = run({18:{'disconnect_output':[0]},20:{'disconnect_output':[]}})
    after_delivery = run({20:{'disconnect_output':[0]}})
    after_emission_clamp = run({2:{'clamp_spikes':[0]}})
    assert before_delivery.spike_steps[before_delivery.spike_neurons==0].tolist()==[1]
    assert before_delivery.spike_counts[1]==0
    assert after_delivery.spike_counts[1]>0
    np.testing.assert_array_equal(after_delivery.spike_counts,after_emission_clamp.spike_counts)
    # Delivery at step 19 survives a later output disconnection; suppression
    # at step 18 discards that event and restoration does not resurrect it.
    assert before_delivery.g_mV[19,1]==0
    assert after_delivery.g_mV[19,1]==pytest.approx(825)


def interface_indices():
    registry = Registry.load()
    roots = [int(node['root_id']) for group in registry.groups.values() for node in group['neurons']]
    return registry.resolve(roots), registry.resolve_sides(roots), len(roots)


def test_encoder_has_no_hidden_directional_escape_or_food_coordinate_channel():
    groups,sides,_ = interface_indices()
    a,b = SensoryEncoder(groups,sides,7431),SensoryEncoder(groups,sides,7431)
    left,right = np.zeros(59),np.zeros(59)
    left[33],right[41] = .8,.8
    right[48:52] = [1,.5,.7,.2]  # Different local chemical cue; no extra input grant.
    right[53:55] = 1
    first,second = a.encode(left,steps=400),b.encode(right,steps=400)
    assert a.last_rates_hz==b.last_rates_hz
    assert all(np.array_equal(x,y) for x,y in zip(first,second,strict=True))
    assert a.last_rates_hz['lplc2']==a.last_rates_hz['lc4']==160


def test_readout_uses_registered_side_and_clamp_preserves_neural_filter_state():
    groups,sides,n = interface_indices()
    active = np.zeros(n,dtype=np.int64)
    active[sides['dna02_left']] = 3
    readout = MotorReadout(sides)
    action = readout.action(active)
    assert action[0]>0 and action[1]>0 and action[2]==action[3]==0
    unselected = active.copy()
    unselected[groups['lplc2']] = 100
    other = MotorReadout(sides)
    np.testing.assert_array_equal(other.action(unselected),action)
    clamped = MotorReadout(sides); clamped.clamped=True
    np.testing.assert_array_equal(clamped.action(active),0)
    assert clamped.filtered==readout.filtered
    zero = np.zeros(n,dtype=np.int64)
    for _ in range(100):
        np.testing.assert_array_equal(clamped.action(zero),0)
