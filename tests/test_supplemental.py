"""Independent boundary probes; no official stress seeds are evaluated here."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neuroterrarium.controllers import LearnedController, Policy, apply_action
from neuroterrarium.supplemental import (balanced_allocation, evaluate_supplement, make_scene,
                                       run_trial, summarize_supplement, trial_metrics, validate_protocol)
from neuroterrarium.world import ACTION_DT, Body, Food, World

MODELS = [f'{kind}-{seed}' for kind in ('feedforward','recurrent','hybrid') for seed in (101,202,303)]


class Probe:
    def __init__(self, action=(0,0,0,0), fail_after=None):
        self.action, self.fail_after, self.seen = np.asarray(action), fail_after, []

    def act(self, observation):
        if self.fail_after == len(self.seen):
            raise RuntimeError('Test backend interruption')
        self.seen.append(observation.copy())
        return self.action.copy()


def test_every_controller_occupies_every_slot_three_times():
    assignments = [balanced_allocation(MODELS, i) for i in range(30)]
    for identity in ['connectome', *MODELS]:
        assert [sum(a[slot] == identity for a in assignments) for slot in range(10)] == [3]*10
    assert assignments == [balanced_allocation(list(reversed(MODELS)), i) for i in range(30)]


def test_frozen_protocol_is_separate_and_budget_checked():
    root = Path(__file__).parents[1]
    config = json.loads((root/'configs/evaluate-supplement-v1.json').read_text())
    validate_protocol(config)
    core = json.loads((root/'configs/evaluate-v1.json').read_text())
    assert not set(config['stress_seeds']) & set(core['environment_seeds'])
    assert not set(config['shared_seeds']) & set(config['stress_seeds'])
    assert config['stress_windows']*ACTION_DT == 1.5
    for field, value in [('wall_budget_seconds', 901), ('stress_windows', 50), ('visual_noise_std', .1)]:
        changed = copy.deepcopy(config); changed[field] = value
        with pytest.raises(ValueError): validate_protocol(changed)


def test_noise_is_in_actual_observation_and_leaves_other_random_streams_untouched():
    a, ae = make_scene('stress', 7301); b, be = make_scene('stress', 7301)
    clean, noisy = Probe(), Probe()
    _, aa, _ = run_trial(a, ae, [clean], windows=12, noise_std=0)
    _, ba, _ = run_trial(b, be, [noisy], windows=12, noise_std=.15)
    assert a.snapshot() == b.snapshot()  # identical actions, same exogenous events
    np.testing.assert_array_equal(aa['unperturbed_observation'], ba['unperturbed_observation'])
    np.testing.assert_array_equal(aa['observation'][:,:,48:], ba['observation'][:,:,48:])
    assert not np.array_equal(aa['observation'][:,:,16:48], ba['observation'][:,:,16:48])
    np.testing.assert_array_equal(np.asarray(noisy.seen), ba['observation'][:,0])
    assert np.all(ba['observation'][:,:,16:32] >= 0)
    assert np.all(np.abs(ba['observation'][:,:,32:48]) <= 1)


def test_shared_food_competition_is_conservative_and_synchronous():
    world = World(7303, 10)
    world.bodies = [Body(40+2*np.cos(i*np.pi/5),28+2*np.sin(i*np.pi/5),0) for i in range(10)]
    world.foods, world.obstacles, world.stimuli = [Food(40,28,.004,3)], [], []
    events = {'case': 'S_shared', 'family': 'shared', 'environment_seed': 7303,
              'onset_step': 10, 'stimuli_after_onset': []}
    initial = world.observe().copy()
    probes = [Probe((0,0,0,1)) for _ in range(10)]
    row, arrays, _ = run_trial(world, events, probes, windows=1)
    assert row['status'] == 'completed'
    assert row['total_food'] == pytest.approx(.004, abs=1e-12)
    np.testing.assert_array_equal(np.asarray([p.seen[0] for p in probes]), initial)
    np.testing.assert_allclose(arrays['body'][0,:,5], .0004, rtol=0, atol=1e-14)


def test_timeout_and_failure_keep_partial_records_and_null_outcomes():
    world, events = make_scene('stress',7305)
    row, arrays, events = run_trial(world,events,[Probe()],windows=10,timeout_seconds=0)
    assert row['status']=='timeout' and row['completed_windows']==0
    assert row['total_food'] is None and arrays['body'].shape==(0,1,7)
    world, events = make_scene('shared',7307)
    row, arrays, _ = run_trial(world,events,[Probe(fail_after=2),*[Probe() for _ in range(9)]],windows=8)
    assert row['status']=='failed' and row['failure']['stage']=='controller'
    assert world.step==row['completed_windows']==2
    assert row['total_food'] is None and arrays['body'].shape==(2,10,7)
    json.dumps(row, allow_nan=False)


def test_cancellation_is_not_completed_and_inapplicable_channels_are_missing():
    world, events = make_scene('stress',7309)
    row, arrays, _ = run_trial(world,events,[Probe()],windows=8,cancel=lambda:world.step==2)
    assert row['status']=='interrupted' and row['completed_windows']==2
    assert np.isnan(arrays['motor_groups_hz']).all()
    assert np.isnan(arrays['raw_policy_log_probability']).all()


def test_hybrid_raw_actions_reproduce_the_training_application_pipeline():
    torch.set_num_threads(1)
    world, events = make_scene('stress',7311)
    policy = LearnedController(Policy('hybrid',202,8),7311,deterministic=True)
    row, arrays, _ = run_trial(world,events,[policy],windows=14,noise_std=.15)
    applied = apply_action(torch.from_numpy(arrays['raw_policy_action'][:,0]),
                           torch.from_numpy(arrays['observation'][:,0]), hybrid=True)
    np.testing.assert_allclose(arrays['applied_action'][:,0],applied.action.numpy(),rtol=0,atol=1e-7)
    assert row['status']=='completed'
    assert np.isfinite(arrays['raw_policy_log_probability']).all()


def test_recomputed_shared_metrics_reject_corrupted_food_or_clock_shape():
    world, events = make_scene('shared',7313)
    row, arrays, events = run_trial(world,events,[Probe() for _ in range(10)],windows=3)
    assert trial_metrics(arrays,events,status='completed')['total_food']==row['total_food']
    broken = {**arrays,'body':arrays['body'].copy()}; broken['body'][-1,0,5]+=.5
    with pytest.raises(ValueError,match='conservation'): trial_metrics(broken,events,status='completed')
    with pytest.raises(ValueError,match='timing'):
        trial_metrics({**arrays,'inference_seconds':arrays['inference_seconds'][:-1]},events,status='completed')


def test_cancelled_full_plan_does_not_load_graph_or_claim_completion(tmp_path, monkeypatch):
    import neuroterrarium.supplemental as supplement
    config = json.loads((Path(__file__).parents[1]/'configs/evaluate-supplement-v1.json').read_text())
    monkeypatch.setattr(supplement, 'load_model_set', lambda _: (dict.fromkeys(MODELS), {'models': MODELS}))
    monkeypatch.setattr(supplement, '_load_brain', lambda *_: pytest.fail('Cancelled run loaded neural graph'))
    result = evaluate_supplement(config, tmp_path, tmp_path, tmp_path/'out', cancel=lambda: True)
    assert result == {'status': 'incomplete', 'records': 630}
    lock = json.loads((tmp_path/'out/protocol-lock.json').read_text())
    assert lock['storage_profile_sha256'] == supplement.sha256_file(
        supplement.default_path().with_name('brain-storage-v1.json'))
    report = json.loads((tmp_path/'out/results.json').read_text())
    assert all(row['status']=='not_started' and row['reason']=='cancelled' for row in report['records'])
    assert not list((tmp_path/'out/raw').iterdir())
    report['status'] = 'completed'
    (tmp_path/'out/results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='completion status'): summarize_supplement(tmp_path/'out')
    report['status'] = 'incomplete'
    (tmp_path/'out/results.json').write_text(json.dumps(report))
    (tmp_path/'out/budget.json').unlink()
    with pytest.raises(ValueError,match='cumulative execution budget'):
        evaluate_supplement(config,tmp_path,tmp_path,tmp_path/'out',cancel=lambda:True)


def test_supplement_loader_reuses_verified_cached_brain_without_coo(tmp_path, monkeypatch):
    import neuroterrarium.brain_cache as cache
    import neuroterrarium.data as data
    import neuroterrarium.neural as neural
    import neuroterrarium.supplemental as supplement

    calls = []
    roots = np.asarray([720575940622838154, 720575940632499757], dtype=np.int64)
    groups, sides = {'gf': [0, 1]}, {'gf': {'left': [0], 'right': [1]}}
    brain = SimpleNamespace(graph_digest='a'*64)
    cached = SimpleNamespace(root_ids=roots, brain=brain, summary={'neurons': 2})
    registry = SimpleNamespace(
        resolve=lambda ids: (calls.append(('groups', ids)) or groups),
        resolve_sides=lambda ids: (calls.append(('sides', ids)) or sides))
    monkeypatch.setattr(cache, 'ensure_cache', lambda directory: calls.append(('ensure', directory)))
    monkeypatch.setattr(cache, 'load_cache', lambda directory: (calls.append(('load', directory)) or cached))
    monkeypatch.setattr(supplement, 'Registry', SimpleNamespace(load=lambda: registry))
    monkeypatch.setattr(data, 'load_graph', lambda *_a, **_k: pytest.fail('Loader reconstructed COO graph'))
    monkeypatch.setattr(neural.SparseBrain, '__init__', lambda *_a, **_k: pytest.fail('Loader rebuilt neural storage'))

    loaded, actual_groups, actual_sides = supplement._load_brain(tmp_path, tmp_path, 'b'*64)
    assert loaded is brain and actual_groups is groups and actual_sides is sides
    assert calls[:2] == [('ensure', tmp_path), ('load', tmp_path)]
    assert all(ids is roots for _, ids in calls[2:])
    assert json.loads((tmp_path/'graph-summary.json').read_text()) == {
        'profile': cached.summary, 'neural_graph_sha256': 'a'*64, 'protocol_sha256': 'b'*64}


def test_supplement_cache_corruption_fails_without_substitution(tmp_path, monkeypatch):
    import neuroterrarium.brain_cache as cache
    import neuroterrarium.supplemental as supplement

    monkeypatch.setattr(cache, 'ensure_cache', lambda _directory: None)
    def corrupt(_directory):
        raise ValueError('Complete CSR checksum mismatch')
    monkeypatch.setattr(cache, 'load_cache', corrupt)
    with pytest.raises(ValueError, match='checksum mismatch'):
        supplement._load_brain(tmp_path, tmp_path, 'b'*64)
    assert not (tmp_path/'graph-summary.json').exists()


def test_summary_recomputes_raw_metrics_and_rejects_forged_duration(tmp_path):
    from neuroterrarium.data import sha256_file
    from neuroterrarium.training import config_digest
    (tmp_path/'raw').mkdir()
    plan = [{'family':'stress','condition':c,'environment_seed':7315,'controller':'connectome',
             'allocation':['connectome'],'windows':3} for c in ('clean','visual_noise')]
    lock = {'config':{'stress_seeds':[7315]},'plan':plan}
    (tmp_path/'protocol-lock.json').write_text(json.dumps(lock))
    rows = []
    for spec in plan:
        world, events = make_scene('stress',7315)
        row, arrays, events = run_trial(world,events,[Probe()],windows=3)
        name = spec['condition']+'.npz'
        np.savez_compressed(tmp_path/'raw'/name,**arrays)
        row.update(**spec,events=events,protocol_sha256=config_digest(lock),trajectory=name,
                   trajectory_sha256=sha256_file(tmp_path/'raw'/name))
        rows.append(row)
    result = {'protocol_sha256':config_digest(lock),'status':'completed','records':rows}
    (tmp_path/'results.json').write_text(json.dumps(result))
    summarize_supplement(tmp_path)
    rows[0]['total_food'] = 100
    (tmp_path/'results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='raw trajectory'): summarize_supplement(tmp_path)
    rows[0]['total_food'] = 0
    rows[0]['completed_windows'] = 4
    (tmp_path/'results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='clock mismatch'): summarize_supplement(tmp_path)
