"""Small non-scientific fixtures exercise supplement persistence and recovery."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

import neuroterrarium.supplemental as supplement
from neuroterrarium.training import config_digest


@pytest.fixture
def short_run(tmp_path, monkeypatch):
    # These two three-window trials test bookkeeping, never the formal split.
    root = Path(__file__).parents[1]
    config = json.loads((root/'configs/evaluate-supplement-v1.json').read_text())
    config.update(stress_seeds=[7317], shared_seeds=[], stress_windows=3)
    monkeypatch.setattr(supplement, 'validate_protocol', lambda _config: None)
    monkeypatch.setattr(supplement, 'load_model_set', lambda _path: ({}, {'models': []}))
    configs = tmp_path/'configs'; configs.mkdir()
    for name in ('interface-v783.json','brain-interface.json','data-v783.json','brain-storage-v1.json'):
        (configs/name).write_bytes((root/'configs'/name).read_bytes())
    summary = {'neurons': 2, 'scope': 'synthetic test fixture'}
    (configs/'brain-storage-v1.json').write_text(json.dumps({
        'graph_digest': 'a'*64, 'summary_sha256': config_digest(summary)}))
    monkeypatch.setattr(supplement, 'default_path', lambda: configs/'interface-v783.json')
    def loader(_directory, output, digest):
        supplement.atomic_json(output/'graph-summary.json', {
            'profile': summary, 'neural_graph_sha256': 'a'*64, 'protocol_sha256': digest})
        return object(), {}, {}
    monkeypatch.setattr(supplement, '_load_brain', loader)
    class Controller:
        def act(self, _observation): return np.asarray([.2, 0., 0., 0.])
    monkeypatch.setattr(supplement, '_controller', lambda *_args: Controller())
    return config, tmp_path/'output'


def run(fixture, **kwargs):
    config, output = fixture
    return supplement.evaluate_supplement(config, output.parent, output.parent, output, **kwargs)


def inventory(directory):
    return {path.relative_to(directory).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


def test_completion_is_committed_after_verified_summary(short_run, monkeypatch):
    config, output = short_run
    write = supplement.atomic_json
    commits = []
    def observe(path, value):
        if path.name == 'summary.json':
            commits.append('summary')
            assert json.loads((output/'results.json').read_text())['status'] == 'summarizing'
            assert json.loads((output/'budget.json').read_text())['reserved_seconds'] > 0
        elif path.name == 'results.json' and value['status'] == 'completed':
            commits.append('completed')
            assert (output/'summary.json').is_file()
            budget = json.loads((output/'budget.json').read_text())
            assert budget['status'] == 'completed' and budget['reserved_seconds'] == 0
            assert budget['elapsed_seconds'] >= value['wall_seconds']
        return write(path, value)
    monkeypatch.setattr(supplement, 'atomic_json', observe)
    assert run(short_run) == {'status': 'completed', 'records': 2}
    assert commits == ['summary', 'completed']


def test_summary_failure_cannot_leave_completed_result(short_run, monkeypatch):
    def fail(*_args, **_kwargs): raise ValueError('Deliberate statistics failure')
    monkeypatch.setattr(supplement, 'summarize_supplement', fail)
    assert run(short_run) == {'status': 'failed', 'records': 2}
    output = short_run[1]
    result = json.loads((output/'results.json').read_text())
    assert result['failure'] == {'kind': 'ValueError', 'stage': 'summary'}
    assert result['status'] == 'failed' and not (output/'summary.json').exists()
    assert all(row['status']=='completed' for row in result['records'])
    assert json.loads((output/'budget.json').read_text())['status'] == 'failed'


def test_completed_verification_and_summary_are_read_only(short_run, monkeypatch):
    assert run(short_run)['status'] == 'completed'
    output = short_run[1]
    before = inventory(output)
    monkeypatch.setattr(supplement, '_load_brain', lambda *_args: pytest.fail('Completed verification loaded a brain'))
    report = supplement.summarize_supplement(output, persist=False)
    assert report == json.loads((output/'summary.json').read_text())
    assert inventory(output) == before
    result = run(short_run)
    assert result['status'] == 'verified' and result['operation'] == 'read_only_verification'
    assert inventory(output) == before
    result = run(short_run, cancel=lambda: True)
    assert result['status'] == 'cancelled' and result['historical_status'] == 'completed'
    assert inventory(output) == before


@pytest.mark.parametrize('mutation', ['summary', 'graph', 'budget', 'record', 'missing_summary'])
def test_completed_verification_rejects_corruption_without_rewriting(short_run, mutation):
    assert run(short_run)['status'] == 'completed'
    output = short_run[1]
    if mutation == 'missing_summary': (output/'summary.json').unlink()
    else:
        name = {'summary':'summary.json', 'graph':'graph-summary.json', 'budget':'budget.json',
                'record':'results.json'}[mutation]
        value = json.loads((output/name).read_text())
        if mutation == 'summary': value['stress_individual'][0]['metrics']['food'] = 999
        elif mutation == 'graph': value['neural_graph_sha256'] = '0'*64
        elif mutation == 'budget': value['reserved_seconds'] = 1
        else: value['records'][0]['total_food'] = 999
        (output/name).write_text(json.dumps(value))
    before = inventory(output)
    with pytest.raises(ValueError): run(short_run)
    assert inventory(output) == before


@pytest.mark.parametrize('name', ['orphan.npz', 'unexpected.json', 'unfinished.npz.partial'])
def test_unplanned_raw_files_are_never_overwritten(short_run, name):
    output = short_run[1]
    raw = output/'raw'; raw.mkdir(parents=True)
    original = raw/name; original.write_bytes(b'unfinished evidence')
    with pytest.raises(ValueError, match='unplanned or partial'): run(short_run)
    assert original.read_bytes() == b'unfinished evidence'
    assert not (output/'budget.json').exists()


@pytest.mark.parametrize('suffix', ['.npz', '.json'])
def test_planned_orphan_pair_is_preserved(short_run, suffix):
    config, output = short_run
    raw = output/'raw'; raw.mkdir(parents=True)
    name = supplement._identifier(supplement._plan(config, {})[0])+suffix
    original = raw/name; original.write_bytes(b'orphaned evidence')
    with pytest.raises(ValueError, match='Orphan'): run(short_run)
    assert original.read_bytes() == b'orphaned evidence'


@pytest.mark.parametrize(('key','value'), [
    ('status','unknown'), ('trajectory','different.npz'), ('completed_windows',True),
    ('completed_windows',1), ('simulation_seconds',.5), ('wall_seconds',-1)])
def test_resume_rejects_invalid_existing_record_before_loading(short_run, monkeypatch, key, value):
    assert run(short_run)['status'] == 'completed'
    output = short_run[1]
    result = json.loads((output/'results.json').read_text()); result['status'] = 'incomplete'
    (output/'results.json').write_text(json.dumps(result))
    record = next((output/'raw').glob('*.json'))
    row = json.loads(record.read_text()); row[key] = value; record.write_text(json.dumps(row))
    before = inventory(output)
    monkeypatch.setattr(supplement, '_load_brain', lambda *_args: pytest.fail('Invalid resume loaded a brain'))
    with pytest.raises(ValueError, match='record integrity'): run(short_run)
    assert inventory(output) == before


def test_recovered_in_flight_reservation_consumes_cumulative_budget(short_run, monkeypatch):
    assert run(short_run, cancel=lambda: True)['status'] == 'incomplete'
    output = short_run[1]
    path = output/'budget.json'; budget = json.loads(path.read_text())
    budget.update(status='running', elapsed_seconds=880., reserved_seconds=45., in_flight_episode='interrupted')
    path.write_text(json.dumps(budget))
    monkeypatch.setattr(supplement, '_load_brain', lambda *_args: pytest.fail('Exhausted resume loaded a brain'))
    assert run(short_run)['status'] == 'incomplete'
    resumed = json.loads(path.read_text())
    assert resumed['elapsed_seconds'] >= 925 and resumed['recovered_reservation_seconds'] == 45
    assert resumed['attempt'] == budget['attempt']+1 and resumed['reserved_seconds'] == 0
    assert all(row['reason']=='wall_budget' for row in json.loads((output/'results.json').read_text())['records'])


def test_active_trial_and_cache_load_have_crash_reservations(short_run, monkeypatch):
    original_loader, original_trial = supplement._load_brain, supplement.run_trial
    path = short_run[1]/'budget.json'
    observed = []
    def loader(*args):
        budget = json.loads(path.read_text())
        observed.append('cache')
        assert budget['reserved_seconds'] > 0 and budget['in_flight_episode'] is None
        return original_loader(*args)
    def trial(*args, **kwargs):
        budget = json.loads(path.read_text())
        observed.append('trial')
        assert budget['reserved_seconds'] == kwargs['timeout_seconds']
        assert budget['in_flight_episode'].startswith('stress--connectome--')
        return original_trial(*args, **kwargs)
    monkeypatch.setattr(supplement, '_load_brain', loader)
    monkeypatch.setattr(supplement, 'run_trial', trial)
    assert run(short_run)['status'] == 'completed'
    assert observed == ['cache', 'trial', 'trial']


def test_missing_trial_resumes_without_rerunning_valid_record(short_run, monkeypatch):
    original_trial = supplement.run_trial
    calls = []
    def trial(*args, **kwargs):
        calls.append(1)
        return original_trial(*args, **kwargs)
    monkeypatch.setattr(supplement, 'run_trial', trial)
    assert run(short_run, cancel=lambda: len(calls)>=1)['status'] == 'incomplete'
    # The cancelled trial is kept; no completed or interrupted unit is retried.
    raw = short_run[1]/'raw'
    first = inventory(raw)
    assert len(first) == 2
    assert run(short_run)['status'] == 'incomplete'
    assert len(calls) == 2
    assert all(inventory(raw)[name] == value for name,value in first.items())


def test_summary_stop_check_interrupts_without_output_replacement(short_run):
    assert run(short_run)['status'] == 'completed'
    output = short_run[1]
    before = inventory(output)
    def stop(): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): supplement.summarize_supplement(output, persist=False, stop_check=stop)
    assert inventory(output) == before


@pytest.mark.parametrize('stage', ['cache', 'trial'])
def test_abrupt_exit_reservation_is_charged_on_real_resume(short_run, monkeypatch, stage):
    """A process exit must not receive a new execution budget on restart."""
    target = '_load_brain' if stage == 'cache' else 'run_trial'
    original = getattr(supplement, target)
    def abrupt(*_args, **_kwargs): raise SystemExit(23)
    monkeypatch.setattr(supplement, target, abrupt)
    with pytest.raises(SystemExit, match='23'): run(short_run)
    output = short_run[1]
    interrupted = json.loads((output/'budget.json').read_text())
    reservation = interrupted['reserved_seconds']
    assert reservation == (600 if stage == 'cache' else 45)
    assert interrupted['status'] == 'running'
    assert not (output/'results.json').exists()
    monkeypatch.setattr(supplement, target, original)
    assert run(short_run)['status'] == 'completed'
    resumed = json.loads((output/'budget.json').read_text())
    assert resumed['recovered_reservation_seconds'] == reservation
    assert resumed['elapsed_seconds'] >= interrupted['elapsed_seconds'] + reservation
    assert resumed['attempt'] == interrupted['attempt'] + 1


def test_interrupted_final_commit_recovers_without_reexecuting_trials(short_run, monkeypatch):
    output = short_run[1]
    original = supplement.atomic_json
    def interrupt_final(path, value):
        if path.name == 'results.json' and value['status'] == 'completed':
            raise KeyboardInterrupt()
        return original(path, value)
    monkeypatch.setattr(supplement, 'atomic_json', interrupt_final)
    with pytest.raises(KeyboardInterrupt): run(short_run)
    assert json.loads((output/'results.json').read_text())['status'] == 'summarizing'
    before = inventory(output/'raw')
    monkeypatch.setattr(supplement, 'atomic_json', original)
    monkeypatch.setattr(supplement, '_load_brain', lambda *_args: pytest.fail('Summary recovery loaded a brain'))
    monkeypatch.setattr(supplement, 'run_trial', lambda *_args, **_kwargs: pytest.fail('Summary recovery reran a trial'))
    assert run(short_run)['status'] == 'completed'
    assert inventory(output/'raw') == before


def test_completed_storage_profile_mismatch_is_rejected_without_writes(short_run):
    assert run(short_run)['status'] == 'completed'
    output = short_run[1]
    storage = supplement.default_path().with_name('brain-storage-v1.json')
    value = json.loads(storage.read_text()); value['graph_digest'] = '0'*64
    storage.write_text(json.dumps(value))
    before = inventory(output)
    with pytest.raises(ValueError, match='source, model identity or frozen protocol changed'):
        run(short_run)
    assert inventory(output) == before


def test_resource_revision_changes_only_the_cumulative_budget():
    config_dir = Path(__file__).parents[1]/'configs'
    original = json.loads((config_dir/'evaluate-supplement-v1.json').read_text())
    revised = json.loads((config_dir/'evaluate-supplement-v1b.json').read_text())
    assert original['wall_budget_seconds'] == 900
    assert revised['wall_budget_seconds'] == 1800
    assert revised['resources'] == original['resources'].replace(
        'Nine hundred cumulative wall seconds', 'Eighteen hundred cumulative wall seconds')
    assert {key for key in original.keys() | revised.keys()
            if original.get(key) != revised.get(key)} == {'wall_budget_seconds', 'resources'}
    supplement.validate_protocol(original)
    supplement.validate_protocol(revised)


@pytest.mark.parametrize('budget', [0, 899, 901, 1799, 1801, True, 900.0, '1800'])
def test_unfrozen_supplement_resource_budget_is_rejected(budget):
    config = json.loads((Path(__file__).parents[1]/'configs/evaluate-supplement-v1.json').read_text())
    config['wall_budget_seconds'] = budget
    with pytest.raises(ValueError, match='frozen resource version'):
        supplement.validate_protocol(config)


@pytest.mark.parametrize('mutation', ['old_allowance', 'extra_text', 'changed_memory_text'])
def test_resource_revision_rejects_inaccurate_or_altered_description(mutation):
    config = json.loads((Path(__file__).parents[1]/'configs/evaluate-supplement-v1b.json').read_text())
    if mutation == 'old_allowance':
        config['resources'] = config['resources'].replace('Eighteen hundred', 'Nine hundred')
    elif mutation == 'extra_text': config['resources'] += ' More processes are allowed.'
    else: config['resources'] = config['resources'].replace('at most one active', 'two active')
    with pytest.raises(ValueError, match='resource description'):
        supplement.validate_protocol(config)


@pytest.mark.parametrize(('field', 'value'), [
    ('stress_windows', 76), ('shared_windows', 101), ('visual_noise_std', .16),
    ('conditions', ['visual_noise', 'clean']), ('neural_repeats', 2),
    ('stress_seeds', list(range(190002, 190032))),
    ('shared_seeds', list(range(290002, 290032))),
    ('episode_timeout_seconds', 46), ('memory_reserve_bytes', 2*2**30),
    ('disk_reserve_bytes', 9*2**30),
])
def test_resource_revision_does_not_relax_science_or_other_limits(field, value):
    config = json.loads((Path(__file__).parents[1]/'configs/evaluate-supplement-v1b.json').read_text())
    config[field] = value
    with pytest.raises(ValueError): supplement.validate_protocol(config)


@pytest.mark.parametrize(('status','expected'), [('completed',0),('verified',0),('incomplete',1),('failed',1),('cancelled',1)])
def test_supplement_wrapper_reports_real_verification_status(tmp_path, monkeypatch, capsys, status, expected):
    path = Path(__file__).parents[1]/'scripts/evaluate_supplement.py'
    spec = importlib.util.spec_from_file_location('supplement_wrapper', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config = tmp_path/'protocol.json'; config.write_text('{}')
    monkeypatch.setattr(module, 'evaluate_supplement', lambda *_args, **_kwargs: {'status': status})
    monkeypatch.setattr(module.signal, 'signal', lambda *_args: None)
    monkeypatch.setattr(sys, 'argv', [str(path), '--config',str(config),'--data',str(tmp_path),'--output',str(tmp_path/'out')])
    assert module.main() == expected
    assert json.loads(capsys.readouterr().out)['status'] == status
