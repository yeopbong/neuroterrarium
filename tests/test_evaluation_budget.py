"""Budget and cancellation checks use development seeds and no complete graph."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neuroterrarium import evaluation as ev


LIMITS = {'max_total_wall_seconds':10800,'max_process_rss_bytes':2*2**30,
          'min_available_memory_bytes':3*2**30,'min_free_disk_bytes':10*2**30}


@pytest.fixture
def resources(monkeypatch):
    values={'rss':100*2**20,'available':5*2**30,'free':20*2**30}
    monkeypatch.setattr(ev.psutil,'Process',lambda:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=values['rss'])))
    monkeypatch.setattr(ev.psutil,'virtual_memory',lambda:SimpleNamespace(available=values['available']))
    monkeypatch.setattr(ev.psutil,'disk_usage',lambda path:SimpleNamespace(free=values['free']))
    return values


def test_budget_accumulates_prior_run_time_without_charging_between_run_idle(tmp_path,monkeypatch,resources):
    clock=[0.]
    monkeypatch.setattr(ev.time,'perf_counter',lambda:clock[0])
    first=ev.EvaluationBudget(tmp_path,'fixed-protocol',LIMITS)
    first.checkpoint('running',episode='development-case',reserve_seconds=120)
    clock[0]=4.;first.checkpoint('cancelled')
    clock[0]=100.
    second=ev.EvaluationBudget(tmp_path,'fixed-protocol',LIMITS)
    assert second.elapsed==4
    clock[0]=107.
    assert second.elapsed==11
    second.checkpoint('completed')
    saved=json.loads((tmp_path/'budget.json').read_text())
    assert saved['elapsed_seconds']==11 and saved['reserved_seconds']==0 and saved['attempt']==2
    assert not list(tmp_path.glob('*.tmp'))


def test_unclosed_operation_charges_its_reservation_on_resume(tmp_path,monkeypatch,resources):
    clock=[0.];monkeypatch.setattr(ev.time,'perf_counter',lambda:clock[0])
    first=ev.EvaluationBudget(tmp_path,'protocol',LIMITS)
    clock[0]=5.;first.checkpoint('running',episode='development-case',reserve_seconds=120)
    clock[0]=100.
    second=ev.EvaluationBudget(tmp_path,'protocol',LIMITS)
    assert second.elapsed==125 and second.recovered_reservation==120
    second.checkpoint('cancelled')
    with pytest.raises(ValueError,match='different protocol'):
        ev.EvaluationBudget(tmp_path,'changed-protocol',LIMITS)


def test_budget_exhaustion_and_cancel_are_explicit_noncompletion(tmp_path,monkeypatch,resources):
    clock=[0.];monkeypatch.setattr(ev.time,'perf_counter',lambda:clock[0])
    limits={**LIMITS,'max_total_wall_seconds':5}
    budget=ev.EvaluationBudget(tmp_path,'protocol',limits,cancel_file=tmp_path/'cancel')
    clock[0]=5.
    with pytest.raises(ev.EvaluationStopped) as error:budget.check()
    assert error.value.status=='budget_exhausted'
    (tmp_path/'cancel').touch()
    with pytest.raises(ev.EvaluationStopped) as error:budget.check()
    assert error.value.status=='cancelled'


@pytest.mark.parametrize('field,value,reason',[('rss',2*2**30+1,'process_rss'),
    ('available',3*2**30-1,'system_available_memory'),('free',10*2**30-1,'free_disk')])
def test_budget_enforces_all_resource_limits(tmp_path,resources,field,value,reason):
    budget=ev.EvaluationBudget(tmp_path,'protocol',LIMITS)
    resources[field]=value
    with pytest.raises(ev.EvaluationStopped) as error:budget.check()
    assert error.value.status=='resource_limited' and error.value.reason==reason


@pytest.mark.parametrize('field,value',[('max_total_wall_seconds',float('nan')),
    ('max_total_wall_seconds',True),('min_available_memory_bytes',1),
    ('min_free_disk_bytes',1),('max_process_rss_bytes',3*2**30)])
def test_invalid_or_relaxed_limits_are_rejected(tmp_path,field,value):
    with pytest.raises(ValueError):ev.EvaluationBudget(tmp_path,'protocol',{**LIMITS,field:value})


def test_window_boundary_cancellation_preserves_partial_actual_trajectory(resources):
    count=[0]
    def stop():
        count[0]+=1
        if count[0]==4:raise ev.EvaluationStopped('cancelled','test_cancel')
    row,arrays,_=ev.run_episode('A_static',7101,'rule',stop_check=stop)
    assert row['status']=='interrupted' and row['completed_windows']==3
    assert row['failure']['stop_status']=='cancelled'
    assert arrays['body'].shape==(3,7) and arrays['observation'].shape==(3,59)
    assert row['food'] is None


def test_keyboard_interrupt_retains_completed_windows(resources):
    class InterruptingController:
        count=0
        def act(self,obs):
            self.count+=1
            if self.count==4:raise KeyboardInterrupt()
            return np.zeros(4)
    row,arrays,_=ev.run_episode('A_static',7103,InterruptingController())
    assert row['status']=='interrupted' and row['completed_windows']==3
    assert row['failure']['stop_status']=='cancelled' and arrays['body'].shape==(3,7)


@pytest.fixture
def evaluated_fixture(tmp_path,monkeypatch,resources):
    from neuroterrarium import brain_cache
    config=json.loads((Path(__file__).parents[1]/'configs/evaluate-v1.json').read_text())
    config['environment_seeds']=list(range(7101,7131))
    monkeypatch.setattr(ev,'load_model_set',lambda path:({}, {'fixture':'synthetic-budget-test'}))
    monkeypatch.setattr(ev,'evaluation_plan',lambda config,policies:[('rule','none',['A_static'])])
    monkeypatch.setattr(brain_cache,'ensure_cache',lambda path:None)
    monkeypatch.setattr(brain_cache,'load_cache',lambda path:SimpleNamespace(root_ids=np.array([1]),
        summary={'profile':'synthetic-budget-test'},brain=SimpleNamespace(graph_digest='synthetic-budget-test')))
    monkeypatch.setattr(ev.Registry,'load',lambda:SimpleNamespace(resolve=lambda roots:{},resolve_sides=lambda roots:{}))
    profiles=tmp_path/'fixture-profiles';profiles.mkdir()
    origin=ev.default_path()
    for name in ('interface-v783.json','brain-interface.json','brain-storage-v1.json','data-v783.json'):
        (profiles/name).write_bytes(origin.with_name(name).read_bytes())
    storage=json.loads((profiles/'brain-storage-v1.json').read_text())
    storage.update(graph_digest='synthetic-budget-test',summary_sha256=ev.config_digest({'profile':'synthetic-budget-test'}))
    (profiles/'brain-storage-v1.json').write_text(json.dumps(storage))
    monkeypatch.setattr(ev,'default_path',lambda:profiles/'interface-v783.json')
    return config,tmp_path


def test_cancel_before_graph_loading_is_preserved_as_nonpass(evaluated_fixture,monkeypatch):
    from neuroterrarium import brain_cache
    config,path=evaluated_fixture
    def forbidden(*args):raise AssertionError('Cancelled evaluation must not load the graph')
    monkeypatch.setattr(brain_cache,'load_cache',forbidden)
    cancel=path/'cancel';cancel.touch()
    result=ev.evaluate(config,'unused','unused',path,cancel_file=cancel)
    assert result['status']=='cancelled' and result['records']==0 and result['missing_records']==30
    assert json.loads((path/'budget.json').read_text())['status']=='cancelled'
    assert json.loads((path/'results.json').read_text())['status']=='cancelled'


def test_resume_preserves_failed_raw_result_without_reexecuting_it(evaluated_fixture,monkeypatch):
    config,path=evaluated_fixture
    original=ev.run_episode
    class FailingController:
        def act(self,obs):raise RuntimeError('synthetic backend failure')
    def once(case,seed,controller,**kwargs):
        return original(case,seed,FailingController() if seed==7101 else controller,**kwargs)
    monkeypatch.setattr(ev,'run_episode',once)
    first=ev.evaluate(config,'unused','unused',path)
    assert first['status']=='incomplete' and first['records']==30
    raw=path/'raw/rule--none--A_static--7101.json'
    saved=raw.read_bytes()
    assert json.loads(saved)['status']=='failed'
    def forbidden(*args,**kwargs):raise AssertionError('Existing frozen result was rerun')
    monkeypatch.setattr(ev,'run_episode',forbidden)
    second=ev.evaluate(config,'unused','unused',path)
    assert second['status']=='incomplete' and second['records']==30
    assert raw.read_bytes()==saved
    assert second['cumulative_wall_seconds']>=first['cumulative_wall_seconds']


def test_mid_episode_cancel_saves_interrupted_raw_and_budget(evaluated_fixture,monkeypatch):
    config,path=evaluated_fixture
    original=ev.run_episode
    def cancelled(case,seed,controller,**kwargs):
        count=[0]
        def stop():
            count[0]+=1
            if count[0]==4:raise ev.EvaluationStopped('cancelled','test_cancel')
        kwargs['stop_check']=stop
        return original(case,seed,controller,**kwargs)
    monkeypatch.setattr(ev,'run_episode',cancelled)
    result=ev.evaluate(config,'unused','unused',path)
    assert result['status']=='cancelled' and result['records']==1 and result['missing_records']==29
    row=json.loads((path/'raw/rule--none--A_static--7101.json').read_text())
    assert row['status']=='interrupted' and row['completed_windows']==3
    with np.load(path/'raw'/row['trajectory'],allow_pickle=False) as arrays:assert arrays['body'].shape==(3,7)
    assert json.loads((path/'budget.json').read_text())['reserved_seconds']==0


def test_resume_cannot_reset_accounting_by_removing_budget(evaluated_fixture):
    config,path=evaluated_fixture
    raw=path/'raw';raw.mkdir()
    (raw/'existing.json').write_text('{}')
    with pytest.raises(ValueError,match='no cumulative budget'):ev.evaluate(config,'unused','unused',path)


def test_crash_before_summary_does_not_mark_completed_and_can_resume_statistics(evaluated_fixture,monkeypatch):
    config,path=evaluated_fixture
    summarize=ev.summarize
    def crash(output):
        assert json.loads((output/'results.json').read_text())['status']=='summarizing'
        raise SystemExit('simulated abrupt interruption before statistics')
    monkeypatch.setattr(ev,'summarize',crash)
    with pytest.raises(SystemExit,match='simulated abrupt'):ev.evaluate(config,'unused','unused',path)
    assert json.loads((path/'results.json').read_text())['status']=='summarizing'
    budget=json.loads((path/'budget.json').read_text())
    assert budget['status']=='summarizing' and budget['reserved_seconds']==600
    saved={item.name:item.read_bytes() for item in (path/'raw').iterdir()}
    monkeypatch.setattr(ev,'summarize',summarize)
    def forbidden(*args,**kwargs):raise AssertionError('Completed raw cases must not be run again')
    monkeypatch.setattr(ev,'run_episode',forbidden)
    result=ev.evaluate(config,'unused','unused',path)
    assert result['status']=='completed' and result['records']==30
    assert json.loads((path/'budget.json').read_text())['recovered_reservation_seconds']==600
    assert {item.name:item.read_bytes() for item in (path/'raw').iterdir()}==saved


def evidence_bytes_and_times(path):
    return {str(item.relative_to(path)):(item.stat().st_mtime_ns,item.read_bytes() if item.is_file() else None)
            for item in [path,*sorted(path.rglob('*'))]}


@pytest.fixture
def completed_fixture(evaluated_fixture,monkeypatch):
    from neuroterrarium import brain_cache
    config,path=evaluated_fixture
    result=ev.evaluate(config,'unused','unused',path)
    assert result['status']=='completed' and result['records']==30
    def forbidden(*args,**kwargs):raise AssertionError('Read-only verification must not mutate or run the experiment')
    monkeypatch.setattr(brain_cache,'ensure_cache',forbidden)
    monkeypatch.setattr(brain_cache,'load_cache',forbidden)
    monkeypatch.setattr(ev,'run_episode',forbidden)
    monkeypatch.setattr(ev,'atomic_json',forbidden)
    monkeypatch.setattr(ev.EvaluationBudget,'check',forbidden)
    monkeypatch.setattr(ev.EvaluationBudget,'checkpoint',forbidden)
    return config,path


def test_completed_resume_verifies_raw_metrics_and_summary_without_writes_or_graph(completed_fixture):
    config,path=completed_fixture
    before=evidence_bytes_and_times(path)
    historical_budget=json.loads((path/'budget.json').read_text())
    result=ev.evaluate(config,'unused','unused',path)
    assert result['status']=='verified' and result['operation']=='read_only_verification'
    assert result['historical_status']=='completed' and result['missing_records']==0
    assert result['cumulative_wall_seconds']==historical_budget['elapsed_seconds']
    assert result['verification_wall_seconds']>=0
    assert evidence_bytes_and_times(path)==before


def test_completed_resume_does_not_charge_original_budget(completed_fixture,monkeypatch):
    config,path=completed_fixture
    # Simulate an invocation beginning with a fully spent prior-run allowance.
    # The verifier has a separate invocation duration and cannot spend that run again.
    original=ev.EvaluationBudget.__init__
    def spent(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self.previous=self.limits['max_total_wall_seconds']
    monkeypatch.setattr(ev.EvaluationBudget,'__init__',spent)
    before=evidence_bytes_and_times(path)
    assert ev.evaluate(config,'unused','unused',path)['status']=='verified'
    assert evidence_bytes_and_times(path)==before


def test_cancelled_completed_verification_preserves_historical_success(completed_fixture):
    config,path=completed_fixture
    cancel=path/'cancel';cancel.touch()
    before=evidence_bytes_and_times(path)
    result=ev.evaluate(config,'unused','unused',path,cancel_file=cancel)
    assert result['status']=='cancelled' and result['historical_status']=='completed'
    assert result['stop']['stage']=='completed_verification'
    assert evidence_bytes_and_times(path)==before


def test_resource_limited_completed_verification_preserves_historical_success(completed_fixture,resources):
    config,path=completed_fixture
    resources['available']=3*2**30-1
    before=evidence_bytes_and_times(path)
    result=ev.evaluate(config,'unused','unused',path)
    assert result['status']=='resource_limited' and result['historical_status']=='completed'
    assert evidence_bytes_and_times(path)==before


def test_completed_verification_time_limit_cannot_overwrite_success(completed_fixture,monkeypatch):
    config,path=completed_fixture
    clock=iter([0.,10800.])
    monkeypatch.setattr(ev.time,'perf_counter',lambda:next(clock,10800.))
    before=evidence_bytes_and_times(path)
    result=ev.evaluate(config,'unused','unused',path)
    assert result['status']=='budget_exhausted' and result['historical_status']=='completed'
    assert result['stop']['reason']=='verification_wall_budget'
    assert evidence_bytes_and_times(path)==before


def test_completed_verification_can_cancel_during_raw_metric_validation(completed_fixture,monkeypatch):
    config,path=completed_fixture
    # Keep the external cancellation file outside the read-only evidence tree.
    cancel=path.parent/(path.name+'-cancel')
    original=ev.recompute_metrics;calls=[]
    def cancelling(*args,**kwargs):
        calls.append(1)
        if len(calls)==3:cancel.touch()
        return original(*args,**kwargs)
    monkeypatch.setattr(ev,'recompute_metrics',cancelling)
    before=evidence_bytes_and_times(path)
    result=ev.evaluate(config,'unused','unused',path,cancel_file=cancel)
    assert result['status']=='cancelled' and len(calls)==3
    assert evidence_bytes_and_times(path)==before


def test_completed_verification_keyboard_interrupt_is_separate_from_historical_success(completed_fixture,monkeypatch):
    config,path=completed_fixture
    def interrupted(*args,**kwargs):raise KeyboardInterrupt()
    monkeypatch.setattr(ev,'summarize',interrupted)
    before=evidence_bytes_and_times(path)
    result=ev.evaluate(config,'unused','unused',path)
    assert result['status']=='cancelled' and result['stop']['kind']=='KeyboardInterrupt'
    assert evidence_bytes_and_times(path)==before


@pytest.mark.parametrize('target,field,value,match',[
    ('results.json','records',[],'index disagrees'),
    ('results.json','expected_records',31,'index integrity'),
    ('budget.json','reserved_seconds',120,'budget integrity'),
    ('graph-summary.json','neural_graph_sha256','changed','graph identity'),
    ('summary.json','single_controller',[],'summary disagrees'),
])
def test_completed_verification_rejects_tampered_evidence_without_repair(completed_fixture,target,field,value,match):
    config,path=completed_fixture
    file=path/target;data=json.loads(file.read_text());data[field]=value;file.write_text(json.dumps(data))
    before=evidence_bytes_and_times(path)
    with pytest.raises(ValueError,match=match):ev.evaluate(config,'unused','unused',path)
    assert evidence_bytes_and_times(path)==before


@pytest.mark.parametrize('target',['protocol-lock.json','budget.json','summary.json','graph-summary.json'])
def test_completed_verification_rejects_missing_evidence_without_recreating(completed_fixture,target):
    config,path=completed_fixture
    (path/target).unlink()
    before=evidence_bytes_and_times(path)
    with pytest.raises(ValueError,match='missing required evidence'):ev.evaluate(config,'unused','unused',path)
    assert evidence_bytes_and_times(path)==before


def test_completed_verification_recomputes_metrics_even_after_matching_metadata_tamper(completed_fixture):
    config,path=completed_fixture
    file=path/'raw/rule--none--A_static--7101.json'
    row=json.loads(file.read_text());row['food']+=.25;file.write_text(json.dumps(row))
    index=path/'results.json';data=json.loads(index.read_text());data['records'][0]=row;index.write_text(json.dumps(data))
    before=evidence_bytes_and_times(path)
    with pytest.raises(ValueError,match='Outcome does not match raw trajectory: food'):
        ev.evaluate(config,'unused','unused',path)
    assert evidence_bytes_and_times(path)==before
