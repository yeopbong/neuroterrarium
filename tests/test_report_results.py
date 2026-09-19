"""Report formatting and refusal checks use no formal results or full graph."""
import csv
import copy
import shutil
import importlib.util
import io
import json
from pathlib import Path

import pytest
import numpy as np

SPEC=importlib.util.spec_from_file_location('report_results',Path(__file__).parents[1]/'scripts/report_results.py')
report=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(report)


def test_unfinished_experiments_cannot_create_a_placeholder_results_page(tmp_path):
    core=tmp_path/'core';core.mkdir();output=tmp_path/'out'
    (core/'results.json').write_text(json.dumps({'schema':'neuroterrarium.evaluation-results.v1','status':'running','records':[]}))
    with pytest.raises(ValueError,match='Only completed'):
        report.generate(core,tmp_path/'supplement',tmp_path/'models',tmp_path,tmp_path/'evidence.json',output,'v0.1.0')
    assert not output.exists()


def test_undefined_latency_is_not_converted_to_zero():
    assert report.interval({'mean':None,'ci95':None,'defined_units':0})=='undefined (n=0)'
    assert report.interval({'mean':.2,'ci95':[.1,.3],'defined_units':7})=='0.2 [0.1, 0.3] (n=7)'
    with pytest.raises(ValueError):report.interval({'mean':.2,'ci95':[.3,.1],'defined_units':7})


def test_complete_csv_preserves_null_status_and_every_signed_difference():
    rows=[{'controller':'a','mean_difference':-.25,'reaction_seconds':None,'reaction_status':'no_response'},
          {'controller':'b','mean_difference':.5,'reaction_seconds':.2,'reaction_status':'responded'}]
    values=list(csv.DictReader(io.StringIO(report.csv_bytes(rows).decode())))
    assert len(values)==2 and values[0]['mean_difference']=='-0.25' and values[1]['mean_difference']=='0.5'
    assert values[0]['reaction_seconds']=='' and values[0]['reaction_status']=='no_response'


def test_confidence_columns_are_numeric_with_explicit_defined_units():
    value=report.flatten({'reaction_seconds':{'mean':None,'ci95':None,'defined_units':0},'paired_ci95':[-.2,.4]})
    assert value=={'reaction_seconds_mean':None,'reaction_seconds_ci95_low':None,'reaction_seconds_ci95_high':None,
                   'reaction_seconds_defined_units':0,'paired_ci95_low':-.2,'paired_ci95_high':.4}


@pytest.mark.parametrize('path',['../record.json','/record.json','a//record.json','C:/record.json'])
def test_supporting_evidence_paths_cannot_escape_artifact_scope(tmp_path,path):
    with pytest.raises(ValueError):report.reference(tmp_path,{'path':path,'sha256':'0'*64})


def test_supporting_evidence_hash_and_symlink_are_checked(tmp_path):
    path=tmp_path/'record.json';path.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):report.reference(tmp_path,{'path':'record.json','sha256':'0'*64})
    alias=tmp_path/'alias';alias.symlink_to(path)
    with pytest.raises(ValueError,match='Symlink'):report.reference(tmp_path,{'path':'alias','sha256':report.digest(path)})


def test_supporting_index_needs_all_finished_evidence_families(tmp_path):
    with pytest.raises(ValueError,match='Complete supporting'):
        report.validate_evidence(tmp_path,{'schema':'neuroterrarium.report-evidence.v1','evidence':{}},{},'0'*64)


def test_example_selection_is_fixed_before_any_result_values_are_loaded():
    assert report.PRESET_EXAMPLES==( ('A_expansion','motor_energy_demand'),('C_relocation','food') )


def test_fixed_report_tables_preserve_real_counts_and_conditional_undefined_values():
    names=['connectome',*[f'{architecture}-{seed}' for architecture in ('feedforward','recurrent','hybrid') for seed in (101,202,303)],'random','rule','untrained']
    cases=['A_expansion','A_static','A_translation','A_contraction','B_food','B_danger','C_occlusion','C_relocation','C_unseen']
    rows=[];groups=[]
    for name in names:
        for case in cases:
            rows.extend({'controller':name,'scenario':case,'ablation':'none','reaction_status':'no_response','recovery_status':'not_recovered','sampled_peak_process_rss_bytes':100} for _ in range(2))
            groups.append({'controller':name,'scenario':case,'ablation':'none',**{
                key:{'mean':None,'ci95':None,'defined_units':0} if key in {'reaction_seconds','recovery_seconds'} else {'mean':.2,'ci95':[.1,.3],'defined_units':2}
                for key in report.METRICS}})
    core={'result':{'records':rows},'lock':{'config':{'environment_seeds':[11,12]}},
          'summary':{'single_controller':groups,'hierarchical_paired_differences':[]}}
    evidence={'stability':{'measured_wall_seconds':3601},'neural':{'graph':{'neurons':17,'directed_pairs':21,'synaptic_contacts':40}}}
    text=report.render(core,{'result':{'records':[]}},evidence,{},'v0.1.0')
    assert '0/2' in text and 'undefined (n=0)' in text
    assert 'undefined (n=0) denotes no defined observations' in text
    assert 'wall-clock inference ms' in text and 'simulation seconds' in text
    assert all(line.startswith(('#','|','n/N')) for line in text.splitlines() if line)
    a_section=text.split('## A.')[1].split('## B.')[0]
    assert a_section.index('| connectome |')<a_section.index('| feedforward-101 |')<a_section.index('| untrained |')
    assert text.count('| hybrid-303 |')==4
    recovery={'parent_status':'resource_limited','reason':'The available-memory guard interrupted a unit.',
              'reused_completed_count':236,'rerun_units':[{'identifier':'connectome--none--C_relocation--90027',
                  'original_status':'interrupted','original_completed_windows':246}],
              'newly_executed_units':5043,'prefix_comparisons':[{'identifier':'connectome--none--C_relocation--90027',
                  'completed_prefix_windows':246}], 'evidence':{'release_asset':'scientific-evidence-v1.tar.gz'}}
    revision={'parent_status':'incomplete','parent_reason':'wall_budget','parent_exit_code':1,'parent_wall_seconds':900.7,
              'parent_status_counts':{'completed':620,'interrupted':1,'not_started':9},'newly_executed_units':630,
              'incomplete_parent_units':[{'identifier':'shared--shared_population--clean--290021',
                  'status':'interrupted','completed_windows':43},{'identifier':'shared--shared_population--clean--290022',
                  'status':'not_started','completed_windows':0}], 'evidence':{'release_asset':'scientific-evidence-v1.tar.gz'}}
    detailed=report.render(core,{'result':{'records':[]}},evidence,{},'v0.1.0',recovery,revision)
    assert detailed==text
    assert '](' not in detailed and '## Records' not in detailed
    assert 'shared machine' not in detailed and 'dedicated-machine' not in detailed
    assert '5043 previously unrecorded' not in detailed



def make_recovered_records(tmp_path,seeds=(7301,7302,7303)):
    """A preserved complete unit, interrupted unit, and new unit; no test data."""
    parent=tmp_path/'evidence/original';core_path=tmp_path/'evidence/recovered'
    (parent/'raw').mkdir(parents=True);(core_path/'raw').mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
    def ref(path):return {'path':str(path.relative_to(tmp_path)),'sha256':report.digest(path),'bytes':path.stat().st_size}
    lock={'source':{'evaluation.py':'a'*64},'config':{'environment_seeds':list(seeds)}}
    protocol=report.canonical(lock);old=[];new=[]
    for seed in seeds:
        name=f'connectome--none--C_relocation--{seed}'
        arrays={key:np.arange(3*2,dtype=np.float64).reshape(3,2) for key in report.RECOVERY_ARRAYS if key!='initial_body'}
        arrays['initial_body']=np.array([0.,1.]);arrays['inference_seconds']=np.arange(3.)
        arrays['sampled_process_rss_bytes']=np.arange(4,dtype=np.int64)
        np.savez(core_path/'raw'/(name+'.npz'),**arrays)
        row={'controller':'connectome','ablation':'none','scenario':'C_relocation','environment_seed':seed,
             'status':'completed','completed_windows':3,'protocol_sha256':protocol,'trajectory':name+'.npz',
             'trajectory_sha256':report.digest(core_path/'raw'/(name+'.npz')),'failure':None}
        new.append(row);write(core_path/'raw'/(name+'.json'),row)
        if seed not in (7301,7302):continue
        if seed==7301:
            (parent/'raw'/(name+'.npz')).write_bytes((core_path/'raw'/(name+'.npz')).read_bytes())
            prior=dict(row)
        else:
            prefix={key:value if key=='initial_body' else value[:2] for key,value in arrays.items()}
            np.savez(parent/'raw'/(name+'.npz'),**prefix)
            prior={**row,'status':'interrupted','completed_windows':2,
                   'trajectory_sha256':report.digest(parent/'raw'/(name+'.npz')),
                   'failure':{'reason':'system_available_memory','stop_status':'resource_limited'}}
        old.append(prior);write(parent/'raw'/(name+'.json'),prior)
    write(parent/'results.json',{'schema':'neuroterrarium.evaluation-results.v1','status':'resource_limited',
          'protocol_sha256':protocol,'expected_records':len(seeds),'records':old,'stop':{'reason':'system_available_memory'}})
    write(parent/'protocol-lock.json',lock);write(parent/'graph-summary.json',{})
    write(parent/'budget.json',{'protocol_sha256':protocol,'limits':{'min_available_memory_bytes':10},'elapsed_seconds':2.})
    write(parent/'progress.json',{})
    names=[row['trajectory'].removesuffix('.npz') for row in old]
    lineage={'schema':'neuroterrarium.evaluation-recovery.v1','parent_experiment':'original',
             'parent_files_sha256':{str(p.relative_to(parent)):report.digest(p) for p in parent.rglob('*') if p.is_file()},
             'protocol_sha256':protocol,'source':lock['source'],'reason':'system_available_memory guard interrupted an episode.',
             'limits':{'min_available_memory_bytes':10},'inherited_wall_seconds':2.,
             'reused_completed_identifiers':[names[0]],'incomplete_parent_records':{names[1]:old[1]},
             'rerun_identifiers':[names[1]],'not_previously_recorded_count':len(seeds)-len(old)}
    write(core_path/'recovery-manifest.json',lineage)
    commands=[]
    for phase in ('preparation','execution'):
        path=tmp_path/'evidence'/(phase+'.json')
        write(path,{'phase':phase,'status':'completed','command':['python','checked-command'],'exit_code':0})
        commands.append(ref(path))
    proof=tmp_path/'evidence/prefix.json'
    write(proof,{'identifier':names[1],'status':'passed','old_trajectory_sha256':old[1]['trajectory_sha256'],
                 'recovery_trajectory_sha256':new[1]['trajectory_sha256'],
                 'checks':{key:{'status':'passed'} for key in report.RECOVERY_ARRAYS}})
    spec={'parent_directory':'evidence/original','manifest':ref(core_path/'recovery-manifest.json'),
          'commands':commands,'prefix_proof':ref(proof),'release_asset':'scientific-evidence-v1.tar.gz'}
    result={'schema':'neuroterrarium.evaluation-results.v1','status':'completed','protocol_sha256':protocol,'expected_records':len(seeds),'records':new}
    write(core_path/'results.json',result);write(core_path/'protocol-lock.json',lock);write(core_path/'graph-summary.json',{})
    write(core_path/'progress.json',{});write(core_path/'budget.json',{'protocol_sha256':protocol,'limits':{'min_available_memory_bytes':10},'elapsed_seconds':3.})
    return {'root':tmp_path,'path':core_path,'core':{'lock':lock,'result':result},
            'spec':spec,'parent':parent,'lineage':lineage,'write':write,'ref':ref,'old':old,'new':new}


@pytest.fixture
def recovered_records(tmp_path):
    return make_recovered_records(tmp_path)


def recovery_result(fixture):
    return report.validate_recovery(fixture['path'],fixture['core'],fixture['root'],fixture['spec'])


def update_lineage(fixture):
    path=fixture['path']/'recovery-manifest.json';fixture['write'](path,fixture['lineage'])
    fixture['spec']['manifest']=fixture['ref'](path)


def test_recovery_reports_every_reused_rerun_and_new_unit(recovered_records):
    out=recovery_result(recovered_records)
    assert out['parent_status']=='resource_limited' and out['reused_completed_count']==1
    assert len(out['rerun_units'])==1 and out['newly_executed_units']==1 and out['omitted_parent_units']==[]
    assert out['prefix_comparisons'][0]['completed_prefix_windows']==2
    assert out['prefix_comparisons'][0]['excluded_cost_arrays']==['inference_seconds','sampled_process_rss_bytes']


@pytest.mark.parametrize('change', ['omit_reuse','omit_rerun','missing_parent_hash','invent_new_count','wrong_protocol'])
def test_recovery_rejects_selective_or_forged_lineage(recovered_records,change):
    f=recovered_records;m=f['lineage']
    if change=='omit_reuse':m['reused_completed_identifiers']=[]
    elif change=='omit_rerun':m['rerun_identifiers']=[]
    elif change=='missing_parent_hash':m['parent_files_sha256'].pop(next(k for k in m['parent_files_sha256'] if k.startswith('raw/')))
    elif change=='invent_new_count':m['not_previously_recorded_count']=0
    else:m['protocol_sha256']='0'*64
    update_lineage(f)
    with pytest.raises(ValueError):recovery_result(f)


def test_recovery_rejects_modified_completed_units(recovered_records):
    recovered_records['new'][0]['unexpected_change']=True
    with pytest.raises(ValueError,match='reused completed'):recovery_result(recovered_records)


def test_recovery_prefix_is_recomputed_instead_of_trusting_pass_label(recovered_records):
    f=recovered_records;row=f['new'][1];path=f['path']/'raw'/row['trajectory']
    with np.load(path) as saved:arrays={key:saved[key] for key in saved.files}
    arrays['applied_action'][0,0]+=1
    np.savez(path,**arrays);row['trajectory_sha256']=report.digest(path)
    f['write'](f['path']/'raw'/(path.stem+'.json'),row)
    with pytest.raises(ValueError,match='prefix differs'):recovery_result(f)


@pytest.mark.parametrize('name', ['../outside','/absolute','evidence//original'])
def test_recovery_parent_directory_cannot_escape(recovered_records,name):
    recovered_records['spec']['parent_directory']=name
    with pytest.raises(ValueError,match='normalized relative'):recovery_result(recovered_records)


def test_recovery_rejects_running_command_and_missing_execution_phase(recovered_records):
    f=recovered_records;path=f['root']/f['spec']['commands'][1]['path'];doc=report.read(path)
    doc['status']='running';f['write'](path,doc);f['spec']['commands'][1]=f['ref'](path)
    with pytest.raises(ValueError,match='successful recorded'):recovery_result(f)
    f['spec']['commands']=f['spec']['commands'][:1]
    with pytest.raises(ValueError,match='both be recorded'):recovery_result(f)


def test_recovery_requires_index_and_no_recovery_fixture_remains_supported(recovered_records,tmp_path):
    with pytest.raises(ValueError,match='explicit recovery'):
        report.validate_recovery(recovered_records['path'],recovered_records['core'],recovered_records['root'],None)
    assert report.validate_recovery(tmp_path,{},tmp_path,None) is None


@pytest.fixture
def recovery_chain(tmp_path):
    f=make_recovered_records(tmp_path,seeds=(7301,7302,7303,7304))
    middle=f['path'];final=tmp_path/'evidence/final';shutil.copytree(middle,final)
    final_core=copy.deepcopy(f['core']);final_rows=final_core['result']['records']
    # The first recovery reran the original interrupted unit, recorded one new
    # interrupted unit, and stopped before the last planned unit.
    partial=copy.deepcopy(final_rows[2]);name=partial['trajectory'].removesuffix('.npz')
    with np.load(middle/'raw'/partial['trajectory']) as saved:
        arrays={key:saved[key] if key=='initial_body' else saved[key][:2] for key in saved.files}
    np.savez(middle/'raw'/partial['trajectory'],**arrays)
    partial.update(status='interrupted',completed_windows=2,trajectory_sha256=report.digest(middle/'raw'/partial['trajectory']),
                   failure={'reason':'system_available_memory','stop_status':'resource_limited'})
    f['write'](middle/'raw'/(name+'.json'),partial)
    for suffix in ('.npz','.json'):(middle/'raw'/(final_rows[3]['trajectory'].removesuffix('.npz')+suffix)).unlink()
    intermediate={**f['core']['result'],'status':'resource_limited','records':[copy.deepcopy(final_rows[0]),copy.deepcopy(final_rows[1]),partial],
                  'stop':{'reason':'system_available_memory'}}
    f['write'](middle/'results.json',intermediate)
    ancestor=copy.deepcopy(f['spec']);ancestor['child_directory']='evidence/recovered'
    command_path=tmp_path/ancestor['commands'][1]['path'];command=report.read(command_path);command['exit_code']=1
    f['write'](command_path,command);ancestor['commands'][1]=f['ref'](command_path)
    lineage={'schema':'neuroterrarium.evaluation-recovery.v1','parent_experiment':'recovered',
             'parent_files_sha256':{str(p.relative_to(middle)):report.digest(p) for p in middle.rglob('*') if p.is_file()},
             'parent_recovery_manifest_sha256':report.digest(middle/'recovery-manifest.json'),'ancestor_experiments':['original'],
             'protocol_sha256':report.canonical(final_core['lock']),'source':final_core['lock']['source'],
             'reason':'system_available_memory guard interrupted the first recovery.',
             'limits':{'min_available_memory_bytes':10},'inherited_wall_seconds':3.,
             'reused_completed_identifiers':[row['trajectory'].removesuffix('.npz') for row in intermediate['records'][:2]],
             'incomplete_parent_records':{name:partial},'rerun_identifiers':[name],'not_previously_recorded_count':1}
    f['write'](final/'recovery-manifest.json',lineage)
    f['write'](final/'budget.json',{'protocol_sha256':report.canonical(final_core['lock']),'limits':{'min_available_memory_bytes':10},'elapsed_seconds':4.})
    commands=[]
    for phase in ('preparation','execution'):
        p=tmp_path/'evidence'/('final-'+phase+'.json');f['write'](p,{'phase':phase,'status':'completed','command':['python','checked-command'],'exit_code':0})
        commands.append(f['ref'](p))
    proof=tmp_path/'evidence/final-prefix.json';f['write'](proof,{'identifier':name,'status':'passed',
            'old_trajectory_sha256':partial['trajectory_sha256'],'recovery_trajectory_sha256':final_rows[2]['trajectory_sha256'],
            'checks':{key:{'status':'passed'} for key in report.RECOVERY_ARRAYS}})
    spec={'parent_directory':'evidence/recovered','manifest':f['ref'](final/'recovery-manifest.json'),
          'commands':commands,'prefix_proof':f['ref'](proof),'release_asset':'scientific-evidence-v1.tar.gz','ancestors':[ancestor]}
    return {**f,'path':final,'core':final_core,'spec':spec,'lineage':lineage,'middle':middle}


def test_recovery_chain_keeps_failed_stage_and_actual_unit_counts(recovery_chain):
    value=recovery_result(recovery_chain);first,last=value['history']
    assert first['child_status']=='resource_limited' and last['child_status']=='completed'
    assert (first['planned_new_units'],first['recorded_new_units'],first['completed_new_units'],first['missing_new_units'])==(2,1,0,1)
    assert len(first['incomplete_new_units'])==1 and len(first['rerun_units'])==1
    assert (last['planned_new_units'],last['recorded_new_units'],last['completed_new_units'],last['missing_new_units'])==(1,1,1,0)
    assert value['independent_final_units']==4 and value['recorded_execution_attempts']==6
    assert first['prefix_comparisons'][0]['completed_prefix_windows']==last['prefix_comparisons'][0]['completed_prefix_windows']==2


@pytest.mark.parametrize('change',['omit_ancestor','duplicate_ancestor','wrong_parent_manifest','wrong_declared_ancestry','changed_inherited_budget','failed_stage_claims_exit_zero','omitted_intermediate_raw'])
def test_recovery_chain_rejects_missing_or_falsely_completed_history(recovery_chain,change):
    f=recovery_chain
    if change=='omit_ancestor':f['spec'].pop('ancestors')
    elif change=='duplicate_ancestor':f['spec']['ancestors'].append(copy.deepcopy(f['spec']['ancestors'][0]))
    elif change=='wrong_parent_manifest':f['lineage']['parent_recovery_manifest_sha256']='0'*64;update_lineage(f)
    elif change=='wrong_declared_ancestry':f['lineage']['ancestor_experiments']=[];update_lineage(f)
    elif change=='changed_inherited_budget':
        p=f['path']/'budget.json';d=report.read(p);d['elapsed_seconds']=2.;f['write'](p,d)
    elif change=='failed_stage_claims_exit_zero':
        a=f['spec']['ancestors'][0];p=f['root']/a['commands'][1]['path'];d=report.read(p);d['exit_code']=0;f['write'](p,d);a['commands'][1]=f['ref'](p)
    else:
        p=f['middle']/'raw/omitted.json';f['write'](p,{'status':'interrupted'})
        f['lineage']['parent_files_sha256']['raw/omitted.json']=report.digest(p);update_lineage(f)
    with pytest.raises(ValueError):recovery_result(f)


def test_release_gate_requires_the_same_complete_recovery_chain(recovery_chain):
    f=recovery_chain;gate=report.release_validator();evidence=gate.Evidence(f['root'])
    ref=f['ref'](f['path']/'results.json');context={'source':{'scripts/report_results.py':report.digest(report.__file__)}}
    verified=gate.verify_evaluation_recovery(evidence,ref,f['core']['result'],f['core']['lock'],context,{'recovery':f['spec']})
    assert verified['independent_final_units']==4 and len(verified['history'])==2
    with pytest.raises(ValueError,match='explicit recovery'):
        gate.verify_evaluation_recovery(evidence,ref,f['core']['result'],f['core']['lock'],context,{})
    context['source']['scripts/report_results.py']='0'*64
    with pytest.raises(ValueError,match='verifier source'):
        gate.verify_evaluation_recovery(evidence,ref,f['core']['result'],f['core']['lock'],context,{'recovery':f['spec']})


def test_report_metadata_drives_the_actual_frozen_allocation_plan():
    from neuroterrarium import evaluation
    root=Path(__file__).parents[1]
    models=report.read(root/'artifacts/models/manifest.json')
    config=report.read(root/'configs/evaluate-v1.json')
    policies=report.policy_plan_metadata(models)
    units=[(name,ablation,case,seed) for name,ablation,cases in evaluation.evaluation_plan(config,policies)
           for case in cases for seed in config['environment_seeds']]
    assert len(units)==len(set(units))==5280
    for architecture in ('feedforward','recurrent','hybrid'):
        assert {name for name,_,_,_ in units if name.startswith(architecture+'-')}=={
            row['id'] for row in models['models'] if row['architecture']==architecture}
