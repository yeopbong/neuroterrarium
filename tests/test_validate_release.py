"""Negative release-gate probes; synthetic evidence cannot produce a release pass."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import zlib

import pytest

SPEC=importlib.util.spec_from_file_location('validate_release',Path(__file__).parents[1]/'scripts/validate_release.py')
gate=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def write(root,name,data):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data if isinstance(data,bytes) else json.dumps(data).encode())
    return {'path':name,'sha256':gate.sha256(path)}


@pytest.fixture
def candidate(tmp_path):
    names=sorted(set().union(*gate.SOURCE_SCOPE.values()))
    source={name:write(tmp_path,'candidate/'+name,('test source '+name).encode()) for name in names}
    training={name:source[name] for name in ('training.py','controllers.py','world.py')}
    trained={name:ref['sha256'] for name,ref in training.items()}
    config={'total_transitions':1000000,'num_envs':4}
    models=[]
    for architecture in ('feedforward','recurrent','hybrid'):
        for seed in (101,202,303):
            name=f'{architecture}-{seed}'
            state={'training_status':'completed','transitions':1000000,'config':config,'config_sha256':gate.canonical(config),
                   'source_sha256':trained,'policy':{'architecture':architecture,'seed':seed,'observation_size':59}}
            files={}
            for component in ('policy.safetensors','optimizer.safetensors','runtime.safetensors','state.json'):
                # Parser fixtures only; no product runtime loads these payloads.
                ref=write(tmp_path,f'models/{name}/{component}',state if component=='state.json' else (name+component).encode())
                files[component]={'sha256':ref['sha256'],'bytes':(tmp_path/ref['path']).stat().st_size}
            checkpoint=write(tmp_path,f'models/{name}/manifest.json',{'schema':'neuroterrarium.ppo-checkpoint.v1','files':files})
            models.append({'id':name,'checkpoint':name,'architecture':architecture,'seed':seed,
                           'policy_sha256':files['policy.safetensors']['sha256'],
                           'checkpoint_manifest_sha256':checkpoint['sha256'],'transitions':1000000,'training_status':'completed'})
    modelset=write(tmp_path,'models/manifest.json',{'schema':'neuroterrarium.model-set.v1','status':'completed','models':models,
                                                 'transitions_per_model':1000000,'source_sha256':trained,'protocol_sha256':gate.canonical(config)})
    data={'expected':{'neurons':138639,'directed_pairs':15091983,'synaptic_contacts':54492922},'sources':{}}
    configs={name:write(tmp_path,'configs/'+name,data if name=='data-v783.json' else config if name=='train-full.json' else {})
             for name in set().union(*gate.CONFIG_SCOPE.values())}
    return {'source':source,'training_sources':training,'configs':configs,
            'model_manifest':modelset}


@pytest.mark.parametrize('path',['/etc/passwd','../other','a/../file','a//file','./file','C:/file','a\\file',''])
def test_artifacts_reject_absolute_traversal_and_ambiguous_paths(tmp_path,path):
    with pytest.raises(gate.GateError):gate.Evidence(tmp_path).path(path)


def test_artifact_hash_mutation_and_symlink_are_rejected(tmp_path):
    ref=write(tmp_path,'record.json',{'status':'passed'})
    evidence=gate.Evidence(tmp_path);evidence.file(ref)
    (tmp_path/'record.json').write_text('damaged')
    with pytest.raises(gate.GateError,match='checksum'):evidence.file(ref)
    (tmp_path/'alias.json').symlink_to(tmp_path/'record.json')
    with pytest.raises(gate.GateError,match='Symlink'):evidence.path('alias.json')


@pytest.mark.parametrize('content',[
    '<testsuite tests="0"/>',
    '<testsuite tests="1"><testcase><skipped/></testcase></testsuite>',
    '<testsuite tests="1"><testcase><failure/></testcase></testsuite>',
    '<testsuite tests="1"><testcase><error/></testcase></testsuite>',
    '<testsuite tests="2"><testcase/></testsuite>',
    '<testsuite tests="1" skipped="1"><testcase/></testsuite>',
    '<!DOCTYPE testsuite><testsuite tests="1"><testcase/></testsuite>',
])
def test_junit_rejects_empty_failed_skipped_and_forged_counts(tmp_path,content):
    path=tmp_path/'junit.xml';path.write_text(content)
    with pytest.raises(gate.GateError):gate.verified_junit(path)


def test_junit_counts_actual_executed_cases(tmp_path):
    path=tmp_path/'junit.xml';path.write_text('<testsuites tests="2"><testsuite tests="2" failures="0" errors="0" skipped="0"><testcase name="a"/><testcase name="b"/></testsuite></testsuites>')
    assert gate.verified_junit(path)==2


def test_missing_categories_stay_blocked_without_remote_dependencies(tmp_path,candidate):
    result=gate.validate({'schema':'neuroterrarium.release-evidence.v1','phase':'local','candidate':candidate,'items':{}},gate.Evidence(tmp_path))
    assert result['status']=='blocked' and len(result['checks'])==16
    assert all(row['status']=='blocked' for row in result['checks'])
    assert not any('remote' in row['name'] for row in result['checks'])
    with pytest.raises(gate.GateError,match='local evidence only'):
        gate.validate({'schema':'neuroterrarium.release-evidence.v1','phase':'remote'},gate.Evidence(tmp_path))


@pytest.mark.parametrize('mutation',['duplicate_weights','smoke_budget','incomplete','source_changed','component_missing'])
def test_model_gate_rejects_forged_model_set(tmp_path,candidate,mutation):
    path=tmp_path/candidate['model_manifest']['path'];models=json.loads(path.read_text())
    if mutation=='duplicate_weights':models['models'][1]['policy_sha256']=models['models'][0]['policy_sha256']
    elif mutation=='smoke_budget':models['transitions_per_model']=4096
    elif mutation=='incomplete':models['models'][0]['training_status']='interrupted'
    elif mutation=='source_changed':models['source_sha256']['world.py']='a'*64
    else:(tmp_path/'models/feedforward-101/optimizer.safetensors').unlink()
    candidate['model_manifest']=write(tmp_path,'models/manifest.json',models)
    with pytest.raises((gate.GateError,OSError)):gate.candidate_context(gate.Evidence(tmp_path),candidate)


def test_failed_short_stability_cannot_pass_even_if_hour_flag_is_true(tmp_path):
    summary={'schema':'neuroterrarium.stability.v1','status':'completed','mode':'Local full graph',
             'behavior_sha256':'a','models_sha256':'b','controllers':['connectome',*[str(i) for i in range(9)]],
             'requested_wall_seconds':3600,'measured_wall_seconds':15,'actual_wall_seconds':16,'hour_gate_passed':True}
    context={'behavior':'a','models_hash':'b','controller_names':set(summary['controllers'])}
    with pytest.raises(gate.GateError,match='full hour'):
        gate.verify_stability(gate.Evidence(tmp_path),{},summary,context,{})


def test_ui_requires_all_six_actual_interaction_records(tmp_path):
    summary={'schema':'neuroterrarium.ui-release-check.v1','mode':'Local full graph','checks':[]}
    with pytest.raises(gate.GateError,match='Six core'):gate.verify_ui(gate.Evidence(tmp_path),summary)
    ref=write(tmp_path,'screen.png',b'not an actual image')
    trace=write(tmp_path,'interaction.json',{'command':'test'})
    summary['checks']=[{'name':name,'status':'passed','operations':1,'observed_effect':'Measured state changed',
                        'screenshot':ref,'trace':trace} for name in gate.UI_CHECKS]
    with pytest.raises(gate.GateError,match='PNG'):gate.verify_ui(gate.Evidence(tmp_path),summary)


def test_status_only_generic_item_does_not_pass(tmp_path,candidate):
    context=gate.candidate_context(gate.Evidence(tmp_path),candidate)
    item={'status':'completed','exit_code':0,'command':['test-probe'],
          'log':write(tmp_path,'check.log',b'probe output\n'),
          'summary':write(tmp_path,'summary.json',{'status':'passed','steps':[]}),
          'identity':{'source_sha256':context['source'],'config_sha256':context['configs'],'model_manifest_sha256':context['models_hash']}}
    result=gate.validate({'schema':'neuroterrarium.release-evidence.v1','phase':'local','candidate':candidate,'items':{'lint':item}},gate.Evidence(tmp_path))
    assert next(c for c in result['checks'] if c['name']=='lint')['status']=='blocked'
    item['exit_code']=None
    result=gate.validate({'schema':'neuroterrarium.release-evidence.v1','phase':'local','candidate':candidate,'items':{'lint':item}},gate.Evidence(tmp_path))
    assert 'successfully' in next(c for c in result['checks'] if c['name']=='lint')['reason']


def test_missing_formal_evaluation_records_cannot_use_completed_label(tmp_path):
    config={'status':'frozen','environment_seeds':list(range(30)),'scenarios':['A_expansion'],'ablations':[]}
    lock={'config':config,'source':{},'model_manifest_sha256':gate.canonical({})}
    ref=write(tmp_path,'lock.json',lock)
    summary={'status':'completed','protocol_sha256':gate.canonical(lock),'records':[]}
    with pytest.raises(gate.GateError,match='omits or duplicates'):
        gate.verify_evaluation(gate.Evidence(tmp_path),{},summary,{'models':{},'source':{},'model_ids':set(),
                              'config_documents':{'evaluate-v1.json':config}}, {'protocol_lock':ref})


def test_ui_dimensions_are_measured_from_bytes_not_claimed_metadata(tmp_path):
    image=write(tmp_path,'tiny.png',png_fixture(1,1))
    trace=write(tmp_path,'trace.json',{'command':'test'})
    summary={'schema':'neuroterrarium.ui-release-check.v1','mode':'Local full graph',
             'checks':[{'name':name,'status':'passed','operations':1,'observed_effect':'Measured state changed',
                        'screenshot':image,'trace':trace,'claimed_width':1920} for name in gate.UI_CHECKS]}
    with pytest.raises(gate.GateError,match='too small'):gate.verify_ui(gate.Evidence(tmp_path),summary)


def test_tested_source_identity_cannot_be_replaced_with_current_file_presence(tmp_path,candidate):
    context=gate.candidate_context(gate.Evidence(tmp_path),candidate)
    identity={'source_sha256':copy.deepcopy(context['source']),'config_sha256':context['configs'],'model_manifest_sha256':context['models_hash']}
    identity['source_sha256']['world.py']='f'*64
    item={'status':'completed','exit_code':0,'command':['test-probe'],'log':write(tmp_path,'source.log',b'test\n'),
          'summary':write(tmp_path,'source.json',{'status':'passed'}),'identity':identity}
    result=gate.validate({'schema':'neuroterrarium.release-evidence.v1','phase':'local','candidate':candidate,'items':{'runtime':item}},gate.Evidence(tmp_path))
    assert 'source identity differs' in next(c for c in result['checks'] if c['name']=='runtime')['reason']


def test_cache_gate_recomputes_arrays_instead_of_trusting_equal_flags(tmp_path):
    import numpy as np
    summary={'neurons':2,'connections':1,'graph_digest':'graph','input_recording_sha256':'tape',
             'windows':20,'all_neuron_spikes':20,'neural_sha256':'neural','runtime_sha256':'runtime'}
    report={'schema':'neuroterrarium.cache-equivalence.v1','status':'passed','equal':True,
            'before':write(tmp_path,'before/summary.json',summary),'after':write(tmp_path,'after/summary.json',summary),'pairs':[]}
    arrays={'spike_steps':np.array([1]),'spike_neurons':np.array([0]),'spike_counts':np.array([1,0]),
            'v_mV':np.array([[-52.,-52.]]),'g_mV':np.zeros((1,2))}
    for i in range(20):
        pair={}
        for branch in ('before','after'):
            name=f'{branch}/window-{i:03d}.npz';np.savez(tmp_path/name,**arrays)
            pair[branch]={'path':name,'sha256':gate.sha256(tmp_path/name)}
        report['pairs'].append(pair)
    report['pairs'].append({branch:write(tmp_path,f'{branch}/final-state.json',{'state':[1,2]}) for branch in ('before','after')})
    context={'data':{'expected':{'neurons':2,'directed_pairs':1}},'source':{'neural.py':'neural','runtime.py':'runtime'}}
    gate.verify_cache(gate.Evidence(tmp_path),report,context)
    changed={**arrays,'v_mV':np.array([[-51.,-52.]])}
    ref=report['pairs'][0]['after'];np.savez(tmp_path/ref['path'],**changed);ref['sha256']=gate.sha256(tmp_path/ref['path'])
    with pytest.raises(gate.GateError,match='arrays differ'):
        gate.verify_cache(gate.Evidence(tmp_path),report,context)


def scoped_identity(name,context):
    source=context['training_source'] if name=='training' else context['source']
    return {'source_sha256':{key:source[key] for key in gate.SOURCE_SCOPE[name]},
            'config_sha256':{key:context['configs'][key] for key in gate.CONFIG_SCOPE[name]},
            'model_manifest_sha256':None if name in gate.WITHOUT_MODELS else context['models_hash']}


def test_typecheck_scope_does_not_claim_python_or_models_were_executed(tmp_path,candidate):
    context=gate.candidate_context(gate.Evidence(tmp_path),candidate)
    identity=scoped_identity('typecheck',context)
    assert set(identity['source_sha256'])=={'web/src/main.ts'}
    gate.verify_identity('typecheck',identity,{},context,gate.Evidence(tmp_path),{})
    # Unrelated new Python modules are inventoried, without invalidating a
    # recorded TypeScript-only command or asserting that it executed Python.
    context['source']['evaluation.py']='b'*64
    gate.verify_identity('typecheck',identity,{},context,gate.Evidence(tmp_path),{})
    context['source']['web/src/main.ts']='b'*64
    with pytest.raises(gate.GateError,match='source identity differs'):
        gate.verify_identity('typecheck',identity,{},context,gate.Evidence(tmp_path),{})


def test_dependency_subset_cannot_omit_required_runtime_module(tmp_path,candidate):
    context=gate.candidate_context(gate.Evidence(tmp_path),candidate)
    identity=scoped_identity('runtime',context);del identity['source_sha256']['world.py']
    with pytest.raises(gate.GateError,match='scope omits'):
        gate.verify_identity('runtime',identity,{},context,gate.Evidence(tmp_path),{})
    identity=scoped_identity('runtime',context);del identity['config_sha256']['interface-v783.json']
    with pytest.raises(gate.GateError,match='configuration scope omits'):
        gate.verify_identity('runtime',identity,{},context,gate.Evidence(tmp_path),{})


def test_native_neural_identity_cannot_be_replaced_by_an_envelope(tmp_path,candidate):
    context=gate.candidate_context(gate.Evidence(tmp_path),candidate);identity=scoped_identity('neural',context)
    native={('src/neuroterrarium/'+key if '/' not in key else key):value for key,value in identity['source_sha256'].items()}
    native.update({'configs/'+key:value for key,value in identity['config_sha256'].items()})
    summary={'checks':{'component_source_unchanged':True},'component_sha256':native}
    gate.verify_identity('neural',identity,summary,context,gate.Evidence(tmp_path),{})
    native['src/neuroterrarium/neural.py']='d'*64
    with pytest.raises(gate.GateError,match='Native neural'):
        gate.verify_identity('neural',identity,summary,context,gate.Evidence(tmp_path),{})


def test_native_data_identity_and_measured_summary_are_required(tmp_path,candidate):
    evidence=gate.Evidence(tmp_path);context=gate.candidate_context(evidence,candidate)
    identity=scoped_identity('data',context);summary={'neurons':138639}
    execution={'exit_code':0,'status':'passed','summary':summary,
               'source_files_sha256':{'neuroterrarium/src/neuroterrarium/data.py':identity['source_sha256']['data.py'],
                'neuroterrarium/configs/data-v783.json':identity['config_sha256']['data-v783.json']}}
    item={'execution':write(tmp_path,'executed.json',execution)}
    gate.verify_identity('data',identity,summary,context,evidence,item)
    execution['summary']={'neurons':5};item['execution']=write(tmp_path,'executed.json',execution)
    with pytest.raises(gate.GateError,match='Native data execution'):
        gate.verify_identity('data',identity,summary,context,evidence,item)


def test_unrelated_model_manifest_cannot_be_attributed_to_pretraining_neural_run(tmp_path,candidate):
    context=gate.candidate_context(gate.Evidence(tmp_path),candidate);identity=scoped_identity('neural',context)
    identity['model_manifest_sha256']=context['models_hash']
    with pytest.raises(gate.GateError,match='inapplicable'):
        gate.verify_identity('neural',identity,{},context,gate.Evidence(tmp_path),{})


def test_content_review_requires_native_candidate_inventory_and_exact_hashes(tmp_path,candidate):
    evidence=gate.Evidence(tmp_path);context=gate.candidate_context(evidence,candidate)
    identity={'source_sha256':context['source'],'config_sha256':context['configs'],'model_manifest_sha256':context['models_hash']}
    files={('src/neuroterrarium/'+key if '/' not in key else key):{'sha256':value} for key,value in context['source'].items()}
    files.update({'configs/'+key:{'sha256':value} for key,value in context['configs'].items()})
    native={'schema':'neuroterrarium.candidate-content-inventory.v1','files':files,'sha256':gate.canonical(files)}
    summary={'candidate_source_unchanged':True,'candidate_inventory_sha256':native['sha256'],
             'candidate_inventory':write(tmp_path,'content/inventory.json',native)}
    gate.verify_identity('public_review',identity,summary,context,evidence,{})
    native['files']['src/neuroterrarium/world.py']['sha256']='e'*64
    native['sha256']=gate.canonical(native['files']);summary['candidate_inventory_sha256']=native['sha256']
    summary['candidate_inventory']=write(tmp_path,'content/inventory.json',native)
    with pytest.raises(gate.GateError,match='Native content source identity differs'):
        gate.verify_identity('public_review',identity,summary,context,evidence,{})


def test_portable_installed_bytes_must_match_the_candidate_envelope(tmp_path,candidate):
    evidence=gate.Evidence(tmp_path);context=gate.candidate_context(evidence,candidate)
    identity=scoped_identity('portable',context);source=identity['source_sha256']
    summary={'package_source_sha256':{key:source[key] for key in {*gate.BEHAVIOR,'validation.py','service.py','cli.py'}},
             'config_sha256':{key:context['configs'][key] for key in gate.BRAIN_CONFIGS},
             'validation_source_sha256':source['validation.py'],
             'verification_source_sha256':{key:source['scripts/'+key] for key in ('portable_check.py','build_package.py','install_check.py')}}
    gate.verify_identity('portable',identity,summary,context,evidence,{})
    summary['package_source_sha256']['neural.py']='e'*64
    with pytest.raises(gate.GateError,match='Native installed package source differs'):
        gate.verify_identity('portable',identity,summary,context,evidence,{})


def png_fixture(width,height):
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    header=struct.pack('>IIBBBBB',width,height,8,2,0,0,0)
    scan=zlib.compress((b'\0'+b'\0'*(3*width))*height)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',header)+chunk(b'IDAT',scan)+chunk(b'IEND',b'')


def jpeg_header_fixture(width,height):
    def segment(marker,payload):return b'\xff'+bytes([marker])+struct.pack('>H',len(payload)+2)+payload
    frame=struct.pack('>BHHB',8,height,width,3)+bytes([1,0x11,0,2,0x11,0,3,0x11,0])
    scan=bytes([3,1,0,2,0,3,0,0,63,0])
    # Structural parser fixture only, never used as UI or release evidence.
    return b'\xff\xd8'+segment(0xc0,frame)+segment(0xda,scan)+b'\x01\x02\xff\xd9'


@pytest.mark.parametrize('kind',['png','jpeg'])
def test_screenshot_dimensions_come_from_complete_image_structure(tmp_path,kind):
    path=tmp_path/'image.data'
    path.write_bytes(png_fixture(800,600) if kind=='png' else jpeg_header_fixture(800,600))
    assert gate.image_dimensions(path)==(800,600)


@pytest.mark.parametrize('damage',['png_truncated','png_crc','jpeg_truncated','jpeg_length','jpeg_frame','jpeg_scan_missing'])
def test_screenshot_parser_rejects_malformed_and_truncated_images(tmp_path,damage):
    raw=png_fixture(800,600) if damage.startswith('png') else jpeg_header_fixture(800,600)
    if damage.endswith('truncated'):raw=raw[:-4]
    elif damage=='png_crc':raw=raw[:20]+bytes([raw[20]^1])+raw[21:]
    elif damage=='jpeg_length':raw=raw[:4]+b'\xff\xff'+raw[6:]
    elif damage=='jpeg_frame':raw=raw[:9]+b'\0\0'+raw[11:]
    else:raw=raw[:raw.index(b'\xff\xda')]+b'\xff\xd9'
    path=tmp_path/'image';path.write_bytes(raw)
    with pytest.raises(gate.GateError):gate.image_dimensions(path)


def test_ui_category_dispatch_checks_native_source_and_actual_image_bytes(tmp_path):
    image=write(tmp_path,'screen.jpg',jpeg_header_fixture(800,600))
    trace=write(tmp_path,'trace.json',{'fixture':'parser-only interaction trace'})
    source={'web/src/main.ts':'a'*64,'web/src/style.css':'b'*64}
    summary={'schema':'neuroterrarium.ui-release-check.v1','mode':'Local full graph','source_sha256':source.copy(),
             'checks':[{'name':name,'status':'passed','operations':1,'observed_effect':'Parser fixture changed the recorded field',
                        'screenshot':image,'trace':trace} for name in gate.UI_CHECKS],
             'viewports':[{'width':800,'height':600,'device_pixel_ratio':1},{'width':1280,'height':720,'device_pixel_ratio':2}],
             'keyboard_verified':True,'zoom_factors':[1,1.25]}
    item={'summary':write(tmp_path,'summary.json',summary)}
    gate.validate_category('ui',gate.Evidence(tmp_path),item,summary,{'source':source})
    summary['source_sha256']['web/src/main.ts']='c'*64
    with pytest.raises(gate.GateError,match='Native UI source identity differs'):
        gate.validate_category('ui',gate.Evidence(tmp_path),item,summary,{'source':source})


def test_quiet_typecheck_has_a_real_empty_log_not_a_missing_log(tmp_path,candidate):
    evidence=gate.Evidence(tmp_path);context=gate.candidate_context(evidence,candidate)
    log=write(tmp_path,'quiet/typecheck.log',b'')
    summary={'status':'passed','steps':[{'name':'typecheck','command':['node','tsc','--noEmit'],
              'exit_code':0,'state':'completed','log':'typecheck.log','log_sha256':log['sha256']}]}
    item={'status':'completed','exit_code':0,'command':['node','tsc','--noEmit'],'log':log,
          'summary':write(tmp_path,'quiet/summary.json',summary),'identity':scoped_identity('typecheck',context)}
    manifest={'schema':'neuroterrarium.release-evidence.v1','phase':'local','candidate':candidate,'items':{'typecheck':item}}
    result=gate.validate(manifest,evidence)
    assert next(row for row in result['checks'] if row['name']=='typecheck')['status']=='passed'
    assert result['status']=='blocked'  # All other release categories remain absent.
    (tmp_path/log['path']).unlink()
    result=gate.validate(manifest,evidence)
    assert next(row for row in result['checks'] if row['name']=='typecheck')['status']=='blocked'


def test_supplement_budget_revision_only_changes_allowance_and_description():
    configs=Path(__file__).parents[1]/'configs'
    old=json.loads((configs/'evaluate-supplement-v1.json').read_text())
    new=json.loads((configs/'evaluate-supplement-v1b.json').read_text())
    assert [row['path'] for row in gate.verify_supplement_config_revision(old,new)]==['/wall_budget_seconds','/resources']
    for field,value in [('visual_noise_std',.2),('wall_budget_seconds',1800.),('resources',old['resources']),('shared_windows',101)]:
        mutated={**new,field:value}
        with pytest.raises(gate.GateError):gate.verify_supplement_config_revision(old,mutated)


@pytest.fixture
def supplement_revision(tmp_path):
    """Synthetic metadata preserves all planned units; never a biological dataset."""
    configs=Path(__file__).parents[1]/'configs'
    old_config=json.loads((configs/'evaluate-supplement-v1.json').read_text())
    new_config=json.loads((configs/'evaluate-supplement-v1b.json').read_text())
    old_source="""def validate_protocol(config):
    expected = {'wall_budget_seconds': 900, 'stress_windows': 75}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('settings differ')

def trajectory(value):
    return value * 2
"""
    new_source="""def validate_protocol(config):
    expected = {'stress_windows': 75}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('settings differ')
    wall_budget = config.get('wall_budget_seconds')
    if type(wall_budget) is not int or wall_budget not in (900, 1800):
        raise ValueError('Supplement wall budget must match a frozen resource version')
    allowance = 'Nine hundred' if wall_budget == 900 else 'Eighteen hundred'
    description = (f'{allowance} cumulative wall seconds including setup across resumed invocations, '
                   'checked at episode boundaries and action-window timeout; small recording and teardown '
                   'overhead can exceed the boundary. Preserve partial records and list not-started units '
                   'on cancellation, budget exhaustion or low resources.')
    if config.get('resources') != ('One immutable full-graph structure and at most one active connectome state. '
                                  'One CPU thread for learned inference. ' + description):
        raise ValueError('Supplement resource description differs from the frozen resource version')

def trajectory(value):
    return value * 2
"""
    before=write(tmp_path,'sources/original.py',old_source.encode());after=write(tmp_path,'sources/revised.py',new_source.encode())
    original=write(tmp_path,'configs/original.json',old_config);revised=write(tmp_path,'configs/revised.json',new_config)
    plan=[{'family':'stress','controller':f'controller-{i//60}','condition':['clean','visual_noise'][(i//30)%2],
           'environment_seed':190001+i%30,'allocation':[f'controller-{i//60}'],'windows':75} for i in range(600)]
    plan.extend({'family':'shared','controller':'shared_population','condition':'clean','environment_seed':290001+i,
                 'allocation':[f'controller-{j}' for j in range(10)],'windows':100} for i in range(30))
    resource_limits={'max_total_wall_seconds':900,'max_process_rss_bytes':2*2**30,
                     'min_available_memory_bytes':3*2**30,'min_free_disk_bytes':10*2**30}
    old_lock={'config':old_config,'source':{'supplemental.py':before['sha256'],'world.py':'f'*64},'plan':plan,'model_manifest_sha256':'a'*64,
              'resource_limits':resource_limits}
    new_lock={**old_lock,'config':new_config,'source':{**old_lock['source'],'supplemental.py':after['sha256']},
              'resource_limits':{**resource_limits,'max_total_wall_seconds':1800}}
    old_protocol=gate.canonical(old_lock);new_protocol=gate.canonical(new_lock);previous=[];current=[]
    directory='evidence/original';parent=tmp_path/directory
    for i,spec in enumerate(plan):
        row={**spec,'protocol_sha256':old_protocol,'status':'completed' if i<620 else 'interrupted' if i==620 else 'not_started'}
        name=f"{spec['family']}--{spec['controller']}--{spec['condition']}--{spec['environment_seed']}"
        if i<621:
            raw=write(tmp_path,directory+'/raw/'+name+'.npz',b'synthetic checksum fixture')
            row.update(trajectory=name+'.npz',trajectory_sha256=raw['sha256'],completed_windows=spec['windows'] if i<620 else 43)
            if i==620:row['failure']={'reason':'wall_budget'}
            write(tmp_path,directory+'/raw/'+name+'.json',row)
        else:row['reason']='wall_budget'
        previous.append(row);current.append({**spec,'status':'completed','protocol_sha256':new_protocol})
    old_result={'schema':'neuroterrarium.supplement-results.v1','protocol_sha256':old_protocol,'status':'incomplete',
                'reason':'wall_budget','wall_seconds':900.7,'records':previous}
    write(tmp_path,directory+'/results.json',old_result);write(tmp_path,directory+'/protocol-lock.json',old_lock)
    write(tmp_path,directory+'/budget.json',{'protocol_sha256':old_protocol,'limits':resource_limits,'elapsed_seconds':900.7})
    write(tmp_path,directory+'/graph-summary.json',{})
    lineage={'schema':'neuroterrarium.supplement-resource-revision.v1','revision_id':'supplement-v1b',
      'parent':{'directory':directory,'status':'incomplete','reason':'wall_budget','exit_code':1,'protocol_sha256':old_protocol,
                'config_sha256':original['sha256'],'source_sha256':{'src/neuroterrarium/'+k:v for k,v in old_lock['source'].items()},
                'wall_seconds':900.7,'status_counts':{'completed':620,'interrupted':1,'not_started':9},
                'evidence_sha256':{str(p.relative_to(parent)):gate.sha256(p) for p in parent.rglob('*') if p.is_file()}},
      'revised':{'directory':'evidence/revised','config_sha256':revised['sha256'],
                 'source_sha256':{'src/neuroterrarium/'+k:v for k,v in new_lock['source'].items()},
                 'expected_total_units':630,'expected_stress_units':600,'expected_shared_world_units':30,'reuse_parent_records':False},
      'configuration_diff':[{'path':'/wall_budget_seconds','before':900,'after':1800},
                            {'path':'/resources','before':old_config['resources'],'after':new_config['resources']}],
      'validator_diff':{'before_sha256':before['sha256'],'after_sha256':after['sha256']}}
    commands=[write(tmp_path,'commands/'+phase+'.json',{'phase':phase,'command':['python','scripts/evaluate_supplement.py'],
                 'status':'completed','exit_code':1 if phase=='parent' else 0,'protocol_sha256':old_protocol if phase=='parent' else new_protocol})
              for phase in ('parent','execution')]
    spec={'manifest':write(tmp_path,'revision.json',lineage),'parent_directory':directory,'parent_source':before,'parent_config':original,
          'commands':commands,'release_asset':'scientific-evidence-v1.tar.gz'}
    context={'config_documents':{'evaluate-supplement-v1.json':old_config,'evaluate-supplement-v1b.json':new_config},
             'configs':{'evaluate-supplement-v1b.json':revised['sha256']},'source':new_lock['source'],
             'source_paths':{'supplemental.py':tmp_path/after['path']}}
    summary={'status':'completed','protocol_sha256':new_protocol,'records':current}
    return {'root':tmp_path,'context':context,'spec':spec,'summary':summary,'lock':new_lock,'lineage':lineage}


def check_supplement_revision(f):
    return gate.verify_supplement_revision(gate.Evidence(f['root']),f['summary'],f['lock'],f['context'],f['spec'])


def test_supplement_revision_exposes_all_failed_predecessors_without_pooling(supplement_revision):
    report=check_supplement_revision(supplement_revision)
    assert report['parent_exit_code']==1 and report['parent_status_counts']=={'completed':620,'interrupted':1,'not_started':9}
    assert len(report['incomplete_parent_units'])==10 and report['incomplete_parent_units'][0]['completed_windows']==43
    assert report['newly_executed_units']==630 and report['reused_parent_units']==report['pooled_parent_units']==0
    assert len(report['parent_files_sha256'])==1246


@pytest.mark.parametrize('mutation',['hide_parent_file','completed_label','reuse','wrong_diff','omit_final','old_row','source_behavior','source_guard','running','old_exit_zero','traversal'])
def test_supplement_revision_rejects_forged_or_incomplete_lineage(supplement_revision,mutation):
    f=supplement_revision;lineage=f['lineage']
    if mutation=='hide_parent_file':lineage['parent']['evidence_sha256'].pop(next(name for name in lineage['parent']['evidence_sha256'] if name.startswith('raw/')))
    elif mutation=='completed_label':lineage['parent']['status']='completed'
    elif mutation=='reuse':lineage['revised']['reuse_parent_records']=True
    elif mutation=='wrong_diff':lineage['configuration_diff']=lineage['configuration_diff'][:1]
    elif mutation=='omit_final':f['summary']['records'].pop()
    elif mutation=='old_row':f['summary']['records'][0]['protocol_sha256']=lineage['parent']['protocol_sha256']
    elif mutation in {'source_behavior','source_guard'}:
        path=f['context']['source_paths']['supplemental.py']
        content=path.read_text().replace('return value * 2','return value * 3') if mutation=='source_behavior' else path.read_text().replace('(900, 1800)','(900, 1800, 3600)')
        path.write_text(content)
    elif mutation in {'running','old_exit_zero'}:
        index=1 if mutation=='running' else 0;ref=f['spec']['commands'][index]
        doc=json.loads((f['root']/ref['path']).read_text())
        if mutation=='running':doc['status']='running'
        else:doc['exit_code']=0
        f['spec']['commands'][index]=write(f['root'],ref['path'],doc)
    else:lineage['parent']['directory']=f['spec']['parent_directory']='../outside'
    f['spec']['manifest']=write(f['root'],'revision.json',lineage)
    with pytest.raises((gate.GateError,OSError)):check_supplement_revision(f)


def test_supplement_v1b_cannot_pass_without_explicit_revision_evidence(supplement_revision):
    supplement_revision['spec']=None
    with pytest.raises(gate.GateError,match='explicit resource-revision'):check_supplement_revision(supplement_revision)


def test_core_native_source_receipt_includes_report_and_validator(tmp_path,candidate):
    context=gate.candidate_context(gate.Evidence(tmp_path),candidate);identity=scoped_identity('core_tests',context)
    native={('src/neuroterrarium/'+key if '/' not in key else key):value for key,value in identity['source_sha256'].items()}
    native.update({'configs/'+key:value for key,value in identity['config_sha256'].items()})
    summary={'tested_source_files_sha256':native,'source_changes_during_run':[]}
    gate.verify_identity('core_tests',identity,summary,context,gate.Evidence(tmp_path),{})
    del native['scripts/report_results.py']
    with pytest.raises(gate.GateError,match='Native test source'):gate.verify_identity('core_tests',identity,summary,context,gate.Evidence(tmp_path),{})


def test_supplement_resource_allowance_mirror_preserves_protection_thresholds():
    config={'wall_budget_seconds':900,'memory_reserve_bytes':3*2**30,'disk_reserve_bytes':10*2**30}
    limits={'max_total_wall_seconds':900,'max_process_rss_bytes':2*2**30,
            'min_available_memory_bytes':3*2**30,'min_free_disk_bytes':10*2**30}
    old={'config':config,'resource_limits':limits}
    new={'config':{**config,'wall_budget_seconds':1800},'resource_limits':{**limits,'max_total_wall_seconds':1800}}
    gate.verify_supplement_resource_limits(old,new)
    mutations=[lambda d:d.pop('resource_limits'),
               lambda d:d['resource_limits'].pop('max_process_rss_bytes'),
               lambda d:d['resource_limits'].update(extra=1),
               lambda d:d['resource_limits'].update(max_process_rss_bytes=3*2**30),
               lambda d:d['resource_limits'].update(min_available_memory_bytes=2*2**30),
               lambda d:d['resource_limits'].update(min_free_disk_bytes=9*2**30),
               lambda d:d['resource_limits'].update(max_total_wall_seconds=1800.),
               lambda d:d['resource_limits'].update(max_process_rss_bytes=True),
               lambda d:d['resource_limits'].update(max_process_rss_bytes=0),
               lambda d:d['resource_limits'].update(max_total_wall_seconds=900),
               lambda d:d['config'].update(wall_budget_seconds=900)]
    for mutate in mutations:
        changed=copy.deepcopy(new);mutate(changed)
        with pytest.raises(gate.GateError):gate.verify_supplement_resource_limits(old,changed)
    for mutate in mutations[:3]:
        changed=copy.deepcopy(old);mutate(changed)
        with pytest.raises(gate.GateError):gate.verify_supplement_resource_limits(changed,new)


@pytest.fixture
def browser_transport(tmp_path):
    old="def serve():\n    app = create_app()\n    if open_browser:\n        import webbrowser\n        threading.Timer(1.5,lambda:webbrowser.open(f'http://127.0.0.1:{port}')).start()\n    uvicorn.run(app,host=host)\n"
    new=old.replace("        threading.Timer(1.5,lambda:webbrowser.open(f'http://127.0.0.1:{port}')).start()", "        browser_host=f'[{host}]' if ':' in host else host\n        threading.Timer(1.5,lambda:webbrowser.open(f'http://{browser_host}:{port}')).start()")
    before=write(tmp_path,'old-service.py',old.encode());after=write(tmp_path,'service.py',new.encode())
    regression_source=write(tmp_path,'tests/test_service.py',b'regression source')
    validator=write(tmp_path,'scripts/validate_release.py',b'verifier source')
    log=write(tmp_path,'regression/core.log',b'3 passed')
    summary=write(tmp_path,'regression/summary.json',{'status':'passed','source_changes_during_run':[],
        'tested_source_files_sha256':{'src/neuroterrarium/service.py':after['sha256'],'tests/test_service.py':regression_source['sha256'],
                                     'scripts/validate_release.py':validator['sha256']},
        'executed_tests':3,'steps':[{'name':'core_tests','state':'completed','exit_code':0,'command':['python','-m','pytest'],
                                    'log':'core.log','log_sha256':log['sha256']}]})
    xml='<testsuite tests="3">'+''.join('<testcase classname="tests.test_service" name="test_browser_opens_the_bound_loopback_address['+host+'-url]"/>' for host in ('127.0.0.1','localhost','::1'))+'</testsuite>'
    junit=write(tmp_path,'regression/core.xml',xml.encode())
    return {'root':tmp_path,'source':old,'new':new,'spec':{'schema':'neuroterrarium.browser-launch-compatibility.v1',
        'original_source':before,'candidate_source':after,'regression_summary':summary,'regression_junit':junit},
        'sources':{'service.py':before['sha256']},'context':{'source':{'service.py':after['sha256'],
            'tests/test_service.py':regression_source['sha256'],'scripts/validate_release.py':validator['sha256']}}}


def transport_result(f,name='ui'):
    return gate.verify_transport_compatibility(name,gate.Evidence(f['root']),f['sources'],f['context'],{'transport_compatibility':f['spec']})


def test_browser_url_exception_preserves_actual_historical_service_identity(browser_transport):
    f=browser_transport;original=dict(f['sources'])
    assert transport_result(f) and transport_result(f,'stability')
    assert f['sources']==original and f['sources']['service.py']!=f['context']['source']['service.py']


@pytest.mark.parametrize('change',['other_function','branch_delay','address','original_branch','source_binding','category',
    'running_regression','wrong_regression_source','missing_both_test_sources','failed_test','missing_host','missing_log','unknown_field'])
def test_transport_exception_rejects_other_edits_or_unproven_regressions(browser_transport,change):
    f=browser_transport;s=f['spec'];root=f['root']
    if change in {'other_function','branch_delay','address'}:
        text=f['new'].replace('create_app()','create_app(privileged=True)') if change=='other_function' else f['new'].replace('1.5','1.6') if change=='branch_delay' else f['new'].replace("':' in host", "'.' in host")
        s['candidate_source']=write(root,'service.py',text.encode());f['context']['source']['service.py']=s['candidate_source']['sha256']
    elif change=='original_branch':
        s['original_source']=write(root,'old-service.py',f['source'].replace('127.0.0.1','localhost').encode());f['sources']['service.py']=s['original_source']['sha256']
    elif change=='source_binding':f['sources']['service.py']='0'*64
    elif change in {'running_regression','wrong_regression_source','missing_both_test_sources'}:
        p=root/s['regression_summary']['path'];d=json.loads(p.read_text())
        if change=='running_regression':d['status']='running'
        elif change=='missing_both_test_sources':
            d['tested_source_files_sha256'].pop('tests/test_service.py');f['context']['source'].pop('tests/test_service.py')
        else:d['tested_source_files_sha256']['src/neuroterrarium/service.py']='0'*64
        s['regression_summary']=write(root,str(p.relative_to(root)),d)
    elif change in {'failed_test','missing_host'}:
        p=root/s['regression_junit']['path'];text=p.read_text()
        text=text.replace('/>','><failure/></testcase>',1) if change=='failed_test' else text.replace('[::1-url]','[unregistered-url]')
        s['regression_junit']=write(root,str(p.relative_to(root)),text.encode())
    elif change=='missing_log':(root/'regression/core.log').unlink()
    elif change=='unknown_field':s['allow_any_change']=True
    with pytest.raises((ValueError,OSError)):
        transport_result(f,'runtime' if change=='category' else 'ui')
