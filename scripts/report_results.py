"""Generate fixed task-level reports from completed, revalidated experiment records."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace

METRICS=('food','collision_substeps','motor_energy_demand','final_simulated_energy',
         'escape_drive_integral','inference_mean_ms','reaction_seconds','recovery_seconds')
REQUIRED_EVIDENCE={'training','neural','runtime','stability','installed_wheel','portable'}
PRESET_EXAMPLES=(('A_expansion','motor_energy_demand'),('C_relocation','food'))
RECOVERY_ARRAYS=('observation','applied_action','body','initial_body','motor_groups_hz',
                 'neural_input_hz','raw_policy_action','raw_policy_log_probability','motor_energy_demand')


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>256*2**20:raise ValueError('Missing or excessive report input')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(path.read_text(),object_pairs_hook=unique,
                      parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def reference(root,ref):
    name=ref['path']
    if not isinstance(name,str) or PurePosixPath(name).is_absolute() or '\\' in name or ':' in name or any(part in {'','.','..'} for part in name.split('/')):
        raise ValueError('Report evidence must use normalized relative paths')
    path=Path(root)
    for part in name.split('/'):
        path=path/part
        if path.is_symlink():raise ValueError('Symlink report evidence')
    if not path.is_file() or digest(path)!=ref['sha256']:raise ValueError('Report evidence checksum mismatch')
    if 'bytes' in ref and path.stat().st_size!=ref['bytes']:raise ValueError('Report evidence length mismatch')
    return path


def directory_reference(root,name):
    if not isinstance(name,str) or PurePosixPath(name).is_absolute() or '\\' in name or ':' in name or any(part in {'','.','..'} for part in name.split('/')):
        raise ValueError('Recovery directory must use a normalized relative path')
    path=Path(root)
    for part in name.split('/'):
        path=path/part
        if path.is_symlink():raise ValueError('Symlink recovery directory')
    if not path.is_dir():raise ValueError('Recovery parent directory is missing')
    return path


def validate_recovery_step(core_path,core,evidence_root,spec,*,require_completed=True):
    """Revalidate one actual transition, including an interrupted intermediate run."""
    core_path=Path(core_path)
    present=(core_path/'recovery-manifest.json').is_file()
    if not present:
        if spec is not None:raise ValueError('Recovery evidence supplied without a recovery manifest')
        return None
    if not isinstance(spec,dict) or not {'parent_directory','manifest','commands','release_asset'}<=spec.keys():
        raise ValueError('A recovered experiment requires an explicit recovery evidence index')
    if set(spec)-{'parent_directory','manifest','commands','prefix_proof','release_asset'}:
        raise ValueError('Unrecognized recovery evidence field')
    asset=spec['release_asset']
    if not isinstance(asset,str) or PurePosixPath(asset).name!=asset or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]+',asset):
        raise ValueError('Recovery evidence requires a versioned release asset name')
    manifest_path=reference(evidence_root,spec['manifest'])
    if manifest_path.resolve()!= (core_path/'recovery-manifest.json').resolve():
        raise ValueError('Recovery reference does not identify the evaluated experiment')
    lineage=read(manifest_path)
    if lineage.get('schema')!='neuroterrarium.evaluation-recovery.v1':raise ValueError('Unsupported recovery manifest')
    parent=directory_reference(evidence_root,spec['parent_directory'])
    if lineage.get('parent_experiment')!=parent.name or parent.resolve()==core_path.resolve():
        raise ValueError('Recovery parent identity mismatch')
    files=lineage.get('parent_files_sha256',{})
    if not isinstance(files,dict) or not {'results.json','protocol-lock.json','graph-summary.json','budget.json','progress.json'}<=files.keys():
        raise ValueError('Recovery predecessor evidence is incomplete')
    actual={str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file()}
    if actual!=set(files):raise ValueError('Recovery manifest omits or invents predecessor files')
    for name,expected in files.items():reference(parent,{'path':name,'sha256':expected})
    old=read(parent/'results.json');old_lock=read(parent/'protocol-lock.json');budget=read(parent/'budget.json')
    if old.get('schema')!='neuroterrarium.evaluation-results.v1' or old.get('status') not in {'resource_limited','cancelled','failed','incomplete','budget_exhausted'}:
        raise ValueError('Recovery predecessor has no registered incomplete execution status')
    protocol=canonical(core['lock'])
    if old_lock!=core['lock'] or lineage.get('protocol_sha256')!=protocol or old.get('protocol_sha256')!=protocol or lineage.get('source')!=core['lock']['source']:
        raise ValueError('Recovery changed the frozen scientific protocol or source')
    if budget.get('protocol_sha256')!=protocol or lineage.get('limits')!=budget.get('limits') or lineage.get('inherited_wall_seconds')!=budget.get('elapsed_seconds'):
        raise ValueError('Recovery resource-accounting provenance differs')

    def indexed(rows):
        result={}
        for row in rows:
            name=f'{row["controller"]}--{row["ablation"]}--{row["scenario"]}--{row["environment_seed"]}'
            if name in result:raise ValueError('Duplicate recovery experiment unit')
            result[name]=row
        return result

    if core['result'].get('schema')!='neuroterrarium.evaluation-results.v1':raise ValueError('Recovery child result schema differs')
    previous=indexed(old['records']);current=indexed(core['result']['records'])
    for directory,records in ((parent,previous),(core_path,current)):
        wanted={name+suffix for name in records for suffix in ('.json','.npz')}
        if {path.name for path in (directory/'raw').iterdir()}!=wanted:
            raise ValueError('Recovery raw directory contains omitted or unexpected experiment units')
    expected=old.get('expected_records');current_status=core['result'].get('status')
    if (type(expected) is not int or expected<len(previous) or core['result'].get('expected_records')!=expected
            or len(current)>expected or set(previous)-set(current)
            or core['result'].get('protocol_sha256')!=protocol):
        raise ValueError('Recovery omitted an original unit or changed the planned population')
    if require_completed:
        if current_status!='completed' or len(current)!=expected or any(r.get('status')!='completed' for r in current.values()):
            raise ValueError('Final recovery still has an incomplete or unrecorded unit')
    elif current_status not in {'resource_limited','cancelled','failed','incomplete','budget_exhausted'}:
        raise ValueError('An intermediate recovery must retain its unsuccessful execution status')
    child_budget=read(core_path/'budget.json')
    elapsed=child_budget.get('elapsed_seconds');inherited=lineage['inherited_wall_seconds']
    if (child_budget.get('protocol_sha256')!=protocol or child_budget.get('limits')!=budget['limits']
            or type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<inherited
            or ('maximum_total_wall_seconds' in lineage and lineage['maximum_total_wall_seconds']!=budget['limits'].get('max_total_wall_seconds'))):
        raise ValueError('Recovery changed resource protections or lost inherited elapsed time')
    parent_manifest=parent/'recovery-manifest.json'
    if parent_manifest.is_file() and lineage.get('parent_recovery_manifest_sha256')!=digest(parent_manifest):
        raise ValueError('Recovery does not bind its immediate predecessor recovery manifest')

    completed={name for name,row in previous.items() if row['status']=='completed'}
    incomplete={name:row for name,row in previous.items() if row['status']!='completed'}
    reused=lineage.get('reused_completed_identifiers',[]);rerun=lineage.get('rerun_identifiers',[])
    if len(reused)!=len(set(reused)) or set(reused)!=completed or len(rerun)!=len(set(rerun)) or set(rerun)!=set(incomplete):
        raise ValueError('Recovery reuse or rerun selection is incomplete or duplicated')
    if lineage.get('incomplete_parent_records')!=incomplete or lineage.get('not_previously_recorded_count')!=expected-len(previous):
        raise ValueError('Recovery does not enumerate every unfinished and unrecorded unit')
    for name,row in previous.items():
        if row.get('protocol_sha256')!=protocol:raise ValueError('Predecessor row protocol mismatch')
        trajectory=row.get('trajectory')
        if not isinstance(trajectory,str) or PurePosixPath(trajectory).name!=trajectory:
            raise ValueError('Unsafe predecessor trajectory reference')
        reference(parent/'raw',{'path':trajectory,'sha256':row['trajectory_sha256']})
        raw=parent/'raw'/(name+'.json')
        if read(raw)!=row:raise ValueError('Predecessor index differs from its raw record')
        if name in completed:
            if current[name]!=row or digest(core_path/'raw'/(name+'.json'))!=digest(raw):
                raise ValueError('A reused completed unit was changed')

    for name,row in current.items():
        if row.get('protocol_sha256')!=protocol:raise ValueError('Recovery child row protocol mismatch')
        if read(core_path/'raw'/(name+'.json'))!=row:raise ValueError('Recovery child index differs from raw record')
        trajectory=row.get('trajectory')
        if not isinstance(trajectory,str) or PurePosixPath(trajectory).name!=trajectory:
            raise ValueError('Unsafe child trajectory reference')
        reference(core_path/'raw',{'path':trajectory,'sha256':row['trajectory_sha256']})

    # Recompute the preserved simulated prefix directly. Wall costs and sampled
    # memory can change between invocations and are deliberately not equated.
    import numpy as np
    prefixes=[]
    for name in sorted(incomplete):
        before=incomplete[name];after=current[name];windows=before['completed_windows']
        if type(windows) is not int or windows<0:raise ValueError('Invalid interrupted prefix length')
        original=reference(parent/'raw',{'path':before['trajectory'],'sha256':before['trajectory_sha256']})
        recovered=reference(core_path/'raw',{'path':after['trajectory'],'sha256':after['trajectory_sha256']})
        with np.load(original,allow_pickle=False) as a,np.load(recovered,allow_pickle=False) as b:
            if not set(RECOVERY_ARRAYS)<=set(a.files) or not set(RECOVERY_ARRAYS)<=set(b.files):
                raise ValueError('Recovery prefix lacks a recorded simulated channel')
            for key in RECOVERY_ARRAYS:
                left=a[key];right=b[key] if key=='initial_body' else b[key][:windows]
                if key!='initial_body' and len(left)!=windows:raise ValueError('Interrupted array length differs from recorded windows')
                if left.dtype!=right.dtype or not np.array_equal(left,right,equal_nan=True):
                    raise ValueError('Recovered simulated prefix differs: '+key)
        prefixes.append({'identifier':name,'completed_prefix_windows':windows,'status':'exact',
            'old_trajectory_sha256':before['trajectory_sha256'],'recovery_trajectory_sha256':after['trajectory_sha256'],
            'compared_arrays':list(RECOVERY_ARRAYS),'excluded_cost_arrays':['inference_seconds','sampled_process_rss_bytes']})
    if 'prefix_proof' in spec:
        proof=read(reference(evidence_root,spec['prefix_proof']))
        matching=next((p for p in prefixes if p['identifier']==proof.get('identifier')),None)
        if matching is None or proof.get('status')!='passed' or proof.get('old_trajectory_sha256')!=matching['old_trajectory_sha256'] or proof.get('recovery_trajectory_sha256')!=matching['recovery_trajectory_sha256'] or set(proof.get('checks',{}))!=set(RECOVERY_ARRAYS) or any(v.get('status')!='passed' for v in proof['checks'].values()):
            raise ValueError('Recorded prefix proof disagrees with raw-array revalidation')
    commands=spec['commands']
    if not isinstance(commands,list) or not commands:raise ValueError('Recovery command lineage is missing')
    phases=set()
    for command_ref in commands:
        command=read(reference(evidence_root,command_ref));argv=command.get('argv',command.get('command'))
        phase=command.get('phase')
        if phase not in {'preparation','execution'} or phase in phases:raise ValueError('Recovery command phase is missing or duplicated')
        phases.add(phase)
        expected_exit=1 if phase=='execution' and not require_completed else 0
        if type(command.get('exit_code')) is not int or command['exit_code']!=expected_exit or command.get('status','completed') not in {'completed','passed'} or not isinstance(argv,(list,str)) or not argv or isinstance(argv,list) and any(not isinstance(value,str) or not value for value in argv):
            raise ValueError('Recovery command has no successful recorded completion or correct failed-stage exit')
        if re.search(r'/(?:Users|home|private/var/folders)/|[A-Za-z]:\\Users\\',json.dumps(command)):
            raise ValueError('Recovery command evidence must normalize private paths before publication')
    if not {'preparation','execution'}<=phases:raise ValueError('Recovery preparation and execution commands must both be recorded')
    new_identifiers=set(current)-set(previous)
    return {'schema':'neuroterrarium.recovery-report.v1','parent_directory':spec['parent_directory'],
        'child_directory':str(core_path.resolve().relative_to(Path(evidence_root).resolve())),
        'child_status':current_status,'expected_units':expected,'parent_recorded_units':len(previous),
        'planned_new_units':expected-len(previous),'recorded_new_units':len(new_identifiers),
        'completed_new_units':sum(current[name]['status']=='completed' for name in new_identifiers),
        'incomplete_new_units':[{'identifier':name,'status':current[name]['status'],'failure':current[name].get('failure')} for name in sorted(new_identifiers) if current[name]['status']!='completed'],
        'missing_new_units':expected-len(current),
        'recorded_execution_attempts':len(new_identifiers)+len(incomplete),
        'parent_status':old['status'],'parent_stop':old.get('stop'),'parent_files_sha256':files,
        'protocol_sha256':protocol,'source':lineage['source'],'reason':lineage['reason'],
        'reused_completed_count':len(completed),'reused_completed_identifiers':sorted(completed),
        'rerun_units':[{'identifier':name,'original_status':incomplete[name]['status'],
            'failure':incomplete[name].get('failure'),'original_completed_windows':incomplete[name]['completed_windows']} for name in sorted(incomplete)],
        'newly_executed_units':len(current)-len(previous),'omitted_parent_units':[],
        'inherited_wall_seconds':lineage['inherited_wall_seconds'],'resource_limits':lineage['limits'],
        'prefix_comparisons':prefixes,'evidence':spec}


def validate_recovery(core_path,core,evidence_root,spec):
    """Bind every chronological recovery stage; only the final run may complete."""
    if not (Path(core_path)/'recovery-manifest.json').is_file():
        if spec is not None:raise ValueError('Recovery evidence supplied without a recovery manifest')
        return None
    if not isinstance(spec,dict):raise ValueError('A recovered experiment requires an explicit recovery evidence index')
    ancestors=spec.get('ancestors',[])
    if not isinstance(ancestors,list) or len(ancestors)>8:raise ValueError('Invalid or excessive recovery ancestry')
    steps=[];seen=set();previous_child=None;parent_names=[]
    for item in [*ancestors,{**{key:value for key,value in spec.items() if key!='ancestors'},
                            'child_directory':str(Path(core_path).resolve().relative_to(Path(evidence_root).resolve()))}]:
        if not isinstance(item,dict) or 'child_directory' not in item:raise ValueError('Recovery ancestor needs an explicit child directory')
        child=directory_reference(evidence_root,item['child_directory']);parent=directory_reference(evidence_root,item['parent_directory'])
        if child in seen or parent==child:raise ValueError('Recovery ancestry contains a duplicate or cycle')
        if previous_child is None:
            if (parent/'recovery-manifest.json').exists():raise ValueError('Recovery ancestry omitted an earlier recovery stage')
        elif parent!=previous_child:raise ValueError('Recovery ancestry is out of order or disconnected')
        seen.add(child);previous_child=child
        step_spec={key:value for key,value in item.items() if key!='child_directory'}
        final=child.resolve()==Path(core_path).resolve()
        data=core if final else {'lock':read(child/'protocol-lock.json'),'result':read(child/'results.json')}
        if data['lock']!=core['lock']:raise ValueError('Recovery ancestry changed the frozen lock')
        lineage=read(reference(evidence_root,step_spec['manifest']))
        if 'ancestor_experiments' in lineage and lineage['ancestor_experiments']!=parent_names:
            raise ValueError('Recovery declared ancestry differs from the actual chain')
        step=validate_recovery_step(child,data,evidence_root,step_spec,require_completed=final)
        steps.append(step);parent_names.append(parent.name)
    final=steps[-1]
    if final['child_status']!='completed':raise ValueError('Recovery chain does not end in the completed experiment')
    return {**final,'history':steps,'evidence':spec,'independent_final_units':final['expected_units'],
            'recorded_execution_attempts':steps[0]['parent_recorded_units']+sum(row['recorded_execution_attempts'] for row in steps)}


def validate_evidence(root,index,models,model_hash):
    if index.get('schema')!='neuroterrarium.report-evidence.v1' or not REQUIRED_EVIDENCE<=set(index.get('evidence',{})):
        raise ValueError('Complete supporting evidence index is required')
    documents={}
    schemas={'training':('neuroterrarium.training-results.v1',{'complete'}),
             'neural':('neural-gate-v1',{'passed'}),'runtime':('neuroterrarium.runtime-validation.v1',{'passed'}),
             'stability':('neuroterrarium.stability.v1',{'completed'}),
             'installed_wheel':('neuroterrarium.installation-check.v1',{'passed'}),
             'portable':('neuroterrarium.portable-check.v1',{'passed'})}
    for name,ref in index['evidence'].items():
        if set(ref)-{'path','sha256','bytes','release_asset'}:raise ValueError('Unrecognized supporting reference metadata')
        doc=read(reference(root,ref));documents[name]=doc
        if name in schemas and (doc.get('schema')!=schemas[name][0] or doc.get('status') not in schemas[name][1]):
            raise ValueError('Supporting evidence is unfinished or has wrong schema: '+name)
        asset=ref.get('release_asset')
        if not isinstance(asset,str) or PurePosixPath(asset).name!=asset or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]+',asset):
            raise ValueError('Each supporting reference needs its versioned release asset name')
    training=documents['training']
    if training.get('model_set_manifest_sha256')!=model_hash or documents['runtime'].get('model_manifest_sha256')!=model_hash or documents['stability'].get('models_sha256')!=model_hash:
        raise ValueError('Supporting evidence model identity mismatch')
    if len(training.get('models',[]))!=9 or {row['id'] for row in training['models']}!={row['id'] for row in models['models']}:
        raise ValueError('Training evidence does not cover nine models')
    if any(row.get('transitions')!=models['transitions_per_model'] for row in training['models']):
        raise ValueError('Training budget mismatch')
    neural=documents['neural']
    if not {'gustation','visual_projection','closed_feeding','closed_expansion','full_graph_reference'}<=set(neural.get('checks',{})) or not all(value is True for value in neural['checks'].values()):
        raise ValueError('Neural positive control evidence is incomplete')
    hour=documents['stability']
    if hour.get('mode')!='Local full graph' or hour.get('measured_wall_seconds',0)<3600 or set(hour.get('controllers',[]))!={'connectome',*[row['id'] for row in models['models']]} or hour.get('error') is not None:
        raise ValueError('A completed live ten-controller hour is required')
    for name in ('installed_wheel','portable'):
        steps=documents[name].get('steps',[])
        if not steps or any(type(step.get('exit_code')) is not int or step['exit_code']!=0 or step.get('state')!='completed' for step in steps):
            raise ValueError('Installation evidence contains unfinished command steps')
    return documents


def release_validator():
    """Use the published standalone artifact validator for revision provenance."""
    spec=importlib.util.spec_from_file_location('release_validator',Path(__file__).with_name('validate_release.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_supplement_revision(supplement_path,supplement,evidence_root,spec):
    from neuroterrarium import supplemental
    from neuroterrarium.registry import default_path
    gate=release_validator();config_root=default_path().parent
    names=('evaluate-supplement-v1.json','evaluate-supplement-v1b.json')
    context={'config_documents':{name:read(config_root/name) for name in names},
             'configs':{name:digest(config_root/name) for name in names},
             'source':supplement['lock']['source'],
             'source_paths':{'supplemental.py':Path(supplemental.__file__)}}
    result=gate.verify_supplement_revision(gate.Evidence(evidence_root),supplement['result'],supplement['lock'],context,spec)
    if result is not None:
        lineage=read(reference(evidence_root,spec['manifest']))
        if directory_reference(evidence_root,lineage['revised']['directory']).resolve()!=Path(supplement_path).resolve():
            raise ValueError('Supplement revision does not name this evaluated directory')
    return result


def policy_plan_metadata(models):
    return {row['id']:SimpleNamespace(architecture=row['architecture']) for row in models['models']}


def load_experiments(core_path,supplement_path,models_path):
    for path,schema in [(Path(core_path),'neuroterrarium.evaluation-results.v1'),(Path(supplement_path),'neuroterrarium.supplement-results.v1')]:
        preliminary=read(path/'results.json')
        if preliminary.get('schema')!=schema or preliminary.get('status')!='completed':
            raise ValueError('Only completed frozen experiment versions can be reported')
    # Importing the shared numerical result readers does not create a brain.
    from neuroterrarium import evaluation as ev, supplemental as sup
    from neuroterrarium.registry import default_path
    models=read(Path(models_path)/'manifest.json')
    if models.get('status')!='completed' or len(models.get('models',[]))!=9:raise ValueError('Completed model manifest required')
    if len({row['id'] for row in models['models']})!=9 or len({row['policy_sha256'] for row in models['models']})!=9 or models.get('transitions_per_model',0)<1000000:
        raise ValueError('Nine distinct formally trained models are required')
    policies=policy_plan_metadata(models)
    outputs=[]
    for path,is_supplement in ((Path(core_path),False),(Path(supplement_path),True)):
        result=read(path/'results.json');lock=read(path/'protocol-lock.json')
        required_schema='neuroterrarium.supplement-results.v1' if is_supplement else 'neuroterrarium.evaluation-results.v1'
        if result.get('schema')!=required_schema or result.get('status')!='completed' or result.get('protocol_sha256')!=canonical(lock):
            raise ValueError('Only completed frozen experiment versions can be reported')
        if lock.get('model_manifest_sha256')!=canonical(models):raise ValueError('Experiment model identity mismatch')
        expected_source={**ev.evaluation_source_digest()}
        if is_supplement:expected_source['supplemental.py']=digest(Path(sup.__file__))
        if lock.get('source')!=expected_source:raise ValueError('Experiment source differs from current result reader')
        for field,name in [('interface_sha256','interface-v783.json'),('brain_interface_sha256','brain-interface.json'),
                           ('storage_profile_sha256','brain-storage-v1.json'),('data_manifest_sha256','data-v783.json')]:
            if lock.get(field)!=digest(default_path().with_name(name)):raise ValueError('Experiment configuration identity mismatch')
        protocol_name=('evaluate-supplement-v1b.json' if lock['config'].get('wall_budget_seconds')==1800 else 'evaluate-supplement-v1.json') if is_supplement else 'evaluate-v1.json'
        if lock['config']!=read(default_path().with_name(protocol_name)):
            raise ValueError('Experiment protocol differs from the frozen release configuration')
        rows=result['records']
        if is_supplement:
            sup.validate_protocol(lock['config'])
            if lock.get('plan')!=sup._plan(lock['config'],policies):raise ValueError('Supplement plan differs from frozen allocation protocol')
            if len(rows)!=630 or len(rows)!=len(lock['plan']):raise ValueError('Missing supplemental trials')
            calculated=sup.summarize_supplement(path,persist=False)
        else:
            config=lock['config'];seeds=config['environment_seeds']
            if config.get('status')!='frozen' or config.get('scenarios')!=list(ev.SCENARIOS) or len(seeds)<30 or len(set(seeds))!=len(seeds):
                raise ValueError('Core task families or frozen environment units are incomplete')
            units=[]
            for name,ablation,cases in ev.evaluation_plan(config,policies):
                for case in cases:
                    for seed in seeds:
                        units.append((f'{name}--{ablation}--{case}--{seed}',name,ablation,case,seed))
            actual=ev._existing_records(path/'raw',units,canonical(lock))
            if result.get('expected_records')!=len(units) or result.get('missing_records')!=0 or rows!=[actual[unit[0]] for unit in units]:
                raise ValueError('Core index omits or changes frozen trials')
            calculated=ev.summarize(path,persist=False)
        if not rows or any(row.get('status')!='completed' for row in rows):raise ValueError('Failed, timed out or unfinished trials cannot enter a completed report')
        if calculated!=read(path/'summary.json'):raise ValueError('Stored statistics differ from raw-record recomputation')
        graph=read(path/'graph-summary.json');storage=read(default_path().with_name('brain-storage-v1.json'))
        if graph.get('protocol_sha256')!=canonical(lock) or graph.get('neural_graph_sha256')!=storage['graph_digest'] or canonical(graph.get('profile'))!=storage['summary_sha256']:
            raise ValueError('Experiment does not identify the complete frozen graph')
        outputs.append({'result':result,'lock':lock,'summary':calculated,
                        'hashes':{name:digest(path/name) for name in ('results.json','protocol-lock.json','summary.json','graph-summary.json')}})
    return outputs[0],outputs[1],models


def interval(value):
    if value.get('mean') is None:return f"undefined (n={value.get('defined_units',0)})"
    lo,hi=value['ci95'];mean=value['mean']
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (mean,lo,hi)) or not lo<=hi:
        raise ValueError('Malformed defined interval')
    return f'{mean:.4g} [{lo:.4g}, {hi:.4g}] (n={value["defined_units"]})'


def cell(value):
    return str(value).replace('|','\\|').replace('\n',' ')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(cell(value) for value in row)+' |' for row in rows]])


def flatten(row,prefix=''):
    result={}
    for key,value in row.items():
        name=prefix+key
        if isinstance(value,dict):result.update(flatten(value,name+'_'))
        elif key.endswith('ci95') and (value is None or isinstance(value,list) and len(value)==2):
            result[name+'_low']=None if value is None else value[0]
            result[name+'_high']=None if value is None else value[1]
        else:result[name]=value
    return result


def csv_bytes(rows):
    rows=list(rows);keys=list(dict.fromkeys(key for row in rows for key in row))
    output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=keys,lineterminator='\n');writer.writeheader()
    for row in rows:writer.writerow({key:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False) if isinstance(value,(dict,list)) else value for key,value in row.items()})
    return output.getvalue().encode()


def render(core,supplement,evidence,refs,tag,recovery=None,supplement_revision=None):
    rows=core['result']['records'];summary=core['summary']
    names=['connectome',*sorted({row['controller'] for row in rows if row['controller'] not in {'connectome','random','rule','untrained'}}),'random','rule','untrained']
    grouped={(row['scenario'],row['controller']):row for row in summary['single_controller'] if row['ablation']=='none'}
    def outcome(name,case,field,value):
        selected=[row for row in rows if row['controller']==name and row['scenario']==case and row['ablation']=='none']
        return f'{sum(row[field]==value for row in selected)}/{len(selected)}'
    def metric(name,case,key):return interval(grouped[case,name][key])
    a=[[name,outcome(name,'A_expansion','reaction_status','responded'),metric(name,'A_expansion','reaction_seconds'),
        *[outcome(name,case,'reaction_status','responded') for case in ('A_static','A_translation','A_contraction')]] for name in names]
    b=[[name,metric(name,'B_food','food'),metric(name,'B_danger','food'),
        metric(name,'B_food','collision_substeps')+' / '+metric(name,'B_danger','collision_substeps'),
        metric(name,'B_food','motor_energy_demand')+' / '+metric(name,'B_danger','motor_energy_demand'),
        metric(name,'B_food','final_simulated_energy')+' / '+metric(name,'B_danger','final_simulated_energy')] for name in names]
    c=[[name,metric(name,'C_occlusion','food'),outcome(name,'C_relocation','recovery_status','recovered'),
        metric(name,'C_relocation','recovery_seconds'),metric(name,'C_unseen','food')] for name in names]
    examples=[]
    for case,key in PRESET_EXAMPLES:
        for row in summary['hierarchical_paired_differences']:
            if (row['scenario'],row['metric'])==(case,key):
                ci=row['hierarchical_paired_ci95'];mean=row['mean_difference_vs_connectome']
                examples.append([case,key,row['architecture'],row['training_seeds'],row['environment_units'],
                    'undefined' if mean is None else f'{mean:.4g} [{ci[0]:.4g}, {ci[1]:.4g}]'])
    costs=[[name,metric(name,'A_expansion','inference_mean_ms'),metric(name,'B_danger','inference_mean_ms'),metric(name,'C_relocation','inference_mean_ms')] for name in names]
    sections=['# Results',
      'n/N is the responding or recovered count over matched environment trials. Latency and recovery intervals use defined responses only; undefined (n=0) denotes no defined observations. Other intervals show their defined count n.',
      '## A. Expansion and controls',
      table(['Controller','Expansion response n/N','Conditional reaction, simulation seconds [95% CI]','Static n/N','Translation n/N','Contraction n/N'],a),
      '## B. Food near danger',
      table(['Controller','Safe food [95% CI]','Danger food [95% CI]','Collision body-substeps safe / danger','Motor demand safe / danger','Final simulated energy safe / danger'],b),
      '## C. Local observation and recovery',
      table(['Controller','Occluded food [95% CI]','Relocation recovered n/N','Conditional recovery, simulation seconds [95% CI]','New-layout food [95% CI]'],c),
      '## Prespecified paired examples',
      table(['Scenario','Metric','Architecture','Training seeds','Matched environments','Mean learned minus connectome [95% CI]'],examples),
      '## Inference cost',
      table(['Controller','Expansion wall-clock inference ms [95% CI]','Danger wall-clock inference ms [95% CI]','Relocation wall-clock inference ms [95% CI]'],costs)]
    return '\n\n'.join(section for section in sections if section!='')+'\n'


def generate(core_path,supplement_path,models_path,evidence_root,evidence_index,output_root,tag):
    if not re.fullmatch(r'v\d+\.\d+\.\d+',tag):raise ValueError('Versioned release tag required')
    core,supplement,models=load_experiments(core_path,supplement_path,models_path)
    refs=read(evidence_index);evidence=validate_evidence(evidence_root,refs,models,digest(Path(models_path)/'manifest.json'))
    recovery=validate_recovery(core_path,core,evidence_root,refs.get('recovery'))
    revision=validate_supplement_revision(supplement_path,supplement,evidence_root,refs.get('supplement_revision'))
    # Collect every output in memory before writing anything: a missing evidence
    # input cannot leave a plausible placeholder results page behind.
    rows=core['result']['records'];summary=core['summary'];sup_summary=supplement['summary']
    fields=('controller','ablation','scenario','environment_seed','neural_repeat','status','completed_windows',
            'simulation_seconds','wall_seconds',*METRICS,'reaction_status','recovery_status','baseline_escape_drive','preactivated',
            'sampled_peak_process_rss_bytes','rss_sampling','reflex_interventions','physical_energy_debit','physical_energy_debit_status')
    tables={'core-episodes.csv':[{key:row.get(key) for key in fields} for row in rows],
            'core-task-summary.csv':[flatten(row) for row in summary['single_controller']],
            'architecture-differences.csv':[flatten(row) for row in summary['hierarchical_paired_differences']],
            'ablations.csv':[flatten(row) for row in summary['paired_ablations']],
            'supplement-units.csv':[{key:value for key,value in row.items() if key not in {'events'}} for row in supplement['result']['records']],
            'supplement-stress-summary.csv':[flatten(row) for row in sup_summary['stress_individual']],
            'supplement-noise-differences.csv':[flatten(row) for row in sup_summary['stress_paired_noise_changes']],
            'shared-allocation.csv':sup_summary['shared_population']['allocation_table']}
    outputs={f'experiments/tables/{name}':csv_bytes(values) for name,values in tables.items()}
    index={'schema':'neuroterrarium.results-index.v1','status':'complete','release_tag':tag,
           'core':{'records':len(rows),'protocol_sha256':canonical(core['lock']),'files':core['hashes'],'source':core['lock']['source']},
           'supplement':{'records':len(supplement['result']['records']),'protocol_sha256':canonical(supplement['lock']),'files':supplement['hashes'],'source':supplement['lock']['source']},
           'model_manifest_sha256':digest(Path(models_path)/'manifest.json'),'supporting_evidence':refs['evidence'],
           'supplement_revision':None if revision is None else {'path':'experiments/supplement-revision-v1b.json',
               'parent_protocol_sha256':revision['parent_protocol_sha256'],'parent_status_counts':revision['parent_status_counts'],
               'newly_executed_units':revision['newly_executed_units'],'reused_parent_units':0,'pooled_parent_units':0,'evidence':revision['evidence']},
           'recovery':None if recovery is None else {'path':'experiments/recovery-v1.json',
               'parent_status':recovery['parent_status'],'reused_completed_count':recovery['reused_completed_count'],
               'rerun_units':[row['identifier'] for row in recovery['rerun_units']],
               'newly_executed_units':recovery['newly_executed_units'],'evidence':recovery['evidence'],
               'independent_final_units':recovery['independent_final_units'],
               'history':[{'child_directory':stage['child_directory'],'status':stage['child_status'],
                           'planned_new_units':stage['planned_new_units'],'recorded_new_units':stage['recorded_new_units'],
                           'completed_new_units':stage['completed_new_units'],'missing_new_units':stage['missing_new_units']} for stage in recovery['history']]},
           'tables':{path:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'rows':len(tables[Path(path).name])} for path,data in outputs.items()},
           'generator_sha256':digest(__file__),'prespecified_examples':[list(item) for item in PRESET_EXAMPLES],
           'availability':'Local candidate files verified. Public asset availability is checked after the tagged release.',
           'limits':['Three training seeds per architecture.','One neural-input repetition per environment.','Short fixed scenes and engineering interfaces.',
                     'Conditional reaction/recovery intervals exclude nonresponding units, with defined counts retained.',
                     'Process RSS includes shared graph and all loaded policies; no exclusive per-model memory comparison.']}
    if recovery is not None:
        raw=(json.dumps(recovery,indent=2,allow_nan=False)+'\n').encode()
        outputs['experiments/recovery-v1.json']=raw
        index['recovery'].update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    if revision is not None:
        raw=(json.dumps(revision,indent=2,allow_nan=False)+'\n').encode()
        outputs['experiments/supplement-revision-v1b.json']=raw
        index['supplement_revision'].update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    outputs['experiments/results-v1.json']=(json.dumps(index,indent=2,allow_nan=False)+'\n').encode()
    outputs['docs/results.md']=render(core,supplement,evidence,refs['evidence'],tag,recovery,revision).encode()
    root=Path(output_root)
    for name,data in outputs.items():
        path=root/name
        current=root
        for part in PurePosixPath(name).parts:
            current=current/part
            if current.is_symlink():raise ValueError('Report output path may not contain a symlink')
        if path.with_suffix(path.suffix+'.partial').exists():raise ValueError('Incomplete older report output must be preserved')
        if path.exists() and path.read_bytes()!=data:raise ValueError('Different existing report output must be preserved: '+name)
        if path.is_symlink():raise ValueError('Report output may not be a symlink')
    for name,data in outputs.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.partial');temporary.write_bytes(data);temporary.replace(path)
    return {'status':'complete','core_records':len(rows),'supplement_records':len(supplement['result']['records']),
            'outputs':{name:hashlib.sha256(data).hexdigest() for name,data in outputs.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('core','supplement','models','evidence-root','evidence-index','output-root'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    result=generate(args.core,args.supplement,args.models,args.evidence_root,args.evidence_index,args.output_root,args.tag)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
