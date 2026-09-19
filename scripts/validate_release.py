"""Verify local release evidence without running commands or changing artifacts."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib


REQUIRED = ('data', 'neural', 'cache_equivalence', 'training', 'core_tests', 'lint',
            'typecheck', 'build', 'runtime', 'evaluation', 'supplement', 'ui',
            'stability', 'installed_wheel', 'portable', 'public_review')
BEHAVIOR = ('brain_cache.py', 'controllers.py', 'data.py', 'interface.py', 'neural.py',
            'reference.py', 'registry.py', 'runtime.py', 'world.py')
NEURAL_SOURCES = {'data.py','interface.py','neural.py','reference.py','registry.py','world.py','scripts/neural_gate.py'}
BRAIN_CONFIGS = {'data-v783.json','brain-interface.json','interface-v783.json','brain-storage-v1.json'}
SOURCE_SCOPE = {
    'data': {'data.py'}, 'neural': NEURAL_SOURCES,
    'cache_equivalence': {'neural.py','runtime.py'},
    'training': {'training.py','controllers.py','world.py'},
    'core_tests': {*BEHAVIOR,'training.py','evaluation.py','supplemental.py','service.py','recording.py','replay.py','validation.py',
                   'scripts/report_results.py','scripts/validate_release.py','tests/test_report_results.py','tests/test_validate_release.py'},
    'lint': {*BEHAVIOR,'training.py','evaluation.py','supplemental.py','service.py'},
    'typecheck': {'web/src/main.ts'},
    'build': {'web/src/main.ts','web/src/style.css','scripts/stage_web.py'},
    'runtime': {*BEHAVIOR,'validation.py'},
    'evaluation': {*BEHAVIOR,'evaluation.py'},
    'supplement': {*BEHAVIOR,'evaluation.py','supplemental.py'},
    'ui': {*BEHAVIOR,'service.py','recording.py','replay.py','web/src/main.ts','web/src/style.css'},
    'stability': {*BEHAVIOR,'service.py','recording.py','replay.py','scripts/stability.py'},
    'installed_wheel': {*BEHAVIOR,'validation.py','cli.py','scripts/install_check.py','scripts/stage_web.py'},
    'portable': {*BEHAVIOR,'validation.py','service.py','cli.py','scripts/portable_check.py','scripts/build_package.py','scripts/install_check.py'},
    'public_review': {'scripts/validate_release.py'},
}
CONFIG_SCOPE = {
    'data': {'data-v783.json'},
    'neural': BRAIN_CONFIGS-{'brain-storage-v1.json'},
    'cache_equivalence': {'data-v783.json','brain-storage-v1.json'},
    'training': {'train-full.json'},
    'core_tests': BRAIN_CONFIGS|{'train-full.json','evaluate-v1.json','evaluate-supplement-v1.json','evaluate-supplement-v1b.json'},
    'lint': set(), 'typecheck': set(), 'build': set(),
    'runtime': BRAIN_CONFIGS, 'evaluation': BRAIN_CONFIGS|{'evaluate-v1.json'},
    'supplement': BRAIN_CONFIGS|{'evaluate-supplement-v1.json','evaluate-supplement-v1b.json'},
    'ui': BRAIN_CONFIGS, 'stability': BRAIN_CONFIGS,
    'installed_wheel': BRAIN_CONFIGS, 'portable': BRAIN_CONFIGS,
    'public_review': {'data-v783.json'},
}
# These focused commands predate the model set and never load learned policies.
WITHOUT_MODELS = {'data','neural'}
UI_CHECKS = {'environment_editing', 'actual_inspector', 'guess_and_reveal',
             'time_save_load_replay', 'fork_comparison', 'sensory_circuit_interventions'}
STABILITY_OPERATIONS = {'resume','pause','step','food_add','stimulus_add','stimulus_remove',
                        'channels','restore_interventions','neural_disconnect','readout_clamp',
                        'sham','noise','snapshot_save','snapshot_restore'}
SHA = re.compile(r'[0-9a-f]{64}\Z')


class GateError(ValueError):
    """Evidence is absent, inconsistent, out of scope or incomplete."""


def require(condition, message):
    if not condition:
        raise GateError(message)


def finite(value, minimum=0):
    return type(value) in (int,float) and math.isfinite(value) and value >= minimum


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read_json(path):
    require(path.stat().st_size <= 128*2**20, 'JSON evidence exceeds size limit')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(GateError('Nonfinite JSON number')))


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        require(self.root.is_dir(), 'Artifact root must be a directory')
        self.verified = {}
        self.stats = {}

    def path(self, name, *, must_exist=True):
        require(isinstance(name,str) and name and '\\' not in name and ':' not in name,
                'Evidence path must be a relative POSIX path')
        parts = name.split('/')
        require(not PurePosixPath(name).is_absolute() and all(part not in {'','.','..'} for part in parts),
                'Evidence path may not be absolute or traverse directories')
        path = self.root.joinpath(*parts)
        # Refuse symlinks, including links resolving inside the root: one path
        # has one independently hashable artifact, without a mutable alias.
        current = self.root
        for part in parts:
            current = current/part
            require(not current.is_symlink(), 'Symlink evidence paths are not accepted')
        require(path.resolve(strict=must_exist).is_relative_to(self.root) and (not must_exist or path.is_file()),
                'Evidence path is not a regular file within the artifact root')
        return path

    def file(self, ref):
        require(isinstance(ref,dict) and {'path','sha256'} <= set(ref), 'A file reference needs path and sha256')
        require(isinstance(ref['sha256'],str) and SHA.fullmatch(ref['sha256']), 'Invalid SHA-256')
        path = self.path(ref['path'])
        stat=path.stat();identity=(stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if self.stats.get(ref['path'])!=identity:
            self.verified[ref['path']] = sha256(path)
            self.stats[ref['path']] = identity
        require(self.verified[ref['path']] == ref['sha256'], 'Artifact checksum mismatch: '+ref['path'])
        if 'bytes' in ref:
            require(type(ref['bytes']) is int and path.stat().st_size == ref['bytes'], 'Artifact byte count mismatch')
        return path

    def json(self, ref):
        return read_json(self.file(ref))

    def sibling(self, parent, name, digest):
        require(isinstance(name,str) and PurePosixPath(name).name == name and name not in {'','.','..'},
                'Nested evidence filename must be a simple name')
        ref = {'path': str(PurePosixPath(parent['path']).parent/name), 'sha256': digest}
        return self.file(ref)


def verified_junit(path):
    require(path.stat().st_size <= 64*2**20, 'JUnit exceeds size limit')
    raw = path.read_bytes()
    require(b'<!DOCTYPE' not in raw.upper() and b'<!ENTITY' not in raw.upper(), 'DTD is not allowed in JUnit')
    root = ET.fromstring(raw)
    require(root.tag in {'testsuite','testsuites'}, 'Unexpected JUnit root')
    cases = list(root.iter('testcase'))
    require(len(cases)>0, 'JUnit contains no executed tests')
    require(not any(list(root.iter(tag)) for tag in ('failure','error','skipped')),
            'JUnit contains failed, errored or skipped tests')
    for suite in root.iter('testsuite'):
        actual = len(list(suite.iter('testcase')))
        require(actual>0 and int(suite.attrib.get('tests','-1'))==actual, 'JUnit declared test count mismatch')
        require(all(int(suite.attrib.get(name,'0'))==0 for name in ('failures','errors','skipped')),
                'JUnit declares a failure, error or skip')
    if 'tests' in root.attrib: require(int(root.attrib['tests'])==len(cases),'JUnit root test count mismatch')
    return len(cases)


def verify_steps(evidence, summary_ref, steps, required):
    require(isinstance(steps,list) and len(steps)>0, 'Missing executed command steps')
    names = []
    for step in steps:
        require(type(step.get('exit_code')) is int and step['exit_code']==0, 'A required command did not exit successfully')
        require(step.get('state','completed')=='completed', 'A required command is incomplete')
        require(isinstance(step.get('command'),list) and step['command'] and all(isinstance(x,str) and x for x in step['command']), 'Missing command arguments')
        evidence.sibling(summary_ref,step['log'],step['log_sha256'])
        names.append(step['name'])
    require(len(names)==len(set(names)) and set(required)<=set(names), 'Required command steps missing or duplicated')


def verify_graph(graph, manifest):
    require(graph.get('profile')=='shiu-v783-full' and graph.get('connectome_version')=='783', 'Wrong graph profile')
    for name,value in manifest['expected'].items():
        if name in graph:
            require(type(graph[name]) is int and graph[name]==value, 'Graph count mismatch: '+name)
    for name in ('neurons','directed_pairs','synaptic_contacts'):
        require(type(graph.get(name)) is int and graph[name]>0 and graph[name]==manifest['expected'][name], 'Missing graph count: '+name)
    require(graph['source_sha256']=={k:v['sha256'] for k,v in manifest['sources'].items()}, 'Graph source identity mismatch')


def verify_runtime(summary, context):
    require(summary.get('schema')=='neuroterrarium.runtime-validation.v1' and summary.get('status')=='passed', 'Runtime verification is incomplete')
    require(summary.get('behavior_sha256')==context['behavior'] and summary.get('model_manifest_sha256')==context['models_hash'], 'Runtime tested identity mismatch')
    require(summary.get('validation_source_sha256')==context['source']['validation.py'], 'Native validation implementation differs')
    verify_graph(summary['graph'],context['data'])
    checks = {item['name']:item for item in summary['checks']}
    required = {'verified_full_profile_and_nine_models','neural_firing_drives_physical_feeding',
                'persistent_nonresting_state','snapshot_recomputation_exact','positive_control_output_disconnection',
                'same_snapshot_sham_branch_exact','clamped_readout_has_no_new_motor_commands',
                'stimulus_removal_changes_delivered_input','shared_sensory_mask_reaches_all_controllers',
                'ten_actual_inspector_inputs','journal_controller_recomputation'}
    require(len(checks)==len(summary['checks']) and required<=checks.keys(), 'Missing runtime causal check')
    require(all(c.get('status')=='passed' for c in checks.values()), 'Runtime causal check failed')
    loaded=checks['verified_full_profile_and_nine_models']
    require(loaded['neurons']==context['data']['expected']['neurons'] and set(loaded['controllers'])==context['controller_names'], 'Runtime uses the wrong bodies or graph')
    feeding=checks['neural_firing_drives_physical_feeding']; persistent=checks['persistent_nonresting_state']
    require(finite(feeding['consumed_food'],1e-15) and finite(feeding['mn9_window_hz_sum'],1e-15), 'Runtime feeding has no measured activity or intake')
    require(finite(persistent['neural_steps'],1) and finite(persistent['queued_events'],1), 'Runtime state was not active and persistent')
    cut=checks['positive_control_output_disconnection']
    require(cut['normal_mn9_window_hz_sum']>cut['cut_mn9_window_hz_sum']>=0, 'Neural positive control has no disconnection effect')
    removed=checks['stimulus_removal_changes_delivered_input']
    require(removed['before_hz']['lplc2']>0 and removed['after_hz']['lplc2']==removed['after_hz']['lc4']==0, 'Stimulus input did not disappear')
    require(checks['journal_controller_recomputation']['action_windows']>0, 'No controller recomputation was checked')


def verify_neural(evidence, ref, summary, context):
    import numpy as np
    require(summary.get('schema')=='neural-gate-v1' and summary.get('status')=='passed', 'Full neural gate incomplete')
    verify_graph(summary['graph'],context['data'])
    require(finite(summary.get('duration_seconds'),.5), 'Neural gate duration is too short')
    checks=summary['checks']
    require({'gustation','visual_projection','closed_feeding','closed_expansion','full_graph_reference'}<=checks.keys() and all(v is True for v in checks.values()), 'A neural gate is missing or failed')
    tests={(r['case'],r['condition']):r for r in summary['tests']}
    require(len(tests)==len(summary['tests']), 'Duplicate neural positive control')
    for case in ('gustation','visual_projection'):
        normal,cut,clamp=(tests[case,c] for c in ('normal','disconnect_output','readout_clamp'))
        require(normal['output_spikes']>0 and cut['output_spikes']==0 and clamp['output_spikes']==normal['output_spikes'], 'Neural firing/clamp/disconnection positive control invalid')
        require(clamp['action']==[0,0,0,0], 'Readout clamp did not zero the actions')
        for condition in ('normal','disconnect_output','readout_clamp'):
            row=tests[case,condition]
            evidence.sibling(ref,f'{case}-{condition}.npz',row['trajectory_sha256'])
        references=[r for r in summary['reference'] if r['case']==case]
        require(len(references)==1, 'Missing or duplicate full Brian reference case')
        reference=references[0]
        require(reference['spike_steps_equal'] is True and reference['spike_neurons_equal'] is True, 'Brian spike correspondence failed')
        require(all(finite(reference[k]) and reference[k]<=1e-9 for k in ('max_voltage_error_mV','max_conductance_error_mV')), 'Brian state error exceeds frozen tolerance')
        a=evidence.sibling(ref,f'{case}-normal.npz',normal['trajectory_sha256'])
        b=evidence.sibling(ref,f'reference-{case}.npz',reference['trajectory_sha256'])
        with np.load(a,allow_pickle=False) as actual,np.load(b,allow_pickle=False) as expected:
            require(len(actual['spike_steps'])>0, 'Silent graph is not an active comparison')
            for key in ('spike_steps','spike_neurons','record_indices'):
                require(np.array_equal(actual[key],expected[key]), 'Raw Brian spike or index mismatch')
            for key in ('v_mV','g_mV'):
                require(actual[key].size>0 and np.isfinite(actual[key]).all() and np.isfinite(expected[key]).all()
                        and actual[key].shape==expected[key].shape and np.allclose(actual[key],expected[key],rtol=0,atol=1e-9), 'Raw Brian trajectory mismatch')
    closed={(r['case'],r['condition']):r for r in summary['closed_loop']}
    for (case,condition),row in closed.items():
        evidence.sibling(ref,f'closed-{case}-{condition}.jsonl',row['trajectory_sha256'])
    for case in ('feeding','expansion'):
        metric='food' if case=='feeding' else 'distance'
        require(closed[case,'normal'][metric]>closed[case,'disconnect_output'][metric]>=0
                and closed[case,'readout_clamp'][metric]==0, 'Embodied neural positive control did not separate conditions')


def verify_cache(evidence, summary, context):
    import numpy as np
    require(summary.get('schema')=='neuroterrarium.cache-equivalence.v1', 'Unsupported cache-equivalence evidence')
    before,after=(evidence.json(summary[key]) for key in ('before','after'))
    expected=context['data']['expected']
    require(before['neurons']==after['neurons']==expected['neurons'] and before['connections']==after['connections']==expected['directed_pairs'], 'Cache comparison is not the full graph')
    require(before['graph_digest']==after['graph_digest'] and before['input_recording_sha256']==after['input_recording_sha256'], 'Cache graph or input tape changed')
    require(after['neural_sha256']==context['source']['neural.py'] and after['runtime_sha256']==context['source']['runtime.py'], 'Cache comparison does not target candidate sources')
    require(before['windows']==after['windows'] and before['windows']>=20 and before['all_neuron_spikes']>0, 'Cache numerical run has insufficient active windows')
    pairs=summary['pairs']; require(len(pairs)==before['windows']+1, 'Cache comparison omits a window or final state')
    seen=set(); spikes=0
    for pair in pairs:
        a,b=evidence.file(pair['before']),evidence.file(pair['after'])
        require(a.name==b.name and a.name not in seen,'Duplicate/mismatched cache comparison file');seen.add(a.name)
        if a.suffix=='.npz':
            with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
                require(set(x.files)==set(y.files) and {'spike_steps','spike_neurons','spike_counts','v_mV','g_mV'}<=set(x.files), 'Cache trace fields missing')
                require(all(np.array_equal(x[k],y[k]) for k in x.files), 'Cache numerical arrays differ')
                require(x['spike_counts'].shape==(expected['neurons'],), 'Cache spike counts omit graph nodes')
                spikes+=len(x['spike_steps'])
        else:
            require(a.name=='final-state.json' and read_json(a)==read_json(b), 'Cache final executable state differs')
    require(spikes==before['all_neuron_spikes']==after['all_neuron_spikes'], 'Cache actual spike count mismatch')


def without_restore(path):
    text=path.read_text();lines=text.splitlines(keepends=True)
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='World')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='restore')
    start=min([method.lineno,*[d.lineno for d in method.decorator_list]])-1
    return ''.join(lines[:start]+lines[method.end_lineno:])


def verify_training(evidence, summary, context, item):
    require(summary.get('schema')=='neuroterrarium.training-results.v1' and summary.get('status')=='complete', 'Formal training evidence incomplete')
    require(summary['model_set_manifest_sha256']==context['models_hash'] and summary['protocol_sha256']==context['models']['protocol_sha256'], 'Training/model protocol mismatch')
    require(summary['source_sha256']==context['training_source'], 'Trained source identity mismatch')
    rows=summary['models'];models={m['id']:m for m in context['models']['models']}
    require(len(rows)==9 and {r['id'] for r in rows}==set(models), 'Nine training identities are required')
    initial=set();final=set()
    for row in rows:
        spec=models[row['id']]
        require(row['transitions']==spec['transitions'] and row['updates']>0 and row['vector_steps']*4==row['transitions'], 'Formal training budget or transition units mismatch')
        require(row['weights_sha256']==spec['policy_sha256'] and row['initial_weights_sha256']!=row['weights_sha256'], 'Weights were not independently updated')
        require(SHA.fullmatch(row['initial_weights_sha256']), 'Missing initial weight identity')
        initial.add(row['initial_weights_sha256']);final.add(row['weights_sha256'])
        require(item['initial_weights'][row['id']]['sha256']==row['initial_weights_sha256'],'Initial checkpoint evidence hash mismatch')
        evidence.file(item['initial_weights'][row['id']])
        logref=item['training_logs'][row['id']]
        require(logref['sha256']==row['training_log_sha256'],'Training log hash mismatch')
        log=evidence.file(logref);record_count=0;last_transitions=0
        with log.open() as handle:
            for line in handle:
                require(len(line)<2**20,'Training log line exceeds size limit')
                entry=json.loads(line);record_count+=1
                require(entry['updates']==record_count and entry['transitions']>last_transitions,'Training update log order mismatch')
                last_transitions=entry['transitions']
        require(record_count==row['updates'] and last_transitions==row['transitions'],'Training logs do not cover full budget')
        for label in ('dev_before','dev_after'):
            dev=row[label]
            require(dev['split']=='dev' and len(dev['episodes'])>=8, 'Missing real paired development episodes')
            require(all(finite(ep['episode_steps'],1) and finite(ep['food']) for ep in dev['episodes']), 'Incomplete development evidence')
        require([r['seed'] for r in row['dev_before']['episodes']]==[r['seed'] for r in row['dev_after']['episodes']], 'Development comparison not paired')
    require(len(initial)==len(final)==9,'Initial or final weights are duplicated')
    if context['training_source']['world.py']!=context['source']['world.py']:
        proof=evidence.json(item['compatibility'])
        require(proof['schema']=='neuroterrarium.world-compatibility-results.v1' and proof['status']=='passed'
                and proof['original_world_sha256']==context['training_source']['world.py']
                and proof['compatible_world_sha256']==context['source']['world.py']
                and proof['model_set_manifest_sha256']==context['models_hash'], 'Training compatibility does not cover candidate World')
        require(without_restore(context['training_source_paths']['world.py'])==without_restore(context['source_paths']['world.py']), 'World changed outside restore after training')
        episodes=proof['episodes'];require(len(episodes)==72 and proof['paired_development_episodes']==72, 'Incomplete compatibility episode set')
        require(len({(r['model'],r['development_seed']) for r in episodes})==72, 'Duplicate compatibility units')
        for row in episodes:
            require(row['exact_equal'] is True and row['action_windows']==250
                    and row['original_numeric_sha256']==row['compatible_numeric_sha256']
                    and row['original_world_tape_sha256']==row['compatible_world_tape_sha256'], 'Training compatibility trace mismatch')


def verify_supplement_config_revision(original, revised):
    """Only the documented cumulative wall allowance and its prose may change."""
    require(type(original.get('wall_budget_seconds')) is int and original['wall_budget_seconds']==900
            and type(revised.get('wall_budget_seconds')) is int and revised['wall_budget_seconds']==1800,
            'Supplement resource versions must use 900 and 1800 seconds')
    before='Nine hundred cumulative wall seconds';after='Eighteen hundred cumulative wall seconds'
    require(isinstance(original.get('resources'),str) and original['resources'].count(before)==1,
            'Original supplement resource description differs')
    expected={**original,'wall_budget_seconds':1800,'resources':original['resources'].replace(before,after)}
    require(revised==expected,'Supplement revision changed scientific fields or an undeclared resource setting')
    return [{'path':'/'+key,'before':original[key],'after':revised[key]} for key in ('wall_budget_seconds','resources')]


def verify_supplement_validator_revision(original, revised):
    """Rebuild the sole permitted validation edit, then compare whole-module ASTs."""
    before=ast.parse(original.read_text());after=ast.parse(revised.read_text())
    functions=[node for node in before.body if isinstance(node,ast.FunctionDef) and node.name=='validate_protocol']
    require(len(functions)==1,'Original supplement protocol validator is missing')
    function=functions[0]
    assignments=[node for node in function.body if isinstance(node,ast.Assign)
                 and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='expected']
    require(len(assignments)==1 and isinstance(assignments[0].value,ast.Dict),'Original supplement settings differ')
    values=assignments[0].value
    positions=[i for i,key in enumerate(values.keys) if isinstance(key,ast.Constant) and key.value=='wall_budget_seconds']
    require(len(positions)==1 and isinstance(values.values[positions[0]],ast.Constant)
            and values.values[positions[0]].value==900,'Original supplement wall guard differs')
    index=positions[0];del values.keys[index];del values.values[index]
    function.body.extend(ast.parse("wall_budget = config.get('wall_budget_seconds')\n"
        "if type(wall_budget) is not int or wall_budget not in (900, 1800):\n"
        "    raise ValueError('Supplement wall budget must match a frozen resource version')\n"
        "allowance = 'Nine hundred' if wall_budget == 900 else 'Eighteen hundred'\n"
        "description = (f'{allowance} cumulative wall seconds including setup across resumed invocations, ' "
        "'checked at episode boundaries and action-window timeout; small recording and teardown ' "
        "'overhead can exceed the boundary. Preserve partial records and list not-started units ' "
        "'on cancellation, budget exhaustion or low resources.')\n"
        "if config.get('resources') != ('One immutable full-graph structure and at most one active connectome state. ' "
        "'One CPU thread for learned inference. ' + description):\n"
        "    raise ValueError('Supplement resource description differs from the frozen resource version')\n").body)
    require(ast.dump(before)==ast.dump(after),'Supplement source revision changed more than the explicit wall-budget guard')


def verify_supplement_resource_limits(original, revised):
    """The lock mirrors its configuration's wall allowance; protections stay fixed."""
    before=original.get('resource_limits');after=revised.get('resource_limits')
    fields={'max_total_wall_seconds','max_process_rss_bytes','min_available_memory_bytes','min_free_disk_bytes'}
    require(isinstance(before,dict) and isinstance(after,dict) and set(before)==set(after)==fields,
            'Supplement resource-limit fields are missing or unexpected')
    require(all(type(value) is int and value>0 for values in (before,after) for value in values.values()),
            'Supplement resource limits must be positive integers')
    for document,limits,expected in ((original,before,900),(revised,after,1800)):
        config=document['config'];allowance=config.get('wall_budget_seconds')
        require(type(allowance) is int and allowance==expected==limits['max_total_wall_seconds'],
                'Supplement wall-budget mirror differs from its configuration')
        require(limits['min_available_memory_bytes']==config.get('memory_reserve_bytes')
                and limits['min_free_disk_bytes']==config.get('disk_reserve_bytes'),
                'Supplement resource-protection mirrors differ from their configuration')
    require(after=={**before,'max_total_wall_seconds':1800},
            'Supplement revision changed a resource protection instead of only its wall allowance')


def verify_supplement_revision(evidence, summary, lock, context, spec):
    """Preserve the unsuccessful 900-second version beside a fresh 630-unit run."""
    if lock['config'].get('wall_budget_seconds')==900:
        require(spec is None,'Original supplement cannot claim a resource-revision lineage')
        return None
    required={'manifest','parent_directory','parent_source','parent_config','commands','release_asset'}
    require(isinstance(spec,dict) and set(spec)==required,'Supplement v1b requires complete explicit resource-revision evidence')
    asset=spec['release_asset']
    require(isinstance(asset,str) and PurePosixPath(asset).name==asset and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]+',asset),
            'Supplement revision requires a versioned evidence asset')
    lineage=evidence.json(spec['manifest'])
    require(lineage.get('schema')=='neuroterrarium.supplement-resource-revision.v1'
            and lineage.get('revision_id')=='supplement-v1b','Unsupported supplement resource revision')
    parent=lineage['parent'];revised=lineage['revised'];directory=spec['parent_directory']
    require(parent['directory']==directory and revised['directory']!=directory,'Supplement predecessor identity differs')
    parent_path=evidence.path(directory+'/results.json').parent
    files=parent['evidence_sha256']
    require({'results.json','protocol-lock.json','budget.json','graph-summary.json'}<=files.keys()
            and {str(p.relative_to(parent_path)) for p in parent_path.rglob('*') if p.is_file()}==set(files),
            'Supplement predecessor file inventory omits or invents records')
    for name,value in files.items():
        evidence.file({'path':directory+'/'+name,'sha256':value})
    old=read_json(parent_path/'results.json');old_lock=read_json(parent_path/'protocol-lock.json');budget=read_json(parent_path/'budget.json')
    old_protocol=canonical(old_lock);new_protocol=canonical(lock)
    require(old.get('schema')=='neuroterrarium.supplement-results.v1' and old.get('status')=='incomplete'
            and old.get('reason')=='wall_budget' and parent.get('status')==old['status'] and parent.get('reason')==old['reason']
            and type(parent.get('exit_code')) is int and parent['exit_code']==1,'Supplement predecessor was not the recorded budget-limited failure')
    require(parent.get('protocol_sha256')==old.get('protocol_sha256')==old_protocol
            and summary.get('protocol_sha256')==new_protocol and summary.get('status')=='completed',
            'Supplement revision protocol or completion mismatch')
    original=evidence.json(spec['parent_config'])
    require(original==old_lock['config']==context['config_documents']['evaluate-supplement-v1.json']
            and spec['parent_config']['sha256']==parent['config_sha256'],'Supplement original configuration differs')
    differences=verify_supplement_config_revision(original,lock['config'])
    require(lock['config']==context['config_documents']['evaluate-supplement-v1b.json']
            and revised['config_sha256']==context['configs']['evaluate-supplement-v1b.json'],
            'Supplement revised configuration differs')
    require(lineage.get('configuration_diff')==differences,'Supplement declared configuration diff differs')
    verify_supplement_resource_limits(old_lock,lock)
    require({k:v for k,v in old_lock.items() if k not in {'config','source','resource_limits'}}==
            {k:v for k,v in lock.items() if k not in {'config','source','resource_limits'}},'Supplement revision changed models, interfaces, graph or plan')
    require(set(old_lock['source'])==set(lock['source']) and all(old_lock['source'][key]==value for key,value in lock['source'].items() if key!='supplemental.py'),
            'Supplement revision changed scientific dependencies')
    archived=evidence.file(spec['parent_source'])
    require(spec['parent_source']['sha256']==old_lock['source']['supplemental.py'], 'Archived supplement source differs from original lock')
    verify_supplement_validator_revision(archived,context['source_paths']['supplemental.py'])
    for key,value in old_lock['source'].items():
        require(parent['source_sha256'].get('src/neuroterrarium/'+key)==value,'Supplement original source lineage differs')
    for key,value in lock['source'].items():
        require(revised['source_sha256'].get('src/neuroterrarium/'+key)==value==context['source'][key],
                'Supplement revised source lineage differs')
    diff=lineage['validator_diff']
    require(diff['before_sha256']==old_lock['source']['supplemental.py'] and diff['after_sha256']==lock['source']['supplemental.py'],
            'Supplement source revision hashes differ')
    require(budget.get('protocol_sha256')==old_protocol and budget.get('limits')==old_lock['resource_limits']
            and finite(old.get('wall_seconds'),900) and budget.get('elapsed_seconds')==old['wall_seconds']==parent.get('wall_seconds'),
            'Supplement predecessor wall-budget evidence differs')
    require(revised.get('expected_total_units')==630 and revised.get('expected_stress_units')==600
            and revised.get('expected_shared_world_units')==30 and revised.get('reuse_parent_records') is False,
            'Supplement revised run must execute all 630 units without reusing parent records')
    keys=('family','controller','condition','environment_seed','allocation','windows')
    def unit(row):return tuple(tuple(row[key]) if key=='allocation' else row[key] for key in keys)
    plan=[unit(row) for row in lock['plan']]
    require(len(plan)==630 and len(set(plan))==630 and [unit(row) for row in old['records']]==plan
            and [unit(row) for row in summary['records']]==plan,'Supplement version omits or duplicates a planned unit')
    require(all(row.get('status')=='completed' and row.get('protocol_sha256')==new_protocol for row in summary['records']),
            'Supplement revised records are incomplete or reused from the old protocol')
    counts=dict(Counter(row['status'] for row in old['records']))
    require(counts==parent.get('status_counts') and counts.get('completed',0)<630
            and set(counts)<={'completed','interrupted','not_started'},'Supplement predecessor status counts differ')
    exclusions=[]
    for row in old['records']:
        require(row.get('protocol_sha256')==old_protocol,'Supplement predecessor row version differs')
        name=f"{row['family']}--{row['controller']}--{row['condition']}--{row['environment_seed']}"
        if row['status']=='not_started':
            require('trajectory' not in row and not (parent_path/'raw'/(name+'.json')).exists(), 'Not-started predecessor has hidden recorded data')
        else:
            require(read_json(parent_path/'raw'/(name+'.json'))==row,'Supplement predecessor index differs from raw record')
            trajectory=row.get('trajectory');require(isinstance(trajectory,str) and PurePosixPath(trajectory).name==trajectory,
                                                     'Unsafe predecessor trajectory reference')
            evidence.file({'path':directory+'/raw/'+trajectory,'sha256':row['trajectory_sha256']})
        if row['status']!='completed':
            exclusions.append({**{key:row[key] for key in keys},'identifier':name,'status':row['status'],
                               'completed_windows':row.get('completed_windows',0),'reason':row.get('reason'), 'failure':row.get('failure')})
    commands=spec['commands'];require(isinstance(commands,list) and len(commands)==2,'Both supplement execution commands are required')
    phases=set()
    for command_ref in commands:
        command=evidence.json(command_ref);phase=command.get('phase');argv=command.get('argv',command.get('command'))
        require(phase in {'parent','execution'} and phase not in phases,'Supplement command phase is duplicated or missing');phases.add(phase)
        require(type(command.get('exit_code')) is int and command['exit_code']==(1 if phase=='parent' else 0)
                and command.get('status') in ({'completed','incomplete'} if phase=='parent' else {'completed','passed'})
                and command.get('protocol_sha256')==(old_protocol if phase=='parent' else new_protocol),
                'Supplement command lacks its actual versioned completion')
        require(isinstance(argv,(str,list)) and argv and (isinstance(argv,str) or all(isinstance(arg,str) and arg for arg in argv)),
                'Supplement command arguments are missing')
        require(not re.search(r'/(?:Users|home|private/var/folders)/|[A-Za-z]:\\Users\\',json.dumps(command)),
                'Supplement command contains private paths')
    return {'schema':'neuroterrarium.supplement-revision-report.v1','revision_id':'supplement-v1b',
            'parent_directory':directory,'parent_protocol_sha256':old_protocol,'protocol_sha256':new_protocol,
            'parent_status':old['status'],'parent_reason':old['reason'],'parent_exit_code':1,'parent_wall_seconds':old['wall_seconds'],
            'parent_status_counts':counts,'parent_files_sha256':files,'incomplete_parent_units':exclusions,
            'configuration_diff':differences,'source_diff':{'before_sha256':old_lock['source']['supplemental.py'],
                'after_sha256':lock['source']['supplemental.py'],'scope':'Only the accepted cumulative wall-budget guard; whole-module AST verified.'},
            'newly_executed_units':630,'reused_parent_units':0,'pooled_parent_units':0,'evidence':spec}


def verify_evaluation_recovery(evidence, ref, summary, lock, context, item):
    """The final gate verifies all preserved recovery stages, not only final rows."""
    directory=evidence.file(ref).parent
    if not (directory/'recovery-manifest.json').exists() and item.get('recovery') is None:
        return None
    script=Path(__file__).with_name('report_results.py')
    require(context['source'].get('scripts/report_results.py')==sha256(script),
            'Recovery verifier source differs from the tested candidate')
    module_spec=importlib.util.spec_from_file_location('release_recovery_report',script)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return module.validate_recovery(directory,{'result':summary,'lock':lock},evidence.root,item.get('recovery'))


def verify_evaluation(evidence, ref, summary, context, item, *, supplement=False):
    lock=evidence.json(item['protocol_lock']); config=lock['config']
    configuration=('evaluate-supplement-v1b.json' if config.get('wall_budget_seconds')==1800 else 'evaluate-supplement-v1.json') if supplement else 'evaluate-v1.json'
    require(config==context['config_documents'][configuration],'Experiment protocol differs from frozen candidate configuration')
    require(summary['status']=='completed' and summary['protocol_sha256']==canonical(lock), 'Experiment is incomplete or lock mismatched')
    require(lock['model_manifest_sha256']==canonical(context['models']), 'Experiment model-set identity mismatch')
    require(all(context['source'].get(name)==digest for name,digest in lock['source'].items()), 'Experiment source differs from candidate')
    if supplement:
        revision=verify_supplement_revision(evidence,summary,lock,context,item.get('resource_revision'))
        if revision is not None:
            native=evidence.json(item['resource_revision']['manifest'])
            require(native['revised']['directory']==str(PurePosixPath(ref['path']).parent),'Supplement revision names a different execution directory')
        expected=[tuple(spec[k] if k!='allocation' else tuple(spec[k]) for k in ('family','controller','condition','environment_seed','allocation','windows')) for spec in lock['plan']]
        rows=summary['records']
        actual=[tuple(row[k] if k!='allocation' else tuple(row[k]) for k in ('family','controller','condition','environment_seed','allocation','windows')) for row in rows]
        require(actual==expected and len(rows)==630 and len(set(expected))==630,'Supplement does not contain every planned unit')
        require(len(config['stress_seeds'])==len(config['shared_seeds'])==30,'Supplement needs thirty environments per configuration')
    else:
        require(config['status']=='frozen' and len(config['environment_seeds'])>=30 and len(set(config['environment_seeds']))==len(config['environment_seeds']), 'Formal evaluation seed set incomplete')
        names=['connectome',*sorted(context['model_ids']),'random','rule','untrained']
        plan=[(name,'none',case,seed) for name in names for case in config['scenarios'] for seed in config['environment_seeds']]
        for ablation in config['ablations']:
            members=['connectome'] if ablation['controller_kind']=='connectome' else [n for n in names if n.startswith(ablation['controller_kind']+'-')]
            plan.extend((name,ablation['intervention'],case,seed) for name in members for case in ablation['scenarios'] for seed in config['environment_seeds'])
        rows=summary['records'];actual=[(r['controller'],r['ablation'],r['scenario'],r['environment_seed']) for r in rows]
        require(len(plan)==len(set(plan)) and set(actual)==set(plan) and len(actual)==len(plan), 'Formal evaluation omits or duplicates planned units')
        require({'A_expansion','A_static','A_translation','A_contraction','B_food','B_danger','C_occlusion','C_relocation','C_unseen'}<=set(config['scenarios']), 'ABC evaluation families missing')
    for row in rows:
        require(row['status']=='completed' and row['protocol_sha256']==summary['protocol_sha256'], 'Experiment contains failed, skipped or unfinished trials')
        windows=row['windows'] if supplement else (100 if row['scenario'].startswith('A_') else 250 if row['scenario'].startswith('B_') else 300)
        require(row['completed_windows']==windows and math.isclose(row['simulation_seconds'],windows*.02,abs_tol=1e-12), 'Trial duration does not match protocol')
        name=row['trajectory'];require(PurePosixPath(name).name==name,'Invalid raw trajectory path')
        raw={'path':str(PurePosixPath(ref['path']).parent/'raw'/name),'sha256':row['trajectory_sha256']}
        evidence.file(raw)
    if not supplement:verify_evaluation_recovery(evidence,ref,summary,lock,context,item)


def image_dimensions(path):
    """Read bounded PNG chunks or JPEG frame headers; never trust an extension."""
    size=path.stat().st_size
    require(0<size<=32*2**20, 'Screenshot size is empty or exceeds the limit')
    with path.open('rb') as handle:
        signature=handle.read(8)
        if signature==b'\x89PNG\r\n\x1a\n':
            dimensions=None;idat=False;chunks=0
            while handle.tell()<size:
                header=handle.read(8)
                require(len(header)==8, 'Truncated PNG chunk header')
                length,kind=struct.unpack('>I4s',header)
                require(length<=size-handle.tell()-4, 'Truncated PNG chunk')
                payload=handle.read(length);crc=handle.read(4)
                require(len(crc)==4 and zlib.crc32(kind+payload)&0xffffffff==struct.unpack('>I',crc)[0], 'Invalid PNG chunk checksum')
                require(chunks>0 or kind==b'IHDR', 'PNG does not begin with IHDR');chunks+=1
                if kind==b'IHDR':
                    require(dimensions is None and length==13, 'Invalid PNG image header')
                    width,height,depth,color,compression,filtering,interlace=struct.unpack('>IIBBBBB',payload)
                    allowed={0:{1,2,4,8,16},2:{8,16},3:{1,2,4,8},4:{8,16},6:{8,16}}
                    require(width>0 and height>0 and depth in allowed.get(color,set()) and compression==filtering==0 and interlace in (0,1), 'Invalid PNG dimensions or format')
                    dimensions=(width,height)
                elif kind==b'IDAT':idat=idat or length>0
                elif kind==b'IEND':
                    require(length==0 and dimensions is not None and idat and handle.tell()==size, 'Invalid or incomplete PNG ending')
                    return dimensions
            raise GateError('Truncated PNG without IEND')
        require(signature[:2]==b'\xff\xd8', 'UI screenshot is not a PNG or JPEG image')
        handle.seek(-2,2);require(handle.read(2)==b'\xff\xd9', 'Truncated JPEG without EOI')
        handle.seek(2);dimensions=None;components=set()
        frame_markers={0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf}
        while handle.tell()<min(size-2,2*2**20):
            require(handle.read(1)==b'\xff', 'Malformed JPEG marker')
            code=handle.read(1)
            while code==b'\xff':code=handle.read(1)
            require(len(code)==1 and code[0] not in {0,0xd8,0xd9,*range(0xd0,0xd8)}, 'Invalid JPEG header marker')
            marker=code[0];length_bytes=handle.read(2)
            require(len(length_bytes)==2, 'Truncated JPEG segment length')
            length=struct.unpack('>H',length_bytes)[0]
            require(length>=2 and handle.tell()+length-2<=size-2 and handle.tell()+length-2<=2*2**20, 'Truncated or excessive JPEG segment')
            payload=handle.read(length-2)
            if marker in frame_markers:
                require(dimensions is None and len(payload)>=6, 'Invalid JPEG frame header')
                precision,height,width,count=struct.unpack('>BHHB',payload[:6])
                require(precision in {8,12,16} and width>0 and height>0 and count in {1,3,4}
                        and len(payload)==6+3*count, 'Invalid JPEG dimensions or components')
                components=set(payload[6::3]);require(len(components)==count, 'Duplicate JPEG frame components')
                dimensions=(width,height)
            elif marker==0xda:
                require(dimensions is not None and len(payload)>=4, 'JPEG scan precedes a valid frame')
                count=payload[0]
                require(1<=count<=len(components) and len(payload)==1+2*count+3
                        and set(payload[1:1+2*count:2])<=components, 'Invalid JPEG scan components')
                require(handle.tell()<size-2, 'Truncated JPEG with no image scan data')
                return dimensions
        raise GateError('JPEG lacks a complete bounded frame and scan header')


def verify_ui(evidence, summary):
    require(summary.get('schema')=='neuroterrarium.ui-release-check.v1' and summary.get('mode')=='Local full graph', 'UI evidence must use the local complete graph')
    checks=summary['checks'];require({r['name'] for r in checks}>=UI_CHECKS and len({r['name'] for r in checks})==len(checks), 'Six core UI interactions are missing or duplicated')
    for row in checks:
        require(row['status']=='passed' and type(row['operations']) is int and row['operations']>0
                and isinstance(row['observed_effect'],str) and len(row['observed_effect'])>10, 'UI interaction has no observed effect')
        evidence.file(row['trace'])
        path=evidence.file(row['screenshot'])
        width,height=image_dimensions(path)
        require(width>=640 and height>=400, 'UI screenshot is too small to inspect')
    views=summary['viewports']
    require(len({(v['width'],v['height']) for v in views})>=2 and any(v['device_pixel_ratio']>=2 for v in views), 'UI resize or high-DPI evidence missing')
    require(summary.get('keyboard_verified') is True and len(set(summary['zoom_factors']))>=2, 'UI keyboard or zoom checks missing')


def verify_stability(evidence, ref, summary, context, item):
    require(summary.get('schema')=='neuroterrarium.stability.v1' and summary.get('status')=='completed' and summary.get('mode')=='Local full graph', 'Stability did not complete a live local run')
    require(summary.get('behavior_sha256')==context['behavior'] and summary.get('models_sha256')==context['models_hash'], 'Stability tested identity mismatch')
    require(set(summary['controllers'])==context['controller_names'] and len(summary['controllers'])==10,'Stability did not run ten distinct controllers')
    measured=summary['measured_wall_seconds']
    require(finite(measured,3600) and finite(summary['requested_wall_seconds'],3600)
            and finite(summary['actual_wall_seconds'],measured), 'A measured full hour is required')
    require(summary['resource_samples']>=measured/5 and summary['active_wall_seconds_estimate']>=.9*measured
            and summary['maximum_unpaused_progress_gap_seconds']<=30, 'Stability was paused, stalled or undersampled')
    require(summary['error'] is None and finite(summary['peak_sampled_rss_bytes'],1)
            and type(summary['post_warmup_memory_slope_bytes_per_second']) in (int,float)
            and math.isfinite(summary['post_warmup_memory_slope_bytes_per_second']), 'Stability resources or error state invalid')
    operations=summary['operations'];names={r.get('operation',r.get('command',{}).get('type')) for r in operations}
    require(STABILITY_OPERATIONS<=names and all(r.get('status_code')==200 for r in operations), 'Stability interactions missing or failed')
    for row in operations:
        if row.get('operation')=='snapshot_save': evidence.sibling(ref,row['snapshot'],row['sha256'])
        if row.get('operation')=='snapshot_restore': require(row.get('exact_executable_state_match') is True,'Stability restore was not exact')
    journal=evidence.file(item['journal']);require(item['journal']['sha256']==summary['execution_journal_sha256'],'Stability journal hash mismatch')
    windows=0;lines=0
    with journal.open() as handle:
        for i,line in enumerate(handle):
            require(len(line)<=4*2**20,'Oversized journal record');row=json.loads(line)
            require(row['sequence']==i,'Stability journal sequence mismatch')
            if i==0:require(row.get('schema')=='neuroterrarium.execution.v1','Missing stability journal identity')
            if 'record' in row:
                require(row['type'] in {'advance','command'} and len(row['record']['decisions'])==10,'Invalid ten-body journal record')
                windows+=1
            lines+=1
    require(lines>1 and windows>=measured/30 and windows==summary['committed_action_windows']==summary['journal_action_windows'], 'Stability committed window count mismatch')
    require(math.isclose(summary['cumulative_simulation_seconds'],windows*.02,rel_tol=0,abs_tol=1e-8),'Stability simulation clock mismatch')


def validate_category(name, evidence, item, summary, context):
    ref=item['summary']
    if name=='data':
        verify_graph(summary,context['data'])
        for key,source in context['data']['sources'].items():
            require(item['data_files'][key]['sha256']==source['sha256'],'Data source hash differs from frozen profile')
            evidence.file(item['data_files'][key])
    elif name=='neural': verify_neural(evidence,ref,summary,context)
    elif name=='cache_equivalence': verify_cache(evidence,summary,context)
    elif name=='training': verify_training(evidence,summary,context,item)
    elif name=='core_tests':
        tests=verified_junit(evidence.file(item['junit']))
        require(summary.get('executed_tests')==tests,'Core test summary count differs from JUnit')
    elif name in {'lint','typecheck','build'}:
        verify_steps(evidence,ref,summary['steps'],{name})
    elif name=='runtime': verify_runtime(summary,context)
    elif name in {'evaluation','supplement'}: verify_evaluation(evidence,ref,summary,context,item,supplement=name=='supplement')
    elif name=='ui':
        require(summary.get('source_sha256')=={key:context['source'][key] for key in ('web/src/main.ts','web/src/style.css')}, 'Native UI source identity differs')
        verify_ui(evidence,summary)
    elif name=='stability': verify_stability(evidence,ref,summary,context,item)
    elif name=='installed_wheel':
        require(summary.get('schema')=='neuroterrarium.installation-check.v1' and summary.get('status')=='passed','Installed wheel verification incomplete')
        require(summary['installation_network']=='offline wheelhouse' and summary['path_case']=='spaces and non-ASCII characters','Clean offline installation not verified')
        require('denied' in summary['runtime_network'],'Installed runtime network was not denied')
        verify_steps(evidence,ref,summary['steps'],{'create-environment','install-offline','dependency-check','installed-contents','installed-cli-help','offline-doctor','offline-complete-data','offline-runtime-validation'})
        require(item['wheel']['sha256']==summary['wheel']['sha256'],'Installed wheel identity mismatch');evidence.file(item['wheel'])
        verify_runtime(evidence.json(item['runtime_summary']),context)
    elif name=='portable':
        require(summary.get('schema')=='neuroterrarium.portable-check.v1' and summary.get('status')=='passed'
                and summary.get('clean_extraction') is True and summary.get('offline_runtime') is True,'Portable clean/offline execution incomplete')
        require(summary.get('path_case')=='spaces and non-ASCII characters','Portable relocation path not tested')
        verify_steps(evidence,ref,summary['steps'],{'extract','doctor','offline-runtime-validation','local-http','relocated-launcher'})
        evidence.file(summary['package']);verify_runtime(evidence.json(item['runtime_summary']),context)
    elif name=='public_review':
        require(summary.get('schema')=='neuroterrarium.public-review.v1' and summary.get('unresolved_findings')==0,'Public-content review has unresolved findings')
        scans={r['name']:r for r in summary['scans']}
        require({'secrets','private_paths','development_traces','licenses','links','asset_availability'}<=scans.keys(),'Public review categories missing')
        require(all(r['status']=='passed' and r['scanned_files']>0 and r['findings']==[] for r in scans.values()),'Public review scan incomplete or found unresolved content')
        require(summary.get('code_license')=='MIT' and summary.get('flywire_data_license')=='CC-BY-NC-4.0','Code and data licensing are not distinguished')
        require(len(summary['license_files'])>=3,'Necessary license notices missing')
        for file in summary['license_files']: evidence.file(file)


def candidate_context(evidence,candidate):
    source_paths={k:evidence.file(v) for k,v in candidate['source'].items()}
    source={k:v['sha256'] for k,v in candidate['source'].items()}
    require(set(BEHAVIOR)<=source.keys(),'Candidate behavior source inventory incomplete')
    config={k:v['sha256'] for k,v in candidate['configs'].items()}
    config_documents={name:evidence.json(ref) for name,ref in candidate['configs'].items()}
    require('data-v783.json' in config,'Missing frozen data manifest')
    data=evidence.json(candidate['configs']['data-v783.json'])
    models=evidence.json(candidate['model_manifest']);model_ref=candidate['model_manifest']
    require(models.get('schema')=='neuroterrarium.model-set.v1' and models.get('status')=='completed','Nine completed model artifacts are required')
    rows=models['models'];require(len(rows)==9,'Model set must contain exactly nine models')
    require(Counter(m['architecture'] for m in rows)=={'feedforward':3,'recurrent':3,'hybrid':3},'Model architecture counts mismatch')
    require(len({m['policy_sha256'] for m in rows})==len({m['id'] for m in rows})==9,'Model IDs or weights are duplicated')
    require(type(models['transitions_per_model']) is int and models['transitions_per_model']>=1_000_000,'Smoke budget cannot satisfy formal training')
    for arch in ('feedforward','recurrent','hybrid'):
        require(len({m['seed'] for m in rows if m['architecture']==arch})==3,'Training seeds duplicated')
    for row in rows:
        require(row['training_status']=='completed' and row['transitions']==models['transitions_per_model'],'Model training is incomplete')
        name=row['checkpoint'];require(PurePosixPath(name).name==name and name not in {'','.','..'},'Invalid checkpoint directory')
        path=str(PurePosixPath(model_ref['path']).parent/name/'manifest.json')
        manifest=evidence.json({'path':path,'sha256':row['checkpoint_manifest_sha256']})
        require(manifest['schema']=='neuroterrarium.ppo-checkpoint.v1' and set(manifest['files'])=={'policy.safetensors','optimizer.safetensors','runtime.safetensors','state.json'},'Checkpoint components missing')
        for filename,details in manifest['files'].items():
            actual=evidence.sibling({'path':path},filename,details['sha256'])
            require(actual.stat().st_size==details['bytes'],'Checkpoint component byte count mismatch')
            if filename=='policy.safetensors':require(details['sha256']==row['policy_sha256'] and actual.stat().st_size>0,'Checkpoint policy hash mismatch')
            if filename=='state.json':
                state=read_json(actual)
                require(state['training_status']=='completed' and state['transitions']==row['transitions']
                        and state['config_sha256']==models['protocol_sha256'] and canonical(state['config'])==models['protocol_sha256']
                        and state['source_sha256']==models['source_sha256'],'Checkpoint training state identity mismatch')
                require(state['policy']['architecture']==row['architecture'] and state['policy']['seed']==row['seed']
                        and state['policy']['observation_size']==59,'Checkpoint architecture or observation schema mismatch')
    training_paths={k:evidence.file(v) for k,v in candidate['training_sources'].items()}
    training={k:v['sha256'] for k,v in candidate['training_sources'].items()}
    require(training==models['source_sha256'],'Archived trained-source bytes do not match checkpoints')
    require(all(training[k]==source[k] for k in ('training.py','controllers.py')),'Learning source changed after training')
    return {'source':source,'source_paths':source_paths,'configs':config,'config_documents':config_documents,'data':data,'models':models,
            'model_ids':{m['id'] for m in rows},'controller_names':{'connectome',*[m['id'] for m in rows]},
            'models_hash':model_ref['sha256'],'behavior':canonical({k:source[k] for k in BEHAVIOR}),
            'training_source':training,'training_source_paths':training_paths}


def verify_transport_compatibility(name,evidence,sources,context,item):
    """Permit only the browser-opening URL correction for prior UI/hour records."""
    spec=item.get('transport_compatibility')
    require(name in {'ui','stability'} and isinstance(spec,dict), 'Tested source differs without a scoped transport proof')
    require(set(spec)=={'schema','original_source','candidate_source','regression_summary','regression_junit'},
            'Transport proof has missing or unknown fields')
    require(spec['schema']=='neuroterrarium.browser-launch-compatibility.v1','Unsupported transport proof')
    original=evidence.file(spec['original_source']);candidate=evidence.file(spec['candidate_source'])
    require(spec['original_source']['sha256']==sources.get('service.py')
            and spec['candidate_source']['sha256']==context['source'].get('service.py'), 'Transport proof source identity differs')
    before=ast.parse(original.read_text());after=ast.parse(candidate.read_text())
    functions=[node for node in before.body if isinstance(node,ast.FunctionDef) and node.name=='serve']
    require(len(functions)==1,'Transport original must contain exactly one serve function')
    blocks=[node for node in ast.walk(functions[0]) if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='open_browser']
    require(len(blocks)==1,'Transport original browser branch is ambiguous')
    old=ast.parse("if open_browser:\n    import webbrowser\n    threading.Timer(1.5,lambda:webbrowser.open(f'http://127.0.0.1:{port}')).start()\n").body[0]
    new=ast.parse("if open_browser:\n    import webbrowser\n    browser_host=f'[{host}]' if ':' in host else host\n    threading.Timer(1.5,lambda:webbrowser.open(f'http://{browser_host}:{port}')).start()\n").body[0]
    require(ast.dump(blocks[0])==ast.dump(old),'Transport original branch differs from the registered URL bug')
    blocks[0].body=new.body
    require(ast.dump(before)==ast.dump(after),'Transport proof includes changes outside the exact browser URL correction')
    summary=evidence.json(spec['regression_summary'])
    require(summary.get('status')=='passed' and summary.get('source_changes_during_run')==[], 'Transport regression execution incomplete')
    tested=summary.get('tested_source_files_sha256',{})
    for key in ('service.py','tests/test_service.py','scripts/validate_release.py'):
        mapped='src/neuroterrarium/'+key if '/' not in key else key
        require(key in context['source'] and mapped in tested and isinstance(context['source'][key],str)
                and SHA.fullmatch(context['source'][key]) and tested[mapped]==context['source'][key],
                'Transport regression tested source differs')
    verify_steps(evidence,spec['regression_summary'],summary.get('steps',[]),{'core_tests'})
    junit=evidence.file(spec['regression_junit'])
    require(verified_junit(junit)==summary.get('executed_tests'),'Transport regression JUnit count differs')
    tests={case.get('name') for case in ET.parse(junit).getroot().iter('testcase')
           if case.get('classname','').endswith('test_service')}
    for host in ('127.0.0.1','localhost','::1'):
        require(any(name.startswith('test_browser_opens_the_bound_loopback_address['+host+'-') for name in tests),
                'Transport regression omitted a supported loopback URL')
    return True


def verify_identity(name,identity,summary,context,evidence,item):
    """Verify the recorded dependency scope, without attributing unrelated files."""
    sources=identity['source_sha256'];configs=identity['config_sha256']
    expected=context['training_source'] if name=='training' else context['source']
    require(isinstance(sources,dict) and SOURCE_SCOPE[name]<=sources.keys(), 'Tested source scope omits required dependencies')
    differences={key for key,value in sources.items() if key not in expected or value!=expected[key]}
    if differences:
        require(differences=={'service.py'} and name in {'ui','stability'}, 'Tested source identity differs from declared candidate')
        verify_transport_compatibility(name,evidence,sources,context,item)
    else:
        require('transport_compatibility' not in item,'Unnecessary transport exception is not allowed')
    require(isinstance(configs,dict) and CONFIG_SCOPE[name]<=configs.keys(), 'Tested configuration scope omits required dependencies')
    require(all(key in context['configs'] and value==context['configs'][key] for key,value in configs.items()), 'Tested configuration identity differs')
    require(identity['model_manifest_sha256']==(None if name in WITHOUT_MODELS else context['models_hash']), 'Tested model identity differs or is inapplicable')
    if name in {'core_tests','lint'}:
        native=summary.get('tested_source_files_sha256',{})
        require(summary.get('source_changes_during_run')==[], 'Source changed during the test or lint execution')
        for key,value in sources.items():
            require(native.get('src/neuroterrarium/'+key if '/' not in key else key)==value,
                    'Native test source scope or hashes differ')
        for key,value in configs.items():
            require(native.get('configs/'+key)==value,'Native test configuration differs')
    elif name=='data':
        execution=evidence.json(item['execution'])
        require(type(execution.get('exit_code')) is int and execution['exit_code']==0 and execution.get('status')=='passed' and execution.get('summary')==summary, 'Native data execution evidence differs')
        native=execution['source_files_sha256']
        require(native.get('neuroterrarium/src/neuroterrarium/data.py')==sources['data.py']
                and native.get('neuroterrarium/configs/data-v783.json')==configs['data-v783.json'], 'Native data source or configuration differs')
    elif name=='neural':
        require(summary.get('checks',{}).get('component_source_unchanged') is True, 'Neural components changed while running')
        native=summary['component_sha256']
        expected_native={('src/neuroterrarium/'+key if '/' not in key else key):value for key,value in sources.items()}
        expected_native.update({'configs/'+key:value for key,value in configs.items()})
        require(native==expected_native, 'Native neural component scope or hashes differ')
    elif name=='ui':
        require(summary.get('source_sha256')=={key:sources[key] for key in ('web/src/main.ts','web/src/style.css')}, 'Native UI source identity differs')
    elif name=='public_review':
        require(sources==context['source'] and configs==context['configs'], 'Whole-candidate content review omits inventoried files')
        native=evidence.json(summary['candidate_inventory'])
        require(native.get('schema')=='neuroterrarium.candidate-content-inventory.v1' and summary.get('candidate_source_unchanged') is True
                and native.get('sha256')==canonical(native['files'])==summary.get('candidate_inventory_sha256'), 'Native candidate content inventory differs')
        for key,value in sources.items():
            path='src/neuroterrarium/'+key if '/' not in key else key
            require(native['files'].get(path,{}).get('sha256')==value, 'Native content source identity differs')
        for key,value in configs.items():
            path='configs/'+key
            require(native['files'].get(path,{}).get('sha256')==value, 'Native content configuration identity differs')
    elif name=='portable':
        required={*BEHAVIOR,'validation.py','service.py','cli.py'}
        require(summary.get('package_source_sha256')=={key:sources[key] for key in required}, 'Native installed package source differs')
        require(summary.get('config_sha256')=={key:configs[key] for key in BRAIN_CONFIGS}, 'Native installed package configuration differs')
        required_scripts={'portable_check.py','build_package.py','install_check.py'}
        require(summary.get('verification_source_sha256')=={key:sources['scripts/'+key] for key in required_scripts}, 'Native portable verification source differs')
        require(summary.get('validation_source_sha256')==sources['validation.py'], 'Native portable validation implementation differs')
    elif name=='runtime':
        require(summary.get('validation_source_sha256')==sources['validation.py'], 'Native validation implementation differs')
    elif name in {'evaluation','supplement'}:
        lock=evidence.json(item['protocol_lock'])
        require(all(sources.get(key)==value for key,value in lock['source'].items())
                and set(lock['source'])==SOURCE_SCOPE[name], 'Native experiment source scope or hashes differ')
        fields={'interface_sha256':'interface-v783.json','brain_interface_sha256':'brain-interface.json',
                'storage_profile_sha256':'brain-storage-v1.json','data_manifest_sha256':'data-v783.json'}
        for field,key in fields.items():
            require(lock.get(field)==configs[key], 'Native experiment interface or data configuration differs')


def validate(manifest,evidence):
    require(manifest.get('schema')=='neuroterrarium.release-evidence.v1','Unsupported release evidence schema')
    require(manifest.get('phase')=='local','This verifier handles local evidence only; remote CI and release have a separate phase')
    context=candidate_context(evidence,manifest['candidate'])
    items=manifest.get('items',{});require(isinstance(items,dict),'Release items must be an object')
    require(not set(items)-set(REQUIRED),'Unknown local gate category')
    checked=[]
    for name in REQUIRED:
        try:
            require(name in items,'Required category is missing')
            item=items[name]
            require(item.get('status')=='completed' and type(item.get('exit_code')) is int and item['exit_code']==0,'Required evidence did not complete successfully')
            require(isinstance(item.get('command'),list) and item['command'] and all(isinstance(x,str) and x for x in item['command']),'Missing actual command arguments')
            evidence.file(item['log'])
            identity=item['identity']
            for ref in item.get('artifacts',[]):evidence.file(ref)
            summary=evidence.json(item['summary'])
            verify_identity(name,identity,summary,context,evidence,item)
            if 'status' in summary: require(summary['status'] in {'passed','complete','completed','verified'},'Evidence summary is incomplete or failed')
            validate_category(name,evidence,item,summary,context)
            checked.append({'name':name,'status':'passed','exit_code':0,'command':item['command'],
                            'log':item['log'],'summary':item['summary'],'identity':identity})
        except (GateError,KeyError,TypeError,ValueError,OSError,StopIteration,ET.ParseError) as error:
            checked.append({'name':name,'status':'blocked','reason':type(error).__name__+': '+str(error)})
    return {'schema':'neuroterrarium.release-gate.v1','phase':'local',
            'status':'passed' if all(r['status']=='passed' for r in checked) else 'blocked',
            'evidence_manifest_sha256':canonical(manifest),'behavior_sha256':context['behavior'],
            'model_manifest_sha256':context['models_hash'],'checks':checked,
            'verified_files':len(evidence.verified),
            'scope':'Local candidate evidence only. This result does not claim a GitHub push, remote CI, Release or Pages deployment.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--manifest',required=True,help='Evidence manifest path relative to artifact root')
    parser.add_argument('--output',help='Optional result path relative to artifact root')
    args=parser.parse_args()
    try:
        evidence=Evidence(args.artifact_root)
        result=validate(read_json(evidence.path(args.manifest)),evidence)
    except (GateError,KeyError,TypeError,ValueError,OSError,StopIteration,ET.ParseError) as error:
        result={'schema':'neuroterrarium.release-gate.v1','phase':'local','status':'blocked',
                'reason':type(error).__name__+': '+str(error)}
    encoded=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.output:
        try:
            output=Evidence(args.artifact_root).path(args.output,must_exist=False)
            output.parent.mkdir(parents=True,exist_ok=True)
            temporary=Evidence(args.artifact_root).path(args.output+'.partial',must_exist=False)
            temporary.write_text(encoded);temporary.replace(output)
        except (GateError,OSError) as error:
            print('Cannot write gate result: '+str(error),file=sys.stderr)
            return 1
    print(encoded,end='')
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())
