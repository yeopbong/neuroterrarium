"""Run bounded full-profile input/output and embodied positive controls."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import psutil

from neuroterrarium.data import load_graph
from neuroterrarium.interface import MotorReadout, SensoryEncoder
from neuroterrarium.neural import SparseBrain
from neuroterrarium.reference import run_reference
from neuroterrarium.registry import Registry
from neuroterrarium.world import Body, Food, Stimulus, World, stream


def source_digest(root):
    h=hashlib.sha256()
    for p in sorted((root/'src').rglob('*.py'))+sorted((root/'configs').glob('*.json'))+sorted((root/'scripts').glob('*.py')):
        h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=0.5)
    parser.add_argument('--reference',action='store_true')
    args=parser.parse_args()
    if not 0.1<=args.duration<=2:raise ValueError('bounded probe duration is 0.1 to 2 seconds')
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():raise ValueError('Existing numerical evidence must be preserved')
    component_paths=[root/'src/neuroterrarium'/name for name in ('data.py','interface.py','neural.py','reference.py','registry.py','world.py')]
    component_paths += [root/'configs'/name for name in ('data-v783.json','brain-interface.json','interface-v783.json')]
    component_paths.append(Path(__file__))
    components={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in component_paths}
    result={'schema':'neural-gate-v1','source_tree_sha256':source_digest(root),
            'duration_seconds':args.duration,'stochastic_input_seed':48173,
            'component_sha256':components,'declared_trace_tolerance_mV':1e-9,
            'memory_samples':[],
            'status':'running','tests':[]}
    def save():
        temp=args.output/'summary.partial';temp.write_text(json.dumps(result,indent=2)+'\n')
        temp.replace(args.output/'summary.json')
    save()
    def headroom(stage):
        sample={'stage':stage,'available_bytes':psutil.virtual_memory().available,'rss_bytes':psutil.Process().memory_info().rss}
        result['memory_samples'].append(sample)
        if sample['available_bytes']<3*1024**3:
            result['status']='failed';result['error']='Less than 3 GiB system memory available';save()
            raise RuntimeError(result['error'])
    headroom('before_load')
    start=time.perf_counter()
    graph=load_graph(args.data_dir,root/'configs/data-v783.json')
    result['load_seconds']=time.perf_counter()-start;result['graph']=graph.summary
    registry=Registry.load(root/'configs/interface-v783.json')
    registry.verify_annotations(args.data_dir/'neuron_annotations_v2.1.0.tsv')
    groups=registry.resolve(graph.root_ids);sides=registry.resolve_sides(graph.root_ids)
    inputs=np.unique(np.concatenate([groups[k] for k in ('sugar','lplc2','lc4','dna02')]))
    motors=np.unique(np.concatenate([groups[k] for k in ('mn9','gf','dna01','dna02')]))
    start=time.perf_counter()
    blank=SparseBrain(len(graph.root_ids),graph.pre,graph.post,graph.signed_counts,input_neurons=inputs)
    result['graph_construction_seconds']=time.perf_counter()-start
    start=time.perf_counter();compiled=blank.fork();compiled.advance([[]]);del compiled
    result['first_compile_and_step_seconds']=time.perf_counter()-start
    headroom('after_graph_and_compilation')
    steps=round(args.duration/0.0001)
    for name,input_names,output_name in [('gustation',['sugar'],'mn9'),('visual_projection',['lplc2','lc4'],'gf')]:
        excited=np.concatenate([groups[k] for k in input_names])
        rng=stream(48173,name)
        events=rng.random((steps,len(excited)))<200*.0001
        tape=[excited[np.flatnonzero(row)] for row in events]
        for condition in ('normal','disconnect_output','readout_clamp'):
            headroom(name+'/'+condition)
            brain=blank.fork()
            if condition=='disconnect_output':brain.set_interventions(disconnect_output=excited)
            start=time.perf_counter();r=brain.advance(tape,motors,record_spikes=True)
            elapsed=time.perf_counter()-start
            readout=MotorReadout(sides);readout.clamped=condition=='readout_clamp'
            action=readout.action(r.spike_counts,args.duration)
            record={'case':name,'condition':condition,'wall_seconds':elapsed,
                    'neural_update_seconds':r.elapsed_seconds,'simulation_seconds':args.duration,
                    'simulation_per_wall':args.duration/elapsed,
                    'total_spikes':int(r.spike_counts.sum()),
                    'input_spikes':int(r.spike_counts[excited].sum()),
                    'output_spikes':int(r.spike_counts[groups[output_name]].sum()),
                    'motor_spikes':{k:int(r.spike_counts[groups[k]].sum()) for k in ('mn9','gf','dna01','dna02')},
                    'action':action.tolist()}
            stem=f'{name}-{condition}'
            start=time.perf_counter()
            np.savez_compressed(args.output/(stem+'.npz'),spike_steps=r.spike_steps,
                spike_neurons=r.spike_neurons,v_mV=r.v_mV,g_mV=r.g_mV,record_indices=motors,
                input_steps=np.nonzero(events)[0],input_neurons=excited[np.nonzero(events)[1]])
            record['record_seconds']=time.perf_counter()-start
            record['trajectory_sha256']=hashlib.sha256((args.output/(stem+'.npz')).read_bytes()).hexdigest()
            result['tests'].append(record);print(json.dumps(record),flush=True);save()
            if condition=='normal' and args.reference:
                reference=run_reference(len(graph.root_ids),graph.pre,graph.post,graph.signed_counts,
                    tape,input_neurons=inputs,record_indices=motors)
                comparison={'case':name,'reference_wall_seconds':reference.elapsed_seconds,
                    'spike_steps_equal':bool(np.array_equal(r.spike_steps,reference.spike_steps)),
                    'spike_neurons_equal':bool(np.array_equal(r.spike_neurons,reference.spike_neurons)),
                    'max_voltage_error_mV':float(np.max(np.abs(r.v_mV-reference.v_mV))),
                    'max_conductance_error_mV':float(np.max(np.abs(r.g_mV-reference.g_mV)))}
                result.setdefault('reference',[]).append(comparison);print(json.dumps(comparison),flush=True);save()
                ref_path=args.output/f'reference-{name}.npz'
                np.savez_compressed(ref_path,spike_steps=reference.spike_steps,spike_neurons=reference.spike_neurons,
                                    v_mV=reference.v_mV,g_mV=reference.g_mV,record_indices=motors)
                comparison['trajectory_sha256']=hashlib.sha256(ref_path.read_bytes()).hexdigest();save()
                del reference;gc.collect();headroom('after_reference/'+name)
            del brain,r;gc.collect()
    result['closed_loop']=[]
    for case in ('feeding','expansion'):
        for condition in ('normal','disconnect_output','readout_clamp'):
            brain=blank.fork();world=World(41,1);world.bodies=[Body(20,28,0)]
            world.foods=[Food(20,28)] if case=='feeding' else []
            world.stimuli=[Stimulus(28,28,0.4,growth=8)] if case=='expansion' else []
            encoder=SensoryEncoder(groups,sides,919);readout=MotorReadout(sides)
            if condition=='disconnect_output':
                brain.set_interventions(disconnect_output=groups['sugar'] if case=='feeding' else np.r_[groups['lplc2'],groups['lc4']])
            readout.clamped=condition=='readout_clamp'
            rows=[];start=time.perf_counter()
            for window in range(round(args.duration/.02)):
                observation=world.observe()[0]
                tape=encoder.encode(observation)
                neural=brain.advance(tape)
                action=readout.action(neural.spike_counts)
                world.advance([action])
                rows.append({'step':world.step,'time':world.time,'body':vars(world.bodies[0]).copy(),
                             'observation':observation.tolist(),'input_rates_hz':dict(encoder.last_rates_hz),
                             'action':action.tolist(),'motor_rates_hz':dict(readout.filtered)})
            elapsed=time.perf_counter()-start
            record={'case':case,'condition':condition,'wall_seconds':elapsed,'simulation_seconds':world.time,
                    'distance':float(np.hypot(world.bodies[0].x-20,world.bodies[0].y-28)),
                    'food':world.bodies[0].food,'final_energy':world.bodies[0].energy,
                    'neural_steps':brain.step}
            path=args.output/f'closed-{case}-{condition}.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            record['trajectory_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            result['closed_loop'].append(record);print(json.dumps(record),flush=True);save()
            del brain;gc.collect()
    checks={}
    for name in ('gustation','visual_projection'):
        rows={x['condition']:x for x in result['tests'] if x['case']==name}
        checks[name]=rows['normal']['output_spikes']>0 and rows['disconnect_output']['output_spikes']<rows['normal']['output_spikes']
        checks[name+'_readout_clamp']=rows['readout_clamp']['action']==[0.0]*4
    for name,metric in [('feeding','food'),('expansion','distance')]:
        rows={x['condition']:x for x in result['closed_loop'] if x['case']==name}
        checks['closed_'+name]=rows['normal'][metric]>0 and rows['disconnect_output'][metric]<rows['normal'][metric] and rows['readout_clamp'][metric]==0
    if args.reference:
        checks['full_graph_reference']=all(x['spike_steps_equal'] and x['spike_neurons_equal'] and x['max_voltage_error_mV']<=1e-9 and x['max_conductance_error_mV']<=1e-9 for x in result['reference'])
    current={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in component_paths}
    checks['component_source_unchanged']=current==components
    result['checks']=checks
    result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
    result['status']='passed' if all(checks.values()) else 'failed';save()
    print(json.dumps({'status':result['status'],'checks':checks}),flush=True)
    return 0 if all(checks.values()) else 1


if __name__=='__main__':raise SystemExit(main())
