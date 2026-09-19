"""Matched single-body protocols with preserved raw trajectories and failures."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from .controllers import LearnedController, Policy, named_seed
from .data import sha256_file
from .interface import MotorReadout, SensoryEncoder
from .registry import Registry, default_path
from .runtime import load_model_set
from .training import atomic_json, config_digest
from .world import ACTION_DT, Body, Food, Obstacle, Stimulus, World, stream

SCENARIOS={
    'A_expansion':100,'A_static':100,'A_translation':100,'A_contraction':100,
    'B_food':250,'B_danger':250,
    'C_occlusion':300,'C_relocation':300,'C_unseen':300,
}


def make_scene(case:str,seed:int) -> tuple[World,dict]:
    if case not in SCENARIOS:raise ValueError('Unknown evaluation scenario')
    rng=stream(seed,'evaluation/environment')
    world=World(seed,1);world.replenish=False
    x,y=40.0,28.0;heading=float(rng.uniform(-math.pi,math.pi))
    world.bodies=[Body(x,y,heading,energy=.85)]
    world.foods=[];world.obstacles=[];world.stimuli=[]
    direction=heading+float(rng.choice([0,math.pi/2,math.pi,-math.pi/2]))
    distance=float(rng.uniform(7,11));speed=float(rng.uniform(2,6));radius=float(rng.uniform(1,1.5))
    event={'onset_step':10 if case.startswith('A_') else 0,'case':case,
           'stimulus_direction_relative_initial_body':math.atan2(math.sin(direction-heading),math.cos(direction-heading)),
           'stimulus_speed':speed,'stimulus_initial_radius':radius,
           'environment_seed':seed,'neural_repeat':0,'neural_stream':'evaluation/neural/0'}
    event['stimulus']={'x':x+distance*math.cos(direction),'y':y+distance*math.sin(direction),
                       'radius':radius,'vx':0.0,'vy':0.0,'growth':0.0,'physical':False}
    if case=='A_expansion':event['stimulus']['growth']=speed
    elif case=='A_translation':
        event['stimulus']['vx']=-math.sin(direction)*speed
        event['stimulus']['vy']=math.cos(direction)*speed
    elif case=='A_contraction':event['stimulus']['growth']=-speed*.5
    if case.startswith('A_'):
        baseline={**event['stimulus'],'vx':0.0,'vy':0.0,'growth':0.0}
        world.stimuli=[Stimulus(**baseline)]
    if not case.startswith('A_'):
        angle=heading+float(rng.uniform(-.8,.8))
        d=float(rng.uniform(7,11) if case in ('C_occlusion','C_unseen') else rng.uniform(2,7))
        world.foods=[Food(x+d*math.cos(angle),y+d*math.sin(angle),1)]
        if case=='B_danger':
            f=world.foods[0]
            world.stimuli=[Stimulus(f.x+4,f.y,1.3,vx=-1.2,physical=True)]
        elif case in ('C_occlusion','C_unseen'):
            f=world.foods[0]
            world.obstacles=[Obstacle((x+f.x)/2,(y+f.y)/2,1.0)]
            if case=='C_unseen':
                world.obstacles += [Obstacle(x+5,y+5,1.5),Obstacle(x-5,y+3,2.0)]
                world.foods.append(Food(x-7,y-5,.7))
        if case=='C_relocation':
            event['relocation_step']=75
            event['relocation']=[x+float(rng.uniform(-8,8)),y+float(rng.uniform(-8,8))]
            event['replacement_food_amount']=1.0
            event['relocation_protocol']='Remove the original resource; introduce one new unit at a fixed location and time.'
    event['initial_world']=world.snapshot()
    return world,event


class BrainRunner:
    def __init__(self,blank,groups,sides,seed):
        self.brain=blank.fork();self.groups,self.sides=groups,sides
        self.encoder=SensoryEncoder(groups,sides,named_seed(seed,'evaluation/neural/0'))
        self.readout=MotorReadout(sides)
        self.last_rates={};self.last_inputs={}

    def act(self,observation):
        tape=self.encoder.encode(observation)
        result=self.brain.advance(tape)
        self.last_rates={k:float(result.spike_counts[v].mean()/ACTION_DT) for k,v in self.groups.items()}
        self.last_inputs=dict(self.encoder.last_rates_hz)
        return self.readout.action(result.spike_counts)


def rule_action(obs):
    chemical=obs[48:52]
    turn=float(np.clip(chemical[1]-chemical[3],-1,1))
    proximity=obs[:16]
    if max(proximity[[15,0,1]])>.75:
        turn=float(np.clip(turn+(proximity[12:16].mean()-proximity[1:5].mean())*4,-1,1))
    return np.array([.6,turn,float(max(obs[32:48])>.25),float(obs[52]>.5)])


class EvaluationStopped(RuntimeError):
    """A resource, time or cancellation limit stopped new experiment work."""
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


class EvaluationBudget:
    """Atomic cumulative wall accounting, including conservative crash recovery."""
    def __init__(self, output, protocol_sha256, limits, *, started=None, cancel_file=None):
        required={'max_total_wall_seconds','max_process_rss_bytes','min_available_memory_bytes','min_free_disk_bytes'}
        if not isinstance(limits,dict) or set(limits)!=required:
            raise ValueError('Frozen evaluation resource limits are required')
        for key,value in limits.items():
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('Resource limits must be finite positive numbers')
        if (limits['min_available_memory_bytes']<3*2**30 or limits['min_free_disk_bytes']<10*2**30
                or limits['max_process_rss_bytes']>2*2**30):
            raise ValueError('Evaluation cannot relax the declared memory or disk protection')
        self.output=Path(output);self.path=self.output/'budget.json';self.protocol=protocol_sha256
        self.limits=dict(limits);self.started=time.perf_counter() if started is None else started
        self.cancel_file=Path(cancel_file) if cancel_file is not None else None
        self.previous=0.;self.recovered_reservation=0.;self.attempt=1
        if self.path.exists():
            value=json.loads(self.path.read_text())
            if (value.get('schema')!='neuroterrarium.evaluation-budget.v1'
                    or value.get('protocol_sha256')!=protocol_sha256 or value.get('limits')!=limits):
                raise ValueError('Existing evaluation budget has a different protocol')
            for key in ('elapsed_seconds','reserved_seconds'):
                if type(value.get(key)) not in (int,float) or not math.isfinite(value[key]) or value[key]<0:
                    raise ValueError('Invalid persisted evaluation budget')
            if type(value.get('attempt')) is not int or value['attempt']<1:raise ValueError('Invalid budget attempt')
            self.recovered_reservation=value['reserved_seconds']
            self.previous=value['elapsed_seconds']+self.recovered_reservation
            self.attempt=value['attempt']+1

    @property
    def elapsed(self):
        return self.previous+time.perf_counter()-self.started

    def checkpoint(self,status,*,episode=None,reserve_seconds=0):
        if not 0<=reserve_seconds<=600:raise ValueError('Invalid in-flight budget reservation')
        atomic_json(self.path,{'schema':'neuroterrarium.evaluation-budget.v1','protocol_sha256':self.protocol,
            'limits':self.limits,'status':status,'elapsed_seconds':self.elapsed,
            'reserved_seconds':reserve_seconds,'in_flight_episode':episode,'attempt':self.attempt,
            'recovered_reservation_seconds':self.recovered_reservation,
            'accounting':'Cumulative process wall time across resumes; an unclosed operation reservation is charged conservatively after abrupt interruption.'})

    def check(self):
        if self.cancel_file is not None and self.cancel_file.exists():
            raise EvaluationStopped('cancelled','cancel_file')
        if self.elapsed>=self.limits['max_total_wall_seconds']:
            raise EvaluationStopped('budget_exhausted','cumulative_wall_budget')
        if psutil.Process().memory_info().rss>self.limits['max_process_rss_bytes']:
            raise EvaluationStopped('resource_limited','process_rss')
        if psutil.virtual_memory().available<self.limits['min_available_memory_bytes']:
            raise EvaluationStopped('resource_limited','system_available_memory')
        if psutil.disk_usage(self.output).free<self.limits['min_free_disk_bytes']:
            raise EvaluationStopped('resource_limited','free_disk')


def run_episode(case,seed,controller,*,ablation='none',timeout_seconds=120,stop_check=None):
    """Run a fixed protocol and preserve every completed action window.

    NaN denotes a non-applicable channel in numeric NPZ arrays; JSON summary
    fields use null. Backend failures are data with a stage and a partial trace,
    and are never converted into a different controller or a successful trial.
    """
    if not isinstance(timeout_seconds,(int,float)) or isinstance(timeout_seconds,bool) or not math.isfinite(timeout_seconds) or timeout_seconds<0:
        raise ValueError('Invalid episode timeout')
    allowed={'none','vision_off','chemical_off'}
    brain=isinstance(controller,BrainRunner);learned=isinstance(controller,LearnedController)
    if brain:allowed|={'output_disconnect','readout_clamp','tonic_off'}
    if learned and controller.policy.recurrent:allowed.add('memory_reset')
    if learned and controller.policy.architecture=='hybrid':allowed.add('reflex_off')
    if ablation not in allowed:raise ValueError('Ablation is not applicable to this controller')
    world,events=make_scene(case,seed)
    events.setdefault('initial_world',world.snapshot())
    rng=stream(seed,'evaluation/random-baseline')
    observations=[];actions=[];bodies=[];rates=[];input_rates=[];raw=[];logp=[];inference=[];rss=[]
    reaction=None;recovery=None;baseline_drive=None;consumption_before_change=0.;reflex_count=0
    start=time.perf_counter();status='completed';failure=None
    names=['sugar','lplc2','lc4','mn9','gf','dna01','dna02']
    initial=world.bodies[0]
    initial_body=np.asarray([initial.x,initial.y,initial.heading,initial.speed,initial.energy,initial.food,initial.collisions],dtype=np.float64)
    physical=any(stimulus.physical for stimulus in world.stimuli)
    if brain:
        if ablation=='output_disconnect':controller.brain.set_interventions(disconnect_output=np.r_[controller.groups['lplc2'],controller.groups['lc4']])
        if ablation=='readout_clamp':controller.readout.clamped=True
        if ablation=='tonic_off':controller.encoder.tonic_hz=0
    if learned and ablation=='reflex_off':controller.reflex_enabled=False
    stage='resource_sampling'
    try:
        process=psutil.Process()
        rss.append(process.memory_info().rss)
        for step in range(SCENARIOS[case]):
            stage='resource_guard'
            if stop_check is not None:stop_check()
            if time.perf_counter()-start>=timeout_seconds:
                status='timeout';break
            stage='external_event'
            if case.startswith('A_') and step==events['onset_step']:
                world.stimuli=[Stimulus(**events['stimulus'])]
                baseline_drive=float(np.mean([a[2] for a in actions[-5:]]))
                body=world.bodies[0];stimulus=world.stimuli[0]
                angle=math.atan2(stimulus.y-body.y,stimulus.x-body.x)-body.heading
                events['onset_body_relative_bearing']=math.atan2(math.sin(angle),math.cos(angle))
                events['onset_body_distance']=math.hypot(stimulus.x-body.x,stimulus.y-body.y)
            if events.get('relocation_step')==step:
                removed=sum(food.amount for food in world.foods)
                amount=events['replacement_food_amount']
                world.foods=[Food(*events['relocation'],amount)]
                events['resource_change']={'time':world.time,'added_food':amount,'removed_unconsumed_food':removed}
                consumption_before_change=world.bodies[0].food
            stage='observation'
            obs=world.observe()[0].copy()
            if ablation=='vision_off':obs[16:48]=0
            if ablation=='chemical_off':obs[48:52]=0
            if ablation=='memory_reset':controller.reset()
            stage='controller';tick=time.perf_counter()
            if controller=='random':action=rng.uniform([0,-1,0,0],[1,1,1,1])
            elif controller=='rule':action=rule_action(obs)
            else:action=controller.act(obs)
            cost=time.perf_counter()-tick
            action=np.asarray(action,dtype=np.float64)
            if action.shape!=(4,) or not np.isfinite(action).all() or np.any(action<[0,-1,0,0]) or np.any(action>1):
                raise ValueError('Invalid applied action')
            raw_action=controller.last['raw_action'] if learned else [np.nan]*4
            likelihood=controller.last['log_probability'] if learned else np.nan
            neural=[controller.last_rates[n] for n in names] if brain else [np.nan]*len(names)
            # Non-input neural groups are not assigned a fictitious input rate.
            inputs=[controller.last_inputs.get(n,np.nan) for n in names] if brain else [np.nan]*len(names)
            stage='world';world.advance([action]);body=world.bodies[0]
            if case.startswith('A_') and baseline_drive is not None and action[2]>=.5 and action[2]-baseline_drive>=.2 and reaction is None:
                reaction=(step-events['onset_step'])*ACTION_DT
            if events.get('relocation_step',10**9)<=step and body.food>consumption_before_change and recovery is None:
                # Consumption is measured at the end of this integration window.
                recovery=(step+1-events['relocation_step'])*ACTION_DT
            observations.append(obs);actions.append(action.copy())
            bodies.append([body.x,body.y,body.heading,body.speed,body.energy,body.food,body.collisions])
            rates.append(neural);input_rates.append(inputs);raw.append(raw_action);logp.append(likelihood);inference.append(cost)
            if learned:reflex_count+=int(controller.last['reflex_triggered'])
            stage='resource_sampling';rss.append(process.memory_info().rss)
    except EvaluationStopped as error:
        status='interrupted'
        failure={'kind':type(error).__name__,'stage':stage,'completed_windows':len(actions),
                 'stop_status':error.status,'reason':error.reason}
    except KeyboardInterrupt:
        status='interrupted'
        failure={'kind':'KeyboardInterrupt','stage':stage,'completed_windows':len(actions),
                 'stop_status':'cancelled','reason':'keyboard_interrupt'}
    except Exception as error:
        status='failed'
        failure={'kind':type(error).__name__,'stage':stage,'completed_windows':len(actions)}
    elapsed=time.perf_counter()-start;n=len(actions)
    if status=='completed' and elapsed>=timeout_seconds:status='timeout'
    applied=np.asarray(actions,dtype=np.float64).reshape(n,4)
    motor=ACTION_DT*(.001+.003*applied[:,0]+.015*applied[:,2]+.001*np.abs(applied[:,1]))
    row={'scenario':case,'environment_seed':seed,'neural_repeat':0,'ablation':ablation,'status':status,'failure':failure,
         'simulation_seconds':n*ACTION_DT,'wall_seconds':elapsed,
         'inference_wall_seconds':sum(inference),'inference_mean_ms':1000*float(np.mean(inference)) if inference else None,
         'food':world.bodies[0].food if status=='completed' else None,
         'collision_substeps':world.bodies[0].collisions if status=='completed' else None,
         'motor_energy_demand':float(motor.sum()) if status=='completed' else None,
         'physical_energy_debit':None if physical else 0.0,
         'physical_energy_debit_status':'not_measured_at_body_substep' if physical else 'not_applicable',
         'final_simulated_energy':world.bodies[0].energy if status=='completed' else None,
         'reaction_seconds':reaction,
         'reaction_status':('not_applicable' if not case.startswith('A_') else 'responded' if reaction is not None else 'no_response' if status=='completed' else status),
         'baseline_escape_drive':baseline_drive,'preactivated':baseline_drive>=.5 if baseline_drive is not None else None,
         'recovery_seconds':recovery,
         'recovery_status':('not_applicable' if case!='C_relocation' else 'recovered' if recovery is not None else 'not_recovered' if status=='completed' else status),
         'escape_drive_integral':float(applied[:,2].sum()*ACTION_DT),
         'reflex_interventions':reflex_count,'sampled_peak_process_rss_bytes':max(rss) if rss else None,
         'rss_sampling':'before episode and after each completed action window; transient allocations may be missed',
         'completed_windows':n,'neural_trace_applicable':brain,'policy_likelihood_applicable':learned}
    arrays={'observation':np.asarray(observations,dtype=np.float64).reshape(n,59),'applied_action':applied,
            'body':np.asarray(bodies,dtype=np.float64).reshape(n,7),'initial_body':initial_body,
            'motor_groups_hz':np.asarray(rates,dtype=np.float64).reshape(n,len(names)),
            'neural_input_hz':np.asarray(input_rates,dtype=np.float64).reshape(n,len(names)),
            'raw_policy_action':np.asarray(raw,dtype=np.float64).reshape(n,4),
            'raw_policy_log_probability':np.asarray(logp,dtype=np.float64),
            'inference_seconds':np.asarray(inference,dtype=np.float64),'motor_energy_demand':motor,
            'sampled_process_rss_bytes':np.asarray(rss,dtype=np.int64)}
    return row,arrays,events


def evaluation_source_digest():
    from .runtime import BEHAVIOR_SOURCE_FILES
    root=Path(__file__).parent
    return {name:sha256_file(root/name) for name in (*BEHAVIOR_SOURCE_FILES,'evaluation.py')}


def recompute_metrics(arrays,events,*,status='completed'):
    """Recompute outcome summaries from numeric records, without a controller."""
    actions=np.asarray(arrays['applied_action']);body=np.asarray(arrays['body']);n=len(actions)
    if actions.shape!=(n,4) or body.shape!=(n,7) or not np.isfinite(actions).all() or not np.isfinite(body).all():
        raise ValueError('Invalid numeric outcome trajectory')
    if np.any(actions<[0,-1,0,0]) or np.any(actions>1):raise ValueError('Action bounds in trajectory')
    case=events['case'];onset=events.get('onset_step');baseline=None;reaction=None;recovery=None
    if case.startswith('A_') and n>onset:
        baseline=float(actions[max(0,onset-5):onset,2].mean())
        candidates=np.flatnonzero((actions[onset:,2]>=.5)&(actions[onset:,2]-baseline>=.2))
        if len(candidates):reaction=int(candidates[0])*ACTION_DT
    if case=='C_relocation' and n>events['relocation_step']:
        change=events['relocation_step']
        before=body[change-1,5] if change else arrays['initial_body'][5]
        candidates=np.flatnonzero(body[change:,5]>before)
        if len(candidates):recovery=(int(candidates[0])+1)*ACTION_DT
    motor=ACTION_DT*(.001+.003*actions[:,0]+.015*actions[:,2]+.001*np.abs(actions[:,1]))
    inference=np.asarray(arrays['inference_seconds'])
    if inference.shape!=(n,) or not np.isfinite(inference).all() or np.any(inference<0):
        raise ValueError('Invalid inference timing trajectory')
    return {'food':float(body[-1,5]) if n and status=='completed' else None,
            'collision_substeps':int(body[-1,6]) if n and status=='completed' else None,
            'final_simulated_energy':float(body[-1,4]) if n and status=='completed' else None,
            'motor_energy_demand':float(motor.sum()) if status=='completed' else None,
            'escape_drive_integral':float(actions[:,2].sum()*ACTION_DT),
            'simulation_seconds':n*ACTION_DT,'inference_wall_seconds':float(inference.sum()),
            'inference_mean_ms':float(inference.mean()*1000) if n else None,
            'baseline_escape_drive':baseline,'reaction_seconds':reaction,'recovery_seconds':recovery}


def evaluation_plan(config,policies):
    """Expand the frozen intervention list without implicit architecture cases."""
    plan=[(name,'none',list(config['scenarios'])) for name in ['connectome',*sorted(policies),'random','rule','untrained']]
    allowed={'connectome':{'vision_off','output_disconnect','readout_clamp','tonic_off'},
             'feedforward':{'chemical_off','vision_off'},
             'recurrent':{'memory_reset','chemical_off','vision_off'},
             'hybrid':{'reflex_off','memory_reset','chemical_off','vision_off'}}
    used={(name,ablation) for name,ablation,_ in plan}
    if not isinstance(config.get('ablations'),list):raise ValueError('Frozen ablation list required')
    for item in config['ablations']:
        if not isinstance(item,dict) or set(item)!={'controller_kind','intervention','scenarios'}:
            raise ValueError('Invalid ablation specification')
        kind,intervention,cases=item['controller_kind'],item['intervention'],item['scenarios']
        if kind not in allowed or intervention not in allowed[kind]:raise ValueError('Inapplicable protocol ablation')
        if not isinstance(cases,list) or not cases or len(set(cases))!=len(cases) or any(case not in config['scenarios'] for case in cases):
            raise ValueError('Invalid ablation scenarios')
        names=['connectome'] if kind=='connectome' else sorted(name for name,policy in policies.items() if policy.architecture==kind)
        if not names:raise ValueError('Ablation references unavailable controller kind')
        for name in names:
            if (name,intervention) in used:raise ValueError('Duplicate protocol ablation')
            used.add((name,intervention));plan.append((name,intervention,list(cases)))
    return plan


def _existing_records(rawdir,units,protocol_sha256,*,stop_check=None):
    records={}
    for identifier,name,ablation,case,seed in units:
        if stop_check is not None:stop_check()
        metadata=rawdir/(identifier+'.json');trajectory=rawdir/(identifier+'.npz')
        if metadata.exists():
            row=json.loads(metadata.read_text())
            windows=row.get('completed_windows')
            if (row.get('protocol_sha256')!=protocol_sha256 or not trajectory.is_file()
                    or row.get('trajectory')!=trajectory.name
                    or sha256_file(trajectory)!=row.get('trajectory_sha256')
                    or (row.get('controller'),row.get('ablation'),row.get('scenario'),row.get('environment_seed'))!=(name,ablation,case,seed)
                    or type(row.get('neural_repeat')) is not int or row['neural_repeat']!=0
                    or type(windows) is not int or not 0<=windows<=SCENARIOS[case]
                    or row.get('simulation_seconds')!=windows*ACTION_DT
                    or row.get('status')=='completed' and windows!=SCENARIOS[case]
                    or row.get('status') not in {'completed','failed','timeout','interrupted'}):
                raise ValueError('Existing experiment record integrity mismatch')
            records[identifier]=row
        elif trajectory.exists():raise ValueError('Orphaned experiment trajectory retained; use a new experiment version')
    expected={unit[0] for unit in units}
    if ({path.stem for path in rawdir.glob('*.json')}|{path.stem for path in rawdir.glob('*.npz')})-expected:
        raise ValueError('Unexpected records outside the frozen experiment plan')
    return records


def _verify_completed(output,data,lock,units,budget):
    """Read a completed version without spending or rewriting its run budget."""
    def check():
        if budget.cancel_file is not None and budget.cancel_file.exists():
            raise EvaluationStopped('cancelled','cancel_file')
        if time.perf_counter()-budget.started>=budget.limits['max_total_wall_seconds']:
            raise EvaluationStopped('budget_exhausted','verification_wall_budget')
        if psutil.Process().memory_info().rss>budget.limits['max_process_rss_bytes']:
            raise EvaluationStopped('resource_limited','process_rss')
        if psutil.virtual_memory().available<budget.limits['min_available_memory_bytes']:
            raise EvaluationStopped('resource_limited','system_available_memory')

    status='verified';stop=None
    try:
        check()
        protocol=config_digest(lock)
        if (data.get('schema')!='neuroterrarium.evaluation-results.v1'
                or data.get('protocol_sha256')!=protocol or data.get('status')!='completed'
                or type(data.get('expected_records')) is not int or data['expected_records']!=len(units)
                or type(data.get('missing_records')) is not int or data['missing_records']!=0
                or data.get('stop') is not None
                or type(data.get('wall_seconds')) not in (int,float)
                or not math.isfinite(data['wall_seconds']) or data['wall_seconds']<0):
            raise ValueError('Completed experiment index integrity mismatch')
        saved_budget=json.loads(budget.path.read_text())
        if (saved_budget.get('status')!='completed' or saved_budget.get('reserved_seconds')!=0
                or saved_budget.get('in_flight_episode') is not None
                or saved_budget['elapsed_seconds']<data['wall_seconds']):
            raise ValueError('Completed experiment budget integrity mismatch')
        records=_existing_records(output/'raw',units,protocol,stop_check=check)
        index=[records[unit[0]] for unit in units if unit[0] in records]
        if (len(index)!=len(units) or any(row['status']!='completed' for row in index)
                or data.get('records')!=index):
            raise ValueError('Completed experiment index disagrees with frozen raw records')
        graph=json.loads((output/'graph-summary.json').read_text())
        profile=json.loads(default_path().with_name('brain-storage-v1.json').read_text())
        if (graph.get('protocol_sha256')!=protocol or graph.get('neural_graph_sha256')!=profile['graph_digest']
                or config_digest(graph.get('profile'))!=profile['summary_sha256']):
            raise ValueError('Completed experiment graph identity mismatch')
        expected_summary=summarize(output,persist=False,stop_check=check)
        if json.loads((output/'summary.json').read_text())!=expected_summary:
            raise ValueError('Completed experiment summary disagrees with raw outcomes')
        check()
    except EvaluationStopped as error:
        status=error.status;stop={'kind':type(error).__name__,'reason':error.reason,'stage':'completed_verification'}
    except KeyboardInterrupt:
        status='cancelled';stop={'kind':'KeyboardInterrupt','reason':'keyboard_interrupt','stage':'completed_verification'}
    return {'status':status,'operation':'read_only_verification','historical_status':'completed',
            'records':len(units),'missing_records':0,'cumulative_wall_seconds':budget.previous,
            'verification_wall_seconds':time.perf_counter()-budget.started,'stop':stop,'output':str(output)}


def evaluate(config,data_dir,models_dir,output_dir,*,cancel_file=None):
    invocation_started=time.perf_counter()
    config=json.loads(Path(config).read_text()) if not isinstance(config,dict) else config
    if config.get('schema')!='neuroterrarium.evaluation.v1' or config.get('status')!='frozen':raise ValueError('Frozen experiment protocol required')
    seeds=config['environment_seeds']
    if len(seeds)<30 or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):raise ValueError('At least 30 unique frozen environment seeds required')
    if config['scenarios']!=list(SCENARIOS):raise ValueError('Scenario protocol mismatch')
    if config.get('neural_stochastic_repeats')!=1:raise ValueError('This protocol implements exactly one named neural repeat')
    for family in ('A','B','C'):
        if any(config.get(family+'_duration_seconds')!=steps*ACTION_DT for name,steps in SCENARIOS.items() if name.startswith(family+'_')):
            raise ValueError('Configured duration does not match executable protocol')
    timeout=config.get('timeout_seconds',120)
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0<timeout<=120:raise ValueError('Invalid evaluation timeout')
    output=Path(output_dir)
    historical=json.loads((output/'results.json').read_text()) if (output/'results.json').is_file() else None
    completed=isinstance(historical,dict) and historical.get('status')=='completed'
    if completed and any(not path.exists() for path in (output/'protocol-lock.json',output/'budget.json',output/'raw',output/'summary.json',output/'graph-summary.json')):
        raise ValueError('Completed experiment is missing required evidence; preserve this experiment version')
    output.mkdir(parents=True,exist_ok=True)
    policies,model_manifest=load_model_set(models_dir)
    specifications=evaluation_plan(config,policies)
    lock={'schema':'neuroterrarium.evaluation-lock.v1','config':config,'source':evaluation_source_digest(),
          'model_manifest_sha256':config_digest(model_manifest),'interface_sha256':sha256_file(default_path()),
          'brain_interface_sha256':sha256_file(default_path().with_name('brain-interface.json')),
          'storage_profile_sha256':sha256_file(default_path().with_name('brain-storage-v1.json')),
          'data_manifest_sha256':sha256_file(default_path().with_name('data-v783.json'))}
    if (output/'protocol-lock.json').exists():
        if json.loads((output/'protocol-lock.json').read_text())!=lock:raise ValueError('Existing experiment uses a different frozen source or protocol')
    else:atomic_json(output/'protocol-lock.json',lock)
    rawdir=output/'raw';rawdir.mkdir(exist_ok=True)
    if not (output/'budget.json').exists() and any(rawdir.glob('*.json')):
        raise ValueError('Existing experiment records have no cumulative budget; preserve this experiment version')
    budget=EvaluationBudget(output,config_digest(lock),config.get('resource_limits'),
                            started=invocation_started,cancel_file=cancel_file)
    units=[]
    for name,ablation,cases in specifications:
        for case in cases:
            for seed in seeds:
                identifier=f'{name}--{ablation}--{case}--{seed}'
                units.append((identifier,name,ablation,case,seed))
    if completed:return _verify_completed(output,historical,lock,units,budget)
    records=_existing_records(rawdir,units,config_digest(lock))
    outcome='running';stop=None;stage='cache_loading'
    try:
        budget.check();budget.checkpoint('running',reserve_seconds=600)
        torch.set_num_threads(1)
        from .brain_cache import ensure_cache, load_cache
        ensure_cache(data_dir);cached=load_cache(data_dir)
        registry=Registry.load()
        groups=registry.resolve(cached.root_ids);sides=registry.resolve_sides(cached.root_ids)
        blank=cached.brain
        atomic_json(output/'graph-summary.json',{'profile':cached.summary,'neural_graph_sha256':blank.graph_digest,
                                               'protocol_sha256':config_digest(lock)})
        budget.checkpoint('running')
        for identifier,name,ablation,case,seed in units:
            if identifier in records:continue
            budget.check();budget.checkpoint('running',episode=identifier,reserve_seconds=timeout)
            stage='episode'
            if name=='connectome':controller=BrainRunner(blank,groups,sides,seed)
            elif name in policies:controller=LearnedController(policies[name],seed,deterministic=True)
            elif name=='untrained':controller=LearnedController(Policy('feedforward',991),seed,deterministic=True)
            else:controller=name
            row,arrays,events=run_episode(case,seed,controller,ablation=ablation,timeout_seconds=timeout,
                                          stop_check=budget.check)
            stage='record_persistence'
            metadata=rawdir/(identifier+'.json');trajectory=rawdir/(identifier+'.npz')
            temporary=trajectory.with_suffix('.npz.partial')
            with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
            temporary.replace(trajectory)
            row.update(controller=name,events=events,protocol_sha256=config_digest(lock),
                       trajectory=trajectory.name,trajectory_sha256=sha256_file(trajectory))
            atomic_json(metadata,row);records[identifier]=row
            budget.checkpoint('running')
            print(json.dumps({'controller':name,'ablation':ablation,'scenario':case,'seed':seed,'status':row['status'],'food':row['food']}),flush=True)
            del controller;gc.collect()
            atomic_json(output/'progress.json',{'recorded_cases':len(records),
                        'completed_cases':sum(record['status']=='completed' for record in records.values()),
                        'last_controller':name,'last_ablation':ablation,'cumulative_wall_seconds':budget.elapsed})
            if row['status']=='interrupted':
                raise EvaluationStopped(row['failure']['stop_status'],row['failure']['reason'])
        budget.check()
        outcome='completed' if len(records)==len(units) and all(row['status']=='completed' for row in records.values()) else 'incomplete'
    except EvaluationStopped as error:
        outcome=error.status;stop={'kind':type(error).__name__,'reason':error.reason,'stage':stage}
    except KeyboardInterrupt:
        outcome='cancelled';stop={'kind':'KeyboardInterrupt','reason':'keyboard_interrupt','stage':stage}
    except Exception as error:
        outcome='failed';stop={'kind':type(error).__name__,'stage':stage}
    index=[records[unit[0]] for unit in units if unit[0] in records]
    interim='summarizing' if outcome in {'completed','incomplete'} else outcome
    budget.checkpoint(interim,reserve_seconds=600 if interim=='summarizing' else 0)
    atomic_json(output/'results.json',{'schema':'neuroterrarium.evaluation-results.v1','protocol_sha256':config_digest(lock),
                                    'status':interim,'records':index,'expected_records':len(units),
                                    'missing_records':len(units)-len(index),'stop':stop,'wall_seconds':budget.elapsed})
    if outcome in {'completed','incomplete'}:
        try:
            budget.check();summarize(output);budget.check()
        except EvaluationStopped as error:
            outcome=error.status;stop={'kind':type(error).__name__,'reason':error.reason,'stage':'summary'}
        except KeyboardInterrupt:
            outcome='cancelled';stop={'kind':'KeyboardInterrupt','reason':'keyboard_interrupt','stage':'summary'}
        except Exception as error:
            outcome='failed';stop={'kind':type(error).__name__,'stage':'summary'}
    budget.checkpoint(outcome)
    final_elapsed=json.loads(budget.path.read_text())['elapsed_seconds']
    atomic_json(output/'results.json',{'schema':'neuroterrarium.evaluation-results.v1','protocol_sha256':config_digest(lock),
                                    'status':outcome,'records':index,'expected_records':len(units),
                                    'missing_records':len(units)-len(index),'stop':stop,'wall_seconds':final_elapsed})
    return {'status':outcome,'records':len(index),'missing_records':len(units)-len(index),
            'cumulative_wall_seconds':final_elapsed,'stop':stop,'output':str(output)}


def summarize(output_dir,*,persist=True,stop_check=None):
    output=Path(output_dir);data=json.loads((output/'results.json').read_text());rows=data['records']
    identities=set()
    for row in rows:
        if stop_check is not None:stop_check()
        identity=tuple(row[key] for key in ('scenario','controller','ablation','environment_seed','neural_repeat'))
        if identity in identities:raise ValueError('Duplicate experiment unit')
        identities.add(identity)
        if row['scenario'] not in SCENARIOS or row['neural_repeat']!=0 or row['status'] not in {'completed','failed','timeout','interrupted'}:
            raise ValueError('Invalid experiment identity or status')
        if 'protocol_sha256' in data and row.get('protocol_sha256')!=data['protocol_sha256']:
            raise ValueError('Experiment row protocol mismatch')
        if 'trajectory' in row:
            name=row['trajectory']
            if Path(name).name!=name:raise ValueError('Trajectory path must be a simple asset name')
            path=output/'raw'/name
            if not path.is_file() or sha256_file(path)!=row['trajectory_sha256']:
                raise ValueError('Raw trajectory checksum mismatch')
            with np.load(path,allow_pickle=False) as saved:
                calculated=recompute_metrics(saved,row['events'],status=row['status'])
            for key,value in calculated.items():
                recorded=row.get(key)
                if value is None:
                    if recorded is not None:raise ValueError('Undefined outcome mismatch: '+key)
                elif recorded is None or not math.isclose(value,recorded,rel_tol=1e-10,abs_tol=1e-10):
                    raise ValueError('Outcome does not match raw trajectory: '+key)
    metrics=('food','collision_substeps','motor_energy_demand','final_simulated_energy',
             'escape_drive_integral','inference_mean_ms','reaction_seconds','recovery_seconds')
    summaries=[]
    combinations=sorted({(row['scenario'],row['controller'],row['ablation']) for row in rows})
    for case,controller,ablation in combinations:
        if stop_check is not None:stop_check()
        subset=[row for row in rows if (row['scenario'],row['controller'],row['ablation'])==(case,controller,ablation)]
        valid=[row for row in subset if row['status']=='completed']
        item={'scenario':case,'controller':controller,'ablation':ablation,'environment_units':len(subset),
              'completed':len(valid),'status_counts':{status:sum(row['status']==status for row in subset)
                                                   for status in sorted({row['status'] for row in subset})},
              'response_fraction':sum(row['reaction_status']=='responded' for row in valid)/len(valid) if valid and case.startswith('A_') else None,
              'preactivated_fraction':sum(row.get('preactivated') is True for row in valid)/len(valid) if valid and case.startswith('A_') else None}
        for metric in metrics:
            values=np.asarray([row[metric] for row in valid if row.get(metric) is not None],dtype=float)
            if not np.isfinite(values).all():raise ValueError('Nonfinite defined outcome')
            rng=stream(11071,f'summary/{case}/{controller}/{ablation}/{metric}')
            if len(values):
                boot=rng.choice(values,(2000,len(values)),replace=True).mean(axis=1)
                item[metric]={'mean':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'defined_units':len(values)}
            else:item[metric]={'mean':None,'ci95':None,'defined_units':0}
        summaries.append(item)

    def matched(case,names,ablations,metric):
        lookups=[{row['environment_seed']:row for row in rows if row['scenario']==case and row['controller']==name and row['ablation']==ablation}
                 for name,ablation in zip(names,ablations,strict=True)]
        all_seeds=sorted(set().union(*(set(lookup) for lookup in lookups)))
        units=[];excluded=[]
        for seed in all_seeds:
            reasons=[]
            for name,ablation,lookup in zip(names,ablations,lookups,strict=True):
                row=lookup.get(seed)
                reason='missing_record' if row is None else row['status'] if row['status']!='completed' else 'undefined_metric' if row.get(metric) is None else None
                if reason:reasons.append({'controller':name,'ablation':ablation,'reason':reason})
            if reasons:excluded.append({'environment_seed':seed,'reasons':reasons})
            else:units.append(seed)
        return lookups,units,excluded

    pairs=[];ablation_pairs=[]
    for case in SCENARIOS:
        if stop_check is not None:stop_check()
        for architecture in ('feedforward','recurrent','hybrid'):
            models=sorted({row['controller'] for row in rows if row['scenario']==case and row['controller'].startswith(architecture+'-') and row['ablation']=='none'})
            if not models:continue
            for metric in ('food','collision_substeps','motor_energy_demand'):
                lookups,units,excluded=matched(case,['connectome',*models],['none']*(len(models)+1),metric)
                item={'scenario':case,'architecture':architecture,'metric':metric,'training_seeds':len(models),
                      'environment_units':len(units),'excluded_environment_units':len(excluded),'exclusions':excluded,
                      'status':'completed' if len(models)==3 and units else 'incomplete',
                      'mean_difference_vs_connectome':None,'hierarchical_paired_ci95':None}
                if item['status']=='completed':
                    values=np.asarray([[lookup[seed][metric]-lookups[0][seed][metric] for seed in units] for lookup in lookups[1:]])
                    rng=stream(11071,f'architecture/{case}/{architecture}/{metric}');boots=[]
                    for _ in range(2000):
                        model_indices=rng.integers(0,3,3);environment_indices=rng.integers(0,len(units),len(units))
                        boots.append(values[model_indices][:,environment_indices].mean())
                    item.update(mean_difference_vs_connectome=float(values.mean()),hierarchical_paired_ci95=np.quantile(boots,[.025,.975]).tolist())
                pairs.append(item)
        for selected_case,controller,ablation in combinations:
            if selected_case!=case or ablation=='none':continue
            for metric in ('food','collision_substeps','motor_energy_demand','escape_drive_integral'):
                lookups,units,excluded=matched(case,[controller,controller],['none',ablation],metric)
                item={'scenario':case,'controller':controller,'ablation':ablation,'metric':metric,
                      'environment_units':len(units),'excluded_environment_units':len(excluded),'exclusions':excluded,
                      'mean_change':None,'paired_ci95':None,'status':'completed' if units else 'incomplete'}
                if units:
                    values=np.asarray([lookups[1][seed][metric]-lookups[0][seed][metric] for seed in units])
                    rng=stream(11071,f'ablation/{case}/{controller}/{ablation}/{metric}')
                    boot=rng.choice(values,(2000,len(values)),replace=True).mean(axis=1)
                    item.update(mean_change=float(values.mean()),paired_ci95=np.quantile(boot,[.025,.975]).tolist())
                ablation_pairs.append(item)
    report={'units':'environment seeds paired across controllers; exactly three training seeds per learned architecture',
        'reaction_interval':'conditional on responding in completed trials; response fraction and completion counts reported separately',
        'single_controller':summaries,'hierarchical_paired_differences':pairs,'paired_ablations':ablation_pairs,
        'limitations':['Simplified sensory and motor interfaces; no organism-level validation.',
                      'One neural stochastic repetition per environment; environmental and input-stream variation are not separated.',
                      'Intervals use complete matched environments; excluded seeds and reasons are explicit and may be nonrandom.',
                      'Only three training seeds; uncertainty in architecture effects is weakly resolved.',
                      'C_unseen tests one new three-obstacle arrangement, not general out-of-distribution ability.',
                      'Motor energy demand follows the action-cost formula. Actual physical-threat debit at each body substep is not recorded.',
                      'Inference costs are wall time; response and recovery latencies are simulation time.']}
    if persist:atomic_json(output/'summary.json',report)
    return report
