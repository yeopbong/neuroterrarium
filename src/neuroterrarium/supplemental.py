"""Frozen short stress and shared-population protocols, separate from core tests."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from .controllers import LearnedController, named_seed
from .data import sha256_file
from .evaluation import (BrainRunner, EvaluationBudget, EvaluationStopped,
                         evaluation_source_digest, recompute_metrics)
from .interface import SensoryEncoder
from .registry import Registry, default_path
from .runtime import load_model_set
from .training import atomic_json, config_digest
from .world import ACTION_DT, Body, Food, Obstacle, Stimulus, World, stream

GROUPS = ('sugar', 'lplc2', 'lc4', 'mn9', 'gf', 'dna01', 'dna02')
METRICS = ('food', 'collision_substeps', 'motor_energy_demand',
           'final_simulated_energy', 'escape_drive_integral', 'inference_mean_ms')


def validate_protocol(config):
    if config.get('schema') != 'neuroterrarium.supplement.v1' or config.get('status') != 'frozen':
        raise ValueError('A frozen supplemental protocol is required')
    for key, start in (('stress_seeds', 190001), ('shared_seeds', 290001)):
        if config.get(key) != list(range(start, start + 30)) or any(type(s) is not int for s in config[key]):
            raise ValueError('Supplemental seeds do not match the predefined independent split')
    expected = {'stress_windows': 75, 'shared_windows': 100, 'visual_noise_std': .15,
                'conditions': ['clean', 'visual_noise'], 'neural_repeats': 1,
                'episode_timeout_seconds': 45,
                'memory_reserve_bytes': 3 * 2**30, 'disk_reserve_bytes': 10 * 2**30}
    if any(type(config.get(k)) is not type(v) or config.get(k) != v for k, v in expected.items()):
        raise ValueError('Executable supplemental settings do not match the frozen protocol')
    # v1b changes only the cumulative resource allowance after v1 exhausted it.
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


def balanced_allocation(model_ids, trial_index):
    """Three randomized cyclic blocks; every identity occupies every slot thrice."""
    names = ['connectome', *sorted(model_ids)]
    if len(names) != 10 or len(set(names)) != 10 or type(trial_index) is not int or not 0 <= trial_index < 30:
        raise ValueError('Allocation requires nine unique models and a trial index from 0 to 29')
    block, shift = divmod(trial_index, 10)
    order = np.asarray(names)
    stream(290000 + block, 'supplement/shared/allocation-block').shuffle(order)
    return np.roll(order, shift).tolist()


def make_scene(family, seed):
    if family not in {'stress', 'shared'} or type(seed) is not int:
        raise ValueError('Unknown supplemental scene or seed')
    rng = stream(seed, f'supplement/{family}/environment')
    count = 1 if family == 'stress' else 10
    world = World(seed, count)
    world.foods, world.obstacles, world.stimuli = [], [], []
    angle = float(rng.uniform(-math.pi, math.pi))
    if family == 'stress':
        world.bodies = [Body(40, 28, angle, energy=.35)]
        bearing = angle + float(rng.uniform(-.5, .5))
        distance = float(rng.uniform(2, 3.5))
        world.foods = [Food(40 + distance*math.cos(bearing), 28 + distance*math.sin(bearing), .7)]
        for delta, radius, distance in ((1.2, .8, 6), (-1.4, 1.1, 8), (2.8, .9, 10)):
            world.obstacles.append(Obstacle(40 + distance*math.cos(angle+delta), 28 + distance*math.sin(angle+delta), radius))
    else:
        # The geometric slots remain distinct from hidden controller identities.
        world.bodies = [Body(40 + 3.2*math.cos(angle+i*math.tau/10),
                             28 + 3.2*math.sin(angle+i*math.tau/10),
                             angle+i*math.tau/10+math.pi, energy=.65) for i in range(10)]
        world.foods = [Food(40, 28, 1, radius=2.7), Food(32, 28, .5), Food(48, 28, .5)]
        world.obstacles = [Obstacle(40, 36, 1.5), Obstacle(34, 22, 1), Obstacle(47, 23, 1.2)]
    stimuli = [dict(x=40+11*math.cos(angle), y=28+11*math.sin(angle), radius=.8,
                    growth=1.5, vx=0., vy=0., physical=False),
               dict(x=40-10*math.cos(angle), y=28-10*math.sin(angle), radius=1.1,
                    growth=0., vx=-2*math.sin(angle), vy=2*math.cos(angle), physical=False)]
    world.stimuli = [Stimulus(**{**item, 'growth': 0., 'vx': 0., 'vy': 0.}) for item in stimuli]
    events = {'case': 'S_'+family, 'family': family, 'environment_seed': seed,
              'onset_step': 10, 'stimuli_after_onset': stimuli, 'neural_repeat': 0,
              'neural_stream': f'supplement/{family}/neural/0',
              'noise_stream': f'supplement/{family}/body/0/visual-noise',
              'initial_world': world.snapshot(), 'physical_threat': False}
    return world, events


def run_trial(world, events, controllers, *, windows, noise_std=0., timeout_seconds=45,
              cancel=None):
    """Observe every body, compute all actions, then advance the same world once.

    Numeric NPZ channels that do not apply use NaN; outcome JSON uses null.
    Failed or timed-out trials retain only completed, synchronous windows.
    """
    count = len(world.bodies)
    if len(controllers) != count or type(windows) is not int or windows < 1 or windows > 100:
        raise ValueError('Invalid controller population or duration')
    if (type(noise_std) not in (int, float) or not math.isfinite(noise_std) or not 0 <= noise_std <= .15
            or type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds < 0):
        raise ValueError('Invalid noise amplitude or timeout')
    fields = ('observation', 'unperturbed_observation', 'applied_action', 'body', 'motor_groups_hz',
              'neural_input_hz', 'raw_policy_action', 'raw_policy_log_probability',
              'inference_seconds', 'remaining_food', 'reflex_triggered')
    history = {name: [] for name in fields}
    initial = np.asarray([[b.x, b.y, b.heading, b.speed, b.energy, b.food, b.collisions] for b in world.bodies])
    noise = [stream(events['environment_seed'], f"supplement/{events['family']}/body/{i}/visual-noise") for i in range(count)]
    events = {**events, 'noise_std': noise_std, 'allocation': events.get('allocation', []),
              'initial_total_food': float(sum(f.amount for f in world.foods))}
    start = time.perf_counter()
    status, failure, stage, rss = 'completed', None, 'resource_sampling', []
    try:
        process = psutil.Process()
        rss.append(process.memory_info().rss)
        for step in range(windows):
            if cancel is not None and cancel():
                status = 'interrupted'
                failure = {'kind': 'CancellationRequested', 'stage': 'before_window', 'completed_windows': step}
                break
            if time.perf_counter()-start >= timeout_seconds:
                status = 'timeout'
                break
            stage = 'external_event'
            if step == events['onset_step']:
                world.stimuli = [Stimulus(**item) for item in events['stimuli_after_onset']]
            stage = 'observation'
            clean = world.observe().copy()
            observation = clean.copy()
            for i, rng in enumerate(noise):
                # Draw at both amplitudes. No other named stream is touched.
                observation[i, 16:48] += noise_std*rng.normal(size=32)
            observation[:, 16:32] = np.clip(observation[:, 16:32], 0, 1)
            observation[:, 32:48] = np.clip(observation[:, 32:48], -1, 1)
            actions = np.empty((count, 4)); cost = np.zeros(count)
            raw = np.full((count, 4), np.nan); logp = np.full(count, np.nan)
            rates = np.full((count, len(GROUPS)), np.nan); inputs = rates.copy()
            reflex = np.zeros(count, dtype=bool)
            stage = 'controller'
            for i, controller in enumerate(controllers):
                tick = time.perf_counter()
                action = np.asarray(controller.act(observation[i]), dtype=np.float64)
                cost[i] = time.perf_counter()-tick
                if action.shape != (4,) or not np.isfinite(action).all() or np.any(action < [0,-1,0,0]) or np.any(action > 1):
                    raise ValueError('Invalid applied action')
                actions[i] = action
                if isinstance(controller, LearnedController):
                    raw[i], logp[i] = controller.last['raw_action'], controller.last['log_probability']
                    reflex[i] = controller.last['reflex_triggered']
                elif isinstance(controller, BrainRunner):
                    rates[i] = [controller.last_rates[name] for name in GROUPS]
                    inputs[i] = [controller.last_inputs.get(name, np.nan) for name in GROUPS]
            stage = 'world'
            world.advance(actions)
            body = [[b.x,b.y,b.heading,b.speed,b.energy,b.food,b.collisions] for b in world.bodies]
            values = (observation,clean,actions,body,rates,inputs,raw,logp,cost,
                      sum(f.amount for f in world.foods),reflex)
            for name, value in zip(fields, values, strict=True):
                history[name].append(value)
            stage = 'resource_sampling'
            rss.append(process.memory_info().rss)
    except KeyboardInterrupt:
        status = 'interrupted'
        failure = {'kind': 'KeyboardInterrupt', 'stage': stage, 'completed_windows': len(history['body'])}
    except Exception as error:
        status = 'failed'
        failure = {'kind': type(error).__name__, 'stage': stage, 'completed_windows': len(history['body'])}
    elapsed = time.perf_counter()-start
    if status == 'completed' and elapsed >= timeout_seconds:
        status = 'timeout'
    n = len(history['body'])
    shapes = {'observation': (count,59), 'unperturbed_observation': (count,59),
              'applied_action': (count,4), 'body': (count,7), 'motor_groups_hz': (count,7),
              'neural_input_hz': (count,7), 'raw_policy_action': (count,4),
              'raw_policy_log_probability': (count,), 'inference_seconds': (count,),
              'remaining_food': (), 'reflex_triggered': (count,)}
    arrays = {name: np.asarray(history[name], dtype=np.float64).reshape((n,*shape)) for name,shape in shapes.items()}
    arrays['initial_body'] = initial
    arrays['sampled_process_rss_bytes'] = np.asarray(rss, dtype=np.int64)
    metrics = trial_metrics(arrays, events, status=status)
    row = {'status': status, 'failure': failure, 'completed_windows': n,
           'wall_seconds': elapsed, 'simulation_seconds': n*ACTION_DT,
           'sampled_peak_process_rss_bytes': max(rss) if rss else None, **metrics}
    return row, arrays, events


def trial_metrics(arrays, events, *, status):
    actions, bodies = np.asarray(arrays['applied_action']), np.asarray(arrays['body'])
    if actions.ndim != 3 or actions.shape[2] != 4 or not 1 <= actions.shape[1] <= 10:
        raise ValueError('Invalid supplemental population trace')
    n, count, _ = actions.shape
    if bodies.shape != (n,count,7) or np.asarray(arrays['initial_body']).shape != (count,7):
        raise ValueError('Invalid supplemental body trace')
    if np.asarray(arrays['inference_seconds']).shape != (n,count):
        raise ValueError('Invalid supplemental timing trace')
    body_metrics = []
    for i in range(count):
        single = {key: np.asarray(arrays[key])[:,i] for key in ('applied_action','body','inference_seconds')}
        single['initial_body'] = arrays['initial_body'][i]
        body_metrics.append(recompute_metrics(single, events, status=status))
    remaining = np.asarray(arrays['remaining_food'])
    if remaining.shape != (n,) or not np.isfinite(remaining).all() or np.any(remaining < -1e-12):
        raise ValueError('Invalid remaining-food trajectory')
    consumed = bodies[:,:,5].sum(axis=1)-np.asarray(arrays['initial_body'])[:,5].sum()
    if n and not np.allclose(consumed+remaining, events['initial_total_food'], rtol=0, atol=1e-10):
        raise ValueError('Shared food conservation violated')
    complete = status == 'completed'
    return {'body_metrics': body_metrics,
            'total_food': float(consumed[-1]) if n and complete else None,
            'total_collision_substeps': sum(m['collision_substeps'] for m in body_metrics) if complete and n else None,
            'total_motor_energy_demand': sum(m['motor_energy_demand'] for m in body_metrics) if complete else None,
            'remaining_food': float(remaining[-1]) if n and complete else None,
            'physical_energy_debit': 0.0, 'physical_energy_debit_status': 'not_applicable'}


def _controller(name, policies, blank, groups, sides, seed, family):
    if name == 'connectome':
        runner = BrainRunner(blank, groups, sides, seed)
        runner.encoder = SensoryEncoder(groups, sides, named_seed(seed, f'supplement/{family}/neural/0'))
        return runner
    return LearnedController(policies[name], named_seed(seed, f'supplement/{family}/{name}/policy'), deterministic=True)


def _plan(config, policies):
    plan = []
    for name in ['connectome', *sorted(policies)]:
        for condition in config['conditions']:
            for seed in config['stress_seeds']:
                plan.append({'family': 'stress', 'condition': condition, 'environment_seed': seed,
                             'controller': name, 'allocation': [name], 'windows': config['stress_windows']})
    for i, seed in enumerate(config['shared_seeds']):
        plan.append({'family': 'shared', 'condition': 'clean', 'environment_seed': seed,
                     'controller': 'shared_population', 'allocation': balanced_allocation(policies, i),
                     'windows': config['shared_windows']})
    return plan


def _load_brain(data_dir, output, digest):
    from .brain_cache import ensure_cache, load_cache

    ensure_cache(data_dir)
    cached = load_cache(data_dir)
    registry = Registry.load()
    groups, sides = registry.resolve(cached.root_ids), registry.resolve_sides(cached.root_ids)
    blank = cached.brain
    atomic_json(output/'graph-summary.json', {'profile': cached.summary, 'neural_graph_sha256': blank.graph_digest,
                                             'protocol_sha256': digest})
    return blank, groups, sides


def _identifier(spec):
    return f"{spec['family']}--{spec['controller']}--{spec['condition']}--{spec['environment_seed']}"


def _existing_records(rawdir, plan, digest, *, stop_check=None):
    """Reject incomplete file pairs before a new run can overwrite evidence."""
    planned = {_identifier(spec): spec for spec in plan}
    if len(planned) != len(plan):
        raise ValueError('Duplicate supplemental trial identity')
    allowed = {identifier+suffix for identifier in planned for suffix in ('.json', '.npz')}
    if any(path.is_symlink() or not path.is_file() or path.name not in allowed for path in rawdir.iterdir()):
        raise ValueError('Supplement raw directory contains unplanned or partial evidence; preserve this version')
    records = {}
    for identifier, spec in planned.items():
        if stop_check is not None: stop_check()
        metadata, trajectory = rawdir/(identifier+'.json'), rawdir/(identifier+'.npz')
        if metadata.exists() != trajectory.exists():
            raise ValueError('Orphan supplemental trajectory or metadata; preserve this version')
        if not metadata.exists(): continue
        row = json.loads(metadata.read_text())
        n, wall = row.get('completed_windows'), row.get('wall_seconds')
        if (row.get('protocol_sha256') != digest or any(row.get(key) != value for key,value in spec.items())
                or row.get('trajectory') != trajectory.name or sha256_file(trajectory) != row.get('trajectory_sha256')
                or row.get('status') not in {'completed','timeout','failed','interrupted'}
                or type(n) is not int or not 0 <= n <= spec['windows']
                or row.get('simulation_seconds') != n*ACTION_DT
                or row['status']=='completed' and n != spec['windows']
                or type(wall) not in (int,float) or not math.isfinite(wall) or wall < 0):
            raise ValueError('Existing supplemental record integrity mismatch')
        records[identifier] = row
    return records


def _verify_completed(output, historical, lock, budget, cancel):
    """Verify a completed version without altering any historical file."""
    def check():
        if cancel is not None and cancel(): raise EvaluationStopped('cancelled', 'cancelled')
        if time.perf_counter()-budget.started >= budget.limits['max_total_wall_seconds']:
            raise EvaluationStopped('budget_exhausted', 'verification_wall_budget')
        if psutil.Process().memory_info().rss > budget.limits['max_process_rss_bytes']:
            raise EvaluationStopped('resource_limited', 'process_rss')
        if psutil.virtual_memory().available < budget.limits['min_available_memory_bytes']:
            raise EvaluationStopped('resource_limited', 'available_memory')
    status, stop = 'verified', None
    try:
        check()
        digest = config_digest(lock)
        wall = historical.get('wall_seconds')
        saved_budget = json.loads(budget.path.read_text())
        if (historical.get('schema') != 'neuroterrarium.supplement-results.v1'
                or historical.get('protocol_sha256') != digest or historical.get('reason') is not None
                or type(wall) not in (int,float) or not math.isfinite(wall) or wall < 0
                or saved_budget.get('status') != 'completed' or saved_budget.get('reserved_seconds') != 0
                or saved_budget.get('in_flight_episode') is not None or saved_budget['elapsed_seconds'] < wall):
            raise ValueError('Completed supplement index or budget integrity mismatch')
        records = _existing_records(output/'raw', lock['plan'], digest, stop_check=check)
        index = [records[_identifier(spec)] for spec in lock['plan'] if _identifier(spec) in records]
        if (len(index) != len(lock['plan']) or any(row['status'] != 'completed' for row in index)
                or historical.get('records') != index):
            raise ValueError('Completed supplement index disagrees with frozen raw records')
        graph = json.loads((output/'graph-summary.json').read_text())
        storage = json.loads(default_path().with_name('brain-storage-v1.json').read_text())
        if (graph.get('protocol_sha256') != digest or graph.get('neural_graph_sha256') != storage['graph_digest']
                or config_digest(graph.get('profile')) != storage['summary_sha256']):
            raise ValueError('Completed supplement graph identity mismatch')
        expected = summarize_supplement(output, persist=False, stop_check=check)
        if json.loads((output/'summary.json').read_text()) != expected:
            raise ValueError('Completed supplement summary disagrees with raw outcomes')
        check()
    except EvaluationStopped as error:
        status, stop = error.status, {'kind': type(error).__name__, 'reason': error.reason}
    except KeyboardInterrupt:
        status, stop = 'cancelled', {'kind': 'KeyboardInterrupt', 'reason': 'keyboard_interrupt'}
    return {'status': status, 'operation': 'read_only_verification', 'historical_status': 'completed',
            'records': len(lock['plan']), 'cumulative_wall_seconds': budget.previous,
            'verification_wall_seconds': time.perf_counter()-budget.started, 'stop': stop}


def evaluate_supplement(config, data_dir, models_dir, output_dir, *, cancel=None):
    invocation_started = time.perf_counter()
    validate_protocol(config)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rawdir = output/'raw'
    resultpath = output/'results.json'
    historical = json.loads(resultpath.read_text()) if resultpath.exists() else None
    completed = isinstance(historical, dict) and historical.get('status') == 'completed'
    if completed and (not rawdir.is_dir() or any(not (output/name).is_file() for name in
                         ('protocol-lock.json','budget.json','graph-summary.json','summary.json'))):
        raise ValueError('Completed supplement evidence is missing required files')
    if not completed: rawdir.mkdir(exist_ok=True)
    policies, manifest = load_model_set(models_dir)
    source = {**evaluation_source_digest(), 'supplemental.py': sha256_file(Path(__file__))}
    limits = {'max_total_wall_seconds': config['wall_budget_seconds'], 'max_process_rss_bytes': 2*2**30,
              'min_available_memory_bytes': config['memory_reserve_bytes'], 'min_free_disk_bytes': config['disk_reserve_bytes']}
    lock = {'schema': 'neuroterrarium.supplement-lock.v1', 'config': config, 'source': source,
            'model_manifest_sha256': config_digest(manifest), 'interface_sha256': sha256_file(default_path()),
            'brain_interface_sha256': sha256_file(default_path().with_name('brain-interface.json')),
            'storage_profile_sha256': sha256_file(default_path().with_name('brain-storage-v1.json')),
            'data_manifest_sha256': sha256_file(default_path().with_name('data-v783.json')),
            'resource_limits': limits, 'plan': _plan(config, policies)}
    lockpath = output/'protocol-lock.json'
    resumed = lockpath.exists()
    if resumed and json.loads(lockpath.read_text()) != lock:
        raise ValueError('Supplement source, model identity or frozen protocol changed')
    if not lockpath.exists(): atomic_json(lockpath, lock)
    digest = config_digest(lock)
    budgetpath = output/'budget.json'
    if resumed and not budgetpath.is_file():
        raise ValueError('Existing supplement is missing its cumulative execution budget')
    budget = EvaluationBudget(output, digest, limits, started=invocation_started)
    if completed: return _verify_completed(output, historical, lock, budget, cancel)
    records = _existing_records(rawdir, lock['plan'], digest)
    reason, blank, failure = None, None, None

    def stopped():
        nonlocal reason
        if reason is not None: return True
        if cancel is not None and cancel(): reason = 'cancelled'; return True
        try: budget.check()
        except EvaluationStopped as error:
            reason = {'cumulative_wall_budget':'wall_budget', 'system_available_memory':'available_memory',
                      'free_disk':'disk_reserve'}.get(error.reason, error.reason)
        return reason is not None

    def index_records():
        return [records.get(_identifier(spec), {**spec, 'protocol_sha256': digest,
                'status': 'not_started', 'reason': reason or 'execution_failed'}) for spec in lock['plan']]

    outcome = 'running'
    budget.checkpoint(outcome)
    try:
        torch.set_num_threads(1)
        for spec in lock['plan']:
            identifier = _identifier(spec)
            metadata = rawdir/(identifier+'.json'); trajectory = rawdir/(identifier+'.npz')
            if identifier in records: continue
            if stopped(): break
            if blank is None:
                budget.checkpoint('running', reserve_seconds=min(600, max(0., config['wall_budget_seconds']-budget.elapsed)))
                blank, groups, sides = _load_brain(data_dir, output, digest)
                gc.collect()
                budget.checkpoint('running')
                if stopped(): break
            remaining = config['wall_budget_seconds']-budget.elapsed
            timeout = min(config['episode_timeout_seconds'], max(0., remaining))
            budget.checkpoint('running', episode=identifier, reserve_seconds=timeout)
            world, events = make_scene(spec['family'], spec['environment_seed'])
            events['allocation'] = spec['allocation']
            controllers = [_controller(name, policies, blank, groups, sides, spec['environment_seed'], spec['family']) for name in spec['allocation']]
            row, arrays, events = run_trial(world, events, controllers, windows=spec['windows'],
                noise_std=config['visual_noise_std'] if spec['condition']=='visual_noise' else 0.,
                timeout_seconds=timeout, cancel=stopped)
            temporary = trajectory.with_suffix('.npz.partial')
            with temporary.open('wb') as handle: np.savez_compressed(handle, **arrays)
            temporary.replace(trajectory)
            row.update(**spec, events=events, protocol_sha256=digest,
                       trajectory=trajectory.name, trajectory_sha256=sha256_file(trajectory))
            atomic_json(metadata, row); records[identifier] = row
            budget.checkpoint('running')
            print(json.dumps({k: row[k] for k in ('family','controller','condition','environment_seed','status','wall_seconds')}), flush=True)
            del controllers; gc.collect()
            if row['failure'] is not None and row['failure'].get('kind') == 'KeyboardInterrupt':
                reason = 'keyboard_interrupt'; break
        stopped()
        outcome = 'completed' if reason is None and all(row['status']=='completed' for row in index_records()) else 'incomplete'
    except KeyboardInterrupt:
        outcome, reason = 'incomplete', 'keyboard_interrupt'
    except Exception as error:
        outcome, reason = 'failed', 'execution_failed'
        failure = {'kind': type(error).__name__, 'stage': 'execution'}

    index = index_records()
    interim = 'summarizing' if reason is None else outcome
    budget.checkpoint(interim, reserve_seconds=min(600, max(0., config['wall_budget_seconds']-budget.elapsed)) if interim=='summarizing' else 0)
    result = {'schema': 'neuroterrarium.supplement-results.v1', 'protocol_sha256': digest,
              'status': interim, 'reason': reason, 'failure': failure, 'wall_seconds': budget.elapsed, 'records': index}
    atomic_json(resultpath, result)
    if interim == 'summarizing':
        try:
            def summary_check():
                if stopped(): raise EvaluationStopped('incomplete', reason)
            report = summarize_supplement(output, persist=False, stop_check=summary_check)
            summary_check()
            atomic_json(output/'summary.json', report)
            summary_check()
        except EvaluationStopped:
            outcome = 'incomplete'
        except KeyboardInterrupt:
            outcome, reason = 'incomplete', 'keyboard_interrupt'
        except Exception as error:
            outcome, reason = 'failed', 'summary_failed'
            failure = {'kind': type(error).__name__, 'stage': 'summary'}
    budget.checkpoint(outcome)
    final_elapsed = json.loads(budgetpath.read_text())['elapsed_seconds']
    atomic_json(resultpath, {**result, 'status': outcome, 'reason': reason, 'failure': failure, 'wall_seconds': final_elapsed})
    return {'status': outcome, 'records': len(index)}


def summarize_supplement(output_dir, *, persist=True, stop_check=None):
    output = Path(output_dir)
    result = json.loads((output/'results.json').read_text())
    lock = json.loads((output/'protocol-lock.json').read_text())
    if result['protocol_sha256'] != config_digest(lock): raise ValueError('Supplement lock hash mismatch')
    rows = result['records']
    if len(rows) != len(lock['plan']): raise ValueError('Missing supplemental experiment units')
    all_completed = all(row['status']=='completed' for row in rows)
    if result['status'] not in {'completed','incomplete','summarizing','failed'}:
        raise ValueError('Unknown supplemental completion status')
    if result['status']=='completed' and not all_completed:
        raise ValueError('Supplement completion status contradicts trial statuses')
    for row, spec in zip(rows, lock['plan'], strict=True):
        if stop_check is not None: stop_check()
        if any(row.get(key) != value for key,value in spec.items()) or row.get('protocol_sha256') != result['protocol_sha256']:
            raise ValueError('Supplemental identity or ordering mismatch')
        if row['status'] == 'not_started':
            if row.get('reason') not in {'cancelled','wall_budget','available_memory','disk_reserve','process_rss',
                                        'keyboard_interrupt','execution_failed'}:
                raise ValueError('Missing not-started reason')
            continue
        if row['status'] not in {'completed','timeout','failed','interrupted'}: raise ValueError('Unknown trial status')
        name = row['trajectory']
        if Path(name).name != name or sha256_file(output/'raw'/name) != row['trajectory_sha256']:
            raise ValueError('Supplement trajectory checksum mismatch')
        with np.load(output/'raw'/name, allow_pickle=False) as arrays:
            count = len(spec['allocation'])
            n = len(arrays['body'])
            if n != row['completed_windows'] or row['simulation_seconds'] != n*ACTION_DT:
                raise ValueError('Supplement recorded clock mismatch')
            if n > spec['windows'] or (row['status']=='completed' and n != spec['windows']):
                raise ValueError('Completed supplement does not cover its full duration')
            if arrays['body'].shape != (n,count,7): raise ValueError('Supplement allocation shape mismatch')
            calculated = trial_metrics(arrays, row['events'], status=row['status'])
        if config_digest(calculated) != config_digest({key: row[key] for key in calculated}):
            raise ValueError('Supplement outcome does not match the raw trajectory')
    stress = [row for row in rows if row['family']=='stress']
    individual = []; contrasts = []
    names = sorted({row['controller'] for row in stress})
    lookups = {(name, condition): {r['environment_seed']: r for r in stress if r['controller']==name and r['condition']==condition}
               for name in names for condition in ('clean','visual_noise')}
    for name in names:
        for condition in ('clean','visual_noise'):
            subset = list(lookups[name,condition].values())
            valid = [r for r in subset if r['status']=='completed']
            individual.append({'controller': name, 'condition': condition, 'environment_units': len(subset),
                               'completed': len(valid), 'status_counts': {s: sum(r['status']==s for r in subset) for s in sorted({r['status'] for r in subset})},
                               'metrics': {m: float(np.mean([r['body_metrics'][0][m] for r in valid])) if valid else None for m in METRICS}})
    # Each contrast changes only visual-noise amplitude. Architectures retain
    # three independently trained models, crossed with matched environment IDs.
    for group in ['connectome','feedforward','recurrent','hybrid']:
        if stop_check is not None: stop_check()
        members = ['connectome'] if group=='connectome' else [n for n in names if n.startswith(group+'-')]
        if not members: continue
        units, exclusions = [], []
        for seed in lock['config']['stress_seeds']:
            reasons = [{'controller': n, 'condition': c, 'reason': lookups[n,c][seed]['status']}
                       for n in members for c in ('clean','visual_noise') if lookups[n,c][seed]['status']!='completed']
            if reasons: exclusions.append({'environment_seed': seed, 'reasons': reasons})
            else: units.append(seed)
        for metric in METRICS:
            if stop_check is not None: stop_check()
            values = np.asarray([[lookups[n,'visual_noise'][s]['body_metrics'][0][metric]-lookups[n,'clean'][s]['body_metrics'][0][metric] for s in units] for n in members])
            estimate, ci = None, None
            if units and len(members)==(1 if group=='connectome' else 3):
                rng = stream(390001, f'supplement/summary/{group}/{metric}')
                draws = [values[rng.integers(0,len(members),len(members))][:,rng.integers(0,len(units),len(units))].mean() for _ in range(2000)]
                estimate, ci = float(values.mean()), np.quantile(draws,[.025,.975]).tolist()
            contrasts.append({'group': group, 'training_seeds': 0 if group=='connectome' else len(members),
                              'environment_units': len(units), 'excluded_units': exclusions, 'metric': metric,
                              'mean_noise_minus_clean': estimate, 'paired_crossed_ci95': ci})
    shared = [r for r in rows if r['family']=='shared']; completed = [r for r in shared if r['status']=='completed']
    shared_summary = {'world_units': len(shared), 'completed_worlds': len(completed),
                      'status_counts': {s: sum(r['status']==s for r in shared) for s in sorted({r['status'] for r in shared})},
                      'allocation_table': [{'environment_seed': r['environment_seed'], 'allocation': r['allocation']} for r in shared],
                      'world_means': {m: float(np.mean([r[m] for r in completed])) if completed else None for m in ('total_food','total_collision_substeps','total_motor_energy_demand')},
                      'identity_descriptions': [{'controller': name, 'world_units': len(completed), 'metrics': {
                          metric: float(np.mean([r['body_metrics'][r['allocation'].index(name)][metric] for r in completed])) if completed else None for metric in METRICS}} for name in names]}
    report = {'status': ('completed' if all_completed else 'incomplete') if result['status']=='summarizing' else result['status'],
        'stress_individual': individual,
        'stress_paired_noise_changes': contrasts, 'shared_population': shared_summary,
        'limitations': ['Short 1.5 and 2 simulation-second trials; no long-term memory or ecological-success inference.',
                       'Stress measures this fixed visual-noise perturbation in one novel arrangement; it is not a tuning split.',
                       'Shared bodies interact and compete. The world is the sampling unit; ten bodies are not ten independent replicates.',
                       'Shared identity descriptions do not constitute a one-versus-nine algorithm comparison.',
                       'Only three training seeds and one neural input repetition per environment; uncertainty is limited.',
                       'Timing is measured wall time. RSS sampling can miss transient allocations.',
                       'Stimuli in this supplement are visual shadows without physical energy debit.']}
    if persist: atomic_json(output/'summary.json', report)
    return report
