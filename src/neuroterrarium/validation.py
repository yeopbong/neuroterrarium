"""Bounded validation against the complete data profile and nine trained models."""
from __future__ import annotations

import gc
from pathlib import Path
import tempfile
import time

import numpy as np
import psutil
import torch

from .data import sha256_file
from .recording import ExecutionJournal, recompute
from .runtime import Session, _wire_digest
from .training import atomic_json
from .world import Food, Stimulus


def dynamics_digest(session):
    state=session._payload()
    for key in ('wall_time','compute_seconds','events','fork','paused'):
        state.pop(key,None)
    return _wire_digest(state)


def validate(data_dir,models_dir,output_dir=None):
    """Actual integration checks, not a substitute for the full release gate."""
    torch.set_num_threads(1)
    output=Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix='neuroterrarium-validation-'))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'validation.json').exists():raise ValueError('Choose a new validation output directory; old evidence is preserved')
    started=time.perf_counter();checks=[]
    result={'schema':'neuroterrarium.runtime-validation.v1','status':'running',
            'scope':'Real complete graph and nine independently trained policies; bounded local integration',
            'checks':checks}
    def check(name,passed,**evidence):
        checks.append({'name':name,'status':'passed' if passed else 'failed',**evidence})
        atomic_json(output/'validation.json',result)
        if not passed:raise RuntimeError(f'Validation failed: {name}')
    try:
        session=Session(data_dir,models_dir,seed=407)
        result['graph']=session.graph.summary
        result['behavior_sha256']=session.behavior_sha256
        result['validation_source_sha256']=sha256_file(Path(__file__))
        result['model_manifest_sha256']=sha256_file(Path(models_dir)/'manifest.json')
        check('verified_full_profile_and_nine_models',session.mode=='Local full graph' and len(set(session.allocation))==10,
              neurons=session.brain.n_neurons,controllers=sorted(session.allocation))
        index=session.allocation.index('connectome');body=session.world.bodies[index]
        session.world.foods=[Food(body.x,body.y)]
        session.world.stimuli=[]
        initial=session.snapshot()
        normal=0.0
        for _ in range(25):
            session.step();normal+=session._last_rates['mn9']
        food=session.world.bodies[index].food
        check('neural_firing_drives_physical_feeding',normal>0 and food>0,mn9_window_hz_sum=normal,consumed_food=food)
        saved=session.snapshot()
        check('persistent_nonresting_state',session.brain.step==5000 and np.any(session.brain.v_mV!=-52),
              neural_steps=session.brain.step,queued_events=sum(len(q) for q in session.brain.snapshot()['queue']))
        for _ in range(3):session.step()
        expected=dynamics_digest(session)
        session.restore(saved)
        for _ in range(3):session.step()
        check('snapshot_recomputation_exact',dynamics_digest(session)==expected)
        session.restore(initial)
        session.execute({'type':'neural_disconnect','group':'sugar'})
        cut=0.0
        for _ in range(25):session.step();cut+=session._last_rates['mn9']
        check('positive_control_output_disconnection',cut<normal,
              normal_mn9_window_hz_sum=normal,cut_mn9_window_hz_sum=cut)
        session.restore(saved)
        session.execute({'type':'fork','replacement':'same'})
        session.execute({'type':'sham','branch':'right'})
        for _ in range(3):session.step()
        check('same_snapshot_sham_branch_exact',dynamics_digest(session)==dynamics_digest(session.fork_session))
        session.execute({'type':'fork_close'})
        session.execute({'type':'readout_clamp','value':True})
        speeds=[]
        for _ in range(5):
            session.step();speeds.append(session.world.bodies[index].speed)
            if np.any(session._last_actions[index]!=0):raise RuntimeError('Clamped motor readout produced an action')
        check('clamped_readout_has_no_new_motor_commands',True,residual_inertial_speeds=speeds)
        session.execute({'type':'restore_interventions'})
        body=session.world.bodies[index]
        session.world.stimuli=[Stimulus(body.x+3,body.y,.8,growth=4)]
        for _ in range(3):session.step()
        before=dict(session.encoder.last_rates_hz)
        session.execute({'type':'stimulus_remove','index':0})
        session.step()
        after=dict(session.encoder.last_rates_hz)
        check('stimulus_removal_changes_delivered_input',after['lplc2']==0 and after['lc4']==0 and before['lplc2']>0,
              before_hz=before,after_hz=after)
        session.execute({'type':'channels','channel':'vision','enabled':False})
        session.step()
        check('shared_sensory_mask_reaches_all_controllers',not np.any(session._last_observations[:,16:48]))
        session.execute({'type':'restore_interventions'})
        for i in range(10):
            panel=session.state(i,True)['selected']
            if panel['observation']!=session.records[-1]['observations'][i]:raise RuntimeError('Inspector differs from delivered observation')
        check('ten_actual_inspector_inputs',True)
        journal=ExecutionJournal(output/'recomputation',session)
        try:
            for _ in range(4):session.step();journal.advanced(session)
            command={'type':'noise','value':.03};session.execute(command);journal.write({'type':'command','command':command})
            for _ in range(3):session.step();journal.advanced(session)
        finally:journal.close()
        recomputed=recompute(session,output/'recomputation')
        check('journal_controller_recomputation',recomputed['action_windows']==7,**{k:v for k,v in recomputed.items() if k!='status'})
        result['status']='passed'
        result['process_rss_bytes_at_end']=psutil.Process().memory_info().rss
        del session;gc.collect()
    except Exception as exc:
        result.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    result['wall_seconds']=time.perf_counter()-started
    atomic_json(output/'validation.json',result)
    return result
