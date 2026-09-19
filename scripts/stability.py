"""Exercise the real ten-controller local service for a measured wall-clock hour."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import psutil
import requests
import torch
import uvicorn

from neuroterrarium.data import sha256_file
from neuroterrarium.runtime import NEURAL_STEPS, Session
from neuroterrarium.service import create_app
from neuroterrarium.training import atomic_json
from neuroterrarium.world import ACTION_DT

REQUIRED_OPERATIONS={'resume','pause','step','food_add','stimulus_add','stimulus_remove',
                     'channels','restore_interventions','neural_disconnect','readout_clamp',
                     'sham','noise','snapshot_save','snapshot_restore'}


class WindowCounter:
    """Count committed controller windows independently of restorable world time."""
    def __init__(self,journal):
        self.windows=0
        original=journal.advanced
        def advanced(session,command=None):
            if (len(session.world.bodies)!=10 or session.allocation.count('connectome')!=1
                    or session.brain.step!=session.world.step*NEURAL_STEPS
                    or len(session.records[-1]['decisions'])!=10):
                raise RuntimeError('A recorded window did not advance the complete ten-controller world')
            original(session,command)
            self.windows+=1
        journal.advanced=advanced


def count_journal_windows(path):
    count=0
    identity_seen=False
    with Path(path).open(encoding='utf-8') as handle:
        for sequence,line in enumerate(handle):
            if len(line)>4*2**20:raise ValueError('Oversized execution journal line')
            item=json.loads(line)
            if item.get('sequence')!=sequence:raise ValueError('Journal sequence mismatch')
            if sequence==0 and item.get('schema')!='neuroterrarium.execution.v1':
                raise ValueError('Journal identity mismatch')
            if sequence==0:identity_seen=True
            if 'record' in item:
                if item['type'] not in {'advance','command'} or len(item['record']['decisions'])!=10:
                    raise ValueError('Invalid completed-window journal entry')
                count+=1
    if not identity_seen:raise ValueError('Missing execution journal identity')
    return count


def executable_snapshot(payload):
    """Exclude only wall-cost, display and historical-log fields from comparison."""
    result={key:value for key,value in payload.items()
            if key not in {'wall_time','compute_seconds','events','records','revealed','fork'}}
    result['fork']=executable_snapshot(payload['fork']) if payload['fork'] is not None else None
    return result


def hour_gate(summary):
    """A short, stalled or replay-only run cannot become an hour pass."""
    names={entry.get('operation',entry.get('command',{}).get('type')) for entry in summary['operations']}
    measured=summary['measured_wall_seconds']
    return bool(summary['status']=='completed' and summary['requested_wall_seconds']>=3600
                and measured>=3600 and summary['resource_samples']>=measured/5
                and summary['active_wall_seconds_estimate']>=.9*measured
                and summary['maximum_unpaused_progress_gap_seconds']<=30
                and summary['committed_action_windows']>0
                and summary['journal_action_windows']==summary['committed_action_windows']
                and REQUIRED_OPERATIONS<=names)


def run(data_dir,models_dir,output,duration=3600,port=8765):
    if type(duration) not in (int,float) or not math.isfinite(duration) or not 0<duration<=86400:
        raise ValueError('Duration must be finite, positive and at most one day')
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('Invalid loopback port')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'stability.json').exists():raise ValueError('Existing stability evidence must be preserved')
    torch.set_num_threads(1)
    session=Session(data_dir,models_dir,seed=519,scenario='open')
    session.paused=True
    app=create_app(session,port=port,record_dir=output/'execution')
    counter=WindowCounter(app.state.controller.journal)
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning',access_log=False))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    http=requests.Session();http.trust_env=False
    base=f'http://127.0.0.1:{port}'
    status='running';operations=[];samples=[];error=None
    summary={'schema':'neuroterrarium.stability.v1','status':status,'requested_wall_seconds':duration,
             'behavior_sha256':session.behavior_sha256,'script_sha256':sha256_file(Path(__file__)),
             'models_sha256':sha256_file(Path(models_dir)/'manifest.json'),
             'mode':'Local full graph','controllers':sorted(session.allocation)}
    start=time.monotonic();last_slot=-1;saved=None;poll=0
    measured=0.;paused_wall=0.;last_sample_time=0.;was_paused=False
    last_count=0;last_progress=0.;maximum_gap=0.;journal_windows=None
    def command(data):
        reply=http.post(base+'/api/command',json=data,timeout=30)
        if reply.status_code!=200:raise RuntimeError(f'Command {data["type"]} failed: HTTP {reply.status_code}')
        operations.append({'wall_seconds':time.monotonic()-start,'command':data,'status_code':reply.status_code})
    try:
        while not server.started:
            if time.monotonic()-start>20:raise RuntimeError('Local service did not start')
            time.sleep(.1)
        start=time.monotonic()
        command({'type':'resume'})
        with (output/'resources.jsonl').open('x') as log:
            while time.monotonic()-start<duration:
                if (output/'cancel').exists():status='cancelled';break
                response=http.get(base+'/api/state',params={'selected':poll%10,'reveal':'true'},timeout=30)
                if response.status_code!=200:raise RuntimeError(f'Backend state failed: HTTP {response.status_code}')
                state=response.json()
                if state['selected']['index']!=poll%10:raise RuntimeError('Selected inspector mismatch')
                now=time.monotonic()-start;sim=state['simulation_time']
                with app.state.controller.lock:completed=counter.windows
                if state['mode']!='Local full graph' or not state['ready'] or len(state['world']['bodies'])!=10:
                    raise RuntimeError('Service is not running the complete local session')
                if was_paused or state['paused']:paused_wall+=now-last_sample_time
                if completed>last_count or state['paused']:last_progress=now
                else:maximum_gap=max(maximum_gap,now-last_progress)
                if not state['paused'] and now-last_progress>30:raise RuntimeError('No complete controller window for 30 unpaused seconds')
                last_count=completed;last_sample_time=now;was_paused=state['paused']
                memory=psutil.Process().memory_info().rss
                sample={'wall_seconds':now,'simulation_time':sim,'cumulative_simulation_seconds':completed*ACTION_DT,
                        'committed_action_windows':completed,
                        'rss_bytes':memory,'available_memory_bytes':psutil.virtual_memory().available,
                        'free_disk_bytes':psutil.disk_usage(output).free,'paused':state['paused'],
                        'recording_buffer_bytes':app.state.controller.frames.bytes}
                log.write(json.dumps(sample)+'\n');log.flush();samples.append(sample)
                if sample['available_memory_bytes']<3*2**30:raise RuntimeError('Less than 3 GiB system memory available')
                if sample['free_disk_bytes']<10*2**30:raise RuntimeError('Less than 10 GiB free disk')
                # Every 30 seconds exercise a different real operation. The
                # first three minutes remain available for direct UI play.
                slot=int(max(0,now-180)//30) if now>=180 else -1
                if slot!=last_slot:
                    last_slot=slot;phase=slot%14
                    if phase==0:command({'type':'food_add','x':22,'y':18})
                    elif phase==1:command({'type':'stimulus_add','x':40,'y':30,'radius':.7,'growth':.08,'physical':False})
                    elif phase==2:command({'type':'channels','channel':'vision','enabled':False})
                    elif phase==3:
                        command({'type':'restore_interventions'})
                        if state['world']['stimuli']:command({'type':'stimulus_remove','index':0})
                    elif phase==4:
                        command({'type':'pause'})
                        snapshot=http.get(base+'/api/snapshot',timeout=30)
                        if snapshot.status_code!=200:raise RuntimeError('Snapshot save failed')
                        saved=snapshot.content
                        path=output/f'snapshot-{slot:04d}.json'
                        temporary=path.with_suffix('.partial');temporary.write_bytes(saved);temporary.replace(path)
                        operations.append({'wall_seconds':time.monotonic()-start,'operation':'snapshot_save',
                                           'snapshot':path.name,'sha256':sha256_file(path),'bytes':len(saved),'status_code':200})
                        command({'type':'step'});command({'type':'resume'})
                    elif phase==5 and saved:
                        restored=http.post(base+'/api/snapshot',data=saved,headers={'Content-Type':'application/json'},timeout=30)
                        if restored.status_code!=200:raise RuntimeError('Snapshot restore failed')
                        check=http.get(base+'/api/snapshot',timeout=30)
                        if check.status_code!=200:raise RuntimeError('Restored state unavailable')
                        before=json.loads(saved)['state'];after=check.json()['state']
                        if executable_snapshot(before)!=executable_snapshot(after):
                            raise RuntimeError('Snapshot restoration changed executable state')
                        operations.append({'wall_seconds':time.monotonic()-start,'operation':'snapshot_restore',
                                           'status_code':200,'exact_executable_state_match':True})
                        command({'type':'resume'})
                    elif phase==6:command({'type':'neural_disconnect','group':'sugar'})
                    elif phase==7:command({'type':'restore_interventions'})
                    elif phase==8:command({'type':'readout_clamp','value':True})
                    elif phase==9:command({'type':'restore_interventions'})
                    elif phase==10:command({'type':'sham'})
                    elif phase==11:command({'type':'noise','value':.02})
                    elif phase==12:command({'type':'restore_interventions'})
                    elif phase==13:
                        # Resource renewal is part of play, not the fixed
                        # termination rule used by formal experiments.
                        if state['world']['stimuli']:command({'type':'stimulus_remove','index':0})
                poll+=1
                if poll%30==0:
                    atomic_json(output/'progress.json',{**sample,'operations':len(operations),'status':'running'})
                    print(json.dumps(sample),flush=True)
                time.sleep(1)
            else:status='completed'
        command({'type':'pause'})
        measured=time.monotonic()-start
        recording=http.get(base+'/api/recording',timeout=30)
        if recording.status_code!=200:raise RuntimeError('Final recording unavailable')
        (output/'representative-replay.json').write_bytes(recording.content)
    except KeyboardInterrupt:
        status='cancelled';error={'type':'KeyboardInterrupt','message':'Run cancelled before the hour gate'}
    except Exception as exc:
        status='failed';error={'type':type(exc).__name__,'message':str(exc)}
    finally:
        if not measured:measured=time.monotonic()-start
        server.should_exit=True;thread.join(timeout=30);http.close()
        if thread.is_alive():status='failed';error={'type':'ShutdownTimeout','message':'Local service did not stop'}
    try:
        journal=output/'execution'/'execution.jsonl'
        journal_windows=count_journal_windows(journal)
        summary['execution_journal_sha256']=sha256_file(journal)
        if journal_windows!=counter.windows:raise RuntimeError('Journal and committed controller-window counts disagree')
    except Exception as exc:
        status='failed';error={'type':type(exc).__name__,'message':'Execution journal verification failed'}
    after_warm=[s for s in samples if s['wall_seconds']>=min(300,duration/2)]
    slope=float(np.polyfit([s['wall_seconds'] for s in after_warm],[s['rss_bytes'] for s in after_warm],1)[0]) if len(after_warm)>2 else None
    advanced=counter.windows*ACTION_DT
    elapsed=time.monotonic()-start
    summary.update(status=status,actual_wall_seconds=elapsed,measured_wall_seconds=measured,
                   active_wall_seconds_estimate=max(0,measured-paused_wall),
                   active_time_estimator='one-second state samples; intervals touching a paused sample are counted paused',
                   maximum_unpaused_progress_gap_seconds=maximum_gap,
                   committed_action_windows=counter.windows,journal_action_windows=journal_windows,
                   cumulative_simulation_seconds=advanced,
                   simulation_per_wall=advanced/measured if measured else 0,
                   peak_sampled_rss_bytes=max((s['rss_bytes'] for s in samples),default=None),
                   post_warmup_memory_slope_bytes_per_second=slope,resource_samples=len(samples),
                   operations=operations,error=error)
    summary['hour_gate_passed']=hour_gate(summary)
    atomic_json(output/'stability.json',summary)
    return 0 if summary['hour_gate_passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True);parser.add_argument('--models',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--duration',type=float,default=3600)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    raise SystemExit(run(args.data_dir,args.models,args.output,args.duration,args.port))
