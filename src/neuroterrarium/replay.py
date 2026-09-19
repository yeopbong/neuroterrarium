"""Serve an inspected recording as playback; no controller execution is implied."""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .world import World


def _number(value,lo,hi,label):
    if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:
        raise ValueError(f'Invalid recorded {label}')


def _action(values):
    if not isinstance(values,list) or len(values)!=4:raise ValueError('Invalid recorded action')
    for v,lo in zip(values,[0,-1,0,0],strict=True):_number(v,lo,1,'action')


def _frame(value,depth=0):
    if depth>1 or not isinstance(value,dict) or not isinstance(value.get('world'),dict):raise ValueError('Invalid recording frame or nested fork')
    world=value['world']
    for key in ('simulation_time','wall_time','rate'):_number(value.get(key),0,2e7,key)
    if world.get('width')!=80 or world.get('height')!=56:raise ValueError('Invalid recorded world dimensions')
    if not isinstance(world.get('bodies'),list) or len(world['bodies'])!=10:raise ValueError('Replay requires ten recorded bodies')
    # Reuse the authoritative structured-world checks without creating or
    # running any controller. Recordings do not contain executable code.
    try:
        World.restore({'schema':'world-v1','seed':0,'scenario':value.get('scenario'),
            'step':round(value['simulation_time']/.02),'bodies':world['bodies'],
            **{key:world.get(key) for key in ('foods','obstacles','stimuli')},
            'external_events':[],'previous_projection':[[0.]*16 for _ in range(10)],'replenish':False})
    except (TypeError,KeyError) as exc:raise ValueError('Invalid recorded world') from exc
    def panel(p):
        if not isinstance(p,dict) or type(p.get('index')) is not int or not 0<=p['index']<10:raise ValueError('Invalid recorded inspector index')
        if p.get('body')!=world['bodies'][p['index']]:raise ValueError('Recorded inspector body mismatch')
        _action(p.get('action'))
        if p['action']!=p['body']['action']:raise ValueError('Recorded inspector action mismatch')
        observation=p.get('observation')
        if not isinstance(observation,list) or len(observation)!=59:raise ValueError('Invalid recorded observation shape')
        for index,v in enumerate(observation):_number(v,-1 if 32<=index<48 or index==56 else 0,1,'observation')
        if 'observation_time' in p:_number(p['observation_time'],0,value['simulation_time'],'observation time')
        if p.get('kind') not in (None,'connectome','feedforward','recurrent','hybrid'):raise ValueError('Invalid recorded controller kind')
        if not isinstance(p.get('label'),str) or len(p['label'])>64:raise ValueError('Invalid inspector label')
        for key in ('neural_rates_hz','neural_inputs_hz'):
            if key in p:
                if not isinstance(p[key],dict) or len(p[key])>16:raise ValueError('Invalid recorded neural groups')
                for rate in p[key].values():_number(rate,0,10000,'neural rate')
        if 'hidden' in p:
            if not isinstance(p['hidden'],list) or len(p['hidden'])>1024:raise ValueError('Invalid hidden-state shape')
            for v in p['hidden']:_number(v,-1,1,'hidden state')
    panel(value.get('selected'))
    if 'panels' in value:
        if not isinstance(value['panels'],list) or len(value['panels'])!=10:raise ValueError('Invalid recorded panels')
        for index,p in enumerate(value['panels']):
            panel(p)
            if p['index']!=index:raise ValueError('Recorded panel ordering mismatch')
    if not isinstance(value.get('events'),list) or len(value['events'])>4096:raise ValueError('Invalid recorded events')
    for event in value['events']:
        if not isinstance(event,dict) or not isinstance(event.get('type'),str) or len(event['type'])>64:raise ValueError('Invalid recorded event')
        _number(event.get('time'),0,value['simulation_time'],'event time')
    if 'fork' in value:_frame(value['fork'],depth+1)


def load_recording(path):
    path=Path(path)
    if path.stat().st_size>32*2**20:raise ValueError('Recording exceeds 32 MiB')
    data=json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite recording value')))
    if not isinstance(data,dict) or data.get('schema')!='neuroterrarium.frame-replay.v1' or data.get('mode')!='Replay':
        raise ValueError('Expected an explicitly labeled frame replay')
    frames=data.get('frames')
    if not isinstance(frames,list) or not 1<=len(frames)<=3000:raise ValueError('Recording must contain 1–3000 frames')
    for value in frames:_frame(value)
    return data


def create_replay_app(recording,*,port=8765,web_dir=None):
    data=load_recording(recording)
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    allowed={f'localhost:{port}',f'127.0.0.1:{port}'}
    @app.middleware('http')
    async def boundaries(request:Request,call_next):
        if request.headers.get('host') not in allowed or (request.headers.get('origin') and request.headers['origin'] not in {f'http://{h}' for h in allowed}):
            return JSONResponse({'detail':'Invalid Host or Origin'},status_code=403)
        if request.method not in ('GET','HEAD'):return JSONResponse({'detail':'Replay is read-only'},status_code=405)
        response=await call_next(request)
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"
        response.headers['X-Content-Type-Options']='nosniff'
        return response
    @app.get('/')
    def index():return RedirectResponse('/index.html?replay=recording.json')
    @app.get('/recording.json')
    def content():return data
    if web_dir is None:
        candidate=Path(__file__).resolve().parents[2]/'web/dist'
        web_dir=candidate if candidate.is_dir() else Path(__file__).parent/'static'
    if not Path(web_dir).is_dir():raise FileNotFoundError('Built application assets missing')
    app.mount('/',StaticFiles(directory=web_dir,html=True))
    return app


def serve_replay(recording,port=8765,open_browser=False):
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('Invalid local port')
    import uvicorn
    app=create_replay_app(recording,port=port)
    if open_browser:
        import webbrowser
        threading.Timer(1.0,lambda:webbrowser.open(f'http://127.0.0.1:{port}')).start()
    uvicorn.run(app,host='127.0.0.1',port=port,log_level='warning',access_log=False)
