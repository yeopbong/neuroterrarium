"""Loopback-only local application transport with one authoritative worker."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .world import ACTION_DT
from .recording import ExecutionJournal, FrameBuffer, frame

MAX_UPLOAD = 32*1024*1024


class Application:
    def __init__(self, session, record_dir=None):
        self.session=session
        self.lock=threading.RLock()
        self.stop=threading.Event()
        self.error=None
        self.selected=0
        self.frames=FrameBuffer()
        self.journal=ExecutionJournal(record_dir,session) if record_dir else None
        self.worker=threading.Thread(target=self.run,name='terrarium-simulation',daemon=True)

    def record_failure(self,exc):
        self.error=f'Recording failed ({type(exc).__name__}); session paused because its executable journal is incomplete.'
        self.session.paused=True

    def run(self):
        while not self.stop.is_set():
            start=time.perf_counter()
            with self.lock:
                if self.error or self.session.paused:
                    wait=.015
                else:
                    try:
                        self.session.step()
                        if self.journal:self.journal.advanced(self.session)
                        if self.session.world.step%5==0:
                            self.frames.append(frame(self.session,self.selected))
                        target=float(getattr(self.session,'speed',1.0))
                        wait=max(.001,ACTION_DT/target-(time.perf_counter()-start))
                    except Exception as exc:
                        self.error=f'{type(exc).__name__}: {exc}'
                        self.session.paused=True
                        wait=.05
            self.stop.wait(wait)


def create_app(session, *, port:int=8765, web_dir:Path|None=None, record_dir=None) -> FastAPI:
    controller=Application(session,record_dir)
    allowed_hosts={f'127.0.0.1:{port}',f'localhost:{port}',f'[::1]:{port}'}
    allowed_origins={f'http://{host}' for host in allowed_hosts}

    @asynccontextmanager
    async def lifespan(app):
        controller.worker.start()
        yield
        controller.stop.set()
        controller.worker.join(timeout=5)
        if controller.journal:controller.journal.close()

    app=FastAPI(title='NeuroTerrarium',docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
    app.state.controller=controller

    @app.middleware('http')
    async def boundaries(request:Request,call_next):
        if request.headers.get('host','') not in allowed_hosts:
            return JSONResponse({'detail':'Invalid Host'},status_code=403)
        origin=request.headers.get('origin')
        if origin and origin not in allowed_origins:
            return JSONResponse({'detail':'Invalid Origin'},status_code=403)
        if request.method in ('POST','PUT','PATCH'):
            if request.headers.get('content-type','').split(';')[0]!='application/json':
                return JSONResponse({'detail':'Structured JSON required'},status_code=415)
            try:
                size=int(request.headers.get('content-length','0'))
            except ValueError:return JSONResponse({'detail':'Invalid length'},status_code=400)
            if size<0 or size>MAX_UPLOAD:return JSONResponse({'detail':'Upload exceeds 32 MiB'},status_code=413)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        if request.url.path.startswith('/api/'):response.headers['Cache-Control']='no-store'
        return response

    async def structured(request:Request,limit:int) -> dict:
        chunks=[];total=0
        async for chunk in request.stream():
            total+=len(chunk)
            if total>limit:raise HTTPException(413,'Payload too large')
            chunks.append(chunk)
        try:
            obj=json.loads(b''.join(chunks),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
        except (ValueError,UnicodeDecodeError,RecursionError) as exc:raise HTTPException(400,'Invalid JSON') from exc
        if not isinstance(obj,dict):raise HTTPException(400,'Expected a JSON object')
        return obj

    @app.get('/api/state')
    def get_state(selected:int=0,reveal:bool=False):
        with controller.lock:
            if controller.error:raise HTTPException(503,controller.error)
            if not 0<=selected<len(session.world.bodies):raise HTTPException(400,'Invalid selected individual')
            controller.selected=selected
            state=session.state(selected,reveal=reveal)
            if hasattr(session,'graph'):state['profile']=session.graph.summary
            return state

    @app.post('/api/command')
    async def command(request:Request):
        data=await structured(request,64*1024)
        with controller.lock:
            if controller.error:raise HTTPException(503,controller.error)
            try:
                result=session.execute(data)
                if controller.journal:
                    try:
                        if data.get('type')=='step':controller.journal.advanced(session,data)
                        else:controller.journal.write({'type':'command','command':data})
                    except Exception as exc:
                        controller.record_failure(exc)
                        raise HTTPException(503,controller.error) from exc
                if data.get('type') in ('reset','scenario'):controller.frames.clear()
                return result if isinstance(result,dict) else {'ok':True}
            except (ValueError,KeyError,TypeError,IndexError) as exc:raise HTTPException(400,str(exc)) from exc

    @app.get('/api/snapshot')
    def snapshot():
        with controller.lock:
            data=json.dumps(session.snapshot(),allow_nan=False,separators=(',',':'))
        return Response(data,media_type='application/json',headers={'Content-Disposition':'attachment; filename="neuroterrarium-snapshot.json"'})

    @app.post('/api/snapshot')
    async def restore(request:Request):
        data=await structured(request,MAX_UPLOAD)
        with controller.lock:
            try:session.restore(data)
            except (ValueError,KeyError,TypeError,IndexError) as exc:raise HTTPException(400,str(exc)) from exc
            controller.error=None
            controller.frames.clear()
            if controller.journal:
                try:controller.journal.restored(session)
                except Exception as exc:
                    controller.record_failure(exc)
                    raise HTTPException(503,controller.error) from exc
        return {'ok':True}

    @app.get('/api/recording')
    def recording():
        with controller.lock:
            data={'schema':'neuroterrarium.frame-replay.v1','mode':'Replay',
                  'verification':'recorded event/action playback; no controller recomputation',
                  'frames':controller.frames.export()}
        return JSONResponse(data,headers={'Content-Disposition':'attachment; filename="neuroterrarium-recording.json"'})

    if web_dir is None:
        candidate=Path(__file__).resolve().parents[2]/'web/dist'
        web_dir=candidate if candidate.is_dir() else Path(__file__).parent/'static'
    if not web_dir.is_dir():raise FileNotFoundError('Built web assets are missing; rebuild the application package')
    app.mount('/',StaticFiles(directory=web_dir,html=True),name='web')
    return app


def serve(data_dir,models_dir,host='127.0.0.1',port=8765,open_browser=False):
    if host not in ('127.0.0.1','localhost','::1'):raise ValueError('The application only listens on loopback')
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('Invalid local port')
    import uvicorn
    from .runtime import Session
    session=Session(data_dir,models_dir)
    record_dir=Path('recordings')/datetime.now(timezone.utc).strftime('session-%Y%m%d-%H%M%S-%f')
    app=create_app(session,port=port,record_dir=record_dir)
    if open_browser:
        import webbrowser
        browser_host=f'[{host}]' if ':' in host else host
        threading.Timer(1.5,lambda:webbrowser.open(f'http://{browser_host}:{port}')).start()
    uvicorn.run(app,host=host,port=port,log_level='warning',access_log=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',required=True);parser.add_argument('--models-dir',required=True);parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();serve(args.data_dir,args.models_dir,port=args.port)
