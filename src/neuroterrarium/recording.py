"""Bounded frame history and an append-only executable session journal."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import shutil

from .data import sha256_file


def frame(session, selected=0):
    result = session.state(selected, reveal=True)
    result['profile']=session.graph.summary
    result['events'] = result['events'][-16:]
    result['panels'] = [session.state(i, reveal=True)['selected'] for i in range(10)]
    if session.fork_session:
        result['fork']['events'] = result['fork']['events'][-16:]
        result['fork']['panels'] = [session.fork_session.state(i, reveal=True)['selected'] for i in range(10)]
    return result


class FrameBuffer:
    """Limit both frame count and serialized bytes; never store every neuron."""
    def __init__(self, max_bytes=20*2**20, max_frames=1500):
        self.frames=deque();self.bytes=0
        self.max_bytes=max_bytes;self.max_frames=max_frames

    def append(self, value):
        encoded=json.dumps(value, allow_nan=False, separators=(',', ':'))
        size=len(encoded.encode('utf-8'))
        if size>self.max_bytes:raise ValueError('One recording frame exceeds the memory budget')
        self.frames.append((encoded,size));self.bytes+=size
        while self.bytes>self.max_bytes or len(self.frames)>self.max_frames:
            self.bytes-=self.frames.popleft()[1]

    def clear(self):
        self.frames.clear();self.bytes=0

    def export(self):
        return [json.loads(encoded) for encoded,_ in self.frames]


class ExecutionJournal:
    """Trusted local output path; no path from an HTTP request is accepted."""
    def __init__(self, directory, session):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.count=0
        session.save(self.directory/'initial.json')
        self.handle=(self.directory/'execution.jsonl').open('x',encoding='utf-8')
        self.write({'type':'identity','schema':'neuroterrarium.execution.v1',
                    'initial_sha256':sha256_file(self.directory/'initial.json'),
                    'verification':'controller recomputation from initial state and ordered external operations'})

    def write(self, value):
        if self.count%500==0 and shutil.disk_usage(self.directory).free<10*2**30:
            raise RuntimeError('Recording stopped: less than 10 GiB free disk space')
        self.handle.write(json.dumps({'sequence':self.count,**value},allow_nan=False,separators=(',',':'))+'\n')
        self.handle.flush();self.count+=1

    def advanced(self, session, command=None):
        record={'type':'advance' if command is None else 'command','record':session.records[-1]}
        if command is not None:record['command']=command
        if session.fork_session:record['fork_record']=session.fork_session.records[-1]
        self.write(record)

    def restored(self, session):
        name=f'restore-{self.count:08d}.json'
        session.save(self.directory/name)
        self.write({'type':'restore','snapshot':name,'sha256':sha256_file(self.directory/name)})

    def close(self):
        self.handle.close()


def recompute(session, directory, *, max_operations=None):
    """Re-execute a bounded journal prefix and compare every logged decision."""
    directory=Path(directory)
    path=directory/'execution.jsonl'
    operations=0;windows=0
    with path.open(encoding='utf-8') as handle:
        identity=json.loads(handle.readline())
        if identity.get('schema')!='neuroterrarium.execution.v1' or identity.get('initial_sha256')!=sha256_file(directory/'initial.json'):
            raise ValueError('Executable journal identity mismatch')
        session.load(directory/'initial.json')
        for line in handle:
            if max_operations is not None and operations>=max_operations:break
            if len(line)>4*2**20:raise ValueError('Journal entry exceeds limit')
            item=json.loads(line)
            if item.get('sequence')!=operations+1:raise ValueError('Journal sequence mismatch')
            if item['type']=='advance':session.step()
            elif item['type']=='command':session.execute(item['command'])
            elif item['type']=='restore':
                name=item['snapshot']
                if name!=f'restore-{item["sequence"]:08d}.json':raise ValueError('Invalid journal snapshot name')
                snapshot=directory/name
                if sha256_file(snapshot)!=item['sha256']:raise ValueError('Journal snapshot checksum mismatch')
                session.load(snapshot)
            else:raise ValueError('Unknown journal operation')
            if 'record' in item:
                if session.records[-1]!=item['record']:raise ValueError(f'Controller recomputation differs at operation {operations+1}')
                if 'fork_record' in item and (session.fork_session is None or session.fork_session.records[-1]!=item['fork_record']):
                    raise ValueError('Branch recomputation differs')
                windows+=1
            operations+=1
    if windows==0:raise ValueError('Journal contains no executed action windows in the requested prefix')
    return {'status':'passed','verification':'controller recomputation','operations':operations,'action_windows':windows,
            'scope':'declared local numerical backend; exact observations, raw decisions and applied actions'}
