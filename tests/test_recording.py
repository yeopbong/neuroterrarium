import json

from fastapi.testclient import TestClient
import pytest

from neuroterrarium.recording import ExecutionJournal, FrameBuffer, frame, recompute
from neuroterrarium.replay import create_replay_app, load_recording
from neuroterrarium.runtime import _wire_digest
pytest_plugins = ['test_runtime']


def test_frame_buffer_bounds_bytes_and_count():
    buffer=FrameBuffer(max_bytes=100,max_frames=2)
    for i in range(10):buffer.append({'step':i,'sample':'small'})
    assert len(buffer.export())==2 and buffer.bytes<=100
    assert buffer.export()[-1]['step']==9
    with pytest.raises(ValueError,match='budget'):buffer.append({'sample':'x'*101})
    assert buffer.export()[-1]['step']==9


def test_recording_captures_each_individuals_actual_input(session,tmp_path):
    session.step()
    recording=frame(session,3)
    for index,panel in enumerate(recording['panels']):
        assert panel['index']==index
        assert panel['observation']==session.records[-1]['observations'][index]
    path=tmp_path/'recording.json'
    path.write_text(json.dumps({'schema':'neuroterrarium.frame-replay.v1','mode':'Replay','frames':[recording]}))
    assert load_recording(path)['frames'][0]['selected']['index']==3
    web=tmp_path/'web';web.mkdir();(web/'index.html').write_text('<title>Replay</title>')
    with TestClient(create_replay_app(path,web_dir=web),base_url='http://127.0.0.1:8765') as client:
        assert client.get('/').url.path=='/index.html'
        assert client.get('/recording.json').json()['mode']=='Replay'
        assert client.post('/api/command',json={'type':'resume'}).status_code==405
        assert client.get('/recording.json',headers={'Origin':'https://untrusted.example'}).status_code==403
        assert client.get('/recording.json',headers={'Origin':'http://127.0.0.1:8765'}).status_code==200


def test_executable_journal_recomputes_actions_interventions_and_restore(session,tmp_path):
    root=tmp_path/'journal';journal=ExecutionJournal(root,session)
    initial=session.snapshot()
    session.step();journal.advanced(session)
    command={'type':'noise','value':.1}
    session.execute(command);journal.write({'type':'command','command':command})
    session.step();journal.advanced(session)
    session.restore(initial);journal.restored(session)
    session.step();journal.advanced(session)
    journal.close()
    result=recompute(session,root)
    assert result['action_windows']==3 and result['operations']==5
    lines=(root/'execution.jsonl').read_text().splitlines()
    changed=json.loads(lines[1]);changed['record']['decisions'][0]['applied_action'][0]+=0.1
    lines[1]=json.dumps(changed);(root/'execution.jsonl').write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='differs'):recompute(session,root)


def test_journal_initial_state_mutation_and_empty_execution_fail(session,tmp_path):
    root=tmp_path/'journal';journal=ExecutionJournal(root,session);journal.close()
    with pytest.raises(ValueError,match='no executed'):recompute(session,root)
    snapshot=json.loads((root/'initial.json').read_text());snapshot['state']['brain']['v_mV'][0]+=1
    snapshot['sha256']=_wire_digest(snapshot['state'])
    (root/'initial.json').write_text(json.dumps(snapshot))
    with pytest.raises(ValueError,match='identity'):recompute(session,root)


def test_replay_rejects_nonfinite_and_unlabeled_data(tmp_path):
    path=tmp_path/'bad.json';path.write_text('{"x": NaN}')
    with pytest.raises(ValueError):load_recording(path)
    path.write_text('{"frames": []}')
    with pytest.raises(ValueError,match='labeled'):load_recording(path)
