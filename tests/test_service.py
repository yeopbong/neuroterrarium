
from fastapi.testclient import TestClient
import pytest

from neuroterrarium.service import MAX_UPLOAD, create_app, serve
from neuroterrarium.world import World


class TransportFixture:
    """Only the transport is under test; real controllers have separate tests."""
    def __init__(self):self.world=World(1,1);self.paused=True;self.commands=[]
    def state(self,selected,reveal=False):return {'selected':selected,'step':self.world.step}
    def execute(self,data):
        if data.get('type')!='sham':raise ValueError('Unknown command')
        self.commands.append(data);return {'ok':True}
    def snapshot(self):return self.world.snapshot()
    def restore(self,snapshot):self.world=World.restore(snapshot)


@pytest.fixture
def client(tmp_path):
    (tmp_path/'index.html').write_text('<title>Transport test</title>')
    fixture=TransportFixture()
    with TestClient(create_app(fixture,web_dir=tmp_path),base_url='http://127.0.0.1:8765') as c:
        yield c


def test_host_and_origin_guards(client):
    assert client.get('/api/state',headers={'Host':'attacker.example'}).status_code==403
    assert client.post('/api/command',json={'type':'sham'},headers={'Origin':'https://attacker.example'}).status_code==403
    assert client.post('/api/command',json={'type':'sham'},headers={'Origin':'http://localhost:8765'}).status_code==200


def test_structured_upload_limits(client):
    assert client.post('/api/command',content='type=sham').status_code==415
    assert client.post('/api/command',json={'type':'sham'},headers={'Content-Length':str(MAX_UPLOAD+1)}).status_code==413
    assert client.post('/api/command',json={'type':'sham','extra':'x'*70000}).status_code==413
    assert client.post('/api/command',content='{"a":NaN}',headers={'Content-Type':'application/json'}).status_code==400
    assert client.post('/api/command',json=['sham']).status_code==400


def test_bad_snapshot_is_atomic_and_no_arbitrary_file_route(client):
    snapshot=client.get('/api/snapshot').json();bad={**snapshot,'bodies':[]}
    assert client.post('/api/snapshot',json=bad).status_code==400
    assert client.get('/api/snapshot').json()==snapshot
    assert client.get('/api/file?path=/etc/passwd').status_code==404
    assert client.post('/api/command',json={'type':'shell','command':'id'}).status_code==400


def test_assets_and_recordings_have_security_and_mode_headers(client):
    response=client.get('/')
    assert response.status_code==200
    assert "frame-ancestors 'none'" in response.headers['content-security-policy']
    recording=client.get('/api/recording').json()
    assert recording['mode']=='Replay'
    assert 'no controller recomputation' in recording['verification']


def test_loopback_only_before_loading():
    with pytest.raises(ValueError,match='loopback'):serve('missing','missing',host='0.0.0.0')


@pytest.mark.parametrize(('host', 'expected_url'), [
    ('127.0.0.1', 'http://127.0.0.1:8766'),
    ('localhost', 'http://localhost:8766'),
    ('::1', 'http://[::1]:8766'),
])
def test_browser_opens_the_bound_loopback_address(monkeypatch, host, expected_url):
    from neuroterrarium import runtime, service
    import uvicorn
    import webbrowser

    opened = []
    servers = []
    sentinel = object()
    monkeypatch.setattr(runtime, 'Session', lambda *_: sentinel)
    monkeypatch.setattr(service, 'create_app', lambda session, **_: session)
    monkeypatch.setattr(webbrowser, 'open', opened.append)
    monkeypatch.setattr(uvicorn, 'run', lambda app, **kwargs: servers.append((app, kwargs)))

    class ImmediateTimer:
        def __init__(self, delay, callback):
            self.callback = callback

        def start(self):
            self.callback()

    monkeypatch.setattr(service.threading, 'Timer', ImmediateTimer)
    serve('unused-data', 'unused-models', host=host, port=8766, open_browser=True)
    assert opened == [expected_url]
    assert servers[0][0] is sentinel
    assert servers[0][1]['host'] == host
    assert servers[0][1]['port'] == 8766
