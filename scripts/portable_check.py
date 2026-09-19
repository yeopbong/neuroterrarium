"""Check a frozen macOS archive by clean extraction, offline execution and relocation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

import psutil

from install_check import PROBE, atomic_json, checksum

MAX_BYTES = 4 * 2**30
MAX_FILES = 100000
RUNTIME_SOURCES = ('brain_cache.py', 'controllers.py', 'data.py', 'interface.py', 'neural.py',
                   'reference.py', 'registry.py', 'runtime.py', 'world.py', 'validation.py', 'service.py', 'cli.py')
CONFIGS = ('data-v783.json', 'brain-interface.json', 'interface-v783.json', 'brain-storage-v1.json')


def within(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Archive path must be a relative POSIX name')
    pieces = relative.split('/')
    if PurePosixPath(relative).is_absolute() or any(part in ('', '.', '..') for part in pieces):
        raise ValueError('Archive path traverses its extraction directory')
    path = root.joinpath(*pieces)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Archive path escapes its extraction directory')
    return path


def verify_contents(bundle):
    metadata = bundle / 'installation.json'
    if metadata.is_symlink() or metadata.stat().st_size > 32 * 2**20:
        raise ValueError('Invalid installation manifest')
    manifest = json.loads(metadata.read_text())
    if manifest.get('schema') != 'neuroterrarium.installation.v1' or manifest.get('platform') != 'macos-arm64':
        raise ValueError('Wrong installation schema or platform')
    files = manifest.get('files')
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError('Invalid installation file inventory')
    total = 0
    for name, info in files.items():
        path = within(bundle, name)
        if not isinstance(info, dict):
            raise ValueError('Invalid installation file metadata')
        if set(info) == {'symlink'}:
            if not path.is_symlink() or os.readlink(path) != info['symlink'] or not path.resolve().is_file():
                raise ValueError('Installation symlink identity mismatch')
        elif set(info) == {'bytes', 'sha256'}:
            if (path.is_symlink() or not path.is_file() or type(info['bytes']) is not int or info['bytes'] < 0
                    or not isinstance(info['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', info['sha256'])
                    or path.stat().st_size != info['bytes'] or checksum(path) != info['sha256']):
                raise ValueError('Installed file hash or size mismatch')
            total += info['bytes']
        else:
            raise ValueError('Unsupported installation file metadata')
        if total > MAX_BYTES:
            raise ValueError('Installation exceeds unpacked size limit')
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob('*') if path.is_file() or path.is_symlink()}
    if actual != set(files) | {'installation.json'}:
        raise ValueError('Installation contains missing or unregistered files')
    if manifest.get('trained_models') != 9:
        raise ValueError('Portable archive must include all nine trained models')
    if checksum(bundle / 'artifacts/models/manifest.json') != manifest.get('model_set_manifest_sha256'):
        raise ValueError('Portable model-set identity mismatch')
    return {'status': 'passed', 'files': len(files), 'bytes': total,
            'source_sha256': manifest['source']['sha256'], 'manifest_sha256': checksum(metadata)}


def extract(package, expected_sha256, destination):
    if not re.fullmatch('[a-f0-9]{64}', expected_sha256) or package.is_symlink() or checksum(package) != expected_sha256:
        raise ValueError('Portable archive checksum mismatch')
    if destination.exists():
        raise ValueError('Clean extraction destination must not already exist')
    destination.mkdir(parents=True)
    with tarfile.open(package, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) > MAX_FILES or sum(item.size for item in members) > MAX_BYTES:
            raise ValueError('Archive extraction exceeds resource bounds')
        names = set()
        for item in members:
            within(destination, item.name)
            if item.name != 'NeuroTerrarium' and not item.name.startswith('NeuroTerrarium/'):
                raise ValueError('Unexpected portable archive root')
            if item.name in names or not (item.isdir() or item.isfile() or item.issym() or item.islnk()):
                raise ValueError('Duplicate or unsupported archive entry')
            names.add(item.name)
        archive.extractall(destination, members=members, filter='data')
    result = verify_contents(destination / 'NeuroTerrarium')
    result['archive_sha256'] = expected_sha256
    return result


def guard(directory):
    if psutil.virtual_memory().available < 3 * 2**30 or shutil.disk_usage(directory).free < 10 * 2**30:
        raise ValueError('Portable verification requires 3 GiB available memory and 10 GiB free disk')


def installed_identity(bundle):
    manifest = json.loads((bundle / 'installation.json').read_text())
    source = manifest['source']['files']
    modules = bundle / 'python/lib/python3.12/site-packages/neuroterrarium'
    configuration = bundle / 'python/share/neuroterrarium/configs'
    hashes = {name: checksum(modules / name) for name in RUNTIME_SOURCES}
    configs = {name: checksum(configuration / name) for name in CONFIGS}
    for name, digest in hashes.items():
        if source.get('src/neuroterrarium/' + name) != digest:
            raise ValueError('Installed runtime source differs from the recorded build input')
    for name, digest in configs.items():
        if source.get('configs/' + name) != digest:
            raise ValueError('Installed configuration differs from the recorded build input')
    return {'validation_source_sha256': hashes['validation.py'], 'package_source_sha256': hashes,
        'config_sha256': configs, 'verification_source_sha256': {
            'portable_check.py': checksum(Path(__file__)),
            'install_check.py': checksum(Path(__file__).with_name('install_check.py')),
            'build_package.py': source['scripts/build_package.py']}}


def native_libraries(raw, bundle):
    paths = set(re.findall(r'^dyld\[[0-9]+\]:.*? (/[^\n]+)$', raw.decode('utf8', errors='replace'), re.M))
    if not paths: raise ValueError('Native loader trace is empty')
    system, local = 0, 0
    for name in paths:
        if name.startswith(('/System/', '/usr/lib/')): system += 1
        elif Path(name).resolve().is_relative_to(bundle.resolve()): local += 1
        else: raise ValueError('A native library loaded from outside the bundle or operating system')
    if not system or not local: raise ValueError('Native loader trace does not cover both runtime and system libraries')
    return {'status': 'passed', 'unique_loaded_paths': len(paths), 'system_paths': system,
            'bundle_paths': local, 'unexpected_paths': []}


HTTP_GUARD = r'''
import socket,sys
allowed={'127.0.0.1','localhost','::1'}
original_socket=socket.socket
original_dns=socket.getaddrinfo
class LocalSocket(original_socket):
    def connect(self,address):
        if not isinstance(address,tuple) or address[0] not in allowed: raise RuntimeError('External networking denied')
        return super().connect(address)
    def connect_ex(self,address):
        if not isinstance(address,tuple) or address[0] not in allowed: raise RuntimeError('External networking denied')
        return super().connect_ex(address)
def local_dns(host,*args,**kwargs):
    if host not in allowed: raise RuntimeError('External DNS denied')
    return original_dns(host,*args,**kwargs)
socket.socket=LocalSocket
socket.getaddrinfo=local_dns
try:
    socket.create_connection(('not-allowed.invalid',443))
except RuntimeError:
    pass
else:
    raise AssertionError('Network guard did not reject external access')
from neuroterrarium.cli import main
raise SystemExit(main(sys.argv[1:]))
'''


def local_http(python, data_dir, models, probe, working, environment, server_log):
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    command = [str(python), '-I', '-B', str(probe), 'app', '--data-dir', str(data_dir),
               '--models', str(models), '--port', str(port)]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f'http://127.0.0.1:{port}'
    def request(path, body=None, headers=None):
        encoded = json.dumps(body).encode() if body is not None else None
        request_headers = {'Content-Type': 'application/json'} if body is not None else {}
        request_headers.update(headers or {})
        with opener.open(urllib.request.Request(base + path, data=encoded, headers=request_headers), timeout=30) as response:
            data = response.read(32 * 2**20 + 1)
            if len(data) > 32 * 2**20: raise ValueError('HTTP response exceeds capacity')
            return response.status, data, dict(response.headers)
    with server_log.open('wb') as log:
        process = subprocess.Popen(command, cwd=working, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None: raise ValueError('Portable application exited before becoming ready')
                try:
                    _, raw, _ = request('/api/state?selected=0&reveal=true')
                    state = json.loads(raw)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline: raise ValueError('Portable application startup timed out')
                    time.sleep(.2)
            if state.get('mode') != 'Local full graph' or not state.get('ready') or len(state['world']['bodies']) != 10:
                raise ValueError('Portable application is not the actual ten-body full-graph mode')
            request('/api/command', {'type': 'pause'})
            _, before_raw, _ = request('/api/state?selected=0&reveal=true')
            before = json.loads(before_raw)
            request('/api/command', {'type': 'step'})
            _, after_raw, _ = request('/api/state?selected=0&reveal=true')
            after = json.loads(after_raw)
            if abs(after['simulation_time'] - before['simulation_time'] - .02) > 1e-9:
                raise ValueError('Portable local step did not advance the authoritative clock')
            index_status, index, _ = request('/index.html')
            scripts = re.findall(rb'<script[^>]+src="([^"]+)"', index)
            if not scripts: raise ValueError('Installed browser entry has no JavaScript resource')
            resources = []
            for url in scripts:
                path = url.decode()
                if path.startswith(('http:', 'https:', '//')): raise ValueError('Installed application requested an external script')
                status, content, _ = request('/' + path.lstrip('./'))
                if status != 200 or not content: raise ValueError('Installed script did not load')
                resources.append({'path': path, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
            try:
                request('/api/state', headers={'Origin': 'https://untrusted.invalid'})
            except urllib.error.HTTPError as error:
                if error.code != 403: raise
            else: raise ValueError('Portable local service accepted an invalid Origin')
            return {'status': 'passed', 'mode': after['mode'], 'bodies': 10, 'advance_seconds': .02,
                    'index_status': index_status, 'index_sha256': hashlib.sha256(index).hexdigest(),
                    'scripts': resources, 'origin_rejected': True, 'network': 'external Python sockets and DNS denied; loopback allowed'}
        finally:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)


def run_check(package, expected_sha256, data_dir, output, artifact_root):
    if Path(package).is_symlink(): raise ValueError('Portable archive must not be a symlink')
    package, data_dir, output, artifact_root = map(lambda p: Path(p).resolve(), (package, data_dir, output, artifact_root))
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('This portable archive check is defined for macOS arm64')
    if output.exists() and any(output.iterdir()): raise ValueError('Portable evidence directory must be empty')
    if not package.is_relative_to(artifact_root): raise ValueError('Package must be inside the release artifact root')
    output.mkdir(parents=True, exist_ok=True); guard(output)
    if not data_dir.is_dir(): raise ValueError('Verified local data must be prepared first')
    summary = {'schema': 'neuroterrarium.portable-check.v1', 'status': 'running',
        'clean_extraction': False, 'offline_runtime': False, 'path_case': 'spaces and non-ASCII characters',
        'package': {'path': package.relative_to(artifact_root).as_posix(), 'sha256': expected_sha256, 'bytes': package.stat().st_size},
        'platform': {'system': platform.system(), 'machine': platform.machine(), 'release': platform.release()}, 'steps': []}
    replacements = [(str(output), '<portable-check>'), (str(package), '<release-package>'), (str(data_dir), '<data-cache>'),
                    (str(Path.cwd()), '<workspace>'), (str(Path(sys.base_prefix)), '<host-python>'), (str(Path.home()), '<home>')]
    def redact(text):
        for old, new in replacements: text = text.replace(old, new)
        return text
    environment = {'PATH': os.defpath, 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1',
        'NUMBA_CACHE_DIR': str(output / 'cache/numba')}
    working = output / '空白 working directory'; working.mkdir()
    def record(name, command, *, timeout=600, native_bundle=None):
        guard(output); started = time.monotonic()
        command_environment = dict(environment)
        if native_bundle is not None: command_environment['DYLD_PRINT_LIBRARIES'] = '1'
        try:
            result = subprocess.run(command, cwd=working, env=command_environment, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
            raw, code, state = result.stdout, result.returncode, 'completed'
        except subprocess.TimeoutExpired as error:
            raw, code, state = error.stdout or b'', None, 'timeout'
        log = output / f'{len(summary["steps"])+1:02d}-{name}.log'; log.write_text(redact(raw.decode('utf8', errors='replace')))
        summary['steps'].append({'name': name, 'command': [redact(str(x)) for x in command], 'exit_code': code,
            'state': state, 'wall_seconds': time.monotonic()-started, 'log': log.name,
            'log_sha256': checksum(log), 'original_output_sha256': hashlib.sha256(raw).hexdigest()})
        atomic_json(output / 'summary.json', summary)
        if code != 0: raise ValueError(f'Portable check failed: {name} ({state}, exit {code})')
        if native_bundle is not None: summary['native_libraries'] = native_libraries(raw, native_bundle)
    started = time.monotonic()
    try:
        destination = output / '初次 clean extraction'
        script = str(Path(__file__).resolve())
        record('extract', [sys.executable, script, '_extract', '--package', str(package), '--sha256', expected_sha256,
                           '--destination', str(destination)])
        summary['clean_extraction'] = True
        bundle = destination / 'NeuroTerrarium'
        summary.update(installed_identity(bundle))
        probe = working / 'offline_probe.py'; probe.write_text(PROBE)
        python = bundle / 'python/bin/python3.12'; models = bundle / 'artifacts/models'
        record('installed-contents', [str(python), '-I', '-B', str(probe), 'contents'])
        record('doctor', [str(python), '-I', '-B', str(probe), 'doctor', '--data-dir', str(data_dir), '--models', str(models)],
               native_bundle=bundle)
        record('offline-runtime-validation', [str(python), '-I', '-B', str(probe), 'validate', '--data-dir', str(data_dir),
                                               '--models', str(models), '--output', str(output / 'runtime-evidence')])
        http_probe = working / 'local_http_guard.py'; http_probe.write_text(HTTP_GUARD)
        record('local-http', [sys.executable, script, '_http', '--python', str(python), '--data-dir', str(data_dir),
            '--models', str(models), '--probe', str(http_probe), '--working', str(working), '--output', str(output)], timeout=180)
        moved_parent = output / '搬移 portable application'; moved_parent.mkdir()
        moved = moved_parent / 'NeuroTerrarium'; bundle.rename(moved)
        record('relocated-launcher', [str(moved / 'neuroterrarium'), 'doctor', '--data-dir', str(data_dir)])
        summary['offline_runtime'] = True; summary['status'] = 'passed'
    except (OSError, ValueError, RuntimeError) as error:
        summary['status'] = 'failed'; summary['failure'] = {'type': type(error).__name__, 'message': redact(str(error))}
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
    finally:
        server_original = output / 'server-original.log'
        if server_original.is_file():
            raw = server_original.read_bytes()
            server_log = output / 'server.log'
            server_log.write_text(redact(raw.decode('utf8', errors='replace')))
            summary['server_log'] = {'log': server_log.name, 'sha256': checksum(server_log),
                                      'original_output_sha256': hashlib.sha256(raw).hexdigest()}
            server_original.unlink()
        summary['wall_seconds'] = time.monotonic() - started
        atomic_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if len(sys.argv) > 1 and sys.argv[1] == '_extract':
        parser.add_argument('_mode'); parser.add_argument('--package', type=Path, required=True)
        parser.add_argument('--sha256', required=True); parser.add_argument('--destination', type=Path, required=True)
        args = parser.parse_args(); result = extract(args.package, args.sha256, args.destination)
    elif len(sys.argv) > 1 and sys.argv[1] == '_http':
        parser.add_argument('_mode')
        for name in ('python','data-dir','models','probe','working','output'): parser.add_argument('--'+name, type=Path, required=True)
        args = parser.parse_args()
        result = local_http(args.python, args.data_dir, args.models, args.probe, args.working, dict(os.environ), args.output/'server-original.log')
        # The parent records normalized command output. Keep server diagnostics
        # private until they are normalized alongside the other step logs.
    else:
        for name in ('package','data-dir','output','artifact-root'): parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--sha256', required=True); args = parser.parse_args()
        result = run_check(args.package,args.sha256,args.data_dir,args.output,args.artifact_root)
    print(json.dumps(result,sort_keys=True,allow_nan=False));return 0 if result['status']=='passed' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f'{type(error).__name__}: {error}',file=sys.stderr);raise SystemExit(1)
