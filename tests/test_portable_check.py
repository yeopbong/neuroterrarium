"""Small archive-boundary tests; no synthetic package is a release artifact."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('portable_check', SCRIPTS / 'portable_check.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def tiny_archive(tmp_path, malicious=None):
    root = tmp_path / 'input/NeuroTerrarium'
    models = root / 'artifacts/models'
    models.mkdir(parents=True)
    model = models / 'manifest.json'
    model.write_text('{"synthetic_unit_fixture":true}')
    manifest = {'schema': 'neuroterrarium.installation.v1', 'platform': 'macos-arm64', 'trained_models': 9,
        'model_set_manifest_sha256': module.checksum(model), 'source': {'sha256': 'a' * 64},
        'files': {'artifacts/models/manifest.json': {'bytes': model.stat().st_size, 'sha256': module.checksum(model)}}}
    (root / 'installation.json').write_text(json.dumps(manifest))
    package = tmp_path / 'fixture.tar.gz'
    with tarfile.open(package, 'w:gz') as archive:
        archive.add(root, arcname='NeuroTerrarium')
        if malicious is not None:
            name, link = malicious
            entry = tarfile.TarInfo(name)
            if link is not None:
                entry.type = tarfile.SYMTYPE; entry.linkname = link
                archive.addfile(entry)
            else:
                entry.size = 1; archive.addfile(entry, io.BytesIO(b'x'))
    return package, module.checksum(package)


def test_clean_hash_checked_extraction(tmp_path):
    package, digest = tiny_archive(tmp_path)
    target = tmp_path / '空格 clean extraction'
    result = module.extract(package, digest, target)
    assert result['status'] == 'passed' and result['files'] == 1
    with pytest.raises(ValueError, match='not already'): module.extract(package, digest, target)


def test_bad_archive_hash_is_rejected_before_extraction(tmp_path):
    package, _ = tiny_archive(tmp_path)
    target = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='checksum'): module.extract(package, 'b' * 64, target)
    assert not target.exists()


@pytest.mark.parametrize('malicious', [('NeuroTerrarium/../../outside', None), ('unrelated/file', None),
    ('NeuroTerrarium/escape', '../../outside'), ('NeuroTerrarium/installation.json', None)])
def test_paths_links_and_duplicate_entries_cannot_escape(tmp_path, malicious):
    package, digest = tiny_archive(tmp_path, malicious)
    with pytest.raises((ValueError, tarfile.FilterError)):
        module.extract(package, digest, tmp_path / 'out')
    assert not (tmp_path / 'outside').exists()


def test_extracted_file_mutation_and_extra_file_fail(tmp_path):
    package, digest = tiny_archive(tmp_path)
    target = tmp_path / 'out'
    module.extract(package, digest, target)
    root = target / 'NeuroTerrarium'
    model = root / 'artifacts/models/manifest.json'
    before = model.read_bytes()
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash or size'): module.verify_contents(root)
    model.write_bytes(before)
    (root / 'unexpected.py').write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError, match='unregistered'): module.verify_contents(root)


def test_symlink_archive_and_memory_reserve_fail(tmp_path, monkeypatch):
    package, digest = tiny_archive(tmp_path)
    link = tmp_path / 'linked.tar.gz'; link.symlink_to(package)
    with pytest.raises(ValueError, match='symlink'):
        module.run_check(link, digest, tmp_path, tmp_path / 'out', tmp_path)
    monkeypatch.setattr(module.psutil, 'virtual_memory', lambda: type('Memory', (), {'available': 1})())
    with pytest.raises(ValueError, match='3 GiB'): module.guard(tmp_path)


def test_native_loader_trace_rejects_empty_or_external_paths(tmp_path):
    bundle = tmp_path / 'bundle'
    trace = f'dyld[42]: <abc> /usr/lib/libSystem.B.dylib\ndyld[42]: <def> {bundle}/python/lib/libpython3.12.dylib\n'.encode()
    result = module.native_libraries(trace, bundle)
    assert result['system_paths'] == result['bundle_paths'] == 1
    with pytest.raises(ValueError, match='empty'): module.native_libraries(b'', bundle)
    with pytest.raises(ValueError, match='outside'):
        module.native_libraries(trace + b'dyld[42]: <ghi> /opt/local/lib/libunrelated.dylib\n', bundle)


def test_installed_source_identity_uses_actual_package_bytes(tmp_path):
    modules = tmp_path / 'python/lib/python3.12/site-packages/neuroterrarium'
    configs = tmp_path / 'python/share/neuroterrarium/configs'
    modules.mkdir(parents=True); configs.mkdir(parents=True)
    source = {'scripts/build_package.py': 'a' * 64}
    for name in module.RUNTIME_SOURCES:
        path = modules / name; path.write_text('# Explicit synthetic source fixture\n')
        source['src/neuroterrarium/' + name] = module.checksum(path)
    for name in module.CONFIGS:
        path = configs / name; path.write_text('{}')
        source['configs/' + name] = module.checksum(path)
    (tmp_path / 'installation.json').write_text(json.dumps({'source': {'files': source}}))
    result = module.installed_identity(tmp_path)
    assert result['validation_source_sha256'] == module.checksum(modules / 'validation.py')
    (modules / 'runtime.py').write_text('# changed')
    with pytest.raises(ValueError, match='runtime source'): module.installed_identity(tmp_path)
