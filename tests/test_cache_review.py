"""Independent cache corruption and active-state checks on explicit small fixtures.

No fixture is a FlyWire graph or distributed as a runtime data asset.
"""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neuroterrarium import brain_cache as cache
from neuroterrarium.neural import NeuralLimits, NeuralResourceError, SparseBrain


def small_brain():
    return SparseBrain(5, [0, 0, 1, 2, 3, 4, 4], [1, 4, 4, 4, 0, 2, 3],
                       [240, 100, 100, -20, 40, 200, -100], input_neurons=[0, 1, 2, 3])


def cache_fixture(tmp_path, monkeypatch):
    original = small_brain()
    roots = np.arange(1, 6, dtype=np.int64)
    digest = 'a' * 64
    data = {'profile': 'small-cache-test', 'connectome_version': '783',
            'expected': {'neurons': 5, 'directed_pairs': 7, 'connection_rows': 7, 'synaptic_contacts': 800},
            'sources': {name: {'sha256': digest, 'filename': name} for name in ('completeness', 'connectivity', 'annotations')}}
    profile = {'profile': data['profile'], **data['expected'], 'graph_digest': original.graph_digest, 'license': 'test fixture'}
    identity = {'profile': data['profile'], 'graph_digest': original.graph_digest, 'source_sha256': {'test.py': digest}}
    monkeypatch.setattr(cache, '_identity', lambda *args: (tmp_path / 'data.json', data, profile, identity))
    monkeypatch.setattr(cache, '_verify_sources', lambda *args: {name: tmp_path / name for name in data['sources']})
    monkeypatch.setattr(cache, '_read_roots', lambda *args: roots.copy())
    class TestRegistry:
        @classmethod
        def load(cls): return cls()
        def verify_annotations(self, path): return {'known': 5}
        def resolve(self, root_ids):
            return {name: np.array([index]) for index, name in enumerate(('sugar', 'lplc2', 'lc4', 'dna02'))}
        def resolve_sides(self, root_ids): return {}
    monkeypatch.setattr(cache, 'Registry', TestRegistry)
    directory = tmp_path / cache.CACHE_DIRECTORY
    directory.mkdir()
    arrays = {'root_ids': roots, 'indptr': original.indptr, 'post_indices': original.post_indices,
              'weights_mV': original.weights_mV, 'input_mask': original.input_mask}
    metadata = {}
    for name, array in arrays.items():
        path = directory / f'{name}.npy'
        np.save(path, array, allow_pickle=False)
        metadata[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
                          'dtype': array.dtype.str, 'shape': list(array.shape)}
    summary = {'schema': 'neuroterrarium.graph-summary.v1', 'profile': data['profile'], 'connectome_version': '783',
               **data['expected'], 'positive_pairs': 5, 'negative_pairs': 2, 'duplicate_pairs': 0,
               'self_pairs': 0, 'neurons_without_outgoing_edges': 0, 'annotations': {'known': 5},
               'source_sha256': {name: digest for name in data['sources']},
               'structure_bytes': 5 * 8 + 7 * 12}
    profile['summary_sha256'] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = {'schema': cache.CACHE_SCHEMA, 'status': 'complete', 'identity': identity,
                'summary': summary, 'arrays': metadata}
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest, original


def test_readonly_mmap_has_independent_persistent_active_state(tmp_path, monkeypatch):
    cache_fixture(tmp_path, monkeypatch)
    first, second = cache.load_cache(tmp_path), cache.load_cache(tmp_path)
    for name in ('indptr', 'post_indices', 'weights_mV', 'input_mask'):
        array = getattr(first.brain, name)
        assert isinstance(array, np.memmap) and array.mode == 'r' and not array.flags.writeable
        with pytest.raises(ValueError): array.flags.writeable = True
    tape = [[0] if step % 7 == 0 else [] for step in range(160)]
    first.brain.advance(tape[:17])
    saved = first.brain.snapshot()
    assert saved['step'] == 17 and sum(map(len, saved['queue'])) > 0
    assert second.brain.step == 0 and not np.shares_memory(first.brain.v_mV, second.brain.v_mV)
    second.brain.restore(copy.deepcopy(saved))
    expected = first.brain.advance(tape[17:], range(5), True)
    actual = second.brain.advance(tape[17:], range(5), True)
    assert len(actual.spike_steps) > 0
    for field in ('spike_steps', 'spike_neurons', 'v_mV', 'g_mV'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize('field,value', [('schema', 'unrelated'), ('connectome_version', '630'),
    ('connection_rows', -1), ('positive_pairs', 1), ('negative_pairs', 9), ('duplicate_pairs', 1),
    ('self_pairs', 7), ('neurons_without_outgoing_edges', 5), ('structure_bytes', -1), ('load_seconds', -1)])
def test_mutated_display_provenance_cannot_pass_cache_verification(tmp_path, monkeypatch, field, value):
    path, manifest, _ = cache_fixture(tmp_path, monkeypatch)
    manifest['summary'][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): cache.load_cache(tmp_path)


@pytest.mark.parametrize('field', ['root_ids', 'input_mask', 'weights_mV', 'post_indices', 'indptr'])
def test_rehashed_array_mutations_still_fail_frozen_identity(tmp_path, monkeypatch, field):
    path, manifest, _ = cache_fixture(tmp_path, monkeypatch)
    target = path.parent / f'{field}.npy'
    values = np.load(target, allow_pickle=False)
    if field == 'root_ids': values[[0, 1]] = values[[1, 0]]
    elif field == 'input_mask': values[0] = False
    elif field == 'weights_mV': values[0] *= 2
    elif field == 'post_indices': values[0] = 4
    else: values[1] += 1
    np.save(target, values, allow_pickle=False)
    manifest['arrays'][field]['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): cache.load_cache(tmp_path)


def test_readonly_view_cannot_leave_a_writable_graph_alias():
    original = small_brain()
    owner = original.weights_mV.copy()
    view = owner.view()
    view.flags.writeable = False
    loaded = SparseBrain.from_csr(original.indptr, original.post_indices, view, original.input_mask,
                                  expected_digest=original.graph_digest)
    before = loaded.weights_mV.copy()
    owner[0] *= 2
    np.testing.assert_array_equal(loaded.weights_mV, before)
    assert loaded.graph_digest == original.graph_digest


def test_resource_rejection_precedes_copy_of_external_arrays():
    class NoCopyArray(np.ndarray):
        def tobytes(self, *args, **kwargs):
            raise AssertionError('The rejected input was copied before checking its budget')
    original = small_brain()
    view = original.weights_mV.view(NoCopyArray)
    with pytest.raises(NeuralResourceError):
        SparseBrain.from_csr(original.indptr, original.post_indices, view, original.input_mask,
                             expected_digest=original.graph_digest,
                             limits=replace(NeuralLimits(), max_graph_working_bytes=1))


def test_old_identity_duplicate_fields_and_missing_files_fail(tmp_path, monkeypatch):
    path, _, _ = cache_fixture(tmp_path, monkeypatch)
    value = json.loads(path.read_text())
    value['identity']['profile'] = 'v630'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='version'): cache.load_cache(tmp_path)
    path.write_text('{"schema":"first","schema":"second"}')
    with pytest.raises(ValueError, match='duplicate'): cache.load_cache(tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match='missing'): cache.load_cache(tmp_path)


def test_failed_rebuild_publication_restores_original_cache(tmp_path, monkeypatch):
    path, manifest, original = cache_fixture(tmp_path, monkeypatch)
    old_manifest = path.read_bytes()
    monkeypatch.setattr(cache, '_resource_guard', lambda directory: None)
    monkeypatch.setattr(cache, 'load_graph', lambda *args: SimpleNamespace(
        root_ids=np.arange(1, 6, dtype=np.int64), pre=[0, 0, 1, 2, 3, 4, 4],
        post=[1, 4, 4, 4, 0, 2, 3], signed_counts=[240, 100, 100, -20, 40, 200, -100],
        summary=manifest['summary']))
    original_replace = cache.os.replace
    def fail_publish(source, destination):
        if Path(source).name.startswith('.full-csr-build-'):
            raise OSError('injected publish failure')
        return original_replace(source, destination)
    monkeypatch.setattr(cache.os, 'replace', fail_publish)
    with pytest.raises(OSError, match='injected'): cache.prepare_cache(tmp_path, rebuild=True)
    assert path.read_bytes() == old_manifest
    assert not list(tmp_path.glob('.full-csr-build-*'))
    assert not list(tmp_path.glob('full-csr-v1-replaced-*'))
    assert cache.load_cache(tmp_path).brain.graph_digest == original.graph_digest


def test_unrecognized_directory_and_symlink_are_never_rebuilt(tmp_path, monkeypatch):
    directory = tmp_path / cache.CACHE_DIRECTORY
    directory.mkdir()
    sentinel = directory / 'user-file.txt'
    sentinel.write_text('preserve')
    with pytest.raises(ValueError): cache.prepare_cache(tmp_path, rebuild=True)
    assert sentinel.read_text() == 'preserve'
    sentinel.unlink(); directory.rmdir()
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    directory.symlink_to(unrelated, target_is_directory=True)
    with pytest.raises(ValueError): cache.prepare_cache(tmp_path, rebuild=True)
    assert directory.is_symlink() and unrelated.is_dir()


def test_cache_subprocess_uses_only_fixed_code_and_separate_arguments(tmp_path, monkeypatch):
    calls = []
    def failed_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1, stderr='injected child failure')
    monkeypatch.setattr(cache.subprocess, 'run', failed_run)
    hostile = tmp_path / 'space; apparent command.py'
    with pytest.raises(RuntimeError, match='injected child failure'): cache.ensure_cache(hostile)
    command, kwargs = calls[0]
    assert kwargs.get('shell', False) is False
    assert command[-1] == str(hostile.resolve()) and command[-2] == '--data-dir'
    assert 'apparent command' not in command[2]
    assert kwargs['timeout'] == 600
