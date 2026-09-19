"""Cache integrity and lifecycle tests use explicitly synthetic small graphs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from neuroterrarium import brain_cache as cache
from neuroterrarium.data import GraphData, sha256_file
from neuroterrarium.interface import SensoryEncoder
from neuroterrarium.neural import SparseBrain
from neuroterrarium.registry import Registry


def readonly(array):
    return np.frombuffer(array.tobytes(), dtype=array.dtype)


def test_csr_constructor_matches_all_active_spikes_traces_and_snapshot(tmp_path):
    original = SparseBrain(3, [0, 1, 2], [1, 2, 0], [40, -20, 10], input_neurons=[0])
    arrays = {}
    for name in ("indptr", "post_indices", "weights_mV", "input_mask"):
        path = tmp_path / (name + ".npy")
        np.save(path, getattr(original, name), allow_pickle=False)
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    mapped = SparseBrain.from_csr(**arrays, expected_digest=original.graph_digest)
    assert all(isinstance(getattr(mapped, name), np.memmap) for name in arrays)
    tape = [[0] if step % 50 == 0 else [] for step in range(200)]
    for _ in range(3):
        a = original.advance(tape, [0, 1, 2], True)
        b = mapped.advance(tape, [0, 1, 2], True)
        assert a.spike_steps.size > 0
        for name in ("spike_steps", "spike_neurons", "spike_counts", "v_mV", "g_mV"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert original.snapshot() == mapped.snapshot()
    clone = mapped.fork()
    assert clone.post_indices is mapped.post_indices
    assert clone.v_mV is not mapped.v_mV


@pytest.mark.parametrize("mutation", ["dtype", "writeable", "offset", "target", "weight", "digest"])
def test_csr_constructor_rejects_unverified_storage(mutation):
    brain = SparseBrain(2, [0], [1], [20], input_neurons=[0])
    arrays = {name: readonly(getattr(brain, name)) for name in ("indptr", "post_indices", "weights_mV", "input_mask")}
    digest = brain.graph_digest
    if mutation == "dtype":arrays["post_indices"] = readonly(np.array([1], dtype=np.int64))
    if mutation == "writeable":arrays["weights_mV"] = np.array([5.5])
    if mutation == "offset":arrays["indptr"] = readonly(np.array([0, 2, 1], dtype=np.int64))
    if mutation == "target":arrays["post_indices"] = readonly(np.array([2], dtype=np.int32))
    if mutation == "weight":arrays["weights_mV"] = readonly(np.array([float("nan")]))
    if mutation == "digest":digest = "0" * 64
    with pytest.raises(ValueError):SparseBrain.from_csr(**arrays, expected_digest=digest)


def test_readonly_view_with_writeable_base_cannot_mutate_attached_csr():
    brain = SparseBrain(2, [0], [1], [20], input_neurons=[0])
    backing = brain.weights_mV.copy()
    alias = backing.view();alias.flags.writeable = False
    mapped = SparseBrain.from_csr(brain.indptr, brain.post_indices, alias, brain.input_mask,
                                 expected_digest=brain.graph_digest)
    backing[0] = 0
    assert mapped.weights_mV[0] == 20 * .275
    with pytest.raises(ValueError):mapped.weights_mV.flags.writeable = True


@pytest.fixture
def small_cache(tmp_path, monkeypatch):
    registry = Registry.load()
    roots = np.array([int(node["root_id"]) for group in registry.groups.values()
                      for node in group["neurons"]], dtype=np.int64)
    pre, post, signed = np.array([0, 1], dtype=np.int32), np.array([1, 2], dtype=np.int32), np.array([40, -20], dtype=np.int32)
    root_path = tmp_path / "roots.csv"
    root_path.write_text(",Completed\n" + "".join(f"{root},True\n" for root in roots))
    annotations = tmp_path / "annotations.tsv"
    annotations.write_text("synthetic-test-only")
    sources = {"completeness": {"filename": root_path.name, "sha256": sha256_file(root_path)},
               "annotations": {"filename": annotations.name, "sha256": sha256_file(annotations)}}
    summary = {"profile": "synthetic-test-only", "neurons": len(roots), "directed_pairs": 2,
               "synaptic_contacts": 60, "source_sha256": {name: item["sha256"] for name, item in sources.items()}}
    graph = GraphData(roots, pre, post, signed, summary)
    encoder = SensoryEncoder(registry.resolve(roots), registry.resolve_sides(roots), 0)
    brain = SparseBrain(len(roots), pre, post, signed, input_neurons=encoder.input_neurons)
    data = {"profile": summary["profile"], "sources": sources}
    profile = {**summary, "graph_digest": brain.graph_digest, "license": "synthetic test fixture",
               "summary_sha256": cache._summary_digest(summary)}
    identity = {"profile": "synthetic-test-only", "graph_digest": brain.graph_digest}
    monkeypatch.setattr(cache, "_identity", lambda path=None: (root_path, data, profile, identity))
    monkeypatch.setattr(cache, "_verify_sources", lambda directory, manifest: {"completeness": root_path, "annotations": annotations})
    monkeypatch.setattr(cache, "load_graph", lambda directory, path: graph)
    monkeypatch.setattr(cache, "_resource_guard", lambda directory: None)
    monkeypatch.setattr(Registry, "verify_annotations", lambda self, path: None)
    return tmp_path, graph


def test_cache_build_load_verify_reuse_and_explicit_rebuild(small_cache):
    directory, graph = small_cache
    first = cache.prepare_cache(directory)
    assert first["status"] == "prepared" and first["directed_pairs"] == len(graph.pre)
    loaded = cache.load_cache(directory)
    assert isinstance(loaded.brain.weights_mV, np.memmap)
    assert not loaded.root_ids.flags.writeable
    assert cache.verify_cache(directory)["status"] == "verified"
    assert cache.prepare_cache(directory)["status"] == "verified"
    rebuilt = cache.prepare_cache(directory, rebuild=True)
    assert rebuilt["replaced_cache_preserved"]
    assert len(list(directory.glob("full-csr-v1-replaced-*"))) == 1


@pytest.mark.parametrize("mutation", ["bytes", "root", "wrong_version", "pickle", "self_rehashed_weight", "summary"])
def test_cache_rejects_corruption_even_if_local_array_hash_is_rewritten(small_cache, mutation):
    directory, _ = small_cache
    cache.prepare_cache(directory)
    target = directory / cache.CACHE_DIRECTORY
    manifest = json.loads((target / "manifest.json").read_text())
    name = "weights_mV"
    if mutation == "wrong_version":manifest["identity"]["profile"] = "different"
    elif mutation == "summary":manifest["summary"]["connectome_version"] = "630"
    elif mutation == "bytes":
        with (target / "weights_mV.npy").open("ab") as handle:handle.write(b"x")
    else:
        if mutation == "root":name = "root_ids"
        path = target / (name + ".npy")
        values = np.load(path, allow_pickle=False).copy()
        if mutation == "pickle":values = np.array([{"script": "not executable"}], dtype=object)
        elif mutation == "root":values[0] += 1
        else:values[0] *= 2
        np.save(path, values, allow_pickle=mutation == "pickle")
        manifest["arrays"][name]["sha256"] = sha256_file(path)
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):cache.load_cache(directory)
    with pytest.raises(ValueError):cache.verify_cache(directory)


def test_failed_build_cannot_publish_an_incomplete_cache(small_cache, monkeypatch):
    directory, _ = small_cache
    monkeypatch.setattr(np, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("interrupted write")))
    with pytest.raises(OSError):cache.prepare_cache(directory)
    assert not (directory / cache.CACHE_DIRECTORY).exists()
    assert not list(directory.glob(".full-csr-build-*"))


def test_rebuild_rejects_unidentified_directory_and_symbolic_link(small_cache):
    directory, _ = small_cache
    target = directory / cache.CACHE_DIRECTORY
    target.mkdir()
    (target / "manifest.json").write_text('{"schema":"unrelated"}')
    with pytest.raises(ValueError, match="identified"):cache.prepare_cache(directory, rebuild=True)


def test_resource_guards_are_not_relaxed(monkeypatch, tmp_path):
    monkeypatch.setattr(cache.psutil, "virtual_memory", lambda: SimpleNamespace(available=3 * 2**30 - 1))
    with pytest.raises(ValueError, match="3 GiB"):cache._resource_guard(tmp_path)


def test_missing_cache_prepares_only_in_a_child_and_failure_is_explicit(monkeypatch, tmp_path):
    calls = []
    def child(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1, stderr="DataValidationError: missing source data")
    monkeypatch.setattr(cache.subprocess, "run", child)
    with pytest.raises(RuntimeError, match="missing source"):cache.ensure_cache(tmp_path)
    assert calls[0][0][0] == cache.sys.executable
    assert calls[0][1]["timeout"] == 600
    assert "shell" not in calls[0][1]
