"""Verified complete-graph CSR preparation in a separate, bounded-lived process."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import psutil

from .data import (_read_roots, _verify_sources, load_graph, read_manifest,
                   sha256_file)
from .interface import SensoryEncoder
from .neural import SparseBrain
from .registry import Registry, default_path

CACHE_SCHEMA = "neuroterrarium.complete-csr-cache.v1"
CACHE_DIRECTORY = "full-csr-v1"
ARRAYS = {"root_ids": "<i8", "indptr": "<i8", "post_indices": "<i4",
          "weights_mV": "<f8", "input_mask": "|b1"}


def _read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("CSR manifest missing, linked or oversized; run data prepare --rebuild")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:raise ValueError("duplicate CSR manifest field")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite CSR metadata")))
    if not isinstance(value, dict):raise ValueError("CSR manifest must be an object")
    return value


def _identity(manifest_path=None):
    configuration = default_path().parent
    path = Path(manifest_path) if manifest_path else configuration / "data-v783.json"
    data = read_manifest(path)
    profile = _read_json(configuration / "brain-storage-v1.json")
    if (profile["schema"] != "neuroterrarium.complete-csr-profile.v1"
            or profile["profile"] != data["profile"]
            or any(profile[key] != data["expected"][key]
                   for key in ("neurons", "directed_pairs", "synaptic_contacts"))):
        raise ValueError("full CSR storage and source profile mismatch")
    modules = Path(__file__).parent
    identity = {"data_manifest_sha256": sha256_file(path),
                "storage_profile_sha256": sha256_file(configuration / "brain-storage-v1.json"),
                "registry_sha256": sha256_file(configuration / "interface-v783.json"),
                "interface_sha256": sha256_file(configuration / "brain-interface.json"),
                "source_sha256": {name: sha256_file(modules / name) for name in
                    ("brain_cache.py", "data.py", "neural.py", "reference.py", "registry.py", "interface.py")},
                "profile": profile["profile"], "graph_digest": profile["graph_digest"]}
    return path, data, profile, identity


def _summary_digest(summary):
    return hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CachedBrain:
    root_ids: np.ndarray
    summary: dict
    brain: SparseBrain
    manifest: dict


def _verified_arrays(data_dir, manifest_path=None):
    data_dir = Path(data_dir)
    _, data, profile, identity = _identity(manifest_path)
    originals = _verify_sources(data_dir, data)
    directory = data_dir / CACHE_DIRECTORY
    if directory.is_symlink():raise ValueError("CSR cache must not be a symbolic link")
    manifest = _read_json(directory / "manifest.json")
    if (manifest.get("schema") != CACHE_SCHEMA or manifest.get("status") != "complete"
            or manifest.get("identity") != identity or set(manifest.get("arrays", {})) != set(ARRAYS)):
        raise ValueError("CSR cache profile or source version mismatch; run data prepare --rebuild")
    n, edges = profile["neurons"], profile["directed_pairs"]
    shapes = {"root_ids": [n], "indptr": [n + 1], "post_indices": [edges],
              "weights_mV": [edges], "input_mask": [n]}
    arrays = {}
    for name, dtype in ARRAYS.items():
        item = manifest["arrays"][name]
        array_path = directory / f"{name}.npy"
        if (not isinstance(item, dict) or set(item) != {"sha256", "bytes", "dtype", "shape"}
                or item["dtype"] != dtype or item["shape"] != shapes[name]
                or array_path.is_symlink() or not array_path.is_file()
                or array_path.stat().st_size != item["bytes"]
                or item["bytes"] != int(np.prod(shapes[name])) * np.dtype(dtype).itemsize + 128
                or sha256_file(array_path) != item["sha256"]):
            raise ValueError(f"CSR {name} checksum or shape mismatch; run data prepare --rebuild")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False, max_header_size=1024)
        if list(array.shape) != shapes[name] or array.dtype.str != dtype or array.flags.writeable:
            raise ValueError("CSR array storage differs from its declared identity")
        arrays[name] = array
    roots = _read_roots(originals["completeness"], n)
    if not np.array_equal(arrays["root_ids"], roots):
        raise ValueError("CSR root ordering differs from the original complete graph")
    digest = hashlib.sha256(b"neuroterrarium.lif-csr.v1\0")
    digest.update(np.asarray([n], dtype="<i8").tobytes())
    for name in ("indptr", "post_indices", "weights_mV", "input_mask"):
        for start in range(0, len(arrays[name]), 65536):
            digest.update(arrays[name][start:start + 65536].tobytes())
    if digest.hexdigest() != profile["graph_digest"]:
        raise ValueError("CSR content differs from the frozen complete-graph digest")
    registry = Registry.load()
    registry.verify_annotations(originals["annotations"])
    groups, sides = registry.resolve(arrays["root_ids"]), registry.resolve_sides(arrays["root_ids"])
    expected_inputs = SensoryEncoder(groups, sides, 0).input_neurons
    if not np.array_equal(np.flatnonzero(arrays["input_mask"]), np.sort(expected_inputs)):
        raise ValueError("CSR injection-site mask differs from registered inputs")
    summary = manifest.get("summary")
    if (not isinstance(summary, dict) or summary.get("profile") != data["profile"]
            or any(summary.get(key) != profile[key] for key in ("neurons", "directed_pairs", "synaptic_contacts"))
            or summary.get("source_sha256") != {name: item["sha256"] for name, item in data["sources"].items()}):
        raise ValueError("CSR graph summary disagrees with the frozen source profile")
    if _summary_digest(summary) != profile["summary_sha256"]:
        raise ValueError("CSR summary metadata differs from the frozen complete-graph summary")
    return arrays, summary, manifest, profile


def verify_cache(data_dir, manifest_path=None) -> dict:
    """Check original data, identities and cache hashes without neural state."""
    _, summary, manifest, profile = _verified_arrays(data_dir, manifest_path)
    return {"status": "verified", "scope": "source and complete CSR asset identity; no dynamics executed",
            "cache": CACHE_DIRECTORY, "graph_digest": profile["graph_digest"],
            "manifest_sha256": sha256_file(Path(data_dir) / CACHE_DIRECTORY / "manifest.json"),
            "bytes": sum(item["bytes"] for item in manifest["arrays"].values()),
            "neurons": summary["neurons"], "directed_pairs": summary["directed_pairs"]}


def load_cache(data_dir, manifest_path=None) -> CachedBrain:
    """Verify source bytes and cache identities, then attach read-only storage."""
    arrays, summary, manifest, profile = _verified_arrays(data_dir, manifest_path)
    brain = SparseBrain.from_csr(arrays["indptr"], arrays["post_indices"], arrays["weights_mV"],
                                 arrays["input_mask"], expected_digest=profile["graph_digest"])
    return CachedBrain(arrays["root_ids"], summary, brain, manifest)


def _resource_guard(directory):
    available = psutil.virtual_memory().available
    if available < 3 * 2**30:
        raise ValueError(f"CSR preparation requires at least 3 GiB available system memory; current {available} bytes")
    if shutil.disk_usage(directory).free < 10 * 2**30:
        raise ValueError("CSR preparation requires at least 10 GiB free disk")


def prepare_cache(data_dir, manifest_path=None, *, rebuild=False) -> dict:
    """Run in the preparation process; atomically publish a verified full CSR."""
    import fcntl
    data_dir = Path(data_dir)
    if not data_dir.is_dir():raise ValueError("prepare the original data directory first")
    target = data_dir / CACHE_DIRECTORY
    with (data_dir / ".full-csr-v1.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists() and not rebuild:return verify_cache(data_dir, manifest_path)
        if target.is_symlink():raise ValueError("rebuild cannot replace a symbolic link")
        if target.exists() and (not target.is_dir()
                                or _read_json(target / "manifest.json").get("schema") != CACHE_SCHEMA):
            raise ValueError("rebuild requires an identified complete-CSR cache directory")
        _resource_guard(data_dir)
        path, data, profile, identity = _identity(manifest_path)
        registry = Registry.load()
        graph = load_graph(data_dir, path)
        registry.verify_annotations(data_dir / data["sources"]["annotations"]["filename"])
        groups, sides = registry.resolve(graph.root_ids), registry.resolve_sides(graph.root_ids)
        encoder = SensoryEncoder(groups, sides, 0)
        brain = SparseBrain(len(graph.root_ids), graph.pre, graph.post, graph.signed_counts,
                            input_neurons=encoder.input_neurons)
        if brain.graph_digest != profile["graph_digest"]:
            raise ValueError("constructed complete CSR differs from the frozen graph digest")
        summary = {key: value for key, value in graph.summary.items() if key != "load_seconds"}
        if _summary_digest(summary) != profile["summary_sha256"]:
            raise ValueError("constructed graph summary differs from the frozen source summary")
        temporary = Path(tempfile.mkdtemp(prefix=".full-csr-build-", dir=data_dir))
        try:
            arrays = {"root_ids": graph.root_ids, "indptr": brain.indptr,
                      "post_indices": brain.post_indices, "weights_mV": brain.weights_mV,
                      "input_mask": brain.input_mask}
            files = {}
            for name, array in arrays.items():
                output = temporary / f"{name}.npy"
                with output.open("xb") as handle:
                    np.save(handle, array, allow_pickle=False)
                    handle.flush();os.fsync(handle.fileno())
                files[name] = {"sha256": sha256_file(output), "bytes": output.stat().st_size,
                               "dtype": array.dtype.str, "shape": list(array.shape)}
            manifest = {"schema": CACHE_SCHEMA, "status": "complete", "identity": identity,
                        "summary": summary, "arrays": files,
                        "license": profile["license"], "preparation": "complete verified graph; separate process"}
            with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.flush();os.fsync(handle.fileno())
            backup = None
            if target.exists():
                backup = data_dir / f"full-csr-v1-replaced-{uuid.uuid4().hex}"
                os.replace(target, backup)
            try:os.replace(temporary, target)
            except BaseException:
                if backup is not None:os.replace(backup, target)
                raise
            return {"status": "prepared", "cache": CACHE_DIRECTORY,
                    "graph_digest": brain.graph_digest, "bytes": sum(item["bytes"] for item in files.values()),
                    "neurons": brain.n_neurons, "directed_pairs": len(brain.post_indices),
                    "replaced_cache_preserved": backup is not None}
        finally:
            if temporary.exists():shutil.rmtree(temporary)


def ensure_cache(data_dir, manifest_path=None) -> None:
    """Prepare missing storage in a child process; existing corruption is fatal."""
    if (Path(data_dir) / CACHE_DIRECTORY).exists():return
    command = [sys.executable, "-c",
               "import sys;sys.path.insert(0,sys.argv.pop(1));from neuroterrarium.brain_cache import main;raise SystemExit(main())",
               str(Path(__file__).resolve().parent.parent), "--data-dir", str(Path(data_dir).resolve())]
    if manifest_path is not None:command += ["--manifest", str(Path(manifest_path).resolve())]
    print("Preparing verified complete CSR in a separate process (about 183 MB of cache storage).", file=sys.stderr)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    if completed.returncode != 0:
        raise RuntimeError("Complete CSR preparation failed; run data prepare for details. " + completed.stderr.strip()[-500:])
    if not (Path(data_dir) / CACHE_DIRECTORY / "manifest.json").is_file():
        raise RuntimeError("Complete CSR preparation produced no verified cache")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = prepare_cache(args.data_dir, args.manifest, rebuild=args.rebuild)
        print(json.dumps(result));return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr);return 1


if __name__ == "__main__":
    raise SystemExit(main())
