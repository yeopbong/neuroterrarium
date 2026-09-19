"""Prepare pinned wheels and build an offline, relocatable macOS arm64 bundle.

Only ``prepare`` accesses the network. ``build`` requires a previously frozen
source digest, the verified wheel cache and nine completed trained models.
Installation happens in a new staging directory, never in the system Python.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

import psutil
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = {
    "schema": "neuroterrarium.python-runtime.v1", "version": "3.12.14", "release": "20260901",
    "revision": "4bb01f09aaf362c71e891be4a41cb6d6ddf830b3", "platform": "macos-arm64",
    "filename": "cpython-3.12.14+20260901-aarch64-apple-darwin-install_only.tar.gz",
    "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%2B20260901-aarch64-apple-darwin-install_only.tar.gz",
    "bytes": 25135464, "sha256": "3ee3ee547cedfeb7c2b16b2b7156039f7b470bb8f857e226fd3d2eb11db83c76",
    "source": "https://github.com/astral-sh/python-build-standalone/tree/4bb01f09aaf362c71e891be4a41cb6d6ddf830b3",
    "accessed_date": "2026-09-13",
}
RUNTIME_FULL = {
    "filename": "cpython-3.12.14+20260901-aarch64-apple-darwin-pgo+lto-full.tar.zst",
    "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%2B20260901-aarch64-apple-darwin-pgo%2Blto-full.tar.zst",
    "bytes": 54682677, "sha256": "dbefa04d4107b449e17022f9c78d3eacbd37a1aa8c166c6220dbd95addf3f318",
}
CYTHON_LICENSES = {
    "COPYING.txt": {"bytes": 756, "sha256": "e1eb1c49a8508e8173dac30157e4a6439a44ad8846194746c424fbc3fc2b95d7"},
    "LICENSE.txt": {"bytes": 10174, "sha256": "9568a2b155e66ac3e0ba1fd80b52b827b9460e6cf6f233125e7cbca8e206ddc3"},
}
CYTHON_SOURCE = "https://raw.githubusercontent.com/cython/cython/bcd04a2966842d80de6ecff4e68e8d892bec5ef7/"
MIN_AVAILABLE_BYTES = 3 * 1024**3
MIN_FREE_BYTES = 10 * 1024**3
MAX_UNPACKED_BYTES = 4 * 1024**3


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def check_resources(path: Path) -> None:
    if psutil.virtual_memory().available < MIN_AVAILABLE_BYTES:
        raise ValueError("build paused: preserve at least 3 GiB available memory")
    if shutil.disk_usage(path).free < MIN_FREE_BYTES:
        raise ValueError("build paused: preserve at least 10 GiB free disk")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("this bundle recipe has only been prepared for macOS arm64")


def source_manifest() -> dict:
    files = [ROOT / "pyproject.toml", ROOT / "LICENSE", ROOT / "THIRD_PARTY_NOTICES.md",
             Path(__file__).resolve()]
    files += sorted((ROOT / "src/neuroterrarium").rglob("*.py"))
    files += sorted((ROOT / "src/neuroterrarium/static").rglob("*"))
    files += sorted((ROOT / "configs").glob("*.json"))
    entries = {str(path.relative_to(ROOT)): digest_file(path) for path in files if path.is_file()}
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": entries}


def dependency_closure() -> list[str]:
    """Freeze the installed versions in the project's actual dependency closure."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pending = [Requirement(value) for value in project["project"]["dependencies"]]
    versions = {}
    while pending:
        requirement = pending.pop()
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(requirement.name)
        distribution = importlib.metadata.distribution(name)
        if distribution.version not in requirement.specifier:
            raise ValueError(f"installed {name} does not satisfy the project requirement")
        if name in versions:
            continue
        versions[name] = distribution.version
        pending.extend(Requirement(value) for value in distribution.requires or [])
    return [f"{name}=={version}" for name, version in sorted(versions.items())]


def verified(path: Path, spec: dict) -> None:
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != spec["bytes"]
            or digest_file(path) != spec["sha256"]):
        raise ValueError(f"asset checksum or size mismatch: {path.name}")


def fetch_asset(cache: Path, specification: dict) -> Path:
    target = cache / specification["filename"]
    if target.exists():
        verified(target, specification)
        return target
    partial = target.with_name(target.name + ".partial")
    # curl refuses to append an ignored Range response. Fixed GitHub release
    # redirects remain HTTPS; exact content is checked before publication.
    command = ["curl", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
               "--retry", "2", "--connect-timeout", "10", "--max-time", "300",
               "--continue-at", "-", "--output", str(partial), specification["url"]]
    subprocess.run(command, check=True)
    verified(partial, specification)
    os.replace(partial, target)
    return target


def prepare_licenses(cache: Path) -> None:
    archive = fetch_asset(cache, RUNTIME_FULL)
    destination = cache / "python-runtime-licenses"
    destination.mkdir(exist_ok=True)
    names = subprocess.check_output(["tar", "-tf", str(archive), "python/licenses/*"], text=True).splitlines()
    if len(names) > 64 or any(not re.fullmatch(r"python/licenses/LICENSE\.[a-zA-Z0-9_.-]+", name) for name in names):
        raise ValueError("unexpected runtime license archive layout")
    names.append("python/PYTHON.json")
    files = {}
    for name in names:
        data = subprocess.check_output(["tar", "-xOf", str(archive), name])
        if len(data) > 1024 * 1024:
            raise ValueError("runtime license or metadata exceeds 1 MiB")
        path = destination / Path(name).name
        path.write_bytes(data)
        files[path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "archive_path": name}
    write_json(destination / "manifest.json", {
        "schema": "neuroterrarium.runtime-licenses.v1", "source": RUNTIME["source"],
        "source_archive_sha256": RUNTIME_FULL["sha256"], "files": files,
        "scope": "Unmodified upstream metadata and license texts, including optional components. macOS zlib is a system library; zlib-ng is not bundled."})
    cython = cache / "cython-licenses"
    cython.mkdir(exist_ok=True)
    for name, spec in CYTHON_LICENSES.items():
        fetch_asset(cython, {**spec, "filename": name, "url": CYTHON_SOURCE + name})
    write_json(cython / "manifest.json", {"source": CYTHON_SOURCE, "version": "3.3.0", "files": CYTHON_LICENSES})


def verify_runtime_licenses(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("schema") != "neuroterrarium.runtime-licenses.v1"
            or manifest.get("source") != RUNTIME["source"]
            or manifest.get("source_archive_sha256") != RUNTIME_FULL["sha256"]
            or "LICENSE.cpython.txt" not in manifest.get("files", {})):
        raise ValueError("runtime license provenance mismatch")
    for name, spec in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("invalid license file path")
        verified(directory / name, spec)
    return manifest


def prepare(cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    check_resources(cache)
    fetch_asset(cache, RUNTIME)
    prepare_licenses(cache)
    requirements = dependency_closure()
    constraints = cache / "requirements-pinned.txt"
    constraints.write_text("\n".join(requirements) + "\n")
    wheelhouse = cache / "wheels"
    wheelhouse.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
                    "--no-deps", "--only-binary=:all:", "--dest", str(wheelhouse),
                    "--requirement", str(constraints)], check=True)
    expected = {canonicalize_name(line.split("==")[0]): line.split("==")[1] for line in requirements}
    files, found = {}, set()
    for path in sorted(wheelhouse.glob("*.whl")):
        name, version, _, _ = parse_wheel_filename(path.name)
        if name not in expected or str(version) != expected[name] or name in found:
            raise ValueError("wheel cache contains a duplicate or unrequested distribution")
        found.add(name)
        files[path.name] = {"bytes": path.stat().st_size, "sha256": digest_file(path),
                            "distribution": str(name), "version": str(version)}
    if found != set(expected):
        raise ValueError("wheel cache is incomplete")
    result = {"schema": "neuroterrarium.build-inputs.v1", "runtime": RUNTIME,
              "requirements": requirements, "wheels": files}
    write_json(cache / "build-inputs.json", result)
    return {"status": "prepared", "wheel_count": len(files),
            "downloaded_wheel_bytes": sum(item["bytes"] for item in files.values())}


def safe_extract_runtime(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        if len(members) > 20000 or sum(item.size for item in members) > 256 * 1024**2:
            raise ValueError("runtime archive exceeds expected extraction bounds")
        if any(not item.name.startswith("python/") for item in members):
            raise ValueError("runtime archive contains an unexpected root")
        stream.extractall(destination, members=members, filter="data")


def linkage_report(directory: Path) -> list[dict]:
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix not in {".so", ".dylib"} and path.name != "python3.12":
            continue
        commands = subprocess.check_output(["otool", "-l", str(path)], text=True)
        # LC_ID_DYLIB identifies this file; it is not an external dependency.
        dependencies = re.findall(
            r"cmd LC_(?:LOAD|LOAD_WEAK|REEXPORT|LAZY_LOAD|LOAD_UPWARD)_DYLIB\s+cmdsize \d+\s+name (.+?) \(offset", commands)
        if any(not value.startswith(("/System/", "/usr/lib/", "@loader_path/", "@rpath/", "@executable_path/"))
               for value in dependencies):
            raise ValueError(f"nonportable dynamic dependency: {path.name}")
        rpaths = re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (.+?) \(offset", commands)
        # Official wheels can retain unused cross-build RPATH candidates. Keep
        # them visible and verify actual loaded libraries during release tests;
        # changing signed third-party Mach-O files is not part of packaging.
        result.append({"file": str(path.relative_to(directory)), "dependencies": dependencies,
                       "declared_search_paths": rpaths})
    return result


LAUNCHER = '''#!/bin/sh
set -eu
bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$bundle_dir"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$bundle_dir/cache/numba"
exec "$bundle_dir/python/bin/python3.12" -I -B -m neuroterrarium "$@"
'''


def _clean_generated_paths(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file() and (path.suffix in {".pyc", ".nbc", ".nbi"} or path.name == "direct_url.json"):
            path.unlink()
    # Runtime use has one relative launcher. pip-generated console scripts carry
    # staging paths, and the optional interpreter helper scripts are unnecessary.
    for path in (directory / "python/bin").iterdir():
        if path.name not in {"python", "python3", "python3.12"}:
            if path.is_dir():
                raise ValueError("unexpected directory in interpreter bin")
            path.unlink()


def _privacy_check(directory: Path) -> None:
    forbidden = [str(Path.home()).encode(), str(ROOT).encode(), str(Path(sys.base_prefix)).encode()]
    overlap = max(map(len, forbidden))
    for path in directory.rglob("*"):
        if path.is_symlink():
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("bundle contains an escaping symbolic link")
        elif path.is_file():
            previous = b""
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    block = previous + chunk
                    if any(value in block for value in forbidden):
                        raise ValueError(f"local build path embedded in {path.relative_to(directory)}")
                    previous = block[-overlap:]


def _file_manifest(directory: Path) -> dict:
    files = {}
    total = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            files[str(path.relative_to(directory))] = {"symlink": os.readlink(path)}
        elif path.is_file():
            total += path.stat().st_size
            files[str(path.relative_to(directory))] = {"bytes": path.stat().st_size, "sha256": digest_file(path)}
        if total > MAX_UNPACKED_BYTES or len(files) > 100000:
            raise ValueError("bundle exceeds file count or unpacked size budget")
    return files


def _archive(directory: Path, output: Path) -> None:
    partial = output.with_name(output.name + ".partial")

    def normalize(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.pax_headers = {}
        return info

    with (partial.open("wb") as raw,
          gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0, compresslevel=6) as zipped,
          tarfile.open(fileobj=zipped, mode="w") as stream):
        stream.add(directory, arcname="NeuroTerrarium", filter=normalize)
    os.replace(partial, output)


def build(cache: Path, models: Path, output: Path, source_sha256: str, runtime_licenses: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    check_resources(output.parent)
    if output.exists():
        raise ValueError("release archives are immutable; choose a new output path")
    source = source_manifest()
    if source["sha256"] != source_sha256:
        raise ValueError("current package source differs from the frozen source digest")
    if not (ROOT / "src/neuroterrarium/static/index.html").is_file():
        raise ValueError("build and package the actual web application first")
    inputs = json.loads((cache / "build-inputs.json").read_text())
    if inputs.get("schema") != "neuroterrarium.build-inputs.v1" or inputs.get("runtime") != RUNTIME:
        raise ValueError("build input manifest mismatch")
    verified(cache / RUNTIME["filename"], RUNTIME)
    wheels = []
    for name, spec in inputs["wheels"].items():
        if Path(name).name != name or not name.endswith(".whl"):
            raise ValueError("invalid wheel filename")
        path = cache / "wheels" / name
        verified(path, spec)
        wheels.append(path)
    sys.path.insert(0, str(ROOT / "src"))
    from neuroterrarium.runtime import load_model_set
    _, models_manifest = load_model_set(models)
    license_manifest = verify_runtime_licenses(runtime_licenses)
    for name, spec in CYTHON_LICENSES.items():
        verified(cache / "cython-licenses" / name, spec)
    with tempfile.TemporaryDirectory(prefix="neuroterrarium-build-", dir=output.parent) as temporary:
        work = Path(temporary)
        bundle = work / "NeuroTerrarium"
        bundle.mkdir()
        safe_extract_runtime(cache / RUNTIME["filename"], bundle)
        wheel_dir = work / "project-wheel"
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                        "--no-index", "--wheel-dir", str(wheel_dir), str(ROOT)], check=True)
        project_wheels = list(wheel_dir.glob("*.whl"))
        if len(project_wheels) != 1 or source_manifest() != source:
            raise ValueError("source changed while building the package wheel")
        with zipfile.ZipFile(project_wheels[0]) as archive:
            if "neuroterrarium/static/index.html" not in archive.namelist():
                raise ValueError("project wheel is missing the application UI")
        python = bundle / "python/bin/python3.12"
        subprocess.run([str(python), "-I", "-B", "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
                        *map(str, wheels), str(project_wheels[0])], check=True)
        subprocess.run([str(python), "-I", "-B", "-m", "pip", "check"], check=True)
        check_resources(output.parent)
        model_destination = bundle / "artifacts/models"
        model_destination.mkdir(parents=True)
        shutil.copy2(models / "manifest.json", model_destination / "manifest.json")
        for item in models_manifest["models"]:
            source_checkpoint = models / item["checkpoint"]
            destination_checkpoint = model_destination / item["checkpoint"]
            destination_checkpoint.mkdir()
            for name in ("manifest.json", "state.json", "policy.safetensors", "optimizer.safetensors", "runtime.safetensors"):
                original = source_checkpoint / name
                if original.is_symlink():
                    raise ValueError("checkpoint payload must not be a symbolic link")
                shutil.copy2(original, destination_checkpoint / name)
        shutil.copytree(runtime_licenses, bundle / "licenses/python-runtime")
        shutil.copytree(cache / "cython-licenses", bundle / "licenses/cython-3.3.0")
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            shutil.copy2(ROOT / name, bundle / name)
        (bundle / "neuroterrarium").write_text(LAUNCHER)
        (bundle / "neuroterrarium").chmod(0o755)
        (bundle / "INSTALL.txt").write_text(
            "NeuroTerrarium for macOS arm64\n\n"
            "This archive is not signed or notarized. No system certification is claimed.\n"
            "Unpack into a writable folder. First prepare public data:\n"
            "  ./neuroterrarium data fetch\n"
            "Then start the local application:\n"
            "  ./neuroterrarium app --open-browser\n"
            "After preparation, runtime and verification work without networking.\n"
            "Data retain CC BY-NC 4.0 terms; application code is MIT.\n"
            "Dependency licenses remain with their distributions and in licenses/.\n"
            "The bundled interpreter omits optional console helper scripts; its standard library is unchanged.\n"
            "Only the release's recorded macOS version and hardware were tested.\n")
        _clean_generated_paths(bundle)
        links = linkage_report(bundle)
        _privacy_check(bundle)
        manifest = {"schema": "neuroterrarium.installation.v1", "platform": "macos-arm64",
                    "source": source, "runtime": RUNTIME, "wheels": inputs["wheels"],
                    "runtime_license_manifest_sha256": digest_file(runtime_licenses / "manifest.json"),
                    "runtime_license_files": len(license_manifest["files"]),
                    "model_set_manifest_sha256": digest_file(models / "manifest.json"),
                    "trained_models": len(models_manifest["models"]), "dynamic_linkage": links,
                    "files": _file_manifest(bundle), "signature": "unsigned", "notarization": "none",
                    "validation": "requires clean relocation, offline runtime and release validation"}
        write_json(bundle / "installation.json", manifest)
        subprocess.run([str(bundle / "neuroterrarium"), "--help"], cwd=work, check=True,
                       stdout=subprocess.DEVNULL)
        _archive(bundle, output)
    result = {"status": "built", "filename": output.name, "bytes": output.stat().st_size,
              "sha256": digest_file(output), "source_sha256": source_sha256,
              "scope": "archive built; release validation is a separate required step"}
    write_json(output.with_name(output.name + ".json"), result)
    output.with_name(output.name + ".sha256").write_text(f"{result['sha256']}  {output.name}\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source-digest", help="report the source to freeze before a build")
    preparation = commands.add_parser("prepare", help="explicitly download pinned runtime and dependency wheels")
    preparation.add_argument("--cache", type=Path, required=True)
    packaging = commands.add_parser("build", help="build offline from frozen inputs; does not certify the release")
    packaging.add_argument("--cache", type=Path, required=True)
    packaging.add_argument("--models", type=Path, required=True)
    packaging.add_argument("--output", type=Path, required=True)
    packaging.add_argument("--source-sha256", required=True)
    packaging.add_argument("--runtime-licenses", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "source-digest":
        result = source_manifest()
    elif args.command == "prepare":
        result = prepare(args.cache.resolve())
    else:
        result = build(args.cache.resolve(), args.models.resolve(), args.output.resolve(),
                       args.source_sha256, args.runtime_licenses.resolve())
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
