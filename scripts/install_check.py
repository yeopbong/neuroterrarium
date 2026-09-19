"""Verify a wheel in a new isolated environment and a non-ASCII working path.

With --wheelhouse, installation and runtime checks are entirely offline. Without
it, only installation may download dependencies; Linux installs the official
CPU-only PyTorch wheel first. Runtime probes deny Python socket connections and
DNS resolution. They run the actual prepared data and nine trained policies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

MIN_AVAILABLE_BYTES = 3 * 1024**3
MIN_FREE_BYTES = 10 * 1024**3


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


PROBE = r'''
import json
import pathlib
import socket
import sys

network_attempts = []
def deny_network(*args, **kwargs):
    network_attempts.append("blocked outbound operation")
    raise RuntimeError("network access forbidden during installed runtime validation")

class OfflineSocket(socket.socket):
    connect = deny_network
    connect_ex = deny_network

socket.socket = OfflineSocket
socket.create_connection = deny_network
socket.getaddrinfo = deny_network
try:
    socket.create_connection(("127.0.0.1", 9))
except RuntimeError:
    pass
else:
    raise AssertionError("network guard was not effective")
network_attempts.clear()

import neuroterrarium
from neuroterrarium.cli import main
from neuroterrarium.download import config_path

prefix = pathlib.Path(sys.prefix).resolve()
package = pathlib.Path(neuroterrarium.__file__).resolve().parent
assert package.is_relative_to(prefix), "package did not load from the isolated environment"
assert not sys.flags.ignore_environment == 0, "interpreter must ignore ambient Python configuration"

if sys.argv[1] == "contents":
    profiles = {}
    for name in ("data-v783.json", "interface-v783.json", "brain-interface.json", "brain-storage-v1.json"):
        path = config_path(name)
        assert path.resolve().is_relative_to(prefix), "configuration came from a source checkout"
        profiles[name] = json.loads(path.read_text())["schema"]
    index = package / "static/index.html"
    assert index.is_file() and index.stat().st_size > 0, "installed application UI missing"
    assert any((package / "static").rglob("*.js")), "installed JavaScript application missing"
    print(json.dumps({"status": "passed", "version": neuroterrarium.__version__,
        "package_location": str(package.relative_to(prefix)), "profiles": profiles,
        "ui": "installed HTML and JavaScript present", "network_guard": "Python outbound sockets and DNS denied"}))
    result = 0
else:
    result = main(sys.argv[1:])
assert not network_attempts, "runtime attempted network access"
raise SystemExit(result)
'''


class InstallationCheckError(RuntimeError):
    """A required installation step failed; its evidence remains on disk."""


def pick_wheel(path: Path) -> Path:
    if path.is_dir():
        candidates = sorted(path.glob("neuroterrarium-*.whl"))
        if len(candidates) != 1:
            raise ValueError("wheel directory must contain exactly one NeuroTerrarium wheel")
        path = candidates[0]
    if not path.is_file() or path.is_symlink() or path.suffix != ".whl":
        raise ValueError("a regular built wheel file is required")
    return path.resolve()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_check(wheel: Path, data_dir: Path, models: Path, output: Path,
              wheelhouse: Path | None = None) -> dict:
    wheel = pick_wheel(wheel)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("installation evidence directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    if psutil.virtual_memory().available < MIN_AVAILABLE_BYTES or shutil.disk_usage(output).free < MIN_FREE_BYTES:
        raise ValueError("installation check requires 3 GiB available memory and 10 GiB free disk")
    if not data_dir.is_dir() or not (models / "manifest.json").is_file():
        raise ValueError("prepare real data and the completed nine-model set before installation validation")
    data_dir, models = data_dir.resolve(), models.resolve()
    environment_dir = output / "isolated environment"
    working = output / "干净 working directory"
    inputs = output / "inputs"
    working.mkdir()
    inputs.mkdir()
    shutil.copy2(wheel, inputs / wheel.name)
    copied_wheel = inputs / wheel.name
    if checksum(copied_wheel) != checksum(wheel):
        raise ValueError("copied wheel checksum differs")
    # Only selected paths are normalized in public evidence. Original log bytes
    # are hashed before normalization, and their text is never presented as raw.
    replacements = [(str(output), "<installation>"), (str(data_dir), "<data-cache>"),
                    (str(models), "<trained-models>"), (str(Path.cwd()), "<workspace>"),
                    (str(Path(sys.base_prefix)), "<host-python>"), (str(Path.home()), "<home>")]

    def redact(value: str) -> str:
        for original, replacement in replacements:
            value = value.replace(original, replacement)
        return value

    environment = {"PATH": os.defpath, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "NUMBA_NUM_THREADS": "1", "NUMBA_CACHE_DIR": str(output / "cache/numba"),
                   "PIP_CONFIG_FILE": os.devnull, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    summary = {"schema": "neuroterrarium.installation-check.v1", "status": "running",
               "wheel": {"filename": wheel.name, "sha256": checksum(wheel)},
               "platform": {"system": platform.system(), "machine": platform.machine(),
                            "release": platform.release(), "python": platform.python_version()},
               "installation_network": "offline wheelhouse" if wheelhouse else "public dependency indexes",
               "runtime_network": "outbound Python socket operations and DNS denied in each fresh process",
               "scope": "new environment, installed resources, actual full graph and nine-model runtime",
               "path_case": "spaces and non-ASCII characters", "steps": []}
    started = time.perf_counter()

    def record(name: str, command: list[str], timeout: int = 300) -> None:
        if psutil.virtual_memory().available < MIN_AVAILABLE_BYTES:
            raise InstallationCheckError("resource reserve reached before the next command")
        before = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=working, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=timeout, check=False)
            raw, code, state = completed.stdout, completed.returncode, "completed"
        except subprocess.TimeoutExpired as error:
            raw, code, state = error.stdout or b"", None, "timeout"
        text = redact(raw.decode("utf-8", errors="replace"))
        log = output / f"{len(summary['steps']) + 1:02d}-{name}.log"
        log.write_text(text, encoding="utf-8")
        summary["steps"].append({"name": name, "command": [redact(value) for value in command],
            "cwd": "<installation>/干净 working directory", "exit_code": code, "state": state,
            "wall_seconds": time.perf_counter() - before, "log": log.name,
            "log_sha256": checksum(log), "original_output_sha256": hashlib.sha256(raw).hexdigest(),
            "log_redactions": "local workspace, cache, account and interpreter paths"})
        atomic_json(output / "summary.json", summary)
        if code != 0:
            raise InstallationCheckError(f"required command failed: {name} ({state}, exit {code})")

    try:
        # Use the running interpreter to create a truly separate site-packages.
        record("create-environment", [sys.executable, "-I", "-m", "venv", str(environment_dir)])
        python = str(environment_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        pip = [python, "-I", "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-cache-dir"]
        if wheelhouse is not None:
            wheelhouse = wheelhouse.resolve()
            lock_path = wheelhouse.parent / "build-inputs.json"
            if not lock_path.is_file():
                raise ValueError("offline wheelhouse requires its build-inputs.json hash manifest")
            lock = json.loads(lock_path.read_text())
            if lock.get("schema") != "neuroterrarium.build-inputs.v1":
                raise ValueError("unsupported wheelhouse manifest")
            local_wheels = inputs / "wheels"
            local_wheels.mkdir()
            for name, specification in lock["wheels"].items():
                if Path(name).name != name or not name.endswith(".whl"):
                    raise ValueError("invalid dependency wheel filename")
                path = wheelhouse / name
                if (path.is_symlink() or path.stat().st_size != specification["bytes"]
                        or checksum(path) != specification["sha256"]):
                    raise ValueError("dependency wheel integrity failure")
                shutil.copy2(path, local_wheels / name)
            summary["wheelhouse_manifest_sha256"] = checksum(lock_path)
            record("install-offline", pip + ["install", "--no-index", "--no-compile", "--find-links", str(local_wheels),
                                            str(copied_wheel)], timeout=600)
        else:
            if platform.system() == "Linux":
                record("install-cpu-torch", pip + ["install", "--no-compile", "torch==2.14.0",
                    "--index-url", "https://download.pytorch.org/whl/cpu"], timeout=900)
            record("install-wheel", pip + ["install", "--no-compile", str(copied_wheel)], timeout=900)
        record("dependency-check", pip + ["check"])
        probe = working / "installed_probe.py"
        probe.write_text(PROBE)
        runner = [python, "-I", "-B", str(probe)]
        record("installed-contents", runner + ["contents"])
        record("installed-cli-help", [python, "-I", "-B", "-m", "neuroterrarium", "--help"])
        record("offline-doctor", runner + ["doctor", "--data-dir", str(data_dir), "--models", str(models)])
        record("offline-complete-data", runner + ["data", "verify", "--data-dir", str(data_dir)])
        record("offline-runtime-validation", runner + ["validate", "--data-dir", str(data_dir), "--models", str(models),
                                                       "--output", str(output / "runtime-evidence")], timeout=600)
        summary["status"] = "passed"
    except (InstallationCheckError, OSError, ValueError) as error:
        summary["status"] = "failed"
        summary["failure"] = {"type": type(error).__name__, "message": redact(str(error))}
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    finally:
        summary["wall_seconds"] = time.perf_counter() - started
        atomic_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path)
    args = parser.parse_args()
    try:
        summary = run_check(args.wheel, args.data_dir, args.models, args.output, args.wheelhouse)
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "not_run", "error": type(error).__name__}))
        return 2
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
