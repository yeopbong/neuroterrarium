"""Offline-first command-line entry points for preparation and local use."""

from __future__ import annotations

import argparse
import importlib
import ipaddress
import json
import platform
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .download import config_path


class CommandError(ValueError):
    """The requested operation is unavailable or has invalid arguments."""


def _loopback(value: str) -> str:
    if value == "localhost":
        return value
    try:
        if ipaddress.ip_address(value).is_loopback:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("the local application must bind to a loopback host")


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1024 and 65535")
    return port


def _emit(value: dict) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


def _data_options(parser: argparse.ArgumentParser, *, manifest: bool = True) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="local data cache (default: ./data)")
    if manifest:
        parser.add_argument("--manifest", type=Path, help="frozen data manifest; defaults to bundled v783")


def _models_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", dest="models_dir", type=Path, default=Path("artifacts/models"),
                        help="prepared nine-model set (default: ./artifacts/models)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuroterrarium", description="Connectome-constrained behavior sandbox")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="report current-process resources and preparation readiness")
    _data_options(doctor)
    _models_option(doctor)
    data = commands.add_parser("data", help="explicit data preparation or offline verification")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch = data_commands.add_parser("fetch", help="download pinned data; this command uses the network")
    _data_options(fetch)
    fetch.add_argument("--restart", action="store_true", help="restart only matching identified partial downloads")
    verify = data_commands.add_parser("verify", help="verify actual complete graph, identities and registered interface")
    _data_options(verify)
    prepare = data_commands.add_parser("prepare", help="build the validated full-graph sparse cache from local data, offline")
    _data_options(prepare)
    prepare.add_argument("--rebuild", action="store_true", help="replace only the identified derived graph cache")
    app = commands.add_parser("app", help="run the prepared local application without downloading or training")
    _data_options(app, manifest=False)
    _models_option(app)
    app.add_argument("--host", type=_loopback, default="127.0.0.1")
    app.add_argument("--port", type=_port, default=8765)
    app.add_argument("--open-browser", action="store_true")
    train = commands.add_parser("train", help="run or resume explicit PPO training")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", dest="output_dir", type=Path, required=True)
    train.add_argument("--architecture", choices=("feedforward", "recurrent", "hybrid"), default="feedforward")
    train.add_argument("--seed", type=int, default=101)
    train.add_argument("--resume", dest="resume_from", type=Path)
    train.add_argument("--stop-after-updates", type=int)
    train.add_argument("--cancel-file", type=Path)
    evaluate = commands.add_parser("evaluate", help="execute a frozen evaluation protocol")
    _data_options(evaluate, manifest=False)
    _models_option(evaluate)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--output", dest="output_dir", type=Path, required=True)
    evaluate.add_argument("--cancel-file", type=Path, help="stop safely when this local file exists")
    replay = commands.add_parser("replay", help="inspect a recorded frame replay")
    replay.add_argument("recording", type=Path)
    replay.add_argument("--port", type=_port, default=8766)
    replay.add_argument("--open-browser", action="store_true")
    validate = commands.add_parser("validate", aliases=["verify"], help="validate the complete graph and nine-model runtime in a bounded run")
    _data_options(validate, manifest=False)
    _models_option(validate)
    validate.add_argument("--output", dest="output_dir", type=Path, required=True)
    return parser


def _gpu_status() -> dict:
    try:
        import torch
    except ImportError:
        return {"scope": "current process", "status": "unavailable", "reason": "torch not installed"}
    return {"scope": "current process; sandbox permissions can affect availability",
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available())}


def doctor(data_dir: Path, models_dir: Path, manifest_path: Path | None = None) -> dict:
    """Report process-visible resources without hostnames, paths or identifiers."""
    import psutil

    from .data import read_manifest
    from .download import verify_file

    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    readiness = {"status": "not_ready", "scope": "local asset bytes and SHA-256; no simulation test"}
    try:
        manifest = read_manifest(manifest_path or config_path())
        data_results = {name: verify_file(data_dir / source["filename"], source)
                        for name, source in manifest["sources"].items()}
        readiness["data"] = {"status": "verified", "files": data_results}
    except (ValueError, OSError) as error:
        readiness["data"] = {"status": "not_ready", "reason": type(error).__name__}
    try:
        from .brain_cache import verify_cache
        readiness["sparse_cache"] = verify_cache(data_dir, manifest_path=manifest_path)
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as error:
        readiness["sparse_cache"] = {"status": "not_ready", "reason": type(error).__name__,
                                     "preparation": "data prepare builds the complete cache from verified local files"}
    try:
        from .runtime import load_model_set
        policies, model_manifest = load_model_set(models_dir)
        readiness["models"] = {"status": "verified", "loaded_models": len(policies),
                               "scope": "independent identities, checkpoint hashes and actual policy loading",
                               "transitions_per_model": model_manifest["transitions_per_model"]}
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as error:
        readiness["models"] = {"status": "not_ready", "reason": type(error).__name__}
    if all(readiness[key]["status"] == "verified" for key in ("data", "sparse_cache", "models")):
        readiness["status"] = "ready"
    return {"schema": "neuroterrarium.doctor.v1", "status": readiness["status"],
            "platform": {"system": platform.system(), "release": platform.release(),
                         "architecture": platform.machine(), "python": platform.python_version()},
            "cpu": {"logical": psutil.cpu_count(), "physical": psutil.cpu_count(logical=False)},
            "memory": {"total_bytes": memory.total, "available_bytes": memory.available},
            "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
            "gpu": _gpu_status(), "readiness": readiness}


def verify_data(data_dir: Path, manifest_path: Path | None = None) -> dict:
    from .data import load_graph, read_manifest
    from .registry import Registry

    path = manifest_path or config_path()
    manifest = read_manifest(path)
    registry = Registry.load()
    graph = load_graph(data_dir, path)
    registry.verify_annotations(data_dir / manifest["sources"]["annotations"]["filename"])
    groups = registry.resolve(graph.root_ids)
    return {"status": "verified", "scope": "complete graph and registered interface",
            "graph": graph.summary, "registered_neurons": {name: len(values) for name, values in groups.items()},
            "interface_sha256": registry.digest}


def _entry(module: str, function: str):
    try:
        loaded = importlib.import_module(f"neuroterrarium.{module}")
    except ImportError as error:
        raise CommandError(f"{module} is unavailable ({type(error).__name__}); operation was not performed") from error
    result = getattr(loaded, function, None)
    if not callable(result):
        raise CommandError(f"{module}.{function} is unavailable; operation was not performed")
    return result


def _fetch(args: argparse.Namespace) -> dict:
    from .data import read_manifest
    from .download import fetch_data

    path = args.manifest or config_path()
    manifest = read_manifest(path)
    print(f"Data cache: {args.data_dir.resolve()}", file=sys.stderr)
    print(f"FlyWire data license: {manifest['licenses']['flywire_data']} "
          f"({manifest['licenses']['flywire_data_license_url']}); code license is separate.", file=sys.stderr)
    print(f"Frozen download size: {sum(source['bytes'] for source in manifest['sources'].values()):,} bytes", file=sys.stderr)
    last = {"time": 0.0, "name": ""}

    def progress(name: str, done: int, total: int) -> None:
        now = time.monotonic()
        if name != last["name"] or done == total or now - last["time"] >= 0.5:
            print(f"{name}: {done:,} / {total:,} bytes ({100 * done / total:.1f}%)", file=sys.stderr)
            last.update(time=now, name=name)

    result = fetch_data(args.data_dir, path, progress=progress, restart=args.restart)
    if result.get("status") not in {"verified", "completed", "complete"}:
        return result
    print("Preparing the complete sparse graph cache from the verified local data.", file=sys.stderr)
    result["sparse_cache"] = _entry("brain_cache", "prepare_cache")(args.data_dir, manifest_path=path, rebuild=False)
    if result["sparse_cache"].get("status") not in {"verified", "completed", "complete", "prepared"}:
        raise CommandError("full-graph cache preparation did not complete")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(args.data_dir, args.models_dir, args.manifest)
            _emit(result)
            return 0 if result["status"] == "ready" else 1
        if args.command == "data":
            if args.data_command == "fetch":
                result = _fetch(args)
            elif args.data_command == "prepare":
                result = _entry("brain_cache", "prepare_cache")(
                    args.data_dir, manifest_path=args.manifest, rebuild=args.rebuild)
            else:
                result = verify_data(args.data_dir, args.manifest)
        elif args.command == "app":
            _entry("service", "serve")(args.data_dir, args.models_dir, host=args.host, port=args.port,
                                       open_browser=args.open_browser)
            return 0
        elif args.command == "train":
            result = _entry("training", "train")(
                args.config, args.output_dir, architecture=args.architecture, seed=args.seed,
                resume_from=args.resume_from, stop_after_updates=args.stop_after_updates,
                cancel_file=args.cancel_file)
        elif args.command == "evaluate":
            result = _entry("evaluation", "evaluate")(
                args.config, args.data_dir, args.models_dir, args.output_dir, cancel_file=args.cancel_file)
        elif args.command == "replay":
            _entry("replay", "serve_replay")(args.recording, port=args.port, open_browser=args.open_browser)
            return 0
        else:
            result = _entry("validation", "validate")(args.data_dir, args.models_dir, args.output_dir)
        if not isinstance(result, dict):
            raise CommandError("operation returned no structured result")
        _emit(result)
        if result.get("status") in {"interrupted", "incomplete", "cancelled", "pending"}:
            return 3
        if result.get("status") in {"failed", "error", "not_ready", "blocked"}:
            return 1
        return 0 if result.get("status") in {"complete", "completed", "prepared", "passed", "verified", "ready"} else 1
    except KeyboardInterrupt:
        print("Cancelled; completed results and identified partial downloads are retained.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as error:
        # Ordinary validation errors are actionable. OSError/transport text can
        # contain local account paths; their class is sufficient at this boundary.
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(f"Error: {message}", file=sys.stderr)
        return 2
