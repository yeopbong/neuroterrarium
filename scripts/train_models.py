"""Train or resume all nine models under one frozen, sequential wall budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

from neuroterrarium.controllers import ARCHITECTURES
from neuroterrarium.data import sha256_file
from neuroterrarium.training import (
    _verified_checkpoint,
    atomic_json,
    behavior_source_digest,
    config_digest,
    evaluate_policy,
    export_model_set,
    load_policy,
    read_training_config,
    train,
)


def verified_run(directory: Path, configuration: dict, architecture: str, seed: int,
                 source: dict) -> tuple[dict | None, Path | None]:
    """Reuse only completed, matching records; otherwise return a checked resume."""
    checkpoint = None
    latest = directory / "latest.json"
    if latest.is_file():
        pointer = json.loads(latest.read_text())
        relative = Path(pointer["checkpoint"])
        if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "checkpoints":
            raise ValueError("invalid checkpoint pointer")
        checkpoint = directory / relative
        if sha256_file(checkpoint / "manifest.json") != pointer["manifest_sha256"]:
            raise ValueError("latest checkpoint manifest hash mismatch")
    elif (directory / "checkpoints" / "initial" / "manifest.json").is_file():
        checkpoint = directory / "checkpoints" / "initial"
    if checkpoint is None:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("existing run has no verified recovery checkpoint")
        return None, None
    _, manifest, state = _verified_checkpoint(checkpoint)
    if state["config_sha256"] != config_digest(configuration) or state["source_sha256"] != source:
        raise ValueError("checkpoint configuration or source mismatch")
    if (state["policy"]["architecture"], state["policy"]["seed"]) != (architecture, seed):
        raise ValueError("checkpoint model identity mismatch")
    run_path = directory / "run.json"
    if not run_path.is_file():
        return None, checkpoint
    run = json.loads(run_path.read_text())
    if run["source_sha256"] != source or run["config_sha256"] != config_digest(configuration):
        raise ValueError("run configuration or source mismatch")
    if run["status"] != "complete":
        return None, checkpoint
    if (run["architecture"], run["seed"]) != (architecture, seed):
        raise ValueError("run identity mismatch")
    if run["transitions"] != configuration["total_transitions"] or state["transitions"] != run["transitions"]:
        raise ValueError("completed run transition count mismatch")
    if state["training_status"] != "completed" or run["protocol_status"] != "frozen":
        raise ValueError("completed run status mismatch")
    if run["weights_sha256"] != manifest["files"]["policy.safetensors"]["sha256"]:
        raise ValueError("run policy hash mismatch")
    if run["checkpoint"] != str(checkpoint.relative_to(directory)):
        raise ValueError("run checkpoint pointer mismatch")
    return run, checkpoint


def train_models(configuration: Path, output: Path, models: Path, *, resume: bool = False,
                 cancel_file: Path | None = None) -> dict:
    config = read_training_config(configuration)
    if config["status"] != "frozen" or config["architectures"] != list(ARCHITECTURES):
        raise ValueError("a frozen three-mechanism protocol is required")
    seeds = config["seeds"]
    if len(seeds) != 3 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 3:
        raise ValueError("three independent integer seeds are required")
    wall_limit = config["suite_wall_max_seconds"]
    if type(wall_limit) not in (int, float) or not 0 < wall_limit <= 86400:
        raise ValueError("invalid suite wall budget")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("existing output requires explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    source, protocol = behavior_source_digest(), config_digest(config)
    budget_path = output / "suite-budget.json"
    previous_elapsed = 0.0
    if budget_path.is_file():
        budget = json.loads(budget_path.read_text())
        if budget["protocol_sha256"] != protocol or budget["source_sha256"] != source:
            raise ValueError("suite budget provenance mismatch")
        previous_elapsed = float(budget["elapsed_wall_seconds"]) + 1.0
    elif resume:
        # Earlier runs retain a per-model journal even if the process stopped
        # before creating a suite journal. Their consumed budget is preserved.
        for path in output.glob("*/budget.json"):
            budget = json.loads(path.read_text())
            if budget["config_sha256"] != protocol:
                raise ValueError("per-model budget provenance mismatch")
            previous_elapsed += float(budget["elapsed_wall_seconds"])
    if not 0 <= previous_elapsed < wall_limit:
        raise ValueError("suite wall budget exhausted")
    cancel = cancel_file or output / "suite.cancel"
    if cancel.exists():
        raise ValueError("cancel request remains present")
    stop = threading.Event()
    clock_errors: list[BaseException] = []
    started = time.perf_counter()

    def elapsed() -> float:
        return previous_elapsed + time.perf_counter() - started

    def journal() -> None:
        while True:
            try:
                atomic_json(budget_path, {"protocol_sha256": protocol, "source_sha256": source,
                                         "elapsed_wall_seconds": elapsed()})
                if elapsed() >= wall_limit:
                    cancel.write_text("Frozen suite wall budget exhausted.\n")
            except BaseException as error:
                clock_errors.append(error)
                cancel.touch()
                return
            if stop.wait(0.5):
                return

    def request_cancel(_number, _frame) -> None:
        cancel.write_text("Cancellation requested; saving at the next rollout boundary.\n")

    prior_signal = signal.signal(signal.SIGINT, request_cancel)
    watcher = threading.Thread(target=journal, daemon=True)
    watcher.start()
    records, checkpoints = [], []
    result = {"schema": "neuroterrarium.training-suite.v1", "protocol_sha256": protocol,
              "source_sha256": source, "status": "running", "models": records}
    try:
        for architecture in config["architectures"]:
            for seed in seeds:
                if clock_errors:
                    raise RuntimeError("suite budget journal failed") from clock_errors[0]
                if behavior_source_digest() != source:
                    raise ValueError("behavior source changed during training")
                if cancel.exists() or elapsed() >= wall_limit:
                    result.update(status="interrupted", suite_elapsed_seconds=elapsed(),
                                  interruption_reason="suite_cancel_or_wall_budget")
                    atomic_json(output / "index.json", result)
                    return result
                name = f"{architecture}-{seed}"
                directory = output / name
                run, checkpoint = verified_run(directory, config, architecture, seed, source)
                reused = run is not None
                if run is None:
                    if checkpoint is not None and not (directory / "dev-before.json").is_file():
                        # A process may have stopped after the immutable initial
                        # checkpoint but before its deterministic dev baseline.
                        initial = directory / "checkpoints" / "initial"
                        _, _, initial_state = _verified_checkpoint(initial)
                        if (initial_state["transitions"] != 0 or initial_state["source_sha256"] != source or
                            initial_state["config_sha256"] != protocol or
                            (initial_state["policy"]["architecture"], initial_state["policy"]["seed"]) != (architecture, seed)):
                            raise ValueError("initial baseline checkpoint mismatch")
                        baseline = evaluate_policy(load_policy(initial), config["dev_seeds"],
                                                   episode_steps=config["dev_episode_steps"])
                        atomic_json(directory / "dev-before.json", baseline)
                    print(json.dumps({"event": "training", "model": name, "resume": checkpoint is not None}), flush=True)
                    run = train(config, directory, architecture=architecture, seed=seed,
                                resume_from=checkpoint, cancel_file=cancel)
                    checkpoint = directory / run["checkpoint"]
                record = {"model": name, **{key: run[key] for key in (
                    "architecture", "seed", "status", "transitions", "transitions_per_second",
                    "weights_sha256", "initial_weights_sha256", "parameters_changed",
                    "dev_before", "dev_after", "interruption_reason", "peak_resources")},
                          "wall_seconds": run["cumulative_wall_seconds"],
                          "checkpoint": str(checkpoint.relative_to(output)), "reused_verified_run": reused}
                records.append(record)
                checkpoints.append(checkpoint)
                result.update(suite_elapsed_seconds=elapsed(), status="running" if run["status"] == "complete" else "interrupted")
                atomic_json(output / "index.json", result)
                print(json.dumps({"event": "verified" if reused else "finished", "model": name,
                                  "status": run["status"], "transitions": run["transitions"]}), flush=True)
                if run["status"] != "complete":
                    return result
        if clock_errors:
            raise RuntimeError("suite budget journal failed") from clock_errors[0]
        if models.exists():
            # A prior immutable export can be retained only when all identities,
            # manifests and bytes still match the verified training outputs.
            model_set = json.loads((models / "manifest.json").read_text())
            if (model_set.get("schema") != "neuroterrarium.model-set.v1" or model_set.get("status") != "completed" or
                model_set["protocol_sha256"] != protocol or model_set["source_sha256"] != source or len(model_set["models"]) != 9 or
                model_set["transitions_per_model"] != config["total_transitions"]):
                raise ValueError("existing model set provenance mismatch")
            expected = {f"{architecture}-{seed}" for architecture in config["architectures"] for seed in seeds}
            if {entry["id"] for entry in model_set["models"]} != expected:
                raise ValueError("existing model set identities mismatch")
            by_identity = {record["model"]: path for record, path in zip(records, checkpoints, strict=True)}
            for entry in model_set["models"]:
                if entry["checkpoint"] != entry["id"]:
                    raise ValueError("invalid existing model directory")
                path = models / entry["id"]
                _, artifact_manifest, metadata = _verified_checkpoint(path)
                if (entry["policy_sha256"] != artifact_manifest["files"]["policy.safetensors"]["sha256"] or
                    entry["training_status"] != "completed" or entry["transitions"] != config["total_transitions"] or
                    entry["architecture"] != metadata["policy"]["architecture"] or entry["seed"] != metadata["policy"]["seed"]):
                    raise ValueError("existing exported identity mismatch")
                if (sha256_file(path / "manifest.json") != entry["checkpoint_manifest_sha256"] or
                    sha256_file(path / "manifest.json") != sha256_file(by_identity[entry["id"]] / "manifest.json")):
                    raise ValueError("existing exported checkpoint mismatch")
        else:
            model_set = export_model_set(checkpoints, models)
        result.update(status="complete", exit_code=0, suite_elapsed_seconds=elapsed(),
                      model_set_manifest=model_set)
        atomic_json(output / "index.json", result)
        return result
    except Exception as error:
        result.update(status="failed", suite_elapsed_seconds=elapsed(), error_type=type(error).__name__)
        atomic_json(output / "index.json", result)
        raise
    finally:
        stop.set()
        watcher.join(timeout=2)
        signal.signal(signal.SIGINT, prior_signal)
        atomic_json(budget_path, {"protocol_sha256": protocol, "source_sha256": source,
                                 "elapsed_wall_seconds": elapsed()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cancel-file", type=Path)
    arguments = parser.parse_args()
    report = train_models(arguments.config, arguments.output, arguments.models,
                          resume=arguments.resume, cancel_file=arguments.cancel_file)
    print(json.dumps({key: value for key, value in report.items() if key not in {"models", "model_set_manifest"}}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 2)
