"""Compare released policies under the original and compatible world readers."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from neuroterrarium.controllers import LearnedController
from neuroterrarium.data import sha256_file
from neuroterrarium import training, world


def function_text(source: str, name: str) -> str:
    definition = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "World")
    return ast.get_source_segment(source, next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == name))


def trajectory(policy, seed, module, episode_steps):
    names = ("World", "Body", "Food", "Obstacle", "Stimulus", "stream")
    previous = {name: getattr(training, name) for name in names}
    try:
        for name in names:
            setattr(training, name, getattr(module, name))
        env = training.TaskEnv(seed, episode_steps, "dev")
        controller = LearnedController(policy, seed, deterministic=True)
        observation = env.reset()
        records, states = [], hashlib.sha256()
        for step in range(episode_steps):
            # Checkpoint restoration is part of the comparison, including the
            # recurrent memory and the map/random state at the midpoint.
            if step == episode_steps // 2:
                env = training.TaskEnv.restore(json.loads(json.dumps(env.snapshot())))
                controller.restore(json.loads(json.dumps(controller.snapshot())))
            action = controller.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            body = env.world.bodies[0]
            records.append(np.r_[observation, action,
                                  [body.x, body.y, body.heading, body.speed, body.energy,
                                   body.food, body.collisions, reward],
                                  controller.hidden.numpy().ravel(),
                                  controller.last["raw_action"], controller.last["log_probability"]])
            states.update(json.dumps(env.world.snapshot(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            observation = next_observation
            if terminated or truncated:
                break
        values = np.asarray(records, dtype="<f8")
        return values, states.hexdigest(), info
    finally:
        for name, value in previous.items():
            setattr(training, name, value)


def verify(original_world: Path, models: Path, manifest: Path, output: Path):
    started = time.perf_counter()
    compatibility = json.loads(manifest.read_text())
    current_world = Path(world.__file__)
    if sha256_file(original_world) != compatibility["original_world_sha256"] or sha256_file(current_world) != compatibility["compatible_world_sha256"]:
        raise ValueError("world compatibility source hash mismatch")
    original_text, current_text = original_world.read_text(), current_world.read_text()
    for name in ("observe", "advance"):
        before, after = function_text(original_text, name), function_text(current_text, name)
        if before != after or hashlib.sha256(before.encode()).hexdigest() != compatibility["unchanged_functions_sha256"][name]:
            raise ValueError("a numerical world function changed")
    spec = importlib.util.spec_from_file_location("neuroterrarium.original_training_world_v1", original_world)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    model_set = json.loads((models / "manifest.json").read_text())
    if sha256_file(models / "manifest.json") != compatibility["model_set_manifest_sha256"]:
        raise ValueError("model set differs from the compatibility protocol")
    if len(model_set["models"]) != 9:
        raise ValueError("nine trained policies are required")
    torch.set_num_threads(1)
    rows, total = [], 0
    for entry in model_set["models"]:
        if Path(entry["checkpoint"]).name != entry["checkpoint"]:
            raise ValueError("invalid checkpoint name")
        checkpoint = models / entry["checkpoint"]
        if sha256_file(checkpoint / "policy.safetensors") != entry["policy_sha256"]:
            raise ValueError("policy hash mismatch")
        policy = training.load_policy(checkpoint)
        for seed in compatibility["development_seeds"]:
            before, before_state, before_info = trajectory(policy, seed, original, compatibility["episode_steps"])
            after, after_state, after_info = trajectory(policy, seed, world, compatibility["episode_steps"])
            exact = before.shape == after.shape and np.array_equal(before, after) and before_state == after_state and before_info == after_info
            if not exact:
                raise ValueError(f"world compatibility differs for {entry['id']} development seed {seed}")
            total += len(after)
            rows.append({"model": entry["id"], "development_seed": seed, "action_windows": len(after),
                         "shape": list(after.shape), "exact_equal": True,
                         "original_numeric_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
                         "compatible_numeric_sha256": hashlib.sha256(after.tobytes()).hexdigest(),
                         "original_world_tape_sha256": before_state, "compatible_world_tape_sha256": after_state,
                         "outcome": after_info})
    result = {"schema": "neuroterrarium.world-compatibility-results.v1", "status": "passed",
              "original_world_sha256": compatibility["original_world_sha256"],
              "compatible_world_sha256": compatibility["compatible_world_sha256"],
              "model_set_manifest_sha256": compatibility["model_set_manifest_sha256"],
              "verification_source_sha256": sha256_file(Path(__file__)),
              "models": len(model_set["models"]), "paired_development_episodes": len(rows),
              "action_windows_per_implementation": total, "wall_seconds": time.perf_counter() - started,
              "scope": "Exact observations, applied and raw actions, likelihoods, recurrent state, rewards, bodies, complete world-state tapes and episode outcomes; midpoint snapshot restoration included.",
              "episodes": rows}
    training.atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-world", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = verify(arguments.original_world, arguments.models, arguments.manifest, arguments.output)
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))
