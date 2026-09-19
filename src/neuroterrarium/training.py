"""Resource-bounded PPO with sequence replay and non-executable checkpoints.

An environment transition is one 20 ms action window in one World. Gaussian
latent samples retain their own likelihood, even when the common action
pipeline transforms them or applies the explicitly labeled hybrid reflex.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import psutil
import torch
from safetensors.torch import load_file, save_file

from .controllers import ARCHITECTURES, LearnedController, Policy, named_seed, policy_metadata
from .data import sha256_file
from .world import Body, Food, Obstacle, Stimulus, World, stream

CHECKPOINT_SCHEMA = "neuroterrarium.ppo-checkpoint.v1"
DEFAULT_RESOURCE_LIMITS = {
    "minimum_disk_free_bytes": 10 * 1024**3,
    "minimum_available_memory_bytes": 3 * 1024**3,
    "maximum_process_rss_bytes": 1024**3,
    "maximum_wall_seconds": 900,
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def behavior_source_digest() -> dict:
    root = Path(__file__).parent
    return {name: sha256_file(root / name) for name in
            ("training.py", "controllers.py", "world.py")}


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_training_config(config: dict | Path | str) -> dict:
    if not isinstance(config, dict):
        path = Path(config)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("training configuration too large")
        config = json.loads(path.read_text())
    config = copy.deepcopy(config)
    if config.get("schema") != "neuroterrarium.ppo.v1":
        raise ValueError("training configuration schema mismatch")
    if config.get("status") not in ("smoke", "frozen"):
        raise ValueError("training budget must be smoke or frozen before execution")
    bounds = {"num_envs": (1, 4), "torch_threads": (1, 2), "hidden_size": (8, 256),
              "rollout_steps": (1, 1024), "sequence_length": (1, 128),
              "minibatch_sequences": (1, 128), "epochs": (1, 16),
              "episode_steps": (1, 10000), "total_transitions": (1, 100000000),
              "checkpoint_updates": (1, 10000)}
    for key, (low, high) in bounds.items():
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError(f"invalid {key}")
    if config["total_transitions"] % config["num_envs"]:
        raise ValueError("total transitions must be divisible by environment count")
    for key in ("learning_rate", "gamma", "gae_lambda", "clip", "value_coefficient", "entropy_coefficient", "max_grad_norm"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {key}")
    if not 0 < config["gamma"] <= 1 or not 0 < config["gae_lambda"] <= 1:
        raise ValueError("invalid return discount")
    limits = config.setdefault("resource_limits", dict(DEFAULT_RESOURCE_LIMITS))
    if not isinstance(limits, dict) or set(limits) != set(DEFAULT_RESOURCE_LIMITS):
        raise ValueError("resource limit schema mismatch")
    for key, value in limits.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid resource limit")
        if key.startswith("maximum") and value == 0:
            raise ValueError("maximum resource limit must be positive")
    return config


def measure_resources(output: Path) -> dict:
    """Read actual host headroom; measurement failures propagate explicitly."""
    return {"disk_free_bytes": shutil.disk_usage(output).free,
            "available_memory_bytes": psutil.virtual_memory().available,
            "process_rss_bytes": psutil.Process().memory_info().rss}


def resource_violations(config: dict, measured: dict, elapsed: float) -> list[str]:
    limits = config["resource_limits"]
    problems = []
    if measured["disk_free_bytes"] < limits["minimum_disk_free_bytes"]:
        problems.append("disk_reserve")
    if measured["available_memory_bytes"] < limits["minimum_available_memory_bytes"]:
        problems.append("available_memory_reserve")
    if measured["process_rss_bytes"] > limits["maximum_process_rss_bytes"]:
        problems.append("process_rss_limit")
    if elapsed >= limits["maximum_wall_seconds"]:
        problems.append("wall_time_budget")
    return problems


class TaskEnv:
    """One body, the authoritative World, and a private map/reset stream.

    Reward may read simulator truth. Only ``World.observe()[0]`` is sent to
    the policy. Food relocation is selected at reset and occurs on a fixed
    schedule independent of subsequent controller actions.
    """
    def __init__(self, seed: int, episode_steps: int = 250, split: str = "train"):
        if split not in ("train", "dev", "test", "stress"):
            raise ValueError("unknown scene partition")
        self.seed, self.episode_steps, self.split = seed, episode_steps, split
        self.random = stream(seed, f"maps/{split}")
        self.episodes = 0
        self.world: World | None = None

    def reset(self) -> np.ndarray:
        rng = self.random
        self.kind = str(rng.choice(["forage", "obstacles", "threat", "relocation"]))
        world_seed = named_seed(self.seed, f"{self.split}/episode/{self.episodes}") % (2**53)
        self.world = World(world_seed, 1)
        x, y = float(rng.uniform(32, 48)), float(rng.uniform(20, 36))
        heading = float(rng.uniform(-math.pi, math.pi))
        self.world.bodies = [Body(x, y, heading, energy=float(rng.uniform(0.55, 0.95)))]
        angle = heading + float(rng.uniform(-1.3, 1.3))
        distance = float(rng.uniform(7, 11) if self.kind == "obstacles" else rng.uniform(0.5, 7))
        self.world.foods = [Food(x + distance*math.cos(angle), y + distance*math.sin(angle), 0.7)]
        second_angle = float(rng.uniform(-math.pi, math.pi))
        self.world.foods.append(Food(x+12*math.cos(second_angle), y+12*math.sin(second_angle), 0.7))
        self.world.obstacles = ([] if self.kind != "obstacles" else
                                [Obstacle(x + 0.5*distance*math.cos(angle), y + 0.5*distance*math.sin(angle), 1.2)])
        self.world.stimuli = []
        if self.kind == "threat":
            theta = float(rng.uniform(-math.pi, math.pi))
            direction = np.array([math.cos(theta), math.sin(theta)])
            speed = float(rng.uniform(2, 6))
            physical = bool(rng.integers(0, 2))
            kind = int(rng.integers(0, 4))
            velocity = (-direction*speed if kind == 0 else
                        np.array([-direction[1], direction[0]])*speed if kind == 1 else np.zeros(2))
            growth = 1.0 if kind == 2 else -0.3 if kind == 3 else 0.0
            self.world.stimuli = [Stimulus(x+10*direction[0], y+10*direction[1],
                                           1.2, float(velocity[0]), float(velocity[1]), growth, physical)]
        self.scheduled_food = [float(rng.uniform(15, 65)), float(rng.uniform(12, 44))]
        self.elapsed = 0
        self.episode_return = 0.0
        self.episodes += 1
        return self.world.observe()[0].astype(np.float32)

    def step(self, action: np.ndarray):
        body = self.world.bodies[0]
        previous_food, previous_energy, previous_collisions = body.food, body.energy, body.collisions
        self.world.advance(np.asarray(action)[None])
        self.elapsed += 1
        if self.kind == "relocation" and self.elapsed == self.episode_steps // 2:
            self.world.foods[0].x, self.world.foods[0].y = self.scheduled_food
        gained, energy_delta = body.food-previous_food, body.energy-previous_energy
        collision_delta = body.collisions-previous_collisions
        applied = np.asarray(body.action)
        reward = 100*gained + energy_delta - 0.02*collision_delta - 0.001*(applied[0]+3*applied[2])
        self.episode_return += reward
        terminated = bool(body.energy <= 0.05)
        truncated = self.elapsed >= self.episode_steps and not terminated
        observation = self.world.observe()[0].astype(np.float32)
        info = {"food": float(body.food), "energy": float(body.energy), "collisions": int(body.collisions),
                "episode_return": float(self.episode_return), "episode_steps": self.elapsed,
                "scene": self.kind, "terminated": terminated, "truncated": truncated}
        return observation, float(reward), terminated, truncated, info

    def snapshot(self) -> dict:
        return {"seed": self.seed, "episode_steps": self.episode_steps, "split": self.split,
                "random": copy.deepcopy(self.random.bit_generator.state), "episodes": self.episodes,
                "world": self.world.snapshot(), "kind": self.kind,
                "scheduled_food": self.scheduled_food, "elapsed": self.elapsed,
                "episode_return": self.episode_return}

    @classmethod
    def restore(cls, state: dict):
        env = cls(state["seed"], state["episode_steps"], state["split"])
        env.world = World.restore(state["world"])
        env.random.bit_generator.state = copy.deepcopy(state["random"])
        for key in ("episodes", "kind", "scheduled_food", "elapsed", "episode_return"):
            setattr(env, key, copy.deepcopy(state[key]))
        return env


def compute_gae(rewards, values, next_values, terminated, truncated, gamma: float, gae_lambda: float):
    """Bootstrap time limits, but never carry advantage across episode resets."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma*next_values[step]*(~terminated[step]) - values[step]
        running = delta + gamma*gae_lambda*(~(terminated[step] | truncated[step]))*running
        advantages[step] = running
    return advantages, advantages+values


def evaluate_policy(policy: Policy, seeds: list[int], *, episode_steps: int = 250,
                    split: str = "dev", reflex_enabled: bool = True) -> dict:
    rows = []
    previous_mode = policy.training
    for seed in seeds:
        env = TaskEnv(seed, episode_steps, split)
        controller = LearnedController(policy, seed, deterministic=True)
        controller.reflex_enabled = reflex_enabled
        observation = env.reset()
        corrections = 0
        for _ in range(episode_steps):
            action = controller.act(observation)
            corrections += int(controller.last["reflex_triggered"])
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        rows.append({"seed": seed, **info, "reflex_interventions": corrections,
                     "status": "terminated" if terminated else "time_limit"})
    policy.train(previous_mode)
    return {"split": split, "deterministic": True, "episodes": rows,
            "mean_return": float(np.mean([row["episode_return"] for row in rows])),
            "mean_food": float(np.mean([row["food"] for row in rows])),
            "mean_collisions": float(np.mean([row["collisions"] for row in rows]))}


def _optimizer_pack(optimizer: torch.optim.Optimizer):
    state = optimizer.state_dict()
    tensors, scalar_state = {}, {}
    for parameter, fields in state["state"].items():
        scalar_state[str(parameter)] = {}
        for key, value in fields.items():
            if isinstance(value, torch.Tensor):
                name = f"optimizer/{parameter}/{key}"
                tensors[name] = value.detach().cpu().contiguous()
                scalar_state[str(parameter)][key] = {"tensor": name}
            else:
                scalar_state[str(parameter)][key] = {"scalar": value}
    return tensors, {"state": scalar_state, "param_groups": state["param_groups"]}


def _optimizer_restore(optimizer, metadata, tensors):
    states = {}
    for parameter, fields in metadata["state"].items():
        states[int(parameter)] = {key: tensors[value["tensor"]] if "tensor" in value else value["scalar"]
                                  for key, value in fields.items()}
    optimizer.load_state_dict({"state": states, "param_groups": metadata["param_groups"]})


def _verified_checkpoint(path: Path | str):
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 1024*1024:
        raise ValueError("checkpoint manifest missing or too large")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    required = {"policy.safetensors", "optimizer.safetensors", "runtime.safetensors", "state.json"}
    if set(manifest["files"]) != required:
        raise ValueError("checkpoint file list mismatch")
    for name, spec in manifest["files"].items():
        file = path/name
        if not file.is_file() or file.stat().st_size > 128*1024*1024 or file.stat().st_size != spec["bytes"] or sha256_file(file) != spec["sha256"]:
            raise ValueError(f"checkpoint integrity failure: {name}")
    metadata = json.loads((path/"state.json").read_text())
    return path, manifest, metadata


def load_policy(path: Path | str) -> Policy:
    path, _, metadata = _verified_checkpoint(path)
    specification = metadata["policy"]
    policy = Policy(specification["architecture"], specification["seed"], specification["hidden_size"])
    policy.load_state_dict(load_file(str(path/"policy.safetensors")), strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
        raise ValueError("nonfinite policy checkpoint")
    return policy.eval()


class PPOTrainer:
    def __init__(self, config: dict | str | Path, architecture: str, seed: int):
        self.config = read_training_config(config)
        torch.set_num_threads(self.config["torch_threads"])
        self.policy = Policy(architecture, seed, self.config["hidden_size"])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config["learning_rate"], eps=1e-5)
        self.generator = torch.Generator().manual_seed(named_seed(seed, f"sample/{architecture}"))
        self.permutations = stream(seed, f"optimizer/{architecture}")
        self.envs = [TaskEnv(named_seed(seed, f"environment/{architecture}/{index}"), self.config["episode_steps"])
                     for index in range(self.config["num_envs"])]
        self.observations = torch.from_numpy(np.stack([env.reset() for env in self.envs]))
        self.hidden = self.policy.initial_state(len(self.envs))
        self.starts = torch.ones(len(self.envs), dtype=torch.bool)
        self.transitions = self.updates = 0
        self.elapsed_wall_seconds = 0.0
        self.episode_history = []
        self.last_rollout: dict | None = None
        self.source = behavior_source_digest()

    def collect(self, steps: int) -> dict:
        fields = {name: [] for name in ("observations", "hidden", "starts", "raw_action", "log_probability", "applied_action",
                                        "reflex_triggered", "correction_l1", "value", "reward", "next_value", "terminated", "truncated")}
        for _ in range(steps):
            fields["observations"].append(self.observations.clone())
            fields["hidden"].append(self.hidden.clone())
            fields["starts"].append(self.starts.clone())
            with torch.no_grad():
                result = self.policy.sample(self.observations, self.hidden, self.starts, self.generator)
            final_observations, rewards, terminations, truncations = [], [], [], []
            for index, env in enumerate(self.envs):
                obs, reward, terminated, truncated, info = env.step(result["applied_action"][index].numpy())
                final_observations.append(obs)
                rewards.append(reward)
                terminations.append(terminated)
                truncations.append(truncated)
                if terminated or truncated:
                    self.episode_history.append({"transition": self.transitions+index+1, **info})
            with torch.no_grad():
                _, next_values, _ = self.policy(torch.from_numpy(np.stack(final_observations)), result["hidden"],
                                                torch.zeros(len(self.envs), dtype=torch.bool))
            for name in ("raw_action", "log_probability", "applied_action", "reflex_triggered", "correction_l1", "value"):
                fields[name].append(result[name].clone())
            fields["reward"].append(torch.tensor(rewards, dtype=torch.float32))
            fields["next_value"].append(next_values)
            fields["terminated"].append(torch.tensor(terminations))
            fields["truncated"].append(torch.tensor(truncations))
            self.starts = fields["terminated"][-1] | fields["truncated"][-1]
            for index, done in enumerate(self.starts):
                if done:
                    final_observations[index] = self.envs[index].reset()
            self.observations = torch.from_numpy(np.stack(final_observations))
            self.hidden = result["hidden"].clone()
            self.transitions += len(self.envs)
        rollout = {name: torch.stack(values) for name, values in fields.items()}
        rollout["advantage"], rollout["return"] = compute_gae(
            rollout["reward"], rollout["value"], rollout["next_value"], rollout["terminated"], rollout["truncated"],
            self.config["gamma"], self.config["gae_lambda"])
        self.last_rollout = rollout
        return rollout

    def update(self, rollout: dict) -> dict:
        config = self.config
        advantages = rollout["advantage"]
        normalized = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        length, environments = advantages.shape
        sequences = [(start, min(start+config["sequence_length"], length), env)
                     for env in range(environments) for start in range(0, length, config["sequence_length"])]
        diagnostics = []
        for _ in range(config["epochs"]):
            ordering = self.permutations.permutation(len(sequences))
            for offset in range(0, len(ordering), config["minibatch_sequences"]):
                chosen = [sequences[index] for index in ordering[offset:offset+config["minibatch_sequences"]]]
                self.optimizer.zero_grad(set_to_none=True)
                width = max(end-start for start, end, _ in chosen)
                batch = len(chosen)
                tensors = {}
                for key in ("observations", "raw_action", "starts", "log_probability", "return"):
                    sample = rollout[key]
                    tensors[key] = torch.zeros((width, batch, *sample.shape[2:]), dtype=sample.dtype)
                tensors["starts"].fill_(True)
                mask = torch.zeros(width, batch, dtype=torch.bool)
                advantage = torch.zeros(width, batch)
                hidden = torch.stack([rollout["hidden"][start, env] for start, _, env in chosen]).detach()
                for index, (start, end, env) in enumerate(chosen):
                    count = end-start
                    mask[:count, index] = True
                    advantage[:count, index] = normalized[start:end, env]
                    for key in tensors:
                        tensors[key][:count, index] = rollout[key][start:end, env]
                if self.policy.recurrent:
                    log_probabilities, values, entropy_values = [], [], []
                    for step in range(width):
                        distribution, value, hidden = self.policy(tensors["observations"][step], hidden,
                                                                  tensors["starts"][step])
                        log_probabilities.append(distribution.log_prob(tensors["raw_action"][step]).sum(-1))
                        values.append(value)
                        entropy_values.append(distribution.entropy().sum(-1))
                    log_probability = torch.stack(log_probabilities)
                    value = torch.stack(values)
                    entropy_samples = torch.stack(entropy_values)
                else:
                    distribution, value, _ = self.policy(tensors["observations"].flatten(0, 1),
                                                          self.policy.initial_state(width*batch))
                    log_probability = distribution.log_prob(tensors["raw_action"].flatten(0, 1)).sum(-1).reshape(width, batch)
                    value = value.reshape(width, batch)
                    entropy_samples = distribution.entropy().sum(-1).reshape(width, batch)
                difference = (log_probability-tensors["log_probability"])[mask]
                ratio = difference.exp()
                actor_loss = -torch.minimum(ratio*advantage[mask], ratio.clamp(1-config["clip"], 1+config["clip"])*advantage[mask]).mean()
                value_loss = (value[mask]-tensors["return"][mask]).square().mean()
                entropy = entropy_samples[mask].mean()
                kl = ((ratio-1)-difference).detach().mean()
                loss = actor_loss+config["value_coefficient"]*value_loss-config["entropy_coefficient"]*entropy
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite PPO objective")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
                self.optimizer.step()
                diagnostics.append([float(actor_loss.detach()), float(value_loss.detach()), float(entropy.detach()),
                                    float(kl), float(gradient_norm)])
        self.updates += 1
        result = dict(zip(("policy_loss", "value_loss", "latent_entropy", "approx_kl", "gradient_norm"),
                          np.mean(diagnostics, axis=0).tolist(), strict=True))
        result.update(transitions=self.transitions, updates=self.updates,
                      reward_mean=float(rollout["reward"].mean()),
                      reflex_interventions=int(rollout["reflex_triggered"].sum()),
                      correction_l1_mean=float(rollout["correction_l1"].mean()))
        return result

    def save_checkpoint(self, directory: Path | str, *, allow_identical_existing: bool = False) -> Path:
        directory = Path(directory)
        if directory.exists() and not allow_identical_existing:
            raise FileExistsError("checkpoint directories are immutable")
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = directory.with_name(directory.name+".tmp-"+uuid.uuid4().hex)
        temporary.mkdir()
        try:
            save_file({name: tensor.detach().cpu().contiguous() for name, tensor in self.policy.state_dict().items()}, str(temporary/"policy.safetensors"))
            optimizer_tensors, optimizer_metadata = _optimizer_pack(self.optimizer)
            save_file(optimizer_tensors, str(temporary/"optimizer.safetensors"))
            save_file({"hidden": self.hidden, "observations": self.observations, "starts": self.starts,
                       "policy_random": self.generator.get_state()}, str(temporary/"runtime.safetensors"))
            metadata = {"schema": CHECKPOINT_SCHEMA, "policy": policy_metadata(self.policy), "config": self.config,
                        "config_sha256": config_digest(self.config), "source_sha256": self.source,
                        "transitions": self.transitions, "updates": self.updates,
                        "elapsed_wall_seconds": self.elapsed_wall_seconds,
                        "training_status": "completed" if self.transitions == self.config["total_transitions"] else "incomplete",
                        "optimizer": optimizer_metadata,
                        "permutations": copy.deepcopy(self.permutations.bit_generator.state),
                        "environments": [env.snapshot() for env in self.envs],
                        "recent_episodes": self.episode_history[-100:],
                        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}}
            atomic_json(temporary/"state.json", metadata)
            manifest = {"schema": CHECKPOINT_SCHEMA, "files": {
                name: {"sha256": sha256_file(temporary/name), "bytes": (temporary/name).stat().st_size}
                for name in ("policy.safetensors", "optimizer.safetensors", "runtime.safetensors", "state.json")}}
            atomic_json(temporary/"manifest.json", manifest)
            if directory.exists():
                _, existing, existing_metadata = _verified_checkpoint(directory)
                old_state, new_state = dict(existing_metadata), dict(metadata)
                old_state.pop("elapsed_wall_seconds", None)
                new_state.pop("elapsed_wall_seconds", None)
                same_tensors = all(existing["files"][key] == manifest["files"][key]
                                   for key in ("policy.safetensors", "optimizer.safetensors", "runtime.safetensors"))
                if not same_tensors or old_state != new_state:
                    raise ValueError("existing checkpoint differs from resumed computation")
                shutil.rmtree(temporary)
                return directory
            os.replace(temporary, directory)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return directory

    @classmethod
    def resume(cls, directory: Path | str, config: dict | str | Path | None = None):
        directory, _, metadata = _verified_checkpoint(directory)
        expected_config = read_training_config(config if config is not None else metadata["config"])
        if config_digest(expected_config) != metadata["config_sha256"] or behavior_source_digest() != metadata["source_sha256"]:
            raise ValueError("checkpoint training configuration or source mismatch")
        policy_spec = metadata["policy"]
        trainer = cls(expected_config, policy_spec["architecture"], policy_spec["seed"])
        trainer.policy.load_state_dict(load_file(str(directory/"policy.safetensors")), strict=True)
        _optimizer_restore(trainer.optimizer, metadata["optimizer"], load_file(str(directory/"optimizer.safetensors")))
        runtime = load_file(str(directory/"runtime.safetensors"))
        if runtime["hidden"].shape != trainer.hidden.shape or runtime["observations"].shape != trainer.observations.shape or runtime["starts"].shape != trainer.starts.shape:
            raise ValueError("checkpoint runtime shape mismatch")
        trainer.hidden, trainer.observations, trainer.starts = runtime["hidden"], runtime["observations"], runtime["starts"]
        trainer.generator.set_state(runtime["policy_random"])
        trainer.permutations.bit_generator.state = metadata["permutations"]
        trainer.envs = [TaskEnv.restore(state) for state in metadata["environments"]]
        trainer.transitions, trainer.updates = metadata["transitions"], metadata["updates"]
        trainer.elapsed_wall_seconds = float(metadata.get("elapsed_wall_seconds", 0))
        trainer.episode_history = metadata["recent_episodes"]
        return trainer


def train(config: dict | Path | str, output_dir: Path | str, *, architecture: str = "feedforward",
          seed: int = 101, resume_from: Path | str | None = None, stop_after_updates: int | None = None,
          cancel_file: Path | str | None = None) -> dict:
    """Train one independent seed; checkpoints and action audit shards persist.

    Cancellation is checked between bounded rollouts. ``stop_after_updates``
    provides the same safe boundary for controlled resumability checks.
    A draft full configuration cannot start until its budget is frozen.
    """
    config = read_training_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trainer = (PPOTrainer.resume(resume_from, config) if resume_from else PPOTrainer(config, architecture, seed))
    if trainer.policy.architecture != architecture or trainer.policy.seed != seed:
        raise ValueError("resume model identity mismatch")
    elapsed_before = trainer.elapsed_wall_seconds
    budget_path = output/"budget.json"
    if resume_from is not None and budget_path.exists():
        budget = json.loads(budget_path.read_text())
        if budget["config_sha256"] != config_digest(config):
            raise ValueError("resume resource journal protocol mismatch")
        elapsed_before = max(elapsed_before, float(budget["elapsed_wall_seconds"]))
    start_transitions = trainer.transitions
    atomic_json(output/"config.json", config)
    if resume_from is None:
        initial = trainer.save_checkpoint(output/"checkpoints"/"initial")
        dev_before = evaluate_policy(trainer.policy, config["dev_seeds"], episode_steps=config["dev_episode_steps"])
        atomic_json(output/"dev-before.json", dev_before)
        initial_hash = sha256_file(initial/"policy.safetensors")
    else:
        dev_before = json.loads((output/"dev-before.json").read_text())
        initial_hash = sha256_file(output/"checkpoints"/"initial"/"policy.safetensors")
    action_dir = output/"actions"
    action_dir.mkdir(exist_ok=True)
    history_path = output/"training.jsonl"
    checkpoint = Path(resume_from) if resume_from else initial
    interruption_reason = None
    peak_resources = {"process_rss_bytes": 0}
    with history_path.open("a", encoding="utf-8", buffering=1) as history:
        while trainer.transitions < config["total_transitions"]:
            if cancel_file is not None and Path(cancel_file).exists():
                interruption_reason = "cancel_file"
                break
            resources = measure_resources(output)
            peak_resources["process_rss_bytes"] = max(peak_resources["process_rss_bytes"], resources["process_rss_bytes"])
            violations = resource_violations(config, resources, elapsed_before+time.perf_counter()-started)
            if violations:
                interruption_reason = "resource:"+",".join(violations)
                history.write(json.dumps({"status": "interrupted", "reason": interruption_reason,
                                           "transitions": trainer.transitions, "resources": resources})+"\n")
                break
            steps = min(config["rollout_steps"], (config["total_transitions"]-trainer.transitions)//config["num_envs"])
            rollout = trainer.collect(steps)
            metrics = trainer.update(rollout)
            trainer.elapsed_wall_seconds = elapsed_before+time.perf_counter()-started
            atomic_json(budget_path, {"config_sha256": config_digest(config),
                                      "elapsed_wall_seconds": trainer.elapsed_wall_seconds})
            audit = {key: rollout[key].contiguous() for key in
                     ("raw_action", "log_probability", "applied_action", "reflex_triggered", "correction_l1")}
            audit_path = action_dir/f"update-{trainer.updates:06d}.safetensors"
            audit_temporary = audit_path.with_suffix(".safetensors.tmp-"+uuid.uuid4().hex)
            save_file(audit, str(audit_temporary))
            replayed = audit_path.exists()
            if replayed:
                if resume_from is None or sha256_file(audit_temporary) != sha256_file(audit_path):
                    audit_temporary.unlink()
                    raise ValueError("existing action audit differs from resumed computation")
                audit_temporary.unlink()
            else:
                os.replace(audit_temporary, audit_path)
            metrics["elapsed_wall_seconds"] = time.perf_counter()-started
            metrics["resume_recomputed_existing_audit"] = replayed
            history.write(json.dumps(metrics, allow_nan=False)+"\n")
            should_stop = stop_after_updates is not None and trainer.updates >= stop_after_updates
            if trainer.updates % config["checkpoint_updates"] == 0 or should_stop or trainer.transitions == config["total_transitions"]:
                checkpoint = trainer.save_checkpoint(output/"checkpoints"/f"transitions-{trainer.transitions:010d}",
                                                       allow_identical_existing=resume_from is not None)
                atomic_json(output/"latest.json", {"checkpoint": str(checkpoint.relative_to(output)),
                                                    "manifest_sha256": sha256_file(checkpoint/"manifest.json")})
            if should_stop:
                interruption_reason = "update_boundary_stop"
                break
    elapsed = time.perf_counter()-started
    trainer.elapsed_wall_seconds = elapsed_before+elapsed
    if checkpoint.name != f"transitions-{trainer.transitions:010d}" and trainer.transitions > start_transitions:
        checkpoint = trainer.save_checkpoint(output/"checkpoints"/f"transitions-{trainer.transitions:010d}",
                                               allow_identical_existing=resume_from is not None)
    atomic_json(output/"latest.json", {"checkpoint": str(checkpoint.relative_to(output)),
                                        "manifest_sha256": sha256_file(checkpoint/"manifest.json")})
    if interruption_reason is not None and (interruption_reason.startswith("resource:") or interruption_reason == "cancel_file"):
        dev_after = {"status": "not_run", "split": "dev", "reason": interruption_reason}
    else:
        dev_after = evaluate_policy(trainer.policy, config["dev_seeds"], episode_steps=config["dev_episode_steps"])
    atomic_json(output/"dev-after.json", dev_after)
    weights_hash = sha256_file(checkpoint/"policy.safetensors")
    report = {"schema": "neuroterrarium.training-run.v1", "status": "complete" if trainer.transitions == config["total_transitions"] else "interrupted",
              "protocol_status": config["status"], "architecture": architecture, "seed": seed,
              "transitions": trainer.transitions, "vector_steps": trainer.transitions//config["num_envs"],
              "updates": trainer.updates, "wall_seconds": elapsed,
              "cumulative_wall_seconds": elapsed_before+time.perf_counter()-started,
              "interruption_reason": interruption_reason, "peak_resources": peak_resources,
              "transitions_per_second": (trainer.transitions-start_transitions)/max(elapsed, 1e-9),
              "initial_weights_sha256": initial_hash, "weights_sha256": weights_hash,
              "parameters_changed": initial_hash != weights_hash, "checkpoint": str(checkpoint.relative_to(output)),
              "dev_before": dev_before, "dev_after": dev_after, "source_sha256": trainer.source,
              "config_sha256": config_digest(config)}
    atomic_json(output/"run.json", report)
    atomic_json(budget_path, {"config_sha256": config_digest(config),
                              "elapsed_wall_seconds": report["cumulative_wall_seconds"]})
    return report


def export_model_set(checkpoint_dirs: list[Path | str], output_dir: Path | str) -> dict:
    """Export nine completed independent models; smoke checkpoints are refused."""
    if len(checkpoint_dirs) != 9:
        raise ValueError("a model set requires nine checkpoints")
    verified = [_verified_checkpoint(path) for path in checkpoint_dirs]
    first = verified[0][2]
    config = first["config"]
    expected = {(architecture, seed) for architecture in ARCHITECTURES for seed in config["seeds"]}
    if len(expected) != 9 or config["status"] != "frozen":
        raise ValueError("model set requires a frozen three-seed training protocol")
    identities, hashes, models = set(), set(), []
    for path, manifest, metadata in verified:
        identity = (metadata["policy"]["architecture"], metadata["policy"]["seed"])
        weight_hash = manifest["files"]["policy.safetensors"]["sha256"]
        if identity in identities or weight_hash in hashes:
            raise ValueError("duplicate model identity or weights")
        if metadata.get("training_status") != "completed" or metadata["transitions"] != config["total_transitions"]:
            raise ValueError("incomplete model training")
        if metadata["config_sha256"] != first["config_sha256"] or metadata["source_sha256"] != first["source_sha256"]:
            raise ValueError("model training protocol or source mismatch")
        load_policy(path)
        identities.add(identity)
        hashes.add(weight_hash)
        model_id = f"{identity[0]}-{identity[1]}"
        models.append({"id": model_id, "checkpoint": model_id, "architecture": identity[0],
                       "seed": identity[1], "policy_sha256": weight_hash,
                       "checkpoint_manifest_sha256": sha256_file(path/"manifest.json"),
                       "transitions": metadata["transitions"], "training_status": "completed"})
    if identities != expected:
        raise ValueError("model identities do not match the three mechanisms and seeds")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("model sets are immutable; choose a new destination")
    temporary = output.with_name(output.name+".tmp-"+uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    try:
        for (path, _, _), model in zip(verified, models, strict=True):
            shutil.copytree(path, temporary/model["checkpoint"])
        manifest = {"schema": "neuroterrarium.model-set.v1", "status": "completed",
                    "protocol_sha256": first["config_sha256"], "source_sha256": first["source_sha256"],
                    "transitions_per_model": config["total_transitions"], "models": models}
        atomic_json(temporary/"manifest.json", manifest)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise
