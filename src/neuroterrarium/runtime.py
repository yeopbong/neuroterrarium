"""Authoritative local sessions, matched branches and structured snapshots.

Each action window samples every body's local observation before any body is
moved. Neural integration and policy inference use simulated time; their wall
cost slows the entire session. A fork shares immutable connectivity and policy
weights, while copying all dynamic state. This module never substitutes a
controller when verified data or weights are unavailable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .controllers import ARCHITECTURES, LearnedController, Policy, named_seed
from .data import GraphData, sha256_file
from .interface import (NEURAL_DT_SECONDS, MotorReadout, SensoryEncoder,
                        _random_state)
from .neural import SparseBrain
from .registry import Registry, default_path
from .world import (ACTION_DT, ACTION_SCHEMA, OBSERVATION_SCHEMA, RADIUS,
                    Food, Obstacle, Stimulus, World, stream)

SESSION_SCHEMA = "neuroterrarium.session.v1"
SNAPSHOT_SCHEMA = "neuroterrarium.snapshot.v1"
NEURAL_STEPS = round(ACTION_DT / NEURAL_DT_SECONDS)
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
MAX_EVENTS = 4096
MAX_RECORDS = 512
CHANNELS = {"proximity": slice(0, 16), "vision": slice(16, 48),
            "chemical": slice(48, 52), "taste": slice(52, 53)}
LOW_OBSERVATION = np.zeros(59)
LOW_OBSERVATION[32:48] = -1
LOW_OBSERVATION[56] = -1
SAFE_ID = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
BEHAVIOR_SOURCE_FILES = tuple(sorted(("brain_cache.py", "controllers.py", "data.py", "interface.py",
                                     "neural.py", "reference.py", "registry.py", "runtime.py", "world.py")))


def _json_bytes(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("state must contain finite structured JSON") from error


def _digest(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _wire_digest(value) -> str:
    """Hash numeric JSON values independently of integral-float spelling.

    JSON.parse/stringify writes 1.0 as 1. Both represent the same bounded model
    value. Large PCG integers are decimal strings and never enter this path.
    """
    def canonical(item):
        if isinstance(item, dict):
            return {key: canonical(value) for key, value in item.items()}
        if isinstance(item, list):
            return [canonical(value) for value in item]
        if isinstance(item, (float, np.floating)) and math.isfinite(item) and float(item).is_integer() and abs(item) <= 2**53 - 1:
            return int(item)
        return item
    return _digest(canonical(value))


def _rng_wire(state, *, decode=False):
    state = copy.deepcopy(state)
    for key in ("state", "inc"):
        value = state["state"][key]
        if decode:
            if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]{0,38}", value):
                raise ValueError("random stream integers must be decimal strings on the JSON boundary")
            state["state"][key] = int(value)
        else:
            state["state"][key] = str(value)
    if decode:
        return _random_state(state)
    return state


def _encoder_wire(state, *, decode=False):
    state = copy.deepcopy(state)
    state["random"] = {key: _rng_wire(value, decode=decode) for key, value in state["random"].items()}
    return state


def _number(value, minimum, maximum, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f"invalid {label}")
    return float(value)


def _integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {label}")
    return value


def _flag(value, label):
    if type(value) is not bool:
        raise ValueError(f"invalid {label}")
    return value


def _array(value, shape, low, high, label):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid {label}")
    if np.any(array < low) or np.any(array > high):
        raise ValueError(f"invalid {label} bounds")
    return array.astype(np.float64)


def _read_json(path: Path, limit=1024 * 1024):
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("required structured asset missing or too large")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def load_model_set(directory: Path | str) -> tuple[dict[str, Policy], dict]:
    """Verify nine completed independent training runs before loading weights."""
    from .training import load_policy

    directory = Path(directory)
    manifest = _read_json(directory / "manifest.json")
    if (not isinstance(manifest, dict) or manifest.get("schema") != "neuroterrarium.model-set.v1"
            or manifest.get("status") != "completed" or len(manifest.get("models", [])) != 9):
        raise ValueError("nine completed trained models are required")
    budget = _integer(manifest.get("transitions_per_model"), 1, 10**10, "training budget")
    identities, weight_hashes, models = set(), set(), {}
    seeds_by_architecture = {architecture: set() for architecture in ARCHITECTURES}
    for item in manifest["models"]:
        name, checkpoint = item.get("id"), item.get("checkpoint")
        if (not isinstance(name, str) or not SAFE_ID.fullmatch(name) or name == "connectome"
                or not isinstance(checkpoint, str) or not SAFE_ID.fullmatch(checkpoint)
                or name in models or item.get("architecture") not in ARCHITECTURES):
            raise ValueError("invalid or duplicate model identity")
        path = directory / checkpoint
        if path.is_symlink() or path.resolve().parent != directory.resolve():
            raise ValueError("checkpoint must remain inside the model set")
        if sha256_file(path / "manifest.json") != item.get("checkpoint_manifest_sha256"):
            raise ValueError("model checkpoint manifest checksum mismatch")
        policy = load_policy(path)
        metadata = _read_json(path / "state.json", 128 * 1024 * 1024)
        _integer(item.get("seed"), 0, 2**53 - 1, "model seed")
        identity = (policy.architecture, policy.seed)
        digest = sha256_file(path / "policy.safetensors")
        if (identity != (item["architecture"], item["seed"]) or identity in identities
                or digest != item.get("policy_sha256") or digest in weight_hashes):
            raise ValueError("model identity or independent weight checksum mismatch")
        if (item.get("training_status") != "completed" or metadata.get("training_status") != "completed"
                or item.get("transitions") != budget or metadata.get("transitions") != budget
                or metadata.get("config_sha256") != manifest.get("protocol_sha256")
                or metadata.get("source_sha256") != manifest.get("source_sha256")
                or metadata.get("config", {}).get("status") != "frozen"
                or metadata["config"].get("total_transitions") != budget
                or policy.seed not in metadata["config"].get("seeds", [])
                or _digest(metadata["config"]) != manifest["protocol_sha256"]):
            raise ValueError("incomplete or incompatible training protocol")
        if (metadata["policy"].get("observation_schema") != OBSERVATION_SCHEMA
                or metadata["policy"].get("action_schema") != ACTION_SCHEMA
                or metadata["policy"].get("observation_size") != 59
                or policy.hidden_size != metadata["config"].get("hidden_size")):
            raise ValueError("trained model observation or action schema mismatch")
        identities.add(identity)
        seeds_by_architecture[policy.architecture].add(policy.seed)
        weight_hashes.add(digest)
        models[name] = policy
    seeds = list(seeds_by_architecture.values())
    if any(len(value) != 3 or value != seeds[0] for value in seeds):
        raise ValueError("model set must contain three independent seeds per mechanism")
    return models, manifest


@dataclass(frozen=True, eq=False)
class GraphIdentity:
    """Read-only provenance after the complete graph is stored in brain CSR.

    This is metadata, not an edge list or a reduced circuit. The simulator's
    immutable CSR arrays retain every verified connection. Keeping the loader's
    three coordinate arrays as well would duplicate their storage in a session.
    """

    root_ids: np.ndarray
    connection_rows: int
    connectivity_sha256: str
    _summary_json: str

    @classmethod
    def from_graph(cls, graph: GraphData, brain: SparseBrain) -> GraphIdentity:
        if brain.n_neurons != len(graph.root_ids) or len(brain.post_indices) != len(graph.pre):
            raise ValueError("complete graph storage and identity disagree")
        # Bytes-backed storage cannot be made writeable by a consumer. It also
        # avoids retaining a view of any loader allocation.
        roots = np.frombuffer(graph.root_ids.tobytes(), dtype=graph.root_ids.dtype)
        return cls(roots, len(graph.pre), brain.graph_digest, _json_bytes(graph.summary).decode("utf-8"))

    @property
    def summary(self) -> dict:
        """Return detached display metadata; callers cannot mutate identity."""
        return json.loads(self._summary_json)


class Session:
    """A complete local world; no network access is needed after preparation."""

    def __init__(self, data_dir: str | Path, models_dir: str | Path, *, seed: int = 0,
                 scenario: str = "open"):
        from .brain_cache import ensure_cache, load_cache
        policies, manifest = load_model_set(models_dir)
        ensure_cache(data_dir)
        cached = load_cache(data_dir)
        registry = Registry.load()
        self._initialize(cached, policies, registry, seed, scenario, "Local full graph", _digest(manifest),
                         prepared_brain=cached.brain)

    @classmethod
    def from_components(cls, graph: GraphData, policies: dict[str, Policy], *,
                        registry: Registry | None = None, seed: int = 0,
                        scenario: str = "open", tonic_hz: float = 15.0) -> Session:
        """Explicit test harness. It cannot label synthetic graphs as full brain."""
        result = object.__new__(cls)
        result._initialize(graph, policies, registry or Registry.load(), seed, scenario,
                           "Test components", "test-components", tonic_hz)
        return result

    def _initialize(self, graph, policies, registry, seed, scenario, mode, model_digest, tonic_hz=15.0,
                    prepared_brain=None):
        self.seed = _integer(seed, 0, 2**53 - 1, "seed")
        if len(policies) != 9 or any(not SAFE_ID.fullmatch(name) or name == "connectome" for name in policies):
            raise ValueError("a ten-body session requires nine identified policies")
        self.registry, self.policies = registry, dict(policies)
        self.groups, self.sides = registry.resolve(graph.root_ids), registry.resolve_sides(graph.root_ids)
        self.mode, self.model_set_digest = mode, model_digest
        modules = Path(__file__).parent
        self.behavior_sha256 = _digest({name: sha256_file(modules / name) for name in BEHAVIOR_SOURCE_FILES})
        self.interface_sha256 = sha256_file(default_path().with_name("brain-interface.json"))
        self._tonic_hz = tonic_hz
        self.encoder = SensoryEncoder(self.groups, self.sides, named_seed(seed, "brain-encoder"), tonic_hz=tonic_hz)
        if prepared_brain is None:
            self.brain = SparseBrain(len(graph.root_ids), graph.pre, graph.post, graph.signed_counts,
                                     input_neurons=self.encoder.input_neurons)
            self.graph = GraphIdentity.from_graph(graph, self.brain)
        else:
            if not np.array_equal(np.flatnonzero(prepared_brain.input_mask), np.sort(self.encoder.input_neurons)):
                raise ValueError("prepared brain input sites differ from the interface")
            self.brain = prepared_brain
            self.graph = GraphIdentity(graph.root_ids, len(self.brain.post_indices), self.brain.graph_digest,
                                       _json_bytes(graph.summary).decode("utf-8"))
        self._initial_brain = self.brain.fork()
        self.readout = MotorReadout(self.sides)
        self.model_weights = {name: LearnedController(policy, 0).weights_sha256 for name, policy in policies.items()}
        self._start_new_world(scenario)

    def _start_new_world(self, scenario):
        self.world = World(self.seed, 10, scenario)
        self.world.replenish = True
        allocation = np.asarray(["connectome", *sorted(self.policies)])
        stream(self.seed, "controller-allocation").shuffle(allocation)
        self.allocation = allocation.tolist()
        self.controllers = [None if name == "connectome" else LearnedController(
            self.policies[name], named_seed(self.seed, f"body/{index}/policy"))
            for index, name in enumerate(self.allocation)]
        self.brain = self._initial_brain.fork()
        self.encoder = SensoryEncoder(self.groups, self.sides, named_seed(self.seed, "brain-encoder"), tonic_hz=self._tonic_hz)
        self.readout = MotorReadout(self.sides)
        self.channels = dict.fromkeys(CHANNELS, True)
        self.noise_std = 0.0
        self.noise_streams = [stream(self.seed, f"body/{index}/observation-noise") for index in range(10)]
        self.paused, self.speed, self.revealed = False, 1.0, False
        self.events, self.records = deque(maxlen=MAX_EVENTS), deque(maxlen=MAX_RECORDS)
        self._last_observations = self.world.observe().copy()
        self._observation_time = self.world.time
        self._last_rates = dict.fromkeys(self.groups, 0.0)
        self._last_inputs = {}
        self._last_actions = np.zeros((10, 4))
        self._wall_start, self._wall_offset = perf_counter(), 0.0
        self.compute_seconds = 0.0
        self.fork_session = None
        self.error = None

    @property
    def scenario(self):
        return self.world.scenario

    @property
    def wall_time(self):
        return self._wall_offset + perf_counter() - self._wall_start

    def _event(self, kind, detail=None):
        self.events.append({"time": self.world.time, "type": kind, "detail": detail or {}})

    def _observations(self):
        observations = self.world.observe().copy()
        for index, rng in enumerate(self.noise_streams):
            # A fixed-size draw on every action window keeps noise amplitude and
            # panel rendering independent of the random-stream schedule.
            noise = rng.normal(size=53)
            observations[index, :53] += self.noise_std * noise
        observations = np.clip(observations, LOW_OBSERVATION, 1)
        for name, section in CHANNELS.items():
            if not self.channels[name]:
                observations[:, section] = 0
        return observations

    def _capture(self):
        # The numerical backend exposes the same bounded transaction helper it
        # uses for advance(); only live delayed events are copied, not capacity.
        return (self.world.snapshot(), self.brain._capture_mutable_state(), self.encoder.snapshot(),
                self.readout.snapshot(), [None if c is None else c.snapshot() for c in self.controllers],
                [copy.deepcopy(r.bit_generator.state) for r in self.noise_streams],
                self._last_observations.copy(), self._observation_time, self._last_actions.copy(),
                dict(self._last_rates), copy.deepcopy(self._last_inputs), self.compute_seconds)

    def _rollback(self, captured):
        (world, brain, encoder, readout, policies, random, self._last_observations,
         self._observation_time, self._last_actions, self._last_rates, self._last_inputs,
         self.compute_seconds) = captured
        self.world = World.restore(world)
        self.brain._restore_mutable_state(brain)
        self.encoder.restore(encoder)
        self.readout.restore(readout)
        for controller, saved in zip(self.controllers, policies, strict=True):
            if controller is not None:
                controller.restore(saved)
        for rng, saved in zip(self.noise_streams, random, strict=True):
            rng.bit_generator.state = saved

    def _step_once(self):
        started = perf_counter()
        observations = self._observations()
        actions = np.zeros((10, 4))
        decisions = []
        for index, controller in enumerate(self.controllers):
            if controller is None:
                tape = self.encoder.encode(observations[index], steps=NEURAL_STEPS)
                result = self.brain.advance(tape)
                actions[index] = self.readout.action(result.spike_counts, ACTION_DT)
                self._last_rates = {name: float(result.spike_counts[indices].mean() / ACTION_DT)
                                    for name, indices in self.groups.items()}
                input_counts = np.bincount(np.concatenate(tape), minlength=self.brain.n_neurons)
                self._last_inputs = {name: int(input_counts[indices].sum())
                                     for name, indices in self.groups.items() if name in self.encoder.random}
                decisions.append({"raw_action": actions[index].tolist(), "log_probability": None,
                                  "applied_action": actions[index].tolist(), "neural_input_counts": self._last_inputs,
                                  "neural_output_rates_hz": dict(self._last_rates)})
            else:
                actions[index] = controller.act(observations[index])
                decisions.append(copy.deepcopy(controller.last))
        self._last_observations = observations
        self._observation_time = self.world.time
        self.world.advance(actions)
        self._last_actions = actions.copy()
        self.compute_seconds += perf_counter() - started
        return {"time": self._observation_time, "observations": observations.tolist(), "decisions": decisions}

    def step(self):
        """Advance one complete window, even when paused (scheduler checks it)."""
        sessions = [self] + ([self.fork_session] if self.fork_session else [])
        captured = [session._capture() for session in sessions]
        try:
            records = [session._step_once() for session in sessions]
        except BaseException:
            for session, saved in zip(sessions, captured, strict=True):
                session._rollback(saved)
                session.paused = True
                session.error = "Controller execution failed; the complete action window was rolled back."
            raise
        for session, record in zip(sessions, records, strict=True):
            session.records.append(record)
            session.error = None
        return self.world.time

    def _world_state(self):
        return {"width": self.world.width, "height": self.world.height,
                **{name: [asdict(item) for item in getattr(self.world, name)]
                   for name in ("bodies", "foods", "obstacles", "stimuli")}}

    def state(self, selected: int = 0, reveal: bool = False):
        selected = _integer(selected, 0, 9, "selected individual")
        reveal = _flag(reveal, "reveal") or self.revealed
        controller = self.controllers[selected]
        kind = "connectome" if controller is None else controller.policy.architecture
        panel = {"index": selected, "label": f"Fly {selected + 1:02d}", "kind": kind if reveal else None,
                 "observation": self._last_observations[selected].tolist(),
                 "observation_time": self._observation_time,
                 "body": asdict(self.world.bodies[selected]), "action": self._last_actions[selected].tolist()}
        if reveal and controller is None:
            panel.update(neural_rates_hz=dict(self._last_rates), neural_inputs_hz=dict(self.encoder.last_rates_hz))
        elif reveal:
            panel["hidden"] = controller.hidden[0].tolist() if controller.policy.recurrent else []
            panel["policy_decision"] = copy.deepcopy(controller.last)
            panel["reflex_enabled"] = controller.reflex_enabled
        wall = self.wall_time
        result = {"mode": self.mode, "ready": self.error is None, "paused": self.paused,
                  "simulation_time": self.world.time, "wall_time": wall,
                  "rate": self.world.time / wall if wall > 0 else 0.0,
                  "compute_seconds": self.compute_seconds, "speed": self.speed, "scenario": self.scenario,
                  "world": self._world_state(), "selected": panel, "events": list(self.events),
                  "channels": dict(self.channels), "noise_std": self.noise_std,
                  "readout_clamped": self.readout.clamped}
        if self.error:
            result["error"] = self.error
        if self.fork_session:
            result["fork"] = self.fork_session.state(selected, reveal)
        return result

    def _clone(self):
        clone = object.__new__(Session)
        clone.__dict__ = self.__dict__.copy()
        clone.world = World.restore(self.world.snapshot())
        clone.brain = self.brain.fork()
        clone.encoder = SensoryEncoder(self.groups, self.sides, 0, tonic_hz=self._tonic_hz)
        clone.encoder.restore(self.encoder.snapshot())
        clone.readout = MotorReadout(self.sides)
        clone.readout.restore(self.readout.snapshot())
        clone.controllers = [None if controller is None else controller.clone() for controller in self.controllers]
        clone.allocation = list(self.allocation)
        clone.channels = dict(self.channels)
        clone.noise_streams = [stream(0, "copy") for _ in self.noise_streams]
        for rng, original in zip(clone.noise_streams, self.noise_streams, strict=True):
            rng.bit_generator.state = copy.deepcopy(original.bit_generator.state)
        clone.events = deque(copy.deepcopy(list(self.events)), maxlen=MAX_EVENTS)
        clone.records = deque(copy.deepcopy(list(self.records)), maxlen=MAX_RECORDS)
        clone._last_observations = self._last_observations.copy()
        clone._last_actions = self._last_actions.copy()
        clone._last_rates, clone._last_inputs = dict(self._last_rates), copy.deepcopy(self._last_inputs)
        clone.fork_session = None
        return clone

    def _reset_controller(self, index, name):
        self.allocation[index] = name
        if name == "connectome":
            interventions = {"disconnect_output": np.flatnonzero(self.brain.output_disconnected),
                             "disconnect_input": np.flatnonzero(self.brain.input_disconnected),
                             "clamp_spikes": np.flatnonzero(self.brain.spike_clamped)}
            clamped = self.readout.clamped
            encoder_channels = dict(self.encoder.channels)
            tonic_hz = self.encoder.tonic_hz
            self.brain = self._initial_brain.fork()
            self.brain.step = self.world.step * NEURAL_STEPS
            self.brain.set_interventions(**interventions)
            self.encoder = SensoryEncoder(self.groups, self.sides, named_seed(self.seed, "brain-encoder"), tonic_hz=tonic_hz)
            self.encoder.channels = encoder_channels
            self.readout = MotorReadout(self.sides)
            self.readout.clamped = clamped
            self.controllers[index] = None
            self._last_rates = dict.fromkeys(self.groups, 0.0)
            self._last_inputs = {}
        else:
            self.controllers[index] = LearnedController(self.policies[name], named_seed(self.seed, f"body/{index}/policy"))
        # Clear the pending/last controller action symmetrically; body inertia is
        # preserved. Both sides see this same action-history reset next window.
        self._last_actions[index] = 0
        self.world.bodies[index].action = [0.0] * 4

    def execute(self, command: dict):
        if not isinstance(command, dict) or len(_json_bytes(command)) > 4096:
            raise ValueError("invalid command object")
        kind = command.get("type")
        branch = command.get("branch", "left")
        if branch not in ("left", "right"):
            raise ValueError("unknown branch")
        if branch == "right":
            if not self.fork_session:
                raise ValueError("right branch does not exist")
            if kind in {"fork", "fork_close", "reset", "scenario", "step", "pause", "resume", "speed"}:
                raise ValueError("clock and branch lifecycle commands apply to the complete session")
            clean = {key: value for key, value in command.items() if key != "branch"}
            return self.fork_session.execute(clean)
        selected = _integer(command.get("selected", 0), 0, 9, "selected individual")
        common = {"type", "branch", "selected"}
        fields = {"pause": set(), "resume": set(), "step": set(), "reset": set(),
                  "scenario": {"scenario"}, "speed": {"value"},
                  "channels": {"channel", "enabled"}, "noise": {"value"},
                  "neural_disconnect": {"group"}, "neural_input_disconnect": {"group"},
                  "spike_clamp": {"group"}, "readout_clamp": {"value"},
                  "restore_interventions": set(), "sham": set(),
                  "fork": {"replacement"}, "fork_close": set(), "guess": {"guess"}, "reveal": {"value"},
                  "food_add": {"x", "y", "radius", "amount"}, "food_move": {"index", "x", "y"},
                  "food_remove": {"index"}, "stimulus_add": {"x", "y", "radius", "vx", "vy", "growth", "physical"},
                  "stimulus_move": {"index", "x", "y"}, "stimulus_remove": {"index"},
                  "obstacle_add": {"x", "y", "radius"}, "obstacle_remove": {"index"},
                  "hybrid_reflex": {"value"}}
        if kind not in fields or set(command) - common - fields[kind]:
            raise ValueError("unknown command or field")
        detail = {key: value for key, value in command.items() if key not in common}
        if kind in {"pause", "resume"}:
            self.paused = kind == "pause"
            if self.fork_session:
                self.fork_session.paused = self.paused
        elif kind == "step":
            if not self.paused:
                raise ValueError("pause before single stepping")
            self.step()
        elif kind in {"reset", "scenario"}:
            scenario = command.get("scenario", self.scenario)
            if scenario not in {"open", "occluded", "looming"}:
                raise ValueError("unknown scenario")
            self._start_new_world(scenario)
        elif kind == "speed":
            self.speed = _number(command.get("value"), 0.05, 4, "requested speed")
            if self.fork_session:
                self.fork_session.speed = self.speed
        elif kind == "channels":
            channel = command.get("channel")
            if channel not in CHANNELS:
                raise ValueError("unknown shared sensory channel")
            self.channels[channel] = _flag(command.get("enabled"), "channel enabled")
        elif kind == "noise":
            self.noise_std = _number(command.get("value"), 0, 0.2, "observation noise")
        elif kind in {"neural_disconnect", "neural_input_disconnect", "spike_clamp"}:
            group = command.get("group")
            if group not in self.groups or "connectome" not in self.allocation:
                raise ValueError("registered group or active connectome unavailable")
            argument, mask = {"neural_disconnect": ("disconnect_output", self.brain.output_disconnected),
                              "neural_input_disconnect": ("disconnect_input", self.brain.input_disconnected),
                              "spike_clamp": ("clamp_spikes", self.brain.spike_clamped)}[kind]
            indices = np.union1d(np.flatnonzero(mask), self.groups[group])
            self.brain.set_interventions(**{argument: indices})
            detail["pending_event_semantics"] = ("output/input masks apply at arrival; spike clamp prevents future emission"
                                                  if kind != "spike_clamp" else "existing queued events remain")
        elif kind == "readout_clamp":
            if "connectome" not in self.allocation:
                raise ValueError("branch has no active connectome readout")
            self.readout.clamped = _flag(command.get("value", True), "readout clamp")
        elif kind == "hybrid_reflex":
            controller = self.controllers[selected]
            if controller is None or controller.policy.architecture != "hybrid":
                raise ValueError("selected controller has no hybrid reflex")
            controller.reflex_enabled = _flag(command.get("value"), "hybrid reflex")
        elif kind == "restore_interventions":
            self.channels = dict.fromkeys(CHANNELS, True)
            self.noise_std = 0.0
            self.brain.set_interventions(disconnect_output=[], disconnect_input=[], clamp_spikes=[])
            self.readout.clamped = False
            self.encoder.channels = {"taste": True, "vision": True}
            for controller in self.controllers:
                if controller is not None:
                    controller.reflex_enabled = True
        elif kind == "fork":
            if self.fork_session:
                raise ValueError("close the existing branch before creating another")
            replacement = command.get("replacement", "same")
            if replacement != "same" and (replacement not in self.policies or self.allocation[selected] != "connectome"):
                raise ValueError("replacement currently supports the selected connectome with a registered learned model")
            self.fork_session = self._clone()
            if replacement != "same":
                self._reset_controller(selected, "connectome")
                self.fork_session._reset_controller(selected, replacement)
                detail["initialization"] = "symmetric_reset"
            else:
                detail["initialization"] = "identical_snapshot"
            self.fork_session._event("fork", detail)
        elif kind == "fork_close":
            self.fork_session = None
        elif kind == "guess":
            guess = command.get("guess")
            if guess not in {"connectome", *ARCHITECTURES}:
                raise ValueError("unknown controller guess")
            actual = self.state(selected, True)["selected"]["kind"]
            detail.update(individual=selected, answer=actual, correct=guess == actual)
            self._event(kind, detail)
            return {"message": f"Fly {selected + 1:02d}: {actual}. {'Correct guess.' if guess == actual else 'Revealed.'}", **detail}
        elif kind == "reveal":
            self.revealed = _flag(command.get("value"), "reveal")
        elif kind == "sham":
            pass
        else:
            # External edits are matched by default. Validate both prospective
            # worlds before mutating either so a blocked edit cannot split them.
            targets = [self] + ([self.fork_session] if self.fork_session else [])
            worlds = [World.restore(target.world.snapshot()) for target in targets]
            for world in worlds:
                self._edit_world(world, kind, command)
            for target, world in zip(targets, worlds, strict=True):
                target.world = world
                if target is not self:
                    target._event(kind, detail)
        self._event(kind, detail)
        return {"ok": True, "simulation_time": self.world.time}

    @staticmethod
    def _edit_world(world, kind, command):
        family, operation = kind.rsplit("_", 1)
        objects = getattr(world, {"food": "foods", "stimulus": "stimuli", "obstacle": "obstacles"}[family])
        if operation in {"remove", "move"}:
            index = _integer(command.get("index"), 0, len(objects) - 1, "object index")
            if operation == "remove":
                del objects[index]
                return
        x = _number(command.get("x"), 0, world.width, "x position")
        y = _number(command.get("y"), 0, world.height, "y position")
        if operation == "move":
            objects[index].x, objects[index].y = x, y
            return
        capacity = 32 if family == "stimulus" else 256
        if len(objects) >= capacity:
            raise ValueError("world object capacity reached")
        radius = _number(command.get("radius", 1.2 if family == "food" else 1.8 if family == "obstacle" else 1),
                         0.1, 10, "object radius")
        if family == "food":
            item = Food(x, y, _number(command.get("amount", 1), 0.01, 10, "food amount"), radius)
        elif family == "obstacle":
            if any(math.hypot(body.x - x, body.y - y) <= radius + RADIUS for body in world.bodies):
                raise ValueError("an obstacle cannot be placed across a body")
            item = Obstacle(x, y, radius)
        else:
            item = Stimulus(x, y, radius, _number(command.get("vx", 0), -20, 20, "stimulus vx"),
                            _number(command.get("vy", 0), -20, 20, "stimulus vy"),
                            _number(command.get("growth", 0), -10, 10, "stimulus growth"),
                            _flag(command.get("physical", False), "physical stimulus"))
        objects.append(item)

    def _payload(self):
        return {"schema": SESSION_SCHEMA, "mode": self.mode, "seed": self.seed,
                "graph_sha256": self.brain.graph_digest, "registry_sha256": self.registry.digest,
                "behavior_sha256": self.behavior_sha256, "interface_sha256": self.interface_sha256,
                "model_set_sha256": self.model_set_digest, "model_weights": self.model_weights,
                "observation_schema": OBSERVATION_SCHEMA, "action_schema": ACTION_SCHEMA,
                "world": self.world.snapshot(), "allocation": list(self.allocation),
                "brain": self.brain.snapshot(), "encoder": _encoder_wire(self.encoder.snapshot()), "readout": self.readout.snapshot(),
                "controllers": [None if c is None else c.snapshot() for c in self.controllers],
                "channels": dict(self.channels), "noise_std": self.noise_std,
                "noise_streams": [_rng_wire(r.bit_generator.state) for r in self.noise_streams],
                "paused": self.paused, "speed": self.speed, "revealed": self.revealed,
                "wall_time": self.wall_time, "compute_seconds": self.compute_seconds,
                "last_observations": self._last_observations.tolist(), "observation_time": self._observation_time,
                "last_actions": self._last_actions.tolist(), "last_rates": dict(self._last_rates),
                "last_input_counts": copy.deepcopy(self._last_inputs), "events": list(self.events),
                "records": list(self.records), "fork": self.fork_session._payload() if self.fork_session else None}

    def snapshot(self):
        payload = self._payload()
        envelope = {"schema": SNAPSHOT_SCHEMA, "sha256": _wire_digest(payload), "state": payload}
        if len(_json_bytes(envelope)) > MAX_SNAPSHOT_BYTES:
            # Records have a separate bounded replay/export channel. Preserve
            # the complete executable state even when history fills the budget.
            payload["records"] = []
            if payload["fork"]:
                payload["fork"]["records"] = []
            envelope["sha256"] = _wire_digest(payload)
            if len(_json_bytes(envelope)) > MAX_SNAPSHOT_BYTES:
                raise ValueError("complete snapshot exceeds the snapshot resource budget")
        return envelope

    def restore(self, envelope):
        if (not isinstance(envelope, dict) or set(envelope) != {"schema", "sha256", "state"}
                or envelope["schema"] != SNAPSHOT_SCHEMA or len(_json_bytes(envelope)) > MAX_SNAPSHOT_BYTES
                or _wire_digest(envelope["state"]) != envelope["sha256"]):
            raise ValueError("snapshot schema, size or checksum mismatch")
        candidate = self._clone()
        candidate._restore_payload(envelope["state"])
        self.__dict__ = candidate.__dict__

    def _restore_payload(self, payload, depth=0):
        expected = set(self._payload())
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("snapshot session fields mismatch")
        identities = {"schema": SESSION_SCHEMA, "mode": self.mode,
                      "graph_sha256": self.brain.graph_digest, "registry_sha256": self.registry.digest,
                      "behavior_sha256": self.behavior_sha256, "interface_sha256": self.interface_sha256,
                      "model_set_sha256": self.model_set_digest, "model_weights": self.model_weights,
                      "observation_schema": OBSERVATION_SCHEMA, "action_schema": ACTION_SCHEMA}
        if any(payload[key] != value for key, value in identities.items()):
            raise ValueError("snapshot data, models or schema mismatch")
        self.seed = _integer(payload["seed"], 0, 2**53 - 1, "snapshot seed")
        world = World.restore(payload["world"])
        if len(world.bodies) != 10 or world.seed != self.seed:
            raise ValueError("snapshot world identity mismatch")
        allocation = payload["allocation"]
        if (not isinstance(allocation, list) or len(allocation) != 10
                or any(name not in {"connectome", *self.policies} for name in allocation)
                or allocation.count("connectome") > 1 or len(payload["controllers"]) != 10):
            raise ValueError("snapshot controller allocation mismatch")
        if depth == 0 and sorted(allocation) != sorted(["connectome", *self.policies]):
            raise ValueError("primary world must retain the complete controller set")
        self.world, self.allocation = world, list(allocation)
        self.brain.restore(payload["brain"])
        if "connectome" in allocation and self.brain.step != world.step * NEURAL_STEPS:
            raise ValueError("neural and body snapshot clocks disagree")
        self.encoder.restore(_encoder_wire(payload["encoder"], decode=True))
        self.readout.restore(payload["readout"])
        self.controllers = []
        for name, state in zip(allocation, payload["controllers"], strict=True):
            if name == "connectome":
                if state is not None:
                    raise ValueError("connectome cannot contain policy hidden state")
                self.controllers.append(None)
            else:
                controller = LearnedController(self.policies[name], 0)
                controller.restore(state)
                if not isinstance(controller.last, dict) or len(_json_bytes(controller.last)) > 4096:
                    raise ValueError("invalid recorded policy decision")
                self.controllers.append(controller)
        if not isinstance(payload["channels"], dict) or set(payload["channels"]) != set(CHANNELS):
            raise ValueError("snapshot channel schema mismatch")
        self.channels = {name: _flag(value, "channel") for name, value in payload["channels"].items()}
        self.noise_std = _number(payload["noise_std"], 0, 0.2, "noise")
        random = payload["noise_streams"]
        if not isinstance(random, list) or len(random) != 10:
            raise ValueError("snapshot noise stream count mismatch")
        self.noise_streams = [stream(0, "restored") for _ in random]
        for rng, saved in zip(self.noise_streams, random, strict=True):
            rng.bit_generator.state = _rng_wire(saved, decode=True)
        self.paused, self.revealed = _flag(payload["paused"], "paused"), _flag(payload["revealed"], "reveal")
        self.speed = _number(payload["speed"], 0.05, 4, "speed")
        self._wall_offset = _number(payload["wall_time"], 0, 10**9, "wall time")
        self._wall_start = perf_counter()
        self.compute_seconds = _number(payload["compute_seconds"], 0, 10**9, "compute time")
        self._last_observations = _array(payload["last_observations"], (10, 59), LOW_OBSERVATION, 1, "observations")
        self._last_actions = _array(payload["last_actions"], (10, 4), [0, -1, 0, 0], 1, "actions")
        self._observation_time = _number(payload["observation_time"], 0, world.time, "observation time")
        if self._observation_time not in {world.time, max(0, world.time - ACTION_DT)} and not math.isclose(self._observation_time, world.time - ACTION_DT, abs_tol=1e-9):
            raise ValueError("snapshot observation clock mismatch")
        rates = payload["last_rates"]
        if not isinstance(rates, dict) or set(rates) != set(self.groups):
            raise ValueError("snapshot neural rate groups mismatch")
        self._last_rates = {key: _number(value, 0, 10000, "neural rate") for key, value in rates.items()}
        counts = payload["last_input_counts"]
        if not isinstance(counts, dict) or any(key not in self.encoder.random for key in counts):
            raise ValueError("snapshot neural input groups mismatch")
        self._last_inputs = {key: _integer(value, 0, 10000000, "input count") for key, value in counts.items()}
        events, records = payload["events"], payload["records"]
        if not isinstance(events, list) or len(events) > MAX_EVENTS or not isinstance(records, list) or len(records) > MAX_RECORDS:
            raise ValueError("snapshot history capacity exceeded")
        for event in events:
            if (not isinstance(event, dict) or set(event) != {"time", "type", "detail"}
                    or not isinstance(event["type"], str) or len(event["type"]) > 64
                    or len(_json_bytes(event["detail"])) > 4096):
                raise ValueError("invalid event history")
            _number(event["time"], 0, world.time, "event time")
        for record in records:
            if not isinstance(record, dict) or set(record) != {"time", "observations", "decisions"}:
                raise ValueError("invalid action record")
            _number(record["time"], 0, world.time, "record time")
            _array(record["observations"], (10, 59), LOW_OBSERVATION, 1, "record observations")
            if not isinstance(record["decisions"], list) or len(record["decisions"]) != 10 or len(_json_bytes(record["decisions"])) > 64 * 1024:
                raise ValueError("invalid recorded decisions")
        self.events = deque(copy.deepcopy(events), maxlen=MAX_EVENTS)
        self.records = deque(copy.deepcopy(records), maxlen=MAX_RECORDS)
        self.fork_session = None
        if payload["fork"] is not None:
            if depth:
                raise ValueError("nested branches are unsupported")
            self.fork_session = self._clone()
            self.fork_session._restore_payload(payload["fork"], depth + 1)
            if self.fork_session.world.step != world.step:
                raise ValueError("branch clocks disagree")
        self.error = None

    def save(self, path: Path | str):
        """Atomic library/CLI export; local web API does not accept file paths."""
        path = Path(path)
        data = _json_bytes(self.snapshot())
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".snapshot-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, path: Path | str):
        self.restore(_read_json(Path(path), MAX_SNAPSHOT_BYTES))
