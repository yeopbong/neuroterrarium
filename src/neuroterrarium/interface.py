"""Explicit engineering sensory injection and low-dimensional motor mapping.

Visual inputs are injected at projection neurons. Both sides receive the same
maximum positive local occupancy-change signal, bypassing retinal/optic-lobe
computation. This engineering input is not selectively responsive to looming.
GF supplies an escape-drive magnitude only. An independent, optional tonic
DNa02 drive supplies baseline activity without reading any observation value.
"""

from __future__ import annotations

import copy
import math
from numbers import Integral, Real

import numpy as np

from .world import ACTION_DT, RAYS, stream

NEURAL_DT_SECONDS = 0.0001
OBSERVATION_SIZE = 59
INPUT_GROUPS = ("sugar", "lplc2", "lc4", "dna02")


def _bounded_number(value: object, low: float, high: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"invalid {label}")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {label}") from error
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"invalid {label}")
    return result


def _channels(value: object) -> dict[str, bool]:
    if not isinstance(value, dict) or set(value) != {"taste", "vision"}:
        raise ValueError("invalid sensory channels")
    if any(type(enabled) is not bool for enabled in value.values()):
        raise ValueError("sensory channels must be boolean")
    return dict(value)


def _random_state(value: object) -> dict:
    """Validate the frozen PCG64 snapshot before assigning any live RNG state."""
    if not isinstance(value, dict) or set(value) != {
        "bit_generator", "state", "has_uint32", "uinteger"
    } or value["bit_generator"] != "PCG64":
        raise ValueError("invalid encoder random state")
    inner = value["state"]
    if not isinstance(inner, dict) or set(inner) != {"state", "inc"}:
        raise ValueError("invalid encoder random state")
    for number, maximum in [(inner["state"], 2**128-1), (inner["inc"], 2**128-1),
                            (value["has_uint32"], 1), (value["uinteger"], 2**32-1)]:
        if isinstance(number, bool) or not isinstance(number, Integral) or not 0 <= number <= maximum:
            raise ValueError("invalid encoder random state")
    if inner["inc"] % 2 != 1:
        raise ValueError("invalid encoder random increment")
    return copy.deepcopy(value)


class SensoryEncoder:
    def __init__(self, groups: dict[str,np.ndarray], sides: dict[str,np.ndarray], seed: int, *, tonic_hz:float=15.0):
        self.tonic_hz = _bounded_number(tonic_hz, 0, 50, "tonic drive")
        self.groups = {}
        for name in INPUT_GROUPS:
            indices = np.asarray(groups[name])
            if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices) or np.any(indices < 0) or np.any(indices > np.iinfo(np.int64).max):
                raise ValueError("invalid sensory group indices")
            self.groups[name] = indices.astype(np.int64, copy=True)
        self.input_neurons = np.concatenate(list(self.groups.values()))
        if len(np.unique(self.input_neurons)) != len(self.input_neurons):
            raise ValueError("duplicate sensory group indices")
        self.input_neurons = np.sort(self.input_neurons)
        self.random = {name:stream(seed,f"neural-input/{name}") for name in INPUT_GROUPS}
        self.channels = {"taste":True,"vision":True}
        self.last_rates_hz: dict[str,float] = {}

    def encode(self, observation: np.ndarray, steps: int = 200) -> list[np.ndarray]:
        raw_observation = np.asarray(observation)
        if raw_observation.dtype.kind not in "fiu":
            raise ValueError("observation schema mismatch")
        obs = np.asarray(raw_observation,dtype=float)
        if obs.shape != (OBSERVATION_SIZE,) or not np.isfinite(obs).all():
            raise ValueError("observation schema mismatch")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or not 1 <= steps <= 10000:
            raise ValueError("invalid neural window")
        low = np.zeros(OBSERVATION_SIZE)
        low[2*RAYS:3*RAYS] = -1
        low[56] = -1  # Previous applied turn action is signed.
        if np.any(obs < low) or np.any(obs > 1):
            raise ValueError("observation values outside schema range")
        channels = _channels(self.channels)
        tonic_hz = _bounded_number(self.tonic_hz, 0, 50, "tonic drive")
        expansion = np.maximum(obs[2*RAYS:3*RAYS],0)
        rate = {"sugar":200*obs[52]*channels["taste"],
                "lplc2":200*float(np.max(expansion))*channels["vision"],
                "lc4":200*float(np.max(expansion))*channels["vision"],
                "dna02":tonic_hz}
        self.last_rates_hz = rate
        events = [[] for _ in range(steps)]
        for name in INPUT_GROUPS:
            indices = self.groups[name]
            draws = self.random[name].random((steps,len(indices)))
            for step, cell in zip(*np.nonzero(draws < rate[name]*NEURAL_DT_SECONDS),strict=True):
                events[int(step)].append(int(indices[cell]))
        return [np.asarray(row,dtype=np.int64) for row in events]

    def snapshot(self) -> dict:
        return {"schema":"encoder-v1", "channels":dict(self.channels),"tonic_hz":self.tonic_hz,
                "random":{k:copy.deepcopy(v.bit_generator.state) for k,v in self.random.items()},
                "last_rates_hz":dict(self.last_rates_hz)}

    def restore(self,state:dict) -> None:
        if not isinstance(state, dict) or set(state) != {
            "schema", "channels", "tonic_hz", "random", "last_rates_hz"
        } or state["schema"] != "encoder-v1":
            raise ValueError("encoder schema mismatch")
        channels = _channels(state["channels"])
        tonic_hz = _bounded_number(state["tonic_hz"], 0, 50, "tonic drive")
        random_states = state["random"]
        if not isinstance(random_states, dict) or set(random_states) != set(INPUT_GROUPS):
            raise ValueError("encoder random streams mismatch")
        restored_random = {}
        for name in INPUT_GROUPS:
            generator = np.random.Generator(np.random.PCG64(0))
            generator.bit_generator.state = _random_state(random_states[name])
            restored_random[name] = generator
        rates = state["last_rates_hz"]
        if not isinstance(rates, dict) or set(rates) not in (set(), set(INPUT_GROUPS)):
            raise ValueError("encoder recorded rates mismatch")
        restored_rates = {
            name: _bounded_number(rate, 0, 50 if name == "dna02" else 200, "recorded input rate")
            for name, rate in rates.items()
        }
        self.channels, self.tonic_hz = channels, tonic_hz
        self.random, self.last_rates_hz = restored_random, restored_rates


class MotorReadout:
    """Accepts registered spike counts only, never a World or Observation.

    80 ms exponential rate filtering; 50 Hz MN9 feeding scale, 60 Hz GF
    escape scale, 60 Hz DNa steering/drive scale. These fixed initial scales
    are engineering assumptions awaiting behavior calibration.
    """
    def __init__(self, sides:dict[str,np.ndarray]):
        names = [f"{group}_{side}" for group in ("mn9","gf","dna01","dna02") for side in ("left","right")]
        self.sides = {}
        for key in names:
            indices=np.asarray(sides[key])
            if indices.ndim!=1 or indices.dtype.kind not in 'iu' or np.any(indices<0) or len(np.unique(indices))!=len(indices):
                raise ValueError('invalid motor group indices')
            self.sides[key]=indices.astype(np.int64)
        if any(len(v)==0 for v in self.sides.values()): raise ValueError("empty motor group")
        self.filtered = {key:0.0 for key in names}
        self.clamped = False

    def action(self, counts:np.ndarray, duration:float=ACTION_DT) -> np.ndarray:
        if duration <= 0 or not math.isfinite(duration): raise ValueError("invalid duration")
        counts = np.asarray(counts)
        if counts.ndim != 1 or counts.dtype.kind not in "iu" or np.any(counts<0):
            raise ValueError("spike counts must be nonnegative integers")
        alpha = 1-math.exp(-duration/0.08)
        for key,indices in self.sides.items():
            hz = float(np.mean(counts[indices]))/duration
            self.filtered[key] += alpha*(hz-self.filtered[key])
        if self.clamped: return np.zeros(4)
        f = self.filtered
        left = (f["dna01_left"]+f["dna02_left"])/2
        right = (f["dna01_right"]+f["dna02_right"])/2
        return np.clip([(left+right)/120,(left-right)/60,
                        max(f["gf_left"],f["gf_right"])/60,
                        max(f["mn9_left"],f["mn9_right"])/50], [0,-1,0,0],[1,1,1,1])

    def snapshot(self) -> dict:
        return {"schema":"readout-v1","filtered":dict(self.filtered),"clamped":self.clamped}

    def restore(self,state:dict) -> None:
        if not isinstance(state, dict) or set(state) != {"schema", "filtered", "clamped"}:
            raise ValueError("readout schema mismatch")
        if state["schema"] != "readout-v1" or not isinstance(state["filtered"], dict) or set(state["filtered"]) != set(self.filtered):
            raise ValueError("readout schema mismatch")
        if type(state["clamped"]) is not bool:
            raise ValueError("readout clamp must be boolean")
        filtered = {
            key: _bounded_number(value, 0, float("inf"), "readout state")
            for key, value in state["filtered"].items()
        }
        self.filtered, self.clamped = filtered, state["clamped"]
