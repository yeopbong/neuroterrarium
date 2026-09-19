"""Persistent full-graph LIF dynamics with ordered sparse synaptic delivery.

The model equations and parameters follow Shiu and Spiller's MIT-licensed
Drosophila_brain_model, commit 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960.
The required notice is retained in :mod:`neuroterrarium.reference`.
No neuron or edge is removed, and no network weight is learned here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import time

import numpy as np
from numba import njit

from .reference import DT_MS, EXTERNAL_INCREMENT_MV, _indices

DELAY_STEPS = 18
QUEUE_SLOTS = DELAY_STEPS + 1
REFRACTORY_STEPS = 22
NEVER_SPIKED = -(1 << 50)
STATE_SCHEMA = "neuroterrarium.neural-state.v1"


class NeuralResourceError(RuntimeError):
    """A requested operation exceeds the configured finite resource budget."""


@dataclass(frozen=True)
class NeuralLimits:
    """Host resource limits, separate from the fixed model dynamics.

    Longer simulations must stream bounded windows. None of these limits
    silently truncates a tape, graph, trace, or spike record.
    """

    max_steps: int = 20_000
    max_input_events: int = 2_000_000
    max_trace_bytes: int = 64 * 1024 * 1024
    max_recorded_spikes: int = 1_000_000
    max_interventions: int = 256
    max_graph_working_bytes: int = 1024 * 1024 * 1024
    max_neuron_updates: int = 3_000_000_000

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@njit(cache=True, fastmath=False)
def _advance_kernel(
    indptr, posts, weights, input_mask,
    v, g, last_spike, queue, queue_counts,
    output_disconnected, input_disconnected, spike_clamped,
    tape_offsets, tape_neurons, recorded, start_step, record_spikes, spike_capacity,
):
    """Integrate every neuron at every step; gather only firing sources' edges."""
    n = len(v)
    steps = len(tape_offsets) - 1
    v_trace = np.empty((steps, len(recorded)), dtype=np.float64)
    g_trace = np.empty((steps, len(recorded)), dtype=np.float64)
    active = np.empty(n, dtype=np.bool_)
    fired = np.empty(n, dtype=np.int32)
    spike_counts = np.zeros(n, dtype=np.int64)
    saved_steps = [np.int64(0) for _ in range(0)]
    saved_neurons = [np.int64(0) for _ in range(0)]
    a = np.exp(-0.1 / 20.0)
    b = np.exp(-0.1 / 5.0)
    coupling = (5.0 / 15.0) * (a - b)
    for local_step in range(steps):
        step = start_step + local_step
        fired_count = 0
        # Brian's groups slot: the refractory boundary is >=, not >.
        for neuron in range(n):
            refractory_steps = 0 if input_mask[neuron] else REFRACTORY_STEPS
            active[neuron] = step - last_spike[neuron] >= refractory_steps
            if active[neuron]:
                old_g = g[neuron]
                v[neuron] = -52.0 + a * (v[neuron] + 52.0) + coupling * old_g
                g[neuron] = b * old_g
        # Brian's thresholds slot: just-spiking cells become read-only for
        # synaptic writes, even for external-input cells with rfc=0.
        for neuron in range(n):
            if active[neuron] and v[neuron] > -45.0 and not spike_clamped[neuron]:
                fired[fired_count] = neuron
                fired_count += 1
                last_spike[neuron] = step
                active[neuron] = False
                spike_counts[neuron] += 1
                if record_spikes:
                    if len(saved_steps) >= spike_capacity:
                        raise RuntimeError("neural spike recording capacity exceeded")
                    saved_steps.append(np.int64(step))
                    saved_neurons.append(np.int64(neuron))
        # Internal synapses run at order -1. The queue carries source events,
        # not weighted currents, so interventions apply at delivery time.
        slot = step % QUEUE_SLOTS
        for event in range(queue_counts[slot]):
            source = queue[slot, event]
            if not output_disconnected[source]:
                for edge in range(indptr[source], indptr[source + 1]):
                    target = posts[edge]
                    if active[target] and not input_disconnected[target]:
                        g[target] += weights[edge]
        queue_counts[slot] = 0
        # External tape increments run at order 0, after internal synapses.
        for event in range(tape_offsets[local_step], tape_offsets[local_step + 1]):
            neuron = tape_neurons[event]
            if active[neuron]:
                v[neuron] += EXTERNAL_INCREMENT_MV
        # Brian's resets slot, followed by end-slot selected-state recording.
        delivery_slot = (step + DELAY_STEPS) % QUEUE_SLOTS
        for event in range(fired_count):
            neuron = fired[event]
            v[neuron] = -52.0
            g[neuron] = 0.0
            queue[delivery_slot, queue_counts[delivery_slot]] = neuron
            queue_counts[delivery_slot] += 1
        for index in range(len(recorded)):
            neuron = recorded[index]
            v_trace[local_step, index] = v[neuron]
            g_trace[local_step, index] = g[neuron]
    return (np.asarray(saved_steps), np.asarray(saved_neurons),
            spike_counts, v_trace, g_trace)


@dataclass(frozen=True)
class NeuralResult:
    spike_steps: np.ndarray
    spike_neurons: np.ndarray
    spike_counts: np.ndarray
    v_mV: np.ndarray
    g_mV: np.ndarray
    record_indices: np.ndarray
    start_step: int
    end_step: int
    dt_ms: float
    elapsed_seconds: float


class SparseBrain:
    """Fixed float64 model with immutable CSR graph and independent state.

    Internal indices are integers. Public FlyWire root IDs belong to the
    separately validated data mapping, not to this numerical interface.
    The 19 queue slots reserve at most one event per source per step.
    """

    @classmethod
    def from_csr(cls, indptr, post_indices, weights_mV, input_mask, *, expected_digest,
                 limits: NeuralLimits | None = None) -> SparseBrain:
        """Attach a verified, read-only complete CSR without rebuilding it.

        Storage changes only: signed contact weights are checked, immutable
        arrays are shared, and each instance receives fresh resting state.
        """
        arrays = ((indptr, np.dtype("int64")), (post_indices, np.dtype("int32")),
                  (weights_mV, np.dtype("float64")), (input_mask, np.dtype("bool")))
        for array, dtype in arrays:
            if (not isinstance(array, np.ndarray) or array.ndim != 1 or array.dtype != dtype
                    or not array.flags.c_contiguous or array.flags.writeable):
                raise ValueError("CSR arrays must be contiguous, read-only and exactly typed")
        n = len(input_mask)
        if not 0 < n <= np.iinfo(np.int32).max or len(indptr) != n + 1:
            raise ValueError("invalid CSR neuron dimensions")
        edges = len(post_indices)
        if (len(weights_mV) != edges or indptr[0] != 0 or indptr[-1] != edges
                or np.any(indptr[1:] < indptr[:-1])):
            raise ValueError("invalid CSR connection offsets")
        selected_limits = NeuralLimits() if limits is None else limits
        if not isinstance(selected_limits, NeuralLimits):
            raise ValueError("limits must be a NeuralLimits instance")
        if 128 * n + 56 * edges > selected_limits.max_graph_working_bytes:
            raise NeuralResourceError("graph exceeds its declared memory budget")
        indptr, post_indices, weights_mV, input_mask = (
            array if isinstance(array, np.memmap) and array.mode == "r"
            else np.frombuffer(array.tobytes(), dtype=array.dtype) for array, _ in arrays)
        for start in range(0, edges, 65536):
            posts, weights = post_indices[start:start + 65536], weights_mV[start:start + 65536]
            counts = np.rint(weights / .275)
            if (np.any(posts < 0) or np.any(posts >= n) or not np.isfinite(weights).all()
                    or np.any(counts == 0) or np.any(np.abs(counts) > np.iinfo(np.int32).max)
                    or not np.array_equal(weights, counts * .275)):
                raise ValueError("invalid CSR targets or signed contact weights")
        digest = hashlib.sha256()
        digest.update(b"neuroterrarium.lif-csr.v1\0")
        digest.update(np.asarray([n], dtype="<i8").tobytes())
        for array, dtype in ((indptr, "<i8"), (post_indices, "<i4"),
                             (weights_mV, "<f8"), (input_mask, "?")):
            for start in range(0, len(array), 65536):
                digest.update(array[start:start + 65536].astype(dtype, copy=False).tobytes())
        if digest.hexdigest() != expected_digest:
            raise ValueError("complete CSR graph checksum mismatch")
        result = object.__new__(cls)
        result.n_neurons, result.n_inputs, result.limits = n, int(input_mask.sum()), selected_limits
        result.indptr, result.post_indices, result.weights_mV = indptr, post_indices, weights_mV
        result.input_mask, result.graph_digest = input_mask, expected_digest
        result.v_mV = np.full(n, -52., dtype=np.float64)
        result.g_mV = np.zeros(n, dtype=np.float64)
        result.last_spike_steps = np.full(n, NEVER_SPIKED, dtype=np.int64)
        result.step = 0
        result._queue = np.empty((QUEUE_SLOTS, n), dtype=np.int32)
        result._queue_counts = np.zeros(QUEUE_SLOTS, dtype=np.int64)
        result.output_disconnected = np.zeros(n, dtype=np.bool_)
        result.input_disconnected = np.zeros(n, dtype=np.bool_)
        result.spike_clamped = np.zeros(n, dtype=np.bool_)
        return result

    def __init__(
        self,
        n_neurons: int,
        pre: Sequence[int],
        post: Sequence[int],
        signed_counts: Sequence[int],
        *,
        input_neurons: Sequence[int],
        initial_v_mV: Sequence[float] | None = None,
        initial_g_mV: Sequence[float] | None = None,
        limits: NeuralLimits | None = None,
    ):
        if (isinstance(n_neurons, bool) or not isinstance(n_neurons, int)
                or not 0 < n_neurons <= np.iinfo(np.int32).max):
            raise ValueError("n_neurons must be a positive int32-compatible integer")
        self.limits = NeuralLimits() if limits is None else limits
        if not isinstance(self.limits, NeuralLimits):
            raise ValueError("limits must be a NeuralLimits instance")
        if not (len(pre) == len(post) == len(signed_counts)):
            raise ValueError("connection arrays must have equal lengths")
        # Conservative estimate for CSR construction/state working arrays.
        # Reject before converting potentially oversized incoming sequences.
        if 128 * n_neurons + 56 * len(pre) > self.limits.max_graph_working_bytes:
            raise NeuralResourceError("graph construction exceeds its memory budget")
        self.n_neurons = n_neurons
        sources = _indices(pre, n_neurons, "pre")
        targets = _indices(post, n_neurons, "post")
        counts = np.asarray(signed_counts, dtype=np.float64)
        if (counts.ndim != 1 or not np.all(np.isfinite(counts))
                or not np.all(counts == np.round(counts))):
            raise ValueError("signed_counts must contain finite integer counts")
        if not (len(sources) == len(targets) == len(counts)):
            raise ValueError("connection arrays must have equal lengths")
        if len(input_neurons) > n_neurons:
            raise NeuralResourceError("too many external input neuron indices")
        inputs = _indices(input_neurons, n_neurons, "input_neurons")
        if len(inputs) != len(np.unique(inputs)):
            raise ValueError("input_neurons must not contain duplicates")
        self.n_inputs = len(inputs)
        # Stable sorting preserves the input edge order within each source.
        order = np.argsort(sources, kind="stable")
        self.indptr = np.zeros(n_neurons + 1, dtype=np.int64)
        self.indptr[1:] = np.cumsum(np.bincount(sources, minlength=n_neurons))
        self.post_indices = targets[order].astype(np.int32)
        self.weights_mV = counts[order] * 0.275
        self.input_mask = np.zeros(n_neurons, dtype=np.bool_)
        self.input_mask[inputs] = True
        digest = hashlib.sha256()
        digest.update(b"neuroterrarium.lif-csr.v1\0")
        digest.update(np.asarray([n_neurons], dtype="<i8").tobytes())
        for array, dtype in ((self.indptr, "<i8"), (self.post_indices, "<i4"),
                             (self.weights_mV, "<f8"), (self.input_mask, "?")):
            digest.update(array.astype(dtype, copy=False).tobytes())
            array.flags.writeable = False
        self.graph_digest = digest.hexdigest()
        self.v_mV = self._initial(initial_v_mV, -52.0, "v")
        self.g_mV = self._initial(initial_g_mV, 0.0, "g")
        self.last_spike_steps = np.full(n_neurons, NEVER_SPIKED, dtype=np.int64)
        self.step = 0
        self._queue = np.empty((QUEUE_SLOTS, n_neurons), dtype=np.int32)
        self._queue_counts = np.zeros(QUEUE_SLOTS, dtype=np.int64)
        self.output_disconnected = np.zeros(n_neurons, dtype=np.bool_)
        self.input_disconnected = np.zeros(n_neurons, dtype=np.bool_)
        self.spike_clamped = np.zeros(n_neurons, dtype=np.bool_)

    def _initial(self, values, default, label):
        if values is not None and len(values) != self.n_neurons:
            raise ValueError(f"initial_{label}_mV needs one finite value per neuron")
        if isinstance(values, np.ndarray):
            if values.shape != (self.n_neurons,) or values.dtype.kind not in "iuf":
                raise ValueError(f"initial_{label}_mV must be a flat numeric vector")
        elif values is not None and any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating)) for value in values
        ):
            raise ValueError(f"initial_{label}_mV must be a flat numeric vector")
        array = (np.full(self.n_neurons, default, dtype=np.float64) if values is None
                 else np.asarray(values, dtype=np.float64))
        if array.shape != (self.n_neurons,) or not np.all(np.isfinite(array)):
            raise ValueError(f"initial_{label}_mV needs one finite value per neuron")
        return array.copy()

    def set_interventions(
        self, *, disconnect_output=None, disconnect_input=None, clamp_spikes=None,
    ) -> None:
        """Replace supplied masks. None leaves a mask unchanged; [] restores it.

        Input disconnection concerns network synapses, not external sensory
        injection. Spike clamp blocks threshold events; it does not erase
        membrane state or queued events. Output weights are checked on arrival.
        """
        changes = [(self.output_disconnected, disconnect_output, "disconnect_output"),
                   (self.input_disconnected, disconnect_input, "disconnect_input"),
                   (self.spike_clamped, clamp_spikes, "clamp_spikes")]
        for _, values, label in changes:
            if values is not None and len(values) > self.n_neurons:
                raise NeuralResourceError(f"{label} contains too many indices")
        validated = [(mask, _indices(values, self.n_neurons, label))
                     for mask, values, label in changes if values is not None]
        for mask, indices in validated:
            mask[:] = False
            mask[indices] = True

    def advance(
        self,
        input_tape: Sequence[Sequence[int]],
        record_indices: Sequence[int] = (),
        record_spikes: bool = False,
        *,
        interventions: Mapping[int, Mapping[str, Sequence[int]]] | None = None,
    ) -> NeuralResult:
        """Advance persistent state using one external-event row per step.

        Intervention steps are relative to this call and apply before the
        corresponding groups slot. Full spike recording is opt-in and finite;
        per-neuron window counts remain available without retaining spike times.
        Oversized requests fail before integration. Runtime recording overflow
        and interrupted/failed integration restore the complete pre-call state.
        """
        steps = len(input_tape)
        if not isinstance(record_spikes, (bool, np.bool_)):
            raise ValueError("record_spikes must be a boolean")
        if not 1 <= steps <= self.limits.max_steps:
            raise NeuralResourceError("neural window exceeds its step budget")
        if self.n_neurons * steps > self.limits.max_neuron_updates:
            raise NeuralResourceError("neural window exceeds its update budget")
        if len(record_indices) > self.n_neurons:
            raise NeuralResourceError("too many recorded neuron indices")
        if steps * len(record_indices) * 16 > self.limits.max_trace_bytes:
            raise NeuralResourceError("selected neural traces exceed their byte budget")
        if interventions is not None and len(interventions) > self.limits.max_interventions:
            raise NeuralResourceError("too many interventions in one neural window")
        recorded = _indices(record_indices, self.n_neurons, "record_indices")
        if len(recorded) != len(np.unique(recorded)):
            raise ValueError("record_indices must not contain duplicates")
        offsets = np.zeros(steps + 1, dtype=np.int64)
        events = []
        for index, row in enumerate(input_tape):
            if len(row) > self.n_inputs or len(events) + len(row) > self.limits.max_input_events:
                raise NeuralResourceError("external input tape exceeds its event budget")
            neurons = _indices(row, self.n_neurons, f"input_tape[{index}]")
            if len(neurons) != len(np.unique(neurons)):
                raise ValueError("a tape step cannot contain repeated input events")
            if not np.all(self.input_mask[neurons]):
                raise ValueError("input_tape references an unregistered input neuron")
            events.extend(neurons.tolist())
            offsets[index + 1] = len(events)
        tape_neurons = np.asarray(events, dtype=np.int64)
        scheduled = {}
        allowed = {"disconnect_output", "disconnect_input", "clamp_spikes"}
        for step, changes in (interventions or {}).items():
            if (isinstance(step, bool) or not isinstance(step, int)
                    or not 0 <= step < steps):
                raise ValueError("intervention step is outside the tape")
            if set(changes) - allowed:
                raise ValueError("unknown intervention type")
            if any(len(value) > self.n_neurons for value in changes.values()):
                raise NeuralResourceError("intervention mask contains too many indices")
            scheduled[step] = {key: _indices(value, self.n_neurons, key)
                               for key, value in changes.items()}
        start_step = self.step
        start = time.perf_counter()
        boundaries = sorted({0, steps, *scheduled})
        parts = []
        counts = np.zeros(self.n_neurons, dtype=np.int64)
        recorded_spikes = 0
        rollback = self._capture_mutable_state()
        try:
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                if left in scheduled:
                    self.set_interventions(**scheduled[left])
                values = _advance_kernel(
                    self.indptr, self.post_indices, self.weights_mV, self.input_mask,
                    self.v_mV, self.g_mV, self.last_spike_steps,
                    self._queue, self._queue_counts,
                    self.output_disconnected, self.input_disconnected, self.spike_clamped,
                    offsets[left:right + 1], tape_neurons, recorded, self.step,
                    bool(record_spikes), self.limits.max_recorded_spikes - recorded_spikes,
                )
                self.step += right - left
                counts += values[2]
                recorded_spikes += len(values[0])
                # Do not retain a full-neuron count array for each segment.
                parts.append((values[0], values[1], values[3], values[4]))
            if not np.all(np.isfinite(self.v_mV)) or not np.all(np.isfinite(self.g_mV)):
                raise FloatingPointError("non-finite neural state after integration")
            arrays = parts[0] if len(parts) == 1 else tuple(
                np.concatenate([part[index] for part in parts]) for index in range(4)
            )
            result = NeuralResult(
                spike_steps=arrays[0], spike_neurons=arrays[1], spike_counts=counts,
                v_mV=arrays[2], g_mV=arrays[3], record_indices=recorded.copy(),
                start_step=start_step, end_step=self.step, dt_ms=DT_MS,
                elapsed_seconds=time.perf_counter() - start,
            )
        except BaseException as error:
            self._restore_mutable_state(rollback)
            if isinstance(error, RuntimeError) and str(error) == "neural spike recording capacity exceeded":
                raise NeuralResourceError("spike recording budget exceeded; neural state unchanged") from error
            raise
        return result

    def _capture_mutable_state(self):
        return (
            self.step,
            {name: getattr(self, name).copy() for name in (
                "v_mV", "g_mV", "last_spike_steps", "output_disconnected",
                "input_disconnected", "spike_clamped")},
            [self._queue[slot, :self._queue_counts[slot]].copy() for slot in range(QUEUE_SLOTS)],
        )

    def _restore_mutable_state(self, state):
        self.step, arrays, queues = state
        for name, array in arrays.items():
            getattr(self, name)[:] = array
        for slot, array in enumerate(queues):
            self._queue_counts[slot] = len(array)
            self._queue[slot, :len(array)] = array

    def snapshot(self) -> dict:
        """Return structured state, including pending events and all masks.

        No executable object, pickle, or random-generator state is embedded.
        The caller owns the external stimulus stream and snapshots it separately.
        """
        return {
            "schema": STATE_SCHEMA, "graph_digest": self.graph_digest,
            "n_neurons": self.n_neurons, "dt_ms": DT_MS, "step": self.step,
            "v_mV": self.v_mV.tolist(), "g_mV": self.g_mV.tolist(),
            "last_spike_steps": self.last_spike_steps.tolist(),
            "queue": [self._queue[slot, :self._queue_counts[slot]].tolist()
                      for slot in range(QUEUE_SLOTS)],
            "disconnect_output": np.flatnonzero(self.output_disconnected).tolist(),
            "disconnect_input": np.flatnonzero(self.input_disconnected).tolist(),
            "clamp_spikes": np.flatnonzero(self.spike_clamped).tolist(),
        }

    def fork(self) -> SparseBrain:
        """Copy neural state while sharing only the immutable graph arrays."""
        clone = object.__new__(SparseBrain)
        for name in ("n_neurons", "n_inputs", "indptr", "post_indices", "weights_mV", "input_mask",
                     "graph_digest", "step", "limits"):
            setattr(clone, name, getattr(self, name))
        for name in ("v_mV", "g_mV", "last_spike_steps", "_queue", "_queue_counts",
                     "output_disconnected", "input_disconnected", "spike_clamped"):
            setattr(clone, name, getattr(self, name).copy())
        return clone

    def restore(self, state: Mapping) -> None:
        """Validate an entire snapshot before atomically replacing state."""
        expected = {"schema", "graph_digest", "n_neurons", "dt_ms", "step", "v_mV", "g_mV",
                    "last_spike_steps", "queue", "disconnect_output", "disconnect_input",
                    "clamp_spikes"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("neural snapshot fields do not match the schema")
        if (state["schema"] != STATE_SCHEMA or state["graph_digest"] != self.graph_digest
                or state["n_neurons"] != self.n_neurons or state["dt_ms"] != DT_MS):
            raise ValueError("neural snapshot graph, profile or schema mismatch")
        step = state["step"]
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < (1 << 50):
            raise ValueError("invalid snapshot step")
        v = self._initial(state["v_mV"], -52.0, "v")
        g = self._initial(state["g_mV"], 0.0, "g")
        if len(state["last_spike_steps"]) != self.n_neurons:
            raise ValueError("invalid snapshot last-spike times")
        if not isinstance(state["last_spike_steps"], np.ndarray) and any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in state["last_spike_steps"]
        ):
            raise ValueError("invalid snapshot last-spike times")
        last = np.asarray(state["last_spike_steps"])
        if (last.shape != (self.n_neurons,) or last.dtype.kind not in "iu"
                or np.any((last != NEVER_SPIKED) & ((last < 0) | (last >= step)))):
            raise ValueError("invalid snapshot last-spike times")
        if len(state["queue"]) != QUEUE_SLOTS:
            raise ValueError("invalid number of delayed-event slots")
        queues = []
        for slot, row in enumerate(state["queue"]):
            if len(row) > self.n_neurons:
                raise NeuralResourceError("queue slot contains too many source events")
            indices = _indices(row, self.n_neurons, "queue")
            if len(indices) != len(np.unique(indices)):
                raise ValueError("duplicate source event in a queue slot")
            if len(indices) > 1 and np.any(indices[1:] <= indices[:-1]):
                raise ValueError("queue source events are not in threshold order")
            arrival = step + ((slot - step) % QUEUE_SLOTS)
            # A committed snapshot follows the previous step's reset. Pending
            # arrivals can only span the next 18 integration steps.
            if len(indices) and not step <= arrival < step + DELAY_STEPS:
                raise ValueError("queue event lies outside the delay horizon")
            emission = arrival - DELAY_STEPS
            if len(indices) and (emission < 0 or np.any(last[indices] < emission)):
                raise ValueError("queue events disagree with the firing history")
            queues.append(indices)
        latest_queued = np.full(self.n_neurons, NEVER_SPIKED, dtype=np.int64)
        for arrival in range(step, step + DELAY_STEPS):
            indices = queues[arrival % QUEUE_SLOTS]
            emission = arrival - DELAY_STEPS
            minimum_interval = np.where(self.input_mask[indices], 2, REFRACTORY_STEPS)
            if np.any(emission - latest_queued[indices] < minimum_interval):
                raise ValueError("queue contains impossible consecutive firing events")
            latest_queued[indices] = emission
        recent = last >= step - DELAY_STEPS
        if np.any(latest_queued[recent] != last[recent]):
            raise ValueError("queue omits a recent firing event")
        elapsed = step - last
        must_be_reset = (elapsed == 1) | (~self.input_mask & (elapsed <= REFRACTORY_STEPS))
        if np.any(v[must_be_reset] != -52.0) or np.any(g[must_be_reset] != 0.0):
            raise ValueError("snapshot refractory state does not match its last spike")
        if any(len(state[key]) > self.n_neurons
               for key in ("disconnect_output", "disconnect_input", "clamp_spikes")):
            raise NeuralResourceError("snapshot intervention mask contains too many indices")
        masks = {key: _indices(state[key], self.n_neurons, key)
                 for key in ("disconnect_output", "disconnect_input", "clamp_spikes")}
        self.v_mV[:] = v
        self.g_mV[:] = g
        self.last_spike_steps[:] = last
        self.step = step
        for slot, indices in enumerate(queues):
            self._queue_counts[slot] = len(indices)
            self._queue[slot, :len(indices)] = indices
        self.set_interventions(**masks)
