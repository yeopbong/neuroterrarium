"""Finite, deterministic numerical comparisons against Brian2.

The equations, parameters and reset statement follow Philip Shiu and Nico
Spiller's Drosophila_brain_model, commit
91bdd1e7dcf193f3e7ca5a8933497fcef63b7960. Their source is MIT licensed.

Copyright (c) 2023 Philip Shiu and Nico Spiller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This runner uses a recorded external input tape for numerical tests. It does
not generate the dynamically varying Bernoulli input used in an interactive
world. Inputs are increments of membrane voltage in the synapses scheduling
slot, order 0, matching the upstream PoissonInput slot. Brian2 is a numerical
reference, not a biological ground truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

DT_MS = 0.1
DELAY_MS = 1.8
REFRACTORY_MS = 2.2
EXTERNAL_INCREMENT_MV = 68.75
UPSTREAM_RESET = "v = v_rst; w = 0; g = 0 * mV"


@dataclass(frozen=True)
class ReferenceResult:
    """Selected states sampled at each step's end, after resets.

    ``v_mV`` and ``g_mV`` have shape (steps, recorded neurons). Spike step
    indices follow Brian's threshold-stage clock, rather than adding a step
    to account for the integration that preceded threshold testing.
    """

    spike_steps: np.ndarray
    spike_neurons: np.ndarray
    v_mV: np.ndarray
    g_mV: np.ndarray
    record_indices: np.ndarray
    dt_ms: float
    elapsed_seconds: float
    brian_version: str
    schedule: str


def _indices(values: Sequence[int], size: int, label: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer)) for value in values):
            raise ValueError(f"{label} must contain flat integer indices")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one dimensional")
    if array.size and array.dtype.kind not in "iu":
        raise ValueError(f"{label} must contain integer indices")
    if array.size and (np.min(array) < 0 or np.max(array) >= size):
        raise ValueError(f"{label} contains an out-of-range index")
    return array.astype(np.int64)


def run_reference(
    n_neurons: int,
    pre: Sequence[int],
    post: Sequence[int],
    signed_counts: Sequence[int],
    input_tape: Sequence[Sequence[int]],
    *,
    input_neurons: Sequence[int],
    record_indices: Sequence[int],
    disconnect_output: Sequence[int] = (),
    disconnect_input: Sequence[int] = (),
    clamp_spikes: Sequence[int] = (),
    interventions: Mapping[int, Mapping[str, Sequence[int]]] | None = None,
    initial_v_mV: Sequence[float] | None = None,
    initial_g_mV: Sequence[float] | None = None,
) -> ReferenceResult:
    """Run the complete supplied graph without pruning or resetting windows.

    Each tape row contains internal indices that receive one external event
    in that step. Duplicates in a row are rejected because the upstream
    N=1 input is a Bernoulli event per cell and step. ``input_neurons`` also
    identifies cells whose refractory duration is set to zero.

    Disconnections affect network synapses, not the separately defined
    external drive. Weights are evaluated when delayed events arrive. A
    scheduled disconnection therefore affects previously queued events too.
    Spike clamp suppresses threshold events while leaving voltage and
    synaptic integration intact; it is distinct from either disconnection.

    ``interventions`` maps step indices to replacement sets for any of
    ``disconnect_output``, ``disconnect_input`` or ``clamp_spikes``.
    Changes apply in the start slot. An empty replacement set restores the
    affected channel; an empty dictionary is a sham.
    """
    import brian2 as b2

    if isinstance(n_neurons, bool) or not isinstance(n_neurons, int) or n_neurons < 1:
        raise ValueError("n_neurons must be a positive integer")
    if len(input_tape) < 1:
        raise ValueError("input_tape must have at least one step")
    pre_array = _indices(pre, n_neurons, "pre")
    post_array = _indices(post, n_neurons, "post")
    weights = np.asarray(signed_counts, dtype=np.float64)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise ValueError("signed_counts must be a finite one-dimensional array")
    if not np.all(weights == np.round(weights)):
        raise ValueError("signed_counts must contain integer synapse counts")
    if not (len(pre_array) == len(post_array) == len(weights)):
        raise ValueError("connection arrays must have equal lengths")
    inputs = _indices(input_neurons, n_neurons, "input_neurons")
    recorded = _indices(record_indices, n_neurons, "record_indices")
    if len(np.unique(inputs)) != len(inputs) or len(np.unique(recorded)) != len(recorded):
        raise ValueError("input_neurons and record_indices must not contain duplicates")
    local_index = {int(neuron): index for index, neuron in enumerate(inputs)}
    tape_neurons: list[int] = []
    tape_steps: list[int] = []
    for step, events in enumerate(input_tape):
        indices = _indices(events, n_neurons, f"input_tape[{step}]")
        if len(indices) != len(np.unique(indices)):
            raise ValueError("a tape step cannot contain repeated input events")
        for index in indices:
            if int(index) not in local_index:
                raise ValueError("input_tape references an unregistered input neuron")
            tape_neurons.append(local_index[int(index)])
            tape_steps.append(step)

    keys = {"disconnect_output", "disconnect_input", "clamp_spikes"}
    active = {
        "disconnect_output": _indices(disconnect_output, n_neurons, "disconnect_output"),
        "disconnect_input": _indices(disconnect_input, n_neurons, "disconnect_input"),
        "clamp_spikes": _indices(clamp_spikes, n_neurons, "clamp_spikes"),
    }
    scheduled: dict[int, dict[str, np.ndarray]] = {}
    for step, changes in (interventions or {}).items():
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < len(input_tape):
            raise ValueError("intervention step is outside the tape")
        if set(changes) - keys:
            raise ValueError("unknown intervention type")
        scheduled[step] = {
            key: _indices(values, n_neurons, key) for key, values in changes.items()
        }

    namespace = {
        "v_0": -52 * b2.mV,
        "v_rst": -52 * b2.mV,
        "v_th": -45 * b2.mV,
        "t_mbr": 20 * b2.ms,
        "tau": 5 * b2.ms,
    }
    neurons = b2.NeuronGroup(
        n_neurons,
        """
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau : volt (unless refractory)
        rfc : second
        spike_clamped : boolean
        """,
        method="linear",
        threshold="v > v_th and not spike_clamped",
        reset=UPSTREAM_RESET,
        refractory="rfc",
        dt=DT_MS * b2.ms,
        codeobj_class=b2.NumpyCodeObject,
        namespace=namespace,
        name="reference_neurons*",
    )
    for name, supplied, default in (
        ("v", initial_v_mV, -52.0),
        ("g", initial_g_mV, 0.0),
    ):
        values = np.full(n_neurons, default) if supplied is None else np.asarray(supplied)
        if values.shape != (n_neurons,) or not np.all(np.isfinite(values)):
            raise ValueError(f"initial_{name}_mV must have one finite value per neuron")
        setattr(neurons, name, values * b2.mV)
    neurons.rfc = REFRACTORY_MS * b2.ms
    neurons.rfc[inputs] = 0 * b2.ms
    synapses = b2.Synapses(
        neurons,
        neurons,
        "w : volt",
        on_pre="g += w",
        delay=DELAY_MS * b2.ms,
        dt=DT_MS * b2.ms,
        codeobj_class=b2.NumpyCodeObject,
        name="reference_synapses*",
    )
    if len(weights):
        synapses.connect(i=pre_array, j=post_array)
    else:
        synapses.active = False

    def apply_masks() -> None:
        if len(weights):
            masked = np.isin(pre_array, active["disconnect_output"]) | np.isin(
                post_array, active["disconnect_input"]
            )
            synapses.w = np.where(masked, 0.0, weights) * 0.275 * b2.mV
        neurons.spike_clamped = False
        neurons.spike_clamped[active["clamp_spikes"]] = True

    apply_masks()
    objects = [neurons, synapses]
    if len(inputs):
        source = b2.SpikeGeneratorGroup(
            len(inputs),
            np.asarray(tape_neurons, dtype=np.int64),
            np.asarray(tape_steps) * DT_MS * b2.ms,
            dt=DT_MS * b2.ms,
            codeobj_class=b2.NumpyCodeObject,
            name="reference_tape*",
        )
        external = b2.Synapses(
            source,
            neurons,
            on_pre=f"v_post += {EXTERNAL_INCREMENT_MV} * mV",
            dt=DT_MS * b2.ms,
            codeobj_class=b2.NumpyCodeObject,
            name="reference_external*",
        )
        external.connect(i=np.arange(len(inputs)), j=inputs)
        external.pre.order = 0
        objects += [source, external]
    if scheduled:
        def update_interventions(t):
            step = int(np.rint(float(t / b2.ms) / DT_MS))
            if step in scheduled:
                active.update(scheduled[step])
                apply_masks()

        operation = b2.NetworkOperation(
            update_interventions,
            dt=DT_MS * b2.ms,
            when="start",
            order=-1,
            name="reference_interventions*",
        )
        objects.append(operation)
    spikes = b2.SpikeMonitor(neurons, name="reference_spikes*")
    states = b2.StateMonitor(
        neurons, ("v", "g"), record=recorded, when="end", name="reference_states*"
    )
    network = b2.Network(*objects, spikes, states)
    previous_target = b2.prefs.codegen.target
    try:
        b2.prefs.codegen.target = "numpy"
        start = perf_counter()
        network.run(len(input_tape) * DT_MS * b2.ms)
        elapsed = perf_counter() - start
    finally:
        b2.prefs.codegen.target = previous_target
    return ReferenceResult(
        spike_steps=np.rint(np.asarray(spikes.t / b2.ms) / DT_MS).astype(np.int64),
        spike_neurons=np.asarray(spikes.i, dtype=np.int64).copy(),
        v_mV=np.asarray(states.v / b2.mV).T.copy(),
        g_mV=np.asarray(states.g / b2.mV).T.copy(),
        record_indices=recorded.copy(),
        dt_ms=DT_MS,
        elapsed_seconds=elapsed,
        brian_version=b2.__version__,
        schedule=str(network.scheduling_summary()),
    )
