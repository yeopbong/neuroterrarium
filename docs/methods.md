# What runs inside the terrarium

NeuroTerrarium compares a connectome-constrained neural model with independently trained artificial controllers through the same local observation and body interfaces. It is a two-dimensional behavior sandbox. It does not reconstruct a complete animal, retina, six-legged body, free flight, metabolism or consciousness.

The authoritative Python path is:

`World → SharedObservation → SensoryEncoder → NeuralState → MotorReadout → ActionLimits → Body`

Training, evaluation and local play use the same world implementation. The learning controllers replace the encoder, neural state and motor readout with their registered policy and action pipeline. All ten bodies have the same dimensions, collision rules, inertia, resource-contact requirement, energy accounting and action limits. Controller assignment is shuffled independently of body placement.

## Four clocks

| Clock | Period | Meaning |
|---|---:|---|
| Neural integration | 0.1 ms | Fixed reference-model numerical step |
| Body integration | 10 ms | Two synchronous body substeps per action |
| Observation/action window | 20 ms | Observe at t; compute the action held during the declared interval [t, t+20 ms) |
| Rendering | Browser animation frames | Draw the latest completed world state; no simulation effect |

All controllers observe before any body advances. Runtime computation duration does not add a simulated reaction delay. Slow neural computation slows the complete world together. The display reports simulation time, elapsed wall time and their ratio; the session's elapsed wall time includes pauses. Rendering smoothness is not evidence of real-time neural execution. Pausing leaves full neural and recurrent state intact. The application’s display-scale control changes only layout and canvas rendering; pointer positions are mapped through the displayed canvas bounds. It does not change the world clock or random streams.

## Measured structure and assumed dynamics

The full v783 directed graph constrains a leaky integrate-and-fire model based on [Shiu et al.](https://doi.org/10.1038/s41586-024-07763-9). It is a computational model, and Brian2 is its numerical reference rather than biological ground truth. The NumPy Brian2 backend and the single sparse Numba backend receive identical prespecified input spike tapes in numerical checks.

The resting/reset voltage is −52 mV, the strict firing threshold is >−45 mV, membrane and synaptic time constants are 20 and 5 ms, synaptic delay is 1.8 ms and refractory duration is 2.2 ms. Pair weights are the signed contact counts multiplied by 0.275 mV. Exact linear integration, frozen refractory state, incoming-write suppression, threshold timing, reset and the delay queue follow the inspected upstream event schedule. External activation adds 68.75 mV using independent per-step Bernoulli events, matching the original one-source PoissonInput discretization. Registered external-injection cells have zero refractory period, including when their current rate is zero.

The sparse backend keeps float64 state and persistent delay queues across every action window. It neither trains these weights nor prunes the graph. Validation checks complete spike identities and steps, spike counts, selected membrane/synaptic traces and readout consequences. Selected trace tolerance is 1e−9 mV; exact spike timing is a separate requirement. A silent case alone cannot establish an active-path match. Tests cover inhibitory and excitatory events, delays, threshold boundaries, refractory release and interventions.

Data preparation verifies the complete graph and writes its full sparse cache in a separate process. Local sessions map that cache read-only, together with immutable root-ID and provenance metadata. Branches share this unchanged connectivity and keep separate neural states; this storage optimization removes no nodes, connections or feedback.

The original README's broad description of silencing differs from its `silence()` implementation and Methods: the implemented operation cuts output transmission. This project exposes the operations separately:

| Operation | Changes | Already queued events |
|---|---|---|
| Disconnect group outputs | Transmission from registered presynaptic cells | Output masks are checked on delivery, so these arrivals are suppressed |
| Disconnect group inputs | Network synapses into registered postsynaptic cells; external sensory injection is separate | Input masks are checked on delivery |
| Suppress future spikes | Threshold emission of registered cells | Existing queued events remain |
| Clamp motor readouts | Every newly returned motor action is zero | Neural dynamics and events continue; body inertia can decay |
| Sensory channel off | The actual shared observation supplied to every controller | Does not retroactively remove neural state or synaptic events |

Restore clears masks and sensory noise settings. Readout filtering continues during a readout clamp, so release can expose retained filtered activity. The intervention log records the precise operation and its event semantics.

## Engineering sensory and motor interfaces

The 59-dimensional observation contains 16 directional proximity samples, 16 dark angular-occupancy samples, 16 signed temporal changes of that projection, four local chemical samples, contact taste, simulated energy, normalized speed and the previous four applied actions. Occluding obstacles block relevant sensory paths. No controller receives absolute target coordinates, future events, target identities, hidden seeds or another body's full state.

The present neural encoder deliberately uses only a subset of this equally available observation. Chemical and proximity samples do not currently inject neural input. Food contact activates the registered sugar-input group. Positive local projection change injects equal-rate input into bilateral LPLC2 and LC4 populations. It bypasses retinal and optic-lobe computation and is not a reconstruction of biological looming selectivity: translation can also activate it. The [looming research](https://pmc.ncbi.nlm.nih.gov/articles/PMC7457385/) supports investigating the pathway, not claiming the encoder reproduces its complete computation.

An explicitly external, world-independent 15 Hz DNa02 input provides baseline movement. Its named random stream has no access to food or threat truth. This is an engineering assumption, not a measured endogenous firing rate. Each neural input group consumes independent random draws even when its channel is disabled. Additional channel noise uses independent per-body streams.

The readout reads registered neural spike counts and an 80 ms low-pass filter only. It cannot read the world or observation. Bilateral DNa01/DNa02 activity supplies engineering forward/yaw commands; GF activity supplies current-heading surge; MN9 activity gates feeding. Gains and all equations are registered in [brain-interface.json](../configs/brain-interface.json). GF activity does not supply a verified escape direction. A surge toward a stimulus is therefore possible. Physical feeding still requires the common body to touch available food.

Interface parameters are fixed engineering choices, not fitted biological response curves. The early positive controls establish executable input→neural-output→body consequences. They do not establish successful ecological avoidance or complete paper replication.

## Branches, randomness and replay

A session snapshot contains body/resource state, pending external events, projection history, independently named random streams, learned recurrent states, full neural voltage/synaptic/refractory state, queued synaptic events, readout filters, last actions and schema/data/model/source hashes. Restoration validates the complete structure before replacing live state. JSON carries no pickle or executable Python.

An unchanged fork copies this complete state and shares only immutable graph structure and policy weights. A sham changes the event log without changing dynamics. Switching the selected connectome controller to a learned controller uses a symmetric reset of that individual's controller history on both sides; incompatible hidden states are never copied. Other individuals retain their state. Shared environment edits apply to both branches by default; the right-branch intervention selector allows a controlled difference.

Longer branches develop different worlds and hence different observations for other bodies. They preserve matched external interventions, not impossible replayed body actions. This measures a total downstream effect; it does not isolate a permanent single direct effect.

The frame buffer retains a bounded recent tail (at most 1,500 frames or 20 MiB); it does not promise a complete session movie. The executable journal separately preserves every committed action window on disk.

The frame player is labeled **Replay**. It shows previously recorded bodies and, where present, each individual's recorded inspector values. It does not recompute controllers or accept environment interventions. The executable journal is separate: it starts from a complete snapshot, records ordered operations and every actual observation/raw decision/applied action, and can recompute those windows for exact comparison under the declared numerical backend.

## Learning and experimental limits

The default learned group has three feedforward, three recurrent and three recurrent-plus-reflex PPO policies, with independent initialization/training seeds. The hybrid branch is explicitly engineered and uses the same observation during training and evaluation. Raw Gaussian samples and their likelihoods remain distinct from the bounded and possibly reflex-modified actions. See [training methods](training.md) and the generated model cards.

Formal comparisons use independent single-body worlds with matched environment seeds, separate from the shared ten-body playground. The frozen protocol specifies stimulus controls, foraging/danger cases, local-observation changes and ablations. Simulated response time, wall-clock cost, food, collisions, motor energy demand and final simulated energy are separate metrics. Failure, timeout, no response and not-applicable values have explicit statuses. With only three training seeds per architecture, broad architecture rankings or universal generalization claims are unwarranted.

The stimulus response measure is an increase in escape drive after the declared stimulus-motion onset, relative to the preceding five action windows. Pre-existing high escape drive is reported separately. This detects a motor response; its direction and physical consequence must be inspected separately. Responses are resolved in 20 ms action windows: a recorded latency of zero means the first action window after onset, not a measured zero-millisecond biological reaction. The unseen-layout case tests one held-out arrangement of obstacles and food, not unrestricted generalization.

Motor energy demand is calculated from applied actions before the simulated energy floor. It is not a measured metabolic quantity or the body's net energy change. Physical-threat energy debit is currently unmeasured in physical-threat trials and explicitly null; food intake and final energy remain separately available. Neural-input repetition is distinct from environment repetition; the initial protocol uses one neural-input repetition per environment.

The stability script counts action windows only after all ten controllers have advanced and their decisions have been written to the executable journal. Its cumulative simulated progress includes windows later revisited after snapshot restoration; final world time can therefore differ. A passing hour requires a full hour of measured operation, monitoring coverage, the declared intervention set, matching journal counts and no prolonged unpaused stall. Memory reports are sampled RSS and a post-warmup trend, not a continuous peak measurement.
