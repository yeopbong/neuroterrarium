# Learned controllers

The default model set contains three independently trained feedforward policies,
three recurrent policies, and three recurrent policies with an explicit reflex
branch. Each uses the same 59 local observations, action limits, body integration,
and food-consumption rules. The weights remain fixed during play. A recurrent
hidden state changes during inference; that is memory, not online training.

The complete protocol is [train-full.json](../configs/train-full.json). Model
identities, weight hashes, measured costs, and development-set outcomes are in
the generated model cards. The short smoke protocol is an execution check and
is not an alternative release model set.

## Policy and action pipeline

All policies use a 64-unit tanh observation embedding and separate linear actor
and value heads. The feedforward policy has another 64-unit tanh layer. The
recurrent and hybrid policies use a 64-unit GRU cell instead. Four independent
Gaussian latent variables parameterize the action. Sigmoid maps forward drive,
escape drive, and feeding gate to [0, 1]; tanh maps turn to [-1, 1]. The initial
escape mean is -2 for all three mechanisms. This is an initialization choice,
not a learned biological parameter.

The hybrid then applies an engineering reflex using shared observations only:
front proximity can reduce drive and bias turning according to side proximity;
positive projected-image change can increase escape drive. Its fixed thresholds
and gain are visible in `controllers.apply_action`. This branch is active both
during training and normal inference. It does not contain another connectome,
identify predators, or read world coordinates. Switching it off is a labeled
ablation. Corrections and the number of affected action windows are recorded.

PPO likelihoods describe the **raw Gaussian sample**, before these deterministic
transforms. Every rollout records that sample, its log probability, the applied
action, and the reflex correction separately. The clipped objective uses the raw
sample under the old and updated distributions. Neither a transformed action
nor a reflex replacement is passed back into the Gaussian likelihood. The
entropy term also describes the latent distribution. Training, evaluation, and
interactive inference call the same action transform.

The implementation uses the clipped objective from
[Schulman et al., *Proximal Policy Optimization Algorithms* (2017)](https://arxiv.org/abs/1707.06347v2).
It is a project-specific implementation with integration tests, not a claim of
reproducing the paper's benchmark results.

## Observations and environments

The fixed `local-rays-v1` observation has 16 proximity samples, 16 dark-object
projection samples, 16 signed projection changes, four local chemical samples,
contact taste, simulated energy, normalized speed, and four previous actions.
These are fixed engineering scales; no fitted running normalizer is used.
Policies receive no absolute object coordinates, controller labels, scene names,
random seeds, future events, or collision countdowns.

Training samples four scene kinds with equal probability: nearby food, food
behind an obstacle, moving or changing visual stimuli, and scheduled food
relocation. The threat family includes physical and visual-only stimuli. A dark
projection is not itself a damage label. Each environment has one body and uses
the same authoritative `World` as the interactive application. Other bodies and
resource competition in the ten-body sandbox are therefore a change from this
training distribution, which is evaluated separately.

Reward is `100 * food_delta + energy_delta - 0.02 * collision_delta - 0.001 *
(drive + 3 * escape)`. Food reward comes from actual contact and feeding, with
resource depletion in the world. The reward can read simulator state; those
values are not appended to observations. Energy and food have simulation units,
not measured metabolic units. These choices encourage consumption and penalize
collisions and sustained escape, but do not establish an optimal ecological
strategy.

One environment transition is one 0.02-second action window in one environment.
Four parallel worlds therefore produce four transitions per vector step.
Body integration is 0.01 seconds. Observation at time t selects the action held
over [t, t + 0.02); computing that action does not add simulated reaction delay.
An episode ends physically at energy <= 0.05. The 250-step (5 simulated seconds)
limit is a time truncation. Time truncation bootstraps the value of its final
observation; physical termination does not. Generalized advantage estimation
stops across either reset.

## Optimization, memory, and independent runs

Each rollout contains 128 steps per environment, followed by four optimization
epochs. Recurrent minibatches contain contiguous chunks of at most 32 steps.
Each chunk begins with its recorded behavior hidden state, gradients stop at
chunk boundaries, and episode-start masks reset memory inside a chunk. The
initial state is held fixed during PPO updates to that chunk; this is truncated
backpropagation, not a gradient through the entire episode. Feedforward chunks
use the same likelihood and reward calculations without recurrent memory.

The frozen settings are Adam with learning rate 0.0003 and epsilon 0.00001,
discount 0.99, GAE parameter 0.95, PPO clipping 0.2, value coefficient 0.5,
latent entropy coefficient 0.005, and gradient-norm limit 0.5. Source hashes for
the controller, training implementation, and world are stored in every
checkpoint and model-set manifest.

Seeds 101, 202, and 303 are independently initialized for each mechanism.
Initialization, action sampling, optimizer shuffling, and each environment's
map/reset process have separate named streams. Architecture names salt the
streams, so the nine runs do not share initial weights. Train, development,
test, and stress scene partitions also use distinct named streams.

Each released model receives 1,000,000 environment transitions. Models run
sequentially in one process with at most four worlds and two Torch CPU threads.
The per-model wall budget is 900 seconds and the suite budget is 5,400 seconds.
Resource checks reserve 10 GiB of disk space and 3 GiB of available host memory
and cap training-process resident memory at 1 GiB. Limits are checked between
bounded rollouts; a violation saves an interrupted checkpoint and its reason.
An interrupted run cannot be exported as a completed model set.

Development evaluation uses the eight fixed seeds 7001 through 7008, with
deterministic Gaussian means before and after training. These are checks of
learning and failure cases, not the formal test set. A higher training reward or
better development result alone does not demonstrate generalization. Three
training seeds are three independent training runs; frames and action windows
are not additional training replicates.

## Checkpoints and recovery

Each checkpoint contains `policy.safetensors`, `optimizer.safetensors`,
`runtime.safetensors`, structured `state.json`, and a SHA-256 file manifest.
Optimizer state, rollout boundaries, hidden state, environment snapshots, and
all active sampling and shuffling streams are retained. Loading does not use
pickle or execute serialized Python. A resume must match the exact frozen
configuration and behavior-source hashes. Checkpoint directories are written
atomically and are immutable after completion.

`training.jsonl` records optimizer updates and real transition counts. Bounded
action-audit shards retain raw samples, log probabilities, applied actions, and
reflex corrections. `run.json` records completion status, development episodes,
resource measurements, and initial/final weight hashes. Existing files alone do
not establish completion: integrity, protocol, and source identifiers must
match. The model-set exporter requires nine completed, distinct identities and
nine distinct final weight hashes at the full transition budget.

Training is optional for ordinary use. To reproduce an individual run after
installing the project:

```sh
neuroterrarium train --config configs/train-full.json --architecture recurrent --seed 101 --output runs/recurrent-101
```

Use `--resume` with a verified checkpoint and the same output directory to
continue after cancellation. A `--cancel-file` path is checked between rollouts;
creating that file requests a safe stop. The cumulative wall budget is retained
across resumes. Removing a cancel request does not reset the budget.

From a source checkout, the complete sequential model set can be reproduced with:

```sh
python scripts/train_models.py --config configs/train-full.json --output runs/full --models artifacts/retrained-models
```

Add `--resume` to retain completed, verified runs and resume the current model.
The suite runner verifies checkpoint contents, source, protocol, seed, and
transition counts before reuse. It journals the cumulative suite budget and
turns Ctrl+C into a request to stop at the next rollout boundary. An existing
model-set export is accepted only if its manifests and bytes match the verified
training outputs; it is never overwritten.
