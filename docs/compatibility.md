# Long-running stimulus snapshots

The released reader accepts moving and growing stimuli that pass the original
fixed import limits during a legitimate run. For example, an expanding stimulus
can grow beyond radius 100, and a moving stimulus can travel beyond coordinate
10,000. The original reader rejected those snapshots even though the unchanged
world integrator could produce them. That also prevented later environment
edits and branching, which validate copied world state.

Only `World.restore` admission checks changed. `World.observe`, `World.advance`,
and all other text in the world implementation are byte-identical to the
original training source. Weights, checkpoint metadata, integration steps,
sensory calculations, body actions, collisions, resource accounting, and
training rewards retain their original values.

The additional domains are finite and depend on the recorded clock and the
stimulus's own rate:

- Absolute x/y coordinate: at most `10000 + abs(vx/vy) * step * 0.02`.
- Positive radius: at most `100 + max(growth, 0) * step * 0.02`.
- An addition-error allowance is `(2 * step + 8) * ulp(upper_bound)`, accounting
  for two binary64 body substeps per action window.

The existing clock limit, finite velocity/growth limits, object-count limits,
and strict action, energy, speed, food, and non-stimulus bounds remain in force.
Static or shrinking stimuli do not gain an enlarged radius domain. This is an
import compatibility repair, not a change to how a stimulus develops.

[The compatibility manifest](../configs/world-restore-compatibility-v1.json)
records both source hashes, unchanged-function hashes, exact bounds, and the
original source archive checksum. `training-source-v1.tar.gz` in the release
contains the exact controller, trainer, world, and configuration used for the
nine training runs. Original checkpoints keep those source hashes. Exact
checkpoint continuation uses that original source; new runs can use the
compatible reader with their own recorded source identity.

[The comparison records](../experiments/world-restore-compatibility-v1.json)
cover all nine fixed trained policies on the eight original development seeds:
72 paired episodes and 18,000 action windows per implementation. Each episode
includes a midpoint world/controller snapshot restoration. Observations, raw
and applied actions, log probabilities, recurrent states, rewards, bodies,
complete world-state tapes, and outcomes match exactly. The comparison uses
deterministic policy means and is separate from the formal held-out experiments.

Boundary tests also execute 501 simulated seconds with a moving, expanding
stimulus, restore the resulting state beyond both old limits, and compare the
continued world exactly. Invalid values beyond the elapsed-time domain still
fail explicitly.

To repeat the development comparison from a prepared source installation and
the extracted original source archive:

```sh
python scripts/verify_world_compatibility.py --original-world training-source-v1/world.py --models artifacts/models --manifest configs/world-restore-compatibility-v1.json --output compatibility-results.json
```
