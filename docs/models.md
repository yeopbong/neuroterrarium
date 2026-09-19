# Model cards

Nine independently initialized PPO runs completed the frozen budget of 1,000,000 environment transitions each. Each model has 250,000 vector steps with 4 environments. An environment transition counts one world's step.

The architecture, observation, action, reward, normalization, and recovery details are in [Training](training.md). All weights stay fixed during play. The recurrent state is 64 float values; the feedforward controller has no persistent policy memory. The hybrid includes a labeled fixed reflex branch.

These tables are generated from the [training identities and raw development episodes](../experiments/training-v1.json). Development uses 8 fixed scenes (seeds 7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008) and deterministic Gaussian means. These outcomes are development checks, not the held-out test comparison. Food, energy, and return are simulation quantities.

| Model | Transitions | Training seconds | Transitions/s | Peak RSS (MiB) |
|---|---:|---:|---:|---:|
| feedforward-101 | 1,000,000 | 250.93 | 3985.2 | 303.5 |
| feedforward-202 | 1,000,000 | 242.36 | 4126.2 | 263.0 |
| feedforward-303 | 1,000,000 | 248.12 | 4030.3 | 230.5 |
| recurrent-101 | 1,000,000 | 359.16 | 2784.2 | 235.1 |
| recurrent-202 | 1,000,000 | 330.56 | 3025.2 | 234.1 |
| recurrent-303 | 1,000,000 | 394.31 | 2536.0 | 234.1 |
| hybrid-101 | 1,000,000 | 350.65 | 2851.8 | 234.2 |
| hybrid-202 | 1,000,000 | 359.57 | 2781.1 | 234.3 |
| hybrid-303 | 1,000,000 | 390.48 | 2561.0 | 234.3 |

The sequential suite took 2932.92 wall-clock seconds, including development checks and checkpoint export. Model training times exclude the before/after development evaluations. Resident memory is the measured training process, not the memory cost of the full neural model.

| Model | Mean food before → after | Mean return before → after | Mean collisions before → after | Dev scenes with no food after |
|---|---:|---:|---:|---:|
| feedforward-101 | 0.0028 → 0.1750 | -0.943 → 17.417 | 49.50 → 0.00 | 6/8 |
| feedforward-202 | 0.0028 → 0.0875 | -0.938 → 8.598 | 49.25 → 0.00 | 7/8 |
| feedforward-303 | 0.0028 → 0.0000 | -0.944 → -0.031 | 49.50 → 0.00 | 8/8 |
| recurrent-101 | 0.0028 → 0.0000 | -0.936 → -0.314 | 49.12 → 0.00 | 8/8 |
| recurrent-202 | 0.0028 → 0.1555 | -0.939 → 15.316 | 49.25 → 0.00 | 6/8 |
| recurrent-303 | 0.0028 → 0.0059 | -0.941 → 0.357 | 49.38 → 0.00 | 7/8 |
| hybrid-101 | 0.0028 → 0.1208 | 0.049 → 11.686 | 0.00 → 0.00 | 6/8 |
| hybrid-202 | 0.0028 → 0.0000 | 0.049 → -0.491 | 0.00 → 0.00 | 8/8 |
| hybrid-303 | 0.0028 → 0.0000 | 0.049 → -0.984 | 0.00 → 0.12 | 8/8 |

Improvement can be concentrated in a few scenes. Zero-consumption episodes are retained. No model is selected or omitted based on these outcomes. Three training seeds per mechanism remain three independent training runs; steps within a run are not independent training replicates.

## Provenance

Frozen protocol SHA-256: `656b93d7c88d9f03a6e21c4b853a2f24a6f472fc98a8b24a316119c45db44245`.

| Behavior source | SHA-256 |
|---|---|
| training.py | `01a2e3866738c692dfc64d1d125d1696daf639df26a1d6c2cd81ffa8c907472e` |
| controllers.py | `d401493d8965dbb9fada08e84d8e3edc38e430e857450a68ef273b7d5b749923` |
| world.py | `6f84eebb1785a91eafd0922524d860581a780fcf0df6facc6317bc06556b7b31` |

| Model | Initial policy file SHA-256 | Released policy file SHA-256 |
|---|---|---|
| feedforward-101 | `0698269da036fccc9e5773bc8d9df1d8cf0fde9168d5dec15632c2f136361066` | `f2f1a61f9ebf69b9b9729adc13d44ac115e197fb3026969e1d540afb12344cc5` |
| feedforward-202 | `176ccc4486bee0e5528c385d927754d031414edf86a46de54c134db608282e34` | `14e9a82f4ab773a5d0d37897558fc4ec45af333769d723c5b7811d2411744fa7` |
| feedforward-303 | `0a17ed6b1d37f8787f47ddcaa083a5d68dd5b4b462c31ef3f38d313d4559200b` | `19aa342edf1b89cf77901aa84c27d1d7ba81a673df7341ff25b12d8dc5f6b2ca` |
| recurrent-101 | `8dc0a0f04ba72bd8ae5a5153a6f8dcf4bc9a3d866847b175809bb5f25f0c5c42` | `43f6cafbbde237386d8ff7a9fdac115bbce1210332ce13f0830bc6c3803fd3f2` |
| recurrent-202 | `e22d5e1bcc9c8c39335ddd2be2bc94cd719060544872584b890d0c2ad629faf6` | `560e3fce30e2050c9a28fc9f854252f88534101942e5f542ef5ab8eb2bb41b11` |
| recurrent-303 | `e7b37fa73f2a3a6e378b884fc8f2589447cea93be135f72539d674de6c586592` | `2fc0b496e958f5833248cc7decc92dd4b9447f36dfa79cc04971244e73411049` |
| hybrid-101 | `49430b718eb5f2a001bbc77c9345d0d60ce2164070fee00c34666d64ec688985` | `535dc022ad32269bed6a6b48308451b1a97cd7863ae7609c4d3b96e1596083f6` |
| hybrid-202 | `c97d03c370ee5d3e70d8b73decdcf4ab70d00b16777ba6659126dddb8c4e2ab2` | `3420833e2251d9f217e07e961a79af4c3eee134080caa8ded3af8f88bbbc875a` |
| hybrid-303 | `8fbde1448237ffa58b72d7b6eaf39c46b104c2f32dc809773127d56046449632` | `0005e77440636b190e7fca94a0542bc846961f2414d1af05cc4d4e3d884316a7` |

Every checkpoint also contains optimizer state, the exact configuration, dependency versions, named initialization seed, recurrent and random states, and a file-integrity manifest. The public model-set manifest ties each identity to its checkpoint manifest and policy file. Full hashes are used for validation; display abbreviations are not integrity checks. The complete training JSONL logs and action-audit shards are distributed as training evidence with the release. Their log hashes are included in the public records; only the first and last optimizer-update summaries are duplicated there.

The runtime includes a bounded reader repair for long-running stimulus snapshots. The original training source and hashes above remain unchanged. See [source compatibility and the exact paired development comparison](compatibility.md).
