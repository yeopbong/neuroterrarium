# Short stress and shared-population experiments

The [frozen supplement protocol](../configs/evaluate-supplement-v1b.json) defines two small experiments alongside the core single-body evaluation. It uses the same full neural graph, fixed nine learned policies, observation schema, motor pipeline and synchronous body integration. Results have their own source, model, data and interface lock.

After installing the project and preparing its data, run from a source checkout:

```sh
python scripts/evaluate_supplement.py --config configs/evaluate-supplement-v1b.json --data ./data --models artifacts/models --output ./experiments/supplement-v1b
```

The data argument is the cache directory reported by `neuroterrarium data fetch`. This command runs the complete graph; it is not a replay. It consumes at most a 1,800-second cumulative execution budget, checked between episodes and during action-window timeouts, with small recording and teardown overhead. Cancellation preserves completed and partial records. Running again checks the same source/protocol and resumes only units that have no record. Failed trials remain failures.

The [original v1 execution](../configs/evaluate-supplement-v1.json) exhausted its 900-second wall budget after 600 stress episodes and 20 shared worlds; the next shared world was interrupted and nine were not started. Its incomplete records remain available. The v1b resource revision changes only the cumulative allowance to 1,800 seconds and its corresponding description. It retains the same scientific settings and resource protections. All 630 units were executed again in a separate directory and completed; this rerun does not add independent scientific samples. A read-only comparison found the same values, shapes and dtypes in all 11 scientific arrays for the original 620 completed units and the interrupted unit's 43-window common prefix. Execution-cost arrays were excluded and paired not-applicable NaNs were treated as equal.

The loader verifies the original data and frozen complete CSR storage profile. Unpaired trajectory/metadata files and unfinished temporary files stop recovery instead of being overwritten. Before an operation starts, the budget records a reservation; after an abrupt process exit, a resumed invocation conservatively charges that reservation. A run becomes completed only after its raw trajectories and summary have been checked. Running the same completed version again returns `verified` through a read-only check and leaves its files and historical execution budget unchanged.

The stress experiment pairs clean local vision with a fixed amount of noise added to actual visual occupancy and projection-change observations. Thirty new environment seeds are used for each controller and condition, giving 600 independent single-body episodes. Each lasts 1.5 simulation seconds in the same novel arrangement of food, obstacles and two visual shadows. Both conditions consume identical-size draws from a separate noise stream. Environment events and neural-input random draws remain paired. The three training seeds for each learned architecture remain three experimental units; bootstrap intervals resample them and the matched environments separately.

The shared experiment runs 30 worlds with all ten bodies for two simulation seconds. Three randomized cyclic allocation blocks make every controller occupy each starting slot exactly three times. Bodies compete for a finite amount of food, with the same collision and consumption rules. The complete assignment table is recorded. World totals and per-controller descriptive measurements have separate labels: the ten interacting bodies are correlated, and one connectome body versus nine learned bodies is not an algorithm comparison.

Both experiments are short checks of the specified conditions. They do not measure long-term memory, general robustness or ecological success. Shadows in this supplement have no physical damage consequence. Food, collisions, requested motor energy, stored energy and inference time stay separate.

The output directory contains `protocol-lock.json`, `budget.json`, `graph-summary.json`, numeric trajectories and per-trial JSON under `raw/`, plus `results.json` and `summary.json`. Undefined outcomes use JSON null; non-applicable numeric neural or likelihood channels use NaN. Incomplete trials cannot make the overall result completed. Summary generation verifies each trajectory checksum, recomputes the metrics and checks food conservation and completed duration:

```sh
python scripts/evaluate_supplement.py --output ./experiments/supplement-v1b --summarize-only
```

The protocol's existence does not establish that an experiment completed. Use the recorded status and raw evidence of the specific published experiment version.
