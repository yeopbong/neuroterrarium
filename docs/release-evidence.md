# Local release evidence

`scripts/validate_release.py` reads an evidence manifest, verifies every referenced file, and checks the measured results. It runs no commands from that manifest and does not load a neural graph. The only accepted phase is `local`; GitHub CI, a published Release and Pages deployment are checked separately after pushing the candidate.

The [JSON Schema](../configs/release-evidence.schema.json) describes the structural envelope. The executable verifier additionally checks file bytes and category-specific measured evidence; schema validation alone cannot pass the gate.

```sh
python scripts/validate_release.py --artifact-root ./release-evidence --manifest evidence.json --output local-gate.json
```

Exit code zero means all 16 local categories passed. Missing files, changed hashes, nonzero or absent command exit codes, required skipped tests and incomplete trials return a nonzero exit code and `status: blocked`. An existing status label alone cannot satisfy a category. A successful quiet command may have an empty stdout/stderr log; that file must still exist and match its hash, with an actual zero exit code and the required native command-step evidence.

Every file reference has `path` and `sha256`; `bytes` is optional. Paths are relative to the artifact root. Absolute paths, `..`, backslashes, empty path components and symlinks are rejected. Keep the candidate files and original evidence inside this root, or copy them into a versioned release bundle. Evidence manifests contain no executable commands: `command` records the actual argument vector for inspection.

The manifest shape is:

```json
{
  "schema": "neuroterrarium.release-evidence.v1",
  "phase": "local",
  "candidate": {
    "source": {"world.py": {"path": "candidate/world.py", "sha256": "<sha256>"}},
    "training_sources": {"world.py": {"path": "training-source/world.py", "sha256": "<original-sha256>"}},
    "configs": {"data-v783.json": {"path": "candidate/data-v783.json", "sha256": "<sha256>"}},
    "model_manifest": {"path": "models/manifest.json", "sha256": "<sha256>"}
  },
  "items": {
    "core_tests": {
      "status": "not_run",
      "command": ["python", "-m", "pytest", "--junitxml=core.xml"],
      "exit_code": null,
      "log": {"path": "core/command.log", "sha256": "<sha256>"},
      "summary": {"path": "core/summary.json", "sha256": "<sha256>"},
      "junit": {"path": "core/core.xml", "sha256": "<sha256>"},
      "identity": {
        "source_sha256": {"world.py": "<sha256>"},
        "config_sha256": {"data-v783.json": "<sha256>"},
        "model_manifest_sha256": "<sha256>"
      },
      "artifacts": []
    }
  }
}
```

This abbreviated example deliberately cannot pass. Expand `candidate.source` to the complete tested source inventory, including all nine `BEHAVIOR` modules listed in the verifier, `training.py`, `evaluation.py` and `supplemental.py`. Source keys are logical filenames used by existing execution manifests; web and script files may use distinct path-like keys. Include all frozen configurations in `candidate.configs`. Each item's `identity` contains the exact dependency subset recorded for that command, rather than attributing every candidate file to every earlier experiment. The verifier's `SOURCE_SCOPE` and `CONFIG_SCOPE` require specific dependencies for each category; all additional declared hashes must also match the candidate. A missing required dependency is a failure. `training` uses the three original training-source hashes instead of the current source inventory. Original training/controller files must still match the candidate; an altered World requires the recorded restore-compatibility proof and byte-identical source outside that method.

For example, `data` requires `data.py` and `data-v783.json`; `neural` requires six neural/world modules, `scripts/neural_gate.py`, and its three interface/data configurations. `runtime` requires the nine behavior modules plus `validation.py` and all four brain configurations. The two experiment categories also require their execution module(s) and frozen protocol. TypeScript checking requires `web/src/main.ts`; list every additional TypeScript dependency that was checked. Lint and build records likewise list their actual input scope. These focused records do not claim that unrelated modules were executed. Candidate inventory remains complete and is checked independently.

Native execution identities are checked as well as the envelope. Data requires an `execution` reference containing the original data-loader command result, nested measured `summary`, and `source_files_sha256`. The extracted graph summary must equal that original measured object. Neural `component_sha256` must exactly match its recorded dependency scope and `component_source_unchanged` must be true. Runtime checks its recorded behavior and validation-source hashes; portable checks additionally match the installed module/configuration bytes and the actual three verification/build scripts; experiment locks check source and interface/data configuration hashes. Training uses checkpoint-native source and protocol hashes. `data` and `neural` require `model_manifest_sha256: null`, because those commands do not load learned policies. Other categories link the candidate model-manifest file hash; this linkage alone does not claim model execution.

When a command has actually completed, set its item to `status: completed` with its real integer `exit_code`. All item summaries are file references. Category-specific fields are listed below.

| Category | Required evidence and objective checks |
| --- | --- |
| `data` | Native graph summary; `data_files` maps completeness/connectivity/annotations to file references. Counts and hashes must match the full v783 manifest. |
| `neural` | Native neural-gate summary and adjacent NPZ/JSONL files. Both sensory positive controls, all three conditions, active exact-spike Brian comparisons, fixed voltage tolerance and closed-loop feeding/motion differences are checked. |
| `cache_equivalence` | `neuroterrarium.cache-equivalence.v1` summary with `before` and `after` summary references, plus `pairs` containing before/after file references. All 20 or more complete windows and the final state must match exactly; node and edge counts remain full-sized. |
| `training` | Published `neuroterrarium.training-results.v1`; `training_logs` and `initial_weights` map all nine model IDs to file references. Actual logs must cover every update and transition; initial/final hashes must be distinct. `compatibility` references the paired World restore proof when needed. |
| `core_tests` | `junit` file reference and summary `executed_tests`. Every actual testcase is counted; zero tests, failures, errors, skips and inconsistent declared counts are rejected. |
| `lint`, `typecheck`, `build` | Summary has `steps`, including a step with the corresponding category name. Each step records `name`, argument-vector `command`, `exit_code`, `state: completed`, adjacent `log` filename and `log_sha256`. |
| `runtime` | Native runtime-validation summary: full graph and ten identities, measured feeding, active queues, positive disconnection, clamp, stimulus removal, identical snapshot/sham and controller recomputation. |
| `evaluation` | Native ABC results and a `protocol_lock` reference. Reconstruct the whole configuration/seed plan, reject missing or duplicated trials, check completion/duration and every raw-trajectory hash. |
| `supplement` | Supplemental results and `protocol_lock`. All 630 planned single-body/shared units, 30 seeds per family, completed durations and raw hashes are required. |
| `ui` | Schema detailed below, including actual interaction traces and PNG or JPEG screenshots. |
| `stability` | Native stability summary plus `journal` reference. Require at least 3,600 measured seconds, at least 90% active time, bounded unpaused progress gaps, all prescribed interactions, sampled resources and actual ten-body journal window counts. A short preflight cannot pass. |
| `installed_wheel` | Native installation-check summary, `wheel` reference and `runtime_summary` reference. Require a clean offline-wheelhouse installation, installed-resource checks, denied runtime networking, full-data verification and real runtime validation from a path with spaces and non-ASCII characters. |
| `portable` | Schema detailed below and a `runtime_summary` reference. A dependency-only or synthetic numerical probe cannot pass. |
| `public_review` | Schema detailed below; necessary code and data notices remain separate. |

The UI summary uses `schema: neuroterrarium.ui-release-check.v1`, `status: passed`, `mode: Local full graph`, and `checks`. Each check has `name`, `status`, positive integer `operations`, an `observed_effect` description, `trace` file reference and `screenshot` PNG or JPEG reference. Required names are `environment_editing`, `actual_inspector`, `guess_and_reveal`, `time_save_load_replay`, `fork_comparison`, and `sensory_circuit_interventions`. `viewports` records width, height and `device_pixel_ratio` for at least two distinct window sizes, including a ratio of at least two. Include `keyboard_verified: true` and two or more tested `zoom_factors`. Screenshot dimensions are read from the actual PNG chunks or JPEG frame header, regardless of filename extension. Bounded structure, PNG chunk checksums, scan/header presence and complete image endings are checked; malformed or truncated files fail. These structural checks accompany the recorded real UI interaction and do not themselves establish what was rendered.

The portable summary uses `schema: neuroterrarium.portable-check.v1`, `status: passed`, `clean_extraction: true`, `offline_runtime: true`, `path_case: spaces and non-ASCII characters`, a `package` file reference and executed `steps`. Required step names are `extract`, `doctor`, `offline-runtime-validation`, `local-http`, and `relocated-launcher`. Steps use the same command/log format as installation evidence. The referenced native runtime-validation summary must match the tested candidate and model set.

The public-content summary uses `schema: neuroterrarium.public-review.v1`, `status: passed`, `unresolved_findings: 0`, `code_license: MIT`, `flywire_data_license: CC-BY-NC-4.0`, and three or more `license_files` references. Its `scans` include `secrets`, `private_paths`, `development_traces`, `licenses`, `links`, and `asset_availability`. Each records a positive `scanned_files` count, `status: passed`, and an empty `findings` list only when the actual review found no unresolved issue. The whole-candidate review also references `candidate_inventory`, with schema `neuroterrarium.candidate-content-inventory.v1`, a `files` map of repository-relative names to file references or `{sha256, bytes}` records, and `sha256` equal to the canonical JSON digest of that map. Its `candidate_inventory_sha256` must match that digest, and `candidate_source_unchanged` must be true. The verifier cross-checks every inventoried candidate source/configuration against the actual scanned bytes. Its identity scope therefore covers the whole candidate. Distinguish repository-relative link validation and hash-verified local release assets from project Release/Pages links, which remain `postpublish_pending` until the separate remote phase. Neither a prepared filename nor a future URL establishes availability. Keep internal review notes outside the code repository.

The verifier checks integrity and consistency of recorded engineering evidence. It does not establish third-party certification or biological validity, and cannot substitute for executing the recorded commands.

## Interrupted predecessors and resource revisions

A recovered core experiment keeps every predecessor directory and a hash inventory of its files. The report evidence index lists earlier stages chronologically in `recovery.ancestors`; each stage binds its `child_directory`, predecessor directory, unchanged recovery manifest, actual preparation and execution command records, and recorded prefix comparison below `--evidence-root`. Intermediate resource-limited runs retain exit 1; only the final completed run has exit 0. The gate rechecks every file, unchanged scientific lock and resource protection, inherited elapsed budget, completed-record reuse, and simulated prefix arrays directly. Inference timing and process RSS are not expected to repeat. `experiments/recovery-v1.json` separates planned new units, actually recorded new units, their completed/incomplete counts, remaining unrecorded units, and reruns of interrupted units at every stage. Only the final unique completed units enter scientific statistics; repeated execution is not an additional sample.

Supplement v1b accepts only the frozen change from 900 to 1800 cumulative wall seconds and the corresponding resource-description text. The release gate checks the full predecessor inventory and the two actual command completions: the original incomplete run with exit 1 and the new completed run with exit 0. It compares the entire source AST after reconstructing only the explicit budget-validation edit. All 630 units must be newly executed under the revised protocol; no predecessor is counted as an additional sample. `experiments/supplement-revision-v1b.json` lists every interrupted or not-started predecessor and both version hashes.

For v1b, a release item's `resource_revision` (or the report evidence index's `supplement_revision`) contains `manifest`, `parent_source`, `parent_config` and `commands` file references, a normalized `parent_directory`, and a versioned `release_asset` filename. Commands have explicit `phase` values `parent` and `execution`, their protocol SHA-256, actual arguments, completion status and exit code. Source and configuration references identify the archived original bytes. Paths are relative to the evidence root; traversal and symlinks are rejected.

Historical UI and hour records retain the hash of the service source they actually ran. A narrowly scoped browser-launch compatibility proof can accompany those two categories when the only source difference corrects the optional browser-opening URL for the bound loopback host. The validator reconstructs exactly that branch and requires the rest of the complete module AST to match, all other scoped source hashes to remain equal, and a completed current-source regression suite covering IPv4, localhost and IPv6 URLs. It does not relabel the original hour as a run of the new file or permit simulation, API, recording, intervention or network-listening changes.
