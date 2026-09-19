"""Generate model cards and public development records from completed runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(suite_path: Path, models_path: Path, output_root: Path) -> None:
    suite = json.loads(suite_path.read_text())
    model_set = json.loads((models_path / "manifest.json").read_text())
    if suite.get("schema") != "neuroterrarium.training-suite.v1" or suite.get("status") != "complete":
        raise ValueError("training suite is incomplete")
    if model_set.get("schema") != "neuroterrarium.model-set.v1" or model_set.get("status") != "completed":
        raise ValueError("model set is incomplete")
    if suite["protocol_sha256"] != model_set["protocol_sha256"] or suite["source_sha256"] != model_set["source_sha256"]:
        raise ValueError("suite and model-set provenance disagree")
    if len(suite["models"]) != 9 or len(model_set["models"]) != 9:
        raise ValueError("nine completed runs are required")
    sources = {entry["model"]: entry for entry in suite["models"]}
    rows = []
    for entry in model_set["models"]:
        name = entry["id"]
        if Path(name).name != name or not name or entry["checkpoint"] != name:
            raise ValueError("invalid checkpoint name")
        run_path = suite_path.parent / name / "run.json"
        run = json.loads(run_path.read_text())
        trained = models_path / name / "policy.safetensors"
        checkpoint_manifest_path = models_path / name / "manifest.json"
        if digest(checkpoint_manifest_path) != entry["checkpoint_manifest_sha256"]:
            raise ValueError("checkpoint manifest hash mismatch")
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
        for filename, specification in checkpoint_manifest["files"].items():
            if Path(filename).name != filename:
                raise ValueError("invalid checkpoint filename")
            artifact = models_path / name / filename
            if artifact.stat().st_size != specification["bytes"] or digest(artifact) != specification["sha256"]:
                raise ValueError("checkpoint file integrity mismatch")
        checkpoint = json.loads((models_path / name / "state.json").read_text())
        if run["status"] != "complete" or run["protocol_status"] != "frozen" or not run["parameters_changed"]:
            raise ValueError("run is not a completed frozen training protocol")
        if run["weights_sha256"] != entry["policy_sha256"] or digest(trained) != entry["policy_sha256"]:
            raise ValueError("weight hash mismatch")
        initial_policy = run_path.parent / "checkpoints" / "initial" / "policy.safetensors"
        if digest(initial_policy) != run["initial_weights_sha256"]:
            raise ValueError("initial policy hash mismatch")
        if run["transitions"] != model_set["transitions_per_model"] or run["transitions"] != sources[name]["transitions"]:
            raise ValueError("transition budget mismatch")
        if checkpoint["source_sha256"] != model_set["source_sha256"] or checkpoint["config_sha256"] != model_set["protocol_sha256"]:
            raise ValueError("checkpoint provenance mismatch")
        if checkpoint["policy"]["seed"] != entry["seed"] or checkpoint["policy"]["architecture"] != entry["architecture"]:
            raise ValueError("checkpoint identity mismatch")
        if any(run[key] != sources[name][key] for key in ("weights_sha256", "initial_weights_sha256", "dev_before", "dev_after")):
            raise ValueError("suite and raw run records disagree")
        for phase in ("dev_before", "dev_after"):
            episodes = run[phase]["episodes"]
            if not episodes:
                raise ValueError("development episodes are missing")
            for summary, field in (("mean_food", "food"), ("mean_return", "episode_return"),
                                   ("mean_collisions", "collisions")):
                computed = math.fsum(episode[field] for episode in episodes) / len(episodes)
                if not math.isclose(computed, run[phase][summary], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("development summary differs from raw episodes")
        curve_path = run_path.parent / "training.jsonl"
        curve = [json.loads(line) for line in curve_path.read_text().splitlines()]
        if not curve or curve[-1]["transitions"] != run["transitions"]:
            raise ValueError("training curve is incomplete")
        rows.append({"id": name, "policy": checkpoint["policy"],
                     "run_sha256": digest(run_path), "training_log_sha256": digest(curve_path),
                     "checkpoint_manifest_sha256": entry["checkpoint_manifest_sha256"],
                     **{key: run[key] for key in (
                         "transitions", "vector_steps", "updates", "wall_seconds",
                         "cumulative_wall_seconds", "peak_resources", "transitions_per_second",
                         "initial_weights_sha256", "weights_sha256", "dev_before", "dev_after")},
                     "training_log_summary": {"record_count": len(curve), "first_update": curve[0],
                                              "last_update": curve[-1]}})
    if len({row["weights_sha256"] for row in rows}) != 9 or len({row["initial_weights_sha256"] for row in rows}) != 9:
        raise ValueError("independent model weight hashes are not unique")
    development_seeds = [episode["seed"] for episode in rows[0]["dev_before"]["episodes"]]
    hidden_sizes = {row["policy"]["hidden_size"] for row in rows}
    if len(hidden_sizes) != 1:
        raise ValueError("model hidden-size protocol mismatch")
    hidden_size = hidden_sizes.pop()
    if not development_seeds or any(
        [episode["seed"] for episode in row[phase]["episodes"]] != development_seeds
        for row in rows for phase in ("dev_before", "dev_after")
    ):
        raise ValueError("development cases are not matched")
    record = {"schema": "neuroterrarium.training-results.v1", "status": "complete",
              "protocol_sha256": model_set["protocol_sha256"],
              "source_sha256": model_set["source_sha256"],
              "model_set_manifest_sha256": digest(models_path / "manifest.json"),
              "suite_wall_seconds": suite["suite_elapsed_seconds"],
              "development_only": True, "models": rows}
    records_path = output_root / "experiments" / "training-v1.json"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    vector_steps = {row["vector_steps"] for row in rows}
    if len(vector_steps) != 1:
        raise ValueError("model vector-step counts disagree")
    budget = model_set["transitions_per_model"]
    vector_steps = vector_steps.pop()
    if budget % vector_steps:
        raise ValueError("invalid vector-step accounting")
    lines = ["# Model cards", "", "Nine independently initialized PPO runs completed the frozen budget of "
             f"{budget:,} environment transitions each. Each model has {vector_steps:,} vector steps "
             f"with {budget//vector_steps} environments. An environment transition counts one world's step.", "",
             "The architecture, observation, action, reward, normalization, and recovery "
             "details are in [Training](training.md). All weights stay fixed during play. "
             f"The recurrent state is {hidden_size} float values; the feedforward controller has no "
             "persistent policy memory. The hybrid includes a labeled fixed reflex branch.", "",
             "These tables are generated from the [training identities and raw development episodes]"
             f"(../experiments/training-v1.json). Development uses {len(development_seeds)} fixed scenes "
             f"(seeds {', '.join(map(str, development_seeds))}) and deterministic Gaussian means. These outcomes are "
             "development checks, not the held-out test comparison. Food, energy, and "
             "return are simulation quantities.", "",
             "| Model | Transitions | Training seconds | Transitions/s | Peak RSS (MiB) |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['transitions']:,} | {row['wall_seconds']:.2f} | "
                     f"{row['transitions_per_second']:.1f} | {row['peak_resources']['process_rss_bytes']/2**20:.1f} |")
    lines += ["", f"The sequential suite took {suite['suite_elapsed_seconds']:.2f} wall-clock "
              "seconds, including development checks and checkpoint export. Model training "
              "times exclude the before/after development evaluations. Resident memory is "
              "the measured training process, not the memory cost of the full neural model.", "",
              "| Model | Mean food before → after | Mean return before → after | Mean collisions before → after | Dev scenes with no food after |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        before, after = row["dev_before"], row["dev_after"]
        no_food = sum(episode["food"] == 0 for episode in after["episodes"])
        lines.append(f"| {row['id']} | {before['mean_food']:.4f} → {after['mean_food']:.4f} | "
                     f"{before['mean_return']:.3f} → {after['mean_return']:.3f} | "
                     f"{before['mean_collisions']:.2f} → {after['mean_collisions']:.2f} | {no_food}/{len(after['episodes'])} |")
    lines += ["", "Improvement can be concentrated in a few scenes. Zero-consumption episodes "
              "are retained. No model is selected or omitted based on these outcomes. "
              "Three training seeds per mechanism remain three independent training runs; "
              "steps within a run are not independent training replicates.", "",
              "## Provenance", "", f"Frozen protocol SHA-256: `{model_set['protocol_sha256']}`.", "",
              "| Behavior source | SHA-256 |", "|---|---|"]
    lines += [f"| {name} | `{value}` |" for name, value in model_set["source_sha256"].items()]
    lines += ["", "| Model | Initial policy file SHA-256 | Released policy file SHA-256 |", "|---|---|---|"]
    lines += [f"| {row['id']} | `{row['initial_weights_sha256']}` | `{row['weights_sha256']}` |" for row in rows]
    lines += ["", "Every checkpoint also contains optimizer state, the exact configuration, "
              "dependency versions, named initialization seed, recurrent and random states, "
              "and a file-integrity manifest. The public model-set manifest ties each "
              "identity to its checkpoint manifest and policy file. Full hashes are used "
              "for validation; display abbreviations are not integrity checks. The complete "
              "training JSONL logs and action-audit shards are distributed as training "
              "evidence with the release. Their log hashes are included in the public "
              "records; only the first and last optimizer-update summaries are duplicated "
              "there.", ""]
    compatibility_path = output_root / "configs" / "world-restore-compatibility-v1.json"
    if compatibility_path.is_file():
        compatibility = json.loads(compatibility_path.read_text())
        if compatibility["original_world_sha256"] != model_set["source_sha256"]["world.py"]:
            raise ValueError("world reader compatibility does not describe the trained source")
        lines += ["The runtime includes a bounded reader repair for long-running stimulus "
                  "snapshots. The original training source and hashes above remain unchanged. "
                  "See [source compatibility and the exact paired development comparison]"
                  "(compatibility.md).", ""]
    documentation = output_root / "docs" / "models.md"
    documentation.parent.mkdir(parents=True, exist_ok=True)
    documentation.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("."))
    arguments = parser.parse_args()
    generate(arguments.suite, arguments.models, arguments.output)
