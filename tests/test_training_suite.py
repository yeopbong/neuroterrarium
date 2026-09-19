"""Small isolated protocols exercise suite recovery, never release weights."""

import importlib.util
import json
from pathlib import Path

import pytest


def load_script(name="train_models"):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location("training_suite_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def protocol(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "configs" / "train-full.json").read_text())
    config.update(total_transitions=4, num_envs=1, rollout_steps=2, sequence_length=2,
                  minibatch_sequences=1, epochs=1, episode_steps=3, dev_episode_steps=2,
                  hidden_size=8, torch_threads=1, checkpoint_updates=1, dev_seeds=[7001],
                  suite_wall_max_seconds=60)
    config["resource_limits"].update(minimum_disk_free_bytes=0, minimum_available_memory_bytes=0)
    path = tmp_path / "isolated-test-protocol.json"
    path.write_text(json.dumps(config))
    return path


def test_complete_suite_reuses_verified_runs_and_rejects_corrupt_export(tmp_path, protocol):
    script = load_script()
    output, models = tmp_path / "runs", tmp_path / "models"
    result = script.train_models(protocol, output, models)
    assert result["status"] == "complete"
    assert len(result["models"]) == 9
    assert len({model["weights_sha256"] for model in result["models"]}) == 9
    before = (output / "feedforward-101" / "training.jsonl").read_bytes()
    resumed = script.train_models(protocol, output, models, resume=True)
    assert all(row["reused_verified_run"] for row in resumed["models"])
    assert (output / "feedforward-101" / "training.jsonl").read_bytes() == before
    documentation = tmp_path / "documentation"
    load_script("model_cards").generate(output / "index.json", models, documentation)
    records = json.loads((documentation / "experiments" / "training-v1.json").read_text())
    assert len(records["models"]) == 9
    assert records["models"][0]["transitions"] == 4
    assert records["models"][0]["training_log_summary"]["record_count"] == 2
    assert "4 environment transitions" in (documentation / "docs" / "models.md").read_text()
    weights = models / "feedforward-101" / "policy.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="integrity"):
        script.train_models(protocol, output, models, resume=True)
    assert json.loads((output / "index.json").read_text())["status"] == "failed"


def test_interrupted_checkpoint_resumes_instead_of_being_treated_complete(tmp_path, protocol):
    script = load_script()
    from neuroterrarium.training import train

    output = tmp_path / "runs"
    directory = output / "feedforward-101"
    stopped = train(protocol, directory, architecture="feedforward", seed=101, stop_after_updates=1)
    assert stopped["status"] == "interrupted"
    result = script.train_models(protocol, output, tmp_path / "models", resume=True)
    assert result["status"] == "complete"
    assert not result["models"][0]["reused_verified_run"]
    assert result["models"][0]["transitions"] == 4


def test_initial_checkpoint_is_recoverable_before_first_periodic_save(tmp_path, protocol):
    script = load_script()
    from neuroterrarium.training import PPOTrainer, behavior_source_digest, read_training_config

    config = read_training_config(protocol)
    output = tmp_path / "runs"
    directory = output / "feedforward-101"
    initial = PPOTrainer(config, "feedforward", 101).save_checkpoint(directory / "checkpoints" / "initial")
    run, checkpoint = script.verified_run(directory, config, "feedforward", 101, behavior_source_digest())
    assert run is None
    assert checkpoint == initial
    result = script.train_models(protocol, output, tmp_path / "models", resume=True)
    assert result["status"] == "complete"
    assert (directory / "dev-before.json").is_file()


def test_corrupted_checkpoint_pointer_fails_before_training(tmp_path, protocol):
    script = load_script()
    from neuroterrarium.training import behavior_source_digest, read_training_config

    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "latest.json").write_text(json.dumps({"checkpoint": "../escape", "manifest_sha256": "0"*64}))
    with pytest.raises(ValueError, match="pointer"):
        script.verified_run(directory, read_training_config(protocol), "feedforward", 101, behavior_source_digest())


def test_exhausted_cumulative_budget_cannot_be_reset_by_resume(tmp_path, protocol):
    script = load_script()
    from neuroterrarium.training import behavior_source_digest, config_digest, read_training_config

    output = tmp_path / "runs"
    output.mkdir()
    (output / "suite-budget.json").write_text(json.dumps({
        "protocol_sha256": config_digest(read_training_config(protocol)),
        "source_sha256": behavior_source_digest(), "elapsed_wall_seconds": 60}))
    with pytest.raises(ValueError, match="exhausted"):
        script.train_models(protocol, output, tmp_path / "models", resume=True)


def test_existing_output_requires_resume_and_cancel_is_explicit(tmp_path, protocol):
    script = load_script()
    output = tmp_path / "runs"
    output.mkdir()
    (output / "suite.cancel").write_text("requested")
    with pytest.raises(ValueError, match="resume"):
        script.train_models(protocol, output, tmp_path / "models")
    with pytest.raises(ValueError, match="cancel"):
        script.train_models(protocol, output, tmp_path / "models", resume=True)
