"""Learning, recurrent boundaries, shared actions and exact resume checks."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from neuroterrarium.controllers import ARCHITECTURES, LearnedController, Policy, apply_action, weights_digest
from neuroterrarium.training import PPOTrainer, TaskEnv, compute_gae, export_model_set, load_policy, train


def small_config():
    config = json.loads((Path(__file__).parents[1]/"configs"/"train-smoke.json").read_text())
    config.update(num_envs=2, hidden_size=16, rollout_steps=8, sequence_length=4,
                  minibatch_sequences=4, epochs=1, episode_steps=5,
                  total_transitions=32, checkpoint_updates=1,
                  dev_seeds=[9001, 9002], dev_episode_steps=5)
    config["resource_limits"] = {"minimum_disk_free_bytes": 0, "minimum_available_memory_bytes": 0,
                                 "maximum_process_rss_bytes": 1024**3, "maximum_wall_seconds": 60}
    return config


def test_time_limit_bootstraps_value_but_cuts_cross_episode_advantage():
    reward = torch.tensor([[1.0], [100.0]])
    values = torch.tensor([[2.0], [3.0]])
    next_values = torch.tensor([[10.0], [99.0]])
    term = torch.tensor([[False], [True]])
    trunc = torch.tensor([[True], [False]])
    advantage, returns = compute_gae(reward, values, next_values, term, trunc, 0.9, 0.95)
    torch.testing.assert_close(advantage, torch.tensor([[8.0], [97.0]]))
    torch.testing.assert_close(returns, torch.tensor([[10.0], [100.0]]))


def test_rollout_boundary_is_not_a_time_limit():
    config = small_config()
    config["episode_steps"] = 250
    trainer = PPOTrainer(config, "recurrent", 101)
    rollout = trainer.collect(3)
    assert not rollout["truncated"].any()
    assert not rollout["terminated"].any()
    expected = rollout["reward"][-1]+config["gamma"]*rollout["next_value"][-1]-rollout["value"][-1]
    torch.testing.assert_close(rollout["advantage"][-1], expected)
    assert trainer.transitions == 6
    assert all(env.world.time == pytest.approx(0.06) for env in trainer.envs)


def test_hybrid_likelihood_uses_raw_sample_and_shared_action_transform():
    policy = Policy("hybrid", 101, 16)
    obs = torch.zeros(2, 59)
    obs[:, [0, 1, 15, 32]] = 1
    hidden = policy.initial_state(2)
    generator = torch.Generator().manual_seed(2)
    sample = policy.sample(obs, hidden, torch.ones(2, dtype=torch.bool), generator)
    distribution, _, _ = policy(obs, hidden)
    formula = (-0.5*((sample["raw_action"]-distribution.mean)/distribution.stddev).square()
               -distribution.stddev.log()-0.5*math.log(2*math.pi)).sum(-1)
    torch.testing.assert_close(sample["log_probability"], formula)
    direct = apply_action(sample["raw_action"], obs, hybrid=True)
    torch.testing.assert_close(sample["applied_action"], direct.action)
    assert sample["reflex_triggered"].all()
    assert not torch.allclose(sample["applied_action"], sample["base_action"])
    assert not torch.allclose(formula, distribution.log_prob(sample["applied_action"]).sum(-1))
    disabled = apply_action(sample["raw_action"], obs, hybrid=True, reflex_enabled=False)
    torch.testing.assert_close(disabled.action, sample["base_action"])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_parameters_really_update_and_collected_actions_match_the_world(architecture):
    trainer = PPOTrainer(small_config(), architecture, 101)
    initial = weights_digest(trainer.policy)
    rollout = trainer.collect(3)
    for index, env in enumerate(trainer.envs):
        np.testing.assert_array_equal(env.world.bodies[0].action, rollout["applied_action"][-1, index].numpy())
    metrics = trainer.update(rollout)
    assert initial != weights_digest(trainer.policy)
    assert math.isfinite(metrics["value_loss"])
    assert metrics["gradient_norm"] > 0


def test_recurrent_episode_reset_blocks_hidden_history_and_gradients():
    policy = Policy("recurrent", 101, 16)
    obs = torch.randn(2, 1, 59, requires_grad=True)
    _, _, hidden = policy(obs[0], policy.initial_state(1))
    distribution, _, _ = policy(obs[1], hidden, torch.tensor([True]))
    fresh, _, _ = policy(obs[1], policy.initial_state(1))
    torch.testing.assert_close(distribution.mean, fresh.mean, rtol=0, atol=0)
    distribution.mean.sum().backward()
    assert torch.count_nonzero(obs.grad[0]) == 0
    assert torch.count_nonzero(obs.grad[1]) > 0


def test_collector_records_episode_starts_inside_sequences():
    config = small_config()
    config["episode_steps"] = 2
    trainer = PPOTrainer(config, "recurrent", 202)
    rollout = trainer.collect(4)
    assert rollout["starts"].tolist() == [[True, True], [False, False], [True, True], [False, False]]
    assert rollout["truncated"].tolist() == [[False, False], [True, True], [False, False], [True, True]]
    assert len(trainer.episode_history) == 4


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_checkpoint_restores_optimizer_hidden_rng_and_continuation_exactly(tmp_path, architecture):
    trainer = PPOTrainer(small_config(), architecture, 303)
    trainer.update(trainer.collect(8))
    checkpoint = trainer.save_checkpoint(tmp_path/"checkpoint")
    restored = PPOTrainer.resume(checkpoint)
    first, second = trainer.collect(8), restored.collect(8)
    for name in first:
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    trainer.update(first)
    restored.update(second)
    for name, tensor in trainer.policy.state_dict().items():
        torch.testing.assert_close(tensor, restored.policy.state_dict()[name], atol=0, rtol=0)
    assert not list(tmp_path.rglob("*.pkl"))
    assert weights_digest(load_policy(checkpoint)) != weights_digest(trainer.policy)


def test_checkpoint_checksum_mutation_is_detected(tmp_path):
    trainer = PPOTrainer(small_config(), "feedforward", 101)
    checkpoint = trainer.save_checkpoint(tmp_path/"checkpoint")
    path = checkpoint/"policy.safetensors"
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="integrity"):
        load_policy(checkpoint)


def test_resume_rejects_different_training_protocol(tmp_path):
    trainer = PPOTrainer(small_config(), "feedforward", 101)
    checkpoint = trainer.save_checkpoint(tmp_path/"checkpoint")
    other = small_config()
    other["gamma"] = 0.9
    with pytest.raises(ValueError, match="configuration or source"):
        PPOTrainer.resume(checkpoint, other)


def test_nine_initializations_have_nine_distinct_weight_hashes():
    hashes = {weights_digest(Policy(architecture, seed, 16)) for architecture in ARCHITECTURES for seed in (101, 202, 303)}
    assert len(hashes) == 9


def test_controller_clone_shares_weights_but_copies_recurrent_and_random_state():
    controller = LearnedController(Policy("recurrent", 101, 16), 7)
    observation = np.full(59, 0.1, dtype=np.float32)
    controller.act(observation)
    clone = controller.clone()
    assert clone.policy is controller.policy
    assert clone.hidden.data_ptr() != controller.hidden.data_ptr()
    np.testing.assert_array_equal(controller.act(observation), clone.act(observation))
    before = copy.deepcopy(clone.snapshot())
    controller.act(observation)
    assert clone.snapshot() == before
    foreign = LearnedController(Policy("recurrent", 202, 16), 7)
    with pytest.raises(ValueError, match="architecture mismatch"):
        foreign.restore(clone.snapshot())


def test_environment_termination_and_time_limit_are_distinct():
    env = TaskEnv(22, 1)
    env.reset()
    env.world.foods = []
    _, _, term, trunc, _ = env.step(np.zeros(4))
    assert not term and trunc
    env.reset()
    env.world.foods = []
    env.world.bodies[0].energy = 0.001
    _, _, term, trunc, _ = env.step(np.zeros(4))
    assert term and not trunc


def test_training_run_can_stop_and_resume_without_reclassifying_smoke(tmp_path):
    config = small_config()
    first = train(config, tmp_path/"run", architecture="hybrid", seed=101, stop_after_updates=1)
    assert first["status"] == "interrupted" and first["transitions"] == 16
    checkpoint = tmp_path/"run"/first["checkpoint"]
    result = train(config, tmp_path/"run", architecture="hybrid", seed=101, resume_from=checkpoint)
    assert result["status"] == "complete" and result["protocol_status"] == "smoke"
    assert result["transitions"] == 32 and result["parameters_changed"]
    assert len(list((tmp_path/"run"/"actions").glob("*.safetensors"))) == 2


def test_draft_full_budget_cannot_start_and_smoke_cannot_be_exported(tmp_path):
    config = small_config()
    config["status"] = "draft"
    with pytest.raises(ValueError, match="budget"):
        PPOTrainer(config, "feedforward", 101)
    trainer = PPOTrainer(small_config(), "feedforward", 101)
    checkpoint = trainer.save_checkpoint(tmp_path/"checkpoint")
    with pytest.raises(ValueError, match="frozen"):
        export_model_set([checkpoint]*9, tmp_path/"models")


@pytest.mark.parametrize("resource,reason", [("disk_free_bytes", "disk_reserve"),
                                              ("available_memory_bytes", "available_memory_reserve"),
                                              ("process_rss_bytes", "process_rss_limit")])
def test_resource_limit_saves_safe_interruption_instead_of_passing(tmp_path, monkeypatch, resource, reason):
    import neuroterrarium.training as training

    config = small_config()
    config["checkpoint_updates"] = 20
    config["resource_limits"].update(minimum_disk_free_bytes=100, minimum_available_memory_bytes=100,
                                      maximum_process_rss_bytes=100)
    calls = 0

    def measured(_):
        nonlocal calls
        calls += 1
        values = {"disk_free_bytes": 1000, "available_memory_bytes": 1000, "process_rss_bytes": 10}
        if calls > 1:
            values[resource] = 200 if resource == "process_rss_bytes" else 0
        return values

    monkeypatch.setattr(training, "measure_resources", measured)
    report = train(config, tmp_path/"run", architecture="recurrent", seed=101)
    assert report["status"] == "interrupted"
    assert report["interruption_reason"] == "resource:"+reason
    assert report["transitions"] == 16
    assert report["dev_after"]["status"] == "not_run"
    restored = PPOTrainer.resume(tmp_path/"run"/report["checkpoint"])
    assert restored.transitions == 16


def test_wall_budget_and_measurement_failures_are_explicit(tmp_path, monkeypatch):
    import neuroterrarium.training as training

    config = small_config()
    config["resource_limits"]["maximum_wall_seconds"] = 1e-12
    report = train(config, tmp_path/"timed", architecture="feedforward", seed=101)
    assert report["status"] == "interrupted"
    assert report["interruption_reason"] == "resource:wall_time_budget"
    assert report["transitions"] == 0

    def fail(_):
        raise OSError("resource measurement unavailable")

    monkeypatch.setattr(training, "measure_resources", fail)
    with pytest.raises(OSError, match="measurement unavailable"):
        train(small_config(), tmp_path/"failed", architecture="feedforward", seed=101)
    assert not (tmp_path/"failed"/"run.json").exists()
