"""Protocol checks using development seeds; no frozen test outcomes are read."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neuroterrarium.controllers import LearnedController, Policy, apply_action
from neuroterrarium.evaluation import SCENARIOS, make_scene, run_episode, summarize
from neuroterrarium.world import ACTION_DT, RADIUS, Body, Food, World


class FixedAction:
    """Test stimulus-response probe, not a product controller or baseline."""
    def __init__(self, before=0.0, after=None):
        self.before, self.after = before, before if after is None else after
        self.observations = []

    def act(self, observation):
        self.observations.append(observation.copy())
        return np.array([0, 0, self.before if len(self.observations) <= 10 else self.after, 0])


def test_matched_scene_seed_is_independent_of_controller_initialization():
    before, before_event = make_scene("C_unseen", 7001)
    Policy("hybrid", 202, 8)
    np.random.default_rng(98).normal(size=1000)
    after, after_event = make_scene("C_unseen", 7001)
    assert before.snapshot() == after.snapshot()
    assert before_event == after_event
    assert not before.replenish


def test_a_controls_match_initial_shadow_and_change_only_after_onset():
    scenes = [make_scene(case, 7003) for case in ("A_expansion", "A_static", "A_translation", "A_contraction")]
    assert all(len(world.stimuli) == 1 for world, _ in scenes)
    for world, event in scenes:
        shadow = world.stimuli[0]
        assert shadow.growth == shadow.vx == shadow.vy == 0
        assert not shadow.physical
        assert event["stimulus_initial_radius"] == shadow.radius
    assert all(world.snapshot() == scenes[0][0].snapshot() for world, _ in scenes[1:])


def test_stationary_probe_has_no_shadow_appearance_artifact_at_onset():
    probe = FixedAction()
    row, arrays, _ = run_episode("A_static", 7005, probe)
    assert row["status"] == "completed"
    assert np.any(arrays["observation"][0, 16:32] > 0)
    np.testing.assert_array_equal(arrays["observation"][:, 32:48], 0)
    assert row["reaction_seconds"] is None and row["reaction_status"] == "no_response"


def test_preexisting_drive_is_not_misclassified_as_stimulus_reaction():
    row, _, _ = run_episode("A_expansion", 7007, FixedAction(.8))
    assert row["preactivated"] is True
    assert row["baseline_escape_drive"] == pytest.approx(.8)
    assert row["reaction_status"] == "no_response" and row["reaction_seconds"] is None


def test_defined_reaction_uses_simulation_time_and_post_baseline_increase():
    row, _, _ = run_episode("A_expansion", 7009, FixedAction(.1, .6))
    assert row["reaction_status"] == "responded"
    assert row["reaction_seconds"] == 0
    assert row["preactivated"] is False
    row, _, _ = run_episode("A_expansion", 7009, FixedAction(.4, .55))
    assert row["reaction_status"] == "no_response" and row["reaction_seconds"] is None


def test_development_scenes_never_start_embedded_in_obstacles():
    for seed in range(7001, 7031):
        for case in ("C_occlusion", "C_unseen", "C_relocation", "B_danger"):
            world, _ = make_scene(case, seed)
            body = world.bodies[0]
            for obstacle in world.obstacles:
                assert math.hypot(body.x - obstacle.x, body.y - obstacle.y) >= RADIUS + obstacle.radius


def test_neural_and_policy_inapplicable_arrays_are_explicitly_missing():
    row, arrays, _ = run_episode("B_food", 7011, "rule")
    assert row["neural_trace_applicable"] is False
    assert row["policy_likelihood_applicable"] is False
    assert np.isnan(arrays["motor_groups_hz"]).all()
    assert np.isnan(arrays["raw_policy_log_probability"]).all()
    assert row["reaction_status"] == "not_applicable" and row["reaction_seconds"] is None
    assert row["recovery_status"] == "not_applicable" and row["recovery_seconds"] is None


def test_timeout_preserves_status_and_well_shaped_empty_arrays():
    row, arrays, _ = run_episode("C_relocation", 7013, "rule", timeout_seconds=0)
    assert row["status"] == "timeout"
    assert row["food"] is None and row["final_simulated_energy"] is None
    assert row["recovery_seconds"] is None and row["recovery_status"] == "timeout"
    assert arrays["observation"].shape == (0, 59)
    assert arrays["applied_action"].shape == (0, 4)
    assert arrays["body"].shape == (0, 7)


def test_learned_raw_likelihood_and_hybrid_applied_pipeline_remain_distinct():
    torch.set_num_threads(1)
    policy = Policy("hybrid", 202, 8)
    controller = LearnedController(policy, 7015)
    row, arrays, _ = run_episode("A_expansion", 7015, controller)
    direct = apply_action(torch.from_numpy(arrays["raw_policy_action"]),
                          torch.from_numpy(arrays["observation"]), hybrid=True)
    np.testing.assert_allclose(arrays["applied_action"], direct.action.numpy(), rtol=0, atol=1e-7)
    assert np.isfinite(arrays["raw_policy_log_probability"]).all()
    assert row["policy_likelihood_applicable"] is True
    assert not np.array_equal(arrays["raw_policy_action"], arrays["applied_action"])


def test_hybrid_ablation_uses_the_same_training_action_pipeline():
    policy = Policy("hybrid", 303, 8)
    row, arrays, _ = run_episode("A_expansion", 7017, LearnedController(policy, 7017), ablation="reflex_off")
    direct = apply_action(torch.from_numpy(arrays["raw_policy_action"]),
                          torch.from_numpy(arrays["observation"]), hybrid=True, reflex_enabled=False)
    np.testing.assert_allclose(arrays["applied_action"], direct.action.numpy(), rtol=0, atol=1e-7)
    assert row["reflex_interventions"] == 0


def test_sensory_ablation_changes_the_actual_policy_observation():
    controller = LearnedController(Policy("recurrent", 101, 8), 7019)
    _, arrays, _ = run_episode("A_expansion", 7019, controller, ablation="vision_off")
    np.testing.assert_array_equal(arrays["observation"][:, 16:48], 0)


def test_raw_recording_contains_initial_state_for_recomputing_metrics(tmp_path):
    row, arrays, events = run_episode("B_food", 7021, "rule")
    assert "initial_body" in arrays or "initial_world" in events
    path = tmp_path / "episode.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        assert saved["body"][-1, 5] == pytest.approx(row["food"])
        assert saved["body"][-1, 6] == row["collision_substeps"]
        assert np.sum(saved["applied_action"][:, 2], dtype=np.float64) * ACTION_DT == pytest.approx(row["escape_drive_integral"], abs=1e-7)
        assert saved["inference_seconds"].sum() == pytest.approx(row["inference_wall_seconds"])


def test_food_energy_saturation_is_not_counted_as_motor_expenditure(monkeypatch):
    import neuroterrarium.evaluation as evaluation

    world = World(7023, 1)
    world.bodies = [Body(40, 28, 0, energy=1)]
    world.foods = [Food(40, 28)]
    world.obstacles, world.stimuli = [], []
    monkeypatch.setattr(evaluation, "make_scene", lambda *_: (world, {"onset_step": 0, "case": "B_food"}))
    class FeedingProbe:
        def act(self, _):
            return np.array([0, 0, 0, 1])
    row, _, _ = run_episode("B_food", 7023, FeedingProbe())
    # World charges exactly .001 simulation-energy units/s for this stationary
    # action. Food credit lost at the energy ceiling is not movement expenditure.
    measured = row.get("motor_energy_demand", row.get("simulated_energy_cost"))
    assert measured == pytest.approx(SCENARIOS["B_food"] * ACTION_DT * .001, abs=1e-10)


def summary_rows():
    rows = []
    for seed in (7025, 7026, 7027, 7028):
        for name, increment in [("connectome", 0), ("feedforward-101", 1),
                                ("feedforward-202", 2), ("feedforward-303", 3)]:
            rows.append({"scenario": "B_food", "controller": name, "environment_seed": seed,
                         "neural_repeat": 0, "ablation": "none", "status": "completed",
                         "food": seed * .01 + increment, "collision_substeps": 2 + increment,
                         "motor_energy_demand": .1 + increment, "inference_mean_ms": 1,
                         "reaction_seconds": None, "reaction_status": "not_applicable"})
    return rows


def test_hierarchical_contrast_uses_three_training_seeds_and_paired_environment_units(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"records": summary_rows()}))
    summarize(tmp_path)
    report = json.loads((tmp_path / "summary.json").read_text())
    pairs = report["hierarchical_paired_differences"]
    food = next(row for row in pairs if row["metric"] == "food")
    assert food["training_seeds"] == 3 and food["environment_units"] == 4
    assert food["mean_difference_vs_connectome"] == pytest.approx(2)
    assert food["hierarchical_paired_ci95"][0] < 2 < food["hierarchical_paired_ci95"][1]


def test_duplicate_experiment_units_are_rejected_before_statistics(tmp_path):
    rows = summary_rows()
    rows.append(dict(rows[0]))
    (tmp_path / "results.json").write_text(json.dumps({"records": rows}))
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        summarize(tmp_path)


def test_missing_matched_unit_is_counted_without_silently_losing_a_training_seed(tmp_path):
    rows = summary_rows()
    rows[-1]["status"] = "timeout"
    rows[-1]["food"] = None
    (tmp_path / "results.json").write_text(json.dumps({"records": rows}))
    summarize(tmp_path)
    report = json.loads((tmp_path / "summary.json").read_text())
    food = next((row for row in report["hierarchical_paired_differences"] if row["metric"] == "food"), None)
    # Retain a complete-case intersection with explicit exclusion accounting,
    # or an explicitly unavailable contrast with its missing-unit reason.
    assert food is not None
    assert food["training_seeds"] == 3
    assert food.get("excluded_environment_units") == 1 or food.get("status") == "incomplete"


def test_backend_failure_is_preserved_with_partial_trace_and_null_outcomes():
    class FailingProbe(FixedAction):
        def act(self, observation):
            if len(self.observations) == 3:
                raise RuntimeError("synthetic test interruption")
            return super().act(observation)
    row, arrays, events = run_episode("A_static", 7031, FailingProbe())
    assert row["status"] == "failed" and row["failure"]["stage"] == "controller"
    assert row["completed_windows"] == 3 and arrays["observation"].shape == (3, 59)
    assert arrays["body"].shape == (3, 7) and row["food"] is None
    assert row["reaction_status"] == "failed" and row["reaction_seconds"] is None
    json.dumps({"row": row, "events": events}, allow_nan=False)


def test_unavailable_resource_measurement_is_failure_not_invented_rss(monkeypatch):
    import neuroterrarium.evaluation as evaluation
    def failure():
        raise OSError("synthetic unavailable resource counter")
    monkeypatch.setattr(evaluation.psutil, "Process", failure)
    row, arrays, _ = run_episode("B_food", 7033, "rule")
    assert row["status"] == "failed" and row["failure"]["stage"] == "resource_sampling"
    assert row["sampled_peak_process_rss_bytes"] is None
    assert arrays["sampled_process_rss_bytes"].size == 0


def test_invalid_or_inapplicable_ablation_is_not_silently_ignored():
    for ablation in ("unknown", "reflex_off", "memory_reset", "output_disconnect"):
        with pytest.raises(ValueError, match="applicable"):
            run_episode("B_food", 7035, "rule", ablation=ablation)


def test_relocation_resource_event_records_fixed_external_addition_and_removed_remainder():
    first, _, event_a = run_episode("C_relocation", 7037, "rule")
    second, _, event_b = run_episode("C_relocation", 7037, FixedAction())
    for event in (event_a, event_b):
        assert event["resource_change"]["added_food"] == 1
        assert 0 <= event["resource_change"]["removed_unconsumed_food"] <= 1
        assert event["resource_change"]["time"] == pytest.approx(1.5)
    assert event_a["relocation"] == event_b["relocation"]
    assert first["status"] == second["status"] == "completed"


def test_outcomes_recompute_from_numeric_traces_for_all_three_task_families():
    from neuroterrarium.evaluation import recompute_metrics
    for case in ("A_expansion", "B_danger", "C_relocation"):
        row, arrays, events = run_episode(case, 7039, "rule")
        computed = recompute_metrics(arrays, events)
        for name, value in computed.items():
            if value is None:
                assert row[name] is None
            else:
                assert row[name] == pytest.approx(value, abs=1e-10)
        assert row["sampled_peak_process_rss_bytes"] == arrays["sampled_process_rss_bytes"].max()
        if case == "B_danger":
            assert row["physical_energy_debit"] is None
            assert row["physical_energy_debit_status"] == "not_measured_at_body_substep"


def test_real_lif_test_graph_uses_identical_brain_pipeline_with_readout_clamp():
    from neuroterrarium.evaluation import BrainRunner
    from neuroterrarium.interface import SensoryEncoder
    from neuroterrarium.neural import SparseBrain
    from neuroterrarium.registry import Registry
    registry = Registry.load()
    roots = np.asarray([int(node["root_id"]) for group in registry.groups.values() for node in group["neurons"]], dtype=np.int64)
    groups, sides = registry.resolve(roots), registry.resolve_sides(roots)
    inputs = SensoryEncoder(groups, sides, 7041).input_neurons
    # Explicitly synthetic empty test graph; no full-data claim is made here.
    blank = SparseBrain(len(roots), [], [], [], input_neurons=inputs)
    controller = BrainRunner(blank, groups, sides, 7041)
    row, arrays, _ = run_episode("A_expansion", 7041, controller, ablation="readout_clamp")
    assert row["status"] == "completed" and row["neural_trace_applicable"]
    np.testing.assert_array_equal(arrays["applied_action"], 0)
    assert np.any(arrays["motor_groups_hz"] > 0)
    assert controller.brain.step == SCENARIOS["A_expansion"] * 200


def test_every_frozen_ablation_expands_to_its_exact_registered_mechanism_and_cases():
    from neuroterrarium.evaluation import evaluation_plan
    config = json.loads((Path(__file__).parents[1] / "configs/evaluate-v1.json").read_text())
    policies = {f"{kind}-{seed}": SimpleNamespace(architecture=kind)
                for kind in ("feedforward", "recurrent", "hybrid") for seed in (101, 202, 303)}
    plan = evaluation_plan(config, policies)
    baseline = [(name, cases) for name, ablation, cases in plan if ablation == "none"]
    assert len(baseline) == 13 and all(cases == list(SCENARIOS) for _, cases in baseline)
    for specification in config["ablations"]:
        names = (["connectome"] if specification["controller_kind"] == "connectome" else
                 [name for name, policy in policies.items() if policy.architecture == specification["controller_kind"]])
        for name in names:
            assert (name, specification["intervention"], specification["scenarios"]) in plan
    assert sum(len(cases) for _, _, cases in plan) == 176
    for kind in ("recurrent", "hybrid"):
        for seed in (101, 202, 303):
            assert (f"{kind}-{seed}", "memory_reset", ["C_occlusion", "C_relocation", "C_unseen"]) in plan
