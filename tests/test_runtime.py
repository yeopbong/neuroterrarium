"""Executable session tests. Synthetic edges occur only in this test harness."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neuroterrarium.controllers import ARCHITECTURES, Policy
from neuroterrarium.data import GraphData
from neuroterrarium.registry import Registry
from neuroterrarium.runtime import (GraphIdentity, MAX_EVENTS, NEURAL_STEPS, Session, _wire_digest,
                                    load_model_set)
from neuroterrarium.world import Food, Stimulus


@pytest.fixture(scope="module")
def components():
    torch.set_num_threads(1)
    registry = Registry.load()
    roots = np.array([int(node["root_id"]) for group in registry.groups.values()
                      for node in group["neurons"]], dtype=np.int64)
    groups, sides = registry.resolve(roots), registry.resolve_sides(roots)
    pre, post, counts = [], [], []
    for sensory, motor, weight in [("sugar", "mn9", 40), ("lplc2", "gf", 10), ("lc4", "gf", 10)]:
        for index in groups[sensory]:
            for target in groups[motor]:
                pre.append(index)
                post.append(target)
                counts.append(weight)
    for side in ("left", "right"):
        pre.append(sides[f"dna02_{side}"][0])
        post.append(sides[f"dna01_{side}"][0])
        counts.append(40)
    graph = GraphData(roots, np.array(pre, dtype=np.int32), np.array(post, dtype=np.int32),
                      np.array(counts, dtype=np.int32), {"profile": "synthetic-test-only"})
    policies = {f"{architecture}-{seed}": Policy(architecture, seed, 8)
                for architecture in ARCHITECTURES for seed in (101, 202, 303)}
    return graph, policies, registry


@pytest.fixture
def session(components):
    graph, policies, registry = components
    return Session.from_components(graph, policies, registry=registry, seed=103)


def dynamics(session):
    payload = session._payload()
    for name in ("wall_time", "compute_seconds", "events", "fork", "paused"):
        payload.pop(name)
    return payload


def rehash(envelope):
    envelope["sha256"] = _wire_digest(envelope["state"])
    return envelope


def test_ten_controllers_share_body_clock_and_have_distinct_states(session):
    assert len(session.controllers) == 10
    assert session.allocation.count("connectome") == 1
    assert len(set(session.allocation)) == 10
    assert sorted(c.policy.architecture for c in session.controllers if c) == sorted(list(ARCHITECTURES) * 3)
    initial = [(b.x, b.y) for b in session.world.bodies]
    for _ in range(5):
        session.step()
    assert session.world.step == 5
    assert session.brain.step == 5 * NEURAL_STEPS
    assert any((b.x, b.y) != position for b, position in zip(session.world.bodies, initial, strict=True))
    assert all(c.last and not c.episode_start for c in session.controllers if c)
    assert session.mode == "Test components"
    assert len(session.records) == 5


def test_session_retains_complete_csr_and_readonly_identity_without_duplicate_edges(session, components):
    graph = components[0]
    assert isinstance(session.graph, GraphIdentity)
    assert not isinstance(session.graph, GraphData)
    assert all(not hasattr(session.graph, name) for name in ("pre", "post", "signed_counts"))
    np.testing.assert_array_equal(session.graph.root_ids, graph.root_ids)
    assert session.graph.connection_rows == len(graph.pre) == len(session.brain.post_indices)
    assert session.graph.connectivity_sha256 == session.brain.graph_digest
    # Independently reconstruct the source's stable ordering: no connection or
    # signed weight may disappear when the duplicate coordinate storage goes.
    order = np.argsort(graph.pre, kind="stable")
    np.testing.assert_array_equal(session.brain.post_indices, graph.post[order])
    np.testing.assert_array_equal(session.brain.weights_mV, graph.signed_counts[order] * .275)
    with pytest.raises(ValueError):
        session.graph.root_ids.flags.writeable = True
    with pytest.raises(ValueError):
        session.graph.root_ids[0] = 1
    summary = session.graph.summary
    summary["profile"] = "mutated"
    assert session.graph.summary == graph.summary
    clone = session._clone()
    assert clone.graph is session.graph
    assert clone.brain.post_indices is session.brain.post_indices
    assert clone.brain.weights_mV is session.brain.weights_mV
    assert clone.brain.v_mV is not session.brain.v_mV


def test_graph_identity_rejects_incomplete_csr(session, components):
    from types import SimpleNamespace
    missing = SimpleNamespace(n_neurons=len(components[0].root_ids), post_indices=[])
    with pytest.raises(ValueError, match="complete graph storage"):
        GraphIdentity.from_graph(components[0], missing)


def test_session_does_not_keep_source_coordinate_allocations_alive(components):
    import gc
    import weakref
    source, policies, registry = components
    graph = GraphData(source.root_ids.copy(), source.pre.copy(), source.post.copy(),
                      source.signed_counts.copy(), dict(source.summary))
    references = [weakref.ref(array) for array in (graph.pre, graph.post, graph.signed_counts)]
    session = Session.from_components(graph, policies, registry=registry)
    del graph
    gc.collect()
    assert all(reference() is None for reference in references)
    assert session.brain.post_indices.size == source.post.size
    session.step()
    assert session.brain.step == NEURAL_STEPS


def test_controller_assignment_is_balanced_and_seeded(session, components):
    second = Session.from_components(components[0], components[1], registry=components[2], seed=103)
    assert second.allocation == session.allocation
    third = Session.from_components(components[0], components[1], registry=components[2], seed=104)
    assert third.allocation != session.allocation
    assert sorted(third.allocation) == sorted(session.allocation)


def test_panel_reads_never_advance_rng_or_change_dynamics(session):
    second = session._clone()
    for _ in range(3):
        for selected in range(10):
            panel = session.state(selected, reveal=False)["selected"]
            assert panel["kind"] is None
            assert "neural_rates_hz" not in panel and "hidden" not in panel
            session.state(selected, reveal=True)
        session.step()
        second.step()
    assert dynamics(session) == dynamics(second)


def test_inspector_displays_the_observation_used_for_its_actual_action(session):
    session.execute({"type": "noise", "value": 0.1})
    session.step()
    for index in range(10):
        panel = session.state(index, True)["selected"]
        assert panel["observation"] == session.records[-1]["observations"][index]
        assert panel["action"] == session.records[-1]["decisions"][index]["applied_action"]
        assert panel["body"] == session.state(index, False)["world"]["bodies"][index]
        assert panel["observation_time"] == 0


def test_reflex_panel_tracks_selected_controller_branch_and_restoration(session):
    indices = [i for i, name in enumerate(session.allocation) if name.startswith("hybrid-")]
    first, second = indices[:2]
    assert "reflex_enabled" not in session.state(first, False)["selected"]
    assert session.state(first, True)["selected"]["reflex_enabled"]
    saved = session.snapshot()
    session.execute({"type": "hybrid_reflex", "selected": first, "value": False})
    assert not session.state(first, True)["selected"]["reflex_enabled"]
    assert session.state(second, True)["selected"]["reflex_enabled"]
    session.execute({"type": "fork", "replacement": "same"})
    session.execute({"type": "hybrid_reflex", "selected": second, "branch": "right", "value": False})
    assert session.state(second, True)["selected"]["reflex_enabled"]
    assert not session.state(second, True)["fork"]["selected"]["reflex_enabled"]
    session.restore(saved)
    assert session.state(first, True)["selected"]["reflex_enabled"]


def test_shared_channels_disable_actual_input_for_every_controller(session):
    session.execute({"type": "noise", "value": 0.2})
    for channel in ("taste", "vision", "chemical", "proximity"):
        session.execute({"type": "channels", "channel": channel, "enabled": False})
    session.step()
    np.testing.assert_array_equal(session._last_observations[:, :53], np.zeros((10, 53)))
    assert session.encoder.last_rates_hz == {"sugar": 0, "lplc2": 0, "lc4": 0, "dna02": 15}
    session.execute({"type": "restore_interventions"})
    assert all(session.channels.values()) and session.noise_std == 0


def test_real_neural_state_drives_action_then_readout_clamp_stops_new_decisions(session):
    index = session.allocation.index("connectome")
    body = session.world.bodies[index]
    session.world.foods = [Food(body.x, body.y)]
    for _ in range(5):
        session.step()
    assert session._last_rates["mn9"] > 0
    assert session._last_actions[index, 3] > 0
    assert session.world.bodies[index].food > 0
    session.execute({"type": "readout_clamp", "value": True})
    for _ in range(3):
        session.step()
        np.testing.assert_array_equal(session._last_actions[index], np.zeros(4))
    assert session.brain.step == session.world.step * NEURAL_STEPS


def test_stimulus_removal_changes_input_after_finite_projection_history(session):
    session.encoder.tonic_hz = 0
    index = session.allocation.index("connectome")
    body = session.world.bodies[index]
    session.world.stimuli = [Stimulus(body.x + 4, body.y, 1, growth=3)]
    for _ in range(3):
        session.step()
    assert any(record["decisions"][index]["neural_input_counts"]["lplc2"] for record in session.records)
    session.execute({"type": "stimulus_remove", "index": 0})
    session.step()
    assert session.encoder.last_rates_hz["lplc2"] == 0
    assert session.encoder.last_rates_hz["lc4"] == 0
    assert session._last_inputs["lplc2"] == session._last_inputs["lc4"] == 0


def test_fork_sham_same_nonzero_execution_and_shared_readonly_graph(session):
    for _ in range(3):
        session.step()
    assert any(b.speed > 0 for b in session.world.bodies)
    session.execute({"type": "fork", "replacement": "same"})
    branch = session.fork_session
    assert branch.brain.indptr is session.brain.indptr
    assert branch.brain.v_mV is not session.brain.v_mV
    for left, right in zip(session.controllers, branch.controllers, strict=True):
        if left:
            assert left.policy is right.policy and left.hidden is not right.hidden
    session.execute({"type": "sham", "branch": "right"})
    for _ in range(4):
        session.step()
    assert dynamics(session) == dynamics(branch)


def test_branch_shared_external_stimuli_and_independent_intervention(session):
    session.execute({"type": "fork"})
    session.execute({"type": "food_add", "x": 30, "y": 10})
    assert session.world.foods == session.fork_session.world.foods
    session.execute({"type": "neural_disconnect", "group": "sugar", "branch": "right"})
    assert session.fork_session.brain.output_disconnected[session.groups["sugar"]].all()
    assert not session.brain.output_disconnected.any()
    session.execute({"type": "channels", "channel": "vision", "enabled": False, "branch": "right"})
    assert session.channels["vision"] and not session.fork_session.channels["vision"]
    session.step()
    assert session.world.time == session.fork_session.world.time


def test_heterogeneous_fork_resets_both_histories_without_copying_hidden(session):
    session.step()
    index = session.allocation.index("connectome")
    session.execute({"type": "fork", "selected": index, "replacement": "recurrent-101"})
    branch = session.fork_session
    assert session.controllers[index] is None
    assert branch.controllers[index].episode_start
    assert not branch.controllers[index].hidden.any()
    assert session.brain.step == session.world.step * NEURAL_STEPS
    assert not np.any(session.brain.g_mV)
    assert session.world.snapshot() == branch.world.snapshot()
    duplicates = [c for c in branch.controllers if c and c.policy is branch.controllers[index].policy]
    assert len(duplicates) == 2 and duplicates[0] is not duplicates[1]
    session.step()
    assert session.world.time == branch.world.time
    restored = session.snapshot()
    session.restore(restored)
    session.step()


def test_snapshot_restores_nonresting_neural_state_and_all_future_randomness(session):
    session.execute({"type": "noise", "value": 0.1})
    for _ in range(3):
        session.step()
    saved = session.snapshot()
    assert saved["state"]["brain"]["step"] == 600
    assert any(value != -52 for value in saved["state"]["brain"]["v_mV"])
    for _ in range(4):
        session.step()
    expected = dynamics(session)
    session.restore(saved)
    for _ in range(4):
        session.step()
    assert dynamics(session) == expected


def test_snapshot_saves_delayed_events_and_fork_independence(session):
    index = session.allocation.index("connectome")
    body = session.world.bodies[index]
    session.world.foods = [Food(body.x, body.y)]
    session.step()
    assert any(session.brain.snapshot()["queue"])
    session.execute({"type": "fork"})
    snapshot = session.snapshot()
    session.step()
    expected = dynamics(session)
    session.restore(snapshot)
    session.step()
    assert dynamics(session) == expected == dynamics(session.fork_session)


@pytest.mark.parametrize("mutation", [
    lambda state: state["brain"].update(graph_digest="0" * 64),
    lambda state: state.update(registry_sha256="0" * 64),
    lambda state: state.update(model_set_sha256="wrong"),
    lambda state: state["brain"].update(step=0),
    lambda state: state["noise_streams"][0].update(bit_generator="MT19937"),
    lambda state: state["channels"].update(vision=0),
    lambda state: state["last_observations"][0].__setitem__(0, -5),
    lambda state: state["controllers"][next(i for i, c in enumerate(state["controllers"]) if c)].update(weights_sha256="0" * 64),
])
def test_invalid_snapshot_rejected_atomically_even_with_recomputed_envelope(session, mutation):
    session.step()
    before = dynamics(session)
    saved = session.snapshot()
    mutation(saved["state"])
    with pytest.raises(ValueError):
        session.restore(rehash(saved))
    assert dynamics(session) == before


def test_checksum_corruption_rejected_and_valid_snapshot_transfers_between_seeds(session, components):
    saved = session.snapshot()
    saved["state"]["world"]["bodies"][0]["food"] = 99
    with pytest.raises(ValueError, match="checksum"):
        session.restore(saved)
    foreign = Session.from_components(components[0], components[1], registry=components[2], seed=109)
    foreign.restore(session.snapshot())
    session.step()
    foreign.step()
    assert dynamics(session) == dynamics(foreign)


def test_atomic_save_load_path_with_spaces_unicode_and_no_pickle(session, tmp_path):
    path = tmp_path / "saved world 공간" / "state.json"
    session.step()
    session.save(path)
    assert json.loads(path.read_text())["schema"] == "neuroterrarium.snapshot.v1"
    session.step()
    expected = dynamics(session)
    session.load(path)
    session.step()
    assert dynamics(session) == expected
    path.write_text('{"schema": "pickle", "__reduce__": "system"}')
    with pytest.raises(ValueError):
        session.load(path)


def test_backend_failure_rolls_back_entire_world_and_other_controllers(session, monkeypatch):
    session.execute({"type": "fork"})
    before, right = dynamics(session), dynamics(session.fork_session)
    def fail(*args, **kwargs):
        raise RuntimeError("injected backend interruption")
    monkeypatch.setattr(session.fork_session.brain, "advance", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        session.step()
    assert dynamics(session) == before
    assert dynamics(session.fork_session) == right
    assert session.paused and session.error


def test_environment_commands_are_real_and_validate_before_mutation(session):
    session.execute({"type": "food_add", "x": 11, "y": 12})
    index = len(session.world.foods) - 1
    session.execute({"type": "food_move", "index": index, "x": 20, "y": 21})
    assert session.world.foods[index].x == 20
    session.execute({"type": "food_remove", "index": index})
    assert len(session.world.foods) == 3
    session.execute({"type": "obstacle_add", "x": 5, "y": 5})
    session.execute({"type": "obstacle_remove", "index": 0})
    assert not session.world.obstacles
    session.execute({"type": "stimulus_add", "x": 30, "y": 15, "vx": 2, "physical": False})
    session.step()
    assert session.world.stimuli[-1].x > 30
    old = session.world.snapshot()
    with pytest.raises(ValueError):
        session.execute({"type": "obstacle_add", "x": session.world.bodies[0].x, "y": session.world.bodies[0].y})
    assert session.world.snapshot() == old


@pytest.mark.parametrize("command", [
    {"type": "channels", "channel": "future", "enabled": True},
    {"type": "channels", "channel": "vision", "enabled": 1},
    {"type": "noise", "value": float("nan")},
    {"type": "noise", "value": 0.3},
    {"type": "readout_clamp", "value": "true"},
    {"type": "food_add", "x": -1, "y": 2},
    {"type": "food_remove", "index": -1},
    {"type": "pause", "shell": "anything"},
    {"type": "fork", "replacement": "missing"},
    {"type": "neural_disconnect", "group": "unknown"},
])
def test_invalid_commands_do_not_mutate(session, command):
    before = dynamics(session)
    with pytest.raises(ValueError):
        session.execute(command)
    assert dynamics(session) == before


def test_clock_controls_reset_and_guess_do_not_train(session):
    session.execute({"type": "pause"})
    session.execute({"type": "step"})
    assert session.world.step == 1 and session.paused
    session.execute({"type": "speed", "value": 0.25})
    assert session.speed == 0.25
    index = session.allocation.index("connectome")
    answer = session.execute({"type": "guess", "selected": index, "guess": "connectome"})
    assert answer["correct"]
    hashes = dict(session.model_weights)
    session.execute({"type": "scenario", "scenario": "occluded"})
    assert session.world.step == 0 and session.world.obstacles
    assert session.model_weights == hashes
    assert len(session.events) == 1


def test_event_log_is_bounded_and_no_commands_record_paths(session):
    for _ in range(MAX_EVENTS + 2):
        session.execute({"type": "sham"})
    assert len(session.events) == MAX_EVENTS
    assert all(event["detail"] == {} for event in session.events)


def test_formal_session_fails_for_missing_models_without_loading_fake_brain(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        Session(tmp_path, tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "neuroterrarium.model-set.v1", "status": "smoke", "models": [{}] * 9}))
    with pytest.raises(ValueError, match="completed"):
        load_model_set(tmp_path)


def test_nine_checkpoint_model_set_integration_and_corrupt_weight_rejection(tmp_path, components):
    from neuroterrarium.training import PPOTrainer, export_model_set

    # The tiny completed protocol is strictly a temporary test fixture. It is
    # never distributed as a trained product model or used by the local app.
    config = json.loads((Path(__file__).parents[1] / "configs/train-smoke.json").read_text())
    config.update(status="frozen", num_envs=2, hidden_size=8, rollout_steps=4,
                  sequence_length=2, minibatch_sequences=4, epochs=1,
                  total_transitions=8, episode_steps=5)
    checkpoints = []
    for architecture in ARCHITECTURES:
        for seed in config["seeds"]:
            trainer = PPOTrainer(config, architecture, seed)
            trainer.update(trainer.collect(4))
            checkpoints.append(trainer.save_checkpoint(tmp_path / f"{architecture}-{seed}"))
    directory = tmp_path / "verified-test-models"
    exported = export_model_set(checkpoints, directory)
    policies, manifest = load_model_set(directory)
    assert manifest == exported and len(policies) == 9
    session = Session.from_components(components[0], policies, registry=components[2])
    session.step()
    assert all(controller.last for controller in session.controllers if controller)
    path = directory / manifest["models"][0]["checkpoint"] / "policy.safetensors"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="integrity"):
        load_model_set(directory)


def test_heterogeneous_reset_preserves_registered_intervention_settings(session):
    index = session.allocation.index("connectome")
    session.execute({"type": "neural_disconnect", "group": "sugar"})
    session.execute({"type": "readout_clamp", "value": True})
    session.execute({"type": "fork", "selected": index, "replacement": "feedforward-101"})
    assert session.brain.output_disconnected[session.groups["sugar"]].all()
    assert session.readout.clamped
    with pytest.raises(ValueError, match="no active"):
        session.execute({"type": "readout_clamp", "value": True, "branch": "right"})


def test_snapshot_random_integer_precision_survives_json_number_normalization(session):
    session.execute({"type": "noise", "value": .1})
    session.step()
    snapshot = session.snapshot()
    streams = snapshot["state"]["noise_streams"] + list(snapshot["state"]["encoder"]["random"].values())
    assert all(isinstance(rng["state"]["state"], str) and isinstance(rng["state"]["inc"], str) for rng in streams)
    # JSON engines may rewrite integral floats as integers without changing the
    # model values. Such spelling differences must not invalidate the checksum.
    text = json.dumps(snapshot)
    decoded = json.loads(text, parse_float=lambda value: int(float(value)) if float(value).is_integer() else float(value))
    session.step()
    expected = dynamics(session)
    session.restore(decoded)
    session.step()
    assert dynamics(session) == expected
    decoded["state"]["noise_streams"][0]["state"]["state"] = 1e30
    with pytest.raises(ValueError, match="decimal strings"):
        session.restore(rehash(decoded))
