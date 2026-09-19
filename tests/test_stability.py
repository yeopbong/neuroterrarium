"""The hour gate must reject short, stalled, incomplete and replay-only evidence."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neuroterrarium.controllers import ARCHITECTURES, Policy
from neuroterrarium.data import GraphData
from neuroterrarium.recording import ExecutionJournal, recompute
from neuroterrarium.registry import Registry
from neuroterrarium.runtime import NEURAL_STEPS, Session


spec = importlib.util.spec_from_file_location("stability_script", Path(__file__).parents[1] / "scripts/stability.py")
stability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stability)


def complete_summary():
    return {"status": "completed", "requested_wall_seconds": 3600, "measured_wall_seconds": 3601,
            "actual_wall_seconds": 3650, "resource_samples": 1000, "active_wall_seconds_estimate": 3590,
            "maximum_unpaused_progress_gap_seconds": 2, "committed_action_windows": 20000,
            "journal_action_windows": 20000,
            "operations": [{"operation": operation} for operation in stability.REQUIRED_OPERATIONS]}


def test_hour_gate_accepts_only_completed_measured_execution():
    assert stability.hour_gate(complete_summary())


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("status", "cancelled"), ("requested_wall_seconds", 120),
    ("measured_wall_seconds", 3599), ("resource_samples", 100),
    ("active_wall_seconds_estimate", 100), ("maximum_unpaused_progress_gap_seconds", 31),
    ("committed_action_windows", 0), ("journal_action_windows", 19999), ("operations", []),
])
def test_hour_gate_cannot_be_padded_by_teardown_or_recorded_frames(field, value):
    summary = complete_summary()
    summary[field] = value
    summary["actual_wall_seconds"] = 7200
    assert not stability.hour_gate(summary)


def test_counter_rejects_missing_bodies_and_neural_progress():
    written = []
    journal = SimpleNamespace(advanced=lambda session, command: written.append(command))
    counter = stability.WindowCounter(journal)
    session = SimpleNamespace(world=SimpleNamespace(bodies=list(range(10)), step=1),
                              brain=SimpleNamespace(step=NEURAL_STEPS),
                              allocation=["connectome"] + ["policy"] * 9,
                              records=[{"decisions": [{}] * 10}])
    journal.advanced(session)
    assert counter.windows == 1 and len(written) == 1
    session.brain.step -= 1
    with pytest.raises(RuntimeError, match="complete ten-controller"):
        journal.advanced(session)
    assert counter.windows == len(written) == 1
    session.brain.step += 1
    session.world.bodies.pop()
    with pytest.raises(RuntimeError):
        journal.advanced(session)
    assert counter.windows == len(written) == 1


def test_counter_does_not_count_a_failed_journal_write():
    def failed_write(session, command):
        raise OSError("disk write failed")
    journal = SimpleNamespace(advanced=failed_write)
    counter = stability.WindowCounter(journal)
    session = SimpleNamespace(world=SimpleNamespace(bodies=list(range(10)), step=1),
                              brain=SimpleNamespace(step=NEURAL_STEPS),
                              allocation=["connectome"] + ["policy"] * 9,
                              records=[{"decisions": [{}] * 10}])
    with pytest.raises(OSError):
        journal.advanced(session)
    assert counter.windows == 0


@pytest.mark.parametrize("entries", [[], [{"sequence": 0, "schema": "frame-replay"}],
    [{"sequence": 0, "schema": "neuroterrarium.execution.v1"},
     {"sequence": 2, "type": "advance", "record": {"decisions": [{}] * 10}}],
    [{"sequence": 0, "schema": "neuroterrarium.execution.v1"},
     {"sequence": 1, "type": "advance", "record": {"decisions": [{}] * 9}}]])
def test_journal_counter_rejects_empty_replay_and_partial_worlds(tmp_path, entries):
    path = tmp_path / "journal.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in entries))
    with pytest.raises(ValueError):
        stability.count_journal_windows(path)


def test_snapshot_comparison_preserves_executable_branch_state():
    before = {"brain": {"v": [0.2]}, "paused": True, "speed": 2,
              "wall_time": 5, "events": ["save"], "records": [], "fork": None}
    after = copy.deepcopy(before)
    after.update(wall_time=6, events=["restore"])
    assert stability.executable_snapshot(before) == stability.executable_snapshot(after)
    after["brain"]["v"][0] = 0
    assert stability.executable_snapshot(before) != stability.executable_snapshot(after)
    after = copy.deepcopy(before)
    after["fork"] = copy.deepcopy(before)
    assert stability.executable_snapshot(before) != stability.executable_snapshot(after)


def test_real_test_components_rewind_does_not_erase_completed_window_count(tmp_path):
    torch.set_num_threads(1)
    registry = Registry.load()
    roots = np.array([int(node["root_id"]) for group in registry.groups.values()
                      for node in group["neurons"]], dtype=np.int64)
    empty = np.zeros(0, dtype=np.int32)
    graph = GraphData(roots, empty, empty, empty, {"profile": "synthetic-test-only"})
    policies = {f"{architecture}-{seed}": Policy(architecture, seed, 8)
                for architecture in ARCHITECTURES for seed in (101, 202, 303)}
    session = Session.from_components(graph, policies, registry=registry, seed=515)
    session.paused = True
    journal = ExecutionJournal(tmp_path / "execution", session)
    counter = stability.WindowCounter(journal)
    for _ in range(2):
        session.step()
        journal.advanced(session)
    saved = session.snapshot()
    session.step()
    journal.advanced(session)
    session.restore(saved)
    journal.restored(session)
    assert session.world.step == 2 and counter.windows == 3
    session.execute({"type": "step"})
    journal.advanced(session, {"type": "step"})
    journal.close()
    assert session.world.step == 3 and counter.windows == 4
    assert stability.count_journal_windows(tmp_path / "execution/execution.jsonl") == 4
    result = recompute(session._clone(), tmp_path / "execution")
    assert result["status"] == "passed" and result["action_windows"] == 4


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), float("inf"), 86401])
def test_duration_validation_precedes_data_loading(tmp_path, duration):
    with pytest.raises(ValueError, match="Duration"):
        stability.run("missing", "missing", tmp_path, duration)
