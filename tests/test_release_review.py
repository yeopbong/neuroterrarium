"""Independent malformed-recording and failed-journal boundary checks."""

import copy
import json

from fastapi.testclient import TestClient
import pytest

from neuroterrarium.recording import frame
from neuroterrarium.replay import load_recording
from neuroterrarium.service import create_app
from neuroterrarium.world import Stimulus, World

pytest_plugins = ["test_runtime"]


@pytest.mark.parametrize("stimulus", [
    Stimulus(40, 28, 99.99, growth=10),
    Stimulus(9999.9, 28, 1, vx=20),
])
def test_reachable_long_running_stimulus_can_restore(stimulus):
    world = World(9, 1)
    world.stimuli = [copy.deepcopy(stimulus)]
    World.restore(world.snapshot())
    world.advance([[0, 0, 0, 0]])
    restored = World.restore(world.snapshot())
    assert restored.snapshot() == world.snapshot()
    world.advance([[0, 0, 0, 0]])
    restored.advance([[0, 0, 0, 0]])
    assert restored.snapshot() == world.snapshot()


@pytest.mark.parametrize("field,value,step,rate", [
    ("radius", 101, 0, 10), ("radius", 121, 100, 10),
    ("radius", 101, 100, -10), ("x", 10041, 100, 20),
    ("y", -10041, 100, -20), ("radius", 10**20, 10**9, 10000),
])
def test_stimulus_outside_finite_elapsed_domain_is_rejected(field, value, step, rate):
    world = World(9, 1)
    stimulus = Stimulus(40, 28)
    setattr(stimulus, field, value)
    setattr(stimulus, {"radius": "growth", "x": "vx", "y": "vy"}[field], rate)
    world.stimuli = [stimulus]
    world.step = step
    with pytest.raises(ValueError):
        World.restore(world.snapshot())


def test_long_stimulus_execution_and_restored_continuation():
    world = World(9, 1)
    world.stimuli = [Stimulus(40, 28, 1, vx=20, growth=3)]
    for _ in range(25050):
        world.advance([[0, 0, 0, 0]])
    assert world.time == 501
    assert world.stimuli[0].x > 10000
    assert world.stimuli[0].radius > 100
    restored = World.restore(world.snapshot())
    for _ in range(20):
        world.advance([[0, 0, 0, 0]])
        restored.advance([[0, 0, 0, 0]])
        assert restored.snapshot() == world.snapshot()


@pytest.mark.parametrize("damage", [
    "string_action", "out_of_range_action", "negative_energy", "bad_width",
    "negative_food", "missing_objects", "bad_observation", "bad_panel_index", "nested_fork",
])
def test_malformed_replay_is_rejected_before_serving(session, tmp_path, damage):
    session.step()
    recorded = frame(session)
    if damage == "string_action":
        recorded["world"]["bodies"][0]["action"] = ["invalid"] * 4
    elif damage == "out_of_range_action":
        recorded["world"]["bodies"][0]["action"][0] = 2
    elif damage == "negative_energy":
        recorded["world"]["bodies"][0]["energy"] = -10
    elif damage == "bad_width":
        recorded["world"]["width"] = "wide"
    elif damage == "negative_food":
        recorded["world"]["foods"][0]["amount"] = -1
    elif damage == "missing_objects":
        del recorded["world"]["stimuli"]
    elif damage == "bad_observation":
        recorded["panels"][0]["observation"] = [0.0]
    elif damage == "bad_panel_index":
        recorded["panels"][0]["index"] = 9
    elif damage == "nested_fork":
        recorded["fork"] = copy.deepcopy(recorded)
        recorded["fork"]["fork"] = copy.deepcopy(recorded)
    path = tmp_path / "malformed-recording.json"
    path.write_text(json.dumps({"schema": "neuroterrarium.frame-replay.v1", "mode": "Replay", "frames": [recorded]}))
    with pytest.raises(ValueError):
        load_recording(path)


class FailingJournal:
    def write(self, _record):
        raise RuntimeError("Simulated storage failure")

    def restored(self, _session):
        raise RuntimeError("Simulated storage failure")

    def close(self):
        pass


@pytest.mark.parametrize("operation", ["command", "snapshot"])
def test_journal_failure_pauses_and_marks_session_unavailable(session, tmp_path, operation):
    session.paused = True
    (tmp_path / "index.html").write_text("<title>Recording transport boundary</title>")
    app = create_app(session, web_dir=tmp_path)
    app.state.controller.journal = FailingJournal()
    with TestClient(app, base_url="http://127.0.0.1:8765", raise_server_exceptions=False) as client:
        if operation == "command":
            response = client.post("/api/command", json={"type": "food_add", "x": 1, "y": 1})
        else:
            response = client.post("/api/snapshot", json=session.snapshot())
        assert response.status_code >= 400
        assert app.state.controller.error is not None
        assert session.paused
        assert client.get("/api/state").status_code == 503
