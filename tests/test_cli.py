"""CLI boundary tests; invocation wiring is distinct from scientific evidence."""

import json
from pathlib import Path

import pytest

from neuroterrarium import cli
from neuroterrarium.download import config_path


def test_installed_or_checkout_configuration_is_frozen_profile():
    data = json.loads(config_path().read_text())
    assert data["profile"] == "shiu-v783-full"
    assert data["expected"]["neurons"] == 138639
    assert all(source["connectome_version"] == "783" for source in data["sources"].values())


def test_help_and_version_do_not_import_heavy_runtime(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert "data" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        cli.main(["--version"])
    assert error.value.code == 0


def test_doctor_missing_assets_is_not_ready_without_personal_information(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_gpu_status", lambda: {"scope": "current process", "mps_available": False})
    result = cli.main(["doctor", "--data-dir", str(tmp_path / "absent"), "--models", str(tmp_path / "models")])
    text = capsys.readouterr().out
    data = json.loads(text)
    assert result == 1 and data["status"] == "not_ready"
    assert data["readiness"]["data"]["status"] == "not_ready"
    assert data["memory"]["total_bytes"] >= data["memory"]["available_bytes"] > 0
    assert str(tmp_path) not in text and str(Path.home()) not in text
    assert "hostname" not in text and "environ" not in text


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.3", "::", "example.com"])
def test_app_rejects_nonloopback_before_starting_service(host):
    with pytest.raises(SystemExit) as error:
        cli.main(["app", "--host", host])
    assert error.value.code == 2


def test_app_defaults_and_arguments_are_wired(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_entry", lambda module, function: lambda *args, **kwargs: calls.append((module, function, args, kwargs)))
    assert cli.main(["app"]) == 0
    module, function, args, kwargs = calls.pop()
    assert (module, function) == ("service", "serve")
    assert args == (Path("data"), Path("artifacts/models"))
    assert kwargs == {"host": "127.0.0.1", "port": 8765, "open_browser": False}


def test_offline_graph_prepare_routes_rebuild_explicitly(monkeypatch, capsys):
    calls = []
    def entry(module, function):
        assert (module, function) == ("brain_cache", "prepare_cache")
        def prepare(data_dir, *, manifest_path, rebuild):
            calls.append((data_dir, manifest_path, rebuild))
            return {"status": "prepared"}
        return prepare
    monkeypatch.setattr(cli, "_entry", entry)
    assert cli.main(["data", "prepare", "--data-dir", "graph cache", "--rebuild"]) == 0
    assert calls == [(Path("graph cache"), None, True)]


def test_fetch_prepares_only_after_verified_downloads(monkeypatch, tmp_path, capsys):
    from neuroterrarium import download
    calls = []
    monkeypatch.setattr(download, "fetch_data", lambda *a, **k: {"status": "verified"})
    def entry(module, function):
        assert (module, function) == ("brain_cache", "prepare_cache")
        def prepare(*args, **kwargs):
            calls.append((args, kwargs))
            return {"status": "verified"}
        return prepare
    monkeypatch.setattr(cli, "_entry", entry)
    assert cli.main(["data", "fetch", "--data-dir", str(tmp_path)]) == 0
    assert len(calls) == 1 and calls[0][1]["rebuild"] is False
    monkeypatch.setattr(download, "fetch_data", lambda *a, **k: {"status": "incomplete"})
    assert cli.main(["data", "fetch", "--data-dir", str(tmp_path)]) == 3
    assert len(calls) == 1
    monkeypatch.setattr(download, "fetch_data", lambda *a, **k: {"status": "verified"})
    monkeypatch.setattr(cli, "_entry", lambda *a: lambda *a, **k: {"status": "failed"})
    assert cli.main(["data", "fetch", "--data-dir", str(tmp_path)]) != 0


def test_training_resume_signature_maps_exactly(monkeypatch, capsys):
    calls = []

    def entry(module, function):
        assert (module, function) == ("training", "train")

        def train(config, output_dir, *, architecture, seed, resume_from, stop_after_updates, cancel_file):
            calls.append((config, output_dir, architecture, seed, resume_from, stop_after_updates, cancel_file))
            return {"status": "completed"}
        return train

    monkeypatch.setattr(cli, "_entry", entry)
    assert cli.main(["train", "--config", "c.json", "--output", "run", "--architecture", "recurrent",
        "--seed", "31", "--resume", "previous", "--stop-after-updates", "3", "--cancel-file", "stop"]) == 0
    assert calls == [(Path("c.json"), Path("run"), "recurrent", 31, Path("previous"), 3, Path("stop"))]
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_missing_module_is_explicit_failure(monkeypatch, capsys):
    def missing(*_):
        raise ImportError("internal private path must not be echoed")

    monkeypatch.setattr(cli.importlib, "import_module", missing)
    assert cli.main(["evaluate", "--config", "protocol.json", "--output", "results"]) == 2
    output = capsys.readouterr()
    assert not output.out and "operation was not performed" in output.err
    assert "private path" not in output.err


def test_data_verify_missing_real_data_fails(tmp_path, capsys):
    assert cli.main(["data", "verify", "--data-dir", str(tmp_path)]) == 2
    assert "missing completeness data" in capsys.readouterr().err


def test_keyboard_interrupt_reports_cancellation(monkeypatch, capsys):
    def cancel(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "_entry", lambda *_: cancel)
    assert cli.main(["app"]) == 130
    assert "Cancelled" in capsys.readouterr().err


@pytest.mark.parametrize("status", ["interrupted", "incomplete", "cancelled", "pending", "failed", "error", "not_ready", "blocked", "running", "timeout", "unknown", None])
def test_incomplete_or_failed_structured_result_has_nonzero_exit(monkeypatch, capsys, status):
    monkeypatch.setattr(cli, "_entry", lambda *_: lambda *args, **kwargs: {"status": status})
    assert cli.main(["train", "--config", "c.json", "--output", "run"]) != 0
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_evaluate_and_replay_real_signatures(monkeypatch, capsys):
    calls = []

    def entry(module, function):
        def call(*args, **kwargs):
            calls.append((module, function, args, kwargs))
            return {"status": "completed"}
        return call

    monkeypatch.setattr(cli, "_entry", entry)
    assert cli.main(["evaluate", "--config", "p.json", "--output", "out"]) == 0
    assert calls.pop() == ("evaluation", "evaluate", (Path("p.json"), Path("data"), Path("artifacts/models"), Path("out")), {"cancel_file": None})
    assert cli.main(["evaluate", "--config", "p.json", "--output", "out", "--cancel-file", "stop"]) == 0
    assert calls.pop()[-1] == {"cancel_file": Path("stop")}
    assert cli.main(["replay", "capture.json"]) == 0
    assert calls.pop() == ("replay", "serve_replay", (Path("capture.json"),), {"port": 8766, "open_browser": False})


@pytest.mark.parametrize('status', ['cancelled', 'resource_limited', 'budget_exhausted', 'incomplete'])
def test_evaluation_partial_status_is_never_success(monkeypatch, capsys, status):
    monkeypatch.setattr(cli, '_entry', lambda *args: lambda *args, **kwargs: {'status': status})
    assert cli.main(['evaluate', '--config', 'p.json', '--output', 'out']) != 0
