"""Installation gate failure handling and isolation contract checks."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def installer():
    location = Path(__file__).resolve().parents[1] / "scripts/install_check.py"
    spec = importlib.util.spec_from_file_location("installation_gate_tests", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_directory_with_multiple_wheels_cannot_choose_a_stale_build(installer, tmp_path):
    (tmp_path / "neuroterrarium-0.1.0-py3-none-any.whl").touch()
    (tmp_path / "neuroterrarium-0.0.9-py3-none-any.whl").touch()
    with pytest.raises(ValueError, match="exactly one"):
        installer.pick_wheel(tmp_path)


def test_failed_evidence_is_never_overwritten(installer, tmp_path):
    wheel = tmp_path / "neuroterrarium-0.1.0-py3-none-any.whl"
    wheel.touch()
    output = tmp_path / "evidence"
    output.mkdir()
    marker = output / "summary.json"
    marker.write_text("original failed run")
    with pytest.raises(ValueError, match="must be empty"):
        installer.run_check(wheel, tmp_path, tmp_path, output)
    assert marker.read_text() == "original failed run"


def test_isolation_probe_is_syntactically_valid_and_rejects_outbound_connection(installer, tmp_path):
    compile(installer.PROBE, "installed_probe.py", "exec")
    # Exercise the actual guard in a separate interpreter; no socket patch is
    # left in the pytest process, and no connection reaches an external host.
    prefix = installer.PROBE.split("import neuroterrarium")[0]
    attempts = '''
for call in (lambda: socket.create_connection(("example.invalid", 443)),
             lambda: socket.socket().connect(("example.invalid", 443)),
             lambda: socket.getaddrinfo("example.invalid", 443)):
    try:
        call()
    except RuntimeError as error:
        assert "network access forbidden" in str(error)
    else:
        raise AssertionError("outbound call bypassed the guard")
assert len(network_attempts) == 3
'''
    probe = tmp_path / "network_guard.py"
    probe.write_text(prefix + attempts)
    subprocess.run([sys.executable, "-I", str(probe)], check=True, timeout=10)
