"""Targeted archive and provenance boundaries, separate from installation tests."""

import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def builder():
    location = Path(__file__).resolve().parents[1] / "scripts/build_package.py"
    spec = importlib.util.spec_from_file_location("package_builder_tests", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_hash_failure_rejects_existing_cache(builder, tmp_path):
    path = tmp_path / "python.tar.gz"
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        builder.verified(path, builder.RUNTIME)


@pytest.mark.parametrize("name", ["python/../../outside", "/absolute", "unexpected/file"])
def test_archive_escape_is_rejected(builder, tmp_path, name):
    archive = tmp_path / "invalid.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        entry = tarfile.TarInfo(name)
        entry.size = 5
        stream.addfile(entry, io.BytesIO(b"bytes"))
    destination = tmp_path / "output"
    destination.mkdir()
    with pytest.raises((ValueError, tarfile.FilterError)):
        builder.safe_extract_runtime(archive, destination)
    assert not (tmp_path / "outside").exists()


def test_local_build_path_detected_across_chunk_boundary(builder, tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"0" * (1024 * 1024 - 3) + str(Path.home()).encode() + b"other")
    with pytest.raises(ValueError, match="local build path"):
        builder._privacy_check(tmp_path)


def test_escaping_bundle_symlink_rejected(builder, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escaping symbolic link"):
        builder._privacy_check(bundle)


def test_archive_metadata_removes_user_identity_and_is_repeatable(builder, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.txt").write_text("actual bytes")
    (source / "relative-link").symlink_to("example.txt")
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    builder._archive(source, first)
    os.utime(source / "example.txt", (2000, 2000))
    builder._archive(source, second)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as stream:
        assert all(item.uid == item.gid == 0 and item.uname == item.gname == "" and item.mtime == 0
                   for item in stream.getmembers())


def test_bundle_build_rejects_unfrozen_source_before_model_loading(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "check_resources", lambda _: None)
    with pytest.raises(ValueError, match="frozen source"):
        builder.build(tmp_path, tmp_path, tmp_path / "output.tar.gz", "0" * 64, tmp_path)
    assert not (tmp_path / "output.tar.gz").exists()


def test_runtime_license_manifest_cannot_substitute_another_archive(builder, tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "neuroterrarium.runtime-licenses.v1",
        "source": builder.RUNTIME["source"], "source_archive_sha256": "0" * 64,
        "files": {"LICENSE.cpython.txt": {}}}))
    with pytest.raises(ValueError, match="provenance"):
        builder.verify_runtime_licenses(tmp_path)


def test_linkage_checks_loaded_libraries_separately_from_own_install_name(builder, tmp_path, monkeypatch):
    (tmp_path / "example.dylib").write_bytes(b"fixture")
    commands = """Load command 1
cmd LC_ID_DYLIB
cmdsize 48
name example.dylib (offset 24)
Load command 2
cmd LC_LOAD_DYLIB
cmdsize 56
name /usr/lib/libSystem.B.dylib (offset 24)
"""
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *args, **kwargs: commands)
    assert builder.linkage_report(tmp_path)[0]["dependencies"] == ["/usr/lib/libSystem.B.dylib"]
    commands += "cmd LC_LOAD_DYLIB\ncmdsize 48\nname /unavailable/libBad.dylib (offset 24)\n"
    with pytest.raises(ValueError, match="nonportable dynamic dependency"):
        builder.linkage_report(tmp_path)
