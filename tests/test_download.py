"""Actual local HTTP transfer tests; these do not certify public-asset access."""

import hashlib
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from neuroterrarium.download import DownloadError, download_file, verify_file


@pytest.fixture(scope="module")
def http_asset():
    state = {"content": bytes(range(256)) * 1280, "mode": "normal", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request_range = self.headers.get("Range")
            state["requests"].append(request_range)
            mode = state["mode"]
            start = int(request_range.removeprefix("bytes=").removesuffix("-")) if request_range else 0
            status = 206 if request_range else 200
            if mode == "ignore_range":
                status, start = 200, 0
            if mode == "http_failure":
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if mode == "redirect":
                self.send_response(302)
                self.send_header("Location", "/other")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            content = state["content"]
            body = content[start:]
            if mode == "corrupt":
                body = bytes(value ^ 1 for value in body)
            self.send_response(status)
            if mode != "missing_length":
                self.send_header("Content-Length", str(len(body) + (1 if mode == "bad_length" else 0)))
            if request_range and status == 206:
                advertised_start = start + (1 if mode == "bad_range" else 0)
                self.send_header("Content-Range", f"bytes {advertised_start}-{len(content)-1}/{len(content)}")
            if mode == "encoding":
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            if mode == "interrupt":
                state["mode"] = "normal"
                self.wfile.write(body[:98304])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when validation rejects response headers.

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/asset.bin"
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


@pytest.fixture
def source(http_asset):
    http_asset["mode"] = "normal"
    http_asset["requests"].clear()
    return {"filename": "真实 graph.bin", "url": http_asset["url"], "bytes": len(http_asset["content"]),
            "sha256": hashlib.sha256(http_asset["content"]).hexdigest()}


def test_transfer_hash_atomic_publication_and_cached_no_request(http_asset, source, tmp_path):
    cache = tmp_path / "path with spaces 数据"
    events = []
    result = download_file(source, cache, progress=lambda done, total: events.append((done, total)))
    assert result["transfer"] == "downloaded"
    assert (cache / source["filename"]).read_bytes() == http_asset["content"]
    assert not (cache / (source["filename"] + ".partial")).exists()
    assert not (cache / (source["filename"] + ".partial.json")).exists()
    assert events[0] == (0, source["bytes"]) and events[-1] == (source["bytes"], source["bytes"])
    assert download_file(source, cache)["transfer"] == "cached"
    assert http_asset["requests"] == [None]


def test_interrupted_transfer_resumes_actual_bytes_and_hash(http_asset, source, tmp_path):
    http_asset["mode"] = "interrupt"
    with pytest.raises(DownloadError, match="interrupted"):
        download_file(source, tmp_path)
    target = tmp_path / source["filename"]
    partial = target.with_name(target.name + ".partial")
    offset = partial.stat().st_size
    assert 0 < offset < source["bytes"] and not target.exists()
    assert partial.read_bytes() == http_asset["content"][:offset]
    result = download_file(source, tmp_path)
    assert result["transfer"] == "resumed"
    assert http_asset["requests"] == [None, f"bytes={offset}-"]
    assert verify_file(target, source)["sha256"] == source["sha256"]


@pytest.mark.parametrize("mode,match", [("ignore_range", "HTTP 200"), ("bad_range", "Content-Range")])
def test_resume_rejects_false_range_without_modifying_partial(http_asset, source, tmp_path, mode, match):
    http_asset["mode"] = "interrupt"
    with pytest.raises(DownloadError):
        download_file(source, tmp_path)
    partial = tmp_path / (source["filename"] + ".partial")
    before = partial.read_bytes()
    http_asset["mode"] = mode
    with pytest.raises(DownloadError, match=match):
        download_file(source, tmp_path)
    assert partial.read_bytes() == before
    http_asset["mode"] = "normal"
    assert download_file(source, tmp_path, restart=True)["transfer"] == "downloaded"
    assert http_asset["requests"][-1] is None


@pytest.mark.parametrize("mode,match", [("http_failure", "HTTP 503"), ("redirect", "HTTP 302"),
    ("bad_length", "Content-Length"), ("missing_length", "Content-Length"), ("encoding", "encoded response")])
def test_rejects_response_before_any_payload(http_asset, source, tmp_path, mode, match):
    http_asset["mode"] = mode
    with pytest.raises(DownloadError, match=match):
        download_file(source, tmp_path)
    assert not (tmp_path / source["filename"]).exists()
    assert not (tmp_path / (source["filename"] + ".partial")).exists()
    assert len(http_asset["requests"]) == 1


def test_bad_checksum_never_publishes_and_can_explicitly_restart(http_asset, source, tmp_path):
    http_asset["mode"] = "corrupt"
    with pytest.raises(DownloadError, match="SHA-256"):
        download_file(source, tmp_path)
    assert not (tmp_path / source["filename"]).exists()
    assert (tmp_path / (source["filename"] + ".partial")).stat().st_size == source["bytes"]
    http_asset["mode"] = "normal"
    with pytest.raises(DownloadError, match="SHA-256"):
        download_file(source, tmp_path)
    assert len(http_asset["requests"]) == 1
    assert download_file(source, tmp_path, restart=True)["transfer"] == "downloaded"


def test_completed_partial_and_empty_partial_recover_without_fake_resume(http_asset, source, tmp_path):
    calls = []

    def cancel(done, total):
        calls.append(done)
        if done == total:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        download_file(source, tmp_path, progress=cancel)
    assert not (tmp_path / source["filename"]).exists()
    assert download_file(source, tmp_path)["transfer"] == "completed_partial"
    assert len(http_asset["requests"]) == 1
    empty_cache = tmp_path / "empty"
    http_asset["mode"] = "http_failure"
    with pytest.raises(DownloadError):
        download_file(source, empty_cache)
    (empty_cache / (source["filename"] + ".partial")).touch()
    http_asset["mode"] = "normal"
    assert download_file(source, empty_cache)["transfer"] == "downloaded"


def test_partial_identity_cannot_be_adopted_or_changed(http_asset, source, tmp_path):
    partial = tmp_path / (source["filename"] + ".partial")
    partial.write_bytes(b"unknown")
    with pytest.raises(DownloadError, match="no source identity"):
        download_file(source, tmp_path)
    partial.unlink()
    http_asset["mode"] = "interrupt"
    with pytest.raises(DownloadError):
        download_file(source, tmp_path)
    changed = {**source, "sha256": "0" * 64}
    with pytest.raises(DownloadError, match="different source"):
        download_file(changed, tmp_path, restart=True)
    assert partial.exists()


def test_existing_corrupt_final_not_overwritten(http_asset, source, tmp_path):
    target = tmp_path / source["filename"]
    target.write_bytes(b"existing unrelated bytes")
    with pytest.raises(DownloadError, match="byte count"):
        download_file(source, tmp_path, restart=True)
    assert target.read_bytes() == b"existing unrelated bytes"
    assert not http_asset["requests"]


@pytest.mark.parametrize("field,value", [("filename", "../escape"), ("filename", "a\\b"),
    ("filename", "a\nb"), ("bytes", True), ("bytes", 2**34), ("sha256", "n" * 64),
    ("url", "https://user:secret@example.com/data"), ("url", "http://example.com/data"),
    ("url", "file:///tmp/data"), ("url", "https://[broken"), ("url", "https://example.com:bad/data")])
def test_invalid_sources_fail_before_creating_cache(http_asset, source, tmp_path, field, value):
    cache = tmp_path / "unused"
    with pytest.raises(DownloadError):
        download_file({**source, field: value}, cache)
    assert not cache.exists() and not http_asset["requests"]


@pytest.mark.parametrize("suffix", ["", ".partial", ".partial.json", ".lock"])
def test_symlinks_are_not_followed(source, tmp_path, suffix):
    unrelated = tmp_path / "outside"
    unrelated.write_text("unchanged")
    (tmp_path / (source["filename"] + suffix)).symlink_to(unrelated)
    with pytest.raises(DownloadError, match="regular file"):
        download_file(source, tmp_path)
    assert unrelated.read_text() == "unchanged"


def test_concurrent_writer_rejected(source, tmp_path):
    from neuroterrarium.download import _download_lock

    with _download_lock(tmp_path / (source["filename"] + ".lock")), pytest.raises(DownloadError, match="already being downloaded"):
        download_file(source, tmp_path)
