"""Explicit preparation downloads; all simulation and verification remain offline.

Data retain their own licenses. A completed cache file is published only after
its exact frozen byte count and SHA-256 have been checked. Partial files and
their source identity survive a failed transfer for a later Range request.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import stat
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

MAX_DOWNLOAD_BYTES = 2 * 1024**3
CHUNK_BYTES = 64 * 1024
DISK_RESERVE_BYTES = 128 * 1024**2
Progress = Callable[[int, int], None]


class DownloadError(ValueError):
    """An incomplete, unverified or unsafe download; never a usable asset."""


def config_path(filename: str = "data-v783.json") -> Path:
    """Locate bundled configuration in an installation or source checkout."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+\.json", filename):
        raise DownloadError("invalid configuration filename")
    installed = Path(sys.prefix) / "share/neuroterrarium/configs" / filename
    checkout = Path(__file__).resolve().parents[2] / "configs" / filename
    for candidate in (installed, checkout):
        if candidate.is_file():
            return candidate
    raise DownloadError(f"missing packaged configuration: {filename}")


def _identity(source: Mapping) -> dict:
    try:
        filename, url, size, digest = (source[key] for key in ("filename", "url", "bytes", "sha256"))
    except (KeyError, TypeError) as error:
        raise DownloadError("incomplete download source") from error
    if (not isinstance(filename, str) or not 1 <= len(filename) <= 128
            or filename in {".", ".."} or Path(filename).name != filename
            or "\\" in filename or any(ord(char) < 32 for char in filename)):
        raise DownloadError("source filename must be a simple relative name")
    if type(size) is not int or not 0 < size <= MAX_DOWNLOAD_BYTES:
        raise DownloadError("source byte count exceeds the download budget")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise DownloadError("invalid source SHA-256")
    if not isinstance(url, str) or len(url) > 4096 or any(char.isspace() or ord(char) < 32 for char in url):
        raise DownloadError("invalid source URL")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        _ = parsed.port  # Validate malformed or out-of-range ports before any I/O.
    except ValueError as error:
        raise DownloadError("invalid source URL") from error
    try:
        is_local = host == "localhost" or (host is not None and ipaddress.ip_address(host).is_loopback)
    except ValueError:
        is_local = False
    if (parsed.scheme != "https" and not (parsed.scheme == "http" and is_local)) or (
        not host or parsed.username is not None or parsed.password is not None or parsed.fragment
    ):
        raise DownloadError("source must use HTTPS without credentials, or loopback HTTP")
    return {"schema": "neuroterrarium.partial.v1", "filename": filename,
            "url": url, "bytes": size, "sha256": digest}


def _regular_or_missing(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise DownloadError("cache asset must be a regular file, never a symbolic link")
    return True


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path | str, source: Mapping) -> dict:
    """Verify a local asset without making a network request."""
    identity = _identity(source)
    path = Path(path)
    if not _regular_or_missing(path):
        raise DownloadError(f"missing data file: {identity['filename']}")
    if path.stat().st_size != identity["bytes"]:
        raise DownloadError(f"byte count mismatch: {identity['filename']}")
    digest = _hash(path)
    if digest != identity["sha256"]:
        raise DownloadError(f"SHA-256 mismatch: {identity['filename']}")
    return {"filename": identity["filename"], "bytes": identity["bytes"],
            "sha256": digest, "status": "verified"}


@contextmanager
def _download_lock(path: Path) -> Iterator[None]:
    # OS locks release on process exit, including a killed preparation process.
    # The small lock file may remain in the cache and never contains user data.
    try:
        import fcntl
    except ImportError as error:
        raise DownloadError("preparation locking is currently supported on POSIX systems") from error
    _regular_or_missing(path)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DownloadError("this asset is already being downloaded") from error
        yield
    finally:
        os.close(descriptor)


def _write_identity(path: Path, identity: dict) -> None:
    # The per-asset lock protects this write; an interrupted metadata write is
    # rejected on the next attempt, never interpreted as a different source.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def download_file(source: Mapping, data_dir: Path | str, *, progress: Progress | None = None,
                  restart: bool = False) -> dict:
    """Download one frozen source, resuming only a correctly identified partial.

    Range responses must name the requested offset and frozen total length.
    Redirects and content encoding are rejected. An ignored Range request must
    be explicitly retried with ``restart=True``; it is never appended to a file.
    Existing completed files are verified rather than overwritten.
    """
    import requests

    identity = _identity(source)
    if type(restart) is not bool:
        raise DownloadError("restart must be a boolean")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / identity["filename"]
    partial = target.with_name(target.name + ".partial")
    metadata = target.with_name(target.name + ".partial.json")
    with _download_lock(target.with_name(target.name + ".lock")):
        if _regular_or_missing(target):
            return {**verify_file(target, source), "transfer": "cached"}
        has_partial = _regular_or_missing(partial)
        has_metadata = _regular_or_missing(metadata)
        if has_metadata:
            try:
                if metadata.stat().st_size > 8192:
                    raise DownloadError("partial metadata exceeds size limit")
                stored = json.loads(metadata.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise DownloadError("invalid partial metadata; cannot safely resume") from error
            if stored != identity:
                raise DownloadError("partial belongs to a different source; cannot resume or restart")
        elif has_partial:
            raise DownloadError("partial has no source identity; cannot safely resume")
        if restart and has_metadata:
            if has_partial:
                partial.unlink()
            metadata.unlink()
            has_partial = has_metadata = False
        if not has_metadata:
            _write_identity(metadata, identity)
        offset = partial.stat().st_size if has_partial else 0
        if offset > identity["bytes"]:
            raise DownloadError("partial exceeds expected size; use --restart")
        if offset == identity["bytes"]:
            verify_file(partial, source)
            os.replace(partial, target)
            metadata.unlink()
            return {**verify_file(target, source), "transfer": "completed_partial"}
        if shutil.disk_usage(data_dir).free < identity["bytes"] - offset + DISK_RESERVE_BYTES:
            raise DownloadError("insufficient free disk space including the 128 MiB reserve")
        headers = {"Accept-Encoding": "identity", "User-Agent": "NeuroTerrarium/0.1 data-preparation"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        if progress:
            progress(offset, identity["bytes"])
        try:
            with requests.Session() as session:
                # Do not inherit netrc credentials, authentication headers or
                # proxy credentials from a user's unrelated shell environment.
                session.trust_env = False
                with session.get(identity["url"], headers=headers, stream=True,
                                 allow_redirects=False, timeout=(10, 30)) as response:
                    expected_status = 206 if offset else 200
                    if response.status_code != expected_status:
                        raise DownloadError(f"unexpected HTTP {response.status_code}; expected {expected_status}")
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise DownloadError("encoded response cannot be used for a byte-exact download")
                    expected_remaining = identity["bytes"] - offset
                    length = response.headers.get("Content-Length")
                    if length is None or not re.fullmatch(r"[0-9]+", length) or int(length) != expected_remaining:
                        raise DownloadError("HTTP Content-Length does not match the frozen byte count")
                    if offset:
                        expected_range = f"bytes {offset}-{identity['bytes'] - 1}/{identity['bytes']}"
                        if response.headers.get("Content-Range") != expected_range:
                            raise DownloadError("HTTP Content-Range does not match the requested resume")
                    elif response.headers.get("Content-Range") is not None:
                        raise DownloadError("unexpected Content-Range on a complete download")
                    written = offset
                    with partial.open("ab" if has_partial else "xb") as stream:
                        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                            if not chunk:
                                continue
                            if written + len(chunk) > identity["bytes"]:
                                raise DownloadError("response exceeds the frozen byte count")
                            stream.write(chunk)
                            written += len(chunk)
                            if progress:
                                progress(written, identity["bytes"])
                        stream.flush()
                        os.fsync(stream.fileno())
                    if written != identity["bytes"]:
                        raise DownloadError("incomplete response; partial retained for resume")
        except requests.RequestException as error:
            # Exception strings can contain proxy URLs or authentication data.
            raise DownloadError(f"transfer interrupted ({type(error).__name__}); partial retained") from error
        result = verify_file(partial, source)
        os.replace(partial, target)
        metadata.unlink()
        return {**result, "transfer": "resumed" if offset else "downloaded"}


def fetch_data(data_dir: Path | str, manifest_path: Path | str | None = None, *,
               progress: Callable[[str, int, int], None] | None = None, restart: bool = False) -> dict:
    """Fetch the three pinned profile assets sequentially with bounded buffers."""
    from .data import read_manifest

    manifest = read_manifest(config_path() if manifest_path is None else manifest_path)
    results = {}
    for name, source in manifest["sources"].items():
        callback = None if progress is None else lambda done, total, name=name: progress(name, done, total)
        results[name] = download_file(source, data_dir, progress=callback, restart=restart)
    return {"status": "verified", "scope": "asset bytes and SHA-256",
            "profile": manifest["profile"], "license": manifest["licenses"]["flywire_data"],
            "files": results}
