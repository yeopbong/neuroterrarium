"""Verified, bounded-memory loading of versioned connectivity data.

Root IDs never pass through floating-point values. The graph preserves the
upstream node ordering and every connection; annotations do not add nodes or
change the supplied signs. Source data is separate from the code license.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class DataValidationError(ValueError):
    """Missing, corrupt, incompatible, or internally inconsistent brain data."""


ROOT_PATTERN = re.compile(r"[1-9][0-9]*\Z")
INT64_MAX = np.iinfo(np.int64).max
INT32_MAX = np.iinfo(np.int32).max
MAX_GRAPH_WORKSPACE_BYTES = 768 * 1024 * 1024
CONNECTION_COLUMNS = (
    "Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index",
    "Postsynaptic_Index", "Connectivity", "Excitatory",
    "Excitatory x Connectivity",
)


def parse_root_id(value: str | int) -> int:
    """Parse a canonical decimal string or integer without any float coercion."""
    if isinstance(value, (bool, np.bool_)):
        raise DataValidationError("root ID must be an integer or decimal string")
    if isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, str) and ROOT_PATTERN.fullmatch(value):
        result = int(value)
    else:
        raise DataValidationError("root ID must be an integer or decimal string")
    if not 0 < result <= INT64_MAX:
        raise DataValidationError("root ID outside positive int64 range")
    return result


def root_ids_to_wire(values: Sequence[int]) -> list[str]:
    """Serialize IDs for JSON/browser use as decimal strings, never numbers."""
    return [str(parse_root_id(value)) for value in values]


def root_ids_from_wire(values: Sequence[str]) -> list[int]:
    """Require string IDs at a JSON/browser boundary, including exact integers."""
    if isinstance(values, (str, bytes)) or any(not isinstance(v, str) for v in values):
        raise DataValidationError("wire root IDs must be decimal strings")
    return [parse_root_id(value) for value in values]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GraphData:
    """Complete immutable structure; each simulator owns its separate state."""

    root_ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    signed_counts: np.ndarray
    summary: dict

    def indices(self, root_ids: Sequence[str | int]) -> np.ndarray:
        requested = [parse_root_id(value) for value in root_ids]
        if len(set(requested)) != len(requested):
            raise DataValidationError("registered root IDs contain duplicates")
        lookup = {int(root): i for i, root in enumerate(self.root_ids)}
        missing = [str(root) for root in requested if root not in lookup]
        if missing:
            raise DataValidationError(f"registered root IDs absent from graph: {missing}")
        return np.asarray([lookup[root] for root in requested], dtype=np.int32)


def read_manifest(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise DataValidationError("data manifest missing or larger than 1 MiB")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise DataValidationError("data manifest must be an object")
        if manifest["schema"] != "neuroterrarium.data.v1":
            raise DataValidationError("unsupported data manifest schema")
        version = manifest["connectome_version"]
        if not isinstance(version, str) or not manifest["profile"]:
            raise DataValidationError("invalid profile version")
        sources = manifest["sources"]
        if not isinstance(sources, dict) or set(sources) != {"completeness", "connectivity", "annotations"}:
            raise DataValidationError("manifest must specify all three data sources")
        for source in sources.values():
            if not isinstance(source, dict):
                raise DataValidationError("source description must be an object")
            filename = source["filename"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise DataValidationError("source filename must be a simple relative name")
            if filename in (".", "..", "") or "\\" in filename:
                raise DataValidationError("invalid source filename")
            if source["connectome_version"] != version:
                raise DataValidationError("data source version mismatch")
            if not re.fullmatch("[a-f0-9]{64}", source["sha256"]):
                raise DataValidationError("invalid source SHA-256")
            if type(source["bytes"]) is not int or source["bytes"] <= 0:
                raise DataValidationError("invalid source byte count")
        for key in ("neurons", "connection_rows", "directed_pairs", "synaptic_contacts"):
            if type(manifest["expected"][key]) is not int or manifest["expected"][key] <= 0:
                raise DataValidationError("invalid expected graph count")
        return manifest
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DataValidationError("malformed data manifest") from exc


def _verify_sources(data_dir: Path, manifest: dict) -> dict[str, Path]:
    paths = {}
    for name, source in manifest["sources"].items():
        path = data_dir / source["filename"]
        if not path.is_file():
            raise DataValidationError(f"missing {name} data; run data preparation")
        if path.stat().st_size != source["bytes"]:
            raise DataValidationError(f"{name} byte count mismatch")
        if sha256_file(path) != source["sha256"]:
            raise DataValidationError(f"{name} SHA-256 mismatch")
        paths[name] = path
    return paths


def _read_roots(path: Path, expected_count: int) -> np.ndarray:
    roots = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["", "Completed"]:
            raise DataValidationError("unexpected completeness columns")
        for row in reader:
            if len(row) != 2 or row[1] != "True":
                raise DataValidationError("invalid or incomplete neuron row")
            root = parse_root_id(row[0])
            if root in seen:
                raise DataValidationError("duplicate completeness root ID")
            seen.add(root)
            roots.append(root)
            if len(roots) > expected_count:
                raise DataValidationError("neuron count exceeds manifest")
    if len(roots) != expected_count or len(roots) > INT32_MAX:
        raise DataValidationError("neuron count differs from manifest")
    return np.asarray(roots, dtype=np.int64)


def _verify_annotations(path: Path, roots: np.ndarray) -> dict:
    graph_roots = set(int(value) for value in roots)
    seen = set()
    unknown = {"type": 0, "side": 0, "predicted_transmitter": 0}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"root_id", "supervoxel_id", "cell_type", "hemibrain_type", "side", "top_nt"}
        if not required.issubset(reader.fieldnames or ()):
            raise DataValidationError("annotation columns missing")
        for row in reader:
            root = parse_root_id(row["root_id"])
            parse_root_id(row["supervoxel_id"])
            if root in seen:
                raise DataValidationError("duplicate annotation root ID")
            seen.add(root)
            if root in graph_roots:
                unknown["type"] += not (row["cell_type"] or row["hemibrain_type"])
                unknown["side"] += row["side"] in ("", "na")
                unknown["predicted_transmitter"] += not row["top_nt"]
    if graph_roots - seen:
        raise DataValidationError("graph root IDs missing from annotations")
    return {"rows": len(seen), "extra_roots_outside_graph": len(seen - graph_roots),
            "unknown_in_graph": unknown}


def load_graph(
    data_dir: Path | str,
    manifest_path: Path | str,
    *,
    required_root_ids: Sequence[str | int] = (),
    batch_size: int = 65536,
) -> GraphData:
    """Verify source bytes, exact integer fields, identities and all graph rows.

    No network request, pruning, sign correction, or synthetic fallback occurs.
    Parquet decoding uses one thread and bounded batches. Duplicate directed
    pairs fail: this profile already aggregates each directed pair upstream.
    The pandas index column is ignored; only explicit neuron index fields
    identify endpoints. Loading preserves physical source row order.
    """
    started = perf_counter()
    if type(batch_size) is not int or not 1 <= batch_size <= 262144:
        raise DataValidationError("batch size must be between 1 and 262144")
    manifest = read_manifest(manifest_path)
    expected = manifest["expected"]
    paths = _verify_sources(Path(data_dir), manifest)
    roots = _read_roots(paths["completeness"], expected["neurons"])
    annotation_summary = _verify_annotations(paths["annotations"], roots)
    try:
        parquet = pq.ParquetFile(paths["connectivity"])
    except (pa.ArrowException, OSError) as exc:
        raise DataValidationError("invalid connectivity parquet") from exc
    if parquet.metadata.num_rows != expected["connection_rows"]:
        raise DataValidationError("connection row count differs from manifest")
    for name in CONNECTION_COLUMNS:
        if name not in parquet.schema_arrow.names or parquet.schema_arrow.field(name).type != pa.int64():
            raise DataValidationError(f"{name} must have exact int64 storage")
    rows = parquet.metadata.num_rows
    # Three int32 arrays plus a temporary uint64 pair key, with node overhead.
    # Leave headroom for Arrow's bounded decoder and the host application.
    if rows * 20 + len(roots) * 32 > MAX_GRAPH_WORKSPACE_BYTES:
        raise DataValidationError("graph exceeds loader memory budget")
    pre = np.empty(rows, dtype=np.int32)
    post = np.empty(rows, dtype=np.int32)
    signed = np.empty(rows, dtype=np.int32)
    pairs = np.empty(rows, dtype=np.uint64)
    neuron_sign_min = np.full(len(roots), 2, dtype=np.int8)
    neuron_sign_max = np.full(len(roots), -2, dtype=np.int8)
    offset = contacts = positive = negative = self_pairs = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(CONNECTION_COLUMNS), use_threads=False):
        if any(column.null_count for column in batch.columns):
            raise DataValidationError("null connectivity fields")
        arrays = {name: batch.column(name).to_numpy(zero_copy_only=False) for name in CONNECTION_COLUMNS}
        src, dst = arrays["Presynaptic_Index"], arrays["Postsynaptic_Index"]
        count, sign = arrays["Connectivity"], arrays["Excitatory"]
        weight = arrays["Excitatory x Connectivity"]
        size = len(src)
        if np.any(src < 0) or np.any(src >= len(roots)) or np.any(dst < 0) or np.any(dst >= len(roots)):
            raise DataValidationError("connection index outside graph")
        if not np.array_equal(roots[src], arrays["Presynaptic_ID"]) or not np.array_equal(roots[dst], arrays["Postsynaptic_ID"]):
            raise DataValidationError("root ID and internal index disagree")
        if np.any(count <= 0) or np.any(count > INT32_MAX):
            raise DataValidationError("synapse counts must be positive int32 values")
        if np.any((sign != 1) & (sign != -1)):
            raise DataValidationError("connection sign must be -1 or 1")
        if not np.array_equal(count * sign, weight):
            raise DataValidationError("signed count disagrees with count and sign")
        np.minimum.at(neuron_sign_min, src, sign.astype(np.int8))
        np.maximum.at(neuron_sign_max, src, sign.astype(np.int8))
        target = slice(offset, offset + size)
        pre[target], post[target], signed[target] = src, dst, weight
        pairs[target] = src.astype(np.uint64) * len(roots) + dst.astype(np.uint64)
        contacts += int(count.sum(dtype=np.int64))
        positive += int(np.count_nonzero(sign == 1))
        negative += int(np.count_nonzero(sign == -1))
        self_pairs += int(np.count_nonzero(src == dst))
        offset += size
    if offset != rows:
        raise DataValidationError("incomplete connectivity read")
    outgoing = neuron_sign_min != 2
    if np.any(neuron_sign_min[outgoing] != neuron_sign_max[outgoing]):
        raise DataValidationError("presynaptic neuron has inconsistent outgoing signs")
    pairs.sort()
    duplicate_pairs = int(np.count_nonzero(pairs[1:] == pairs[:-1]))
    del pairs
    if duplicate_pairs:
        raise DataValidationError("duplicate directed pairs in aggregated connectivity")
    if rows != expected["directed_pairs"] or contacts != expected["synaptic_contacts"]:
        raise DataValidationError("graph pair or synaptic-contact count differs from manifest")
    summary = {
        "schema": "neuroterrarium.graph-summary.v1", "profile": manifest["profile"],
        "connectome_version": manifest["connectome_version"], "neurons": len(roots),
        "connection_rows": rows, "directed_pairs": rows, "synaptic_contacts": contacts,
        "positive_pairs": positive, "negative_pairs": negative,
        "duplicate_pairs": duplicate_pairs, "self_pairs": self_pairs,
        "neurons_without_outgoing_edges": int(np.count_nonzero(~outgoing)),
        "annotations": annotation_summary,
        "source_sha256": {name: item["sha256"] for name, item in manifest["sources"].items()},
        "structure_bytes": roots.nbytes + pre.nbytes + post.nbytes + signed.nbytes,
        "load_seconds": perf_counter() - started,
    }
    for array in (roots, pre, post, signed):
        array.flags.writeable = False
    result = GraphData(roots, pre, post, signed, summary)
    result.indices(required_root_ids)
    return result
