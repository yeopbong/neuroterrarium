"""Versioned, evidence-backed neuron groups with exact integer ID resolution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ANNOTATION_SHA256 = "30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6"
_ROOT_PATTERN = re.compile(r"[1-9][0-9]{0,18}\Z")
_MAX_CONFIG_BYTES = 2_000_000
_INTERFACE_ROLES = {
    "sugar": "sensory_input", "mn9": "motor_readout",
    "lplc2": "sensory_input", "lc4": "sensory_input",
    "gf": "motor_readout", "dna01": "motor_readout", "dna02": "motor_readout",
}


class RegistryError(ValueError):
    """An interface, ID mapping or profile is incomplete or inconsistent."""


def _decimal_root(value: object) -> int:
    if not isinstance(value, str) or not _ROOT_PATTERN.fullmatch(value):
        raise RegistryError("Registry root IDs must be canonical decimal strings")
    root = int(value)
    if root > np.iinfo(np.int64).max:
        raise RegistryError("Root ID exceeds the supported signed 64-bit range")
    return root


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegistryError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def default_path() -> Path:
    """Find the installed profile or the profile in a source checkout."""
    installed = Path(sys.prefix) / "share/neuroterrarium/configs/interface-v783.json"
    if installed.is_file():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "configs/interface-v783.json"
    if checkout.is_file():
        return checkout
    raise RegistryError("Missing interface-v783.json; provide a verified profile path")


class Registry:
    """An immutable copy of a validated interface profile.

    Anatomical coordinates do not participate in identity, grouping or readout.
    Candidate status denotes a proposed interface, not successful simulation.
    """

    def __init__(self, document: Mapping[str, Any]) -> None:
        self._document = deepcopy(dict(document))
        self._validate()
        encoded = json.dumps(self._document, sort_keys=True, separators=(",", ":"))
        self.digest = hashlib.sha256(encoded.encode()).hexdigest()

    @classmethod
    def load(cls, path: str | Path | None = None) -> Registry:
        profile_path = default_path() if path is None else Path(path)
        try:
            with profile_path.open("rb") as stream:
                raw = stream.read(_MAX_CONFIG_BYTES + 1)
            if len(raw) > _MAX_CONFIG_BYTES:
                raise RegistryError("Interface profile exceeds size limit")
            document = json.loads(raw, object_pairs_hook=_unique_json)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RegistryError(f"Cannot load interface profile: {type(error).__name__}") from error
        if not isinstance(document, dict):
            raise RegistryError("Interface profile must be a JSON object")
        return cls(document)

    @property
    def document(self) -> dict[str, Any]:
        return deepcopy(self._document)

    @property
    def groups(self) -> dict[str, Any]:
        return deepcopy(self._document["groups"])

    def group(self, name: str) -> dict[str, Any]:
        try:
            return deepcopy(self._document["groups"][name])
        except KeyError as error:
            raise RegistryError(f"Unknown registered neuron group: {name}") from error

    def _validate(self) -> None:
        data = self._document
        if data.get("schema") != "neuroterrarium.interface.v1" or data.get("profile") != "v783":
            raise RegistryError("Unsupported interface schema or materialization profile")
        annotation = data.get("annotation")
        if not isinstance(annotation, dict) or (
            annotation.get("materialization") != 783
            or annotation.get("version") != "v2.1.0"
            or annotation.get("sha256") != ANNOTATION_SHA256
        ):
            raise RegistryError("Annotation version or checksum does not match v783")
        groups = data.get("groups")
        if not isinstance(groups, dict) or not groups:
            raise RegistryError("The registry has no neuron groups")
        if set(groups) != set(_INTERFACE_ROLES):
            raise RegistryError("Required interface groups are missing or unknown")
        evidence = data.get("evidence", {})
        seen: set[int] = set()
        for name, group in groups.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not isinstance(group, dict):
                raise RegistryError("Invalid neuron group")
            if group.get("role") != _INTERFACE_ROLES[name]:
                raise RegistryError(f"Incorrect interface role for group {name}")
            neurons = group.get("neurons")
            if not isinstance(neurons, list) or not neurons:
                raise RegistryError(f"Empty neuron group: {name}")
            for neuron in neurons:
                if not isinstance(neuron, dict):
                    raise RegistryError(f"Invalid neuron record in {name}")
                root = _decimal_root(neuron.get("root_id"))
                if root in seen:
                    raise RegistryError(f"Duplicate registered root ID: {root}")
                seen.add(root)
                if neuron.get("side") not in {"left", "right", "unknown", "center"}:
                    raise RegistryError(f"Missing or invalid side for root {root}")
                if not neuron.get("type") or neuron.get("type_field") not in {
                    "cell_type", "hemibrain_type"
                }:
                    raise RegistryError(f"Missing annotated type for root {root}")
                if neuron.get("role") not in {"sensory_input", "motor_readout"}:
                    raise RegistryError(f"Unknown interface role for root {root}")
                if neuron["role"] != group["role"]:
                    raise RegistryError(f"Conflicting interface role for root {root}")
                references = neuron.get("evidence")
                if not isinstance(references, list) or not references or any(
                    reference not in evidence for reference in references
                ):
                    raise RegistryError(f"Missing evidence for root {root}")
        for migration in data.get("migrations", []):
            for field in ("source_root_id", "root_id", "supervoxel_anchor"):
                _decimal_root(migration.get(field))
            if migration.get("root_id") not in {str(root) for root in seen}:
                raise RegistryError("Migration target is not registered")
            if migration.get("method") != "identical_supervoxel_anchor":
                raise RegistryError("Unsupported root migration method")

    @staticmethod
    def _graph_index(root_ids: Iterable[int]) -> dict[int, int]:
        index: dict[int, int] = {}
        for position, value in enumerate(root_ids):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise RegistryError("Graph root IDs must be integers, never floats or strings")
            root = int(value)
            if root <= 0 or root > np.iinfo(np.int64).max:
                raise RegistryError("Graph contains an out-of-range root ID")
            if root in index:
                raise RegistryError(f"Duplicate graph root ID: {root}")
            index[root] = position
        return index

    def resolve(self, root_ids: Iterable[int], *, profile: str = "v783") -> dict[str, np.ndarray]:
        """Resolve each registered root to its actual position in the loaded graph."""
        if profile != self._document["profile"]:
            raise RegistryError("Graph and interface materialization versions differ")
        index = self._graph_index(root_ids)
        resolved: dict[str, np.ndarray] = {}
        for name, group in self._document["groups"].items():
            roots = [int(neuron["root_id"]) for neuron in group["neurons"]]
            missing = [str(root) for root in roots if root not in index]
            if missing:
                raise RegistryError(f"Graph is missing {name} neurons: {', '.join(missing)}")
            indices = np.asarray([index[root] for root in roots], dtype=np.int64)
            indices.flags.writeable = False
            resolved[name] = indices
        return resolved

    def resolve_sides(self, root_ids: Iterable[int], *, profile: str = "v783") -> dict[str, np.ndarray]:
        """Split resolved groups by recorded biological side, including unknown."""
        resolved = self.resolve(root_ids, profile=profile)
        result: dict[str, np.ndarray] = {}
        for name, group in self._document["groups"].items():
            sides = [neuron["side"] for neuron in group["neurons"]]
            for side in sorted(set(sides)):
                selected = resolved[name][np.asarray([value == side for value in sides])]
                selected.flags.writeable = False
                result[f"{name}_{side}"] = selected
        return result

    def verify_annotations(self, path: str | Path) -> dict[str, int]:
        """Check every registered identity against the exact frozen source bytes."""
        source = Path(path)
        try:
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != self._document["annotation"]["sha256"]:
                raise RegistryError("Annotation source checksum mismatch")
            registered = {
                node["root_id"]: node
                for group in self._document["groups"].values()
                for node in group["neurons"]
            }
            verified: set[str] = set()
            with source.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    root = row["root_id"]
                    if root not in registered:
                        continue
                    node = registered[root]
                    if root in verified:
                        raise RegistryError(f"Duplicate annotated root ID: {root}")
                    if row[node["type_field"]] != node["type"] or row["side"] != node["side"]:
                        raise RegistryError(f"Annotation identity mismatch for root {root}")
                    if row["top_nt"] != node["top_nt"]:
                        raise RegistryError(f"Neurotransmitter annotation mismatch for root {root}")
                    verified.add(root)
            if len(verified) != len(registered):
                raise RegistryError("Registered neurons are absent from the source annotations")
            return {"registered_neurons": len(registered), "verified_neurons": len(verified)}
        except (OSError, KeyError, UnicodeDecodeError) as error:
            raise RegistryError(f"Cannot verify annotation source: {type(error).__name__}") from error


def resolve(root_ids: Iterable[int], *, profile: str = "v783") -> dict[str, np.ndarray]:
    """Resolve the default frozen interface against a loaded graph."""
    return Registry.load().resolve(root_ids, profile=profile)
