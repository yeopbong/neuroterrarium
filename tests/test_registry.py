"""Check source identities and reject precision loss, stale versions and bad joins."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from neuroterrarium.registry import Registry, RegistryError


PROFILE = Path(__file__).resolve().parents[1] / "configs/interface-v783.json"


@pytest.fixture
def registry():
    return Registry.load(PROFILE)


@pytest.fixture
def roots(registry):
    return np.asarray([
        int(neuron["root_id"])
        for group in registry.groups.values()
        for neuron in group["neurons"]
    ], dtype=np.int64)


def test_steering_ids_match_explicit_v783_paper(registry):
    # Rayshubskiy et al., doi:10.7554/eLife.102230.3, EM reconstruction Methods.
    expected = {
        "dna01": {"left": "720575940627787609", "right": "720575940644438551"},
        "dna02": {"left": "720575940629327659", "right": "720575940604737708"},
    }
    for name, neurons in expected.items():
        assert {row["side"]: row["root_id"] for row in registry.group(name)["neurons"]} == neurons
    assert all(row["type_field"] == "cell_type" for row in registry.group("dna01")["neurons"])
    assert "720575940618167579" not in str(registry.group("dna01"))


def test_actual_frozen_population_and_legacy_migrations(registry):
    assert {name: len(group["neurons"]) for name, group in registry.groups.items()} == {
        "sugar": 21, "mn9": 2, "lplc2": 210, "lc4": 104, "gf": 2, "dna01": 2, "dna02": 2,
    }
    assert all(row["side"] == "left" for row in registry.group("sugar")["neurons"])
    assert {
        (row["source_root_id"], row["root_id"], row["supervoxel_anchor"])
        for row in registry.document["migrations"]
    } == {
        ("720575940620900446", "720575940639259967", "78678166201187069"),
        ("720575940645521262", "720575940618238523", "78536329000155754"),
    }


def test_resolution_uses_roots_not_rows(registry, roots):
    shuffled = roots[::-1].copy()
    resolved = registry.resolve(shuffled)
    for name, indices in resolved.items():
        assert [str(value) for value in shuffled[indices]] == [
            row["root_id"] for row in registry.group(name)["neurons"]
        ]
        assert not indices.flags.writeable
    side_groups = registry.resolve_sides(shuffled)
    assert shuffled[side_groups["gf_left"]].tolist() == [720575940622838154]
    assert len(side_groups["lplc2_left"]) == 108
    assert len(side_groups["lplc2_right"]) == 102


def test_precision_mutation_is_rejected(registry, roots):
    with pytest.raises(RegistryError, match="never floats"):
        registry.resolve(roots.astype(np.float64))
    corrupted = roots.astype(np.float64).astype(np.int64)
    with pytest.raises(RegistryError):
        registry.resolve(corrupted)
    document = registry.document
    document["groups"]["gf"]["neurons"][0]["root_id"] = 720575940622838154
    with pytest.raises(RegistryError, match="decimal strings"):
        Registry(document)


def test_missing_duplicate_unknown_and_wrong_version_fail(registry, roots):
    with pytest.raises(RegistryError, match="missing"):
        registry.resolve(roots[1:])
    with pytest.raises(RegistryError, match="Duplicate graph"):
        registry.resolve(np.append(roots, roots[0]))
    with pytest.raises(RegistryError, match="Unknown registered"):
        registry.group("imaginary")
    with pytest.raises(RegistryError, match="versions differ"):
        registry.resolve(roots, profile="v630")
    document = registry.document
    document["annotation"]["sha256"] = "0" * 64
    with pytest.raises(RegistryError, match="checksum"):
        Registry(document)


def test_config_mutations_fail(registry, tmp_path):
    incomplete = registry.document
    del incomplete["groups"]["gf"]
    with pytest.raises(RegistryError, match="groups are missing"):
        Registry(incomplete)
    document = registry.document
    document["groups"]["gf"]["neurons"].append(deepcopy(document["groups"]["gf"]["neurons"][0]))
    with pytest.raises(RegistryError, match="Duplicate registered"):
        Registry(document)
    path = tmp_path / "配置.json"
    path.write_text('{"schema": "a", "schema": "b"}')
    with pytest.raises(RegistryError, match="Duplicate JSON"):
        Registry.load(path)
    path.write_text('malformed')
    with pytest.raises(RegistryError, match="Cannot load"):
        Registry.load(path)
    path.write_text(json.dumps([]))
    with pytest.raises(RegistryError, match="JSON object"):
        Registry.load(path)


def test_registry_does_not_expose_mutable_internal_state(registry, roots):
    original = registry.digest
    external = registry.groups
    external["sugar"]["neurons"].clear()
    assert len(registry.resolve(roots)["sugar"]) == 21
    assert registry.digest == original


def test_corrupt_annotation_bytes_fail(registry, tmp_path):
    path = tmp_path / "annotation.tsv"
    path.write_text("root_id\tside\n720575940622838154\tleft\n")
    with pytest.raises(RegistryError, match="source checksum mismatch"):
        registry.verify_annotations(path)
