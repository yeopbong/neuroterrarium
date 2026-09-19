"""Corruption and identity tests on explicitly synthetic file fixtures.

Synthetic graphs exist only in temporary test directories. The full profile
has separate acceptance evidence from loading every row of the real files.
"""

import csv
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from neuroterrarium.data import (
    DataValidationError, load_graph, parse_root_id, root_ids_from_wire,
    root_ids_to_wire, sha256_file,
)


# Adjacent identifiers above 2**53 detect accidental floating-point conversion.
ROOTS = [720575940660219265, 720575940660219266, 720575940660219267]


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "nodes.csv").write_text(
        ",Completed\n" + "".join(f"{root},True\n" for root in ROOTS), encoding="utf-8"
    )
    columns = {
        "Presynaptic_ID": [ROOTS[1], ROOTS[0]],
        "Postsynaptic_ID": [ROOTS[2], ROOTS[1]],
        "Presynaptic_Index": [1, 0], "Postsynaptic_Index": [2, 1],
        "Connectivity": [3, 5], "Excitatory": [-1, 1],
        "Excitatory x Connectivity": [-3, 5], "__index_level_0__": [900, 4],
    }
    pq.write_table(pa.table({key: pa.array(value, type=pa.int64()) for key, value in columns.items()}), tmp_path / "edges.parquet")
    with (tmp_path / "annotations.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["root_id", "supervoxel_id", "cell_type", "hemibrain_type", "side", "top_nt"])
        for index, root in enumerate(ROOTS):
            writer.writerow([str(root), str(root - 100), "" if index == 0 else "fixture_type", "", "na", ""])
    manifest = {
        "schema": "neuroterrarium.data.v1", "profile": "synthetic-test-fixture",
        "connectome_version": "fixture", "sources": {},
        "expected": {"neurons": 3, "connection_rows": 2, "directed_pairs": 2, "synaptic_contacts": 8},
    }
    for key, filename in {"completeness": "nodes.csv", "connectivity": "edges.parquet", "annotations": "annotations.tsv"}.items():
        path = tmp_path / filename
        manifest["sources"][key] = {"filename": filename, "connectome_version": "fixture", "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, manifest_path


def rehash(dataset, source):
    directory, manifest_path = dataset
    manifest = json.loads(manifest_path.read_text())
    spec = manifest["sources"][source]
    path = directory / spec["filename"]
    spec.update(bytes=path.stat().st_size, sha256=sha256_file(path))
    manifest_path.write_text(json.dumps(manifest))


def mutate_column(dataset, column, values, dtype=pa.int64()):
    path = dataset[0] / "edges.parquet"
    table = pq.read_table(path)
    table = table.set_column(table.column_names.index(column), column, pa.array(values, type=dtype))
    pq.write_table(table, path)
    rehash(dataset, "connectivity")


def test_exact_precision_row_order_and_counts(dataset):
    graph = load_graph(*dataset, batch_size=1, required_root_ids=[str(ROOTS[0])])
    assert graph.root_ids.dtype == np.int64
    assert graph.pre.dtype == graph.post.dtype == graph.signed_counts.dtype == np.int32
    assert graph.root_ids.tolist() == ROOTS
    assert graph.pre.tolist() == [1, 0]
    assert graph.post.tolist() == [2, 1]
    assert graph.signed_counts.tolist() == [-3, 5]
    assert graph.summary["directed_pairs"] == 2
    assert graph.summary["synaptic_contacts"] == 8
    assert graph.summary["annotations"]["unknown_in_graph"] == {"type": 1, "side": 3, "predicted_transmitter": 3}
    for array in (graph.root_ids, graph.pre, graph.post, graph.signed_counts):
        assert not array.flags.writeable
    assert graph.indices([str(ROOTS[2]), ROOTS[0]]).tolist() == [2, 0]


def test_json_boundary_uses_strings_and_keeps_adjacent_ids_distinct():
    encoded = json.dumps(root_ids_to_wire(ROOTS))
    decoded = json.loads(encoded)
    assert decoded == ["720575940660219265", "720575940660219266", "720575940660219267"]
    assert root_ids_from_wire(decoded) == ROOTS
    with pytest.raises(DataValidationError, match="decimal strings"):
        root_ids_from_wire(ROOTS)


@pytest.mark.parametrize("value", [float(ROOTS[0]), True, "7.20575940660219265e17", "720575940660219265.0", "0720575940660219265", " 720575940660219265", -1, 2**63])
def test_invalid_or_rounded_root_identifiers_fail(value):
    with pytest.raises(DataValidationError):
        parse_root_id(value)


def test_same_length_corruption_fails_checksum(dataset):
    path = dataset[0] / "nodes.csv"
    path.write_text(path.read_text().replace(str(ROOTS[0]), str(ROOTS[0] + 9)))
    with pytest.raises(DataValidationError, match="SHA-256"):
        load_graph(*dataset)


def test_truncated_data_fails_size_check(dataset):
    path = dataset[0] / "edges.parquet"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(DataValidationError, match="byte count"):
        load_graph(*dataset)


def test_missing_data_does_not_select_another_controller(dataset):
    (dataset[0] / "edges.parquet").unlink()
    with pytest.raises(DataValidationError, match="missing connectivity"):
        load_graph(*dataset)


def test_root_id_precision_mutation_in_parquet_fails_even_after_rehash(dataset):
    mutate_column(dataset, "Presynaptic_ID", [float(ROOTS[1]), float(ROOTS[0])], pa.float64())
    with pytest.raises(DataValidationError, match="exact int64"):
        load_graph(*dataset)


def test_wrong_integer_root_mapping_is_detected(dataset):
    mutate_column(dataset, "Presynaptic_ID", [ROOTS[1] + 1, ROOTS[0]])
    with pytest.raises(DataValidationError, match="internal index disagree"):
        load_graph(*dataset)


@pytest.mark.parametrize("indices", [[0, 0], [1, 3], [1, -1]])
def test_index_misalignment_or_out_of_range_is_detected(dataset, indices):
    mutate_column(dataset, "Presynaptic_Index", indices)
    with pytest.raises(DataValidationError, match="index"):
        load_graph(*dataset)


@pytest.mark.parametrize("counts", [[0, 5], [-3, 5], [2**31, 5]])
def test_nonpositive_or_overflowing_counts_fail(dataset, counts):
    mutate_column(dataset, "Connectivity", counts)
    with pytest.raises(DataValidationError, match="positive int32"):
        load_graph(*dataset)


def test_sign_is_independent_of_signed_count(dataset):
    mutate_column(dataset, "Excitatory", [1, 1])
    with pytest.raises(DataValidationError, match="signed count disagrees"):
        load_graph(*dataset)


def test_unknown_sign_is_not_guessed(dataset):
    mutate_column(dataset, "Excitatory", [0, 1])
    with pytest.raises(DataValidationError, match="sign must be"):
        load_graph(*dataset)


def test_nulls_fail_before_numpy_float_coercion(dataset):
    mutate_column(dataset, "Connectivity", [None, 5])
    with pytest.raises(DataValidationError, match="null connectivity"):
        load_graph(*dataset)


def test_duplicate_directed_pair_cannot_inflate_count(dataset):
    mutate_column(dataset, "Presynaptic_ID", [ROOTS[0], ROOTS[0]])
    mutate_column(dataset, "Postsynaptic_ID", [ROOTS[1], ROOTS[1]])
    mutate_column(dataset, "Presynaptic_Index", [0, 0])
    mutate_column(dataset, "Postsynaptic_Index", [1, 1])
    mutate_column(dataset, "Excitatory", [1, 1])
    mutate_column(dataset, "Excitatory x Connectivity", [3, 5])
    with pytest.raises(DataValidationError, match="duplicate directed"):
        load_graph(*dataset, batch_size=1)


def test_neuron_cannot_change_outgoing_sign_between_batches(dataset):
    mutate_column(dataset, "Presynaptic_ID", [ROOTS[0], ROOTS[0]])
    mutate_column(dataset, "Presynaptic_Index", [0, 0])
    with pytest.raises(DataValidationError, match="inconsistent outgoing signs"):
        load_graph(*dataset, batch_size=1)


def test_missing_registered_root_and_duplicate_registration_fail(dataset):
    with pytest.raises(DataValidationError, match="absent from graph"):
        load_graph(*dataset, required_root_ids=[ROOTS[0] + 100])
    with pytest.raises(DataValidationError, match="contain duplicates"):
        load_graph(*dataset, required_root_ids=[ROOTS[0], ROOTS[0]])


def test_missing_annotation_is_not_fabricated(dataset):
    path = dataset[0] / "annotations.tsv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    rehash(dataset, "annotations")
    with pytest.raises(DataValidationError, match="missing from annotations"):
        load_graph(*dataset)


def test_duplicate_node_is_not_silently_deduplicated(dataset):
    path = dataset[0] / "nodes.csv"
    path.write_text(path.read_text().replace(str(ROOTS[2]), str(ROOTS[1])))
    rehash(dataset, "completeness")
    with pytest.raises(DataValidationError, match="duplicate completeness"):
        load_graph(*dataset)


def test_version_mismatch_fails_before_loading(dataset):
    manifest = json.loads(dataset[1].read_text())
    manifest["sources"]["annotations"]["connectome_version"] = "630"
    dataset[1].write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="version mismatch"):
        load_graph(*dataset)


def test_untrusted_manifest_cannot_reference_parent_file(dataset):
    manifest = json.loads(dataset[1].read_text())
    manifest["sources"]["annotations"]["filename"] = "../outside.tsv"
    dataset[1].write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="simple relative"):
        load_graph(*dataset)


def test_wrong_synaptic_contact_total_does_not_pass(dataset):
    manifest = json.loads(dataset[1].read_text())
    manifest["expected"]["synaptic_contacts"] = 9
    dataset[1].write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="synaptic-contact count"):
        load_graph(*dataset)
