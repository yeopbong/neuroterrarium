# Data and provenance

The default `shiu-v783-full` profile uses all 138,639 neurons in the paired completeness file, all 15,091,983 directed pair rows, and 54,492,922 synaptic contacts obtained by summing the export's integer contact counts. These are three different quantities, calculated from the actual files by `data verify`; the pair export is not an individual-contact table. No weak edges, inhibitory connections or feedback loops are removed. The connectivity export has one unique directed pair per row; its stored pandas index is not a neuron index.

[The machine-readable manifest](../configs/data-v783.json) fixes every URL, revision, byte length, SHA-256 and access date. It also records schema checks and the exact filtering and aggregation rules. Downloads total 131,147,197 bytes. Preparation verifies each file before parsing; offline verification repeats the structural checks.

| Source | Frozen revision | Use |
|---|---|---|
| [Shiu model repository](https://github.com/philshiu/Drosophila_brain_model) | `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` | Paired `Completeness_783.csv` and `Connectivity_783.parquet`; original model semantics |
| [FlyWire annotations](https://github.com/flyconnectome/flywire_annotations) | `ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0`, tag `v2.1.0` | Exact root-ID annotation joins for version 783 |
| [FlyWire deposit](https://zenodo.org/records/10676866) | Version 783.0 | Dataset citation and depositor metadata; not an unverified replacement download |

The annotation table contains 139,255 roots. All execution neurons have a matching row; 616 additional annotated roots are not added to the neural model. Unknown annotations remain unknown. The loader rejects duplicate IDs, incompatible versions, wrong integer schemas, absent endpoints, inconsistent transmitter signs, zero counts and corrupted files. The sign supplied by the upstream model is retained, even when later annotations offer a different prediction.

Root IDs are Python integers and decimal strings in JSON. Internal indices follow the completeness file's physical row order. JavaScript floating-point numbers are never used to transport root IDs. The session format also encodes large PCG random-state integers as decimal strings so browser JSON round trips preserve future random draws.

Data preparation constructs a complete CSR cache in a separate process. The local application maps that cache read-only and creates independent neural state. Its 183,461,307 bytes of arrays preserve every source connection. [The storage profile](../configs/brain-storage-v1.json) pins the complete graph and summary digests; the cache also records original data, interface and implementation identities, plus each array's shape, type, byte length and checksum. Loading verifies these and the original source files. Missing storage can be prepared offline from those files; an incompatible or damaged existing cache fails explicitly and requires `data prepare --rebuild`. Rebuild preserves the replaced cache for inspection.

## Registered interface

[The interface registry](../configs/interface-v783.json) contains 343 actual cells: 21 labellar sugar inputs, 210 LPLC2, 104 LC4, two GF/DNp01, two MN9, two DNa01 and two DNa02 neurons. Each entry includes its decimal root ID, type, biological side, role and evidence. Loading resolves these IDs against the complete graph and checks the frozen annotation values. Missing cells are an error.

LPLC2 and LC4 are selected using the registry's explicitly named annotation fields. DNa01 requires special care: the current `cell_type` identifies the intended cells; an older `hemibrain_type` label also occurs on different DNae001 cells and is not treated as equivalent. Direct full-graph links include 189 LPLC2→GF pairs carrying 1,080 contacts and 104 LC4→GF pairs carrying 805 contacts.

The original model's default v630 is a separate data profile. This release executes v783 and does not claim to reproduce v630 paper numbers. Two necessary v630→v783 identity updates in the registry use a unique shared supervoxel and the same annotation anchor, with the old and new IDs preserved. No migration uses numerical proximity, row position or a similar name. Biological side annotations are retained where an old notebook's informal left/right names disagree.

## Separate code and data licenses

The project code is MIT. The Shiu reference implementation is MIT, copyright 2023 Philip Shiu and Nico Spiller; its full notice is preserved in the reference module. See [third-party notices](../THIRD_PARTY_NOTICES.md).

The FlyWire-derived graph and annotation assets are distributed under the noncommercial conditions in the [FlyWire public-data guidelines](https://flywire.ai/guidelines), [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The Zenodo depositor metadata lists CC BY 4.0; the manifest records that difference rather than silently treating the downloaded FlyWire-derived bundle as permissively licensed code. Recorded neural activity and accompanying replay data retain the same noncommercial conditions. The project MIT license does not relicense these data or remove required citations. The downloader identifies the data license before downloading.

Primary scientific citations are in the manifest, including [Shiu et al., Nature (2024)](https://doi.org/10.1038/s41586-024-07763-9), [Dorkenwald et al., Nature (2024)](https://doi.org/10.1038/s41586-024-07558-y), [Schlegel et al., Nature (2024)](https://doi.org/10.1038/s41586-024-07686-5), and the original acquisition, reconstruction and neurotransmitter-prediction work linked there.
