# Third-party attribution

The neural equations and reference implementation follow Philip Shiu and Nico
Spiller's [Drosophila brain model](https://github.com/philshiu/Drosophila_brain_model),
revision `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`, under MIT. Copyright (c) 2023
Philip Shiu and Nico Spiller. Their complete permission and warranty notice is
retained in `src/neuroterrarium/reference.py`.

FlyWire connectivity, annotations, interface cell records, and derived graph
assets, recorded neural activity and accompanying replay data follow [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
as specified by the [FlyWire public-data guidelines](https://flywire.ai/guidelines).
The software's MIT license does not relicense these data. Source records,
revisions, checksums, selection rules, and credits are in the data and interface
manifests. The official metadata for Zenodo 10676866 separately lists CC BY 4.0;
this project preserves that source distinction and follows FlyWire's
noncommercial condition for its data assets.

Please cite the computational model by [Shiu et al. (2024)](https://doi.org/10.1038/s41586-024-07763-9),
and FlyWire by [Dorkenwald et al. (2024)](https://doi.org/10.1038/s41586-024-07558-y)
and [Schlegel et al. (2024)](https://doi.org/10.1038/s41586-024-07686-5).
Underlying imaging, synapse detection, and transmitter-prediction citations
appear in `configs/data-v783.json`; visual and descending-neuron functional
citations appear in `configs/interface-v783.json`.

The alternative fly-brain implementation was consulted for engineering context.
Its GPL-2.0 implementation is not incorporated into this package.
