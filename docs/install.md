# Installation and offline use

NeuroTerrarium uses Python 3.12. Data preparation is an explicit network operation;
local controllers, interventions, snapshots and validation use local files. A
prepared model set contains nine independently trained policies. Training is not
part of application startup.

## From source

Use Python 3.12, Node.js 24 and pnpm **11.19.0** on your PATH. The checked source
installation used Python 3.12.14 and Node.js 24.19.0 on the macOS arm64 platform
described below. Start in the repository root and keep its `artifacts/models`
directory, which contains the nine prepared policies.

Run these commands in the same shell. The thread limits match the tested local
configuration:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd web
pnpm --version
pnpm install --frozen-lockfile --ignore-scripts
pnpm run build
cd ..
python3.12 scripts/stage_web.py
python3.12 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/neuroterrarium data fetch --data-dir data
.venv/bin/neuroterrarium app --data-dir data --models artifacts/models --open-browser
```

Check that `pnpm --version` prints `11.19.0` before continuing. Dependency
installation and the first data download may need internet access. Later launches
use the final command with the same thread settings. After source or browser
changes, rebuild, stage and reinstall the package.

## Prepare data

After the source installation above, or installing a built wheel into an isolated
environment, data preparation and a readiness check are:

```sh
.venv/bin/neuroterrarium data fetch --data-dir data
.venv/bin/neuroterrarium doctor --data-dir data --models artifacts/models
```

The three pinned v783 inputs total **131,147,197 bytes**. The downloader reports
the cache directory, license, byte progress and final SHA-256 verification. It
publishes a file only after its size and checksum match the manifest. Interrupted
transfers retain identified partial files and resume with a checked HTTP Range
response. `data fetch --restart` restarts matching partial downloads; it does not
overwrite an existing corrupted final file.

After downloading, preparation builds a checksummed sparse cache containing the
complete graph. This derived storage changes the memory layout, without pruning
nodes or edges. To prepare the cache from data already present, without a network
connection, use `neuroterrarium data prepare --data-dir data`. A corrupt cache
fails explicitly; `data prepare --rebuild` regenerates that identified cache from
the verified source files.

FlyWire data retain **CC BY-NC 4.0** terms. The application's MIT license does not
relicense those data. See [data sources and citations](data.md).

`doctor` reports resources available to its current process and verifies data
bytes, the sparse cache, model identities and checkpoint hashes. It actually loads the nine policies.
It returns `not_ready` with a nonzero exit code when assets are missing or invalid.
GPU availability can depend on process permissions; this report does not infer
hardware absence from an unavailable backend.

## Local application

The portable archive includes the compiled browser application and nine-model
directory. A plain wheel contains the application and browser assets; keep the
separately supplied `artifacts/models` directory when using a wheel or source
checkout. The local entry point is:

```sh
.venv/bin/neuroterrarium app --data-dir data --models artifacts/models --open-browser
```

The service binds to `127.0.0.1:8765` by default. Non-loopback hosts are rejected.
Keep the data and model directories when moving an installation. Data are not
downloaded and policies are not trained by `app`.

Full data integrity checks and bounded runtime validation have separate commands:

```sh
.venv/bin/neuroterrarium data verify --data-dir data
.venv/bin/neuroterrarium validate --data-dir data --models artifacts/models --output validation-run
```

`validate` runs the complete graph with the nine prepared models and checks
feeding, intervention and state restoration. This bounded run does not replace
the numerical, experimental, interface or long-duration release checks. Each
validation output directory is new, so previous evidence is retained.

## Portable macOS packaging

The packaging recipe uses the fixed CPython 3.12.14 arm64 standalone distribution
and the project's exact dependency wheels. The Python archive checksum and source
revision are recorded in `scripts/build_package.py`; wheel and license hashes
are recorded during preparation. The package keeps dependency license texts and
uses a relative launcher, without a system Python prerequisite.

The assembled archive was tested on **macOS 26.6.2, Apple M2 arm64, with 16 GiB
memory**, using the CPU neural backend. Its bundled interpreter and 40 pinned
dependencies require no system Python installation. Clean extraction, actual
full-graph and nine-policy execution, local HTTP resources, native-library
locations and the launcher after relocation all passed in paths containing
spaces and non-ASCII characters. A separate clean wheel installation passed its
eight required checks. Archive-specific hashes and logs accompany the
[release evidence](results.md). This is the tested installation platform; the
bundle is unsigned and not notarized.

`scripts/build_package.py prepare` is the only packaging operation that downloads
runtime assets or dependency wheels. `build` uses verified local inputs and
requires the frozen source digest, compiled browser resources and nine completed
models. The build checks for at least 3 GiB available memory and a 10 GiB free-disk
reserve before major stages.

The portable launcher's normal commands are `./neuroterrarium data fetch` for
preparation and `./neuroterrarium app --open-browser` for local use. Archive
availability and final validation status belong to the release's asset manifest.

## Verify an installed wheel

For a source build, compile `web/` and run `python scripts/stage_web.py` before
building the wheel. The staging script updates only its own marked generated
directory, rejects unexpected files, and removes obsolete generated assets.
Generated `src/neuroterrarium/static/` files are excluded from version control.

`scripts/install_check.py` creates a fresh virtual environment and empty working
directory with spaces and non-ASCII characters. With `--wheelhouse`, dependency
installation uses only the locally checksummed wheel cache. Without that option,
installation can download public dependencies; Linux first installs PyTorch from
its official CPU-wheel index.

After installation, fresh isolated Python processes verify that code and
configuration came from the installed environment, inspect the packaged HTML and
JavaScript, run `doctor`, verify the actual complete graph and execute the bounded
nine-model runtime checks. These probes reject outbound Python socket calls and
DNS resolution. This is a process-level network prohibition, not a claim that
the physical network connection was unplugged.

Every step records its exit code, elapsed time and log checksum. Public summaries
normalize local paths. A timeout, missing resource, failed check or interrupted
run cannot be reported as passed. Browser interaction, the 60-minute stability
run, public asset re-download and post-deployment checks remain separate required
release evidence.

`scripts/portable_check.py` checks the assembled archive separately: clean
extraction, the installed file inventory, the actual complete graph and nine
policies with outbound Python networking denied, local HTTP and JavaScript
resources, native library locations, and the launcher after relocation. Its
`portable-check.v1` result can pass only after every required command completes.
Use the recorded archive checksum to match these results to the downloaded bytes.
