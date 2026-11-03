# Data and reproducibility

## Image inputs

The benchmark uses two public ZTF Phase-I science images served by NASA/IPAC IRSA. Image data are excluded from version control and retain their separate data terms. Downloads are verified against the packaged SHA-256 manifest:

```bash
python scripts/download_data.py
export BLINDSPOT_FITS_ZIP=data/ztf
python -m blindspot.cli verify-data --fits-zip "$BLINDSPOT_FITS_ZIP"
python -m blindspot.cli smoke-fits --fits-zip "$BLINDSPOT_FITS_ZIP"
```

A directory containing the FITS files or an existing ZIP such as `data/ztf_images.zip` is accepted. The uncompressed size limit is 128 MiB per FITS file. Missing, duplicated, oversized or checksum-mismatched inputs are rejected.

| Field | FITS filename | SHA-256 |
| --- | --- | --- |
| `E4_r` | `ztf_20180411467847_000535_zr_c11_o_q3_sciimg.fits` | `51f8a813654603f26b7c6db47960479db520ac51de7cee87d6803b51729fab61` |
| `E5_g` | `ztf_20180325251921_000314_zg_c08_o_q2_sciimg.fits` | `a8cc7164b81be22e72f75d9bc4db892c974f65b28c15fe1fcc3ef7491f323a8c` |

## Installation and tests

Commands are run from the repository root. Python 3.10 or later is required. An isolated environment can be created in Bash on Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,agent]" httpx
python -m ruff check src tests scripts examples
python -m ruff format --check src tests scripts examples
python -m pytest -ra
```

On Windows with Git Bash, environment creation and activation use `py -3.11 -m venv .venv` and `source .venv/Scripts/activate`. PowerShell environment variables use `$env:BLINDSPOT_FITS_ZIP="data/ztf"`.

Image-dependent tests run when `BLINDSPOT_FITS_ZIP` identifies valid inputs. Other tests use synthetic arrays, simulated clients and mock transports. No test requires a model API credential. The optional historical reproduction test additionally requires `BLINDSPOT_REFERENCE_ROOT` to identify its original frozen checkout; it is skipped when that source is unavailable.

A pinned Python 3.12 reproduction environment is provided in `requirements/reproduce-py312.txt`. Installation uses `python -m pip install -r requirements/reproduce-py312.txt`, followed by `python -m pip install --no-deps -e .`. This environment is separate from the compatibility ranges in `pyproject.toml`.

## Verify the published evaluation without model calls

The published collection is stored in `evidence/model_evaluation/` as `protocol.json`, `ledger.json` and `case_results.csv`. Reproduction uses the current implementation and the original image inputs:

```bash
python -m scripts.reproduce_evaluation --fits-zip "$BLINDSPOT_FITS_ZIP"
```

Successful execution reports `REPRODUCED: 36 cases, 108 responses, 0 API calls`. The command verifies the input seals, experimental settings, all request identities and recorded token accounting. It then recomputes every pilot observation, baseline feature, probe measurement and baseline policy from the verified FITS images. Exact agreement with all 36 recorded cases and all 108 CSV rows is required.

Outputs are written separately under `evidence/runs/publication-reproduction/`:

| Output | Purpose |
| --- | --- |
| `report.md` | Regenerated model-policy results and uncertainty summaries |
| `case_results.csv` | Regenerated per-case decisions, outcomes, usage and calculated costs |
| `reproduction.json` | Verification status, input seals, runtime versions and current implementation identities |

The publication inputs are not overwritten. A mismatch returns a nonzero exit status and no successful reproduction record. Source hashes in the original protocol identify the collection implementation; the reproduction record identifies the implementation used for remeasurement. Reproducing measured values does not establish byte identity with the original runtime, provider authenticity or invoice reconciliation.

## Comparative Meta model evaluation

Preparation measures the shared cases and freezes the model settings, randomized request schedule, source fingerprints and budget before inference:

```bash
python -m scripts.evaluate_models prepare --fits-zip "$BLINDSPOT_FITS_ZIP" --directory evidence/runs/model-evaluation --budget 10
```

An empty experiment directory is required. Preparation makes no model calls. Execution requires the optional agent dependencies, `MODEL_API_KEY`, and explicit authorization through `--live`:

```bash
python -m scripts.evaluate_models run --directory evidence/runs/model-evaluation --live
```

The API key is supplied through the process environment. `MODEL_BASE_URL` defaults to `https://api.meta.ai/v1`. `.env.example` lists the variable names; environment files are not loaded automatically. Credentials must remain outside version control and published experiment artifacts.

The three model profiles use one response per case, model-specific maximum reasoning and a common 8192-token completion limit. The shared allowance is USD 10 across up to 108 inference requests. Rates, reasoning settings and interpretation criteria are described in [Benchmark protocol](BENCHMARK_PROTOCOL.md#paired-evaluation-at-model-specific-maximum-reasoning).

Strict source and configuration identity checks apply to preparation, execution and resumption. Changed implementations require a newly prepared experiment; archived protocols cannot authorize new requests from different source. Completed records are reused. Unresolved reservations, identity discrepancies and stale lockfiles require inspection before continuation. SDK retries are disabled.

Reports for an experiment prepared with the current implementation can be regenerated without API requests:

```bash
python -m scripts.evaluate_models report --directory evidence/runs/model-evaluation
```

The standalone example `python examples/live_test.py --live` authorizes at most one request within its configured USD 0.05 allowance. Importing the example makes no request.

## Local artifacts and historical records

New benchmark outputs, standalone reproduction outputs and paired comparisons default to subdirectories of `evidence/runs/`. This directory is ignored by Git. Explicit paths remain available. Offline benchmark execution uses canned replies and is not evidence of model performance.

Earlier reference, comparison and prototype records remain separate under `evidence/`. Their stored paths, hashes, replies and numerical values are collection provenance. Reproduction of an earlier implementation requires that implementation and its corresponding inputs; the published three-model reproducer does not pool or reinterpret those historical experiments.

## Acknowledging ZTF

Both images were taken in 2018 (ZTF Phase I). Publications using these images should retain the relevant ZTF acknowledgement and cite Masci et al., *The Zwicky Transient Facility: Data Processing, Products, and Archive*, PASP:

> Based on observations obtained with the Samuel Oschin 48-inch Telescope at the Palomar Observatory as part of the Zwicky Transient Facility project. ZTF is supported by the National Science Foundation under Grant No. AST-1440341 and a collaboration including Caltech, IPAC, the Weizmann Institute for Science, the Oskar Klein Center at Stockholm University, the University of Maryland, the University of Washington, Deutsches Elektronen-Synchrotron and Humboldt University, Los Alamos National Laboratories, the TANGO Consortium of Taiwan, the University of Wisconsin at Milwaukee, and Lawrence Berkeley National Laboratories. Operations are conducted by COO, IPAC, and UW.

Data documentation and current acknowledgement guidance are available from [NASA/IPAC IRSA](https://irsa.ipac.caltech.edu/Missions/ztf.html).