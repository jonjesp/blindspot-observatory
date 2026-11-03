# Architecture

## Principle: one source of numbers

Deterministic tools produce the measured counts and reported numerical outcomes. The model may
**choose** probes, declare an ambiguity, or abstain. Written justifications remain unverified
hypotheses and never overwrite measured values. Ground-truth causes live in the private case
generator, outside model requests.

## Data flow

1. Verify the original FITS identities and extract 128 by 128 pixel patches.
2. Generate controlled conditions, inject artificial sources, and calculate pilot measurements and all holdout probe outcomes once per case.
3. Build a strictly whitelisted public observation. Hidden causes and precomputed probe outcomes stay outside model requests.
4. Freeze requests and settings, reserve the estimated cost before each call, and record the model's probe selection, ambiguity, or abstention.
5. Replay selected probes and fixed-order continuation on the shared outcomes. Score intervention closures and control abstentions, attributing direct closures separately.
6. Preserve replies, request identities, usage, and provenance; derive the case table and statistical report from those records.

## Modules

| Module | Responsibility |
|---|---|
| `config` | Loads and validates typed detector and experiment settings from `configs/*.yaml` |
| `datasets.ztf` | Checksum-verified FITS decoding, patch extraction |
| `injection.psf` | Gaussian PSF, injection, train/holdout position grids |
| `detection.detector` | Median background, MAD noise, matched-filter peaks |
| `metrics.completeness` | One-to-one matching, Wilson interval, gain and its lower bound |
| `agent.schemas` | Decision JSON schema and validator |
| `agent.observation` | Closed whitelist of what the model may see |
| `agent.muse` | Request building, cost cap, reservation-first ledger |
| `agent.client` / `agent.offline` | Live client factory / network-free test client |
| `benchmark.design` | 36 non-overlapping cases, classes shuffled per exposure |
| `benchmark.interventions` | **Private.** Condition A to B: noise, PSF width, flux; strong and minor magnitudes |
| `benchmark.measurement` | Deterministic measurements on one arm: recovered counts, source-profile diagnostics |
| `benchmark.observations` | Builds the public observation, per-probe baseline features and the holdout probe measurements |
| `benchmark.generator` | **Private.** Deterministic causes, dominant cause and masking for each planned case |
| `benchmark.calibration` | Runs every case on the real images and summarises cause separation |
| `benchmark.baselines` | `random`, `heuristic`, `elimination` probe orders |
| `benchmark.scoring` | Probe-count scoring with the closure rule |
| `provenance.ledger` | Exclusive transaction locks, atomic JSON writes and SHA-256 digests |
| `scripts.evaluate_models` | Frozen three-model protocol, shared budget, request scheduling, and offline validation |
| `scripts.compare_models` | Archived paired-comparison workflow and reusable reservation-first request accounting |
| `scripts.reproduce_evaluation` | Independent publication remeasurement and case-table verification |
| `scripts.model_report` | Derived model-policy metrics, conditional bootstrap intervals, Markdown report, and CSV table |

## Contracts

* **Decision**: `{"decision": "probe|ambiguous|abstain", "probes": [...], "justification": "..."}`.
  One, two or zero distinct probes from `probe_sky`, `probe_psf`, `probe_flux`.
* **Observation**: only `pilot_measurements`, `indicators` and `available_probes`, each with a closed
  set of keys. An unlisted key is rejected, not filtered. Masked diagnostics are simply omitted.
* **Indicators** are relative changes `|B - A| / A` of three measurements (background noise, integrated
  amplitude, RMS radius of the source profile), one per probe. Missingness uses a separate random
  stream in every case class, independent of causes. An absent key must not encode the true cause.
* **Private/public split**: `interventions`, `measurement` and `generator` know the truth;
  `agent` must never import `blindspot.benchmark` (enforced by an AST test). Private means isolated
  from the prompt, not secret: the generator is deterministic and public.
* **Closure**: a probe closes a case when holdout gain >= `gain_threshold` AND the Wilson
  non-overlap lower bound of the gain is above zero. This operational rule has no guaranteed
  confidence coverage for dependent repeated-image trials.
* **Execution**: observations and probe outcomes determine actions and stopping. The hidden class
  is consulted only afterwards to score controls. Invalid replies receive the fixed-order continuation
  on every class, so they cannot be mistaken for a valid abstention.
* **Attribution**: `muse` remains the machine-readable combined-policy key for compatibility; terminal
  output labels it `muse+fallback`. `closed_without_fallback` reports the contribution of selected probes.
* **Evaluation scope**: all probe outcomes are precomputed, and one model call chooses the initial order.
  There is no model feedback loop after a probe, nor a measured runtime/injection saving.
* **Ledger**: reservation written before the call; repeated case returns the stored record; changed
  request raises; unresolved reservations count at worst-case cost. The seal catches accidental
  edits only; it is not keyed.
* **Multi-model provenance**: the evaluation records the requested and returned model identifiers,
  verified SDK base URL, response ID, token usage, reported reasoning-token count, and request duration.
  Missing reasoning counts remain unknown. These fields do not provide cryptographic provider attestation.
* **Numerical compatibility**: the aggregate reservation bound uses `math.fsum` and an absolute
  comparison tolerance of 1e-12 USD with zero relative tolerance. Discrepancies beyond that tolerance
  remain errors. Source identities, request identities, response identities, and seals remain strict.
* **Archived-source verification**: the offline report command can verify the exact recorded scripts
  in a separate directory and write regenerated outputs elsewhere. This option is unavailable to paid
  execution; live runs continue to require current source hashes matching their frozen protocol.

Every decision ledger is protected by an exclusive process lock covering reservation, request and
response persistence. Concurrent or reentrant writers are rejected before another request can be
sent. Atomic writes use unique temporary files. An interrupted process can leave a lockfile and an
unresolved reservation; inspection is required before resumption.

## Configuration

`src/blindspot/configs/detector.yaml` and `experiment.yaml` are validated at import. The plan hash pins
geometry and classes; it does not cover every experimental parameter. The freeze additionally
fingerprints all packaged Python/YAML/JSON files and intervention settings, so implementation changes
are detected directly from their content. A changed definition requires a new freeze.

The independent model evaluation pins its model profiles, supported reasoning levels, output-token
limit, rates, source fingerprints, and randomized schedule in `protocol.json`. Its shared allowance
is at most USD 10 across three models. The published collection is retained unchanged in
`evidence/model_evaluation/`; derived local audits are written under the ignored `evidence/runs/`.
`scripts.reproduce_evaluation` verifies the publication through fresh pixel measurements and
exact comparison of all recorded observations, probe outcomes, baseline policies and CSV rows.
Historical collection fingerprints remain unchanged. Reproduction records current source identities
and does not claim reconstruction of the original runtime. Paid execution retains strict source
and configuration checks.

## Extending

* **New probe**: add it to `agent.schemas.TOOLS`, give it a diagnostic in the observation
  whitelist, a feature for the baselines, and the restoration logic in `benchmark.interventions`.
* **New baseline**: add the name to `benchmark.baselines.METHODS` and a branch in `policy_order`.
* **New public measurement**: extend the whitelist in `agent.observation` only, then re-freeze.
