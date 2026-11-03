# Benchmark protocol

This document specifies the experimental protocol. The supplied calibration, frozen Spark run,
reproduction report, and Wonder comparison are indexed in [Results and evidence](RESULTS.md#evidence-and-reproducibility).
Those reference records are distinct from the historical runs and the published three-model collection.

## Design

* 36 new cases: 18 per exposure (`E4_r`, `E5_g`), all non-overlapping and disjoint from every site
  used by earlier experiments. Per exposure: 2 no-intervention, 8 single-cause, 8 two-cause.
* Interventions on the anomalous condition B (see `benchmark.interventions`):

  | Probe | Factor degraded | Strong | Minor |
  |---|---|---|---|
  | `probe_sky` | extra noise, in units of the reference noise | 0.8 | 0.3 |
  | `probe_psf` | injected PSF width vs the detector's | x1.3 | x1.1 |
  | `probe_flux` | injected flux | x0.8 | x0.93 |

  A single cause is strong. In a two-cause case one cause is strong (dominant) and the other minor.
  Each case, including controls and two-cause cases, has probability 0.5 of one uniformly chosen
  diagnostic being masked. A separate random stream makes missingness independent of class and causes.
* Pilot: 4 batches of 25 injections per arm at two significance levels (nominal 6, bright 9).
* Probe: restore ONE factor; holdout recovery before and after (4 batches, n = 100), separately sampled
  source centres, same noise realisation. Centres can repeat across batches; PSF footprints overlap
  train footprints and all batches share the same underlying patch.
* Compared policies: Muse, `random`, `heuristic`, `elimination` (see `benchmark.baselines`).
  All see the same public observation.
* Model setup: JSON Schema output, `reasoning_effort=minimal`, no retries, one call per case.
* Execution never consults hidden case labels. An explicit valid abstention is final; otherwise
  selected probes are followed by fixed-order continuation, stopping on measured closure.
  Invalid responses cannot count as correct abstentions. Hidden classes are used only for scoring.
* `muse+fallback` is a combined policy. Closures from the selected probes without fallback are
  reported separately. All probe outcomes are precomputed; counts measure policy replay cost,
  not an observed reduction in computation time or total source injections.

## Metric

Primary: number of probes (maximum 3) until the holdout recovery gain is >= 0.10 **and** its
conservative lower bound is positive. Failing to close counts as a failure at its true cost.
Also reported: abstention accuracy on no-intervention cases, invalid replies at their real cost,
and case-bootstrap intervals conditional on these patches, plus descriptive results by exposure.
This is not a cluster bootstrap over exposures; with only two exposures no population claims are made.
Wilson non-overlap is an operational closure rule, not guaranteed coverage for dependent batches.

## Prediction to freeze before running

At equal or better closure and abstention than the best baseline, Muse uses fewer probes. If Muse
ties or loses, the value is described as traceability and tool orchestration, not as optimisation.

## Before an independent paid run

1. Verify the original FITS identities and the intended case geometry.
2. Review pilot, probe, and holdout sampling and the dependence limitations below.
3. Create and commit a new freeze for any changed definition; do not overwrite the supplied reference.
   The freeze fingerprints packaged implementation, runtime intervention settings, plan,
   configuration, prompt, and FITS identities.
4. Check current provider pricing and the configured budget. Route new artifacts to
   `evidence/runs/` with explicit output and ledger paths.

## Historical generator calibration

From `blindspot calibrate`; the full per-case record is `evidence/generator_calibration.json`.
The reference calibration is `evidence/calibration.json`; the older file remains historical.
The historical calibration does not validate the reference experiment.
Intervention strengths were tuned on these same sites: this is a calibrated synthetic benchmark,
not an out-of-sample test of the generator or a demonstration of natural-cause discovery.

| Class | Result |
|---|---|
| Single cause (16) | Recovery falls from 0.70 to 0.28. True probe gains 0.40 on average (minimum 0.30) and closes 16/16. Wrong probes: gain at most 0.05, 0 false closures in 32 |
| Two causes (16) | Dominant gain 0.39 (16/16 closed); minor gain 0.09 (2/16 closed). The largest-gain probe is the dominant one in 16/16 |
| No intervention (4) | 0 false closures; largest indicator 0.046, under the 0.10 baseline tolerance |

Median indicator by cause (single cause, unmasked cases): sky moves `background_mad_change` to 0.27,
flux moves `integrated_amplitude_change` to 0.26, but PSF moves `rms_radius_change` only to 0.09.
The PSF cause is therefore the hardest to see and sits at the baseline tolerance.

## Stop criteria

Do not run if a FITS checksum fails, a site overlaps an earlier one, or the smoke test fails.

## Paired evaluation at model-specific maximum reasoning

The `scripts.evaluate_models` entry point defines an independent evaluation of the same 36 case observations. It does not replace the archived minimal-reasoning reference or change the detector, intervention generator, decision schema, or scoring rules. Preparation measures the shared cases once and freezes the complete schedule before any model request.

| Model API identifier | Reasoning effort | Completion-token limit |
| --- | --- | ---: |
| `muse-spark-1.1` | `xhigh` | 8192 |
| `muse-spark-1.3` | `max` | 8192 |
| `muse-wonder-1.0` | `max` | 8192 |

The reasoning levels follow the [Meta reasoning documentation](https://dev.meta.ai/docs/reasoning). The common completion limit covers reasoning tokens and the visible answer together. Requests use temperature 1.0 and the existing strict decision schema. A truncated or malformed answer is retained as an invalid response; its usage remains charged and the same fixed-order continuation rule applies. Neither reasoning levels nor token limits are changed during a collection.

A model absent from the account's model catalog receives no inference requests and no synthetic score. Available models can complete their requests, while the report identifies the overall collection as incomplete when a selected API model is unavailable.

The schedule contains 108 requests: one response for each of the three models on every case. Case blocks are shuffled and model order is randomized within each block, using seed 20261031. Observations and numerical probe outcomes remain identical across models. Models see no FITS pixels, hidden causes, or precomputed outcomes. An 8192-token output cap makes these model-specific maximum-reasoning conditions bounded rather than unrestricted or matched-compute evaluations.

### Accounting and provenance

The experiment allowance is at most USD 10 in total across models. Each request reserves its full output-token allowance and a conservative input-token bound before transmission. Preparation rejects a protocol whose aggregate reservation bound exceeds the allowance. The reservation estimate uses uncached input rates; actual accounting uses reported cached tokens where available. Published rates are frozen from the [Meta pricing documentation](https://dev.meta.ai/docs/pricing-rate-limits), checked on 2026-11-02.

The shared ledger records the requested and returned model identities, verified SDK base URL, visible reply, token usage, reported reasoning-token count, and request duration. Reasoning tokens are part of completion tokens and are not charged twice. Missing reasoning counts remain unknown. Private reasoning text is not collected. SDK retries are disabled. A transport failure leaves an unresolved reservation; inspection is required before resumption, and the same request is not automatically billed again. A provider identity mismatch or an accounting discrepancy stops further inference.

These are local accounting controls, not a provider-enforced account limit or invoice reconciliation. A process lock prevents concurrent writers to the same experiment ledger. Protocol and ledger seals detect accidental edits; they are unkeyed and do not establish provider authenticity. Source fingerprints include the packaged benchmark and all four evaluation scripts. Every resumed request must match its frozen hash.

### Prespecified interpretation

Elimination is the primary comparator. For each model, a conditional improvement requires a complete live collection, valid replies, no reduction in intervention closures or correct control abstentions, and an upper bound below zero for the paired mean probe-count difference. The report uses 10,000 paired case-bootstrap resamples with seed 1729. Nominal Bonferroni adjustment across the three primary comparisons gives 98.33% intervals; 95% intervals are descriptive. Model-to-model comparisons are exploratory and do not define an additional winner-selection rule.

An interval containing zero does not establish equivalence. All results remain conditional on these calibrated synthetic interventions and two exposures. One response per case does not estimate model-repeat variability. Policy replay is separated from actual request latency and calculated token cost; no full-pipeline runtime or source-injection saving is inferred.

The derived outputs are one readable `report.md` and one `case_results.csv`. The two JSON files retain the frozen protocol and the request ledger for reproducibility. Execution instructions are provided in [Data and local tests](DATA.md#comparative-meta-model-evaluation).

## Published collection and reproduction

The completed three-model collection is retained under `evidence/model_evaluation/`. It contains 108 live requests, 105 valid decisions, no unresolved reservation, and calculated total expenditure of USD 2.990419. Spark 1.1 uses 39 intervention probes, Spark 1.3 uses 43, Wonder uses 40, and elimination uses 41. All combined model policies close 32/32 interventions and correctly abstain on 4/4 controls. No model meets the primary superiority criterion. Detailed attribution, token-limited replies, costs, and intervals are consolidated in [Results](RESULTS.md#published-three-model-evaluation).

The protocol, ledger and CSV retain the original collection values and identities. Publication reproduction uses verified FITS inputs and exact agreement of all 36 cases with their recorded pilot observations, baseline features, holdout probe outcomes and reference policies. The regenerated 108-row case table must match the publication. This procedure makes no model calls and writes separate local outputs.

Aggregate reservation arithmetic uses `math.fsum` and absolute tolerance 1e-12 USD with relative tolerance zero. The recorded bound, seals and request identities remain unchanged.

Historical source fingerprints identify collection provenance. The reproduction record identifies the current implementation and measured agreement; it does not assert that the original runtime has been reconstructed. Strict source checks remain mandatory for all new paid execution. Commands and output contracts are specified in [Data and reproducibility](DATA.md#verify-the-published-evaluation-without-model-calls).
