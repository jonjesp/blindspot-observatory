# Results and limitations

## Summary

BlindSpot Observatory evaluates probe selection for auditing synthetic-source recovery in astronomical images. Deterministic tools produce the measurements, execute the probe policies, and record the resulting traces and costs.

The published three-model evaluation completed 108 requests. Spark 1.1, Spark 1.3, and Wonder each produced 35 valid decisions out of 36. With fixed-order continuation, they closed 32/32 intervention cases and correctly abstained on 4/4 controls. Spark 1.1 used 39 intervention probes, Wonder 40, Spark 1.3 43, and elimination 41. No model met the prespecified superiority criterion. These records do not establish a model advantage over elimination, equivalence, a general ranking, an end-to-end runtime improvement, an actual reduction in source injections, or a natural astrophysical discovery.

The findings concern the prototype detector and controlled interventions on two ZTF exposures, E4_r and E5_g. They do not evaluate the official ZTF production pipeline or measure the completeness of a production catalogue. Current and historical experiments remain distinct throughout this report.

## Experimental scope and design

- 36 cases: 16 single-cause cases, 16 two-cause cases, and 4 controls without intervention.
- Three available probes: `probe_sky`, `probe_psf`, and `probe_flux`.
- One response per model and case. The published comparison uses `xhigh` for Spark 1.1 and `max` for Spark 1.3 and Wonder, with a common 8192-token completion cap. The earlier reference uses `minimal` and 3072 tokens and remains a separate experiment.
- Models receive pilot measurements and available indicators; hidden causes and probe outcomes are excluded from their requests.
- A valid decision selects one probe, two ordered probes for an ambiguous decision, or no probes for an abstention.
- If the selected probes do not close the case, automatic fallback tries the remaining probes in the fixed order sky, PSF, flux. An explicit abstention stops execution.
- The comparison replays policies on previously calculated probe outcomes. It does not run a new injection campaign for each policy.

Closure requires a recovery gain of at least 0.10, or 10 percentage points, and a post-probe Wilson lower bound above the pre-probe Wilson upper bound. Execution stops at the first accepted closure. For controls, a correct decision is an abstention without executing probes.

## Published three-model evaluation

The complete live collection contains 36 requests for each of `muse-spark-1.1`, `muse-spark-1.3`, and `muse-wonder-1.0`. All models were present in the account catalog, all 108 response identifiers are distinct, and the returned model identities match the requested identities. Every record identifies the SDK base URL as `https://api.meta.ai/v1`. No unresolved reservation remains. These recorded identities are not cryptographic provider attestations.

The experiment was prepared and executed with Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, PyYAML 6.0.3, and OpenAI SDK 3.42.1. No Git revision was recorded. The frozen protocol retains packaged-source and evaluation-script fingerprints, the randomized schedule, shared measurements, rates, and request hashes. Published inputs are preserved byte-for-byte under `evidence/model_evaluation/`.

### Policy performance and attribution

| Policy | Valid replies | Probes in 32 interventions | Mean per intervention | Closures without continuation | Correct abstentions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | Not applicable | 66 | 2.06250 | Not applicable | 0/4 |
| Heuristic | Not applicable | 48 | 1.50000 | Not applicable | 4/4 |
| Elimination | Not applicable | 41 | 1.28125 | Not applicable | 4/4 |
| Spark 1.1 with continuation | 35/36 | 39 | 1.21875 | 28/32 | 4/4 |
| Spark 1.3 with continuation | 35/36 | 43 | 1.34375 | 27/32 | 4/4 |
| Wonder with continuation | 35/36 | 40 | 1.25000 | 27/32 | 4/4 |

Every policy closes 32/32 interventions. First-probe closures are 28/32 for Spark 1.1, 25/32 for Spark 1.3, 27/32 for Wonder, and 25/32 for elimination. The model-selected probes alone close 28, 27, and 27 interventions, respectively. The remaining closures require fixed-order continuation. Combined-policy success must not be attributed entirely to model-selected probes.

Continuation occurs in F03, F15, F16, and F18 for Spark 1.1; F12, F15, F18, F23, and F33 for Spark 1.3; and F03, F08, F15, F18, and F33 for Wonder. All four controls receive explicit valid abstentions.

### Response failures

| Model and case | Recorded condition | Interpretation under the frozen protocol |
| --- | --- | --- |
| Spark 1.1, F16 | `finish_reason: stop`; justification length 364 characters | Invalid decision because the maximum permitted length is 350 characters |
| Spark 1.3, F23 | `finish_reason: length`; 8192 completion tokens, including 8190 reasoning tokens; no visible decision | Invalid, token-limited response |
| Wonder, F08 | `finish_reason: length`; 8192 completion tokens, including 8190 reasoning tokens; no visible decision | Invalid, token-limited response |

These responses remain in the ledger with their usage and calculated cost. They were not retried, repaired, or reclassified after collection. The two truncated replies show that the requested maximum reasoning level can consume the fixed completion allowance before a usable decision is produced. The trial measures bounded maximum-reasoning conditions, not unrestricted model capability.

### Resource use

| Model | Prompt / cached tokens | Completion tokens | Reported reasoning tokens | Request median / p95, seconds | Calculated USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spark 1.1 | 10238 / 3842 | 90007 | 86323 | 20.20 / 51.45 | 0.391101 |
| Spark 1.3 | 13442 / 7712 | 119962 | 117143 | 18.09 / 54.00 | 0.518158 |
| Wonder | 10238 / 1740 | 102341 | 99573 | 15.63 / 63.54 | 2.081160 |

Total calculated expenditure is USD 2.990419, below the authorized USD 10 allowance and the original conservative reservation bound of approximately USD 8.853447. Reasoning tokens are included in completion tokens and are not billed twice. The latency measurements cover client HTTP/generation time, not the full scientific pipeline. Tokenizer accounting, caching, and service conditions can differ by model. No provider invoice or population latency distribution has been independently established.

### Prespecified comparison and uncertainty

The primary comparator is elimination. The criterion requires a complete live collection, valid replies, no reduction in closures or correct abstentions, and an upper confidence bound below zero for the paired probe-count difference. The calculation uses 10,000 paired case-bootstrap resamples with seed 1729. Nominal Bonferroni adjustment covers three primary comparisons.

| Difference against elimination | Mean, probes per intervention | 95% interval | 98.33% interval |
| --- | ---: | --- | --- |
| Spark 1.1 | -0.06250 | [-0.28125, +0.12500] | [-0.31250, +0.15625] |
| Spark 1.3 | +0.06250 | [-0.09375, +0.25000] | [-0.12500, +0.28125] |
| Wonder | -0.03125 | [-0.15625, +0.09375] | [-0.18750, +0.12500] |

**No model meets the criterion.** Each has an invalid response, and every interval against elimination includes zero. Spark 1.1 has the lowest observed model-policy probe count and calculated API cost, but its two-probe difference against elimination is not conclusive evidence of superiority.

| Exploratory paired difference | Mean, probes per intervention | 95% interval |
| --- | ---: | --- |
| Spark 1.1 minus Spark 1.3 | -0.12500 | [-0.37500, +0.09375] |
| Spark 1.1 minus Wonder | -0.03125 | [-0.18750, +0.09375] |
| Spark 1.3 minus Wonder | +0.09375 | [-0.06250, +0.31250] |

The model-to-model intervals are descriptive and unadjusted. They establish neither a winner nor equivalence. One response per model and case does not estimate model-repeat variability. All intervals remain conditional on the calibrated cases and two exposures; they are not population coverage guarantees.

### Offline verification and numerical compatibility

The protocol and ledger seals, all 108 request identities, returned model identities, per-request token costs, and complete case table were checked. The supplied Markdown report and CSV were regenerated from the ledger and matched their recorded contents after newline normalization. Stored physical measurements and baseline features match the preceding Windows image validation.

The aggregate reservation bound is compared with an absolute tolerance of 1e-12 USD and zero relative tolerance, using `math.fsum` for accumulation. The recorded value, seals and request identities remain unchanged.

The [reproduction procedure](DATA.md#verify-the-published-evaluation-without-model-calls) recomputes the 36 published cases from verified FITS images, validates the recorded requests and accounting, and regenerates a matching 108-row case table. Outputs record the current implementation identities. This verifies numerical reproducibility without asserting byte identity with the historical collection implementation.

## Archived reference benchmark results

All policies close the 32 intervention cases, counting the probes actually executed by each policy, including automatic fallback where applicable.

| Policy | Probes in 32 interventions | Mean per intervention | Closure with first probe | Correct abstentions | Probes in controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 66 | 2.0625 | 12/32 | 0/4 | 12 |
| Heuristic | 48 | 1.5000 | 23/32 | 4/4 | 0 |
| Elimination | 41 | 1.28125 | 25/32 | 4/4 | 0 |
| Spark with fallback | 41 | 1.28125 | 26/32 | 4/4 | 0 |
| Wonder with fallback | 43 | 1.34375 | 26/32 | 3/4 | 3 |

These means exclude the four controls. Across all 36 cases, Spark and `elimination` each execute 41 probes, while Wonder executes 46.

### Attribution and calculated cost

| Attribution and cost | Spark | Wonder |
| --- | ---: | ---: |
| Valid responses | 36/36 | 36/36 |
| Closures from selected probes, without fallback | 28/32 | 27/32 |
| Intervention cases using fallback | 4 | 5 |
| Controls using fallback | 0 | 1 |
| Calculated API cost, USD | 0.0642064 | 0.2375688 |

Wonder costs approximately 3.70 times the recorded Spark cost, based on token counts and configured rates. These are ledger calculations, not reconciled provider invoices. API cost does not represent the project's total computing cost.

The 32 closures achieved by each model with fallback must not be attributed exclusively to the model-selected probes.

## Archived Wonder improvement criterion and uncertainty

The Wonder plan specifies `elimination` as the primary comparator. Improvement requires a complete live run, an upper confidence bound below zero for the mean probe-count difference, and no reduction in closures or correct abstentions.

Paired bootstrap intervals are reported across the 32 intervention cases, using 5000 resamples and seed 1729. A negative difference favours Wonder.

| Difference: Wonder minus comparator | Mean, probes per case | 95% interval |
| --- | ---: | ---: |
| Elimination | +0.06250 | [-0.06250, +0.18750] |
| Spark | +0.06250 | [-0.06250, +0.25000] |
| Heuristic | -0.15625 | [-0.43750, +0.12500] |
| Random | -0.71875 | [-1.06250, -0.34375] |

**The primary criterion was not met (`primary_gate_passed: false`).** The interval against `elimination` includes zero, and Wonder achieves 3/4 correct abstentions against 4/4. The observed advantage over random ordering does not establish an advantage over the primary comparator. An interval containing zero does not establish equivalence either.

The intervals are conditional on these cases and exposures. They do not quantify variability across repeated model responses or justify generalization to other exposures. Four controls are insufficient to estimate a population false-alarm rate precisely.

## Archived Wonder cases explaining the outcome

Compared with Spark, Wonder uses one fewer probe in F09, one additional probe in F18, and two additional probes in F33. The remaining 29 intervention cases have equal probe counts. The net difference is two additional probes for Wonder.

In control F17, nominal pilot recovery changes from 67/100 to 65/100 and does not trigger the statistical anomaly criterion. Wonder selects PSF; fallback eventually executes all three probes without closure. Spark, heuristic, and elimination correctly abstain. This case accounts for Wonder's three additional control probes.

Wonder uses fallback in F03, F12, F15, F18, and F33, as well as control F17. In F12, the sky probe improves recovery by 15 percentage points but fails the Wilson condition; rejecting it as a closure is consistent with the protocol.

These cases illustrate selection differences. The models' written justifications are hypotheses, not demonstrated causal diagnoses.

## Historical complete runs

The historical collection includes `evidence/history/benchmark_results.json` and `evidence/history/rep2.json`, marked live,
with 36 valid responses each. For the 32 intervention cases, total probes were:

| Policy | Run 1 | Run 2 |
|---|---:|---:|
| Random | 66 | 66 |
| Heuristic | 49 | 49 |
| Elimination | 38 | 38 |
| Muse plus fallback | 38 | 39 |

The combined policy closed 32/32 in both runs. The selected model probes alone closed 30/32 and
28/32; the remaining closures used fixed-order continuation. All four controls received explicit
abstentions. No invalid reply was recorded in these two complete runs, so the invalid-control
scoring defect found later does not change those particular totals.

The mean probe differences against elimination were 0 and +0.03125. The stored case-bootstrap
intervals include zero; these runs demonstrate neither superiority nor a formal equivalence result.
Both use the same cases and measurements: the second run is a model-repeat check, not 36 new cases.

These are policy replays on precomputed probes, not measured runtime or injection savings.
The reference experiment changes masking and execution safeguards, so these historical runs
do not validate it. Its calibration, frozen reference, reproduction report, and Wonder comparison are
archived separately under `evidence/`. Their findings are reported in the archived reference
sections above. Results from the two definitions are not pooled.

## Earlier prototype runs (not claims for the final benchmark)

| Run | Observation |
|---|---|
| E6 | About 33 % fewer **training** injections; total saving was 9 %. Not an accuracy gain. No LLM involved |
| E13 | Muse 3/3 on single causes with clear diagnostics; the visible deterministic heuristic also 3/3 |
| E14 | 8 valid and correct of 9 attempted calls; a deterministic elimination rule solves 12/12 after the fact |
| E15 | Muse optimal on 2/12 two-cause cases and valid JSON on 7/12; heuristic 6/12 |
| E16 | Two valid decisions with gains of 0 and +6 points, below the +10 required |
| E17-E18 | A deterministic fallback recovered sources on some conditions; not additional Muse wins |
| E19 | Fixed order 12 probes, Muse+fallback 13; 4/6 closures and one false closure each; cost US$0.0122 in the ledger |

The E19 audit records a recomputation from the per-case data in `evidence/E19_audit.json`. No run shows
reliable Muse superiority or a natural astrophysical discovery.

The E6 and E19 summaries are accompanied by audit JSON files. The other prototype figures are retained as historical report summaries; their original per-case inputs are not all included in the repository. They are not additional cases in the 36-case benchmark.

## Historical detector and generator checks

The historical audit reports the following synthetic-noise tests and measurements on the two original images. They describe the corresponding historical implementation and conditions.

* **Detector on white noise** (synthetic): 54 % recovery at SNR 5, 86 % at SNR 6, about 100 % at
  SNR 8, zero false detections in 200 pure-noise patches. The threshold formula is correct for
  white noise.
* **Detector on real noise**: over the 36 planned sites, recovery at *nominal* SNR 5 has median 0.31
  (range 0.17-0.43) on `E4_r` and 0.42 (0.12-0.51) on `E5_g`. The robust width of the matched-filter
  response exceeds the white-noise prediction by a median factor of 1.16 (`E4_r`) and 1.05
  (`E5_g`), up to 1.46. So "SNR 5" is a nominal label, not a 50 % completeness point.
* **A locally poor recovery region**: at site `F20` (`E5_g`) only 17 of 25 sources are recovered even at
  SNR 12. The missed sources sit where the local noise is 4.6-7.3 times the global estimate (halo
  of a very bright object), where the single global noise value overstates their significance.
* All 36 planned patches are finite, have a measurable background and noise, and do not overlap.

* **Historical generator calibration**: see [Benchmark protocol](BENCHMARK_PROTOCOL.md). With the chosen magnitudes the
  true probe is clearly separable (gain 0.30 to 0.71) and wrong probes show no gain. This makes the
  *benchmark* solvable; it says nothing yet about whether Muse solves it.

## Limitations

* Simplified detector; the background is not re-estimated after injection; sources are placed on
  integer pixels with no photon noise of their own; a Gaussian PSF of fixed width is assumed.
* Two exposures only. Batches within a patch share pixels and noise; source centres may repeat.
  Train and holdout grids separate centres, not every pixel in PSF footprints. Binomial interval
  coverage for the aggregate batches is not guaranteed; bootstrap intervals are conditional on cases.
* Indicators carry real cross-talk: noise and real bright objects perturb the profile measurements, and
  the historical calibration reports a weak PSF diagnostic (median change 0.09). Probes are exact interventions, so a probe on an
  unchanged factor shows only sampling noise.
* Anomalies are synthetic with known interventions. The agent's task is to pick the right probe,
  not to discover a physical cause.
* Costs come from the token ledger and the configured price list, not from provider billing.
* The earlier experiment masked only the true-cause diagnostic in selected single-cause cases (7/7 missing
  diagnostics identified that cause). The reference experiment removes this construction artifact; historical
  performance must not be claimed to generalise to the revised masking scheme.
* Probe outcomes are precomputed. The model selects an initial order without a model feedback loop after each probe; policy replay costs do not measure end-to-end runtime or actual injection savings.
* Four controls and two exposures do not establish population false-alarm rates, generalization to other exposures, or variability across repeated model responses.
* The ledger seal is an unkeyed SHA-256 digest: it detects accidental changes but is not a provider signature or protection against deliberate recomputation.

## Evidence and reproducibility

The retained machine-readable evidence is stored under `evidence/`, with comparison and historical records in their existing subfolders. The retained file contents, seals, numeric values, and embedded historical paths remain unchanged. Paths within a stored JSON can describe an earlier runtime location; rewriting those strings would alter the record.

The archived reproduction report records 36/36 matching reference cases and zero API calls. The record audit and image reproduction have different scopes: internal consistency does not independently establish provider authenticity or reproduce pixel measurements.

The archived comparison analysis reports checks of the plan and ledger seals, their links, reference identity, and all 36 requests. Changing only the Wonder model identifier to Spark recovers the reference request hashes. The recorded 108 probe outcomes, candidate policy sequences, summary metrics, intervals, and calculated costs are internally consistent. These checks are not provider signatures.

The sum of recorded Wonder request durations is 121.84 seconds. No comparable Spark latency or complete-pipeline timing is available to establish a speed improvement. Token-based costs use the configured historical rates and do not constitute provider billing reconciliation or a statement of current prices.

### Published three-model evidence

| File | Role |
| --- | --- |
| [model_evaluation/protocol.json](../evidence/model_evaluation/protocol.json) | Unchanged frozen observations, profiles, rates, schedule, and source identities |
| [model_evaluation/ledger.json](../evidence/model_evaluation/ledger.json) | Unchanged 108 provider-response records, token usage, reservations, and ledger seal |
| [model_evaluation/case_results.csv](../evidence/model_evaluation/case_results.csv) | Unchanged case-level table supplied with the completed experiment |

The generated report can be reproduced from the publication records after fresh measurement verification. The original protocol, ledger and case table remain the authoritative collection artifacts. Earlier reference and prototype records retain their separate definitions and limitations.

### Archived reference and comparison

| File | Role |
| --- | --- |
| [benchmark_freeze.json](../evidence/benchmark_freeze.json) | Frozen reference implementation, configuration, plan, and data identities |
| [calibration.json](../evidence/calibration.json) | Reference generator calibration |
| [benchmark_results.json](../evidence/benchmark_results.json) | Complete Spark reference and baseline policy results |
| [benchmark_ledger.json](../evidence/benchmark_ledger.json) | Spark request hashes, responses, usage, and calculated costs |
| [reproduction.json](../evidence/reproduction.json) | Supplied report recording reproduction of all 36 reference cases |
| [comparison/plan.json](../evidence/comparison/plan.json) | Frozen Wonder comparison settings and requests |
| [comparison/wonder_results.json](../evidence/comparison/wonder_results.json) | Wonder results and comparison summary |
| [comparison/wonder_ledger.json](../evidence/comparison/wonder_ledger.json) | Wonder request and cost ledger |

Publication reproduction requires the exact original images and validates the current implementation against every published case. Older archives require their own recorded implementations and are not reconstructed by this procedure.

### Historical artifacts

| File or path | Role |
| --- | --- |
| [E6_local_audit.json](../evidence/E6_local_audit.json) | Audit supporting earlier allocation results |
| [E19_audit.json](../evidence/E19_audit.json) | Audit supporting the earlier six-case comparison |
| [generator_calibration.json](../evidence/generator_calibration.json) | Historical generator calibration |
| [history/benchmark_results.json](../evidence/history/benchmark_results.json) | First historical live run |
| [history/benchmark_ledger.json](../evidence/history/benchmark_ledger.json) | Corresponding call ledger |
| [history/rep2.json](../evidence/history/rep2.json) | Second historical live run, on the same cases |
| [history/rep2_ledger.json](../evidence/history/rep2_ledger.json) | Corresponding second-run ledger |

Historical live runs remain separate from the principal reference. The historical calibration is retained in `evidence/generator_calibration.json`.

### Local reproduction outputs

The root [README](../README.md) documents local testing. Image-based checks use verified FITS inputs, downloaded from IRSA or provided through a local archive such as `data/ztf_images.zip`. The optional historical comparison also requires the original source in an external directory. Configuration is described in [Data](DATA.md).

New CLI outputs default to `evidence/runs/`, which is ignored by Git. Live and offline benchmarks use separate `live/` and `dry_run/` subfolders; explicit output and ledger options remain available. Offline dry runs use canned responses and are not evidence of model performance.

The archived reference remains bound to its recorded implementation hashes. Source changes, including CLI output routing, require a new freeze for new runs. Auditing or reproducing the archived measurements requires their original source revision; the archived hashes are not replaced with hashes of edited code.

The published three-model evaluation and earlier records are reported separately. Local verification preserves curated evidence, with generated outputs written under the ignored local-run directory. The detailed protocol is documented in [Benchmark protocol](BENCHMARK_PROTOCOL.md).
