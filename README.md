# BlindSpot Observatory

BlindSpot Observatory evaluates language-model policies for diagnostic-probe selection in astronomical injection–recovery experiments. The benchmark compares model-selected probes with deterministic baselines on controlled interventions in two ZTF exposures. None of the evaluated models met the prespecified criterion for superiority over the elimination baseline.

![A real ZTF exposure, a 128×128 patch before and after injecting 25 artificial sources, and the benchmark's probe counts.](docs/assets/auditor-demonstration.png)

## The problem

A survey only sees what its detector catches. To measure what it **misses**, artificial sources are injected into real images and the fraction recovered is counted (*injection–recovery*). When recovery drops, the hard question is **why**: extra sky noise, blurrier optics (PSF), or fainter flux? Each cause calls for a different follow-up experiment, and every experiment costs computing time and expert attention.

## What the project does

1. Takes two **real ZTF exposures** and injects artificial point sources into 36 patches.
2. Degrades each patch with a hidden cause (noise, PSF width, flux, or two at once). Four patches are controls with nothing wrong.
3. Shows a **Muse model only the pilot numbers** (recovery counts and three indicators). The model selects one probe (`probe_sky`, `probe_psf`, `probe_flux`), requests two probes, or abstains.
4. **Measures everything with deterministic code.** The model performs no measurements or numerical calculations and never sees pixels, the hidden cause, or the probe outcomes. An automated test (AST check) enforces that the agent code cannot import the benchmark internals.
5. Records every call in a sealed ledger: request hash, response, tokens, cost, and a hard budget cap enforced by reserving the worst-case cost of each request before it is sent.

## Result

108 live calls to the Meta Model API: Muse Spark 1.1, Muse Spark 1.3, and Muse Wonder 1.0, with 36 cases each. **Total API cost: USD 2.99.**

| Policy | Probes needed (32 fault cases) | Closed by the model's own choice | Meta API cost (USD) |
| --- | ---: | ---: | ---: |
| Random order | 66 | n/a | none (no API call) |
| Heuristic (largest change first) | 48 | n/a | none (no API call) |
| **Elimination** | **41** | n/a | none (no API call) |
| Muse Spark 1.1 | 39 | 28/32 | 0.39 |
| Muse Wonder 1.0 | 40 | 27/32 | 2.08 |
| Muse Spark 1.3 | 43 | 27/32 | 0.52 |

*The cost column is the charge recorded for calls to the Meta Model API and nothing else. Local computation (image injection, detection, and probe measurements) is not measured and is not included in any figure, so the table is neither a total-cost nor a runtime comparison.*

All policies ultimately close 32/32 intervention cases (models continue in a fixed order when their first choice fails). Each model correctly abstained on all four controls; the random baseline executed probes on every control.

**Interpretation: the models did not demonstrate an advantage over elimination.** The prespecified criterion was "fewer probes, with the whole confidence interval below zero." No model met it: every interval against the rule includes zero. The results are also inconclusive about equivalence; the intervals do not establish it.

### Invalid model responses

Each model produced one invalid response out of 36 cases (3 of 108 in total). None was retried, repaired, or reclassified.

| Model | Case | Recorded condition |
| --- | --- | --- |
| Muse Spark 1.1 | F16 | A complete reply whose justification was 364 characters, above the 350-character limit. |
| Muse Spark 1.3 | F23 | All 8,192 completion tokens were used (8,190 on reasoning); no decision was returned. |
| Muse Wonder 1.0 | F08 | All 8,192 completion tokens were used (8,190 on reasoning); no decision was returned. |

The two token-limited cases show that maximum reasoning can consume a fixed completion allowance before a usable decision exists. The trial measures bounded maximum-reasoning conditions, not unrestricted model capability.

## Interpretation and reproducibility

- **Prespecified comparison.** The criterion required fewer probes with the full confidence interval below zero; it was not met. The primary baseline is implemented in [`baselines.py`](src/blindspot/benchmark/baselines.py).
- **Baseline comparison.** The model policies required 39 to 43 intervention probes, compared with 41 for elimination. Spark 1.1 incurred approximately USD 0.01 in recorded API cost per decision. These observations do not establish superiority or equivalence.
- **Measured failure modes.** The invalid responses above are recorded with their token usage rather than assumed.
- **Reproducibility.** The published observations, probe measurements and 108-row case table can be reproduced from the verified images without model calls. See [Data and reproducibility](docs/DATA.md#verify-the-published-evaluation-without-model-calls).

Further evaluation could use sequential decisions informed by probe outcomes, natural causes and additional images. Those conditions are outside the current experiment.

## Quick start

```bash
python -m pip install -e ".[dev]"
python scripts/download_data.py          # 2 public ZTF images from IRSA, ~76 MB, SHA-256 verified
export BLINDSPOT_FITS_ZIP=data/ztf       # PowerShell: $env:BLINDSPOT_FITS_ZIP="data/ztf"
python -m blindspot.cli smoke-fits --fits-zip data/ztf
python -m pytest
python -m scripts.reproduce_evaluation --fits-zip "$BLINDSPOT_FITS_ZIP"
```

Without the images, `blindspot plan` and the tests that do not need pixels still run. No API key is needed to reproduce the published tables. Live calls are optional, opt-in, and capped (see [Data](docs/DATA.md)).

## Meta tooling used

| Model (Meta Model API) | Reasoning | Role |
| --- | --- | --- |
| `muse-spark-1.1` | `xhigh` | probe planner |
| `muse-spark-1.3` | `max` | probe planner |
| `muse-wonder-1.0` | `max` | probe planner |

Structured outputs use a strict JSON schema. Requests go through the OpenAI-compatible client against `https://api.meta.ai/v1`, with retries disabled so that a call cannot be billed twice silently.

## Limits

- The detector is a **simplified matched filter**, not the ZTF production pipeline.
- Two exposures are used; the causes are **synthetic interventions** whose strengths were tuned on the same 36 patches that the benchmark evaluates.
- The model makes **one decision per case**; probe outcomes are precomputed. There is no closed loop yet, and no runtime or injection savings were measured.
- One response per model and case, so run-to-run variability is not measured.

Intervals, per-case details, and every caveat: [Results and limitations](docs/RESULTS.md) · [Benchmark protocol](docs/BENCHMARK_PROTOCOL.md) · [Architecture](docs/ARCHITECTURE.md).

## Provenance

Development period, authorship, and reused components are described in [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Data, license, and citation

Code: [MIT License](LICENSE). The images are public ZTF products served by NASA/IPAC IRSA. They are excluded from version control and remain under their own terms. Local image archives and session records are distributed separately from the source package. Acknowledgement text and citation: [docs/DATA.md](docs/DATA.md#acknowledging-ztf). Software metadata: [CITATION.cff](CITATION.cff).
