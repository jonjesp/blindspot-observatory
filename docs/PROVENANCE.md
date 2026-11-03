# Provenance

This document states what the repository contains, who developed it, when, and which parts come from elsewhere.

## Authorship and period

BlindSpot Observatory is a single-author project. The author reports development starting on **27 October 2026**, with planned completion on 4 November 2026 for the Meta Global AI Developer Hackathon (reported event dates: 26 October to 4 November 2026). The dates describe project provenance rather than a complete version-control chronology.

## Developed for this project

- The `blindspot` package: ZTF reader, Gaussian injection, simplified detector, completeness statistics, agent contract, cost-capped ledger, benchmark generator, baselines and scoring.
- The 36-case benchmark and its frozen protocol.
- All live model calls to the Meta Model API recorded under `evidence/`. The records carry dates from 2026-10-31 to 2026-11-03; earlier experiments carry no dates.
- The three-model evaluation (108 requests), its report, the tests and the continuous-integration workflow.

## Not developed for this project

- The two ZTF science images: public IRSA products, obtained with `scripts/download_data.py` and not stored in the repository.
- Third-party libraries: NumPy, SciPy, PyYAML and the OpenAI Python SDK.
- The Meta models called through the API.

## Earlier prototype runs (E6 to E19)

`docs/RESULTS.md` lists earlier prototype experiments (E6 to E19) that led to the final benchmark. They were run within the development period above. They are kept as history, are not claims for the final benchmark, and are not pooled with it. Their per-case inputs are only partly preserved.

## Repository contents

The intended versioned contents are project code, documentation and curated evidence. Local image data and run records are retained separately from the source distribution. API keys, `.env` files, local virtual environments, image data and local run folders are excluded by `.gitignore`.
