# Model Laboratory

The local browser interface runs one case from the sealed ZTF model evaluation.
It supports the E4_r and E5_g exposures, all 36 benchmark cutouts, and the Muse
Spark 1.1, Muse Spark 1.3, and Muse Wonder 1.0 profiles.

## Archived response

Commands are run from the repository root with the project environment active.
The verified FITS archive is expected at `data/ztf_images.zip`. A different
archive or a directory of FITS files can be supplied with `--fits PATH`.

```bash
python -m pip install -e .
python -m scripts.laboratory
```

The terminal prints the local address. With the default settings, the interface
is available at [127.0.0.1:8765](http://127.0.0.1:8765/). A different port can be
set with `--port`. The listener is restricted to IPv4 loopback.

## Inspect a run

The controls are grouped under **Run configuration**. The selected exposure,
cutout, model, and execution mode remain visible when the menu is collapsed.
The run monitor advances as the detector and model stages complete. Its
elapsed time covers the browser run. Provider time is reported separately
under **Request provenance and token usage**.

For a first check, select **E4_r**, **F01**, **Muse Spark 1.3**, and
**Archived response**, then select **Run analysis**. The pilot bars show
counts for the two arms. The model panel displays the validated decision and
its rationale as soon as the response is available.

The original FITS view can display peaks detected on the unmodified cutout.
After scoring, **Inspect batches** opens the holdout views for a probe that
either policy executed. Before and after frames show the synthetic injection
images for the selected batch. The three overlays distinguish all detector
peaks, injected source positions, and matched peaks. The recovery totals
combine the four batches in each condition. Before and after positions are
sampled independently, so markers are compared through recovery counts.

For F01, the point-spread function probe should show 17/100 recoveries before
the probe and 65/100 after it. Provenance should report zero provider calls.
The selected and executed probes can be checked against the matching row in
[case_results.csv](../evidence/model_evaluation/case_results.csv).

## Live provider request

The `agent` extra supplies the client. The credentials are read by the Python
process from `MODEL_API_KEY` and `MODEL_BASE_URL`. The endpoint must match the
sealed protocol (`https://api.meta.ai/v1`). When the existing, trusted local
`.env` uses Bash-compatible assignments, it can be loaded from Git Bash as
shown below. A budget of USD 0.25 is an example session allowance.

```bash
python -m pip install -e '.[agent]'
set -a
source ./.env
set +a
python -m scripts.laboratory --live-budget-usd 0.25
```

Live mode requires the model to be available to the configured provider
account. The interface displays the estimated reservation before execution
and requires authorization for each request. One live attempt per model and
case is allowed per server session. The sealed session ledger is written under
`evidence/runs/laboratory/`. If an attempt leaves an unresolved reservation or
returns inconsistent identity or usage, live execution is paused for that
session. The ledger and provider activity should be reconciled before a new
request. Session budgets are independent of provider account limits.

## Interpretation

The FITS checksum and the newly computed pilot and holdout values are checked
against the manifest and sealed protocol before the model stage. The detector
computes the measurements locally. The model receives the protocol prompt and
its numerical observation. The policies are then scored on the same
measurements in their respective probe orders. The results identify any
fixed-order continuation and distinguish a case closure from a correct
abstention on a control case.

Historical time, token usage, and calculated cost describe the archived
request in replay mode. Live values describe the new request. Single-case
comparisons describe the selected case. The campaign-level comparison is
reported in [RESULTS.md](RESULTS.md).
