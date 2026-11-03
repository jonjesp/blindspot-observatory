"""Human-readable results and case tables for the paired Meta model evaluation."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from blindspot.benchmark.report import _policy_block, paired_bootstrap_ci
from blindspot.provenance.ledger import digest
from scripts.compare_models import evaluate_candidate, spent_usd


def scored_cases(document: dict, ledger: dict, model: str) -> list[dict]:
    rows = []
    for case in document["cases"]:
        key = f"{model}/{case['definition']['id']}"
        record = ledger["attempts"].get(key)
        if record is not None and record["state"] != "reserved_unknown":
            row = evaluate_candidate(case, record)
            row["record"] = record
            rows.append(row)
    return rows


def _interval_text(interval: dict | None) -> str:
    if interval is None:
        return "Not estimated"
    lower, upper = interval["ci95"]
    return f"{interval['mean']:+.4f} [{lower:+.4f}, {upper:+.4f}]"


def _comparison(rows: list[dict], document: dict) -> tuple[dict | None, dict | None]:
    if len(rows) != len(document["cases"]):
        return None, None
    differences = [
        row["policies"]["candidate"]["probes"] - row["policies"]["elimination"]["probes"]
        for row in rows
        if row["kind"] != "no_intervention"
    ]
    criterion = document["criterion"]
    settings = {"seed": criterion["bootstrap_seed"], "n_boot": criterion["bootstrap_samples"]}
    descriptive = paired_bootstrap_ci(differences, **settings)
    confidence = 1 - criterion["familywise_alpha"] / criterion["primary_comparisons"]
    adjusted = paired_bootstrap_ci(differences, coverage=confidence, **settings)
    return descriptive, adjusted


def _write_cases(document: dict, ledger: dict, directory: Path) -> None:
    columns = [
        "model",
        "reasoning_effort",
        "case_id",
        "case_type",
        "exposure",
        "response_status",
        "returned_model",
        "selected_probes",
        "executed_probes",
        "closed",
        "correct_abstention",
        "continuation_used",
        "closed_without_continuation",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "request_seconds",
        "calculated_cost_usd",
    ]
    with (directory / "case_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for profile in document["profiles"]:
            for row in scored_cases(document, ledger, profile["model"]):
                record, candidate = row["record"], row["candidate"]
                trace = row["policies"]["candidate"]
                usage = record["usage"]
                values = {
                    "model": profile["model"],
                    "reasoning_effort": profile["reasoning_effort"],
                    "case_id": row["id"],
                    "case_type": row["kind"],
                    "exposure": row["field"],
                    "response_status": record["state"],
                    "returned_model": record.get("response_model"),
                    "selected_probes": "|".join((record.get("decision") or {}).get("probes", [])),
                    "executed_probes": "|".join(trace["executed"]),
                    "closed": trace["success"] if row["kind"] != "no_intervention" else "",
                    "correct_abstention": trace["success"] if row["kind"] == "no_intervention" else "",
                    "continuation_used": candidate["used_fallback"],
                    "closed_without_continuation": candidate["closed_without_fallback"],
                    **usage,
                    "reasoning_tokens": record.get("reasoning_tokens"),
                    "request_seconds": f"{record['elapsed_seconds']:.6f}",
                    "calculated_cost_usd": f"{record['cost_usd']:.8f}",
                }
                for name, value in values.items():
                    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                        values[name] = "'" + value
                writer.writerow(values)


def write_report(document: dict, ledger: dict, directory: Path, *, error: str | None = None) -> None:
    """Regenerate derived outputs from the preserved protocol and request ledger."""
    mode = ledger["mode"]
    rows_by_model = {
        profile["model"]: scored_cases(document, ledger, profile["model"]) for profile in document["profiles"]
    }
    baselines = [case["reference"] for case in document["cases"]]
    primary = _policy_block(baselines, "elimination")
    complete = all(len(rows) == len(document["cases"]) for rows in rows_by_model.values())
    ledger_seal = digest({field: value for field, value in ledger.items() if field != "seal_sha256"})
    lines = [
        "# Comparative model evaluation",
        "",
        f"Execution mode: **{mode}**. "
        f"Data collection: **{'complete' if complete and not error else 'incomplete'}**.",
        "",
        "The experiment evaluates diagnostic-probe selection from numerical observations on "
        "two ZTF exposures. "
        "Models receive identical case observations and a fixed decision schema. They do not "
        "receive FITS pixels "
        "or hidden intervention labels. Measurements and policy scoring are deterministic.",
        "",
    ]
    if mode != "live":
        lines += ["**Simulated execution: no model performance or paid inference was measured.**", ""]
    if error:
        lines += [f"Execution stopped: {error.replace(chr(10), ' ')}", ""]
    lines += [
        "## Protocol and availability",
        "",
        "| Model | Requested reasoning | Output-token limit | Availability |",
        "| --- | --- | ---: | --- |",
    ]
    for profile in document["profiles"]:
        available = ledger.get("availability", {}).get(profile["model"])
        status = "Available" if available else "Unavailable" if available is False else "Not checked"
        lines.append(
            f"| {profile['name']} | {profile['reasoning_effort']} | "
            f"{document['output_token_limit']} | {status} |"
        )
    lines += [
        "",
        "There are 36 matched cases: 16 single-cause interventions, 16 two-cause "
        "interventions, and four controls. "
        "One response is requested per model and case. Model order is randomized within each "
        "shuffled case block. "
        "The largest documented reasoning level supported by each selected API model is "
        "requested. The common "
        "output-token limit includes private reasoning and the final answer; maximum reasoning is not an "
        "unrestricted token allowance. Truncated responses remain observed failures and are "
        "not automatically retried.",
        "",
        "## Policy performance",
        "",
        "All model policy totals include fixed-order continuation after unsuccessful selected probes. "
        "Direct closures count only model-selected probes. Means exclude controls.",
        "",
        "| Policy | Valid responses | Interventions closed | Direct closures | Controls "
        "correct | Mean probes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("random", "heuristic", "elimination"):
        block = _policy_block(baselines, name)
        lines.append(
            f"| {name.title()} | — | {block['intervention_cases']['k']}/32 | — | "
            f"{block['correct_abstentions']['k']}/4 | {block['mean_probes']:.4f} |"
        )
    for profile in document["profiles"]:
        rows = rows_by_model[profile["model"]]
        if not rows:
            lines.append(f"| {profile['name']} + continuation | 0/0 | Not evaluated | — | — | — |")
            continue
        block = _policy_block(rows, "candidate")
        valid = sum(row["record"]["state"] == "completed" for row in rows)
        direct = sum(
            row["candidate"]["closed_without_fallback"] for row in rows if row["kind"] != "no_intervention"
        )
        active, controls = block["intervention_cases"], block["correct_abstentions"]
        average = "—" if block["mean_probes"] is None else f"{block['mean_probes']:.4f}"
        lines.append(
            f"| {profile['name']} + continuation | {valid}/{len(rows)} | {active['k']}/{active['n']} | "
            f"{direct}/{active['n']} | {controls['k']}/{controls['n']} | {average} |"
        )
    lines += [
        "",
        "Fractions use evaluated cases only. Incomplete collections are not ranked.",
        "",
        "## Primary comparison with elimination",
        "",
        "The prespecified criterion requires a complete live collection, valid model responses, no loss in "
        "intervention closures or correct control abstentions, and an upper confidence bound below zero for "
        "the paired mean probe difference. Nominal Bonferroni adjustment covers the three "
        "primary comparisons. "
        "Intervals are conditional case-bootstrap summaries, not population coverage guarantees.",
        "",
        "| Model | Mean difference [95% interval] | Mean difference [98.33% interval] | Criterion |",
        "| --- | --- | --- | --- |",
    ]
    for profile in document["profiles"]:
        rows = rows_by_model[profile["model"]]
        descriptive, adjusted = _comparison(rows, document)
        block = _policy_block(rows, "candidate")
        accepted = bool(
            mode == "live"
            and not error
            and adjusted is not None
            and adjusted["ci95"][1] < 0
            and all(row["record"]["state"] == "completed" for row in rows)
            and block["intervention_cases"]["k"] >= primary["intervention_cases"]["k"]
            and block["correct_abstentions"]["k"] >= primary["correct_abstentions"]["k"]
        )
        verdict = "Met for this case set" if accepted else "Not established" if adjusted else "Incomplete"
        lines.append(
            f"| {profile['name']} | {_interval_text(descriptive)} | {_interval_text(adjusted)} | {verdict} |"
        )
    lines += [
        "",
        "A negative difference favors the model policy. An interval containing zero establishes neither "
        "superiority nor equivalence. Bootstrap resampling uses 10,000 paired samples and the frozen seed.",
        "",
        "## Exploratory model comparisons",
        "",
        "| Difference | Matched interventions | Mean difference [95% interval] |",
        "| --- | ---: | --- |",
    ]
    for first, second in combinations(document["profiles"], 2):
        left, right = rows_by_model[first["model"]], rows_by_model[second["model"]]
        if len(left) != len(document["cases"]) or len(right) != len(document["cases"]):
            continue
        differences = [
            first_case["policies"]["candidate"]["probes"] - second_case["policies"]["candidate"]["probes"]
            for first_case, second_case in zip(left, right, strict=True)
            if first_case["kind"] != "no_intervention"
        ]
        interval = paired_bootstrap_ci(
            differences, seed=document["criterion"]["bootstrap_seed"], n_boot=10000
        )
        lines.append(
            f"| {first['name']} minus {second['name']} | {len(differences)} | {_interval_text(interval)} |"
        )
    lines += [
        "",
        "These pairwise intervals are descriptive and unadjusted; they do not define a "
        "separate winner-selection rule.",
        "",
        "## Resource use and response quality",
        "",
        "| Model | Prompt / cached tokens | Completion tokens | Reported reasoning tokens | "
        "Request median / p95, s | Calculated USD | Truncated | Continuation cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in document["profiles"]:
        rows = rows_by_model[profile["model"]]
        records = [row["record"] for row in rows]
        times = [row["elapsed_seconds"] for row in records]
        timing = f"{np.median(times):.2f} / {np.percentile(times, 95):.2f}" if times else "Not measured"
        tokens = [row["reasoning_tokens"] for row in records if row.get("reasoning_tokens") is not None]
        reasoning = f"{sum(tokens)} ({len(tokens)}/{len(records)} responses)" if tokens else "Not reported"
        lines.append(
            f"| {profile['name']} | {sum(row['usage']['prompt_tokens'] for row in records)} / "
            f"{sum(row['usage']['cached_tokens'] for row in records)} | "
            f"{sum(row['usage']['completion_tokens'] for row in records)} | {reasoning} | {timing} | "
            f"{sum(row['cost_usd'] for row in records):.6f} | "
            f"{sum(row['finish_reason'] == 'length' for row in records)} | "
            f"{sum(row['candidate']['used_fallback'] for row in rows)} |"
        )
    unresolved = [key for key, record in ledger["attempts"].items() if record["state"] == "reserved_unknown"]
    lines += [
        "",
        f"Accounted expenditure, including unresolved reservations: USD {spent_usd(ledger):.6f} "
        f"within the experiment allowance of USD {document['budget_usd']:.2f}.",
        f"Unresolved requests: {len(unresolved)}. Such reservations require inspection before resumption.",
        "Pending request identifiers: " + (", ".join(unresolved) if unresolved else "none") + ".",
        "",
        "Reasoning tokens are included in completion tokens and are not billed a second time. "
        "Missing reasoning "
        "counts remain unknown. Request latency covers the client HTTP/generation interval, "
        "not the full scientific "
        "pipeline. Token costs use frozen published rates and are not reconciled provider "
        "invoices. The reservation "
        "bound is a local estimate rather than a provider-enforced account limit.",
        "",
        "## Stratified observations",
        "",
        "| Model | Exposure / case class | Cases evaluated | Mean probes | Successful outcomes |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for profile in document["profiles"]:
        rows = rows_by_model[profile["model"]]
        for field in ("E4_r", "E5_g"):
            for kind in ("single_cause", "two_causes", "no_intervention"):
                group = [row for row in rows if row["field"] == field and row["kind"] == kind]
                if group:
                    traces = [row["policies"]["candidate"] for row in group]
                    lines.append(
                        f"| {profile['name']} | {field} / {kind} | {len(group)} | "
                        f"{np.mean([t['probes'] for t in traces]):.4f} | "
                        f"{sum(t['success'] for t in traces)}/{len(group)} |"
                    )
    lines += [
        "",
        "For controls, a successful outcome means an explicit abstention without executing probes.",
        "",
        "## Interpretation and reproducibility",
        "",
        "This is one response per model and case on two exposures. It does not estimate "
        "model-repeat variance "
        "or performance on unseen exposures. Probe outcomes are precomputed; the model "
        "chooses an initial order "
        "without a feedback loop after each probe. Counts do not establish elapsed-time or "
        "injection savings. "
        "Maximum reasoning settings differ by supported API capability; the comparison is not "
        "matched compute. "
        "The fixed output limit can constrain reasoning and must be considered alongside truncation rates. "
        "Model identity and endpoint fields are recorded observations, not cryptographic "
        "provider attestations.",
        "",
        f"Protocol seal: `{document['seal_sha256']}`.",
        f"Ledger seal: `{ledger_seal}`.",
        f"Provider base URL: `{document['base_url']}`.",
        f"Source revision: `{document['git']['commit'] or 'not recorded'}`.",
        "Prepared environment: "
        + ", ".join(f"{name} {value}" for name, value in document["environment"].items())
        + ".",
        f"Documentation and rate snapshot: {document['documentation']['checked_date']}.",
        "",
        "The protocol retains the shared observations, source fingerprints, case order, model "
        "settings, and request "
        "hashes. The ledger retains provider identities, visible decisions, usage, timing, and reservations. "
        "`case_results.csv` provides the per-case table; private reasoning text is not collected.",
        "",
        f"Sources: [reasoning]({document['documentation']['reasoning']}), "
        f"[pricing]({document['documentation']['pricing']}), "
        f"[model availability]({document['documentation']['models']}).",
        "",
    ]
    directory.mkdir(parents=True, exist_ok=True)
    _write_cases(document, ledger, directory)
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
