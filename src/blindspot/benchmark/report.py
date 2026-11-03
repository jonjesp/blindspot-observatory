"""Aggregation of per-case results into the numbers the benchmark reports."""

from __future__ import annotations

from statistics import mean

import numpy as np

from blindspot.metrics.completeness import wilson

POLICIES = ("random", "heuristic", "elimination", "muse")
BASELINES = POLICIES[:-1]


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (0.0, 1.0)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [round(lo, 3), round(hi, 3)]}


def paired_bootstrap_ci(
    diffs: list[float], *, seed: int = 0, n_boot: int = 5000, coverage: float = 0.95
) -> dict:
    """Percentile bootstrap of the mean of paired differences (resampling cases)."""
    values = np.asarray(diffs, dtype=float)
    if values.size == 0:
        raise ValueError("No paired differences to resample")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    tail = (1 - coverage) / 2
    lo, hi = np.quantile(means, [tail, 1 - tail])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)], "n": int(values.size)}


def _policy_block(rows: list[dict], name: str) -> dict:
    interv = [r for r in rows if r["kind"] != "no_intervention"]
    none = [r for r in rows if r["kind"] == "no_intervention"]
    traces = [r["policies"][name] for r in interv]
    successes = sum(t["success"] for t in traces)
    total_probes = sum(t["probes"] for t in traces)
    block = {
        "intervention_cases": _rate(successes, len(interv)),
        "mean_probes": mean(t["probes"] for t in traces) if traces else None,
        "probes_per_success": (total_probes / successes) if successes else None,
        "first_probe_closes": _rate(
            sum(bool(r["policies"][name]["first_probe_closed"]) for r in interv), len(interv)
        ),
        "correct_abstentions": _rate(sum(r["policies"][name]["success"] for r in none), len(none)),
        "by_exposure": {},
    }
    for field in sorted({r["field"] for r in interv}):
        sub = [r["policies"][name] for r in interv if r["field"] == field]
        block["by_exposure"][field] = {
            "success": _rate(sum(t["success"] for t in sub), len(sub)),
            "mean_probes": mean(t["probes"] for t in sub),
        }
    return block


def summarise(rows: list[dict]) -> dict:
    """Per-policy rates, probe costs and paired Muse-versus-baseline differences."""
    interv = [r for r in rows if r["kind"] != "no_intervention"]
    summary = {
        "cases": len(rows),
        "evaluation_scope": "Policy replay on precomputed probes; no measured runtime or injection saving.",
        "interval_scope": "Case bootstrap conditional on these exposures; not a population interval.",
        "solvable_by_some_probe": _rate(sum(r["solvable"] for r in interv), len(interv)),
        "policies": {name: _policy_block(rows, name) for name in POLICIES},
        "muse_minus_baseline_probes": {},
    }
    if interv:
        for base in BASELINES:
            diffs = [r["policies"]["muse"]["probes"] - r["policies"][base]["probes"] for r in interv]
            summary["muse_minus_baseline_probes"][base] = paired_bootstrap_ci(diffs, seed=len(base))
    muse = [r["muse"] for r in rows]
    summary["muse_replies"] = {
        "valid": sum(m["state"] == "completed" for m in muse),
        "invalid": sum(m["state"] != "completed" for m in muse),
        "used_fallback": sum(bool(m["used_fallback"]) for m in muse),
        "closed_without_fallback": _rate(
            sum(bool(r["muse"]["closed_without_fallback"]) for r in interv), len(interv)
        ),
        "cost_usd": round(sum(m["cost_usd"] for m in muse), 6),
    }
    return summary


def format_summary(summary: dict, *, dry_run: bool = False) -> str:
    """Compact terminal table, explicitly attributing the Muse-plus-fallback policy."""
    head = (
        f"{'policy':<18}{'solved':>10}{'mean probes':>13}{'probes/solve':>14}"
        f"{'1st-probe':>11}{'abstain ok':>12}"
    )
    lines = [head, "-" * len(head)]
    for name, b in summary["policies"].items():
        solved, first, ab = b["intervention_cases"], b["first_probe_closes"], b["correct_abstentions"]
        per = "-" if b["probes_per_success"] is None else f"{b['probes_per_success']:.2f}"
        mp = "-" if b["mean_probes"] is None else f"{b['mean_probes']:.2f}"
        label = name
        if name == "muse":
            label = "canned+fallback" if dry_run else "muse+fallback"
        lines.append(
            f"{label:<18}{solved['k']:>5}/{solved['n']:<4}{mp:>13}{per:>14}"
            f"{first['k']:>7}/{first['n']:<3}{ab['k']:>8}/{ab['n']:<3}"
        )
    lines.append("")
    lines.append(
        f"cases solvable by at least one probe: {summary['solvable_by_some_probe']['k']}"
        f"/{summary['solvable_by_some_probe']['n']}"
    )
    for base, d in summary["muse_minus_baseline_probes"].items():
        lines.append(
            f"muse - {base}: mean probes {d['mean']:+.2f}  95% CI [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
        )
    r = summary["muse_replies"]
    direct = r["closed_without_fallback"]
    lines.append(f"cases closed by selected probes without fallback: {direct['k']}/{direct['n']}")
    lines.append(
        f"muse replies: {r['valid']} valid, {r['invalid']} invalid, fallback used {r['used_fallback']}, "
        f"cost ${r['cost_usd']:.4f}"
    )
    lines.append(summary["evaluation_scope"])
    lines.append(summary["interval_scope"])
    return "\n".join(lines)
