"""Complete-decision metrics and a prospective development spending gate.

Inputs are evaluator-only truth and already frozen decisions. No fitting or
candidate construction belongs in this module. Abstention is never a correct NIL.
"""
from __future__ import annotations

import math
import hashlib
import json
import numpy as np

BOOTSTRAP_SEED = 20261007
BOOTSTRAP_REPLICATES = 2000
GATE_GAIN_PER_QUERY = 0.05
GATE_MAX_COVERAGE_LOSS = 0.05


def outcomes(truth, predictions):
    """Truth: query_id -> iterable IDs. Predictions: query_id -> decision dict.

    Decisions contain status (linked/nil/review/failed) and target_ids. Missing
    predictions are failed. Malformed decisions are failed with no target set.
    Every truth query remains in the denominator, including failed retrieval.
    """
    extra = set(predictions) - set(truth)
    if extra:
        raise ValueError("Predictions contain queries outside the frozen denominator.")
    rows = []
    for query_id in sorted(truth):
        gold = set(truth[query_id])
        p = predictions.get(query_id, {"status": "failed", "target_ids": []})
        if not isinstance(p, dict):
            p = {"status": "failed", "target_ids": []}
        status = p.get("status")
        raw = p.get("target_ids", [])
        valid = (status in {"linked", "nil", "review", "failed"}
                 and isinstance(raw, list) and all(isinstance(x, str) for x in raw)
                 and len(raw) == len(set(raw))
                 and ((status == "linked" and len(raw) > 0)
                      or (status != "linked" and not raw)))
        if not valid:
            status, raw = "failed", []
        guessed = set(raw)
        automatic = status in {"linked", "nil"}
        rows.append({"query_id": query_id, "status": status,
                     "correct": automatic and guessed == gold,
                     "automatic": automatic, "true_nil": not gold,
                     "false_links": len(guessed - gold),
                     "missed_links": len(gold - guessed),
                     "true_links": len(guessed & gold),
                     "false_assignment": bool(guessed - gold),
                     "false_nil": status == "nil" and bool(gold),
                     "linked_on_nil": status == "linked" and not gold})
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        raise ValueError("A result must have a nonempty denominator.")
    automatic = sum(r["automatic"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    return {"queries": n, "correct_complete_decisions": correct,
            "query_denominator_sha256": hashlib.sha256(json.dumps(sorted(r["query_id"] for r in rows), separators=(",", ":")).encode()).hexdigest(),
            "complete_decision_accuracy": correct / n,
            "automatic_decisions": automatic, "automatic_coverage": automatic / n,
            "wrong_automatic_decisions": automatic - correct,
            "error_among_automatic": (automatic - correct) / automatic if automatic else None,
            "false_links": sum(r["false_links"] for r in rows),
            "missed_links": sum(r["missed_links"] for r in rows),
            "true_links": sum(r["true_links"] for r in rows),
            "queries_with_false_assignment": sum(r["false_assignment"] for r in rows),
            "true_nil_queries": sum(r["true_nil"] for r in rows),
            "false_nil_decisions": sum(r["false_nil"] for r in rows),
            "linked_on_nil_queries": sum(r["linked_on_nil"] for r in rows),
            "review_queries": sum(r["status"] == "review" for r in rows),
            "failed_queries": sum(r["status"] == "failed" for r in rows)}


def paired_comparison(reference_rows, candidate_rows, groups):
    """Paired query-weighted difference; uncertainty resamples whole groups."""
    left = {r["query_id"]: r for r in reference_rows}
    right = {r["query_id"]: r for r in candidate_rows}
    if set(left) != set(right) or not set(left) <= set(groups):
        raise ValueError("Paired methods and cluster mapping must cover the same queries.")
    grouped = {}
    for key in sorted(left):
        group = groups[key]
        stats = grouped.setdefault(group, [0, 0])
        stats[0] += int(right[key]["correct"]) - int(left[key]["correct"])
        stats[1] += 1
    values = np.asarray(list(grouped.values()), dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        estimates.append(sample[0] / sample[1])
    return {"queries": len(left), "source_groups": len(grouped),
            "corrections": sum(not left[k]["correct"] and right[k]["correct"] for k in left),
            "regressions": sum(left[k]["correct"] and not right[k]["correct"] for k in left),
            "accuracy_difference": float(values[:, 0].sum() / values[:, 1].sum()),
            "cluster_bootstrap_95_percent_interval": np.quantile(estimates, [.025, .975]).tolist(),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interval_scope": "This source sample and family definition; not NSO population uncertainty."}


def development_gate(baseline_results, candidate_result, *, eligible_methods):
    """A spending gate, not a statistical or production acceptance guarantee.

    Compare with the conventional method having the most correct complete dev
    decisions; alphabetical method name breaks ties. This selection rule is
    fixed before results. No baseline or prompt is revised from gate outcomes.
    """
    if not eligible_methods or not set(eligible_methods) <= set(baseline_results):
        raise ValueError("Supply every prospectively eligible operational comparator.")
    baseline_results = {k: baseline_results[k] for k in eligible_methods}
    name = sorted(baseline_results, key=lambda k: (
        -baseline_results[k]["correct_complete_decisions"], k))[0]
    baseline = baseline_results[name]
    n = candidate_result["queries"]
    if any(b["queries"] != n or b["query_denominator_sha256"] != candidate_result["query_denominator_sha256"] for b in baseline_results.values()):
        raise ValueError("All development comparisons require the same denominator.")
    required_gain = math.ceil(GATE_GAIN_PER_QUERY * n)
    gain = candidate_result["correct_complete_decisions"] - baseline["correct_complete_decisions"]
    conditions = {
        "at_least_five_extra_correct_per_hundred": gain >= required_gain,
        "no_additional_false_links": candidate_result["false_links"] <= baseline["false_links"],
        "no_additional_queries_with_false_assignment": candidate_result["queries_with_false_assignment"] <= baseline["queries_with_false_assignment"],
        "coverage_loss_at_most_five_percentage_points": (
            candidate_result["automatic_decisions"] >= baseline["automatic_decisions"] - math.floor(GATE_MAX_COVERAGE_LOSS * n)),
    }
    return {"pass": all(conditions.values()), "reference": name,
            "eligible_methods": sorted(eligible_methods),
            "queries": n, "required_additional_correct": required_gain,
            "observed_additional_correct": gain, "conditions": conditions,
            "interpretation": "A prospective budget decision, not evidence of production safety."}
