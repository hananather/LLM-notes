"""A finite-population review illustration for frozen linkage decisions.

The plan uses no truth labels. It reviews every pair accepted by any method,
then samples pairs rejected by all methods without replacement. Audit labels
enter only after the plan is fixed. Bounds concern this pair universe alone.
"""

from hashlib import sha256
import json
from math import comb
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_KEYS = ["left_id", "right_id"]
ACCEPTED = "accepted_by_any"
REJECTED = "rejected_by_all"


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


def _check_pairs(frame):
    if not isinstance(frame, pd.DataFrame) or not set(PAIR_KEYS).issubset(frame):
        raise ValueError("Pair data require left_id and right_id columns.")
    if frame[PAIR_KEYS].isna().any().any() or frame.duplicated(PAIR_KEYS).any():
        raise ValueError("Pair identifiers must be complete and unique.")
    if not all(isinstance(value, str) for value in frame[PAIR_KEYS].to_numpy().flat):
        raise ValueError("Use opaque string record identifiers in the audit plan.")


def _check_booleans(frame, columns):
    for column in columns:
        if column not in frame or not all(
            isinstance(value, (bool, np.bool_)) for value in frame[column]
        ):
            raise ValueError(f"{column} must contain complete Boolean decisions.")


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode()).hexdigest()


def make_audit_plan(decisions, method_columns, *, rejected_sample_size=50, seed=20261006):
    """Freeze a label-free union census and a random all-rejected sample.

    Supply only pair IDs and the named Boolean decision columns. The returned
    dictionary is JSON serializable and includes sampled IDs and probabilities.
    Sorting by pair ID makes the result independent of input row order.
    """
    method_columns = list(method_columns)
    if not method_columns or len(set(method_columns)) != len(method_columns):
        raise ValueError("Provide distinct method decision columns.")
    reserved = set(PAIR_KEYS + ["is_match", "entity_id", "split", "rec_id", "stratum"])
    if any(not isinstance(c, str) or c in reserved for c in method_columns):
        raise ValueError("Method columns must be named decisions, not labels or metadata.")
    rejected_sample_size = _integer(rejected_sample_size, "rejected_sample_size", 1)
    seed = _integer(seed, "seed")
    _check_pairs(decisions)
    if set(decisions.columns) != set(PAIR_KEYS + method_columns):
        raise ValueError("Pass only pair IDs and method decisions; exclude truth columns.")
    if decisions.empty:
        raise ValueError("The audit population must contain at least one pair.")
    _check_booleans(decisions, method_columns)
    population = decisions[PAIR_KEYS + method_columns].sort_values(PAIR_KEYS).reset_index(drop=True)
    union = population[method_columns].any(axis=1)
    accepted = population.loc[union, PAIR_KEYS]
    rejected = population.loc[~union, PAIR_KEYS]
    n_rejected = min(rejected_sample_size, len(rejected))
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(rejected), size=n_rejected, replace=False))
    sampled_rejected = rejected.iloc[positions]
    sample = []
    strata = []
    for name, frame, total in [
        (ACCEPTED, accepted, len(accepted)),
        (REJECTED, sampled_rejected, len(rejected)),
    ]:
        probability = len(frame) / total if total else None
        strata.append({"stratum": name, "population_pairs": total,
                       "sampled_pairs": len(frame), "inclusion_probability": probability})
        sample.extend({**pair, "stratum": name, "inclusion_probability": probability}
                      for pair in frame.to_dict("records"))
    payload = {"schema": 1, "seed": seed, "rejected_sample_size": rejected_sample_size,
               "method_columns": method_columns, "population_pairs": len(population),
               "population": population.to_dict("records"), "strata": strata, "sample": sample}
    return {**payload, "plan_sha256": _digest(payload)}


def _validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema") != 1:
        raise ValueError("Unsupported audit plan.")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != _digest(payload):
        raise ValueError("The frozen audit plan failed its integrity check.")
    reconstructed = make_audit_plan(pd.DataFrame(plan["population"]), plan["method_columns"],
                                   rejected_sample_size=plan["rejected_sample_size"], seed=plan["seed"])
    if reconstructed != plan:
        raise ValueError("Audit sample or inclusion probabilities differ from the declared design.")


def save_audit_plan(path, plan):
    """Save sampled IDs before review; preserve a different existing plan."""
    _validate_plan(plan)
    return _save_preserved_json(path, plan)


def _save_preserved_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    except FileExistsError:
        if json.loads(path.read_text()) != payload:
            raise ValueError("An audit record already exists here; use a new path for changed evidence.")
    return path


def _labels(frame):
    _check_pairs(frame)
    _check_booleans(frame, ["is_match"])
    return frame[PAIR_KEYS + ["is_match"]].copy()


def simulate_review(plan, truth_pairs):
    """Reveal benchmark truth only for the fixed sample, as simulated review.

    This does not assess human adjudication quality. The caller must freeze
    method decisions and save the audit plan before calling this function.
    """
    _validate_plan(plan)
    sample = pd.DataFrame(plan["sample"])
    reviewed = sample.merge(_labels(truth_pairs), on=PAIR_KEYS, how="left", validate="one_to_one")
    if reviewed.is_match.isna().any():
        raise ValueError("Each sampled pair needs an independent review label.")
    return reviewed


def finite_population_bounds(population_size, sample_size, observed):
    """Invert exact hypergeometric tails for the unknown number of true pairs.

    Return equal-tailed 95% limits and a separate one-sided 95% upper bound.
    Integer comparisons avoid numerical error in the 2.5% and 5% tail tests.
    This enumeration is intended for the notebook's small pair populations.
    """
    population_size = _integer(population_size, "population_size")
    sample_size = _integer(sample_size, "sample_size")
    observed = _integer(observed, "observed")
    if sample_size > population_size or observed > sample_size:
        raise ValueError("Observed and sampled counts exceed their population.")
    if sample_size == 0:
        return {"lower_95": 0, "upper_95": population_size,
                "upper_one_sided_95": population_size}
    if sample_size == population_size:
        return {"lower_95": observed, "upper_95": observed,
                "upper_one_sided_95": observed}
    denominator = comb(population_size, sample_size)
    two_sided, one_sided = [], []
    for total in range(observed, population_size - sample_size + observed + 1):
        minimum = max(0, sample_size - (population_size - total))
        maximum = min(sample_size, total)
        cdf = sum(comb(total, x) * comb(population_size - total, sample_size - x)
                  for x in range(minimum, min(observed, maximum) + 1))
        sf = sum(comb(total, x) * comb(population_size - total, sample_size - x)
                 for x in range(max(observed, minimum), maximum + 1))
        if 40 * cdf >= denominator and 40 * sf >= denominator:
            two_sided.append(total)
        if 20 * cdf >= denominator:
            one_sided.append(total)
    return {"lower_95": min(two_sided), "upper_95": max(two_sided),
            "upper_one_sided_95": max(one_sided)}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def summarize_audit(plan, reviewed):
    """Estimate missed links from reviewed labels without unseen truth.

    Accepted-link precision is exact for this finite slice because the union
    of accepted links is censused. Recall is a ratio estimate with bounds from
    the all-rejected sample. A zero-error sample does not prove perfect recall.
    """
    _validate_plan(plan)
    labels = _labels(reviewed)
    expected = pd.DataFrame(plan["sample"])
    if set(map(tuple, labels[PAIR_KEYS].to_numpy())) != set(map(tuple, expected[PAIR_KEYS].to_numpy())):
        raise ValueError("Supply exactly the labels in the fixed audit sample.")
    audited = expected.merge(labels, on=PAIR_KEYS, validate="one_to_one")
    union_true = int(audited.loc[audited.stratum.eq(ACCEPTED), "is_match"].sum())
    rejected_spec = next(item for item in plan["strata"] if item["stratum"] == REJECTED)
    n0, N0 = rejected_spec["sampled_pairs"], rejected_spec["population_pairs"]
    k0 = int(audited.loc[audited.stratum.eq(REJECTED), "is_match"].sum())
    bounds = finite_population_bounds(N0, n0, k0)
    estimated_rejected_true = N0 * k0 / n0 if n0 else 0.0
    estimated_true = union_true + estimated_rejected_true
    lower_true = union_true + bounds["lower_95"]
    upper_true = union_true + bounds["upper_95"]
    population = pd.DataFrame(plan["population"])
    known = population.merge(labels, on=PAIR_KEYS, how="left", validate="one_to_one")
    metric_rows = []
    for method in plan["method_columns"]:
        accepted = known.loc[known[method]]
        if accepted.is_match.isna().any():
            raise ValueError("The union census must label every accepted pair.")
        tp = int(accepted.is_match.sum())
        accepted_count = len(accepted)
        missed_in_union = union_true - tp
        recall = _ratio(tp, estimated_true)
        note = "Sampling interval covers this fixed pair universe only."
        if lower_true == 0:
            note = "Recall is undefined if the slice contains no true pairs."
        elif recall == 1 and upper_true > estimated_true:
            note = "The point estimate is 100%; the interval still permits missed links."
        metric_rows.append({
            "method": method, "accepted_pairs": accepted_count,
            "true_accepted_pairs": tp, "false_accepted_pairs": accepted_count - tp,
            "precision_census": _ratio(tp, accepted_count),
            "observed_missed_true_pairs": missed_in_union + k0,
            "estimated_missed_true_pairs": missed_in_union + estimated_rejected_true,
            "missed_true_pairs_lower_95": missed_in_union + bounds["lower_95"],
            "missed_true_pairs_upper_95": missed_in_union + bounds["upper_95"],
            "missed_true_pairs_upper_one_sided_95": missed_in_union + bounds["upper_one_sided_95"],
            "recall_estimate": recall,
            "recall_lower_95": _ratio(tp, upper_true),
            "recall_upper_95": _ratio(tp, lower_true), "recall_note": note,
        })
    strata = []
    for item in plan["strata"]:
        is_union = item["stratum"] == ACCEPTED
        strata.append({**item, "observed_true_pairs": union_true if is_union else k0,
                       "estimated_true_pairs": union_true if is_union else estimated_rejected_true,
                       "true_pairs_lower_95": union_true if is_union else bounds["lower_95"],
                       "true_pairs_upper_95": union_true if is_union else bounds["upper_95"],
                       "true_pairs_upper_one_sided_95": union_true if is_union else bounds["upper_one_sided_95"]})
    return {"plan_sha256": plan["plan_sha256"], "strata": pd.DataFrame(strata),
            "metrics": pd.DataFrame(metric_rows).set_index("method"),
            "reviewed": audited, "population_pairs": len(population),
            "reviewed_pairs": len(audited), "estimated_true_pairs": estimated_true,
            "true_pairs_lower_95": lower_true, "true_pairs_upper_95": upper_true,
            "scope": "Simulated review of a fixed finite benchmark; assumes correct audit labels."}


def save_audit_review(path, plan, reviewed):
    """Preserve sampled IDs, inclusion probabilities and simulated review labels."""
    result = summarize_audit(plan, reviewed)
    rows = result["reviewed"].to_dict("records")
    payload = {"schema": 1, "plan_sha256": plan["plan_sha256"],
               "review_type": "simulated review using independent benchmark identity labels",
               "reviewed": rows, "reviewed_sha256": _digest(rows)}
    return _save_preserved_json(path, payload)


def census_metrics(plan, truth_pairs):
    """Score all frozen decisions against full truth, separate from the sample."""
    _validate_plan(plan)
    labels = _labels(truth_pairs)
    population = pd.DataFrame(plan["population"])
    if set(map(tuple, labels[PAIR_KEYS].to_numpy())) != set(map(tuple, population[PAIR_KEYS].to_numpy())):
        raise ValueError("The truth census must cover exactly the frozen pair population.")
    full = population.merge(labels, on=PAIR_KEYS, validate="one_to_one")
    rows = []
    for method in plan["method_columns"]:
        actual, prediction = full.is_match, full[method]
        tp = int((actual & prediction).sum())
        fp = int((~actual & prediction).sum())
        fn = int((actual & ~prediction).sum())
        tn = int((~actual & ~prediction).sum())
        rows.append({"method": method, "pairs": len(full), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                     "F1": _ratio(2 * tp, 2 * tp + fp + fn)})
    return pd.DataFrame(rows).set_index("method")
