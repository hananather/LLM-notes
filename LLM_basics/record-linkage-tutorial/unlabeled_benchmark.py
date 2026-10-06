"""Fit and audit an unlabeled Splink extension of the fixed FEBRL teaching slice.

The protocol is frozen before fitting; predictions are saved before identity
labels are read for evaluation. Existing data, notebooks and snapshots are not
modified. Run with the tutorial's pinned Python environment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import io
import json
import logging
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl


FIELDS = ["given_name", "surname", "date_of_birth", "soc_sec_id",
          "street_number", "postcode"]
INPUT_COLUMNS = ["source", "unique_id", *FIELDS]
PRIOR_RULE_FIELDS = [["soc_sec_id"], ["given_name", "surname", "date_of_birth"]]
EM_FIELDS = ["date_of_birth", "postcode"]
THRESHOLDS = [0.1, 0.5, 0.9, 0.99]
MAIN_THRESHOLD = 0.5
ASSUMED_RECALL = 0.8
MAX_ITERATIONS = 100
EM_TOLERANCE = 0.0001
REQUIRED_SPLINK = "5.0.0"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path, value):
    """Use exclusive creation so repeated runs cannot overwrite earlier evidence."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _protocol(support):
    return {
        "schema": 1,
        "scope": "Unlabeled estimation on the existing FEBRL training partition",
        "evidence_boundary": (
            "Frozen extension of a previously inspected benchmark, not a blind "
            "preregistration. No new choices depend on this run's test outcomes."
        ),
        "training_columns": INPUT_COLUMNS,
        "partition_metadata": {
            "source": "data/record_truth.csv",
            "columns_read_before_prediction_freeze": ["unique_id", "split"],
            "rule": "Keep rows whose existing split is train; discard split before fitting.",
            "boundary": (
                "The original partition and test slice were constructed using entity "
                "identities. Their existing membership is reused; identities are not "
                "supplied to the unlabeled estimator."
            ),
        },
        "prior": {
            "method": "estimate_probability_two_random_records_match",
            "exact_agreement_rules": PRIOR_RULE_FIELDS,
            "rule_combination": "union, with duplicate pairs removed",
            "assumed_recall": ASSUMED_RECALL,
            "record_sample_proportion": 1.0,
            "assumptions": [
                "The strict rules have approximately perfect precision.",
                "The strict rules recover 80% of training matches; this is assumed, not measured.",
            ],
            "not_used": "Neither entity counts nor test-slice match prevalence set the prior.",
        },
        "u_estimation": {"max_pairs": 1_000_000, "seed": 2026,
                         "min_count_per_level": None, "num_chunks": 1},
        "em": {
            "ordered_exact_block_fields": EM_FIELDS,
            "fix_u_probabilities": True,
            "fix_m_probabilities": False,
            "fix_probability_two_random_records_match": False,
            "populate_probability_two_random_records_match_from_trained_values": False,
            "estimate_without_term_frequencies": True,
            "max_iterations": MAX_ITERATIONS,
            "em_convergence": EM_TOLERANCE,
            "initial_m": "Splink 5.0.0 default comparison-level values",
            "between_passes": "Splink carries trained values forward; numeric estimates are combined by median.",
            "outcome_driven_retries": False,
        },
        "prediction": {
            "comparisons": "Same six comparisons and term-frequency settings as the existing informed baseline",
            "term_frequency_adjustments": True,
            "blocking_rule": "1=1",
            "input_slice": "data/evaluation_slice.csv",
            "expected_pairs": 225,
            "main_threshold": MAIN_THRESHOLD,
            "threshold_sensitivity": THRESHOLDS,
            "threshold_selection": "Fixed before fitting; no validation or test labels used",
        },
        "limitations": [
            "Random-pair u estimation assumes matches are rare among sampled pairs.",
            "EM assumes the specified comparison model and can reach a local solution.",
            "Training blocks can induce selection bias when conditional independence fails.",
            "The enriched 225-pair test slice does not establish population calibration or deployment accuracy.",
            "No live LLM calls or LOTUS optimizer measurements are part of this extension.",
        ],
        "sources": [
            "https://moj-analytical-services.github.io/splink/demos/examples/duckdb/febrl4.html",
            "https://moj-analytical-services.github.io/splink/topic_guides/training/training_rationale.html",
            "https://moj-analytical-services.github.io/splink/api_docs/training.html",
        ],
        "input_sha256": {
            name: _file_hash(support / "data" / name)
            for name in ["records.csv", "record_truth.csv", "evaluation_slice.csv"]
        },
        "implementation_sha256": _file_hash(__file__),
        "packages": {name: version(name) for name in ["splink", "duckdb", "pandas", "numpy"]},
    }


def _settings():
    return SettingsCreator(
        link_type="link_only", source_dataset_column_name="source",
        # This starting default is replaced by the deterministic-rule estimate.
        probability_two_random_records_match=0.0001,
        comparisons=[
            cl.NameComparison("given_name").configure(term_frequency_adjustments=True),
            cl.NameComparison("surname").configure(term_frequency_adjustments=True),
            cl.DateOfBirthComparison("date_of_birth", input_is_string=True,
                datetime_format="%Y%m%d", invalid_dates_as_null=True),
            cl.DamerauLevenshteinAtThresholds("soc_sec_id", [1, 2]),
            cl.ExactMatch("street_number").configure(term_frequency_adjustments=True),
            cl.DamerauLevenshteinAtThresholds("postcode", [1, 2]).configure(
                term_frequency_adjustments=True),
        ],
        blocking_rules_to_generate_predictions=["1=1"],
        retain_intermediate_calculation_columns=True,
        max_iterations=MAX_ITERATIONS, em_convergence=EM_TOLERANCE,
    )


def _observed_inputs(support):
    records = pd.read_csv(support / "data/records.csv", dtype=str, usecols=INPUT_COLUMNS)
    # No entity_id column is parsed or returned by this read.
    partition = pd.read_csv(support / "data/record_truth.csv", dtype=str,
                            usecols=["unique_id", "split"])
    train_ids = partition.loc[partition.split.eq("train"), ["unique_id"]]
    training = records.merge(train_ids, on="unique_id", validate="one_to_one")
    slice_ids = pd.read_csv(support / "data/evaluation_slice.csv", dtype=str)
    selected = slice_ids.merge(records, on=["source", "unique_id"], validate="one_to_one")
    a, b = (selected.loc[selected.source.eq(source), INPUT_COLUMNS].reset_index(drop=True)
            for source in ["febrl4a", "febrl4b"])
    if set(training.unique_id) & (set(a.unique_id) | set(b.unique_id)):
        raise ValueError("Training and evaluation records overlap.")
    if len(a) * len(b) != 225:
        raise ValueError("The fixed evaluation slice no longer has 225 pairs.")
    return training[INPUT_COLUMNS], a, b


def _matching_pairs(a, b, fields):
    """Count equality-rule pairs without treating missing values as agreements."""
    left = a[["unique_id", *fields]].dropna(subset=fields)
    right = b[["unique_id", *fields]].dropna(subset=fields)
    joined = left.merge(right, on=fields, suffixes=("_l", "_r"))
    return set(zip(joined.unique_id_l, joined.unique_id_r))


def _level_records(core):
    """Audit raw and effective values; private metadata is pinned to Splink 5.0.0."""
    records = []
    for comparison in core.comparisons:
        for level in comparison.comparison_levels:
            row = {"comparison": comparison.output_column_name,
                   "gamma": level.comparison_vector_value,
                   "label": level.label_for_charts, "is_null": level.is_null_level}
            if not level.is_null_level:
                row.update({
                    "m_raw": level._m_probability, "u_raw": level._u_probability,
                    "m_effective": level.m_probability, "u_effective": level.u_probability,
                    "m_has_numeric_estimate": level._has_estimated_m_values,
                    "u_has_numeric_estimate": level._has_estimated_u_values,
                })
            records.append(row)
    return records


def _session_audit(session, field, pair_count):
    history = []
    previous = None
    for iteration, core in enumerate(session._core_model_settings_history):
        levels = _level_records(core)
        parameters = {"prior": float(core.probability_two_random_records_match)}
        for level in levels:
            if not level["is_null"]:
                for parameter in ["m", "u"]:
                    key = f"{level['comparison']}:{level['gamma']}:{parameter}"
                    parameters[key] = float(level[f"{parameter}_effective"])
        delta = None if previous is None else max(
            abs(parameters[key] - previous[key]) for key in parameters)
        history.append({"iteration": iteration,
                        "training_match_fraction": parameters["prior"],
                        "max_abs_parameter_change": delta, "levels": levels})
        previous = parameters
    iterations = len(history) - 1
    final_delta = history[-1]["max_abs_parameter_change"]
    converged = final_delta is not None and final_delta < EM_TOLERANCE
    return {
        "block_field": field, "training_pairs": pair_count,
        "excluded_comparisons": [c.output_column_name for c in session._comparisons_that_cannot_be_estimated],
        "iterations": iterations, "iteration_cap": MAX_ITERATIONS,
        "tolerance": EM_TOLERANCE, "final_max_abs_parameter_change": final_delta,
        "convergence_criterion_met": converged,
        "cap_reached": iterations >= MAX_ITERATIONS,
        "stopping_reason": "parameter_tolerance" if converged else "iteration_cap",
        "history": history,
    }


def fit_unlabeled(training):
    """Accept observed fields and opaque IDs only; return model and training audit."""
    if version("splink") != REQUIRED_SPLINK:
        raise ValueError(f"Expected Splink {REQUIRED_SPLINK} for audited metadata.")
    if set(training.columns) != set(INPUT_COLUMNS):
        raise ValueError("Unlabeled fitting accepts only source, opaque unique_id and the six observed fields.")
    if not training.unique_id.is_unique or training.unique_id.isna().any():
        raise ValueError("Training record IDs must be complete and unique.")
    parts = [training.loc[training.source.eq(source), INPUT_COLUMNS].copy()
             for source in ["febrl4a", "febrl4b"]]
    if sum(map(len, parts)) != len(training) or any(part.empty for part in parts):
        raise ValueError("Expected nonempty FEBRL4a and FEBRL4b training tables.")
    db = DuckDBAPI()
    tables = [db.register(part, dataset_display_name=f"unlabeled_train_{index}")
              for index, part in enumerate(parts)]
    linker = Linker(tables, _settings(), log_level="WARNING")
    warning_stream = io.StringIO()
    handler = logging.StreamHandler(warning_stream)
    handler.setLevel(logging.WARNING)
    splink_logger = logging.getLogger("splink")
    splink_logger.addHandler(handler)
    started = perf_counter()
    try:
        rule_counts = [len(_matching_pairs(*parts, fields)) for fields in PRIOR_RULE_FIELDS]
        union = set().union(*(_matching_pairs(*parts, fields) for fields in PRIOR_RULE_FIELDS))
        total_pairs = len(parts[0]) * len(parts[1])
        linker.training.estimate_probability_two_random_records_match(
            [block_on(*fields) for fields in PRIOR_RULE_FIELDS],
            recall=ASSUMED_RECALL, record_sample_proportion=1.0)
        prior = float(linker.misc.save_model_to_json()["probability_two_random_records_match"])
        expected_prior = len(union) / ASSUMED_RECALL / total_pairs
        np.testing.assert_allclose(prior, expected_prior, rtol=1e-12, atol=0)
        if not 0 < prior < 1:
            raise ValueError("The declared prior rules did not produce a usable prior.")
        linker.training.estimate_u_using_random_sampling(
            max_pairs=1_000_000, seed=2026, min_count_per_level=None, num_chunks=1)
        u_after_sampling = _level_records(linker._settings_obj.core_model_settings)
        sessions = []
        for field in EM_FIELDS:
            pair_count = len(_matching_pairs(*parts, [field]))
            session = linker.training.estimate_parameters_using_expectation_maximisation(
                block_on(field), estimate_without_term_frequencies=True,
                fix_u_probabilities=True, fix_m_probabilities=False,
                fix_probability_two_random_records_match=False,
                populate_probability_two_random_records_match_from_trained_values=False)
            sessions.append(_session_audit(session, field, pair_count))
        model = linker.misc.save_model_to_json()
        if model["probability_two_random_records_match"] != prior:
            raise ValueError("EM unexpectedly changed the separately estimated global prior.")
        final_levels = _level_records(linker._settings_obj.core_model_settings)
        missing = [row for row in final_levels if not row["is_null"] and
                   not (row["m_has_numeric_estimate"] and row["u_has_numeric_estimate"])]
        zero_values = [{"comparison": row["comparison"], "gamma": row["gamma"],
                        "parameter": parameter} for row in final_levels if not row["is_null"]
                       for parameter in ["m", "u"] if row[f"{parameter}_effective"] == 0]
        u_fixed = all(
            (before.get("u_effective") == after.get("u_effective"))
            for before, after in zip(u_after_sampling, final_levels)
        )
        if not u_fixed:
            raise ValueError("EM changed a u probability that was declared fixed.")
        audit = {
            "rows": len(training), "source_rows": dict(zip(["febrl4a", "febrl4b"], map(len, parts))),
            "input_columns": list(training.columns),
            "observed_training_rows_sha256": _digest(training.fillna("").to_dict(orient="records")),
            "prior": {"probability": prior, "log2_odds": float(np.log2(prior / (1 - prior))),
                      "deterministic_rule_counts": rule_counts, "union_pairs": len(union),
                      "total_possible_pairs": total_pairs, "assumed_recall": ASSUMED_RECALL,
                      "estimated_matching_pairs": len(union) / ASSUMED_RECALL,
                      "unchanged_after_em": True},
            "em_sessions": sessions,
            "coverage": {"all_nonnull_levels_estimated": not missing,
                         "missing_estimates_with_effective_fallbacks": missing,
                         "zero_probability_levels": zero_values,
                         "levels": final_levels,
                         "u_unchanged_during_em": u_fixed},
            "warnings": warning_stream.getvalue().splitlines(),
            "fit_seconds": perf_counter() - started,
            "fitted_settings": model,
        }
        return linker, db, audit
    finally:
        splink_logger.removeHandler(handler)


def _score(linker, db, a, b):
    tables = [db.register(frame, dataset_display_name=f"unlabeled_eval_{index}")
              for index, frame in enumerate([a, b])]
    started = perf_counter()
    scores = linker.inference.predict_between(
        *tables, blocking_rules_to_generate_predictions=["1=1"]).as_pandas_dataframe()
    elapsed = perf_counter() - started
    scores = scores.rename(columns={"unique_id_l": "left_id", "unique_id_r": "right_id"})
    expected = a[["unique_id"]].rename(columns={"unique_id": "left_id"}).merge(
        b[["unique_id"]].rename(columns={"unique_id": "right_id"}), how="cross")
    if len(scores) != len(expected) or scores.duplicated(["left_id", "right_id"]).any():
        raise ValueError("Prediction did not return the fixed pair universe exactly once.")
    scores = expected.merge(scores[["left_id", "right_id", "match_probability", "match_weight"]],
                            on=["left_id", "right_id"], validate="one_to_one", how="left")
    if not np.isfinite(scores[["match_probability", "match_weight"]].to_numpy()).all():
        raise ValueError("An evaluated pair has a missing or nonfinite score.")
    if not scores.match_probability.between(0, 1).all():
        raise ValueError("An evaluated probability is outside [0,1].")
    return scores.to_dict(orient="records"), elapsed


def _metrics(truth, predictions):
    tp = int((truth & predictions).sum())
    fp = int((~truth & predictions).sum())
    fn = int((truth & ~predictions).sum())
    tn = int((~truth & ~predictions).sum())
    return {"pairs": len(truth), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def _evaluate_frozen(prediction_path, support):
    """The first identity-label read occurs after the prediction artifact exists."""
    frozen = json.loads(prediction_path.read_text())
    if _digest(frozen["predictions"]) != frozen["predictions_sha256"]:
        raise ValueError("Frozen predictions failed their integrity check.")
    # Identity labels enter only this evaluator, never fit_unlabeled or _score.
    truth = pd.read_csv(support / "data/record_truth.csv", dtype=str,
                        usecols=["unique_id", "entity_id"])
    entities = truth.set_index("unique_id").entity_id
    pairs = pd.DataFrame(frozen["predictions"])
    a = pairs.left_id.map(entities)
    b = pairs.right_id.map(entities)
    if a.isna().any() or b.isna().any():
        raise ValueError("Missing identity labels for evaluated pairs.")
    pairs["is_match"] = a.eq(b)
    rows = []
    for threshold in frozen["protocol"]["prediction"]["threshold_sensitivity"]:
        decision = pairs.match_probability >= threshold
        rows.append({"threshold": threshold, **_metrics(pairs.is_match, decision)})
    main = next(row for row in rows if row["threshold"] == MAIN_THRESHOLD)
    pairs["decision"] = pairs.match_probability >= MAIN_THRESHOLD
    errors = pairs.loc[pairs.decision.ne(pairs.is_match)].to_dict(orient="records")
    return {"evaluated_at_utc": _now(), "main": {"threshold": MAIN_THRESHOLD,
            "metrics": {key: value for key, value in main.items() if key != "threshold"},
            "errors": errors}, "threshold_sensitivity": rows,
            "true_matches": int(pairs.is_match.sum()),
            "truth_use": "Evaluation only, after predictions were frozen; no threshold optimization"}


def run_unlabeled_benchmark(support, *, result_dir=None):
    """Write a new protocol, frozen predictions and evaluated result; return its path."""
    support = Path(support).resolve()
    result_dir = Path(result_dir or support / "results")
    result_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"unlabeled-febrl4-{stamp}"
    protocol_path = result_dir / f"{stem}-protocol.json"
    prediction_path = result_dir / f"{stem}-predictions.json"
    result_path = result_dir / f"{stem}.json"
    protocol = _protocol(support)
    protocol_sha256 = _digest(protocol)
    _write_new(protocol_path, {"frozen_at_utc": _now(), "protocol_sha256": protocol_sha256,
                               "protocol": protocol})
    training, a, b = _observed_inputs(support)
    linker, db, audit = fit_unlabeled(training)
    predictions, scoring_seconds = _score(linker, db, a, b)
    frozen = {
        "schema": 1, "prediction_frozen_at_utc": _now(),
        "protocol": protocol, "protocol_sha256": protocol_sha256,
        "protocol_file": protocol_path.name,
        "protocol_file_sha256": _file_hash(protocol_path),
        "training": audit, "predictions": predictions,
        "predictions_sha256": _digest(predictions), "scoring_seconds": scoring_seconds,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "label_budget": {"fit_identity_labels": 0, "threshold_identity_labels": 0,
                         "prior_identity_labels": 0,
                         "partition_membership": "Reused existing entity-aware split and slice metadata",
                         "deterministic_prior": "Assumed strict-rule precision and recall; not labeled training"},
    }
    _write_new(prediction_path, frozen)
    evaluation = _evaluate_frozen(prediction_path, support)
    result = {**frozen, "prediction_file": prediction_path.name,
              "prediction_file_sha256": _file_hash(prediction_path), "evaluation": evaluation}
    _write_new(result_path, result)
    return result_path


def load_unlabeled_result(path, *, support):
    """Verify a saved local result and its immutable evidence before displaying it.

    The protocol's implementation hash identifies the fit-time source. Loader-only
    maintenance does not change that historical hash or require refitting.
    """
    path, support = Path(path), Path(support)
    result = json.loads(path.read_text())
    if result.get("schema") != 1:
        raise ValueError("Unsupported unlabeled benchmark result schema.")
    if _digest(result["protocol"]) != result["protocol_sha256"]:
        raise ValueError("Protocol content changed after it was frozen.")
    if _digest(result["predictions"]) != result["predictions_sha256"]:
        raise ValueError("Prediction content changed after it was frozen.")
    artifacts = {}
    for key in ["protocol_file", "prediction_file"]:
        artifact = path.parent / result[key]
        if _file_hash(artifact) != result[f"{key}_sha256"]:
            raise ValueError(f"The {key} integrity check failed.")
        artifacts[key] = json.loads(artifact.read_text())
    frozen = artifacts["prediction_file"]
    expected_keys = set(frozen) | {"prediction_file", "prediction_file_sha256", "evaluation"}
    if set(result) != expected_keys or any(result.get(key) != value for key, value in frozen.items()):
        raise ValueError("The evaluated result differs from its frozen prediction artifact.")
    declared = artifacts["protocol_file"]
    if (declared["protocol"] != result["protocol"] or
            declared["protocol_sha256"] != result["protocol_sha256"]):
        raise ValueError("The result differs from the original frozen protocol.")
    if not (declared["frozen_at_utc"] < result["prediction_frozen_at_utc"] <
            result["evaluation"]["evaluated_at_utc"]):
        raise ValueError("The saved protocol, prediction and evaluation chronology is invalid.")
    recorded_splink = result["protocol"]["packages"]["splink"]
    if recorded_splink != REQUIRED_SPLINK or version("splink") != recorded_splink:
        raise ValueError("Replay requires the pinned Splink version recorded in the protocol.")
    for name, digest in result["protocol"]["input_sha256"].items():
        if _file_hash(support / "data" / name) != digest:
            raise ValueError(f"Benchmark input changed: {name}")
    if len(result["predictions"]) != result["protocol"]["prediction"]["expected_pairs"]:
        raise ValueError("Saved pair coverage is incomplete.")
    pairs = pd.DataFrame(result["predictions"])
    if pairs.duplicated(["left_id", "right_id"]).any():
        raise ValueError("Saved predictions contain duplicate pairs.")
    slice_ids = pd.read_csv(support / "data/evaluation_slice.csv", dtype=str)
    a = slice_ids.loc[slice_ids.source.eq("febrl4a"), "unique_id"]
    b = slice_ids.loc[slice_ids.source.eq("febrl4b"), "unique_id"]
    expected = [{"left_id": left_id, "right_id": right_id} for left_id in a for right_id in b]
    if pairs[["left_id", "right_id"]].to_dict(orient="records") != expected:
        raise ValueError("Saved predictions differ from the fixed ordered pair universe.")
    numeric_scores = pairs[["match_probability", "match_weight"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_scores).all() or not pairs.match_probability.between(0, 1).all():
        raise ValueError("Saved predictions contain invalid scores.")
    return result


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    # The standalone run has no reason to connect to a network service.
    def block_network(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError(f"Network disabled for this local benchmark: {event}")
    sys.addaudithook(block_network)
    result_path = run_unlabeled_benchmark(Path(__file__).resolve().parent,
                                         result_dir=args.output_dir)
    result = load_unlabeled_result(result_path, support=Path(__file__).resolve().parent)
    print(result_path)
    print(json.dumps({"prior": result["training"]["prior"],
                      "em_sessions": [{key: value for key, value in session.items() if key != "history"}
                                      for session in result["training"]["em_sessions"]],
                      "missing_estimates": result["training"]["coverage"]["missing_estimates_with_effective_fallbacks"],
                      "main": result["evaluation"]["main"],
                      "threshold_sensitivity": result["evaluation"]["threshold_sensitivity"]}, indent=2))


if __name__ == "__main__":
    _main()
