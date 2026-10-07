"""Score one sealed, free CORDIS holdout comparison without changing predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

from baseline import read_lines
from export_predictions import verify_freeze
from prepare import DATA, HERE, digest, freeze_bytes, freeze_json, freeze_lines

OUT = HERE / "results" / "free-evaluation-v1"
METHODS = ["lexical_fixed_policy", "lexical_top1_forced", "splink_fixed_policy"]
QUESTION = {
    "question": "On 1065 held-out CORDIS queries, do the development accuracy, false-assignment and coverage trade-offs persist for three unchanged conventional policies?",
    "scope": "Historical CORDIS structured organization linkage against all 108476 ROR v1.41 entries; transductive unlabeled fitting already complete.",
    "methods": METHODS,
    "forced_top1_role": "Ranking diagnostic only; never an eligible operational gate comparator.",
    "prediction_changes": "None: no model calls, refitting, retrieval, threshold selection, prior selection, repairs or row exclusions.",
    "denominator": "Every one of the 1065 predeclared evaluation query IDs, including NIL, review, failed and inconsistent source annotations.",
    "metrics": "Complete decisions, automatic coverage, wrong automatic decisions, false/missed links, false-assignment queries, false NIL, links on NIL, review and failure.",
    "candidate_recall_ranks": [1, 5, 10, 25, 40],
    "bootstrap": {"replicates": 2000, "seed": 20261007,
        "unit": "Actual frozen connected groups of shared positive ROR identity or normalized duplicate inputs",
        "estimand": "Query-weighted metric rates; resample entire evaluation groups with replacement",
        "interval": "Percentile 95%; descriptive for this historical source and grouping, not NSO population or calibration uncertainty"},
    "paired_comparisons": "Every pair of the three frozen policies, with the same source-group bootstrap draws.",
    "source_policy_strata": ["historical NIL", "positive target", "inactive or withdrawn target", "flagged inconsistent duplicate-input labels"],
    "stopping": "One scoring pass; preserve every result and do not change methods from outcomes. Native affiliation sources remain unscored.",
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def seal():
    verify_freeze()
    exports = json.loads((HERE / "results" / "export-manifest.json").read_text())
    assert digest((HERE / "export_predictions.py").read_bytes()) == exports["code_sha256"]
    for item in exports["files"]:
        assert digest((HERE / item["path"]).read_bytes()) == item["sha256"]
    source = json.loads((DATA / "source-manifest.json").read_text())
    paths = ["baseline.py", "prepare.py", "export_predictions.py", "protocol.json",
        "data/source-manifest.json", "data/partitions.json", "results/freeze-manifest.json",
        "results/export-manifest.json", "results/conventional-evaluation.json",
        "results/candidates.jsonl.gz", "results/splink-fit-audit.json",
        "results/splink-pair-predictions.jsonl.gz"]
    return {
        "question": QUESTION,
        "scorer_sha256": digest(Path(__file__).read_bytes()),
        "prediction_and_fit_hashes": {p: digest((HERE / p).read_bytes()) for p in paths},
        "expected_gold_hashes_from_existing_curation_manifest": {
            x["path"]: x["sha256"] for x in source["prepared_files"]
            if x["path"] in {"data/gold/reference.jsonl", "data/gold/sampling-design.json", "data/gold/curation-audit.json"}},
    }


def freeze_question():
    sealed = seal()
    evaluator = HERE.parent / "evaluation.py"
    freeze_bytes(OUT / "evaluator-snapshot.py", evaluator.read_bytes())
    sealed["evaluator_sha256"] = digest((OUT / "evaluator-snapshot.py").read_bytes())
    path = OUT / "question-and-prediction-freeze.json"
    if path.exists():
        existing = json.loads(path.read_text())
        assert {k: v for k, v in existing.items() if k != "frozen_utc"} == sealed
    else:
        sealed["frozen_utc"] = stamp()
        freeze_json(path, sealed)
    print("Question and prediction hashes sealed; no gold opened.")


RATE_NAMES = ["complete_decision_accuracy", "automatic_coverage", "false_links_per_query",
    "missed_links_per_query", "false_assignment_query_rate", "false_nil_query_rate",
    "review_query_rate", "failed_query_rate", "wrong_automatic_per_query"]


def grouped_matrix(rows, group_index, query_groups):
    values = np.zeros((len(group_index), 1 + len(RATE_NAMES)))
    for r in rows:
        values[group_index[query_groups[r["query_id"]]]] += [1, r["correct"], r["automatic"],
            r["false_links"], r["missed_links"], r["false_assignment"], r["false_nil"],
            r["status"] == "review", r["status"] == "failed", r["automatic"] and not r["correct"]]
    return values


def rate_samples(values, draws):
    return np.asarray([sample[1:] / sample[0] for ids in draws for sample in [values[ids].sum(axis=0)]])


def intervals(samples):
    bounds = np.quantile(samples, [.025, .975], axis=0)
    return {name: bounds[:, i].tolist() for i, name in enumerate(RATE_NAMES)}


def score():
    path = OUT / "question-and-prediction-freeze.json"
    frozen = json.loads(path.read_text())
    current = seal()
    assert all(frozen[k] == v for k, v in current.items()), "Sealed code, policy, fitting or predictions changed"
    assert digest((OUT / "evaluator-snapshot.py").read_bytes()) == frozen["evaluator_sha256"]
    if (OUT / "audit.json").exists():
        audit = json.loads((OUT / "audit.json").read_text())
        for name, expected in audit["output_hashes"].items():
            assert digest((OUT / name).read_bytes()) == expected
        print("Existing sealed evaluation replay verified; no rescore performed.")
        return
    opened = stamp()
    # This is the first gold access; the separate question seal must already exist.
    for name, expected in frozen["expected_gold_hashes_from_existing_curation_manifest"].items():
        assert digest((HERE / name).read_bytes()) == expected, name
    partitions = json.loads((DATA / "partitions.json").read_text())
    ids = set(partitions["evaluation"])
    assert len(ids) == 1065
    references = {r["query_id"]: r for r in read_lines(DATA / "gold" / "reference.jsonl") if r["query_id"] in ids}
    assert set(references) == ids and all(r["source_population"] == "cordis" for r in references.values())
    truth = {q: r["target_ids"] for q, r in references.items()}
    sampling = json.loads((DATA / "gold" / "sampling-design.json").read_text())
    eval_groups = sampling["group_partitions"]["evaluation"]
    query_groups = {q: g for g in eval_groups for q in sampling["group_members"][g]}
    assert set(query_groups) == ids and len(eval_groups) == 1000
    group_index = {g: i for i, g in enumerate(sorted(eval_groups))}
    rng = np.random.default_rng(QUESTION["bootstrap"]["seed"])
    draws = rng.integers(0, len(group_index), size=(QUESTION["bootstrap"]["replicates"], len(group_index)))
    spec = importlib.util.spec_from_file_location("sealed_evaluation", OUT / "evaluator-snapshot.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    decisions = json.loads((HERE / "results" / "conventional-evaluation.json").read_text())
    assert set(decisions) == set(METHODS) and all(set(v) == ids for v in decisions.values())
    all_rows = {m: evaluator.outcomes(truth, decisions[m]) for m in METHODS}
    samples = {m: rate_samples(grouped_matrix(rows, group_index, query_groups), draws) for m, rows in all_rows.items()}
    summaries = {m: {**evaluator.summarize(rows), "family_bootstrap_95_percent_intervals": intervals(samples[m])} for m, rows in all_rows.items()}
    paired = {}
    for reference, candidate in itertools.combinations(METHODS, 2):
        a = {r["query_id"]: r for r in all_rows[reference]}
        z = {r["query_id"]: r for r in all_rows[candidate]}
        paired[f"{candidate}_minus_{reference}"] = {
            "reference": reference, "candidate": candidate,
            "corrections": sum(not a[q]["correct"] and z[q]["correct"] for q in ids),
            "regressions": sum(a[q]["correct"] and not z[q]["correct"] for q in ids),
            "family_bootstrap_95_percent_intervals": intervals(samples[candidate] - samples[reference]),
            "point_differences_per_query": {
                key: float((grouped_matrix(all_rows[candidate], group_index, query_groups).sum(axis=0)[i + 1]
                          - grouped_matrix(all_rows[reference], group_index, query_groups).sum(axis=0)[i + 1]) / len(ids))
                for i, key in enumerate(RATE_NAMES)},
        }
    candidates = defaultdict(list)
    for r in read_lines(HERE / "results" / "candidates.jsonl.gz"):
        if r["query_id"] in ids:
            candidates[r["query_id"]].append(r)
    recall = []
    for k in QUESTION["candidate_recall_ranks"]:
        found = total = complete = positives = 0
        for q, targets in truth.items():
            wanted = set(targets)
            retrieved = {r["record_id"] for r in candidates[q] if r["rank"] <= k}
            found += len(wanted & retrieved)
            total += len(wanted)
            positives += bool(wanted)
            complete += bool(wanted) and wanted <= retrieved
        recall.append({"k": k, "targets_found": found, "positive_targets": total,
                       "all_targets_found_positive_queries": complete, "positive_queries": positives})
    curation = json.loads((DATA / "gold" / "curation-audit.json").read_text())
    flagged = {q for r in curation["source_label_policy_flags"] for q in r["query_ids"]} & ids
    strata = {
        "historical_nil": {q for q, ts in truth.items() if not ts},
        "positive_target": {q for q, ts in truth.items() if ts},
        "inactive_or_withdrawn_target": {q for q, r in references.items() if any(s != "active" for s in r["target_statuses"])},
        "flagged_inconsistent_duplicate_labels": flagged,
    }
    source_strata = {name: {"queries": len(qs), "methods": {
        m: evaluator.summarize([r for r in rows if r["query_id"] in qs]) if qs else None
        for m, rows in all_rows.items()}} for name, qs in strata.items()}
    result = {"question": QUESTION["question"], "queries": len(ids), "source_groups": len(group_index),
        "historical_nil_queries": len(strata["historical_nil"]), "positive_queries": len(strata["positive_target"]),
        "methods": summaries, "candidate_recall": recall, "paired_comparisons": paired,
        "source_policy_strata": source_strata, "bootstrap": QUESTION["bootstrap"],
        "scope": "One unchanged free evaluation; no paid arm or operational gate, and no method changes from results.",
        "forced_top1_is_diagnostic": True, "native_sources_scored": False}
    freeze_json(OUT / "summary.json", result)
    freeze_lines(OUT / "query-outcomes.jsonl.gz", [
        {"method": m, "source_group": query_groups[r["query_id"]], **r} for m in METHODS for r in all_rows[m]])
    assert seal() == current, "Fit or prediction hashes changed during scoring"
    freeze_json(OUT / "audit.json", {
        "question_frozen_utc": frozen["frozen_utc"], "gold_first_opened_utc": opened, "completed_utc": stamp(),
        "question_freeze_sha256": digest(path.read_bytes()), "fit_and_prediction_hashes_unchanged": True,
        "gold_hashes_verified": frozen["expected_gold_hashes_from_existing_curation_manifest"],
        "evaluation_queries": len(ids), "actual_source_groups": len(group_index),
        "group_size_counts": dict(Counter(Counter(query_groups.values()).values())),
        "source_rows_excluded": 0, "paid_calls": 0, "fits_run": 0, "retrieval_runs": 0,
        "output_hashes": {name: digest((OUT / name).read_bytes()) for name in ["summary.json", "query-outcomes.jsonl.gz"]},
    })
    print(json.dumps({"methods": summaries, "candidate_recall": recall, "audit": "saved; fitting and prediction hashes unchanged"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    freeze_question() if args.freeze_only else score()
