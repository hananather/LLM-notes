"""Verify published native-affiliation evidence offline without changing files.

Uses Python's standard library only. The frozen experiment implementation is
hashed, never imported. No fitting, model calls, bootstrap reruns or writes occur.
"""
from __future__ import annotations

import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORGS = ROOT / "adversarial-2026/organizations"
ADAPTATION = ROOT / "native-affiliation-adaptation-20261009"


def read(path: Path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(description)


def same(actual, expected, description: str) -> None:
    if isinstance(expected, float):
        equal = isinstance(actual, (int, float)) and math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12
        )
    else:
        equal = actual == expected
    if not equal:
        if isinstance(expected, (dict, list, set)):
            raise ValueError(f"{description}: structured values differ")
        raise ValueError(f"{description}: {actual!r} != {expected!r}")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    return " ".join(re.findall(
        r"\w+", "".join(c for c in text if unicodedata.category(c) != "Mn")
    ))


def opaque(ror: str) -> str:
    return hashlib.sha256(("ror-1.41:" + ror).encode()).hexdigest()[:24]


def metric(states, gold, diagnostic=False):
    total = Counter()
    for qid, targets in gold.items():
        state = states[qid]
        status = state["status"]
        require(status in {"automatic", "review", "invalid"}, "Unknown response status")
        total[status] += 1
        valid = status == "automatic" or (diagnostic and status == "review")
        found = set(state["targets"]) if valid else set()
        truth = set(targets)
        total["exact"] += int(valid and found == truth)
        total["tp"] += len(found & truth)
        total["fp"] += len(found - truth)
        total["fn"] += len(truth - found)
        total["nil"] += int(not truth)
        total["nil_correct"] += int(valid and not truth and not found)
        total["false_nil"] += int(valid and bool(truth) and not found)
        total["false_assignment_nil"] += int(not truth and bool(found))
        total["conditional"] += int(status == "automatic" and found == truth)
    n = len(gold)
    tp, fp, fn = (total[k] for k in ["tp", "fp", "fn"])
    automatic = total["automatic"]
    return {
        "rows": n, "exact_sets": total["exact"],
        "exact_set_accuracy": total["exact"] / n if n else None,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "automatic_rows": automatic, "review_rows": total["review"],
        "invalid_rows": total["invalid"],
        "automatic_coverage": automatic / n if n else None,
        "loss_fp1": fp + fn, "loss_fp2": 2 * fp + fn, "loss_fp5": 5 * fp + fn,
        "nil_rows": total["nil"], "nil_correct": total["nil_correct"],
        "false_nil_rows": total["false_nil"],
        "false_assignment_on_nil_rows": total["false_assignment_nil"],
        "conditional_automatic_exact_sets": total["conditional"],
        "conditional_automatic_accuracy": total["conditional"] / automatic if automatic else None,
        "false_edges_per_100_queries": 100 * fp / n if n else None,
        "missing_edges_per_100_queries": 100 * fn / n if n else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def allocation(states, gold):
    reference, predicted = Counter(), Counter()
    for qid, targets in gold.items():
        reference.update(targets)
        if states[qid]["status"] == "automatic":
            predicted.update(states[qid]["targets"])
    ids = sorted(set(reference) | set(predicted))
    delta = {cid: predicted[cid] - reference[cid] for cid in ids}
    return {
        "reference_allocations": sum(reference.values()),
        "predicted_automatic_allocations": sum(predicted.values()),
        "organization_count_l1_error": sum(abs(v) for v in delta.values()),
        "overallocated_count_units": sum(max(0, v) for v in delta.values()),
        "underallocated_count_units": sum(max(0, -v) for v in delta.values()),
        "organizations_with_reference_allocations": len(reference),
        "organizations_with_nonzero_count_error": sum(v != 0 for v in delta.values()),
        "counts": [{"record_id": cid, "reference_count": reference[cid],
                    "automatic_predicted_count": predicted[cid], "difference": delta[cid]}
                   for cid in ids],
    }


def verify():
    protocol = read(HERE / "protocol.json")
    binding = read(HERE / "sources/source-binding.json")
    frozen = read(HERE / "results/protocol-seal.json")
    same(sha(HERE / "study.py"), frozen["code_sha256"], "Frozen experiment code")
    same(sha(HERE / "protocol.json"), frozen["protocol_sha256"], "Frozen protocol")
    same(set(binding["frozen_source_paths"]), set(protocol["source_sha256"]), "Source names")
    for name, relative in binding["frozen_source_paths"].items():
        same(sha(HERE / relative), protocol["source_sha256"][name], f"Public source {name}")
    for source in binding["published_files"]:
        path = HERE / source["path"]
        same(sha(path), source["sha256"], f"Published source {path.name}")
        same(path.stat().st_size, source["bytes"], f"Published source size {path.name}")

    checked = set()
    for filename in ["protocol-seal.json", "fit-and-policy-seal.json", "prediction-seal.json",
                     "reference-prediction-seal.json"]:
        seal = read(HERE / "results" / filename)
        require(seal["native_gold_opened"] is False, f"Unexpected gold access in {filename}")
        for relative, expected in seal["artifact_sha256"].items():
            same(sha(HERE / relative), expected, f"Sealed artifact {relative}")
            checked.add(relative)
    reference_freeze = read(HERE / "results/reference-protocol-seal.json")
    same(sha(HERE / "trained_reference.py"), reference_freeze["code_sha256"], "Frozen supplement code")
    same(sha(HERE / "reference-protocol.json"), reference_freeze["protocol_sha256"], "Frozen supplement protocol")

    manifest = read(ORGS / "data/source-manifest.json")
    for source in manifest["sources"]:
        if Path(source["file"]).stem in {"french", "multilingual", "multi-org"}:
            same(sha(ORGS / source["file"]), source["sha256"], "Publisher native source")
    reference_hash = next(r["sha256"] for r in manifest["prepared_files"]
                          if r["path"] == "data/gold/reference.jsonl")
    same(sha(ORGS / "data/gold/reference.jsonl"), reference_hash, "Prepared native labels")

    native = read(HERE / "data/native-inputs.json.gz")
    inputs = {r["query_id"]: r for r in native}
    gold, cohort = {}, {}
    for population in ["french", "multilingual", "multi-org"]:
        with (ORGS / "sources" / f"{population}.csv").open(newline="", encoding="utf-8-sig") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                qid = hashlib.sha256(f"{population}:{index}".encode()).hexdigest()[:24]
                gold[qid] = sorted(opaque(ror) for ror in set(re.findall(r"https://ror.org/[a-zA-Z0-9]+", row["label"])))
                cohort[qid] = population
    same(set(inputs), set(gold), "Native query IDs")
    same(len(gold), 1104, "Native denominator")
    prepared_truth = {r["query_id"]: sorted(r["target_ids"]) for r in
                      (json.loads(line) for line in (ORGS / "data/gold/reference.jsonl").read_text().splitlines())
                      if r["query_id"] in gold}
    same(prepared_truth, gold, "Publisher-derived reference targets")

    # Test annotations are skipped before parsing their label field.
    old = []
    with (HERE / "sources/gold_affiliation_annotations.csv").open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if row["split"] == "test":
                continue
            old.append({"query_id": f"s2aff-{index:04d}", "text": row["original_affiliation"],
                        "split": row["split"], "gold": sorted(opaque(v) for v in ast.literal_eval(row["labels"])
                                                               if v.startswith("https://ror.org/"))})
    held = {norm(r["text"]) for r in old if r["split"] == "val"} | {norm(r["text"]) for r in native}
    old = [r for r in old if r["split"] != "train" or norm(r["text"]) not in held]
    same(old, read(HERE / "data/original-training-validation.json.gz"), "Old training/validation packet")

    prepared = {r["query_id"]: r for r in read(HERE / "data/prepared-pairs.json.gz") if r["split"] == "native"}
    methods = {**read(HERE / "results/conventional-predictions.json.gz"),
               **read(HERE / "results/reference-predictions.json.gz")}
    semantic = {r["query_id"]: {"status": r["status"], "targets": sorted({m["record_id"] for m in
                r.get("answer", {}).get("matches", [])})} for r in read(HERE / "results/selection-answers.json.gz")}
    methods["semantic_selector"] = {"fixed": semantic}
    result = read(HERE / "results/evaluation.json")
    same(sha(HERE / "evaluate.py"), result["evaluator_sha256"], "Frozen evaluator")
    same(sha(HERE / "results/reference-policy-audit.json"), result["reference_audit_sha256"], "Reference audit")
    same(set(methods), set(result["scores"]), "Retained methods")
    groups = {"all": list(gold), **{c: [q for q in gold if cohort[q] == c] for c in ["french", "multilingual", "multi-org"]},
              "single_target": [q for q in gold if len(gold[q]) == 1],
              "multiple_targets": [q for q in gold if len(gold[q]) > 1],
              "historical_nil": [q for q in gold if not gold[q]],
              "candidate_complete_positive": [q for q in gold if gold[q] and set(gold[q]) <= set(prepared[q]["candidate_ids"])],
              "candidate_missing_positive": [q for q in gold if gold[q] and not set(gold[q]) <= set(prepared[q]["candidate_ids"])]}
    same({k: len(v) for k, v in groups.items()}, result["group_sizes"], "Group denominators")
    metric_rows = allocation_rows = outputs = 0
    for arm, policies in methods.items():
        same(set(policies), set(result["scores"][arm]), f"Policies for {arm}")
        for policy, states in policies.items():
            outputs += 1
            same(set(states), set(gold), f"Full denominator for {arm}/{policy}")
            for qid, state in states.items():
                same(len(state["targets"]), len(set(state["targets"])), "No repeated assignment")
                require(set(state["targets"]) <= set(prepared[qid]["candidate_ids"]), "Assignment outside common candidates")
            for group, qids in groups.items():
                truth = {qid: gold[qid] for qid in qids}
                for view, diagnostic in [("automatic", False), ("valid_set_diagnostic", True)]:
                    expected = metric(states, truth, diagnostic)
                    recorded = result["scores"][arm][policy][group][view]
                    same(set(expected), set(recorded), "Metric fields")
                    for key, value in expected.items():
                        same(recorded[key], value, f"{arm}/{policy}/{group}/{view}/{key}")
                    metric_rows += 1
                if group in {"all", "french", "multilingual", "multi-org"}:
                    recorded = result["allocation_count_errors"][arm][policy][group]
                    for key, value in allocation(states, truth).items():
                        same(recorded[key], value, f"{arm}/{policy}/{group}/{key}")
                    allocation_rows += 1
    for group in ["all", "french", "multilingual", "multi-org"]:
        qids = groups[group]
        total = sum(len(gold[q]) for q in qids)
        positive = sum(bool(gold[q]) for q in qids)
        for route in ["raw_retrieval_ids", "extracted_retrieval_ids", "candidate_ids"]:
            found = sum(len(set(gold[q]) & set(prepared[q][route])) for q in qids)
            complete = sum(bool(gold[q]) and set(gold[q]) <= set(prepared[q][route]) for q in qids)
            expected = {"target_total": total, "target_found": found, "target_recall": found / total if total else None,
                        "positive_queries": positive, "complete_positive_queries": complete,
                        "all_targets_per_positive_query_recall": complete / positive if positive else None}
            for key, value in expected.items():
                same(result["candidate_recall"][group][route][key], value, f"Candidate recall {group}/{route}/{key}")

    policies = read(HERE / "results/validation-policies.json")
    all_validation = {**policies["arms"], **read(HERE / "results/reference-validation-policies.json")["arms"]}
    raw = min((arm for arm in all_validation if arm.startswith("raw_") and not arm.startswith("raw_matched_")),
              key=lambda arm: (all_validation[arm]["policies"]["risk_2"]["loss_fp2"],
                               -all_validation[arm]["policies"]["risk_2"]["exact_sets"], arm))
    same(raw, result["primary_raw"], "Validation-selected raw champion")
    same(policies["best_extracted_risk2"], result["primary_extracted"], "Validation-selected extraction champion")

    prediction = read(HERE / "results/prediction-seal.json")
    reference_prediction = read(HERE / "results/reference-prediction-seal.json")
    adaptation = read(ADAPTATION / "prediction-seal.json")
    access = read(ADAPTATION / "label-access.json")
    same(sha(HERE / "results/prediction-seal.json"), result["prediction_seal_sha256"], "Main seal binding")
    same(sha(HERE / "results/reference-prediction-seal.json"), result["reference_prediction_seal_sha256"], "Supplement seal binding")
    same(sha(ADAPTATION / "prediction-seal.json"), result["adaptation_prediction_seal_sha256"], "Adaptation seal binding")
    same(sha(ADAPTATION / "label-access.json"), adaptation["sha256"]["label-access.json"], "Scoped label-access record")
    require(max(prediction["sealed_at"], reference_prediction["sealed_at"]) < access["opened_at_utc"]
            < adaptation["sealed_at_utc"] < result["main_evaluation_labels_opened_at"], "Prediction/label-access chronology")
    require(access["test_labels_loaded"] is False and adaptation["test_gold_opened"] is False, "Heldout adaptation boundary")
    return {"verified_at": datetime.now(timezone.utc).isoformat(), "verification": "PASS",
            "sealed_artifacts_verified": len(checked), "frozen_source_hashes_verified": len(protocol["source_sha256"]),
            "publisher_reference_rows": len(gold), "old_training_validation_rows": len(old),
            "method_policy_outputs_verified": outputs, "metric_view_group_rows_verified": metric_rows,
            "allocation_rows_verified": allocation_rows, "candidate_recall_verified": True,
            "validation_champions_verified": True, "label_access_chronology_verified": True,
            "paid_calls": 0, "files_written": 0, "experimental_reruns": 0,
            "interval_scope": "Saved bootstrap intervals are retained; this verifier checks counts and source/seal bindings without resampling."}


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
