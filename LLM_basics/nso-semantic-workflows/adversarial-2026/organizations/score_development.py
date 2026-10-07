"""Score only the predeclared development set after the offline freeze is verified."""
from collections import defaultdict
import importlib.util
import json

from baseline import read_lines
from prepare import DATA, HERE, digest, freeze_json


def metrics(gold, decisions):
    count = len(gold)
    exact = false_targets = missed_targets = nil_errors = false_nil = automatic = automatic_wrong = reviews = 0
    for qid, true in gold.items():
        pred = decisions.get(qid)
        if pred is None:
            raise ValueError(f"Missing declared development prediction: {qid}")
        targets = set(pred["target_ids"])
        reviewed = pred["decision"] == "review"
        exact += (not reviewed and targets == true)
        false_targets += len(targets - true)
        missed_targets += len(true - targets)
        nil_errors += not true and bool(targets)
        false_nil += bool(true) and pred["decision"] == "nil"
        reviews += reviewed
        automatic += not reviewed
        automatic_wrong += not reviewed and targets != true
    return {"queries": count, "correct_complete_decisions": exact, "false_target_assignments": false_targets,
        "missed_targets": missed_targets, "false_assignment_on_nil_rows": nil_errors, "false_nil_rows": false_nil,
        "review_rows": reviews, "automatic_rows": automatic, "automatic_wrong": automatic_wrong,
        "accepted_coverage": automatic / count, "error_among_automatic": automatic_wrong / automatic if automatic else None}


def main():
    out = HERE / "results"
    manifest = json.loads((out / "freeze-manifest.json").read_text())
    if digest((HERE / "baseline.py").read_bytes()) != manifest["code_sha256"]:
        raise ValueError("Baseline code changed after predictions froze")
    if digest((HERE / "prepare.py").read_bytes()) != manifest["normalizer_code_sha256"]:
        raise ValueError("Normalization code changed after predictions froze")
    if digest((HERE / "protocol.json").read_bytes()) != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed after predictions froze")
    for filename, expected in manifest["input_hashes"].items():
        if digest((DATA / filename).read_bytes()) != expected:
            raise ValueError(f"Input changed: {filename}")
    for entry in manifest["files"]:
        if digest((HERE / entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"Prediction/fit artifact changed: {entry['path']}")
    exports = json.loads((out / "export-manifest.json").read_text())
    if digest((HERE / "export_predictions.py").read_bytes()) != exports["code_sha256"]:
        raise ValueError("Export code changed after decisions froze")
    for entry in exports["files"]:
        if digest((HERE / entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"Export changed: {entry['path']}")
    evaluator_path = HERE.parent / "evaluation.py"
    freeze_json(out / "development-evaluator-freeze.json", {
        "score_code_sha256": digest((HERE / "score_development.py").read_bytes()),
        "common_evaluator_sha256": digest(evaluator_path.read_bytes()),
        "export_manifest_sha256": digest((out / "export-manifest.json").read_bytes()),
        "scope": "Development outcomes only; saved before opening labels",
    })
    dev = set(json.loads((DATA / "partitions.json").read_text())["development"])
    gold = {r["query_id"]: set(r["target_ids"]) for r in read_lines(DATA / "gold" / "reference.jsonl") if r["query_id"] in dev}
    lexical = {r["query_id"]: r for r in read_lines(out / "lexical-predictions.jsonl") if r["query_id"] in dev}
    candidates = defaultdict(list)
    for r in read_lines(out / "candidates.jsonl.gz"):
        if r["query_id"] in dev:
            candidates[r["query_id"]].append(r)
    recall = []
    positives = sum(bool(ids) for ids in gold.values())
    for k in [1, 5, 10, 25, 40]:
        all_found = found = total = 0
        for qid, ids in gold.items():
            retrieved = {r["record_id"] for r in candidates[qid] if r["rank"] <= k}
            found += len(ids & retrieved)
            total += len(ids)
            all_found += bool(ids) and ids <= retrieved
        recall.append({"k": k, "targets_found": found, "positive_targets": total,
            "all_targets_found_positive_queries": all_found, "positive_queries": positives})
    fs = defaultdict(list)
    for r in read_lines(out / "splink-pair-predictions.jsonl.gz"):
        if r["query_id"] in dev:
            fs[r["query_id"]].append(r)
    fs_decisions = {}
    for qid, rows in fs.items():
        rows.sort(key=lambda r: (-r["match_probability"], r["record_id"]))
        top = rows[0]["match_probability"]
        second = rows[1]["match_probability"] if len(rows) > 1 else 0.
        decision = "linked" if top >= .99 and second < .5 else "nil" if top < .1 else "review"
        fs_decisions[qid] = {"query_id": qid, "decision": decision, "target_ids": [rows[0]["record_id"]] if decision == "linked" else []}
    top1 = {qid: {"decision": "linked", "target_ids": r["top1_target_ids"]} for qid, r in lexical.items()}
    spec = importlib.util.spec_from_file_location("common_evaluator", evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    normalized = json.loads((out / "conventional-development.json").read_text())
    common_metrics = {method: evaluator.summarize(evaluator.outcomes(gold, decisions))
                      for method, decisions in normalized.items()}
    result = {"population": "CORDIS predeclared development groups only", "test_metrics_opened": False,
        "n": len(dev), "positive_queries": positives, "historical_nil_queries": len(dev) - positives,
        "lexical_fixed_policy": metrics(gold, lexical), "lexical_top1_forced": metrics(gold, top1), "candidate_recall": recall,
        "splink_fixed_policy": metrics(gold, fs_decisions) if fs_decisions else {"status": "not_scored_failed_fit"},
        "common_evaluator_metrics": common_metrics,
        "interpretation": "Development evidence only. Review is not a correct identity decision. Thresholds and retrieval are not tuned from these outcomes.",
        "freeze_manifest_sha256": digest((out / "freeze-manifest.json").read_bytes())}
    freeze_json(out / "development-score.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
