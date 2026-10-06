"""Reconstruct descriptive results from complete frozen runs, without model calls.

Run from any directory. Only results/analysis/ is written; raw evidence is never
changed. Pending runs are reported as incomplete and excluded from comparisons.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "analysis"
READS: dict[str, str] = {}


def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    READS[str(relative)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def module(name):
    path = ROOT / f"{name}.py"
    READS[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(f"nso_analysis_{name}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def write(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    content = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def batch(relative, expected):
    if not (ROOT / relative).exists():
        return None, {"status": "incomplete", "completed": 0, "expected": len(expected)}
    value = read(relative)
    actual = [row["case_id"] for row in value["rows"]]
    if len(actual) != len(set(actual)) or not set(actual) <= set(expected):
        raise ValueError(f"Duplicate or unexpected cases in {relative}")
    if value["status"] != "complete":
        return None, {"status": "incomplete", "completed": len(actual), "expected": len(expected)}
    if set(actual) != set(expected) or value["spec"]["case_count"] != len(expected):
        raise ValueError(f"Complete batch does not contain its full denominator: {relative}")
    return value, {"status": "complete", "completed": len(actual), "expected": len(expected)}


def paired(rows):
    return {
        "n": len(rows), "baseline_correct": sum(r["baseline_correct"] for r in rows),
        "model_correct": sum(r["model_correct"] for r in rows),
        "corrected": sum(not r["baseline_correct"] and r["model_correct"] for r in rows),
        "regressed": sum(r["baseline_correct"] and not r["model_correct"] for r in rows),
        "both_wrong": sum(not r["baseline_correct"] and not r["model_correct"] for r in rows),
        "both_correct": sum(r["baseline_correct"] and r["model_correct"] for r in rows),
    }


def examples(rows):
    ordered = sorted(rows, key=lambda r: r["id"])
    groups = {
        "corrected": lambda r: not r["baseline_correct"] and r["model_correct"],
        "regressed": lambda r: r["baseline_correct"] and not r["model_correct"],
        "missed_candidate": lambda r: r.get("candidate_target_present") is False,
        "domain_invalid": lambda r: r.get("domain_invalid", False),
        "wrong_automatic": lambda r: r.get("automatic", False) and not r["model_correct"],
    }
    return {key: next((r for r in ordered if predicate(r)), None) for key, predicate in groups.items()}


def failure_partition(rows):
    """Mutually exclusive accounting; separate flags retain overlapping causes."""
    counts = Counter()
    for row in rows:
        if row["model_correct"]:
            continue
        if row.get("api_or_schema_invalid"):
            category = "api_or_schema_invalid"
        elif row.get("domain_invalid"):
            category = "domain_invalid"
        elif row.get("candidate_target_present") is False:
            category = "candidate_missing"
        else:
            category = "valid_decision_disagrees_with_reference"
        counts[category] += 1
    return {"counts": dict(counts), "precedence": ["api_or_schema_invalid", "domain_invalid", "candidate_missing", "valid_decision_disagrees_with_reference"],
            "note": "Validation and candidate failures can overlap; use case flags to inspect both. This precedence is accounting, not causal attribution."}


def load_ledger():
    path = ROOT / "results" / "api-ledger.jsonl"
    raw = path.read_bytes() if path.exists() else b""
    READS["results/api-ledger.jsonl"] = hashlib.sha256(raw).hexdigest()
    # A concurrent append may leave an unfinished last line. It is pending, not
    # an API failure. Every complete line must still parse successfully.
    complete = raw[:raw.rfind(b"\n") + 1]
    events = [json.loads(line) for line in complete.splitlines() if line.strip()]
    attempts = {r["tag"]: r for r in events if r["event"] == "attempt"}
    results = {r["tag"]: r for r in events if r["event"] == "result"}
    if len(attempts) != sum(r["event"] == "attempt" for r in events):
        raise ValueError("Duplicate attempt tags in ledger")
    if len(results) != sum(r["event"] == "result" for r in events):
        raise ValueError("Duplicate result tags in ledger")
    return attempts, results


def costs(attempts, results, tags):
    chosen = [results[t] for t in tags if t in results]
    usage = [r.get("response", {}).get("usage", {}) for r in chosen]
    elapsed = sorted(r["elapsed_seconds"] for r in chosen if "elapsed_seconds" in r)
    return {
        "attempted_calls": sum(t in attempts for t in tags), "finished_calls": len(chosen),
        "pending_calls": sum(t in attempts and t not in results for t in tags),
        "api_or_schema_invalid_calls": sum(r["status"] != "valid" for r in chosen),
        "input_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("completion_tokens", 0) for u in usage),
        "cache_write_tokens": sum(u.get("prompt_tokens_details", {}).get("cache_write_tokens", 0) for u in usage),
        "cached_input_tokens": sum(u.get("prompt_tokens_details", {}).get("cached_tokens", 0) for u in usage),
        "calls_with_reported_usage": sum(bool(u) for u in usage),
        "estimated_token_price_usd": sum(r.get("estimated_token_price_usd", 0) for r in chosen),
        "reserved_usd": sum(attempts[t]["reserved_usd"] for t in tags if t in attempts),
        "sum_call_seconds": sum(elapsed),
        "median_call_seconds": ((elapsed[(len(elapsed)-1)//2] + elapsed[len(elapsed)//2])/2 if elapsed else None),
        "maximum_call_seconds": max(elapsed) if elapsed else None,
        "price_boundary": "Recorded token-price estimate, not invoice or measured staff cost; reservations include headroom. Concurrent call times do not sum to wall time.",
    }


def verify_provider_records(records, attempts, results):
    if any(r["tag"] not in results for r in records if r["status"] == "valid"):
        # The batch may have completed between the first ledger snapshot and
        # reading the atomic result file. Refresh; do not call it a failure.
        latest_attempts, latest_results = load_ledger()
        attempts.update(latest_attempts)
        results.update(latest_results)
    for record in records:
        if record["status"] == "valid":
            saved = results[record["tag"]]
            assert saved["status"] == "valid" and saved["parsed"] == record["answer"]


def linkage_analysis(attempts, results):
    scoring = module("linkage")
    inputs = {r["case_id"]: r for r in read("data/linkage/s2aff-inputs.json")}
    gold = {r["case_id"]: r for r in read("data/linkage/s2aff-gold.json")}
    jobs = {r["case_id"]: r for r in read("data/linkage/s2aff-model-jobs.json")}
    registry = {r["id"]: r for r in read("data/linkage/s2aff-candidate-registry.json")}
    baseline = read("results/linkage/s2aff-baseline-predictions.json")["fused_selective_set"]
    source = read("data/linkage/sources.json")[0]
    output = {}
    for split in ("val", "test"):
        expected = [key for key, item in inputs.items() if item["split"] == split]
        raw, status = batch(f"results/linkage/model-{split}-v1.json", expected)
        if raw is None:
            output[split] = status
            continue
        verify_provider_records(raw["rows"], attempts, results)
        records = {r["case_id"]: r for r in raw["rows"]}
        rows = []
        for case_id in expected:
            record, job, truth = records[case_id], jobs[case_id], set(gold[case_id]["gold_ids"])
            valid, error, domain_invalid = record["status"] == "valid", None, False
            if valid:
                try:
                    scoring.validate_model_output(job, record["answer"])
                except (ValueError, KeyError, TypeError) as exc:
                    valid, domain_invalid, error = False, True, str(exc)
            else:
                error = record.get("error_type")
            prediction = set(record["answer"]["target_ids"]) if valid else set()
            candidate_ids = {c["id"] for c in job["user"]["candidates"]}
            describe = lambda ids: [{"id": target, "name": registry.get(target, {}).get("name")} for target in sorted(ids)]
            rows.append({
                "id": case_id, "split": split, "text": inputs[case_id]["text"],
                "source_label": {"url": source["url"], "sha256": source["sha256"], "zero_based_source_row": inputs[case_id]["source_row"], "prepared_reference": "data/linkage/s2aff-gold.json", "case_id": case_id, "non_ror_labels": gold[case_id]["non_ror_labels"]},
                "reference_targets": describe(truth), "baseline_targets": describe(baseline[case_id]), "model_targets": describe(prediction),
                "raw_prediction": record.get("answer"), "baseline_correct": set(baseline[case_id]) == truth,
                "model_correct": valid and prediction == truth, "domain_valid": valid,
                "api_or_schema_invalid": record["status"] != "valid", "domain_invalid": domain_invalid, "validation_error": error,
                "review_required": valid and record["answer"]["review_required"],
                "automatic": valid and not record["answer"]["review_required"],
                "candidate_target_present": truth <= candidate_ids, "missed_target_ids": sorted(truth - candidate_ids),
                "nil_reference": not truth, "multiple_reference_targets": len(truth) > 1,
                "false_target_assignments": len(prediction-truth), "missed_target_assignments": len(truth-prediction),
                "baseline_false_target_assignments": len(set(baseline[case_id])-truth),
            })
        automatic = [r for r in rows if r["automatic"]]
        target_frequency = Counter(target["id"] for r in rows for target in r["reference_targets"])
        summary = {**status, **paired(rows),
            "candidate_all_targets_present": sum(r["candidate_target_present"] for r in rows),
            "candidate_all_targets_present_nonempty": sum(r["candidate_target_present"] and not r["nil_reference"] for r in rows),
            "nonempty_target_rows": sum(not r["nil_reference"] for r in rows),
            "candidate_miss_case_ids": [r["id"] for r in rows if not r["candidate_target_present"]],
            "false_target_assignments": sum(r["false_target_assignments"] for r in rows),
            "baseline_false_target_assignments": sum(r["baseline_false_target_assignments"] for r in rows),
            "missed_target_assignments": sum(r["missed_target_assignments"] for r in rows),
            "review_flags": sum(r["review_required"] for r in rows),
            "review_flagged_correct": sum(r["review_required"] and r["model_correct"] for r in rows),
            "review_flagged_wrong": sum(r["review_required"] and not r["model_correct"] for r in rows),
            "domain_invalid": sum(r["domain_invalid"] for r in rows),
            "api_or_schema_invalid": sum(r["api_or_schema_invalid"] for r in rows),
            "automatic_count": len(automatic), "automatic_correct": sum(r["model_correct"] for r in automatic),
            "automatic_wrong": sum(not r["model_correct"] for r in automatic),
            "automatic_coverage": len(automatic)/len(rows),
            "automatic_nil_rows": sum(r["nil_reference"] for r in automatic),
            "automatic_nil_wrong": sum(r["nil_reference"] and not r["model_correct"] for r in automatic),
            "distinct_reference_organizations": len(target_frequency),
            "organizations_repeated_across_rows": sum(n > 1 for n in target_frequency.values()),
            "maximum_rows_per_reference_organization": max(target_frequency.values(), default=0),
            "cost": costs(attempts, results, [r["tag"] for r in raw["rows"]]), "wall_seconds": raw["wall_seconds"],
            "examples": examples(rows),
            "failure_partition": failure_partition(rows),
            "boundary": "Historical research-affiliation annotations and ROR registry; a proxy for organization identity work, not an NSO establishment test. Full annotation agreement includes review-flagged selections. Repeated organizations make independent-trial population intervals inappropriate.",
        }
        saved_score = ROOT / f"results/linkage/model-{split}-v1-score.json"
        if saved_score.exists():
            stored = read(str(saved_score.relative_to(ROOT)))
            assert stored["all_cases"]["exact_sets"] == summary["model_correct"]
            assert stored["all_cases"]["fp"] == summary["false_target_assignments"]
            assert stored["automatic_cases"]["exact_sets"] == summary["automatic_correct"]
        write(f"linkage-{split}-cases.json", rows)
        output[split] = summary
    return output


def coding_analysis(attempts, results):
    scoring = module("coding")
    cases = {r["id"]: r for r in read("data/coding/cases.json")}
    references = {r["id"]: r for r in read("data/coding/facts-gold.json")}
    jobs = {r["case_id"]: r for r in read("results/coding/audited-v2-all-direct-jobs.json")}
    baseline_saved = read("results/coding/audited-v2-baseline.json")
    baseline = {r["id"]: r for r in baseline_saved["rows"]}
    evidence = read("data/coding/source-evidence.json")
    sources = {s["filename"]: s for s in read("data/coding/sources.json")}
    output = {"official_exact_controls": {"correct": baseline_saved["official_control_correct"], "n": len(baseline_saved["official_control_rows"]), "model_calls": 0}}
    for split in ("dev", "test"):
        expected = [key for key, case in cases.items() if case["split"] == split]
        raw, status = batch(f"results/coding/model-{split}-v2.json", expected)
        if raw is None:
            output[split] = status
            continue
        verify_provider_records(raw["rows"], attempts, results)
        records = {r["case_id"]: r for r in raw["rows"]}
        rows = []
        for case_id in expected:
            record, case, job = records[case_id], cases[case_id], jobs[case_id]
            reference = references[case_id]["reference"]
            assert baseline[case_id]["reference"] == reference
            valid, error, domain_invalid = record["status"] == "valid", None, False
            if valid:
                try:
                    scoring.validate_selection(record["answer"], job["user"]["candidates"], case["text"])
                except (ValueError, KeyError, TypeError) as exc:
                    valid, domain_invalid, error = False, True, str(exc)
            else:
                error = record.get("error_type")
            prediction = record.get("answer") if valid else {"decision": "failed", "code": None}
            candidates = {c["code"] for c in job["user"]["candidates"]}
            labels = []
            for key in references[case_id]["source_keys"]:
                label = dict(evidence[key])
                label["source_urls"] = {name: item["url"] for name, item in sources.items() if name.startswith(case["scheme"])}
                labels.append(label)
            rows.append({
                "id": case_id, "scheme": case["scheme"], "split": split, "family": case["family"],
                "variant": case["variant"], "text": case["text"], "reference": reference,
                "source_label": {"prepared_reference": "data/coding/facts-gold.json", "case_id": case_id, "source_evidence": labels},
                "baseline_prediction": baseline[case_id]["prediction"], "raw_prediction": record.get("answer"), "prediction": prediction,
                "baseline_correct": scoring.decision_correct(baseline[case_id]["prediction"], reference),
                "model_correct": valid and scoring.decision_correct(prediction, reference),
                "domain_valid": valid, "api_or_schema_invalid": record["status"] != "valid", "domain_invalid": domain_invalid, "validation_error": error,
                "automatic": valid and prediction["decision"] == "code", "review_required": valid and prediction["decision"] == "review",
                "candidate_target_present": reference["code"] in candidates if reference["decision"] == "code" else None,
                "review_alternatives_present": set(reference["compatible_codes"]) <= candidates if reference["decision"] == "review" else None,
                "unsupported_single_code_on_review": reference["decision"] == "review" and prediction["decision"] == "code",
            })
        output[split] = {"status": "complete", "schemes": {}}
        for scheme in ("naics", "noc"):
            group = [r for r in rows if r["scheme"] == scheme]
            unique = [r for r in group if r["reference"]["decision"] == "code"]
            review = [r for r in group if r["reference"]["decision"] == "review"]
            automatic = [r for r in group if r["automatic"]]
            summary = {**paired(group), "families": len({r["family"] for r in group}),
                "unique_target_n": len(unique), "unique_target_correct": sum(r["model_correct"] for r in unique),
                "candidate_target_present": sum(r["candidate_target_present"] for r in unique),
                "candidate_miss_case_ids": [r["id"] for r in unique if not r["candidate_target_present"]],
                "review_n": len(review), "review_correct": sum(r["model_correct"] for r in review),
                "review_alternatives_present": sum(r["review_alternatives_present"] for r in review),
                "unsupported_single_code_on_review": sum(r["unsupported_single_code_on_review"] for r in group),
                "domain_invalid": sum(r["domain_invalid"] for r in group),
                "api_or_schema_invalid": sum(r["api_or_schema_invalid"] for r in group),
                "review_outputs": sum(r["review_required"] for r in group),
                "review_outputs_on_unique_target": sum(r["review_required"] for r in unique),
                "automatic_count": len(automatic), "automatic_wrong": sum(not r["model_correct"] for r in automatic),
                "cost": costs(attempts, results, [records[r["id"]]["tag"] for r in group]),
                "by_family": {family: paired([r for r in group if r["family"] == family]) for family in sorted({r["family"] for r in group})},
                "examples": examples(group),
                "failure_partition": failure_partition(group),
                "boundary": "Source-grounded synthetic convenience families, related variants, authored references. No population confidence interval, production error-rate estimate, or measured staff-time saving.",
            }
            if split == "dev":
                summary["development_gate_passed"] = summary["model_correct"] - summary["baseline_correct"] >= 2 and summary["unsupported_single_code_on_review"] == 0
            output[split]["schemes"][scheme] = summary
        saved_score = ROOT / f"results/coding/model-{split}-v2-score.json"
        if saved_score.exists():
            stored = read(str(saved_score.relative_to(ROOT)))
            for scheme in ("naics", "noc"):
                assert stored["summaries"][f"{scheme}_{split}"]["decision_correct"] == output[split]["schemes"][scheme]["model_correct"]
        write(f"coding-{split}-cases.json", rows)
    return output


def statistics_analysis(attempts, results):
    scoring = module("statistics")
    questions = {r["id"]: r for r in read("data/statistics/questions.json") if r["split"] == "test"}
    references = read("data/statistics/reference.json")
    retrieval = read("results/statistics/catalogue-baseline-test.json")
    retrieval_rows = {r["id"]: r for r in retrieval["records"]}
    manifest_path = ROOT / "data/statistics/execution-manifest.json"
    output = {"catalogue_baseline": {k: v for k, v in retrieval.items() if k != "records"},
        "boundary": "Authored test requests. Numerical arms receive four known tables' metadata and query 1–12 rows per answer; 401,944 retained rows are queryable data, not model context. Separate discovery searches 8,270 catalog records; its gold is designated-source recovery, not exhaustive relevance. Review scoring checks action and nonempty reason, not reason quality."}
    if not manifest_path.exists():
        output["status"] = "incomplete"
        output["reason"] = "Awaiting frozen statistics execution manifest and complete results."
        return output
    contract = read("data/statistics/execution-manifest.json")
    execution_sha = READS["data/statistics/execution-manifest.json"]
    output["execution_manifest_sha256"] = execution_sha
    baseline_path = f"results/statistics/baseline-test-{execution_sha[:16]}.json"
    baseline = read(baseline_path) if (ROOT / baseline_path).exists() else None
    if baseline is None:
        output.update(status="incomplete", reason="Awaiting baseline under active execution manifest.")
        return output
    baseline_rows = {r["id"]: r for r in baseline["records"]}
    assert set(baseline_rows) == set(questions)
    output["baseline"] = baseline["summary"]
    numerical = None
    scored_arms = {}
    for arm in ("one_shot", "agent", "discovery"):
        expected = [key for key in questions if arm != "discovery" or references[key]["outcome"]["status"] == "answer"]
        relative = f"results/statistics/{arm}-test-{execution_sha[:16]}.json"
        if not (ROOT / relative).exists():
            output[arm] = {"status": "incomplete", "expected": len(expected)}
            continue
        saved = read(relative)
        assert saved["execution_manifest_sha256"] == execution_sha
        assert saved["implementation_sha256"] == contract["files"]["statistics.py"]
        records = saved["records"]
        assert [r["id"] for r in records] == expected
        if any(request["tag"] not in results for r in records for request in r["requests"] if request["status"] == "valid"):
            latest_attempts, latest_results = load_ledger()
            attempts.update(latest_attempts)
            results.update(latest_results)
        rows, tags = [], []
        for record in records:
            case_id, reference = record["id"], references[record["id"]]
            raw = {k: v for k, v in record.items() if k != "record_sha256"}
            assert record["record_sha256"] == hashlib.sha256(scoring.canonical_bytes(raw)).hexdigest()
            tags.extend(r["tag"] for r in record["requests"])
            for request in record["requests"]:
                assert request["payload_sha256"] == hashlib.sha256(scoring.canonical_bytes(request["payload"])).hexdigest()
                if request["status"] == "valid":
                    assert attempts[request["tag"]]["payload"] == request["payload"]
                    provider = results[request["tag"]]
                    assert provider["status"] == "valid"
                    assert request["parsed_sha256"] == hashlib.sha256(scoring.canonical_bytes(provider["parsed"])).hexdigest()
            outcome = record["outcome"]
            if arm == "discovery":
                candidates = retrieval_rows[case_id]["candidate_ids"]
                proposed = record.get("plan", {})
                model_correct = proposed.get("status") == "selected" and proposed.get("product_id") in reference["expected_product_ids"] and proposed.get("product_id") in candidates
                score = {"complete": model_correct}
                baseline_correct = retrieval_rows[case_id]["top1"]
                candidate_present = retrieval_rows[case_id]["recall_at_12"]
                domain_invalid = proposed.get("status") == "selected" and proposed.get("product_id") not in candidates
            else:
                if numerical is None:
                    numerical = scoring.FrozenStatistics()
                    for baseline_record in baseline_rows.values():
                        assert scoring.evaluate_plan(numerical, baseline_record["plan"]) == baseline_record["outcome"]
                if "plan" in record and (arm == "one_shot" or any(t["action"] == "final" for t in record["trace"])):
                    assert scoring.evaluate_plan(numerical, record["plan"]) == outcome
                score = scoring.certificate_score(outcome, reference)
                baseline_correct = scoring.certificate_score(baseline_rows[case_id]["outcome"], reference)["complete"]
                candidate_present = None
                domain_invalid = outcome["status"] == "error" and all(r["status"] == "valid" for r in record["requests"])
            assert score == record["score"]
            rows.append({"id": case_id, "family": questions[case_id]["family"], "question": questions[case_id]["question"],
                "source_label": {"prepared_reference": "data/statistics/reference.json", "case_id": case_id, "source_row_ids": reference["outcome"].get("source_row_ids", []), "expected_product_ids": reference["expected_product_ids"], "table_urls": [f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={pid}01" for pid in reference["expected_product_ids"]]},
                "reference": reference, "baseline_prediction": retrieval_rows[case_id] if arm == "discovery" else baseline_rows[case_id],
                "prediction": record.get("plan"), "outcome": outcome, "trace": record["trace"], "score": score,
                "baseline_correct": baseline_correct, "model_correct": score["complete"], "candidate_target_present": candidate_present,
                "domain_invalid": domain_invalid, "api_or_schema_invalid": any(r["status"] != "valid" for r in record["requests"]),
                "review_required": outcome["status"] == "review", "automatic": outcome["status"] in {"answer", "selected"},
                "execute_calls": sum(t["action"] == "execute" for t in record["trace"]), "model_calls": len(record["requests"]),
                "trace_errors": sum(t.get("observation", {}).get("status") == "error" for t in record["trace"]),
            })
        summary = {"status": "complete", "metric": "designated_source_recovery" if arm == "discovery" else "complete_certificate_or_review_action_correctness", **paired(rows),
            "review_outputs": sum(r["review_required"] for r in rows),
            "wrong_nonabstaining_answers": sum(r["automatic"] and not r["model_correct"] for r in rows),
            "domain_invalid": sum(r["domain_invalid"] for r in rows), "api_or_schema_invalid": sum(r["api_or_schema_invalid"] for r in rows),
            "execute_calls": sum(r["execute_calls"] for r in rows), "trace_errors": sum(r["trace_errors"] for r in rows),
            "cost": costs(attempts, results, tags), "examples": examples(rows)}
        if arm == "discovery":
            summary["candidate_ceiling"] = sum(r["candidate_target_present"] for r in rows)
            summary["candidate_miss_case_ids"] = [r["id"] for r in rows if not r["candidate_target_present"]]
            summary["selected_nonreference_tables"] = summary["wrong_nonabstaining_answers"]
            summary["nonreference_selection_boundary"] = "These fail designated-source recovery; other tables have not been adjudicated for relevance."
        else:
            summary["continuation_gate_passed"] = summary["model_correct"] >= 32 and summary["wrong_nonabstaining_answers"] <= 1
            answerable = [r for r in rows if r["reference"]["outcome"]["status"] == "answer"]
            reviews = [r for r in rows if r["reference"]["outcome"]["status"] == "review"]
            summary.update(answerable_n=len(answerable), answerable_complete=sum(r["model_correct"] for r in answerable),
                required_review_n=len(reviews), required_review_correct=sum(r["model_correct"] for r in reviews),
                source_correct_count=sum(r["score"]["source_correct"] for r in answerable),
                value_and_unit_correct_count=sum(r["score"]["value_correct"] for r in answerable))
        assert saved["summary"]["complete_correct"] == summary["model_correct"]
        output[arm] = summary
        scored_arms[arm] = rows
        write(f"statistics-{arm}-test-cases.json", rows)
    output["status"] = "complete" if all(output[a]["status"] == "complete" for a in ("one_shot", "agent", "discovery")) else "incomplete"
    if output["one_shot"]["status"] == output["agent"]["status"] == "complete":
        one, agent = output["one_shot"], output["agent"]
        one_tokens = one["cost"]["input_tokens"] + one["cost"]["output_tokens"]
        agent_tokens = agent["cost"]["input_tokens"] + agent["cost"]["output_tokens"]
        output["agent_increment"] = {"additional_complete_answers": agent["model_correct"]-one["model_correct"],
            "token_ratio": agent_tokens/one_tokens if one_tokens else None,
            "additional_estimated_token_price_usd": agent["cost"]["estimated_token_price_usd"]-one["cost"]["estimated_token_price_usd"],
            "required_gain_when_token_ratio_above_two": 2}
        one_by_id = {r["id"]: r for r in scored_arms["one_shot"]}
        incremental = [{**r, "baseline_correct": one_by_id[r["id"]]["model_correct"]} for r in scored_arms["agent"]]
        output["agent_increment"]["paired_vs_one_shot"] = paired(incremental)
        output["agent_increment"]["examples"] = examples(incremental)
    return output


def main():
    attempts, results = load_ledger()
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
        "linkage": linkage_analysis(attempts, results), "coding": coding_analysis(attempts, results),
        "nces": read("results/linkage/nces-baseline.json"), "statistics": statistics_analysis(attempts, results),
        "ledger_snapshot": costs(attempts, results, sorted(attempts)),
        "example_selection": "First case ID in each correction, regression, missed-candidate, domain-invalid and wrong-automatic category; null means no such observed case.",
        "interpretation": "Separate descriptive task results; no pooled ranking, national confidence intervals, or invented human minutes. Complete-run raw evidence is preserved; incomplete runs are not scored."}
    report["inputs_sha256"] = READS
    write("summary.json", report)
    lines = ["# Offline result analysis", "", f"Snapshot: {report['generated_utc']}", "",
        "Counts below use complete saved runs and reconstructed task validation. The full paired cases, source-label references and deterministic examples are in the JSON files beside this report.", "",
        "| Task | Split | Baseline correct | Model correct | Corrections | Regressions |", "|---|---|---:|---:|---:|---:|"]
    for split, summary in report["linkage"].items():
        if summary["status"] == "complete":
            lines.append(f"| Affiliation exact target sets | {split} | {summary['baseline_correct']}/{summary['n']} | {summary['model_correct']}/{summary['n']} | {summary['corrected']} | {summary['regressed']} |")
    for split in ("dev", "test"):
        panel = report["coding"][split]
        if panel["status"] == "complete":
            for scheme, summary in panel["schemes"].items():
                lines.append(f"| {scheme.upper()} synthetic decisions | {split} | {summary['baseline_correct']}/{summary['n']} | {summary['model_correct']}/{summary['n']} | {summary['corrected']} | {summary['regressed']} |")
    for arm in ("one_shot", "agent", "discovery"):
        summary = report["statistics"].get(arm, {})
        if summary.get("status") == "complete":
            lines.append(f"| Statistics {arm} | test | {summary['baseline_correct']}/{summary['n']} | {summary['model_correct']}/{summary['n']} | {summary['corrected']} | {summary['regressed']} |")
    lines += ["", f"Statistics model-run status: **{report['statistics']['status']}**.", "",
        "NCES school continuity stopped before model calls: 300/300 probability-sample matches were correct. The separate changed-name-and-address panel was 72/100, with 96/100 targets retrieved; these populations are not pooled.", "",
        "Affiliation scores include review-flagged selections. Automatic acceptance, review flags and domain-invalid responses are reported separately. Synthetic coding families and authored statistical questions do not estimate national workload accuracy. Exact source quotes verify text presence, not semantic entailment. API price is not human effort or operational savings.", "",
        "Rebuild without model calls: `.venv/bin/python LLM_basics/nso-semantic-workflows/analyze.py`.", ""]
    write("README.md", "\n".join(lines))
    print(json.dumps({"output": str(OUTPUT), "statistics_status": report["statistics"]["status"], "finished_calls": report["ledger_snapshot"]["finished_calls"], "pending_calls": report["ledger_snapshot"]["pending_calls"]}))


if __name__ == "__main__":
    main()
