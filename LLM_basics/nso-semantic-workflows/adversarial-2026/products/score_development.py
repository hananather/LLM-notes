"""Score only the frozen product DEVELOPMENT experiment; never call a model API."""
from __future__ import annotations

import argparse
from decimal import Decimal
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import decision_adapters as adapters
import evaluation
import model_jobs
import runtime
import baseline

MANIFEST = ROOT / "contracts/products-development-v1.json"
MANIFEST_SHA256 = "792ed7904c721d96a56b159531abb3010759d4d198b03a8302f4da785ed797c3"
MANIFEST_FILE_SHA256 = "6371cc2fce29aa1d2c52853b1b2f3b46bc96af536e3d9ebf638cb0148229db8f"
OUTPUT = HERE / "results/model-development-v1"
BASELINE = HERE / "results/baseline-v6/amazon-google/dev"
PRIMARY_FS = "assumed_recall=0.8;threshold=0.9"
ELIGIBLE = ["lexical_all_pairs", "splink_all_pairs"]


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    baseline.freeze(OUTPUT / name, value)


def verify_contract():
    if baseline.file_hash(MANIFEST) != MANIFEST_FILE_SHA256:
        raise ValueError("Development request manifest changed")
    manifest = read(MANIFEST)
    if runtime.digest(manifest) != MANIFEST_SHA256:
        raise ValueError("Canonical development manifest changed")
    if manifest["stage"] != "development" or manifest["contract"]["query_count"] != 144:
        raise ValueError("Only the frozen 144-query development denominator is permitted")
    if manifest["contract"]["gate_eligible_methods"] != ELIGIBLE:
        raise ValueError("Prospective reference methods changed")
    if len(manifest["jobs"]) != 640 or any(not j["tag"].startswith("products/dev/") for j in manifest["jobs"]):
        raise ValueError("Expected exactly the frozen 640 development jobs")
    model_jobs.verify_bindings(manifest)
    runtime.verify_manifest(manifest)
    return manifest


def development_data():
    location = HERE / "data/amazon-google/v2"
    partitions = read(location / "partitions.json")["dev"]
    left, right = baseline.load_observed("amazon-google", "dev")
    truth = {rid: [] for rid in partitions["left"]}
    right_ids = set(partitions["right"])
    # The source reference file spans both splits; only development edges are
    # retained, evaluated or written. Test decisions are never opened here.
    for edge in read(location / "truth.json")["edges"]:
        if edge["left_id"] in truth:
            if edge["right_id"] not in right_ids:
                raise ValueError("Development reference crosses the frozen partition")
            truth[edge["left_id"]].append(edge["right_id"])
    metadata = read(location / "metadata.json")
    groups = {}
    for rid in truth:
        if metadata[rid]["split"] != "dev" or metadata[rid]["side"] != "left":
            raise ValueError("Unexpected source partition")
        groups[rid] = metadata[rid]["family_id"]
    candidates = read(BASELINE / "candidates.json")
    if {row["record_id"] for row in candidates} != set(truth):
        raise ValueError("Candidates have a different query denominator")
    return left, right, truth, groups, candidates


def restricted(rows, candidates):
    pools = {row["record_id"]: set(row["candidate_ids"]) for row in candidates}
    return [{"record_id": row["record_id"], "target_ids": [rid for rid in row["target_ids"] if rid in pools[row["record_id"]]]}
            for row in rows]


def score(truth, decisions, cost=None):
    rows = evaluation.outcomes(truth, decisions)
    result = evaluation.summarize(rows)
    if cost is not None:
        cost = dict(cost)
        cost["reserved_usd_per_attempted_query"] = str(Decimal(cost["reserved_usd"]) / len(rows))
        cost["known_nominal_usd_per_attempted_query"] = str(Decimal(cost["known_nominal_token_price_usd"]) / len(rows))
        correct = result["correct_complete_decisions"]
        cost["nominal_usd_per_correct_complete_decision"] = (str(Decimal(cost["known_nominal_token_price_usd"])/correct)
            if correct and cost["nominal_token_price_complete"] else None)
        result["cost"] = cost
    return {"summary": result, "outcomes": rows, "decisions": decisions}


def local_cost(arm):
    completion = read(BASELINE / "completion.json")
    return {"provider_calls": 0, "known_nominal_token_price_usd": "0", "reserved_usd": "0", "nominal_token_price_complete": True,
            "local_wall_seconds": completion["lexical" if arm == "lexical" else "splink"]["wall_seconds"],
            "scope": "Offline compute time; provider cost zero; hardware and labour unpriced."}


def score_conventional(truth, groups, candidates):
    lex = read(BASELINE / "lexical-decisions.json")
    fs = read(BASELINE / "splink-decisions.json")
    audit = read(BASELINE / "splink-fit-audit.json")
    if audit["status"] != "predictions_frozen_without_truth_scoring":
        raise ValueError("Conventional reference fit is invalid")
    definitions = {
        "lexical_all_pairs": (lex["thresholds"]["0.65"], "lexical"),
        "splink_all_pairs": (fs["decisions"][PRIMARY_FS], "splink"),
        "lexical_same_candidates": (restricted(lex["thresholds"]["0.65"], candidates), "lexical"),
        "splink_same_candidates": (restricted(fs["decisions"][PRIMARY_FS], candidates), "splink"),
    }
    primary = {name: score(truth, adapters.product_decisions(rows), local_cost(arm))
               for name, (rows, arm) in definitions.items()}
    sensitivities = {}
    for arm, grid in [("lexical", lex["thresholds"]), ("splink", fs["decisions"])]:
        for setting, rows in grid.items():
            sensitivities[f"{arm}/{setting}"] = {
                "all_pairs": evaluation.summarize(evaluation.outcomes(truth, adapters.product_decisions(rows))),
                "same_candidates": evaluation.summarize(evaluation.outcomes(truth, adapters.product_decisions(restricted(rows, candidates))))}
    reference = sorted(ELIGIBLE, key=lambda name: (-primary[name]["summary"]["correct_complete_decisions"], name))[0]
    candidate_map = {r["record_id"]: set(r["candidate_ids"]) for r in candidates}
    candidate_audit = {
        "queries": len(truth), "candidate_pairs": sum(len(v) for v in candidate_map.values()),
        "reference_links": sum(len(v) for v in truth.values()),
        "reference_links_retrieved": sum(len(set(v) & candidate_map[rid]) for rid,v in truth.items()),
        "queries_with_all_reference_targets_retrieved": sum(set(v) <= candidate_map[rid] for rid,v in truth.items()),
        "non_nil_queries": sum(bool(v) for v in truth.values()),
        "non_nil_queries_with_all_reference_targets_retrieved": sum(bool(v) and set(v) <= candidate_map[rid] for rid,v in truth.items()),
        "boundary": "Candidate misses remain errors in complete-decision metrics; no target is inserted from truth."}
    save("conventional-primary.json", primary)
    save("conventional-query-outcomes.json", [{"query_id":rid,"source_family":groups[rid],"truth_target_ids":sorted(truth[rid]),
        "methods":{name:{"decision":result["decisions"][rid],"outcome":next(row for row in result["outcomes"] if row["query_id"]==rid)} for name,result in primary.items()}}
        for rid in sorted(truth)])
    save("conventional-sensitivity.json", {"results":sensitivities,"boundary":"Descriptive frozen grids only; no best threshold or prior selected."})
    save("candidate-audit.json", candidate_audit)
    save("development-reference.json", {"reference":reference,"eligible_methods":ELIGIBLE,
        "summary":primary[reference]["summary"],"minimum_correct":primary[reference]["summary"]["correct_complete_decisions"]+8,
        "maximum_false_links":primary[reference]["summary"]["false_links"],
        "maximum_false_assignment_queries":primary[reference]["summary"]["queries_with_false_assignment"],
        "minimum_automatic_decisions":primary[reference]["summary"]["automatic_decisions"]-7,
        "family_groups":len(set(groups.values()))})
    save("conventional-paired.json", evaluation.paired_comparison(primary["lexical_all_pairs"]["outcomes"], primary["splink_all_pairs"]["outcomes"], groups))
    return primary, reference


def ledger_snapshot(path):
    """Take a consistent read-only snapshot; never append, admit or execute."""
    with Path(path).open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        return [json.loads(line) for line in handle if line.strip()]


def terminal_results(manifest, events):
    tags = {j["tag"] for j in manifest["jobs"]}
    rows = [row for row in events if row.get("event") == "result" and row.get("tag") in tags]
    if any(row.get("batch_id") != manifest["batch_id"] for row in rows):
        raise ValueError("Ledger batch mismatch")
    if len({row["tag"] for row in rows}) != len(rows):
        raise ValueError("Duplicate terminal results")
    if any(row.get("status") not in {"valid", "failed"} for row in rows):
        raise ValueError("Unexpected nonterminal result status")
    if len(rows) != len(tags):
        return None, {"status":"waiting_for_all_640_terminal_results","completed":len(rows),"expected":len(tags)}
    return rows, None


def paid_cost(manifest, rows, arm):
    marker = f"/{arm}/"
    chosen = [row for row in rows if marker in row["tag"]]
    bounds = [bound for job,bound in zip(manifest["jobs"],manifest["job_bounds"]) if marker in job["tag"]]
    unpriced = sum("estimated_token_price_usd" not in row for row in chosen)
    cost = {"provider_calls":len(chosen), "provider_valid_calls":sum(row["status"]=="valid" for row in chosen),
        "provider_failed_calls":sum(row["status"]=="failed" for row in chosen),
        "reserved_usd":str(sum((Decimal(b["reserved_usd"]) for b in bounds),Decimal(0))),
        "known_nominal_token_price_usd":str(sum((Decimal(row["estimated_token_price_usd"]) for row in chosen if "estimated_token_price_usd" in row),Decimal(0))),
        "unpriced_calls":unpriced, "nominal_token_price_complete":unpriced==0,
        "summed_call_elapsed_seconds":sum(row.get("elapsed_seconds",0) for row in chosen),
        "scope":"Token-price estimate from reported usage, not invoice. Reservation includes every attempt; unknown usage remains unpriced. Shared catalog extraction charged fully to this development lane."}
    return cost


def extraction_fit(left, right, fields, candidates, prior_bundle):
    location = OUTPUT / "extraction-fit"
    audit_path = location / "splink-fit-audit.json"
    if audit_path.exists():
        saved = read(location / "recordwise-input-manifest.json")
        expected = sha256(json.dumps(fields,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if saved["extractions_sha256"] != expected or saved["conventional_prior_bundle"] != prior_bundle:
            raise ValueError("Existing extracted fit differs from the frozen inputs")
    else:
        if location.exists() and any(location.iterdir()):
            raise FileExistsError("Partial extracted fit exists; preserve it rather than silently rerun")
        baseline.run_recordwise(left,right,fields,candidates,location,prior_bundle=prior_bundle)
    audit = read(audit_path)
    if audit["priors"] != prior_bundle["priors"] or audit["main_prior"] != prior_bundle["main_prior"]:
        raise ValueError("Extracted fit changed the numeric conventional priors")
    return location, audit


def score_semantic(manifest, events, terminal, left, right, truth, groups, candidates, primary, reference):
    save("terminal-collection.json", {"batch_id":manifest["batch_id"],"expected_calls":640,"terminal_calls":len(terminal),
        "results":[{"tag":row["tag"],"status":row["status"],"result_sha256":runtime.digest(row)} for row in sorted(terminal,key=lambda r:r["tag"])],
        "boundary":"All requests have a terminal result before semantic fitting. Source ledger read under a shared lock without modification."})
    fields, extraction_audit = adapters.product_extractions(manifest, events)
    expected_ids = {row["record_id"] for row in left+right}
    if set(fields) != expected_ids:
        raise ValueError("Recordwise extraction denominator is incomplete")
    failed_ids = sorted(rid for rid,parsed in fields.items() if parsed is None)
    if (OUTPUT / "extractions.json").exists():
        if read(OUTPUT / "extractions.json") != fields:
            raise ValueError("Existing collected extractions differ from the verified ledger")
    else:
        save("extractions.json", fields)
    save("extraction-audit.json", {"records":extraction_audit,"failed_extraction_ids":failed_ids})
    location, fit = extraction_fit(left,right,fields,candidates,manifest["contract"]["prior_bundle"])
    fit_valid = fit["status"] == "predictions_frozen_without_truth_scoring"
    costs = {arm:paid_cost(manifest,terminal,arm) for arm in ["extraction","selection"]}
    if fit_valid:
        grid = read(location / "splink-decisions.json")["decisions"]
        rows = grid[PRIMARY_FS]
    else:
        rows = [{"record_id":rid,"target_ids":[]} for rid in truth]
        grid = {}
    semantic = {
        "extraction_splink_all_pairs":score(truth,adapters.product_decisions(rows,failed_extraction_ids=failed_ids,fit_valid=fit_valid),costs["extraction"]),
        "extraction_splink_same_candidates":score(truth,adapters.product_decisions(restricted(rows,candidates),failed_extraction_ids=failed_ids,fit_valid=fit_valid),costs["extraction"]),
        "semantic_selection":score(truth,adapters.product_selections(manifest,events,candidates),costs["selection"]),
    }
    save("semantic-primary.json", semantic)
    sensitivity = {}
    for setting, records in grid.items():
        sensitivity[setting] = {"all_pairs":evaluation.summarize(evaluation.outcomes(truth,adapters.product_decisions(records,failed_extraction_ids=failed_ids))),
                               "same_candidates":evaluation.summarize(evaluation.outcomes(truth,adapters.product_decisions(restricted(records,candidates),failed_extraction_ids=failed_ids)))}
    save("semantic-extraction-sensitivity.json", {"fit_status":fit["status"],"results":sensitivity,"boundary":"Fixed original and broad prior-only grids; no best result selected."})
    gates, paired = {}, {}
    for name in ["extraction_splink_all_pairs", "semantic_selection"]:
        gates[name] = evaluation.development_gate({k:v["summary"] for k,v in primary.items()},semantic[name]["summary"],eligible_methods=ELIGIBLE)
    for name,result in semantic.items():
        paired[name] = evaluation.paired_comparison(primary[reference]["outcomes"],result["outcomes"],groups)
    paired["extraction_vs_conventional_FS_same_candidates"] = evaluation.paired_comparison(primary["splink_same_candidates"]["outcomes"],semantic["extraction_splink_same_candidates"]["outcomes"],groups)
    paired["selection_vs_conventional_FS_same_candidates"] = evaluation.paired_comparison(primary["splink_same_candidates"]["outcomes"],semantic["semantic_selection"]["outcomes"],groups)
    save("development-gates.json", gates)
    save("semantic-paired.json", paired)
    methods = primary | semantic
    common = [{"query_id":rid,"source_family":groups[rid],"truth_target_ids":sorted(truth[rid]),
               "methods":{name:{"decision":result["decisions"][rid], "outcome":next(row for row in result["outcomes"] if row["query_id"]==rid)} for name,result in methods.items()}}
              for rid in sorted(truth)]
    save("common-query-outcomes.json", common)
    save("completion.json", {"status":"development_scored_only", "queries":len(truth),"jobs":len(terminal),
        "source_families":len(set(groups.values())),"manifest_sha256":MANIFEST_SHA256,"manifest_file_sha256":MANIFEST_FILE_SHA256,
        "integration_sha256":baseline.file_hash(__file__),"extracted_fit_status":fit["status"],"reference":reference,
        "gates":gates,"test_identity_outcomes_scored":False,
        "files":{str(p.relative_to(OUTPUT)):baseline.file_hash(p) for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name!="completion.json"}})
    return {"status":"development_scored_only","reference":reference,"summaries":{k:v["summary"] for k,v in semantic.items()},"gates":gates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-semantic",action="store_true")
    parser.add_argument("--ledger",type=Path,default=runtime.LEDGER)
    args = parser.parse_args()
    manifest = verify_contract()
    left,right,truth,groups,candidates = development_data()
    primary,reference = score_conventional(truth,groups,candidates)
    if not args.include_semantic:
        print(json.dumps({"status":"conventional_development_scored","reference":reference,"summaries":{k:v["summary"] for k,v in primary.items()}},indent=2))
        return
    events = ledger_snapshot(args.ledger)
    terminal, pending = terminal_results(manifest,events)
    if pending:
        print(json.dumps(pending,indent=2)); return
    result = score_semantic(manifest,events,terminal,left,right,truth,groups,candidates,primary,reference)
    model_jobs.verify_bindings(manifest)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
