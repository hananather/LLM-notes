"""Seal-gated held-out tradeoff scoring. No API calls, fitting, or rule selection."""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import decision_adapters as adapters
import evaluation
import model_jobs
import runtime

OUTPUT = HERE / "results/model-evaluation-v1"
DATA = HERE / "data/amazon-google/v2"
BASELINE = HERE / "results/baseline-v6/amazon-google/test"
AMENDMENT = "contracts/product-tradeoff-amendment-v1.json"
QUERY_COUNT = 1219
CATALOG_COUNT = 2874
LEXICAL_THRESHOLDS = ["0.5", "0.65", "0.8", "0.9"]
FS_THRESHOLDS = ["0.5", "0.9", "0.99", "0.999"]
PRIORS = ["assumed_recall=0.5", "assumed_recall=0.8", "assumed_recall=1.0",
          "links_per_left=0.1", "links_per_left=0.5", "links_per_left=1.0", "links_per_left=2.0"]
LOSS_RATIOS = [0, .1, .25, .5, 1, 2, 5, 10, 20, 50, 100]
METRICS = {
    "exact_complete_decision_rate": ("correct", "fraction of queries"),
    "false_edges_per_query": ("false_links", "edges per query"),
    "missed_edges_per_query": ("missed_links", "edges per query"),
    "false_assignment_query_rate": ("false_assignment", "fraction of queries"),
    "automatic_coverage": ("automatic", "fraction of queries"),
    "false_nil_query_rate": ("false_nil", "fraction of queries"),
    "linked_on_nil_query_rate": ("linked_on_nil", "fraction of queries"),
}
PRIMARY = {
    "lexical_all_pairs": "lexical/threshold=0.65/all_pairs",
    "splink_all_pairs": "splink/assumed_recall=0.8;threshold=0.9/all_pairs",
    "lexical_same_candidates": "lexical/threshold=0.65/same_candidates",
    "splink_same_candidates": "splink/assumed_recall=0.8;threshold=0.9/same_candidates",
    "semantic_selection": "semantic_selection",
}


def read(path):
    value = json.loads(Path(path).read_text())
    json.dumps(value, allow_nan=False)
    return value


def save(name, value):
    path = OUTPUT / name
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"Preserve existing evaluation evidence: {path}")
        return
    with path.open("xb") as handle:
        handle.write(raw)


def rooted(relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Evidence paths must be relative to the experiment root")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Evidence path leaves the experiment root")
    return path


def check_count_map(value, total=None):
    if not isinstance(value, dict) or any(not isinstance(k,str) or type(v) is not int or v<0 for k,v in value.items()):
        raise ValueError("Expected named nonnegative integer counts")
    if total is not None and sum(value.values()) != total:
        raise ValueError("Execution counts do not preserve every attempted query")


def verify_seal(manifest_path, seal_path):
    """Read only contracts, inference IDs, candidates and sealed decisions."""
    manifest, seal = read(manifest_path), read(seal_path)
    runtime.verify_manifest(manifest)
    model_jobs.verify_bindings(manifest)
    bindings = manifest["contract"]["files"]
    required = [AMENDMENT, "products/score_evaluation.py", "evaluation.py", "decision_adapters.py",
                "products/data/amazon-google/v2/partitions.json", "products/data/amazon-google/v2/truth.json",
                "products/data/amazon-google/v2/metadata.json",
                "products/results/baseline-v6/amazon-google/test/lexical-decisions.json",
                "products/results/baseline-v6/amazon-google/test/splink-decisions.json",
                "products/results/baseline-v6/amazon-google/test/splink-fit-audit.json",
                "products/results/baseline-v6/amazon-google/test/candidates.json"]
    if not set(required) <= set(bindings):
        raise ValueError("The manifest must bind the evaluator, amendment, labels and every fixed comparator")
    if manifest["transport"] != "batch" or len(manifest["jobs"]) != QUERY_COUNT:
        raise ValueError("Only the amended complete selector Batch is permitted")
    if any(not j["tag"].startswith("products/test/selection/") or j["model"] != "gpt-6-luna" or j["max_output_tokens"] != 256 for j in manifest["jobs"]):
        raise ValueError("Unexpected intervention in held-out manifest")
    if seal.get("status") != "predictions_sealed" or seal.get("query_count") != QUERY_COUNT or seal.get("terminal_result_count") != QUERY_COUNT:
        raise ValueError("Every allowed query needs a sealed terminal decision")
    if seal.get("manifest_sha256") != runtime.digest(manifest) or seal.get("evaluator_sha256") != runtime.file_hash(__file__):
        raise ValueError("Seal does not bind this request manifest and evaluator")
    stamp = datetime.fromisoformat(seal["sealed_at_utc"].replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Seal timestamp needs a timezone")
    execution = seal["execution_audit"]
    if execution.get("terminal_status") not in {"completed", "expired", "failed", "cancelled"}:
        raise ValueError("An unresolved provider batch cannot be scored")
    check_count_map(execution["parsed_result_status_counts"], QUERY_COUNT)
    check_count_map(execution["provider_model_id_counts"])
    if sum(execution["provider_model_id_counts"].values()) > QUERY_COUNT:
        raise ValueError("Provider model counts exceed attempts")
    prediction_path = rooted(seal["predictions_path"])
    if runtime.file_hash(prediction_path) != seal["predictions_sha256"]:
        raise ValueError("Sealed decisions changed")
    predictions = read(prediction_path)
    partition = read(DATA / "partitions.json")["test"]
    if len(partition["left"]) != QUERY_COUNT or len(partition["right"]) != CATALOG_COUNT or set(predictions) != set(partition["left"]):
        raise ValueError("Incomplete evaluation denominator")
    jobs = [j["tag"].rsplit("/",1)[1] for j in manifest["jobs"]]
    if len(set(jobs)) != QUERY_COUNT or set(jobs) != set(partition["left"]):
        raise ValueError("Job roster differs from evaluation roster")
    candidates = read(BASELINE / "candidates.json")
    pool = {row["record_id"]:set(row["candidate_ids"]) for row in candidates}
    if set(pool) != set(predictions):
        raise ValueError("Candidate denominator differs from sealed decisions")
    for rid, decision in predictions.items():
        if not isinstance(decision, dict):
            raise ValueError("Malformed sealed decision; parser failure must already be explicit")
        status, targets = decision.get("status"), decision.get("target_ids")
        if status not in {"linked","nil","review","failed"} or not isinstance(targets,list) or not all(isinstance(t,str) for t in targets):
            raise ValueError("Malformed sealed decision")
        if len(targets)!=len(set(targets)) or ((status=="linked") != bool(targets)) or not set(targets)<=pool[rid]:
            raise ValueError("Sealed decision violates the unchanged candidate/parser policy")
    amendment = read(rooted(AMENDMENT))
    if amendment["allowed_run"]["queries"] != QUERY_COUNT or amendment["descriptive_loss"]["lambda_grid"] != LOSS_RATIOS:
        raise ValueError("Amendment and evaluator specification differ")
    return {"manifest":manifest,"seal":seal,"predictions":predictions,"partition":partition,
            "candidates":candidates,"manifest_path":str(Path(manifest_path).resolve()),
            "seal_path":str(Path(seal_path).resolve()),"seal_file_sha256":runtime.file_hash(seal_path)}


def evaluation_truth(verified):
    """Called only after seal verification AND the explicit scoring CLI flag."""
    truth = {rid:[] for rid in verified["partition"]["left"]}
    right = set(verified["partition"]["right"])
    for edge in read(DATA / "truth.json")["edges"]:
        if edge["left_id"] in truth:
            if edge["right_id"] not in right:
                raise ValueError("Reference crosses the frozen source partition")
            truth[edge["left_id"]].append(edge["right_id"])
    metadata = read(DATA / "metadata.json")
    groups = {}
    for rid in truth:
        if metadata[rid]["split"] != "test" or metadata[rid]["side"] != "left":
            raise ValueError("Evaluation metadata disagrees with the query roster")
        groups[rid] = metadata[rid]["family_id"]
    return truth, groups


def restrict(rows, pool):
    return [{"record_id":r["record_id"],"target_ids":[t for t in r["target_ids"] if t in pool[r["record_id"]]]} for r in rows]


def fixed_methods(verified):
    lexical = read(BASELINE / "lexical-decisions.json")["thresholds"]
    fs = read(BASELINE / "splink-decisions.json")["decisions"]
    expected = {f"{prior};threshold={threshold}" for prior in PRIORS for threshold in FS_THRESHOLDS}
    if set(lexical)!=set(LEXICAL_THRESHOLDS) or set(fs)!=expected:
        raise ValueError("Frozen operating-rule grid is incomplete")
    fit = read(BASELINE / "splink-fit-audit.json")
    if fit["status"] != "predictions_frozen_without_truth_scoring" or fit["prediction_consistency"]["status"] != "passed":
        raise ValueError("Conventional full-universe fit lacks valid predictions")
    pool = {r["record_id"]:set(r["candidate_ids"]) for r in verified["candidates"]}
    methods = {}
    for arm, settings in [("lexical",[(f"threshold={t}",lexical[t]) for t in LEXICAL_THRESHOLDS]),
                          ("splink",[(f"{p};threshold={t}",fs[f"{p};threshold={t}"]) for p in PRIORS for t in FS_THRESHOLDS])]:
        for label,rows in settings:
            if {r["record_id"] for r in rows} != set(pool) or len(rows)!=len(pool):
                raise ValueError("Conventional decisions have a different denominator")
            methods[f"{arm}/{label}/all_pairs"] = adapters.product_decisions(rows)
            methods[f"{arm}/{label}/same_candidates"] = adapters.product_decisions(restrict(rows,pool))
    methods["semantic_selection"] = verified["predictions"]
    return methods


def bootstrap_plan(query_ids, groups):
    ordered = list(dict.fromkeys(groups[rid] for rid in query_ids))
    group_index = {group:i for i,group in enumerate(ordered)}
    query_groups = np.array([group_index[groups[rid]] for rid in query_ids],dtype=int)
    group_sizes = np.bincount(query_groups,minlength=len(ordered))
    rng = np.random.default_rng(evaluation.BOOTSTRAP_SEED)
    weights = np.array([np.bincount(rng.integers(0,len(ordered),len(ordered)),minlength=len(ordered))
                        for _ in range(evaluation.BOOTSTRAP_REPLICATES)],dtype=float)
    return {"query_groups":query_groups,"group_count":len(ordered),"weights":weights,
            "denominators":weights @ group_sizes,"query_ids":query_ids}


def paired_metrics(reference_rows, candidate_rows, plan):
    left = {r["query_id"]:r for r in reference_rows}
    right = {r["query_id"]:r for r in candidate_rows}
    if set(left)!=set(right) or set(left)!=set(plan["query_ids"]):
        raise ValueError("Paired methods require exactly the same complete denominator")
    keys = [name for name in METRICS]
    values = np.array([[float(right[rid][METRICS[name][0]])-float(left[rid][METRICS[name][0]]) for name in keys]
                       for rid in plan["query_ids"]])
    grouped = np.zeros((plan["group_count"],len(keys)))
    np.add.at(grouped,plan["query_groups"],values)
    samples = (plan["weights"] @ grouped) / plan["denominators"][:,None]
    result = {"queries":len(left),"source_families":plan["group_count"],"direction":"semantic selector minus named conventional rule",
              "bootstrap_replicates":evaluation.BOOTSTRAP_REPLICATES,"bootstrap_seed":evaluation.BOOTSTRAP_SEED,
              "interval_scope":"Pointwise descriptive paired family-bootstrap intervals for this historical source sample; no multiplicity correction or NSO-population claim.",
              "corrections":sum(not left[k]["correct"] and right[k]["correct"] for k in left),
              "regressions":sum(left[k]["correct"] and not right[k]["correct"] for k in left),"metrics":{}}
    for i,name in enumerate(keys):
        result["metrics"][name] = {"difference":float(values[:,i].mean()),"total_difference":int(values[:,i].sum()),
            "unit":METRICS[name][1],"cluster_bootstrap_95_percent_interval":np.quantile(samples[:,i],[.025,.975]).tolist()}
    return result, samples


def subgroup_summaries(rows, truth):
    def summarize(chosen):
        return evaluation.summarize(chosen) if chosen else {"queries":0,"status":"empty subgroup"}
    return {
        "reference_cardinality":{
            "NIL":summarize([r for r in rows if len(truth[r["query_id"]])==0]),
            "single_target":summarize([r for r in rows if len(truth[r["query_id"]])==1]),
            "multiple_targets":summarize([r for r in rows if len(truth[r["query_id"]])>1])},
        "decision_status":{status:summarize([r for r in rows if r["status"]==status]) for status in ["linked","nil","review","failed"]},
        "boundary":"Reference-cardinality groups are shared across methods. Status groups are method-dependent descriptive denominators, not selected paired populations."}


def run_scoring(verified):
    if (OUTPUT / "completion.json").exists():
        complete = read(OUTPUT / "completion.json")
        if complete["seal_file_sha256"] != verified["seal_file_sha256"] or complete["evaluator_sha256"] != runtime.file_hash(__file__):
            raise ValueError("Existing evaluation belongs to a different seal or evaluator")
        for name,digest in complete["files"].items():
            if runtime.file_hash(OUTPUT/name)!=digest:
                raise ValueError("Completed evaluation evidence changed")
        return complete["primary_summary"]
    # Record verified seal before the first outcome-label read. Existing differing
    # records stop execution; one result is never silently replaced by another.
    save("seal-verification.json", {"status":"verified_before_truth_access","seal_file_sha256":verified["seal_file_sha256"],
        "seal":verified["seal"],"manifest_sha256":runtime.digest(verified["manifest"]),"evaluator_sha256":runtime.file_hash(__file__)})
    truth,groups = evaluation_truth(verified)
    methods = fixed_methods(verified)
    rows = {name:evaluation.outcomes(truth,predictions) for name,predictions in methods.items()}
    summaries = {name:evaluation.summarize(outcomes) for name,outcomes in rows.items()}
    primary = {name:summaries[key] for name,key in PRIMARY.items()}
    query_ids = sorted(truth)
    plan = bootstrap_plan(query_ids,groups)
    paired,loss_differences = {},{}
    for name,outcomes in rows.items():
        if name=="semantic_selection":
            continue
        comparison,samples = paired_metrics(outcomes,rows["semantic_selection"],plan)
        paired[name] = comparison
        fp = list(METRICS).index("false_edges_per_query")
        missed = list(METRICS).index("missed_edges_per_query")
        loss_differences[name] = [{"lambda":ratio,
            "difference_per_query":ratio*comparison["metrics"]["false_edges_per_query"]["difference"]+comparison["metrics"]["missed_edges_per_query"]["difference"],
            "pointwise_family_bootstrap_95_percent_interval":np.quantile(ratio*samples[:,fp]+samples[:,missed],[.025,.975]).tolist()}
            for ratio in LOSS_RATIOS]
    subgroups = {name:subgroup_summaries(outcomes,truth) for name,outcomes in rows.items()}
    loss = {}
    for name,summary in summaries.items():
        unresolved = [r for r in rows[name] if r["status"] in {"review","failed"}]
        loss[name] = {"false_edges":summary["false_links"],"missed_edges":summary["missed_links"],
            "review_queries":summary["review_queries"],"failed_queries":summary["failed_queries"],
            "unresolved_positive_queries":sum(bool(truth[r["query_id"]]) for r in unresolved),
            "unresolved_nil_queries":sum(not truth[r["query_id"]] for r in unresolved),
            "grid":[{"lambda":ratio,"edge_loss":ratio*summary["false_links"]+summary["missed_links"],
                     "edge_loss_per_query":(ratio*summary["false_links"]+summary["missed_links"])/QUERY_COUNT} for ratio in LOSS_RATIOS]}
    pools = {r["record_id"]:set(r["candidate_ids"]) for r in verified["candidates"]}
    candidate_audit = {"queries":QUERY_COUNT,"candidate_pairs":sum(map(len,pools.values())),
        "reference_edges":sum(map(len,truth.values())),"reference_edges_retrieved":sum(len(set(v)&pools[rid]) for rid,v in truth.items()),
        "queries_with_all_reference_targets_retrieved":sum(set(v)<=pools[rid] for rid,v in truth.items()),
        "matched_queries":sum(bool(v) for v in truth.values()),
        "matched_queries_with_all_reference_targets_retrieved":sum(bool(v) and set(v)<=pools[rid] for rid,v in truth.items()),
        "boundary":"No true target is inserted; every candidate miss remains in end-to-end denominators."}
    all_rows = {name:{r["query_id"]:r for r in outcomes} for name,outcomes in rows.items()}
    common = [{"query_id":rid,"source_family":groups[rid],"reference_target_ids":sorted(truth[rid]),
        "methods":{name:{"decision":methods[name][rid],"outcome":all_rows[name][rid]} for name in methods}} for rid in query_ids]
    save("primary-summary.json", {"study":"Post-development, prospectively specified held-out tradeoff characterization",
        "original_development_gate":"failed; unchanged by this amendment or held-out result", "development_selected_reference":"lexical_all_pairs",
        "queries":QUERY_COUNT,"catalog_records":CATALOG_COUNT,"source_families":len(set(groups.values())),"results":primary,
        "execution_audit":verified["seal"]["execution_audit"],"extraction_held_out":"not run; remains development-only",
        "boundary":"No test winner, threshold or prior selected; one amended selector run; no production, ROI or NSO-population claim."})
    save("all-fixed-rule-summaries.json", summaries)
    save("paired-family-uncertainty.json", paired)
    save("subgroup-summaries.json", subgroups)
    save("candidate-audit.json", candidate_audit)
    save("common-query-outcomes.json", common)
    save("descriptive-loss-grid.json", {"formula":"lambda * false_edges + missed_edges","methods":loss,
        "paired_selector_minus_baseline":loss_differences,
        "unresolved_accounting":"Review and failure supply no links; positive targets remain missed. Unresolved NIL is never a correct complete decision. Human review is neither assumed successful nor priced as free.",
        "boundary":"Hypothetical edge-cost ratios; not query costs, measured NSO costs or ROI. No operational winner or precise crossover is selected."})
    model_jobs.verify_bindings(verified["manifest"])
    if runtime.file_hash(rooted(verified["seal"]["predictions_path"]))!=verified["seal"]["predictions_sha256"]:
        raise ValueError("Predictions changed during scoring")
    save("completion.json", {"status":"sealed_held_out_tradeoff_scored_once","original_development_gate":"failed; unchanged",
        "manifest_sha256":runtime.digest(verified["manifest"]),"seal_file_sha256":verified["seal_file_sha256"],
        "evaluator_sha256":runtime.file_hash(__file__),"primary_summary":primary,
        "files":{str(p.relative_to(OUTPUT)):runtime.file_hash(p) for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name!="completion.json"}})
    return primary


def self_check():
    truth = {"a":["x","y"],"b":[],"c":["z"],"d":[]}
    groups = {"a":"family1","b":"family1","c":"family2","d":"family3"}
    reference = {"a":{"status":"linked","target_ids":["x"]},"b":{"status":"nil","target_ids":[]},
                 "c":{"status":"nil","target_ids":[]},"d":{"status":"nil","target_ids":[]}}
    candidate = {"a":{"status":"linked","target_ids":["x","y"]},"b":{"status":"linked","target_ids":["bad"]},
                 "c":{"status":"review","target_ids":[]},"d":{"status":"failed","target_ids":[]}}
    left,right = evaluation.outcomes(truth,reference),evaluation.outcomes(truth,candidate)
    paired,_ = paired_metrics(left,right,bootstrap_plan(sorted(truth),groups))
    check = evaluation.paired_comparison(left,right,groups)
    assert paired["metrics"]["exact_complete_decision_rate"]["difference"] == check["accuracy_difference"]
    assert np.allclose(paired["metrics"]["exact_complete_decision_rate"]["cluster_bootstrap_95_percent_interval"],check["cluster_bootstrap_95_percent_interval"])
    assert paired["metrics"]["false_edges_per_query"]["total_difference"] == 1
    assert paired["metrics"]["missed_edges_per_query"]["total_difference"] == -1
    assert paired["metrics"]["automatic_coverage"]["total_difference"] == -2
    assert not next(r for r in right if r["query_id"]=="d")["correct"]
    assert next(r for r in right if r["query_id"]=="c")["missed_links"] == 1
    assert subgroup_summaries(right,truth)["decision_status"]["failed"]["queries"]==1
    return {"status":"synthetic_checks_passed","source_truth_opened":False,
            "checks":["same frozen bootstrap draws and exact-decision CI","edge and coverage differences","unresolved positive missed edges","unresolved NIL incorrect","status subgroup denominator"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--seal",type=Path)
    parser.add_argument("--score-sealed",action="store_true",help="Explicit authorization to open held-out labels after verifying the seal")
    parser.add_argument("--self-check",action="store_true",help="Synthetic fixtures only; no source truth or predictions opened")
    args = parser.parse_args()
    if args.self_check:
        if args.score_sealed or args.manifest or args.seal:
            raise ValueError("Synthetic checking cannot be combined with outcome scoring")
        print(json.dumps(self_check(),indent=2));return
    if args.manifest is None or args.seal is None:
        parser.error("--manifest and --seal are required; --score-sealed separately authorizes scoring")
    verified = verify_seal(args.manifest,args.seal)
    if not args.score_sealed:
        print(json.dumps({"status":"seal_verified_only","queries":QUERY_COUNT,"truth_opened":False},indent=2));return
    print(json.dumps(run_scoring(verified),indent=2))


if __name__=="__main__":
    main()
