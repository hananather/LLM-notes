"""Freeze, then score the existing free Abt–Buy conventional reference only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT))
import decision_adapters as adapters
import evaluation
import runtime
from score_evaluation import bootstrap_plan, paired_metrics, METRICS, PRIORS, FS_THRESHOLDS, LEXICAL_THRESHOLDS

DATA = HERE / "data/abt-buy/v2"
BASELINE = HERE / "results/baseline-v6/abt-buy/test"
OUTPUT = HERE / "results/free-abt-reference-v1"
PRIMARY = {
    "lexical_all_pairs":"lexical/threshold=0.65/all_pairs",
    "splink_all_pairs":"splink/assumed_recall=0.8;threshold=0.9/all_pairs",
    "lexical_same_candidates":"lexical/threshold=0.65/same_candidates",
    "splink_same_candidates":"splink/assumed_recall=0.8;threshold=0.9/same_candidates",
}


def read(path):
    result = json.loads(Path(path).read_text())
    json.dumps(result,allow_nan=False)
    return result


def save(name,value):
    path = OUTPUT/name
    raw = (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:
            raise FileExistsError(f"Preserve existing reference evidence: {path}")
        return
    with path.open("xb") as stream:
        stream.write(raw)


def bindings():
    paths = list(DATA.glob("*.json"))+[p for p in BASELINE.iterdir() if p.is_file()]
    paths += [Path(__file__), HERE/"score_evaluation.py", ROOT/"evaluation.py", ROOT/"decision_adapters.py",
              ROOT/"model_jobs.py", ROOT/"runtime.py"]
    return {str(path.relative_to(ROOT)):runtime.file_hash(path) for path in sorted(paths)}


def verify_protocol():
    protocol = read(OUTPUT/"protocol.json")
    if protocol["files"]!=bindings():
        raise ValueError("Frozen Abt reference code, inputs or predictions changed")
    return protocol


def freeze_protocol():
    if (OUTPUT/"protocol.json").exists():
        return verify_protocol()
    # Fit and completion diagnostics contain no reference identities or outcome
    # scores. Preserve them and all prediction hashes before opening truth.
    audit = read(BASELINE/"splink-fit-audit.json")
    if audit["status"]!="predictions_frozen_without_truth_scoring" or audit["prediction_consistency"]["status"]!="passed":
        raise ValueError("A valid frozen conventional fit is required")
    completion = read(BASELINE/"completion.json")
    for name,digest in completion["files"].items():
        if runtime.file_hash(BASELINE/name)!=digest:
            raise ValueError("Prediction completion hash mismatch")
    protocol = {
        "study":"Free conventional Abt–Buy source reference",
        "question":"How do the already fixed conventional operating rules recover complete publisher target sets on the intact Abt–Buy graph?",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "freeze_boundary":"This file is saved before this evaluator parses Abt reference identities. Existing v6 predictions are neither refitted nor retuned.",
        "source":"abt-buy","curation":"v2","baseline":"v6","left_records":1081,"right_records":1092,
        "scope":"Free conventional reference on the whole source graph; no paid semantic method, no semantic replication and no NSO-population inference.",
        "primary_methods":PRIMARY,"lexical_thresholds":LEXICAL_THRESHOLDS,"splink_prior_labels":PRIORS,
        "splink_thresholds":FS_THRESHOLDS,"numeric_priors":audit["priors"],"main_prior":audit["main_prior"],
        "uncertainty":{"unit":"actual source families among all left records","replicates":evaluation.BOOTSTRAP_REPLICATES,
                       "seed":evaluation.BOOTSTRAP_SEED,"metrics":list(METRICS),"intervals":"pointwise descriptive; no multiplicity adjustment"},
        "reporting":"All fixed grids and candidate scopes remain visible. No outcome-selected winner, threshold or prior. Compare each rule with lexical threshold 0.65 all-pairs only as a fixed descriptive reference.",
        "failure_policy":"Every left record remains in the denominator; review/failure is unresolved and not correct NIL. Preserve the publisher many-to-many relation.",
        "paid_calls":0,"amazon_test_outcomes_accessed":False,"files":bindings()}
    save("protocol.json",protocol)
    return protocol


def load_reference():
    partition = read(DATA/"partitions.json")["test"]
    if len(partition["left"])!=1081 or len(partition["right"])!=1092:
        raise ValueError("Intact source denominator changed")
    truth = {rid:[] for rid in partition["left"]}
    right = set(partition["right"])
    for edge in read(DATA/"truth.json")["edges"]:
        if edge["left_id"] not in truth or edge["right_id"] not in right:
            raise ValueError("Reference graph leaves the frozen complete source")
        truth[edge["left_id"]].append(edge["right_id"])
    metadata = read(DATA/"metadata.json")
    groups = {rid:metadata[rid]["family_id"] for rid in truth}
    candidates = read(BASELINE/"candidates.json")
    pool = {r["record_id"]:set(r["candidate_ids"]) for r in candidates}
    if set(pool)!=set(truth) or len(candidates)!=len(truth):
        raise ValueError("Candidate denominator differs")
    return truth,groups,pool


def fixed_decisions(pool):
    lex = read(BASELINE/"lexical-decisions.json")["thresholds"]
    fs = read(BASELINE/"splink-decisions.json")["decisions"]
    if set(lex)!=set(LEXICAL_THRESHOLDS) or set(fs)!={f"{p};threshold={t}" for p in PRIORS for t in FS_THRESHOLDS}:
        raise ValueError("Fixed operating grid differs from the prospective specification")
    result = {}
    grids = [("lexical",[(f"threshold={t}",lex[t]) for t in LEXICAL_THRESHOLDS]),
             ("splink",[(f"{p};threshold={t}",fs[f"{p};threshold={t}"]) for p in PRIORS for t in FS_THRESHOLDS])]
    for arm,grid in grids:
        for label,records in grid:
            if {r["record_id"] for r in records}!=set(pool) or len(records)!=len(pool):
                raise ValueError("A baseline decision omitted or duplicated a query")
            restricted = [{"record_id":r["record_id"],"target_ids":[t for t in r["target_ids"] if t in pool[r["record_id"]]]} for r in records]
            result[f"{arm}/{label}/all_pairs"] = adapters.product_decisions(records)
            result[f"{arm}/{label}/same_candidates"] = adapters.product_decisions(restricted)
    return result


def absolute_intervals(rows,plan):
    index = {row["query_id"]:row for row in rows}
    values = np.array([[float(index[rid][field]) for field,_ in METRICS.values()] for rid in plan["query_ids"]])
    grouped = np.zeros((plan["group_count"],len(METRICS)))
    np.add.at(grouped,plan["query_groups"],values)
    samples = (plan["weights"]@grouped)/plan["denominators"][:,None]
    return {name:{"estimate":float(values[:,i].mean()),"unit":METRICS[name][1],
                  "family_bootstrap_95_percent_interval":np.quantile(samples[:,i],[.025,.975]).tolist()} for i,name in enumerate(METRICS)}


def score_frozen():
    protocol = verify_protocol()
    if (OUTPUT/"completion.json").exists():
        completion = read(OUTPUT/"completion.json")
        for name,digest in completion["files"].items():
            if runtime.file_hash(OUTPUT/name)!=digest:
                raise ValueError("Completed reference evidence changed")
        return read(OUTPUT/"primary-summary.json")
    truth,groups,pool = load_reference()
    methods = fixed_decisions(pool)
    rows = {name:evaluation.outcomes(truth,decisions) for name,decisions in methods.items()}
    summaries = {name:evaluation.summarize(outcomes) for name,outcomes in rows.items()}
    for value in summaries.values():
        predicted_edges = value["true_links"]+value["false_links"]
        reference_edges = value["true_links"]+value["missed_links"]
        value["pair_precision"] = value["true_links"]/predicted_edges if predicted_edges else None
        value["pair_recall"] = value["true_links"]/reference_edges if reference_edges else None
    plan = bootstrap_plan(sorted(truth),groups)
    uncertainty = {}
    for name,outcomes in rows.items():
        paired,_ = paired_metrics(rows[PRIMARY["lexical_all_pairs"]],outcomes,plan)
        paired["direction"] = "Named conventional rule minus fixed lexical threshold 0.65 all-pairs"
        uncertainty[name] = {"absolute":absolute_intervals(outcomes,plan),"paired_vs_fixed_lexical":paired}
    primary = {name:summaries[key] for name,key in PRIMARY.items()}
    summary = {"study":protocol["study"],"scope":protocol["scope"],"queries":len(truth),"catalog_records":1092,
        "reference_edges":sum(map(len,truth.values())),"reference_nil_queries":sum(not targets for targets in truth.values()),
        "reference_multiple_target_queries":sum(len(targets)>1 for targets in truth.values()),
        "query_source_families":len(set(groups.values())),"results":primary,"paid_calls":0,
        "boundary":"These are conventional source-reference results only. All operating rules were frozen; no semantic result or outcome-tuned setting is implied."}
    candidate_audit = {"queries":len(truth),"candidate_pairs":sum(map(len,pool.values())),"reference_edges":sum(map(len,truth.values())),
        "reference_edges_retrieved":sum(len(set(v)&pool[rid]) for rid,v in truth.items()),
        "queries_with_complete_reference_set_available":sum(set(v)<=pool[rid] for rid,v in truth.items()),
        "boundary":"Candidate misses count in end-to-end errors; no true target was inserted."}
    indexed = {name:{row["query_id"]:row for row in outcomes} for name,outcomes in rows.items()}
    common = [{"query_id":rid,"source_family":groups[rid],"reference_target_ids":sorted(truth[rid]),
        "methods":{name:{"decision":methods[name][rid],"outcome":indexed[name][rid]} for name in methods}} for rid in sorted(truth)]
    save("primary-summary.json",summary)
    save("all-fixed-rule-summaries.json",summaries)
    save("family-uncertainty.json",uncertainty)
    save("candidate-audit.json",candidate_audit)
    save("common-query-outcomes.json",common)
    lines = ["# Free Abt–Buy conventional reference", "", "The fixed primary lexical and Fellegi–Sunter rules give the following complete target-set results. This is an intact-source conventional reference, not a semantic replication.", "",
             "| Fixed rule | Exact sets / 1,081 | False edges | Missed edges | Automatic decisions |", "|---|---:|---:|---:|---:|"]
    for name,value in primary.items():
        lines.append(f"| {name} | {value['correct_complete_decisions']} | {value['false_links']} | {value['missed_links']} | {value['automatic_decisions']} |")
    lines += ["", "The full prior grid below uses the fixed FS threshold 0.9 and all-pairs predictions. It changes prior odds at fixed fitted likelihoods; it does not refit or choose a preferred setting.", "",
              "| Prior assumption | Exact sets | False edges | Missed edges |", "|---|---:|---:|---:|"]
    for prior in PRIORS:
        value = summaries[f"splink/{prior};threshold=0.9/all_pairs"]
        lines.append(f"| {prior} | {value['correct_complete_decisions']} | {value['false_links']} | {value['missed_links']} |")
    lines += ["", f"The publisher graph has {summary['reference_edges']} edges, {summary['reference_nil_queries']} NIL queries and {summary['reference_multiple_target_queries']} multiple-target queries. Uncertainty resamples {summary['query_source_families']} actual query families, using 2,000 replicates and seed 20261007. Full lexical and FS grids, candidate-restricted results, raw query outcomes and pointwise family intervals are saved alongside this table.", "",
              f"Candidates retain {candidate_audit['reference_edges_retrieved']} of {candidate_audit['reference_edges']} reference edges and every target for {candidate_audit['queries_with_complete_reference_set_available']} of {len(truth)} queries. No target was inserted from truth. All identity outcomes were scored after the question and prediction hashes were frozen. No new fitting, paid calls, source acquisition or Amazon evaluation outcomes were used.", ""]
    report = OUTPUT/"reference-summary.md"
    text = "\n".join(lines)
    if report.exists() and report.read_text()!=text:
        raise FileExistsError("Preserve existing reference report")
    report.write_text(text)
    verify_protocol()
    save("completion.json",{"status":"free_conventional_reference_scored","protocol_sha256":runtime.file_hash(OUTPUT/"protocol.json"),
        "files":{p.name:runtime.file_hash(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name!="completion.json"},
        "paid_calls":0,"amazon_test_outcomes_accessed":False})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--score-frozen",action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze_protocol()
        print(json.dumps({"status":"question_and_hashes_frozen_before_outcomes","protocol_sha256":runtime.file_hash(OUTPUT/"protocol.json")},indent=2))
    else:
        print(json.dumps(score_frozen(),indent=2))


if __name__=="__main__":
    main()
