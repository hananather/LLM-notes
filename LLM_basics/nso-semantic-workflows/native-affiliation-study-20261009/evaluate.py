"""Score sealed native affiliation predictions without fitting or API calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study


def verify_seal():
    study.verify_protocol()
    seal=study.read(study.RESULTS/"prediction-seal.json")
    for name,expected in seal["artifact_sha256"].items():
        if study.sha(HERE/name)!=expected:
            raise ValueError(f"Sealed evidence changed: {name}")
    reference=study.read(study.RESULTS/"reference-prediction-seal.json")
    for name,expected in reference["artifact_sha256"].items():
        assert study.sha(HERE/name)==expected
    adaptation_path=study.ROOT/"native-affiliation-adaptation-20261009/prediction-seal.json"
    if not adaptation_path.exists():
        raise RuntimeError("The separately frozen adaptation predictions must be sealed before complete native evaluation labels are opened")
    study.read(adaptation_path)
    source_manifest=study.read(study.ORGS/"data/source-manifest.json")
    gold_expected=next(x["sha256"] for x in source_manifest["prepared_files"] if x["path"]=="data/gold/reference.jsonl")
    assert study.sha(study.ORGS/"data/gold/reference.jsonl")==gold_expected
    for name,expected in study.read(HERE/"protocol.json")["source_sha256"].items():
        actual=study.protocol()["source_sha256"][name]
        assert actual==expected
    seal["supplement_sealed_at"]=reference["sealed_at"]
    seal["adaptation_prediction_seal_sha256"]=study.sha(adaptation_path)
    seal["reference_label_source_sha256"]=gold_expected
    return seal


def detail(state, truth, diagnostic=False):
    valid=state["status"]=="automatic" or (diagnostic and state["status"]=="review")
    pred=set(state["targets"]) if valid else set();truth=set(truth)
    return {"exact":int(valid and pred==truth),"tp":len(pred&truth),"fp":len(pred-truth),"fn":len(truth-pred),
        "nil_correct":int(valid and not truth and not pred),"false_nil":int(valid and bool(truth) and not pred),
        "false_assignment_nil":int(not truth and bool(pred)),"accepted_targets":sorted(pred),"reference_targets":sorted(truth)}


def complete_metrics(states,gold,diagnostic=False):
    value=study.metrics(states,gold,diagnostic)
    rows=[detail(states[qid],truth,diagnostic) for qid,truth in gold.items()]
    value.update({"nil_rows":sum(not t for t in gold.values()),"nil_correct":sum(r["nil_correct"] for r in rows),
        "false_nil_rows":sum(r["false_nil"] for r in rows),"false_assignment_on_nil_rows":sum(r["false_assignment_nil"] for r in rows),
        "false_edges_per_100_queries":100*value["fp"]/len(gold) if gold else None,
        "missing_edges_per_100_queries":100*value["fn"]/len(gold) if gold else None})
    value["micro_f1"]=2*value["tp"]/(2*value["tp"]+value["fp"]+value["fn"]) if 2*value["tp"]+value["fp"]+value["fn"] else 0.0
    accepted={qid:gold[qid] for qid in gold if states[qid]["status"]=="automatic"}
    value["conditional_automatic_exact_sets"]=sum(detail(states[qid],gold[qid])["exact"] for qid in accepted)
    value["conditional_automatic_accuracy"]=value["conditional_automatic_exact_sets"]/len(accepted) if accepted else None
    return value


def allocation_error(states,gold):
    predicted,reference=Counter(),Counter()
    for qid,truth in gold.items():
        reference.update(truth)
        if states[qid]["status"]=="automatic":
            predicted.update(states[qid]["targets"])
    ids=sorted(set(predicted)|set(reference))
    differences={i:predicted[i]-reference[i] for i in ids}
    return {"reference_allocations":sum(reference.values()),"predicted_automatic_allocations":sum(predicted.values()),
        "organization_count_l1_error":sum(abs(d) for d in differences.values()),
        "overallocated_count_units":sum(max(0,d) for d in differences.values()),
        "underallocated_count_units":sum(max(0,-d) for d in differences.values()),
        "organizations_with_reference_allocations":len(reference),"organizations_with_nonzero_count_error":sum(d!=0 for d in differences.values()),
        "interpretation":"Counts of benchmark source affiliation rows allocated per organization; rows are not established unique publications and these are not official publication counts or population-bias estimates",
        "counts":[{"record_id":i,"reference_count":reference[i],"automatic_predicted_count":predicted[i],"difference":differences[i]} for i in ids]}


def candidate_metrics(prepared,gold,key):
    target_total=target_found=positive=complete=0
    for r in prepared:
        truth=set(gold[r["query_id"]]);found=set(r[key])
        target_total+=len(truth);target_found+=len(truth&found)
        positive+=bool(truth);complete+=bool(truth) and truth<=found
    return {"target_total":target_total,"target_found":target_found,"target_recall":target_found/target_total if target_total else None,
        "positive_queries":positive,"complete_positive_queries":complete,"all_targets_per_positive_query_recall":complete/positive if positive else None}


def paired_interval(new,reference,gold,inputs):
    groups=defaultdict(list)
    for qid in gold:
        groups[study.norm(inputs[qid]["text"])].append(qid)
    # Duplicate groups are sampled within their observed source-population mixture.
    bystratum=defaultdict(list)
    for text,qids in groups.items():
        stratum="+".join(sorted({inputs[qid]["source_population"] for qid in qids}))
        row=np.zeros(6)
        row[0]=len(qids)
        for qid in qids:
            a,b=detail(new[qid],gold[qid]),detail(reference[qid],gold[qid])
            row[1]+=a["exact"]-b["exact"];row[2]+=a["fp"]-b["fp"];row[3]+=a["fn"]-b["fn"]
            row[4]+=(a["fp"]+a["fn"])-(b["fp"]+b["fn"])
            row[5]+=(2*a["fp"]+a["fn"])-(2*b["fp"]+b["fn"])
        bystratum[stratum].append(row)
    rng=np.random.default_rng(study.SEED)
    draws=np.zeros((2000,6))
    for rows in bystratum.values():
        matrix=np.asarray(rows)
        indices=rng.integers(0,len(rows),(2000,len(rows)))
        draws+=matrix[indices].sum(axis=1)
    observed=sum((np.asarray(v).sum(axis=0) for v in bystratum.values()),np.zeros(6))
    out={"rows":len(gold),"normalized_duplicate_groups":len(groups),"draws":2000,"interval":"95% percentile paired duplicate-group bootstrap; descriptive benchmark-sample uncertainty"}
    for i,name in enumerate(["exact_set_difference_pp","fp_difference_per100","fn_difference_per100","edge_loss_fp1_difference_per100","edge_loss_fp2_difference_per100"],1):
        ratios=100*draws[:,i]/draws[:,0]
        out[name]={"estimate":float(100*observed[i]/observed[0]),"lower":float(np.quantile(ratios,.025)),"upper":float(np.quantile(ratios,.975))}
    return out


def score():
    seal=verify_seal()
    if (study.RESULTS/"evaluation.json").exists():
        result=study.read(study.RESULTS/"evaluation.json")
        assert result["prediction_seal_sha256"]==study.sha(study.RESULTS/"prediction-seal.json")
        return result
    # The evaluator implementation and question are fixed before this first read.
    policies=study.read(study.RESULTS/"validation-policies.json")
    reference_policies=study.read(study.RESULTS/"reference-validation-policies.json")
    all_validation={**policies["arms"],**reference_policies["arms"]}
    raw=min((name for name in all_validation if name.startswith("raw_") and not name.startswith("raw_matched_")),key=lambda name:(all_validation[name]["policies"]["risk_2"]["loss_fp2"],-all_validation[name]["policies"]["risk_2"]["exact_sets"],name))
    extracted=policies["best_extracted_risk2"]
    boundary={"frozen_at":study.stamp(),"evaluator_sha256":study.sha(__file__),"prediction_seal_sha256":study.sha(study.RESULTS/"prediction-seal.json"),"reference_prediction_seal_sha256":study.sha(study.RESULTS/"reference-prediction-seal.json"),"adaptation_prediction_seal_sha256":seal["adaptation_prediction_seal_sha256"],"primary_raw_selected_only_on_validation":raw,
        "question":"Score every frozen arm/policy on all 1104 native source rows, retain every result, and calculate source-row organization-allocation count error without changing predictions."}
    study.save(study.RESULTS/"evaluation-boundary.json",boundary)
    opened=study.stamp()
    native=study.read(study.DATA/"native-inputs.json.gz");inputs={r["query_id"]:r for r in native};ids=set(inputs)
    reference_rows=[r for r in study.lines(study.ORGS/"data/gold/reference.jsonl") if r["query_id"] in ids]
    assert len(reference_rows)==1104
    gold={r["query_id"]:r["target_ids"] for r in reference_rows}
    prepared=[r for r in study.read(study.DATA/"prepared-pairs.json.gz") if r["split"]=="native"]
    methods={**study.read(study.RESULTS/"conventional-predictions.json.gz"),**study.read(study.RESULTS/"reference-predictions.json.gz")}
    semantic={}
    for r in study.read(study.RESULTS/"selection-answers.json.gz"):
        semantic[r["query_id"]]={"status":r["status"],"targets":sorted({x["record_id"] for x in r.get("answer",{}).get("matches",[])})}
    methods["semantic_selector"]={"fixed":semantic}
    assert all(set(states)==ids for policies in methods.values() for states in policies.values())
    groups={"all":list(ids),**{p:[r["query_id"] for r in native if r["source_population"]==p] for p in ["french","multilingual","multi-org"]},
        "single_target":[k for k,v in gold.items() if len(v)==1],"multiple_targets":[k for k,v in gold.items() if len(v)>1],"historical_nil":[k for k,v in gold.items() if not v]}
    candidates={r["query_id"]:set(r["candidate_ids"]) for r in prepared}
    groups["candidate_complete_positive"]=[k for k,v in gold.items() if v and set(v)<=candidates[k]]
    groups["candidate_missing_positive"]=[k for k,v in gold.items() if v and not set(v)<=candidates[k]]
    scores,allocation={},{}
    for arm,policies in methods.items():
        scores[arm]={};allocation[arm]={}
        for policy,states in policies.items():
            scores[arm][policy]={group:{"automatic":complete_metrics(states,{k:gold[k] for k in keys}),"valid_set_diagnostic":complete_metrics(states,{k:gold[k] for k in keys},True)} for group,keys in groups.items()}
            allocation[arm][policy]={group:allocation_error(states,{k:gold[k] for k in groups[group]}) for group in ["all","french","multilingual","multi-org"]}
    pairs={}
    comparisons=[("semantic_vs_primary_raw",semantic,methods[raw]["risk_2"]),("semantic_vs_primary_extracted",semantic,methods[extracted]["risk_2"])]
    for model in ["logistic","tree","splink_compact","splink_expanded"]:
        comparisons.append(("extraction_mechanism_"+model,methods["extracted_"+model]["risk_2"],methods["raw_matched_"+model]["risk_2"]))
    for label,new,baseline in comparisons:
        pairs[label]={group:paired_interval(new,baseline,{k:gold[k] for k in groups[group]},inputs) for group in ["all","french","multilingual","multi-org"]}
    duplicates=defaultdict(list)
    for r in reference_rows:
        duplicates[study.norm(inputs[r["query_id"]]["text"])].append(r)
    conflicts=[{"query_ids":[r["query_id"] for r in rows],"source_labels":[r["source_label"] for r in rows],"target_sets":[r["target_ids"] for r in rows]} for rows in duplicates.values() if len({tuple(r["target_ids"]) for r in rows})>1]
    brace_audit=[]
    for r in reference_rows:
        for content in re.findall(r"\{([^}]*)\}",r["source_label"]):
            target_ids=re.findall(r"https://ror.org/[a-zA-Z0-9]+",content)
            if len(target_ids)>1:
                brace_audit.append({"query_id":r["query_id"],"brace_content":content,"ror_ids":target_ids})
    old=study.read(study.DATA/"original-training-validation.json.gz")
    train_targets={i for r in old if r["split"]=="train" for i in r["gold"]}
    oldtexts={study.norm(r["text"]) for r in old}
    audit={"source_cardinalities":dict(Counter(len(v) for v in gold.values())),"shared_training_target_queries":sum(bool(set(v)&train_targets) for v in gold.values()),
        "normalized_text_overlap_with_original_train_validation":sum(study.norm(r["text"]) in oldtexts for r in native),"normalized_duplicate_groups":len(duplicates),"duplicate_target_conflicts":conflicts,
        "multiple_ids_inside_one_brace":brace_audit,"target_statuses":dict(Counter(s for r in reference_rows for s in r["target_statuses"])),
        "scoring_relation":"Strict complete flat published ROR-ID set; preserve annotation conflicts and multi-ID braces as a separate audit. Do not reinterpret targets to improve scores.",
        "source_label_examples":[{"query_id":r["query_id"],"population":r["source_population"],"source_label":r["source_label"]} for r in sorted(reference_rows,key=lambda r:r["query_id"])[:12]]}
    study.save(study.RESULTS/"reference-policy-audit.json",audit)
    registry={r["record_id"]:r for r in study.lines(study.ORGS/"data/ror-v1.41-observed.jsonl.gz")}
    outcomes=[]
    for qid in sorted(ids):
        outcomes.append({**inputs[qid],"reference_targets":gold[qid],"candidate_complete":set(gold[qid])<=candidates[qid],
            "methods":{arm:{"state":states[qid],**detail(states[qid],gold[qid])} for arm,states in [("semantic_selector",semantic),(raw,methods[raw]["risk_2"]),(extracted,methods[extracted]["risk_2"])]}})
    study.save(study.RESULTS/"case-outcomes.json.gz",outcomes)
    examples={}
    for cohort in ["french","multilingual","multi-org"]:
        subset=[r for r in outcomes if r["source_population"]==cohort]
        for label,a,b in [("semantic_correct_raw_wrong","semantic_selector",raw),("raw_correct_semantic_wrong",raw,"semantic_selector")]:
            rows=[r for r in subset if r["methods"][a]["exact"] and not r["methods"][b]["exact"]]
            if rows:
                row=rows[0];mentioned=set(row["reference_targets"])|{i for s in row["methods"].values() for i in s["state"]["targets"]}
                examples[cohort+"_"+label]={**row,"organization_names":{i:registry.get(i,{}).get("name","outside candidate registry") for i in mentioned}}
            else:
                examples[cohort+"_"+label]=None
    study.save(study.RESULTS/"examples.json",{"selection":"First query-ID-sorted example in each cohort and paired-disagreement direction, after scoring; illustrative, not additional evidence", "cases":examples})
    candidate={group:{key:candidate_metrics([r for r in prepared if r["query_id"] in set(groups[group])],{k:gold[k] for k in groups[group]},key) for key in ["raw_retrieval_ids","extracted_retrieval_ids","candidate_ids"]} for group in ["all","french","multilingual","multi-org"]}
    result={"evaluated_at":study.stamp(),"main_evaluation_labels_opened_at":opened,"label_access_scope":"This timestamp is the main evaluator's complete evaluation-label read, not any earlier separately authorized adaptation training/validation label access.","adaptation_prediction_seal_sha256":seal["adaptation_prediction_seal_sha256"],"prediction_sealed_at":max(seal["sealed_at"],seal["supplement_sealed_at"]),"prediction_seal_sha256":study.sha(study.RESULTS/"prediction-seal.json"),"reference_prediction_seal_sha256":study.sha(study.RESULTS/"reference-prediction-seal.json"),
        "evaluator_sha256":study.sha(__file__),"rows":1104,"group_sizes":{k:len(v) for k,v in groups.items()},"primary_raw":raw,"primary_extracted":extracted,
        "scores":scores,"allocation_count_errors":allocation,"paired_intervals":pairs,"candidate_recall":candidate,"cost":seal["cost"],"reference_audit_sha256":study.sha(study.RESULTS/"reference-policy-audit.json"),
        "claim_boundary":"External public affiliation benchmark with shared institutions, historical annotations and possible public model exposure. NSO-relevant institution-level row allocation; no official publication-count, population-bias or universal-method superiority claim."}
    study.save(study.RESULTS/"evaluation.json",result)
    with (study.RESULTS/"summary.csv").open("x",newline="") as f:
        fields=["arm","policy","cohort","rows","exact_sets","exact_set_accuracy","tp","fp","fn","precision","recall","automatic_coverage","review_rows","invalid_rows","loss_fp1","loss_fp2","loss_fp5","organization_count_l1_error"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for arm,pols in scores.items():
            for policy,cohorts in pols.items():
                for cohort in ["all","french","multilingual","multi-org"]:
                    value={k:v for k,v in cohorts[cohort]["automatic"].items() if k in fields}
                    writer.writerow({"arm":arm,"policy":policy,"cohort":cohort,**value,"organization_count_l1_error":allocation[arm][policy][cohort]["organization_count_l1_error"]})
    return result


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result=score();raw=result["primary_raw"];ext=result["primary_extracted"]
    arms=[(raw,"risk_2","Trained raw control"),(ext,"risk_2","Extraction → trained control"),("semantic_selector","fixed","Semantic selector")]
    colors=["#737574","#427E86","#CC6D4D"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.labelcolor":"#242424","text.color":"#242424","axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,3,figsize=(15,5.3),gridspec_kw={"wspace":.42})
    properties=[("exact_set_accuracy","Complete organization sets","% of all 1,104 source rows",100), ("false_edges_per_100_queries","False organization allocations","Edges per 100 source rows",1), ("automatic_coverage","Automatic decisions","% of all 1,104 source rows",100)]
    for ax,(key,title,ylabel,mult) in zip(axs,properties):
        vals=[result["scores"][a][p]["all"]["automatic"][key]*mult for a,p,_ in arms]
        bars=ax.bar(range(3),vals,color=colors,width=.62)
        ax.set_title(title,loc="left",fontweight="bold",fontsize=13,pad=18);ax.set_ylabel(ylabel)
        ax.set_xticks(range(3),[x[2].replace(" → ","\n→ ").replace(" raw ","\nraw ").replace(" selector","\nselector") for x in arms],fontsize=10)
        ax.set_ylim(0,(max(vals)*1.2 if max(vals)>0 else 1));ax.set_axisbelow(True);ax.grid(axis="y",alpha=.17)
        for bar,value in zip(bars,vals):
            ax.annotate(f"{value:.1f}"+("%" if mult==100 else ""),(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,6),textcoords="offset points",ha="center",fontweight="bold")
    fig.suptitle("Native affiliations: allocation accuracy and automatic coverage",x=.075,y=1.0,ha="left",fontweight="bold",fontsize=17)
    fig.text(.075,.015,"All 1,104 published affiliation rows; ROR v1.41. Controls and policies selected on original S2AFF validation.\nReview and invalid responses remain unresolved. Descriptive public benchmark; source rows are not established unique publications.",fontsize=9)
    fig.subplots_adjust(bottom=.27,top=.82,left=.075,right=.98)
    fig.savefig(HERE/"comparison.png",dpi=180,bbox_inches="tight");fig.savefig(HERE/"comparison.svg",bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.3))
    values=[result["allocation_count_errors"][a][p]["all"]["organization_count_l1_error"] for a,p,_ in arms]
    bars=ax.barh(range(3),values,color=colors,height=.6)
    ax.set_yticks(range(3),[x[2] for x in arms]);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.2);ax.set_axisbelow(True);ax.grid(axis="x",alpha=.17)
    ax.set_title("Error in organization-level affiliation-row counts",loc="left",fontweight="bold",fontsize=16,pad=20)
    ax.set_xlabel("Sum of absolute count errors across organizations, Σ |predicted − reference|")
    for bar,value in zip(bars,values):
        ax.annotate(f"{value:,}",(bar.get_width(),bar.get_y()+bar.get_height()/2),xytext=(7,0),textcoords="offset points",va="center",fontweight="bold")
    fig.text(.03,.02,"Counts are allocations of benchmark source affiliation rows, not official publication counts. Review is unresolved.\nOver- and under-allocation can cancel within an organization; pair-level false/missing edges are reported separately.",fontsize=9)
    fig.subplots_adjust(bottom=.22,left=.29,right=.94,top=.83)
    fig.savefig(HERE/"allocation-count-error.png",dpi=180,bbox_inches="tight");fig.savefig(HERE/"allocation-count-error.svg",bbox_inches="tight");plt.close(fig)
    cohorts=["french","multilingual","multi-org"]
    fig,ax=plt.subplots(figsize=(11,5.2));x=np.arange(3);width=.24
    for i,(arm,policy,label) in enumerate(arms):
        vals=[result["scores"][arm][policy][c]["automatic"]["exact_set_accuracy"]*100 for c in cohorts]
        bars=ax.bar(x+(i-1)*width,vals,width,color=colors[i],label=label)
        for bar,value in zip(bars,vals):
            ax.annotate(f"{value:.1f}%",(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords="offset points",ha="center",fontsize=9)
    ax.set_ylim(0,110);ax.set_xticks(x,["French · 614 rows","Multilingual · 322 rows","Multiple organizations · 168 rows"]);ax.set_ylabel("Complete automatic organization sets (% of source rows)");ax.set_axisbelow(True);ax.grid(axis="y",alpha=.17)
    ax.set_title("The three native-text populations remain separate",loc="left",fontweight="bold",fontsize=16,pad=20)
    ax.legend(loc="upper left",bbox_to_anchor=(0,1.0),frameon=False,fontsize=9,ncol=3)
    fig.text(.07,.02,"Frozen methods and policies; review and invalid remain unresolved. Historical published complete-set annotations.\nCohorts are benchmark source collections and do not estimate a national research-output population.",fontsize=9)
    fig.subplots_adjust(bottom=.23,top=.8,left=.07,right=.98)
    fig.savefig(HERE/"cohort-comparison.png",dpi=180,bbox_inches="tight");fig.savefig(HERE/"cohort-comparison.svg",bbox_inches="tight");plt.close(fig)
    return {"plots":["comparison.png","allocation-count-error.png","cohort-comparison.png"]}


def verify():
    seal=verify_seal();result=score()
    assert study.sha(__file__)==result["evaluator_sha256"]
    assert study.sha(study.RESULTS/"reference-policy-audit.json")==result["reference_audit_sha256"]
    for arm,policies in result["scores"].items():
        for policy,cohorts in policies.items():
            assert cohorts["all"]["automatic"]["rows"]==1104
            assert sum(cohorts[c]["automatic"]["rows"] for c in ["french","multilingual","multi-org"])==1104
            value=cohorts["all"]["automatic"]
            assert value["automatic_rows"]+value["review_rows"]+value["invalid_rows"]==1104
            assert value["tp"]+value["fn"]==result["candidate_recall"]["all"]["candidate_ids"]["target_total"]
    output={"verified_at":study.stamp(),"sealed_artifacts_verified":len(seal["artifact_sha256"]),"cohort_denominators_verified":True,"edge_accounting_verified":True,"source_hashes_verified":True,"paid_replay_calls":0,"evaluation_before_gold_boundary_verified":result["prediction_sealed_at"]<result["main_evaluation_labels_opened_at"]}
    if not (study.RESULTS/"verification.json").exists():
        study.save(study.RESULTS/"verification.json",output)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["score","plots","verify"]);args=parser.parse_args()
    result={"score":score,"plots":plots,"verify":verify}[args.command]()
    if args.command=="score":
        result={"primary_raw":result["primary_raw"],"primary_extracted":result["primary_extracted"],"all":{arm:{policy:r["all"]["automatic"] for policy,r in policies.items()} for arm,policies in result["scores"].items()}}
    print(study.canonical(result))
