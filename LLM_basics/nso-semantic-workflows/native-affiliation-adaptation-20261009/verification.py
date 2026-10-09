"""Offline verification of the published secondary adaptation control."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import experiment

HERE=Path(__file__).resolve().parent


def metrics(states,gold):
    exact=tp=fp=fn=automatic=review=invalid=false_nil=false_assignment_nil=0
    predicted_counts=Counter();true_counts=Counter()
    for qid,truth in gold.items():
        state=states[qid];valid=state["status"]=="automatic"
        found=set(state["targets"]) if valid else set();target=set(truth)
        exact+=bool(valid and found==target);tp+=len(found&target);fp+=len(found-target);fn+=len(target-found)
        automatic+=valid;review+=state["status"]=="review";invalid+=state["status"]=="invalid"
        false_nil+=bool(valid and target and not found);false_assignment_nil+=bool(valid and not target and found)
        true_counts.update(truth)
        if valid:predicted_counts.update(state["targets"])
    difference={qid:predicted_counts[qid]-true_counts[qid] for qid in set(true_counts)|set(predicted_counts)}
    n=len(gold)
    return {"rows":n,"exact_sets":exact,"exact_set_accuracy":exact/n,"tp":tp,"fp":fp,"fn":fn,
        "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
        "automatic_rows":automatic,"review_rows":review,"invalid_rows":invalid,"automatic_coverage":automatic/n,
        "loss_fp1":fp+fn,"loss_fp2":2*fp+fn,"loss_fp5":5*fp+fn,
        "false_nil_rows":false_nil,"false_assignment_on_nil_rows":false_assignment_nil,
        "reference_allocations":sum(true_counts.values()),"predicted_automatic_allocations":sum(predicted_counts.values()),
        "organization_count_l1_error":sum(abs(d) for d in difference.values()),
        "overallocated_count_units":sum(max(0,d) for d in difference.values()),
        "underallocated_count_units":sum(max(0,-d) for d in difference.values()),
        "organizations_with_nonzero_count_error":sum(d!=0 for d in difference.values())}


def states(scores,rows,cutoff,extracted=False):
    output={}
    for row,values in zip(rows,scores):
        status=row["extraction_status"] if extracted and row["extraction_status"] in {"review","invalid"} else "automatic"
        output[row["query_id"]]={"status":status,"targets":sorted(cid for cid,value in zip(row["candidate_ids"],values) if value>=cutoff)}
    return output


def policy(scores,rows,gold,thresholds,extracted=False):
    candidates=[{"threshold":cutoff,**metrics(states(scores,rows,cutoff,extracted),gold)} for cutoff in thresholds]
    tie=lambda value:(-value["exact_sets"],value["fp"],value["fn"],-value["threshold"])
    output={}
    for cost in [1,2,5]:
        output[f"risk_{cost}"]=min(candidates,key=lambda v:(cost*v["fp"]+v["fn"],*tie(v)))
        allowed=[value for value in candidates if value["fp"]*100<=cost*value["rows"]]
        output[f"quality_{cost}"]=min(allowed,key=tie) if allowed else {"threshold":1.,"status":"constraint_not_met"}
    return output


def fs_levels(row,route,profile):
    x=np.asarray(row[route+"_features"])
    name=np.where((x[:,2]>0)|(x[:,3]>0),4,np.where(x[:,4]>=.95,3,np.where(x[:,4]>=.85,2,np.where(x[:,4]>=.7,1,0)))).astype(int)
    city=np.where(bool(row[route+"_structure"]["cities"]),x[:,11].astype(int),-1)
    country=np.where(bool(row[route+"_structure"]["countries"]),x[:,12].astype(int),-1)
    values=[name,city,country]
    if profile=="expanded":values.extend([np.digitize(x[:,0],[.3,.5,.7,.9]),np.digitize(x[:,1],[.2,.5,.8]),np.digitize(x[:,9]*x[:,18],[2,4,7,10])])
    return np.stack(values,axis=1)


def verify():
    manifest=experiment.read(HERE/"publication-manifest.json")
    for name,expected in manifest["files"].items():
        if experiment.digest(HERE/name)!=expected:raise ValueError("Published file differs: "+name)
    sealed=experiment.read(HERE/"prediction-seal.json")
    for name,expected in sealed["sha256"].items():
        if experiment.digest(HERE/name)!=expected:raise ValueError("Sealed scientific file differs: "+name)
    protocol=experiment.read(HERE/"protocol.json")
    frozen=experiment.read(HERE/"protocol-seal.json")
    if experiment.digest(HERE/"protocol.json")!=frozen["protocol_sha256"]:raise ValueError("Frozen design differs.")
    rows=experiment.read(HERE/"data/observed-pairs.json.gz")
    partitions={role:[r for r in rows if r["adaptation_split"]==role] for role in ["train","validation","test"]}
    for row in rows:
        if experiment.split(row["text"])[2]!=row["adaptation_split"]:raise ValueError("Text-group split differs.")
    if [len(partitions[r]) for r in ["train","validation","test"]]!=[195,212,697]:raise ValueError("Split counts differ.")
    train,val,test=[partitions[r] for r in ["train","validation","test"]]
    labels=experiment.read(HERE/"data/training-validation-labels.json.gz")
    if set(labels)!={r["query_id"] for r in train+val}:raise ValueError("Training includes undesignated labels.")
    access=experiment.read(HERE/"label-access.json")
    if access["test_labels_loaded"] is not False or access["opened_at_utc"]>=sealed["sealed_at_utc"]:raise ValueError("Scoped label-access chronology differs.")
    gold=experiment.read(HERE/"data/final-test-labels.json.gz")
    if set(gold)!={r["query_id"] for r in test}:raise ValueError("Test denominator differs.")
    order=experiment.read(HERE/"data/final-test-query-order.json")["query_ids"]
    if len(order)!=len(gold) or set(order)!=set(gold):raise ValueError("Final reference query order differs.")
    gold={qid:gold[qid] for qid in order}
    saved=experiment.read(HERE/"predictions.json.gz");result=experiment.read(HERE/"results.json")
    observed={(r["arm"],r["policy"]):r for r in result["outcomes"]}
    if len(observed)!=151:raise ValueError("Complete policy table differs.")
    for arm,choices in saved.items():
        for p,decisions in choices.items():
            if set(decisions)!=set(gold):raise ValueError("Methods do not share a test denominator.")
            for name,value in metrics(decisions,gold).items():
                expected=observed[(arm,p)][name]
                if value is None:
                    if expected is not None:raise ValueError("Undefined metric differs: "+name)
                elif not np.isclose(value,expected,rtol=0,atol=1e-12):raise ValueError("Outcome metric differs: "+name)
    candidate_edges=sum(len(set(gold[r["query_id"]])&set(r["candidate_ids"])) for r in test)
    if candidate_edges!=result["candidate_true_target_edges"] or sum(map(len,gold.values()))!=result["true_target_edges"]:
        raise ValueError("Candidate and truth counts differ.")
    configurations=experiment.read(HERE/"validation-policies.json")
    scores=experiment.read(HERE/"scores.json.gz")
    fits=experiment.read(HERE/"fit-audit.json")
    y=np.array([cid in labels[r["query_id"]] for r in train for cid in r["candidate_ids"]],int)
    maxdifference=0.
    for name,model_scores in scores.items():
        route="extracted" if name.startswith("extracted") else "raw"
        if "splink" in name:
            profile="expanded" if name.endswith("expanded") else "compact"
            vector=np.vstack([fs_levels(r,route,profile) for r in train]);sizes=[5,2,2]+([5,4,5] if profile=="expanded" else [])
            parameters=[]
            for j,size in enumerate(sizes):
                mc=np.bincount(vector[y==1,j][vector[y==1,j]>=0],minlength=size)
                uc=np.bincount(vector[y==0,j][vector[y==0,j]>=0],minlength=size)
                m=(mc+.5)/(mc.sum()+.5*size);u=(uc+.5)/(uc.sum()+.5*size)
                if not np.allclose(m,fits[name]["parameters"][j]["m"],rtol=0,atol=1e-12) or not np.allclose(u,fits[name]["parameters"][j]["u"],rtol=0,atol=1e-12):
                    raise ValueError("Native-trained FS evidence probabilities differ.")
                parameters.append((m,u))
            computed={}
            for part,items in [("validation",val),("test",test)]:
                values=[]
                for row in items:
                    level=fs_levels(row,route,profile);prior=y.mean();logodds=np.full(40,np.log(prior/(1-prior)))
                    for j,(m,u) in enumerate(parameters):
                        mask=level[:,j]>=0;logodds[mask]+=np.log(m[level[mask,j]]/u[level[mask,j]])
                    values.append((1/(1+np.exp(-np.clip(logodds,-700,700)))).tolist())
                computed[part]=values
        else:
            feature="raw_full32_features" if name.startswith("raw32") else route+"_features"
            X=np.vstack([r[feature] for r in train]);Xv=np.vstack([r[feature] for r in val]);Xt=np.vstack([r[feature] for r in test])
            if name.endswith("logistic"):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight="balanced",max_iter=2000,random_state=experiment.SEED))
            else:
                model=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=.05,min_samples_leaf=20,l2_regularization=1,
                    early_stopping=False,class_weight="balanced",random_state=experiment.SEED)
            model.fit(X,y)
            computed={"validation":model.predict_proba(Xv)[:,1].reshape(len(val),40).tolist(),
                      "test":model.predict_proba(Xt)[:,1].reshape(len(test),40).tolist()}
        for part in ["validation","test"]:
            difference=float(np.max(np.abs(np.asarray(computed[part])-np.asarray(model_scores[part]))))
            maxdifference=max(maxdifference,difference)
            if difference>1e-10:raise ValueError("Independent model score reconstruction differs: "+name)
        nominated=policy(computed["validation"],val,{r["query_id"]:labels[r["query_id"]] for r in val},protocol["thresholds"],route=="extracted")
        for choice,value in nominated.items():
            if value["threshold"]!=configurations["models"][name]["policies"][choice]["threshold"]:
                raise ValueError("Native validation threshold differs: "+name)
            decisions=states(computed["test"],test,value["threshold"],route=="extracted")
            if decisions!=saved[name][choice]:raise ValueError("Calibrated decisions differ: "+name)
    key=lambda name:(configurations["models"][name]["policies"]["risk_2"]["loss_fp2"],
        -configurations["models"][name]["policies"]["risk_2"]["exact_sets"],configurations["models"][name]["policies"]["risk_2"]["fp"],
        configurations["models"][name]["policies"]["risk_2"]["fn"],name)
    for route in ["raw","extracted"]:
        champion=min((name for name in scores if name.startswith(route)),key=key)
        if champion!=configurations["champions"][route]:raise ValueError("Validation-only model nomination differs.")
    semantic=saved["semantic_selector"]["fixed"]
    for route,champion in configurations["champions"].items():
        comparison=experiment.paired_interval(semantic,saved[champion]["risk_2"],gold,test)
        if comparison!=result["comparisons"]["semantic_vs_adapted_"+route]:raise ValueError("Paired text-group interval differs.")
    report={"status":"verified","outcome_policy_rows":len(observed),"test_rows":len(test),"train_rows":len(train),"validation_rows":len(val),
        "reconstructed_adapted_models":len(scores),"maximum_model_probability_difference":maxdifference,"teacher_calls":0,"result_files_written":0}
    print(json.dumps(report,indent=2,sort_keys=True))
    return report


if __name__=="__main__":
    verify()
