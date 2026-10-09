"""Retain the earlier complete 32-feature conventional model family.

This free supplement fits only original training labels and chooses policies
only on original validation. Native reference labels are never read here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import study

LEGACY=study.ROOT/"affiliation-value/run.py"


def load_legacy():
    spec=importlib.util.spec_from_file_location("affiliation_reference",LEGACY)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.linkage.norm=study.norm
    return module


def protocol():
    legacy=load_legacy()
    return {"version":"full-feature-reference-v1","frozen_before_native_labels":True,
        "purpose":"Retain the previous full32-feature trained conventional family as an external native-text control",
        "feature_names":legacy.FEATURES,"normalization":"Same Unicode-preserving normalization as main experiment",
        "source_feature_code_sha256":study.sha(LEGACY),"prepared_pairs_sha256":study.sha(study.DATA/"prepared-pairs.json.gz"),
        "registry_sha256":study.sha(study.ORGS/"data/ror-v1.41-observed.jsonl.gz"),"old_training_validation_sha256":study.sha(study.DATA/"original-training-validation.json.gz"),
        "candidate_information":"Same40 candidate IDs; raw route scores, names/aliases/acronyms/cities/countries from complete pinned registry; no target insertion",
        "fit":"Full original training only; validation588 original rows; old test and native evaluation labels never read",
        "models":{"logistic":{"C":1,"class_weight":"balanced","max_iter":2000,"standardize":True},
            "tree":{"max_depth":3,"max_iter":200,"learning_rate":.05,"min_samples_leaf":20,"l2_regularization":1,"class_weight":"balanced","early_stopping":False}},
        "thresholds":study.THRESHOLDS,"policies":study.read(HERE/"protocol.json")["policies"],
        "primary_family_choice":"Choose on original validation risk2, then higher exact sets; include full-feature arms with the other full-validation raw controls",
        "native_model_calls":0,"seed":study.SEED}


def run():
    study.verify_protocol()
    if (study.RESULTS/"reference-prediction-seal.json").exists():
        return study.read(study.RESULTS/"reference-prediction-seal.json")
    start=time.perf_counter();legacy=load_legacy()
    study.save(HERE/"reference-protocol.json",protocol())
    study.save(study.RESULTS/"reference-protocol-seal.json",{"frozen_at":study.stamp(),"code_sha256":study.sha(__file__),"protocol_sha256":study.sha(HERE/"reference-protocol.json"),"native_gold_opened":False})
    roster=study.lines(study.ORGS/"data/ror-v1.41-observed.jsonl.gz")
    organizations=[{"id":r["record_id"],"names":[r["name"],*r["aliases"],*r["acronyms"],*[x["label"] for x in r["labels"]]],
        "acronyms":r["acronyms"],"cities":r["cities"],"country":r["country"],"country_code":r["country_code"]} for r in roster]
    metadata,idf=legacy.registry_features(organizations)
    prepared=study.read(study.DATA/"prepared-pairs.json.gz")
    feature_rows=[]
    for i,r in enumerate(prepared):
        cs=[{"id":cid,"char_score":f[0],"word_score":f[1],"location_score":(f[11]+f[12])/2,
            "contained_alias":bool(f[2]),"score":f[17]} for cid,f in zip(r["candidate_ids"],r["raw_features"])]
        cs.sort(key=lambda c:(-c["score"],c["id"]))
        features=legacy.pair_features(r["text"],cs,metadata,idf)
        byid={c["id"]:f for c,f in zip(cs,features)}
        feature_rows.append({"query_id":r["query_id"],"split":r["split"],"candidate_ids":r["candidate_ids"],"features":[byid[cid] for cid in r["candidate_ids"]]})
        if (i+1)%500==0:
            print(study.canonical({"full_feature_rows":i+1,"total":len(prepared),"elapsed_seconds":time.perf_counter()-start}),flush=True)
    study.save(study.DATA/"full-reference-features.json.gz",{"feature_names":legacy.FEATURES,"rows":feature_rows})
    parts={split:[r for r in feature_rows if r["split"]==split] for split in ["train","val","native"]}
    native=[r for r in prepared if r["split"]=="native"];val=[r for r in prepared if r["split"]=="val"]
    old=study.read(study.DATA/"original-training-validation.json.gz");gold={r["query_id"]:r["gold"] for r in old}
    matrices={split:np.vstack([r["features"] for r in rows]) for split,rows in parts.items()}
    y=np.array([cid in gold[r["query_id"]] for r in parts["train"] for cid in r["candidate_ids"]],int)
    outputs,policies={},{}
    models={"raw_full32_logistic":make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight="balanced",max_iter=2000,random_state=study.SEED)),
        "raw_full32_tree":HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=.05,min_samples_leaf=20,l2_regularization=1,early_stopping=False,class_weight="balanced",random_state=study.SEED)}
    for name,model in models.items():
        model.fit(matrices["train"],y)
        vs=model.predict_proba(matrices["val"])[:,1].reshape(len(val),study.K).tolist()
        ns=model.predict_proba(matrices["native"])[:,1].reshape(len(native),study.K).tolist()
        policies[name]=study.select_policies(vs,val,{r["query_id"]:gold[r["query_id"]] for r in val})
        outputs[name]={policy:study.states(ns,native,value["threshold"]) for policy,value in policies[name]["policies"].items()}
        study.save(study.RESULTS/(name+"-scores.json.gz"),{"validation":[{"query_id":r["query_id"],"candidate_ids":r["candidate_ids"],"scores":s} for r,s in zip(val,vs)],"native":[{"query_id":r["query_id"],"candidate_ids":r["candidate_ids"],"scores":s} for r,s in zip(native,ns)]})
        print(study.canonical({"model":name,"validation_risk2":policies[name]["policies"]["risk_2"]}),flush=True)
    study.save(study.RESULTS/"reference-validation-policies.json",{"arms":policies,"native_gold_opened":False,"training_pairs":len(y),"positive_training_pairs":int(y.sum())})
    study.save(study.RESULTS/"reference-predictions.json.gz",outputs)
    paths=[HERE/"trained_reference.py",HERE/"reference-protocol.json",study.RESULTS/"reference-protocol-seal.json",study.DATA/"full-reference-features.json.gz",study.RESULTS/"reference-validation-policies.json",study.RESULTS/"reference-predictions.json.gz"]
    seal={"sealed_at":study.stamp(),"native_gold_opened":False,"elapsed_seconds":time.perf_counter()-start,"artifact_sha256":{str(p.relative_to(HERE)):study.sha(p) for p in paths}}
    study.save(study.RESULTS/"reference-prediction-seal.json",seal)
    return seal


if __name__=="__main__":
    print(study.canonical(run()),flush=True)
