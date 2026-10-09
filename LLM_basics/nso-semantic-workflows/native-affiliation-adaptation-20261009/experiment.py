"""Secondary conventional adaptation to a frozen native affiliation cohort.

No API calls. Observed evidence is copied before labels. Training and policy
selection receive only their designated normalized-text groups. Final labels
are parsed only after all adapted and reference predictions have been sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import time
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/"native-affiliation-study-20261009"
SEED=20261009


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path=Path(path)
    data=gzip.decompress(path.read_bytes()).decode() if path.suffix==".gz" else path.read_text()
    return json.loads(data)


def save(path,value,exclusive=True):
    path=Path(path)
    if exclusive and path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n"
    data=gzip.compress(text.encode(),mtime=0) if path.suffix==".gz" else text.encode()
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_bytes(data)
    temp.replace(path)


def native_science():
    specification=importlib.util.spec_from_file_location("native_affiliation_science",NATIVE/"study.py")
    module=importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def norm(text):
    text=unicodedata.normalize("NFKD",str(text or "")).casefold()
    return " ".join(re.findall(r"\w+","".join(c for c in text if unicodedata.category(c)!="Mn"),flags=re.UNICODE))


def split(text):
    group=sha256(("native-adaptation-v1:"+norm(text)).encode()).hexdigest()
    bucket=int(group[:8],16)%100
    role="train" if bucket<20 else "validation" if bucket<40 else "test"
    return group,bucket,role


def verify_protocol():
    seal=read(HERE/"protocol-seal.json")
    if digest(HERE/"protocol.json")!=seal["protocol_sha256"] or seal["native_gold_opened"] is not False:
        raise ValueError("The adaptation specification differs from its pre-label freeze.")
    if digest(NATIVE/"protocol.json")!=read(HERE/"protocol.json")["native_protocol_sha256"]:
        raise ValueError("Native model/feature protocol changed.")
    return seal


def bind_observed():
    """Read only observed inputs and already computed candidate-pair evidence."""
    verify_protocol()
    prepared=read(NATIVE/"data/prepared-pairs.json.gz")
    native_inputs={r["query_id"]:r for r in read(NATIVE/"data/native-inputs.json.gz")}
    rows=[{**r,"source_population":native_inputs[r["query_id"]]["source_population"]}
          for r in prepared if r["split"]=="native"]
    if len(rows)!=1104:
        raise ValueError("Complete native observed cohort is required.")
    membership=[]
    for row in rows:
        group,bucket,role=split(row["text"])
        row["adaptation_split"]=role
        membership.append({"query_id":row["query_id"],"duplicate_group":group,"bucket":bucket,
                           "split":role,"source_population":row["source_population"]})
        if len(row["candidate_ids"])!=40 or np.asarray(row["raw_features"]).shape!=(40,19) or np.asarray(row["extracted_features"]).shape!=(40,19):
            raise ValueError("Expected frozen40-candidate19-feature evidence.")
    optional=NATIVE/"data/full-reference-features.json.gz"
    family32=None
    if optional.exists():
        raw=read(optional)
        feature_lookup={r["query_id"]:r for r in raw["rows"] if r["split"]=="native"}
        for row in rows:
            frozen=feature_lookup[row["query_id"]]
            if row["candidate_ids"]!=frozen["candidate_ids"] or np.asarray(frozen["features"]).shape!=(40,32):
                raise ValueError("Native32-feature candidate alignment differs.")
            row["raw_full32_features"]=frozen["features"]
        family32={"feature_names":raw["feature_names"],"source_sha256":digest(optional)}
    save(HERE/"data/observed-pairs.json.gz",rows)
    pd.DataFrame(membership).to_csv(HERE/"data/split-membership.csv",index=False)
    counts={role:{"rows":sum(r["split"]==role for r in membership),
        "duplicate_groups":len({r["duplicate_group"] for r in membership if r["split"]==role})}
        for role in ["train","validation","test"]}
    binding={"bound_at_utc":now(),"native_gold_opened":False,"counts":counts,
        "native_prepared_sha256":digest(NATIVE/"data/prepared-pairs.json.gz"),
        "observed_pairs_sha256":digest(HERE/"data/observed-pairs.json.gz"),
        "membership_sha256":digest(HERE/"data/split-membership.csv"),"raw_full32":family32,
        "protocol_sha256":digest(HERE/"protocol.json")}
    save(HERE/"data-binding.json",binding)
    print(json.dumps(binding,indent=2),flush=True)
    return binding


def verify_main_seal():
    """Require immutable main/reference predictions before new label access."""
    verify_protocol()
    main=read(NATIVE/"results/prediction-seal.json")
    reference=read(NATIVE/"results/reference-prediction-seal.json")
    for seal in [main,reference]:
        if seal["native_gold_opened"] is not False:
            raise ValueError("Main predictions must have been sealed before native truth.")
        for name,value in seal["artifact_sha256"].items():
            if digest(NATIVE/name)!=value:
                raise ValueError("Main scientific evidence changed: "+name)
    return {"main_prediction_seal_sha256":digest(NATIVE/"results/prediction-seal.json"),
            "reference_prediction_seal_sha256":digest(NATIVE/"results/reference-prediction-seal.json")}


def gold_for_ids(allowed_ids):
    """Parse targets only on requested IDs; other groups' targets are not loaded."""
    allowed=set(allowed_ids)
    path=HERE.parent/"adversarial-2026/organizations/data/gold/reference.jsonl"
    answer={}
    query_pattern=re.compile(r'"query_id"\s*:\s*("(?:\\.|[^"\\])*")')
    with path.open() as stream:
        for line in stream:
            match=query_pattern.search(line)
            if not match:
                raise ValueError("Reference row lacks an accessible query identifier.")
            qid=json.loads(match.group(1))
            if qid in allowed:
                row=json.loads(line)
                answer[qid]=row["target_ids"]
    if set(answer)!=allowed:
        raise ValueError("Incomplete label allocation.")
    return answer


def fit_and_seal():
    if (HERE/"prediction-seal.json").exists():
        raise FileExistsError("Adaptation predictions already sealed.")
    (HERE/"models").mkdir(exist_ok=True)
    main_seals=verify_main_seal()
    binding=read(HERE/"data-binding.json")
    if digest(HERE/"data/observed-pairs.json.gz")!=binding["observed_pairs_sha256"]:
        raise ValueError("Observed adaptation evidence changed.")
    start=time.perf_counter()
    science=native_science()
    rows=read(HERE/"data/observed-pairs.json.gz")
    partitions={role:[r for r in rows if r["adaptation_split"]==role] for role in ["train","validation","test"]}
    train,val,test=[partitions[r] for r in ["train","validation","test"]]
    allowed_ids=[r["query_id"] for r in train+val]
    labels=gold_for_ids(allowed_ids)
    train_gold={r["query_id"]:labels[r["query_id"]] for r in train}
    val_gold={r["query_id"]:labels[r["query_id"]] for r in val}
    save(HERE/"data/training-validation-labels.json.gz",labels)
    save(HERE/"label-access.json",{"opened_at_utc":now(),"train_ids":[r["query_id"] for r in train],
        "validation_ids":[r["query_id"] for r in val],"test_labels_loaded":False,"main_seals":main_seals})
    target=np.array([cid in train_gold[r["query_id"]] for r in train for cid in r["candidate_ids"]],int)
    if len(np.unique(target))!=2:
        raise ValueError("Frozen native training partition lacks two classes; retain the design and stop.")
    predictions={};policies={};fits={};scores={}
    for route in ["raw","extracted"]:
        X=np.vstack([r[route+"_features"] for r in train])
        Xv=np.vstack([r[route+"_features"] for r in val]);Xt=np.vstack([r[route+"_features"] for r in test])
        models={"logistic":make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight="balanced",max_iter=2000,random_state=SEED)),
                "tree":HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=.05,min_samples_leaf=20,
                    l2_regularization=1,early_stopping=False,class_weight="balanced",random_state=SEED)}
        for kind,model in models.items():
            model.fit(X,target)
            vs=model.predict_proba(Xv)[:,1].reshape(len(val),40).tolist()
            ts=model.predict_proba(Xt)[:,1].reshape(len(test),40).tolist()
            name=route+"19_"+kind
            policies[name]=science.select_policies(vs,val,val_gold,route=="extracted")
            predictions[name]={policy:science.states(ts,test,choice["threshold"],route=="extracted")
                for policy,choice in policies[name]["policies"].items()}
            scores[name]={"validation":vs,"test":ts}
            joblib.dump(model,HERE/"models"/(name+".joblib"))
            fits[name]={"train_queries":len(train),"train_pairs":len(target),"positive_pairs":int(target.sum()),"model_parameters":repr(model.get_params(deep=False))}
        for profile in ["compact","expanded"]:
            name=route+"19_splink_"+profile
            fs=science.fit_fs(train,train_gold,route,profile)
            vs=science.fs_scores(val,fs,route);ts=science.fs_scores(test,fs,route)
            policies[name]=science.select_policies(vs,val,val_gold,route=="extracted")
            predictions[name]={policy:science.states(ts,test,choice["threshold"],route=="extracted")
                for policy,choice in policies[name]["policies"].items()}
            scores[name]={"validation":vs,"test":ts};fits[name]=fs
    if binding["raw_full32"] is not None:
        X=np.vstack([r["raw_full32_features"] for r in train]);Xv=np.vstack([r["raw_full32_features"] for r in val]);Xt=np.vstack([r["raw_full32_features"] for r in test])
        for kind,model in models.items():
            model.fit(X,target)
            vs=model.predict_proba(Xv)[:,1].reshape(len(val),40).tolist();ts=model.predict_proba(Xt)[:,1].reshape(len(test),40).tolist()
            name="raw32_"+kind
            policies[name]=science.select_policies(vs,val,val_gold)
            predictions[name]={policy:science.states(ts,test,choice["threshold"])
                for policy,choice in policies[name]["policies"].items()}
            scores[name]={"validation":vs,"test":ts};joblib.dump(model,HERE/"models"/(name+".joblib"))
            fits[name]={"train_queries":len(train),"train_pairs":len(target),"positive_pairs":int(target.sum()),"model_parameters":repr(model.get_params(deep=False))}
    key=lambda name:(policies[name]["policies"]["risk_2"]["loss_fp2"],-policies[name]["policies"]["risk_2"]["exact_sets"],
                     policies[name]["policies"]["risk_2"]["fp"],policies[name]["policies"]["risk_2"]["fn"],name)
    champions={"raw":min((n for n in policies if n.startswith("raw")),key=key),
               "extracted":min((n for n in policies if n.startswith("extracted")),key=key)}
    save(HERE/"validation-policies.json",{"models":policies,"champions":champions,"test_truth_loaded":False})
    save(HERE/"fit-audit.json",fits);save(HERE/"scores.json.gz",scores)
    # Transfer and semantic references retain all original policies and decisions.
    test_ids={r["query_id"] for r in test}
    legacy={**read(NATIVE/"results/conventional-predictions.json.gz"),**read(NATIVE/"results/reference-predictions.json.gz")}
    for name,choices in legacy.items():
        predictions["legacy_"+name]={policy:{qid:state for qid,state in states.items() if qid in test_ids}
                                     for policy,states in choices.items()}
    semantic={r["query_id"]:{"status":r["status"],"targets":sorted({x["record_id"] for x in r.get("answer",{}).get("matches",[])})}
              for r in read(NATIVE/"results/selection-answers.json.gz") if r["query_id"] in test_ids}
    predictions["semantic_selector"]={"fixed":semantic}
    if any(set(states)!=test_ids for choices in predictions.values() for states in choices.values()):
        raise ValueError("Predictions do not cover the same final text-group holdout.")
    save(HERE/"predictions.json.gz",predictions)
    paths=["protocol.json","protocol-seal.json","data-binding.json","data/observed-pairs.json.gz","data/split-membership.csv",
        "data/training-validation-labels.json.gz","label-access.json","validation-policies.json","fit-audit.json","scores.json.gz","predictions.json.gz","experiment.py"]
    paths += [str(p.relative_to(HERE)) for p in (HERE/"models").glob("*.joblib")]
    seal={"sealed_at_utc":now(),"test_gold_opened":False,"sha256":{p:digest(HERE/p) for p in paths},
          "main_seals":main_seals,"champions":champions,"fit_seconds":time.perf_counter()-start,"paid_calls":0}
    save(HERE/"prediction-seal.json",seal)
    print(json.dumps({"champions":champions,"fit_seconds":seal["fit_seconds"],"test_gold_opened":False},indent=2),flush=True)
    return seal


def complete_metrics(states,gold):
    value=native_science().metrics(states,gold)
    value["false_nil_rows"]=sum(bool(gold[qid]) and states[qid]["status"]=="automatic" and not states[qid]["targets"] for qid in gold)
    value["false_assignment_on_nil_rows"]=sum(not gold[qid] and states[qid]["status"]=="automatic" and bool(states[qid]["targets"]) for qid in gold)
    return value


def allocation(states,gold):
    predicted,reference=Counter(),Counter()
    for qid,truth in gold.items():
        reference.update(truth)
        if states[qid]["status"]=="automatic":predicted.update(states[qid]["targets"])
    identifiers=set(reference)|set(predicted)
    differences={qid:predicted[qid]-reference[qid] for qid in identifiers}
    return {"reference_allocations":sum(reference.values()),"predicted_automatic_allocations":sum(predicted.values()),
        "organization_count_l1_error":sum(abs(d) for d in differences.values()),
        "overallocated_count_units":sum(max(0,d) for d in differences.values()),
        "underallocated_count_units":sum(max(0,-d) for d in differences.values()),
        "organizations_with_nonzero_count_error":sum(d!=0 for d in differences.values())}


def per_row(state,truth):
    automatic=state["status"]=="automatic"
    found=set(state["targets"]) if automatic else set();truth=set(truth)
    return np.array([int(automatic and found==truth),len(found-truth),len(truth-found)],float)


def paired_interval(new,old,gold,rows):
    lookup={r["query_id"]:r for r in rows};groups=defaultdict(list)
    for qid in gold:groups[norm(lookup[qid]["text"])].append(qid)
    strata=defaultdict(list)
    for qids in groups.values():
        population="+".join(sorted({lookup[q]["source_population"] for q in qids}))
        values=np.zeros(6);values[0]=len(qids)
        for qid in qids:
            difference=per_row(new[qid],gold[qid])-per_row(old[qid],gold[qid])
            values[1:4]+=difference;values[4]+=difference[1]+difference[2];values[5]+=2*difference[1]+difference[2]
        strata[population].append(values)
    rng=np.random.default_rng(SEED);draws=np.zeros((2000,6));observed=np.zeros(6)
    for entries in strata.values():
        matrix=np.asarray(entries);indices=rng.integers(0,len(entries),(2000,len(entries)))
        draws+=matrix[indices].sum(axis=1);observed+=matrix.sum(axis=0)
    out={"rows":len(gold),"duplicate_groups":len(groups),"draws":2000,"interpretation":"Paired descriptive normalized-text-group intervals; not population inference."}
    for i,name in enumerate(["exact_set_difference_pp","fp_difference_per100","fn_difference_per100","loss_fp1_difference_per100","loss_fp2_difference_per100"],1):
        values=100*draws[:,i]/draws[:,0]
        out[name]={"estimate":float(100*observed[i]/observed[0]),"lower":float(np.quantile(values,.025)),"upper":float(np.quantile(values,.975))}
    return out


def evaluate():
    if (HERE/"results.json").exists():raise FileExistsError("Adaptation outcomes already exist; preserve the frozen run.")
    seal=read(HERE/"prediction-seal.json")
    for path,expected in seal["sha256"].items():
        if digest(HERE/path)!=expected:raise ValueError("Adapted sealed artifact differs: "+path)
    rows=[r for r in read(HERE/"data/observed-pairs.json.gz") if r["adaptation_split"]=="test"]
    gold=gold_for_ids([r["query_id"] for r in rows])
    save(HERE/"data/final-test-labels.json.gz",gold)
    predictions=read(HERE/"predictions.json.gz");policies=read(HERE/"validation-policies.json")
    legacy_policies={**read(NATIVE/"results/validation-policies.json")["arms"],**read(NATIVE/"results/reference-validation-policies.json")["arms"]}
    legacy_raw=min((n for n in legacy_policies if n.startswith("raw_") and not n.startswith("raw_matched_")),
        key=lambda name:(legacy_policies[name]["policies"]["risk_2"]["loss_fp2"],-legacy_policies[name]["policies"]["risk_2"]["exact_sets"],name))
    legacy_extracted=read(NATIVE/"results/validation-policies.json")["best_extracted_risk2"]
    outcomes=[]
    for name,choices in predictions.items():
        for policy,states in choices.items():
            outcomes.append({"arm":name,"policy":policy,**complete_metrics(states,gold),**allocation(states,gold)})
    pd.DataFrame(outcomes).to_csv(HERE/"results.csv",index=False)
    semantic=predictions["semantic_selector"]["fixed"]
    comparisons={}
    for route,champion in policies["champions"].items():
        adapted=predictions[champion]["risk_2"]
        legacy=predictions["legacy_"+(legacy_raw if route=="raw" else legacy_extracted)]["risk_2"]
        comparisons["semantic_vs_adapted_"+route]=paired_interval(semantic,adapted,gold,rows)
        comparisons["adapted_vs_legacy_"+route]=paired_interval(adapted,legacy,gold,rows)
    candidates={r["query_id"]:set(r["candidate_ids"]) for r in rows}
    result={"created_at_utc":now(),"prediction_seal_sha256":digest(HERE/"prediction-seal.json"),
        "scope":"Secondary within-curated-cohort unseen-text adaptation; main all1104 external evaluation remains primary; no unseen-organization claim.",
        "test_rows":len(rows),"test_duplicate_groups":len({norm(r["text"]) for r in rows}),
        "train_validation_counts":read(HERE/"data-binding.json")["counts"],"champions":policies["champions"],
        "legacy_champions":{"raw":legacy_raw,"extracted":legacy_extracted},"true_target_edges":sum(map(len,gold.values())),
        "candidate_true_target_edges":sum(len(set(truth)&candidates[qid]) for qid,truth in gold.items()),
        "candidate_complete_positive_rows":sum(bool(truth) and set(truth)<=candidates[qid] for qid,truth in gold.items()),
        "outcomes":outcomes,"comparisons":comparisons,"paid_calls":0,
        "allocation_interpretation":"Source affiliation rows allocated to organizations; rows are not established unique publications or official production counts."}
    save(HERE/"results.json",result)
    print(pd.DataFrame(outcomes).loc[lambda frame:(frame.arm.isin([*policies["champions"].values(),"semantic_selector"]))&(frame.policy.isin(["risk_2","fixed"]))].to_string(index=False),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["bind","fit","evaluate"])
    args=parser.parse_args()
    (HERE/"models").mkdir(exist_ok=True)
    {"bind":bind_observed,"fit":fit_and_seal,"evaluate":evaluate}[args.mode]()
