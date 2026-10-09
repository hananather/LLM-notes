"""Read-only scientific verification using published files and source data.

No API calls, file writes, baseline refits, acquisition retuning or result edits.
The saved pair-similarity students are independently refitted on exactly the
published training pairs to check their learned scores and calibration.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import experiment as science

HERE = Path(__file__).resolve().parent


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def check_manifest(directory):
    manifest=json.loads((directory/"publication-manifest.json").read_text())
    for name,expected in manifest["files"].items():
        path=directory/name
        if not path.is_file() or file_hash(path)!=expected:
            raise ValueError("Published artifact hash differs: "+name)
    for name,expected in manifest["external_inputs"].items():
        path=(directory/name).resolve()
        if not path.is_file() or file_hash(path)!=expected:
            raise ValueError("Published source input hash differs: "+name)
    science.validate_protocol_freeze(directory)
    return len(manifest["files"])+len(manifest["external_inputs"])


def check_equal(observed,expected,label):
    if not np.array_equal(np.asarray(observed),np.asarray(expected)):
        raise ValueError("Scientific verification differs: "+label)


def verify(directory=HERE,emit=False):
    directory=Path(directory).resolve()
    hash_count=check_manifest(directory)
    partitions,truth=science.parts()
    membership=pd.read_csv(directory/"partition-membership.csv",dtype=str)
    expected=truth[["unique_id","role"]].sort_values("unique_id").reset_index(drop=True)
    check_equal(membership.sort_values("unique_id").reset_index(drop=True),expected,"entity partition membership")
    config=json.loads((directory/"configurations.json").read_text())
    baseline=json.loads((directory/"preflight.json").read_text())["selected_baseline"]
    pairs={r:pd.read_parquet(directory/f"pairs-{baseline}-{r}.parquet") for r in partitions}
    # Rebuild diagnostics from observed fields and the saved conventional scores.
    rebuilt={r:science.feature_frame(p) for r,p in pairs.items()}
    for role,features in rebuilt.items():
        saved=pd.read_parquet(directory/f"features-{role}.parquet")
        if list(features)!=list(saved) or not np.allclose(features,saved,rtol=0,atol=1e-12):
            raise ValueError("Published feature reconstruction differs: "+role)
    columns=config["scaler_columns"]
    scaler=StandardScaler().fit(rebuilt["baseline_fit"][columns])
    if not np.allclose(scaler.mean_,config["scaler_mean"],rtol=0,atol=1e-12):
        raise ValueError("Development-only feature means differ.")
    if not np.allclose(scaler.scale_,config["scaler_scale"],rtol=0,atol=1e-12):
        raise ValueError("Development-only feature scales differ.")
    z={r:scaler.transform(f[columns]) for r,f in rebuilt.items()}
    ledger=pd.DataFrame(json.loads((directory/"label-ledger.json").read_text()))
    if len(ledger)!=60 or set(ledger.arm)!={"random","uncertainty_diversity","clusters","error_risk","shuffled_clusters","score_groups"}:
        raise ValueError("Acquisition ledger does not contain all frozen arms/seeds/budgets.")
    if not (ledger.seed_queries==100).all():
        raise ValueError("Common seed query allocation differs.")
    check_equal(ledger.additional_queries,ledger.additional_budget,"additional pair query allocation")
    for field,value in [("common_baseline_audit_queries",400),("shared_repair_validation_queries",241),
                        ("shared_calibration_queries",255),("test_label_access",0)]:
        if not (ledger[field]==value).all():
            raise ValueError("Query allocation differs: "+field)
    frozen=pd.read_parquet(directory/"sealed-test-predictions.parquet")
    outcomes=pd.read_csv(directory/"results.csv")
    test_truth=truth.loc[truth.role.eq("test")]
    target=science.labels(pairs["test"],test_truth)
    count=len(test_truth)//2
    if count!=781 or len(outcomes)!=372:
        raise ValueError("FEBRL4 evaluation denominator differs.")
    for row in outcomes.itertuples(index=False):
        pred=frozen[row.column].to_numpy().astype(bool)
        tp=int((pred&(target==1)).sum());fp=int((pred&(target==0)).sum())
        check_equal([tp,fp,count-tp,row.fp_cost*fp+count-tp],
                    [row.tp,row.fp,row.fn,row.decision_loss],row.column+" pair counts")
        graph=science.graph_metrics(pairs["test"],pred,test_truth)
        for key,value in graph.items():
            if not np.isclose(value,getattr(row,key),rtol=0,atol=1e-12):
                raise ValueError("FEBRL4 graph metric differs: "+key)
        assigned=science.one_to_one(pairs["test"],pred,frozen[row.score_column].to_numpy())
        assigned_tp=int((assigned&(target==1)).sum());assigned_fp=int((assigned&(target==0)).sum())
        check_equal([assigned_tp,assigned_fp,count-assigned_tp],
                    [row.assigned_tp,row.assigned_fp,row.assigned_fn],row.column+" assignment counts")
    teacher=json.loads((directory/"teacher-results.json").read_text())
    jobs=json.loads((directory/"teacher-jobs.json").read_text())
    if teacher["jobs_sha256"]!=file_hash(directory/"teacher-jobs.json") or len(jobs)!=300:
        raise ValueError("Teacher source/query provenance differs.")
    for job in jobs:
        if set(job["user"])!={"record_a","record_b"}:
            raise ValueError("Unexpected teacher information.")
        for record in job["user"].values():
            if set(record)!=set(science.FIELDS):
                raise ValueError("Teacher inputs include extra metadata.")
    decisive=[r for r in teacher["results"] if r["status"]=="valid" and r["parsed"]["decision"]!="insufficient_evidence"]
    indices=np.array([r["pair_index"] for r in decisive],dtype=int)
    if len(indices)!=289:
        raise ValueError("Retained student training query set differs.")
    teacher_y=np.array([int(r["parsed"]["decision"]=="same_person") for r in decisive])
    gold=science.labels(pairs["discovery"].iloc[indices],truth.loc[truth.role.eq("discovery")])
    pseudo=(rebuilt["discovery"].score.iloc[indices].to_numpy()>=0).astype(int)
    cal_ix,cal_weight=science.score_sample(pairs["calibration"],400,7081)
    cal_y=science.labels(pairs["calibration"].iloc[cal_ix],truth.loc[truth.role.eq("calibration")])
    maximum_probability_difference=0.0
    for name,y in [("teacher_student",teacher_y),("pseudo_student",pseudo),("oracle_student",gold)]:
        student=LogisticRegression(C=1,max_iter=1000,random_state=7101).fit(z["discovery"][indices],y)
        matching=[m for m in config["methods"] if m["arm"]==name]
        if len(matching)!=1:
            raise ValueError("Student specification is missing or repeated.")
        method=matching[0]
        probability=student.predict_proba(z["test"])[:,1]
        difference=float(np.max(np.abs(probability-frozen[method["score_column"]].to_numpy())))
        maximum_probability_difference=max(maximum_probability_difference,difference)
        if difference>1e-10:
            raise ValueError("Independent student refit differs: "+name)
        cal_score=student.predict_proba(z["calibration"][cal_ix])[:,1]
        for cost in science.COSTS:
            cutoff,loss=science.threshold(cal_score,cal_y,cal_weight,cost)
            specified=method["thresholds"][str(cost)]
            if not np.isclose(cutoff,specified["threshold"],rtol=0,atol=1e-10):
                raise ValueError("Student calibration threshold differs: "+name)
            check_equal(probability>=cutoff,frozen[specified["prediction_column"]],name+" calibrated decisions")
    # Secondary corpus identity and split metadata are rebuilt from the pinned raw source.
    raw=pd.read_csv(directory/"febrl3-transfer/source-dataset3.csv",dtype=str,keep_default_na=False)
    raw.columns=raw.columns.str.strip()
    for column in raw:raw[column]=raw[column].str.strip()
    entity=raw.rec_id.str.extract(r"^rec-(\d+)-",expand=False)
    ids=raw.rec_id.map(lambda r:sha256(("febrl3-transfer:"+r).encode()).hexdigest()[:24])
    entity_ids=entity.map(lambda e:sha256(("febrl3-entity:"+e).encode()).hexdigest()[:24])
    bucket=entity.map(lambda e:int(sha256(("clustering-febrl3-transfer-v1:"+e).encode()).hexdigest(),16)%100)
    role=bucket.map(lambda n:"calibration" if 70<=n<85 else "test" if n>=85 else "unused_development")
    secondary_truth=pd.DataFrame({"unique_id":ids,"entity_id":entity_ids,"role":role})
    secondary_test=secondary_truth.loc[secondary_truth.role.eq("test")]
    secondary_pairs=pd.read_parquet(directory/"febrl3-transfer/pairs-test.parquet")
    secondary_frozen=pd.read_parquet(directory/"febrl3-transfer/sealed-predictions.parquet")
    secondary_rows=pd.read_csv(directory/"febrl3-transfer/results.csv")
    secondary_y=science.labels(secondary_pairs,secondary_test)
    true_pairs=sum(n*(n-1)//2 for n in secondary_test.groupby("entity_id").size())
    check_equal([len(secondary_test),secondary_test.entity_id.nunique(),true_pairs,int(secondary_y.sum())],
                [733,296,943,942],"FEBRL3 full population")
    for row in secondary_rows.itertuples(index=False):
        pred=secondary_frozen[f"{row.arm}_c{row.fp_cost}"].to_numpy().astype(bool)
        tp=int((pred&(secondary_y==1)).sum());fp=int((pred&(secondary_y==0)).sum())
        check_equal([tp,fp,true_pairs-tp,row.fp_cost*fp+true_pairs-tp],
                    [row.tp,row.fp,row.fn,row.decision_loss],row.arm+" transfer pair counts")
        graph=science.graph_metrics(secondary_pairs,pred,secondary_test)
        for key,value in graph.items():
            if not np.isclose(value,getattr(row,key),rtol=0,atol=1e-12):
                raise ValueError("FEBRL3 graph metric differs: "+key)
    summary={"status":"verified","published_and_input_hashes":hash_count,"febrl4_outcome_rows":len(outcomes),
        "febrl3_outcome_rows":len(secondary_rows),"student_refit_max_probability_difference":maximum_probability_difference,
        "student_training_pairs":len(indices),"teacher_calls_made":0,"files_written":0,
        "scope":"Offline reconstruction of published exploratory outcomes and identical-pair students; no new scientific run or operational claim."}
    if emit:print(json.dumps(summary,indent=2,sort_keys=True))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=HERE)
    args=parser.parse_args()
    verify(args.data_dir,emit=True)
