"""Prospective, no-new-teacher transfer to multi-record synthetic FEBRL3."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from scipy.special import expit
from splink import DuckDBAPI, Linker

import experiment as study

HERE = Path(__file__).resolve().parent
OUT = HERE / "febrl3-transfer"
SOURCE_URL = "https://raw.githubusercontent.com/moj-analytical-services/splink_datasets/75876d806d9eff72072d21878150ee50a96d5f41/data/febrl/dataset3.csv"


def load_source():
    OUT.mkdir(exist_ok=True)
    source=OUT/"source-dataset3.csv"
    if not source.exists():
        with urllib.request.urlopen(SOURCE_URL,timeout=30) as response:
            data=response.read()
        with source.open("xb") as f:
            f.write(data)
    data=pd.read_csv(source,dtype=str,keep_default_na=False)
    data.columns=data.columns.str.strip()
    for column in data:
        data[column]=data[column].str.strip()
    identity=data.rec_id.str.extract(r"^rec-(\d+)-",expand=False)
    assert identity.notna().all()
    ids=data.rec_id.map(lambda r:sha256(("febrl3-transfer:"+r).encode()).hexdigest()[:24])
    entity_id=identity.map(lambda e:sha256(("febrl3-entity:"+e).encode()).hexdigest()[:24])
    bucket=identity.map(lambda e:int(sha256(("clustering-febrl3-transfer-v1:"+e).encode()).hexdigest(),16)%100)
    role=bucket.map(lambda n:"calibration" if 70<=n<85 else "test" if n>=85 else "unused_development")
    truth=pd.DataFrame({"unique_id":ids,"entity_id":entity_id,"role":role})
    observed=data.drop(columns="rec_id").replace("",None)
    observed.insert(0,"unique_id",ids)
    observed.insert(0,"source","febrl3")
    assert len(data)==5000 and observed.unique_id.is_unique
    assert truth.groupby("entity_id").role.nunique().max()==1
    return observed,truth


def dedupe_predict(observed,model):
    model=json.loads(json.dumps(model))
    model["link_type"]="dedupe_only"
    pairs=set()
    for keys in [["soc_sec_id"],["given_name","surname","date_of_birth"]]:
        clean=observed.dropna(subset=keys)
        matched=clean.merge(clean,on=keys,suffixes=("_l","_r"))
        pairs.update(zip(matched.loc[matched.unique_id_l<matched.unique_id_r,"unique_id_l"],
                         matched.loc[matched.unique_id_l<matched.unique_id_r,"unique_id_r"]))
    population=len(observed)*(len(observed)-1)/2
    model["probability_two_random_records_match"]=len(pairs)/0.8/population
    if not 0<model["probability_two_random_records_match"]<1:
        raise ValueError("Unlabeled strict-rule prior is not usable.")
    db=DuckDBAPI()
    table=db.register(observed,dataset_display_name="febrl3_dedupe")
    linker=Linker(table,model,log_level="WARNING")
    return linker.inference.predict().as_pandas_dataframe().sort_values(["unique_id_l","unique_id_r"]).reset_index(drop=True)


def freeze():
    if (OUT/"prediction-seal.json").exists():
        raise FileExistsError("FEBRL3 predictions already frozen.")
    observed,truth=load_source()
    config=json.load(open(HERE/"configurations.json"))
    baseline_name=json.load(open(HERE/"preflight.json"))["selected_baseline"]
    model=json.load(open(HERE/f"model-{baseline_name}.json"))
    manifest={"created_utc":study.now(),"population":"Pinned public synthetic FEBRL3;5000rows; repeated entities with variable group sizes",
        "source_url":SOURCE_URL,"source_sha256":study.digest(OUT/"source-dataset3.csv"),
        "license":"Original FEBRL ANUOS1.2 and Splink datasets MIT; license notices retained in ../record-linkage-tutorial/data",
        "whole_entity_split":"SHA256(clustering-febrl3-transfer-v1:original_entity_digits)mod100:70-84calibration,85-99test,0-69unused development",
        "counts_by_role":truth.groupby("role").agg(records=("unique_id","size"),entities=("entity_id","nunique")).to_dict(orient="index"),
        "max_entity_records":int(truth.groupby("entity_id").size().max()),
        "observed_entity_size_counts":dict(sorted(Counter(truth.groupby("entity_id").size()).items())),
        "baseline":"Reuse stronger FEBRL4 ten-field Splinkm/u; dedupe_only graph; strict-rule prior estimated from observed data only",
        "students":"Frozen FEBRL4 teacher/pseudo/oracle logistic similarities; no retraining or new teacher inference",
        "features":"Frozen FEBRL4 feature definitions and standardization; rank diagnostics retain deterministic opaque-ID orientation in dedupe, an additional transfer shift",
        "calibration":"At most400 fixed score-stratified candidate queries; same labels for all methods; threshold only; no final-test labels",
        "primary_loss":"2FP+FN across all true within-entity pairs; sensitivityFPcost1/2/5",
        "graph":"Connected components, full implied pair errors, false merge/split entities and B-cubed precision/recall; one-to-one is inappropriate for repeated records",
        "boundary":"Cross-corpus synthetic transfer, not human adjudication or NSO operational performance. New corpus split does not remove possible public model exposure.",
        "test_truth_unblinded":False,"implementation_sha256":study.digest(__file__),"parent_config_sha256":study.digest(HERE/"configurations.json")}
    study.dump(OUT/"protocol.json",manifest,exclusive=True)
    pairs={};features={}
    for role in ["calibration","test"]:
        records=observed.merge(truth.loc[truth.role.eq(role),["unique_id"]],on="unique_id",validate="one_to_one")
        pairs[role]=dedupe_predict(records,model)
        pairs[role].to_parquet(OUT/f"pairs-{role}.parquet",index=False)
        features[role]=study.feature_frame(pairs[role])
    ix,w=study.score_sample(pairs["calibration"],400,7231)
    y=study.labels(pairs["calibration"].iloc[ix],truth.loc[truth.role.eq("calibration")])
    cols=config["scaler_columns"]
    standard={r:(f[cols].to_numpy()-np.array(config["scaler_mean"]))/np.array(config["scaler_scale"]) for r,f in features.items()}
    probabilities={"baseline":{r:expit(f.score.to_numpy()*np.log(2)) for r,f in features.items()}}
    for student in [m for m in config["methods"] if m["branch"]=="student"]:
        probabilities[student["arm"]]={r:expit(standard[r]@np.array(student["coefficients"])+student["intercept"]) for r in standard}
    output=pairs["test"][["unique_id_l","unique_id_r"]].copy()
    thresholds={}
    for name,p in probabilities.items():
        output[name+"_probability"]=p["test"]
        thresholds[name]={}
        for cost in study.COSTS:
            t,loss=study.threshold(p["calibration"][ix],y,w,cost)
            thresholds[name][str(cost)]={"threshold":t,"calibration_loss":loss}
            output[f"{name}_c{cost}"]=p["test"]>=t
    output.to_parquet(OUT/"sealed-predictions.parquet",index=False)
    study.dump(OUT/"thresholds.json",thresholds)
    study.dump(OUT/"calibration-ledger.json",{"queries":len(ix),"teacher_calls":0,"indices":ix.tolist(),"weight":w.tolist(),"labels":y.tolist()})
    study.dump(OUT/"prediction-seal.json",{"created_utc":study.now(),"test_truth_unblinded":False,
        "sha256":{p:study.digest(OUT/p) for p in ["protocol.json","sealed-predictions.parquet","thresholds.json","calibration-ledger.json","source-dataset3.csv"]},
        "implementation_sha256":study.digest(__file__)},exclusive=True)
    print(json.dumps(manifest["counts_by_role"],indent=2))


def evaluate():
    if (OUT/"results.json").exists():
        raise FileExistsError("FEBRL3 results already evaluated.")
    seal=json.load(open(OUT/"prediction-seal.json"))
    for p,h in seal["sha256"].items():
        assert study.digest(OUT/p)==h
    assert study.digest(__file__)==seal["implementation_sha256"]
    observed,truth=load_source()
    truth=truth.loc[truth.role.eq("test")]
    true_pairs=int(sum(n*(n-1)//2 for n in truth.groupby("entity_id").size()))
    nrecords=len(truth)
    pairs=pd.read_parquet(OUT/"pairs-test.parquet")
    frozen=pd.read_parquet(OUT/"sealed-predictions.parquet")
    y=study.labels(pairs,truth)
    rows=[]
    for name in ["baseline","teacher_student","pseudo_student","oracle_student"]:
        for cost in study.COSTS:
            pred=frozen[f"{name}_c{cost}"].to_numpy().astype(bool)
            tp=int((pred&(y==1)).sum());fp=int((pred&(y==0)).sum());fn=true_pairs-tp
            rows.append({"arm":name,"fp_cost":cost,"tp":tp,"fp":fp,"fn":fn,"decision_loss":cost*fp+fn,
                "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/true_pairs,
                **study.graph_metrics(pairs,pred,truth)})
    pd.DataFrame(rows).to_csv(OUT/"results.csv",index=False)
    study.dump(OUT/"results.json",{"created_utc":study.now(),"test_records":nrecords,"test_entities":truth.entity_id.nunique(),
        "test_true_identity_pairs":true_pairs,"possible_pairs":nrecords*(nrecords-1)//2,"candidate_pairs":len(pairs),
        "candidate_true_pairs":int(y.sum()),"candidate_misses":true_pairs-int(y.sum()),"results":rows,
        "boundary":"Frozen similarity transfer between public synthetic corpora; calibration uses separately charged oracle queries; one fixed test, no retuning.",
        "prediction_seal_sha256":study.digest(OUT/"prediction-seal.json")},exclusive=True)
    print(pd.DataFrame(rows).loc[lambda d:d.fp_cost.eq(2)].to_string(index=False))


if __name__=="__main__":
    import sys
    {"freeze":freeze,"evaluate":evaluate}[sys.argv[1]]()
