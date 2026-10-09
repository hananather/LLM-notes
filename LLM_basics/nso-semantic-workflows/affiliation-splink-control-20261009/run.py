"""Supervised Splink evidence scoring on unchanged affiliation candidate pools.

The original tree-control feature packet is a read-only input. Native Splink
scores two frozen categorical comparison representations; training class counts
estimate m/u with additive smoothing. Validation selects a representation and
threshold. All test scores and decisions are sealed before test-label parsing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from splink import DuckDBAPI, Linker, SettingsCreator
import splink.comparison_library as cl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / "affiliation-value"
RESULTS = HERE / "results"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("affiliation_features", OLD / "run.py")
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)

MODELS = {
    "compact": ["joint_name", "character", "location"],
    "expanded": ["joint_name", "character", "location", "word", "rarity", "position"],
}
LEVELS = {"joint_name": 6, "character": 6, "location": 3,
          "word": 6, "rarity": 5, "position": 5}
THRESHOLDS = [round(i / 40, 3) for i in range(1, 41)]
INPUTS = [
    OLD / "run.py", OLD / "protocol.json", OLD / "results/pair-features.json.gz",
    OLD / "results/candidates.json.gz", OLD / "results/test-predictions.json",
    OLD / "results/prediction-seal.json", OLD / "results/evaluation.json",
    ROOT / "cache/linkage/gold_affiliation_annotations.csv",
    ROOT / "data/linkage/s2aff-inputs.json", ROOT / "data/linkage/s2aff-gold.json",
    ROOT / "results/linkage/model-test-v1.json", ROOT / "linkage.py",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    return str(path.relative_to(ROOT))


def protocol():
    return {
        "study": "retrospective supervised Splink affiliation control",
        "relation": "the complete original historical annotated ROR set",
        "evidence_boundary": "Unchanged inspected S2AFF benchmark, not a new blind holdout or native NSO data evaluation.",
        "original_rows": {"train": 1132, "validation": 588, "test": 644},
        "training_rows_after_observed_duplicate_exclusion": 1127,
        "training_exclusions": features.EXCLUDED,
        "candidate_set": "The same 25 candidates and permitted source/name/alias/acronym/location evidence as the retained trained-tree and saved LLM controls; no gold insertion.",
        "representation": {
            "joint_name": "highest applicable level: exact acronym token=5; whole registry name contained=4; complete token coverage=3; token coverage>=.75 or abbreviation coverage>=.9=2; token or abbreviation coverage>=.5=1; else=0",
            "character": "registry character TF-IDF cut points .1,.2,.35,.5,.7",
            "location": "existing observed city/country signal at 0,.5,1",
            "word": "registry word TF-IDF cut points .05,.1,.2,.35,.5",
            "rarity": "registry-only token-IDF name coverage cut points .25,.5,.75,.95",
            "position": "contained-name earliest source position cut points .05,.25,.5,.75; no containment=0",
        },
        "models": MODELS, "levels": LEVELS,
        "supervision": "Per-level training positive/negative frequencies, additive smoothing alpha=1 for every level; prior=(positive_pairs+1)/(training_pairs+2). Reference absent candidates remain misses, not inserted training examples.",
        "prior_population": "Candidate-conditioned 25-pair pools, not a uniformly sampled full-registry cross product.",
        "splink_execution": "Native Splink 5 custom-comparison inference. Each fixed candidate pair is encoded as a feature row and an opaque same-key anchor row; blocking yields exactly one permitted comparison. No identity labels or ROR identity enter comparison SQL.",
        "conditional_independence": "FS multiplies comparison likelihood ratios; derived lexical signals are dependent. Compact/expanded families are fixed and chosen on validation. Scores are not asserted to be calibrated.",
        "term_frequency": "Token rarity is computed from the full registry as a separate expanded comparison; native exact-string term-frequency weighting is not applied to categorical feature values.",
        "thresholds": THRESHOLDS,
        "selection": "Validation exact sets; ties fewer false edges, then fewer missed edges, then higher threshold, then compact representation.",
        "set_policy": "Every candidate whose score >= selected threshold, permitting empty and multiple sets; all outputs automatic.",
        "test_use": "Write scores, settings, selected thresholds, decisions and hash seal before opening reference labels; no outcome-driven refit.",
        "evaluation": "Retain logistic, trained tree and saved semantic decisions. Separate valid sets including review from automatic resolved and automatic conditional views; all 644 cases remain in end-to-end denominators.",
        "loss_sensitivity": "FP+FN, 2*FP+FN and 5*FP+FN, descriptive counts on the fixed exact-set-selected policy; no official-statistics cost assumption.",
        "bootstrap": "2000 paired cluster resamples; link cases sharing a positive reference ROR or normalized duplicate input. NIL cases group only by duplicate text. Grouping defines dependence, not institution identity.",
        "versions": {p: version(p) for p in ["splink", "duckdb", "numpy", "pandas", "scipy"]},
        "source_sha256": {relative(p): features.sha(p) for p in INPUTS},
        "code_sha256": features.sha(__file__), "paid_calls": 0,
    }


def freeze():
    features.freeze(HERE / "protocol.json", protocol())
    return {"protocol_sha256": features.sha(HERE / "protocol.json"), "paid_calls": 0}


def verify_protocol():
    if features.read(HERE / "protocol.json") != protocol():
        raise ValueError("Frozen specification, source evidence or implementation changed")


def categorical_matrix(matrix, names):
    f = {n: matrix[:, i] for i, n in enumerate(names)}
    recall = f["best_name_token_recall"]
    abbreviation = f["best_name_abbreviation_coverage"]
    name = np.zeros(len(matrix), dtype=np.int64)
    name[(recall >= .5) | (abbreviation >= .5)] = 1
    name[(recall >= .75) | (abbreviation >= .9)] = 2
    name[recall >= 1 - 1e-12] = 3
    name[f["any_whole_name_contained"] >= .5] = 4
    name[f["exact_acronym_token"] >= .5] = 5
    position = np.zeros(len(matrix), dtype=np.int64)
    for i, bound in enumerate([.75, .5, .25, .05], 1):
        position[(f["any_whole_name_contained"] >= .5) &
                 (f["earliest_contained_name_position"] <= bound)] = i
    return {
        "joint_name": name,
        "character": np.digitize(f["char_tfidf"], [.1, .2, .35, .5, .7]),
        "location": np.digitize(f["location_score"], [.25, .75]),
        "word": np.digitize(f["word_tfidf"], [.05, .1, .2, .35, .5]),
        "rarity": np.digitize(f["best_name_idf_coverage"], [.25, .5, .75, .95]),
        "position": position,
    }


def fit_parameters(categories, y, fields):
    pri = (int(y.sum()) + 1) / (len(y) + 2)
    out = {"prior": pri, "training_pairs": len(y), "training_positive_pairs": int(y.sum()), "comparisons": {}}
    for name in fields:
        count_m = np.bincount(categories[name][y == 1], minlength=LEVELS[name])
        count_u = np.bincount(categories[name][y == 0], minlength=LEVELS[name])
        m = (count_m + 1) / (int(y.sum()) + LEVELS[name])
        u = (count_u + 1) / (int((y == 0).sum()) + LEVELS[name])
        out["comparisons"][name] = {"m": m.tolist(), "u": u.tolist(),
                                    "positive_counts": count_m.tolist(), "negative_counts": count_u.tolist()}
    return out


def native_scores(categories, parameters):
    n = len(next(iter(categories.values())))
    left = pd.DataFrame({"unique_id": [f"L{i:07d}" for i in range(n)], "pair_key": np.arange(n)})
    right = pd.DataFrame({"unique_id": [f"R{i:07d}" for i in range(n)], "pair_key": np.arange(n)})
    comparisons = []
    for name, p in parameters["comparisons"].items():
        left[name] = categories[name]
        right[name] = 0
        levels = [{"sql_condition": f"{name}_l IS NULL OR {name}_r IS NULL", "is_null_level": True}]
        for level in range(LEVELS[name] - 1, 0, -1):
            levels.append({"sql_condition": f"{name}_l = {level} AND {name}_r = 0",
                           "label_for_charts": f"category {level}",
                           "m_probability": p["m"][level], "u_probability": p["u"][level]})
        levels.append({"sql_condition": "ELSE", "label_for_charts": "category 0",
                       "m_probability": p["m"][0], "u_probability": p["u"][0]})
        comparisons.append(cl.CustomComparison(levels, output_column_name=name,
                                               comparison_description=f"Observed candidate-pair {name} evidence"))
    api = DuckDBAPI()
    tables = [api.register(left, dataset_display_name="a_candidate_features"),
              api.register(right, dataset_display_name="b_candidate_anchor")]
    settings = SettingsCreator(link_type="link_only", probability_two_random_records_match=parameters["prior"],
        comparisons=comparisons, blocking_rules_to_generate_predictions=["l.pair_key = r.pair_key"],
        retain_intermediate_calculation_columns=True, additional_columns_to_retain=["pair_key"])
    linker = Linker(tables, settings, log_level="WARNING")
    pred = linker.inference.predict().as_pandas_dataframe()
    if len(pred) != n or not pred.pair_key_l.is_unique:
        raise ValueError("Native scoring did not retain precisely the fixed pair universe")
    pred = pred.sort_values("pair_key_l")
    if pred.unique_id_l.str.startswith("R").any():
        raise ValueError("Splink source orientation changed")
    probability = pred.match_probability.to_numpy()
    logodds = np.full(n, np.log(parameters["prior"] / (1 - parameters["prior"])))
    for name, p in parameters["comparisons"].items():
        logodds += np.log(np.asarray(p["m"])[categories[name]] / np.asarray(p["u"])[categories[name]])
    disagreement = float(np.max(np.abs(probability - expit(logodds))))
    if disagreement > 1e-10 or not np.isfinite(probability).all():
        raise ValueError(f"Native/independent likelihood scores disagree: {disagreement}")
    return probability, linker.misc.save_model_to_json(), disagreement


def fit():
    verify_protocol()
    if (RESULTS / "prediction-seal.json").exists():
        return verify_seal()
    features.verify_seal()
    start = time.perf_counter()
    rows = features.input_rows()
    development_gold = {r["case_id"]: r["gold_ids"] for r in rows if r["split"] != "test"}
    packet = features.read_gz(OLD / "results/pair-features.json.gz")
    if packet["feature_names"] != features.FEATURES:
        raise ValueError("Feature packet changed")
    parts = {s: [r for r in packet["rows"] if r["split"] == s] for s in ["train", "val", "test"]}
    categories = {s: categorical_matrix(np.asarray([x for r in parts[s] for x in r["features"]]), packet["feature_names"]) for s in parts}
    y = np.asarray([int(cid in development_gold[r["case_id"]]) for r in parts["train"] for cid in r["candidate_ids"]])
    val_gold = {r["case_id"]: development_gold[r["case_id"]] for r in parts["val"]}
    sweep, selected, predictions, disagreements = [], {}, {}, {}
    for name, fields in MODELS.items():
        params = fit_parameters(categories["train"], y, fields)
        features.freeze(RESULTS / f"{name}-training.json", params)
        saved_scores = {}
        for split in ["val", "test"]:
            probabilities, model, delta = native_scores(categories[split], params)
            saved_scores[split] = probabilities
            disagreements[f"{name}-{split}"] = delta
            features.freeze(RESULTS / f"{name}-{split}-splink-model.json", model)
        features.freeze_gz(RESULTS / f"{name}-scores.json.gz", {s: [
            {"case_id": r["case_id"], "candidate_ids": r["candidate_ids"],
             "scores": saved_scores[s][i * 25:(i + 1) * 25].tolist()}
            for i, r in enumerate(parts[s])] for s in saved_scores})
        model_sweep = []
        for t in THRESHOLDS:
            decision = features.predictions_from_scores(parts["val"], saved_scores["val"], t)
            model_sweep.append({"model": name, "threshold": t, **features.linkage.set_metrics(decision, val_gold)})
        sweep += model_sweep
        selected[name] = max(model_sweep, key=lambda r: (r["exact_sets"], -r["fp"], -r["fn"], r["threshold"]))
        predictions[name] = features.predictions_from_scores(parts["test"], saved_scores["test"], selected[name]["threshold"])
        print(f"{name}: validation {selected[name]['exact_sets']}/588, threshold {selected[name]['threshold']}", flush=True)
    primary = max(selected.values(), key=lambda r: (r["exact_sets"], -r["fp"], -r["fn"], r["threshold"], r["model"] == "compact"))
    features.freeze(RESULTS / "validation-selection.json", {"primary_model": primary["model"], "primary_selection": primary,
        "selected_by_model": selected, "complete_sweep": sweep, "native_max_absolute_difference": disagreements,
        "test_labels_parsed": False, "elapsed_seconds": time.perf_counter() - start})
    features.freeze(RESULTS / "test-predictions.json", {"primary_model": primary["model"], "arms": predictions})
    paths = [HERE / "protocol.json", HERE / "run.py", *sorted(RESULTS.glob("*"))]
    seal = {"sealed_at": now(), "test_labels_parsed_during_fit": False, "prediction_rows_per_arm": {n: len(p) for n, p in predictions.items()},
            "artifact_sha256": {str(p.relative_to(HERE)): features.sha(p) for p in paths}, "source_sha256": protocol()["source_sha256"], "paid_calls": 0}
    features.freeze(RESULTS / "prediction-seal.json", seal)
    return seal


def verify_seal():
    verify_protocol()
    seal = features.read(RESULTS / "prediction-seal.json")
    for name, digest in seal["artifact_sha256"].items():
        if features.sha(HERE / name) != digest:
            raise ValueError(f"Sealed artifact changed: {name}")
    return seal


def grouped_difference(states, reference, gold, texts):
    keys = sorted(gold)
    parent = {k: k for k in keys}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    first = {}
    for key in keys:
        identifiers = [("text", features.linkage.norm(texts[key]))] + [("ror", x) for x in gold[key]]
        for token in identifiers:
            if token in first:
                a, b = find(key), find(first[token])
                parent[max(a, b)] = min(a, b)
            else:
                first[token] = key
    group_keys = sorted({find(k) for k in keys})
    groups = {g: [k for k in keys if find(k) == g] for g in group_keys}
    rng = np.random.default_rng(2026100921)
    rows = {}
    for view in ["valid_set", "automatic_resolved"]:
        delta = {k: int(features.valid(states, k, view) and set(states[k]["targets"]) == set(gold[k])) -
                    int(features.valid(reference, k, view) and set(reference[k]["targets"]) == set(gold[k])) for k in keys}
        num = np.asarray([sum(delta[k] for k in groups[g]) for g in group_keys])
        den = np.asarray([len(groups[g]) for g in group_keys])
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=2000)
        estimates = weights @ num / (weights @ den)
        rows[view] = {"difference_fraction": sum(delta.values()) / len(keys), "percentile_95": np.quantile(estimates, [.025, .975]).tolist()}
    return {"groups": len(groups), "group_size_counts": dict(sorted(Counter(map(len, groups.values())).items())), "views": rows,
            "scope": "Paired descriptive source/reference-group uncertainty; not NSO population inference or a fresh holdout claim"}


def score():
    seal = verify_seal()
    target = RESULTS / "evaluation.json"
    if target.exists():
        return features.read(target)
    gold_all = {r["case_id"]: r["gold_ids"] for r in features.read(ROOT / "data/linkage/s2aff-gold.json")}
    test = [r for r in features.read(ROOT / "data/linkage/s2aff-inputs.json") if r["split"] == "test"]
    gold = {r["case_id"]: gold_all[r["case_id"]] for r in test}
    saved = features.read(RESULTS / "test-predictions.json")
    states = {f"splink_{name}": {k: {"targets": v, "status": "automatic"} for k, v in pred.items()} for name, pred in saved["arms"].items()}
    original = features.read(OLD / "results/test-predictions.json")
    states.update({name: {k: {"targets": v, "status": "automatic"} for k, v in pred.items()} for name, pred in original["arms"].items()})
    states["saved_llm"] = features.saved_llm_states()
    if any(set(s) != set(gold) for s in states.values()):
        raise ValueError("A comparison dropped test cases")
    results = {}
    for name, method in states.items():
        automatic_gold = {k: v for k, v in gold.items() if method[k]["status"] == "automatic"}
        results[name] = {
            "valid_set": features.metric_view(method, gold, "valid_set"),
            "automatic_resolved": features.metric_view(method, gold, "automatic_resolved"),
            "automatic_conditional": features.metric_view(method, automatic_gold, "automatic_resolved"),
        }
        results[name]["loss_sensitivity"] = {view: {str(c): c * results[name][view]["fp"] + results[name][view]["fn"] for c in [1, 2, 5]}
                                              for view in ["valid_set", "automatic_resolved"]}
    primary = f"splink_{saved['primary_model']}"
    paired = {n: {view: features.paired(states[primary], s, gold, view, view) for view in ["valid_set", "automatic_resolved"]}
              for n, s in states.items() if n != primary}
    texts = {r["case_id"]: r["text"] for r in test}
    bootstrap = {n: grouped_difference(states[primary], states[n], gold, texts) for n in ["hist_gradient_boosting", "saved_llm"]}
    features.freeze_gz(RESULTS / "case-outcomes.json.gz", [{"case_id": k, "truth": gold[k], "methods": {n: s[k] for n, s in states.items()}} for k in gold])
    out = {"evaluated_at": now(), "prediction_sealed_at": seal["sealed_at"],
           "prediction_seal_sha256": features.sha(RESULTS / "prediction-seal.json"), "primary_model": primary,
           "rows": len(gold), "scores": results, "paired_comparisons": paired, "group_bootstrap": bootstrap,
           "scope": protocol()["evidence_boundary"], "paid_calls": 0,
           "case_outcomes_sha256": features.sha(RESULTS / "case-outcomes.json.gz")}
    features.freeze(target, out)
    return out


def verify():
    seal = verify_seal()
    result = score()
    if features.sha(RESULTS / "case-outcomes.json.gz") != result["case_outcomes_sha256"]:
        raise ValueError("Case outcomes changed")
    return {"source_files_unchanged": True, "sealed_artifacts": len(seal["artifact_sha256"]),
            "test_rows": result["rows"], "native_splink_verified": True, "paid_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "fit", "score", "verify"])
    args = parser.parse_args()
    out = {"freeze": freeze, "fit": fit, "score": score, "verify": verify}[args.mode]()
    if args.mode == "score":
        out = {"primary_model": out["primary_model"], "scores": out["scores"], "paid_calls": 0}
    elif args.mode == "fit":
        out = {k: out[k] for k in ["sealed_at", "prediction_rows_per_arm", "paid_calls"]}
    print(json.dumps(out, indent=2, allow_nan=False))
