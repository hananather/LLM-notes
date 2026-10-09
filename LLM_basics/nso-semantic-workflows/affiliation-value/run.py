"""Train and audit conventional affiliation selectors without model API calls.

Commands are separate so test predictions can be sealed before evaluation reads
test reference labels. Original study artifacts are read-only inputs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
from difflib import SequenceMatcher
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
CACHE = ROOT / "cache" / "affiliation-value"
sys.path.insert(0, str(ROOT))
import linkage  # noqa: E402

SEED = 20261009
FEATURES = [
    "char_tfidf", "word_tfidf", "location_score", "contained_alias",
    "fused_score", "candidate_rank_fraction", "fused_gap_from_top",
    "fused_gap_to_next", "char_gap_from_best", "word_gap_from_best",
    "best_name_token_precision", "best_name_token_recall",
    "best_name_token_f1", "best_name_token_jaccard",
    "best_name_character_ratio", "best_name_idf_coverage",
    "best_name_abbreviation_coverage", "any_whole_name_contained",
    "earliest_contained_name_position", "exact_acronym_token",
    "acronym_query_coverage", "city_token_recall", "whole_city_contained",
    "country_token_recall", "whole_country_contained", "country_code_token",
    "query_token_count", "query_character_count", "minimum_name_token_count",
    "maximum_name_token_count", "number_of_names", "number_of_acronyms",
]
EXCLUDED = ["s2aff-0329", "s2aff-0555", "s2aff-1038", "s2aff-1061", "s2aff-1227"]
THRESHOLDS = [round(i / 40, 3) for i in range(1, 41)]
SOURCE_FILES = [
    "cache/linkage/gold_affiliation_annotations.csv",
    "cache/linkage/v1.1-2022-06-16-ror-data.zip",
    "data/linkage/s2aff-inputs.json", "data/linkage/s2aff-candidates.json",
    "data/linkage/s2aff-candidate-registry.json", "data/linkage/s2aff-model-jobs.json",
    "data/linkage/s2aff-gold.json", "data/linkage/model-contract.json",
    "results/linkage/s2aff-baseline-predictions.json",
    "results/linkage/model-test-v1.json", "results/linkage/model-test-v1-score.json",
    "linkage.py",
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value, pretty=True):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       indent=2 if pretty else None,
                       separators=None if pretty else (",", ":")) + "\n").encode()


def freeze_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite different evidence: {path}")
    else:
        with path.open("xb") as f:
            f.write(payload)
    return sha(path)


def freeze(path, value):
    return freeze_bytes(path, json_bytes(value))


def freeze_gz(path, value):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as f:
        f.write(json_bytes(value, pretty=False))
    return freeze_bytes(path, buffer.getvalue())


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    return {
        "version": "affiliation-value-v1", "seed": SEED,
        "status": "retrospective extension on a previously inspected public test split",
        "question": "Does supervised conventional matching close the saved semantic-selector advantage on exactly the same candidate sets?",
        "official_s2aff_replication": False,
        "relation": "complete historical annotated ROR organization set for an affiliation string",
        "original_split_rows": {"train": 1132, "val": 588, "test": 644},
        "training_exclusion_rule": "remove training rows whose linkage.norm(text) appears in validation or test observed strings; no reference labels used",
        "excluded_training_case_ids": EXCLUDED, "training_rows_after_exclusion": 1127,
        "candidate_count": 25, "registry_organizations": 102742,
        "retrieval": "existing linkage.build_ror_candidates over entire pinned registry; no gold insertion; use original saved lists for the existing 744 rows",
        "candidate_reproduction_check": "regenerate all 588 validation lists and require exact agreement for the 100 already saved; test lists are reused, not regenerated",
        "features": FEATURES,
        "feature_information": "observed query and candidate name/alias/acronym/location fields, existing retrieval scores and margins; token IDF from full registry only",
        "prohibited_features": ["ROR identifier or organization-label encoding", "test labels", "LLM answers or embeddings", "external descriptions"],
        "models": {
            "logistic": {"standardize": True, "C": 1.0, "solver": "lbfgs", "max_iter": 2000, "class_weight": "balanced", "random_state": SEED},
            "hist_gradient_boosting": {"max_depth": 3, "max_iter": 200, "learning_rate": 0.05,
                                       "min_samples_leaf": 20, "l2_regularization": 1.0,
                                       "early_stopping": False, "class_weight": "balanced", "random_state": SEED},
        },
        "training_pairs": "all 25 retrieved candidates per included training query; membership in training reference set is target; missing reference candidates are not inserted",
        "thresholds": THRESHOLDS,
        "selection": "validation exact sets, then fewer false edges, then fewer missed edges, then higher threshold, then logistic before tree; both arms and all thresholds retained",
        "set_policy": "select every candidate with score >= threshold; zero and multiple targets allowed; no maximum-cardinality rule",
        "primary_policy": "fully automatic set decisions; learned scores are not asserted to be calibrated probabilities",
        "test_use": "freeze protocol, fit, validation selection and all test predictions; seal their hashes before evaluation loads test gold; no post-score tuning",
        "evaluation": ["all 644", "single target", "empty ROR set", "multiple targets", "candidate-complete positive", "candidate-missing positive"],
        "comparison_views": {
            "valid_set": "valid LLM selections count even if review flagged; invalid outputs never count correct, including empty-reference cases",
            "automatic_resolved": "LLM review and invalid outputs unresolved; no accepted edges from them; all 644 remain in denominator",
            "automatic_conditional": "accuracy among each method's automatic resolved cases; display coverage and denominator; not the same population across methods",
        },
        "prior_exposure": "original test results and examples have been inspected; this is not a fresh blind holdout or unseen-organization test",
        "source_licenses": {"S2AFF": "Apache-2.0; existing data/linkage/S2AFF-LICENSE.txt",
                            "ROR": "CC0; pinned Zenodo ROR release"},
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "paid_calls": 0,
    }


def freeze_protocol():
    path = HERE / "protocol.json"
    freeze(path, protocol())
    return {"protocol_sha256": sha(path), "frozen_before_fitting": True}


def verify_protocol():
    frozen = read(HERE / "protocol.json")
    if frozen != protocol():
        raise ValueError("Protocol, source files or declared settings changed")
    return frozen


def input_rows():
    """Read observed test strings for overlap checks, never parse test labels."""
    rows = []
    with (ROOT / "cache/linkage/gold_affiliation_annotations.csv").open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            item = {"case_id": f"s2aff-{i:04d}", "source_row": i,
                    "split": row["split"], "text": row["original_affiliation"]}
            if item["split"] != "test":
                item["gold_ids"] = sorted(x for x in ast.literal_eval(row["labels"]) if x.startswith("https://ror.org/"))
            rows.append(item)
    assert dict(Counter(r["split"] for r in rows)) == {"train": 1132, "val": 588, "test": 644}
    other = {linkage.norm(r["text"]) for r in rows if r["split"] != "train"}
    excluded = sorted(r["case_id"] for r in rows if r["split"] == "train" and linkage.norm(r["text"]) in other)
    assert excluded == EXCLUDED
    return [r for r in rows if r["case_id"] not in set(excluded)]


def token_stats(query, name, idf):
    qt, nt = set(query.split()), set(name.split())
    common = qt & nt
    precision = len(common) / len(qt) if qt else 0.0
    recall = len(common) / len(nt) if nt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = len(common) / len(qt | nt) if qt | nt else 0.0
    denom = sum(idf.get(t, 1.0) for t in nt)
    coverage = sum(idf.get(t, 1.0) for t in common) / denom if denom else 0.0
    abbreviation = sum(any(t == q or (min(len(t), len(q)) >= 3 and (t.startswith(q) or q.startswith(t)))
                           for q in qt) for t in nt) / len(nt) if nt else 0.0
    return precision, recall, f1, jaccard, coverage, abbreviation


def contained(query, name):
    return bool(name) and " " + name + " " in " " + query + " "


def registry_features(organizations):
    df = Counter()
    prepared = {}
    for org in organizations:
        names = sorted({linkage.norm(s) for s in org["names"] if linkage.norm(s)})
        df.update({t for s in names for t in s.split()})
        prepared[org["id"]] = {
            "names": names, "acronyms": sorted({linkage.norm(s) for s in org["acronyms"] if linkage.norm(s)}),
            "cities": sorted({linkage.norm(s) for s in org["cities"] if linkage.norm(s)}),
            "country": linkage.norm(org["country"]), "country_code": linkage.norm(org["country_code"]),
        }
    idf = {t: math.log((len(organizations) + 1) / (count + 1)) + 1 for t, count in df.items()}
    return prepared, idf


def pair_features(text, candidates, metadata, idf):
    query = linkage.norm(text)
    qt = set(query.split())
    best_char = max(c["char_score"] for c in candidates)
    best_word = max(c["word_score"] for c in candidates)
    top_score = candidates[0]["score"]
    out = []
    for rank, c in enumerate(candidates):
        m = metadata[c["id"]]
        stats = [token_stats(query, n, idf) for n in m["names"]]
        maxima = [max((s[i] for s in stats), default=0.0) for i in range(6)]
        hits = [n for n in m["names"] if contained(query, n)]
        positions = [query.find(n) / max(1, len(query)) for n in hits]
        city_recall = max((token_stats(query, n, idf)[1] for n in m["cities"]), default=0.0)
        acronym_coverage = max((len(qt & set(n.split())) / max(1, len(qt)) for n in m["acronyms"]), default=0.0)
        name_counts = [len(n.split()) for n in m["names"]]
        features = [
            c["char_score"], c["word_score"], c["location_score"], float(c["contained_alias"]),
            c["score"], rank / max(1, len(candidates) - 1), top_score - c["score"],
            c["score"] - candidates[rank + 1]["score"] if rank + 1 < len(candidates) else 0.0,
            best_char - c["char_score"], best_word - c["word_score"],
            *maxima[:4], max((SequenceMatcher(None, query, n, autojunk=False).ratio() for n in m["names"]), default=0.0),
            maxima[4], maxima[5], float(bool(hits)), min(positions, default=1.0),
            float(any(contained(query, n) for n in m["acronyms"])), acronym_coverage,
            city_recall, float(any(contained(query, n) for n in m["cities"])),
            token_stats(query, m["country"], idf)[1], float(contained(query, m["country"])),
            float(bool(m["country_code"]) and m["country_code"] in qt),
            len(query.split()), len(query), min(name_counts, default=0), max(name_counts, default=0),
            len(m["names"]), len(m["acronyms"]),
        ]
        assert len(features) == len(FEATURES)
        assert all(math.isfinite(float(x)) for x in features)
        out.append([float(x) for x in features])
    return out


def prepare():
    verify_protocol()
    target = RESULTS / "preparation.json"
    if target.exists():
        result = read(target)
        for name, expected in result["artifact_sha256"].items():
            assert sha(HERE / name) == expected
        return result
    start = time.perf_counter()
    rows = input_rows()
    existing = {r["case_id"]: r for r in read(ROOT / "data/linkage/s2aff-candidates.json")}
    development = [{k: r[k] for k in ["case_id", "source_row", "split", "text"]}
                   for r in rows if r["split"] != "test"]
    generated, _, index_info = linkage.build_ror_candidates(development)
    generated = {r["case_id"]: r for r in generated}
    overlap = sorted(set(existing) & set(generated))
    assert len(overlap) == 100
    for key in overlap:
        if generated[key] != existing[key]:
            raise ValueError(f"Original validation candidates do not reproduce: {key}")
    candidates = {**generated, **existing}
    assert set(candidates) == {r["case_id"] for r in rows}
    assert all(len(r["candidates"]) == 25 for r in candidates.values())
    freeze_gz(RESULTS / "candidates.json.gz", [candidates[r["case_id"]] for r in rows])
    organizations = linkage.registry()
    metadata, idf = registry_features(organizations)
    feature_rows = []
    for i, row in enumerate(rows):
        cs = candidates[row["case_id"]]["candidates"]
        feature_rows.append({"case_id": row["case_id"], "split": row["split"],
                             "candidate_ids": [c["id"] for c in cs],
                             "features": pair_features(row["text"], cs, metadata, idf)})
        if (i + 1) % 200 == 0:
            print(f"Features {i + 1}/{len(rows)}", flush=True)
    freeze_gz(RESULTS / "pair-features.json.gz", {"feature_names": FEATURES, "rows": feature_rows})
    split_metadata = {}
    for split in ["train", "val"]:
        subset = [r for r in rows if r["split"] == split]
        gold = {r["case_id"]: r["gold_ids"] for r in subset}
        split_metadata[split] = {
            "query_rows": len(subset), "pair_rows": len(subset) * 25,
            "label_cardinalities": {str(k): v for k, v in sorted(Counter(len(r["gold_ids"]) for r in subset).items())},
            "candidate_coverage": linkage.candidate_metrics([candidates[r["case_id"]] for r in subset], gold),
        }
    result = {
        "protocol_sha256": sha(HERE / "protocol.json"), "implementation_sha256": sha(__file__),
        "created_at": utc_now(), "rows_by_split": dict(Counter(r["split"] for r in rows)),
        "excluded_training_case_ids": EXCLUDED, "split_metadata": split_metadata,
        "test_labels_parsed": False, "full_registry_index": index_info,
        "existing_validation_lists_reproduced": len(overlap), "original_test_lists_reused": 644,
        "feature_names": FEATURES, "elapsed_seconds": time.perf_counter() - start,
        "artifact_sha256": {f"results/{name}": sha(RESULTS / name) for name in ["candidates.json.gz", "pair-features.json.gz"]},
        "source_versions": {"python": platform.python_version(), "platform": platform.platform(), "sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    freeze(target, result)
    return result


def predictions_from_scores(rows, scores, threshold):
    result = {}
    offset = 0
    for row in rows:
        count = len(row["candidate_ids"])
        result[row["case_id"]] = sorted(cid for cid, p in zip(row["candidate_ids"], scores[offset:offset + count]) if p >= threshold)
        offset += count
    assert offset == len(scores)
    return result


def selection_key(row):
    return (row["exact_sets"], -row["fp"], -row["fn"], row["threshold"], row["model"] == "logistic")


def fit():
    verify_protocol()
    prep = prepare()
    seal_path = RESULTS / "prediction-seal.json"
    if seal_path.exists():
        return verify_seal()
    start = time.perf_counter()
    rows = input_rows()
    gold = {r["case_id"]: r["gold_ids"] for r in rows if r["split"] != "test"}
    data = read_gz(RESULTS / "pair-features.json.gz")
    assert data["feature_names"] == FEATURES
    parts = {s: [r for r in data["rows"] if r["split"] == s] for s in ["train", "val", "test"]}
    matrices = {s: np.array([pair for r in parts[s] for pair in r["features"]], dtype=np.float64) for s in parts}
    y = np.array([int(cid in set(gold[r["case_id"]])) for r in parts["train"] for cid in r["candidate_ids"]])
    val_gold = {r["case_id"]: gold[r["case_id"]] for r in parts["val"]}
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight="balanced", random_state=SEED)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, class_weight="balanced", random_state=SEED),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    sweep, selected_by_model, prediction_arms, fit_metadata = [], {}, {}, {}
    for name, model in models.items():
        tick = time.perf_counter()
        model.fit(matrices["train"], y)
        val_scores = model.predict_proba(matrices["val"])[:, 1]
        test_scores = model.predict_proba(matrices["test"])[:, 1]
        assert np.isfinite(val_scores).all() and np.isfinite(test_scores).all()
        model_sweep = []
        for threshold in THRESHOLDS:
            pred = predictions_from_scores(parts["val"], val_scores, threshold)
            model_sweep.append({"model": name, "threshold": threshold, **linkage.set_metrics(pred, val_gold)})
        sweep.extend(model_sweep)
        selected = max(model_sweep, key=selection_key)
        selected_by_model[name] = selected
        prediction_arms[name] = predictions_from_scores(parts["test"], test_scores, selected["threshold"])
        joblib.dump(model, CACHE / f"{name}.joblib")
        freeze_gz(RESULTS / f"{name}-scores.json.gz", {
            "val": [{"case_id": r["case_id"], "candidate_ids": r["candidate_ids"], "scores": [float(p) for p in val_scores[i * 25:(i + 1) * 25]]} for i, r in enumerate(parts["val"])],
            "test": [{"case_id": r["case_id"], "candidate_ids": r["candidate_ids"], "scores": [float(p) for p in test_scores[i * 25:(i + 1) * 25]]} for i, r in enumerate(parts["test"])],
        })
        fit_metadata[name] = {"elapsed_seconds": time.perf_counter() - tick,
                              "fitted_model_sha256": sha(CACHE / f"{name}.joblib"),
                              "iterations": int(model[-1].n_iter_[0]) if name == "logistic" else int(model.n_iter_)}
        print(f"{name}: validation {selected['exact_sets']}/588, threshold {selected['threshold']}", flush=True)
    primary = max(selected_by_model.values(), key=selection_key)
    selection = {"protocol_sha256": sha(HERE / "protocol.json"), "primary_model": primary["model"],
                 "primary_selection": primary, "selected_by_model": selected_by_model, "complete_sweep": sweep,
                 "fit_metadata": fit_metadata, "training_pairs": int(len(y)), "training_positive_pairs": int(y.sum()),
                 "training_negative_pairs": int((1 - y).sum()), "test_labels_parsed": False,
                 "elapsed_seconds": time.perf_counter() - start}
    freeze(RESULTS / "validation-selection.json", selection)
    freeze(RESULTS / "test-predictions.json", {"primary_model": primary["model"], "arms": prediction_arms,
                                              "policy": "all predictions are valid, automatic sets, including empty sets"})
    names = ["protocol.json", "run.py", "results/preparation.json", "results/candidates.json.gz",
             "results/pair-features.json.gz", "results/logistic-scores.json.gz", "results/hist_gradient_boosting-scores.json.gz",
             "results/validation-selection.json", "results/test-predictions.json"]
    seal = {"sealed_at": utc_now(), "test_labels_parsed_during_fit": False,
            "test_prediction_rows_per_arm": {k: len(v) for k, v in prediction_arms.items()},
            "artifact_sha256": {name: sha(HERE / name) for name in names},
            "source_sha256": read(HERE / "protocol.json")["source_sha256"],
            "paid_calls": 0, "evaluation_must_verify_this_seal": True}
    freeze(seal_path, seal)
    return seal


def verify_seal():
    verify_protocol()
    seal = read(RESULTS / "prediction-seal.json")
    for name, expected in seal["artifact_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError(f"Sealed prediction input changed: {name}")
    return seal


def saved_llm_states():
    jobs = {j["case_id"]: j for j in linkage.model_jobs("test")}
    raw = {r["case_id"]: r for r in read(ROOT / "results/linkage/model-test-v1.json")["rows"]}
    states = {}
    for case_id, job in jobs.items():
        try:
            answer = linkage.validate_model_output(job, raw[case_id]["answer"])
            states[case_id] = {"targets": sorted(answer["target_ids"]),
                               "status": "review" if answer["review_required"] else "automatic"}
        except (KeyError, ValueError, TypeError):
            states[case_id] = {"targets": [], "status": "invalid"}
    return states


def valid(states, case_id, view):
    status = states[case_id]["status"]
    return status != "invalid" if view == "valid_set" else status == "automatic"


def metric_view(states, gold, view):
    correct = tp = fp = fn = nil_correct = false_nil = false_assignment_nil = 0
    automatic = review = invalid = 0
    for key, ids in gold.items():
        state = states[key]
        accepted = valid(states, key, view)
        pred = set(state["targets"]) if accepted else set()
        truth = set(ids)
        correct += accepted and pred == truth
        tp += len(pred & truth); fp += len(pred - truth); fn += len(truth - pred)
        nil_correct += accepted and not truth and not pred
        false_nil += accepted and bool(truth) and not pred
        false_assignment_nil += not truth and bool(pred)
        automatic += state["status"] == "automatic"
        review += state["status"] == "review"
        invalid += state["status"] == "invalid"
    n = len(gold)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"rows": n, "exact_sets": int(correct), "exact_set_accuracy": correct / n if n else None,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "micro_precision": precision, "micro_recall": recall,
            "nil_rows": sum(not ids for ids in gold.values()), "nil_correct": int(nil_correct),
            "false_nil_rows": int(false_nil), "false_assignment_on_nil_rows": int(false_assignment_nil),
            "automatic_rows": automatic, "review_rows": review, "invalid_rows": invalid,
            "automatic_coverage": automatic / n if n else None,
            "unresolved_rows_in_this_view": invalid if view == "valid_set" else invalid + review}


def paired(new, reference, gold, new_view, reference_view):
    corrections, regressions, both_correct, both_wrong = [], [], [], []
    for key, truth in gold.items():
        a = valid(new, key, new_view) and set(new[key]["targets"]) == set(truth)
        b = valid(reference, key, reference_view) and set(reference[key]["targets"]) == set(truth)
        (both_correct if a and b else corrections if a else regressions if b else both_wrong).append(key)
    return {"corrections": len(corrections), "regressions": len(regressions), "net_correct_change": len(corrections) - len(regressions),
            "both_correct": len(both_correct), "both_wrong_or_unresolved": len(both_wrong),
            "corrected_case_ids": corrections, "regressed_case_ids": regressions,
            "new_view": new_view, "reference_view": reference_view}


def score():
    seal = verify_seal()
    if (RESULTS / "evaluation.json").exists():
        existing = read(RESULTS / "evaluation.json")
        assert existing["prediction_seal_sha256"] == sha(RESULTS / "prediction-seal.json")
        return existing
    # This is the first evaluation-stage reference-label read.
    gold_all = {r["case_id"]: r["gold_ids"] for r in read(ROOT / "data/linkage/s2aff-gold.json")}
    test_rows = [r for r in read(ROOT / "data/linkage/s2aff-inputs.json") if r["split"] == "test"]
    gold = {r["case_id"]: gold_all[r["case_id"]] for r in test_rows}
    assert len(gold) == 644
    saved = read(RESULTS / "test-predictions.json")
    models = {name: {key: {"targets": targets, "status": "automatic"} for key, targets in predictions.items()}
              for name, predictions in saved["arms"].items()}
    lexical = read(ROOT / "results/linkage/s2aff-baseline-predictions.json")["fused_selective_set"]
    models["lexical"] = {key: {"targets": lexical[key], "status": "automatic"} for key in gold}
    models["saved_llm"] = saved_llm_states()
    assert all(set(states) == set(gold) for states in models.values())
    candidates = {r["case_id"]: {c["id"] for c in r["candidates"]} for r in read(ROOT / "data/linkage/s2aff-candidates.json")}
    groups = {"all": list(gold), "single": [k for k, v in gold.items() if len(v) == 1],
              "nil": [k for k, v in gold.items() if not v], "multi": [k for k, v in gold.items() if len(v) > 1],
              "candidate_complete_positive": [k for k, v in gold.items() if v and set(v) <= candidates[k]],
              "candidate_missing_positive": [k for k, v in gold.items() if v and not set(v) <= candidates[k]]}
    scores, pairs = {}, {}
    for name, states in models.items():
        scores[name] = {}
        for group, ids in groups.items():
            subset = {k: gold[k] for k in ids}
            automatic_gold = {k: gold[k] for k in ids if states[k]["status"] == "automatic"}
            scores[name][group] = {"valid_set": metric_view(states, subset, "valid_set"),
                                   "automatic_resolved": metric_view(states, subset, "automatic_resolved"),
                                   "automatic_conditional": metric_view(states, automatic_gold, "automatic_resolved")}
        if name in saved["arms"]:
            pairs[name] = {}
            for group, ids in groups.items():
                subset = {k: gold[k] for k in ids}
                pairs[name][group] = {
                    "versus_lexical": paired(states, models["lexical"], subset, "valid_set", "valid_set"),
                    "versus_llm_valid_set": paired(states, models["saved_llm"], subset, "valid_set", "valid_set"),
                    "versus_llm_automatic_resolved": paired(states, models["saved_llm"], subset, "automatic_resolved", "automatic_resolved"),
                }
    assert scores["lexical"]["all"]["valid_set"]["exact_sets"] == 456
    llm = scores["saved_llm"]["all"]
    assert (llm["valid_set"]["exact_sets"], llm["valid_set"]["fp"], llm["valid_set"]["fn"]) == (593, 30, 27)
    assert (llm["automatic_conditional"]["rows"], llm["automatic_conditional"]["exact_sets"]) == (576, 546)
    assert (llm["valid_set"]["review_rows"], llm["valid_set"]["invalid_rows"]) == (65, 3)
    case_outcomes = []
    for row in test_rows:
        key = row["case_id"]
        case_outcomes.append({"case_id": key, "source_row": row["source_row"], "text": row["text"],
                              "reference_targets": gold[key], "candidate_complete": set(gold[key]) <= candidates[key],
                              "methods": {name: {**states[key],
                                  "valid_set_correct": valid(states, key, "valid_set") and set(states[key]["targets"]) == set(gold[key]),
                                  "automatic_correct": valid(states, key, "automatic_resolved") and set(states[key]["targets"]) == set(gold[key])}
                                          for name, states in models.items()}})
    freeze_gz(RESULTS / "case-outcomes.json.gz", case_outcomes)
    primary = saved["primary_model"]
    bycase = {r["case_id"]: r for r in case_outcomes}
    orgs = {r["id"]: r["name"] for r in linkage.registry()}
    examples = {}
    comparison = pairs[primary]["all"]["versus_llm_valid_set"]
    for label, field in [("trained_correct_llm_wrong", "corrected_case_ids"), ("llm_correct_trained_wrong", "regressed_case_ids")]:
        ids = sorted(comparison[field])
        if ids:
            example = bycase[ids[0]]
            mentioned = set(example["reference_targets"]) | {t for m in example["methods"].values() for t in m["targets"]}
            examples[label] = {**example, "target_names": {key: orgs[key] for key in sorted(mentioned)}}
        else:
            examples[label] = None
    freeze(RESULTS / "examples.json", {"selection": "first case ID in each paired-disagreement direction, after scoring", "primary_model": primary, "cases": examples})
    result = {"evaluated_at": utc_now(), "prediction_seal_sha256": sha(RESULTS / "prediction-seal.json"),
              "prediction_sealed_at": seal["sealed_at"], "primary_model": primary,
              "group_sizes": {k: len(v) for k, v in groups.items()}, "scores": scores, "paired_comparisons": pairs,
              "raw_case_outcomes_sha256": sha(RESULTS / "case-outcomes.json.gz"),
              "examples_sha256": sha(RESULTS / "examples.json"),
              "uncertainty": "Descriptive paired counts; repeated organizations and prior benchmark inspection preclude independent-trial or pristine-holdout claims. No test-tuned threshold or population interval is reported.",
              "paid_calls": 0}
    freeze(RESULTS / "evaluation.json", result)
    return result


def verify():
    seal = verify_seal()
    result = score()
    assert sha(RESULTS / "case-outcomes.json.gz") == result["raw_case_outcomes_sha256"]
    assert sha(RESULTS / "examples.json") == result["examples_sha256"]
    for path in RESULTS.glob("*.json"):
        json_bytes(read(path))
    for path in RESULTS.glob("*.json.gz"):
        json_bytes(read_gz(path), pretty=False)
    return {"sealed_artifacts_verified": len(seal["artifact_sha256"]), "strict_json_verified": True,
            "original_sources_unchanged": True, "primary_model": result["primary_model"], "paid_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "prepare", "fit", "score", "verify"])
    args = parser.parse_args()
    result = {"freeze": freeze_protocol, "prepare": prepare, "fit": fit, "score": score, "verify": verify}[args.command]()
    if args.command == "score":
        result = {"primary_model": result["primary_model"], "all_cases": {name: values["all"] for name, values in result["scores"].items()}}
    elif args.command in {"prepare", "fit"}:
        result = {k: result[k] for k in ["rows_by_split", "elapsed_seconds", "sealed_at", "test_prediction_rows_per_arm", "paid_calls"] if k in result}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
