"""Full-roster, label-isolated organization retrieval and bounded Splink 5 EM."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import importlib.metadata
import io
import json
import logging
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from prepare import DATA, HERE, digest, freeze_json, freeze_lines, norm

PROTOCOL = {
    "version": "organizations-offline-v1", "fit_seed": 2026100722,
    "population": "CORDIS structured organization/city/country against all 108476 ROR v1.41 rows, including inactive and withdrawn records",
    "label_use": "None in normalization, retrieval, fitting, threshold or prior choice. Source labels only determined split grouping during separate curation.",
    "fit_population": "All 3329 observed CORDIS records, including evaluation inputs; transductive unsupervised fit",
    "retrieval": {"alias_char_ngrams": [3, 5], "word_ngrams": [1, 2], "max_features_each": 200000,
        "min_df": 1, "per_route_top": 100, "final_candidates": 40,
        "fusion": {"best_alias_character_cosine": .60, "registry_word_cosine": .30, "city": .06, "country": .04, "exact_alias": 1.0},
        "exact_alias": "Complete normalized name/alias/acronym/multilingual-label equality; union all owners before reranking",
        "ties": "Opaque registry record ID ascending", "gold_target_insertion": False},
    "normalization": "Unicode NFKD, remove combining marks, casefold, retain Unicode word characters; preserve non-Latin scripts",
    "lexical_decisions": {"linked": "top fused score >=0.85 and top-minus-second >=0.10", "nil": "top fused score <0.25", "otherwise": "review", "top1": "Always retain separate top-ranked retrieval result"},
    "splink": {"version": "5.0.0", "comparisons": ["Joint exact/alias/fuzzy organization name", "City exact/fuzzy", "Country"],
        "name_jaro_winkler_thresholds": [.95, .85, .70], "city_jaro_winkler_threshold": .90,
        "u_random_pairs": 1000000, "em_blocks": ["exact city and country", "exact primary normalized name"],
        "maximum_pairs_per_em_pass": 100000, "maximum_iterations": 30, "convergence_tolerance": .0001,
        "strict_prior_rule": "One unique complete-registry exact alias candidate with agreeing nonmissing country, one anchor per query",
        "assumed_anchor_recall_grid": [.5, .8, .95, 1.0],
        "main_prior_rule": "Recall 0.8 if implied matches <=3329, otherwise 0.95 then 1.0; choice uses observed anchor count only",
        "sensitivity": "Fit at main feasible prior once; recompute posterior sensitivity with fixed learned likelihood ratios for every feasible prior scenario",
        "capacity": "At most one target per CORDIS query, many queries may map to same registry entity; no one-to-one assignment",
        "sparse_levels": "Record estimated/missing/default/zero probabilities; use explicit effective Splink values, never silently replace a failed fit",
        "decision": "Top posterior >=0.99 and no second candidate >=0.5 -> linked; top<0.1 -> nil; otherwise review",
        "posterior_grid": [.5, .9, .99, .999], "calibration": "Unestablished; fixed rules and descriptive sensitivity only"},
    "native": "Prepared separately; this run does not score native affiliation sets or fit a multi-target policy",
}


def read_lines(path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def inference_inputs():
    queries = read_lines(DATA / "cordis-observed.jsonl")
    if any(set(q) != {"query_id", "ORG", "CITY", "COUNTRY"} for q in queries):
        raise ValueError("Unexpected query fields")
    roster = read_lines(DATA / "ror-v1.41-observed.jsonl.gz")
    assert len(queries) == 3329 and len(roster) == 108476
    for r in roster:
        r["names"] = list(dict.fromkeys(norm(x) for x in [r["name"], *r["aliases"], *r["acronyms"], *[v["label"] for v in r["labels"]]]))
        r["names"] = [x for x in r["names"] if x]
        r["city_names"] = [norm(x) for x in r["cities"] if norm(x)]
    return queries, roster, json.loads((DATA / "partitions.json").read_text())


def top_indices(values, k, ids):
    nonzero = np.flatnonzero(values > 0)
    if len(nonzero) > k:
        cut = np.partition(values[nonzero], -k)[-k]
        nonzero = nonzero[values[nonzero] >= cut]
    return sorted(nonzero, key=lambda i: (-values[i], ids[i]))[:k]


def retrieve(queries, roster, partitions):
    owners = defaultdict(list)
    for i, r in enumerate(roster):
        for name in r["names"]:
            owners[name].append(i)
    aliases = sorted(owners)
    alias_position = {name: i for i, name in enumerate(aliases)}
    config = PROTOCOL["retrieval"]
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False,
        min_df=1, max_features=config["max_features_each"], dtype=np.float32)
    word = TfidfVectorizer(ngram_range=(1, 2), lowercase=False,
        min_df=1, max_features=config["max_features_each"], dtype=np.float32)
    cm = char.fit_transform(aliases)
    wm = word.fit_transform([" ".join(r["names"]) for r in roster])
    ids = [r["record_id"] for r in roster]
    wanted = set(partitions["development"] + partitions["evaluation"])
    jobs = [q for q in queries if q["query_id"] in wanted]
    cq = char.transform([norm(q["ORG"]) for q in jobs])
    wq = word.transform([norm(q["ORG"]) for q in jobs])
    candidate_rows, decisions = [], []
    anchors = 0
    for q in queries:
        exact = [i for i in owners.get(norm(q["ORG"]), []) if q["COUNTRY"].strip().upper() == roster[i]["country_code"]]
        anchors += len(set(exact)) == 1
    for j, q in enumerate(jobs):
        cs = (cq[j] @ cm.T).toarray().ravel()
        ws = (wq[j] @ wm.T).toarray().ravel()
        char_best = defaultdict(float)
        for ai in top_indices(cs, config["per_route_top"], aliases):
            for i in owners[aliases[ai]]:
                char_best[i] = max(char_best[i], float(cs[ai]))
        pool = set(char_best) | set(top_indices(ws, config["per_route_top"], ids)) | set(owners.get(norm(q["ORG"]), []))
        rows = []
        for i in pool:
            r = roster[i]
            # All aliases for each pooled entity contribute to the reported best cosine.
            best_char = max((float(cs[alias_position[name]]) for name in r["names"]), default=0.)
            exact = norm(q["ORG"]) in r["names"]
            city = bool(norm(q["CITY"])) and norm(q["CITY"]) in r["city_names"]
            country = bool(q["COUNTRY"].strip()) and q["COUNTRY"].strip().upper() == r["country_code"]
            score = .60 * best_char + .30 * float(ws[i]) + .06 * city + .04 * country + float(exact)
            rows.append({"query_id": q["query_id"], "record_id": ids[i], "score": score,
                "char_cosine": best_char, "word_cosine": float(ws[i]),
                "exact_alias": exact, "city_agreement": city, "country_agreement": country})
        rows.sort(key=lambda r: (-r["score"], r["record_id"]))
        rows = rows[:config["final_candidates"]]
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        candidate_rows.extend(rows)
        top = rows[0]["score"] if rows else 0.
        gap = top - (rows[1]["score"] if len(rows) > 1 else 0.)
        decision = "linked" if top >= .85 and gap >= .10 else "nil" if top < .25 else "review"
        decisions.append({"query_id": q["query_id"], "decision": decision,
            "target_ids": [rows[0]["record_id"]] if decision == "linked" else [],
            "top1_target_ids": [rows[0]["record_id"]] if rows else [], "top_score": top, "gap": gap})
        if (j + 1) % 200 == 0:
            print(json.dumps({"retrieved_queries": j + 1, "total": len(jobs)}), flush=True)
    return candidate_rows, decisions, anchors, {"alias_count": len(aliases), "char_features": len(char.vocabulary_), "word_features": len(word.vocabulary_)}


def comparisons():
    def level(sql, label, null=False):
        return {"sql_condition": sql, "label_for_charts": label, **({"is_null_level": True} if null else {})}
    return [
        {"output_column_name": "organization_name", "comparison_levels": [
            level("name_l IS NULL OR name_r IS NULL", "Missing name", True),
            level("name_l = name_r OR list_has_any(names_l, names_r)", "Exact primary or alias"),
            *[level(f"org_name_similarity(names_l, names_r) >= {v}", f"Best alias Jaro-Winkler >= {v}") for v in [.95, .85, .70]],
            level("ELSE", "Other name")]},
        {"output_column_name": "city_evidence", "comparison_levels": [
            level("city_l IS NULL OR city_r IS NULL", "Missing city", True),
            level("city_l = city_r OR list_has_any(city_names_l, city_names_r)", "Exact city"),
            level("org_name_similarity(city_names_l, city_names_r) >= 0.9", "City Jaro-Winkler >= 0.9"), level("ELSE", "Other city")]},
        {"output_column_name": "country_evidence", "comparison_levels": [
            level("country_l IS NULL OR country_r IS NULL", "Missing country", True),
            level("country_l = country_r", "Country agreement"), level("ELSE", "Country disagreement")]},
    ]


def splink_frames(queries, roster):
    q = pd.DataFrame([{"unique_id": r["query_id"], "source": "a_query", "name": norm(r["ORG"]) or None,
        "names": [norm(r["ORG"])] if norm(r["ORG"]) else [], "city": norm(r["CITY"]) or None,
        "city_names": [norm(r["CITY"])] if norm(r["CITY"]) else [], "country": r["COUNTRY"].strip().upper() or None} for r in queries])
    r = pd.DataFrame([{"unique_id": r["record_id"], "source": "b_registry", "name": norm(r["name"]) or None,
        "names": r["names"], "city": (r["city_names"] or [None])[0], "city_names": r["city_names"],
        "country": r["country_code"] or None} for r in roster])
    return q, r


def fit_and_score(queries, roster, candidates, anchors):
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    q, r = splink_frames(queries, roster)
    scenarios = [{"assumed_anchor_recall": recall, "implied_matches": anchors / recall,
        "feasible": anchors > 0 and anchors / recall <= len(q),
        "prior": anchors / recall / (len(q) * len(r))} for recall in [.5, .8, .95, 1.0]]
    main = next((s for recall in [.8, .95, 1.] for s in scenarios if s["assumed_anchor_recall"] == recall and s["feasible"]), None)
    audit = {"strict_unique_anchor_count": anchors, "prior_scenarios": scenarios,
        "main_prior": main, "fit_status": "not_started", "model_calibration": "unestablished"}
    if main is None:
        return [], {**audit, "fit_status": "stopped_no_feasible_prior"}
    db = DuckDBAPI()
    db._con.execute("CREATE MACRO org_name_similarity(a,b) AS list_max(list_transform(a,x -> list_max(list_transform(b,y -> jaro_winkler_similarity(x,y)))))")
    tables = [db.register(q, dataset_display_name="a_query"), db.register(r, dataset_display_name="b_registry")]
    settings = SettingsCreator(link_type="link_only", unique_id_column_name="unique_id", source_dataset_column_name="source",
        comparisons=comparisons(), probability_two_random_records_match=main["prior"],
        blocking_rules_to_generate_predictions=[block_on("name")], max_iterations=30, em_convergence=.0001,
        retain_intermediate_calculation_columns=True)
    linker = Linker(tables, settings, log_level="WARNING")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    log = logging.getLogger("splink")
    log.addHandler(handler)
    started = time.perf_counter()
    try:
        linker.training.estimate_u_using_random_sampling(max_pairs=1000000, seed=2026100722, min_count_per_level=None, num_chunks=1)
        sessions = []
        for block in [block_on("city", "country"), block_on("name")]:
            session = linker.training.estimate_parameters_using_expectation_maximisation(block,
                estimate_without_term_frequencies=True, fix_u_probabilities=True,
                populate_probability_two_random_records_match_from_trained_values=False,
                max_pairs=100000, record_sample_proportion=1.)
            history = []
            previous = None
            for i, core in enumerate(session._core_model_settings_history):
                values = {"prior": core.probability_two_random_records_match}
                for c in core.comparisons:
                    for lev in c.comparison_levels:
                        if not lev.is_null_level:
                            values[f"{c.output_column_name}:{lev.comparison_vector_value}:m"] = lev.m_probability
                            values[f"{c.output_column_name}:{lev.comparison_vector_value}:u"] = lev.u_probability
                delta = None if previous is None else max(abs(values[k] - previous[k]) for k in values)
                history.append({"iteration": i, "parameters": values, "maximum_absolute_change": delta})
                previous = values
            sessions.append({"block": str(block), "iterations": len(history) - 1,
                "excluded_comparisons": [c.output_column_name for c in session._comparisons_that_cannot_be_estimated],
                "convergence_criterion_met": history[-1]["maximum_absolute_change"] is not None and history[-1]["maximum_absolute_change"] < .0001,
                "history": history})
        model = linker.misc.save_model_to_json()
        levels = []
        for comp in linker._settings_obj.core_model_settings.comparisons:
            for lev in comp.comparison_levels:
                row = {"comparison": comp.output_column_name, "sql_condition": lev.sql_condition,
                    "is_null_level": lev.is_null_level, "gamma": lev.comparison_vector_value,
                    }
                if not lev.is_null_level:
                    row.update(m=lev.m_probability, u=lev.u_probability,
                        m_raw=lev._m_probability, u_raw=lev._u_probability,
                        m_estimated=lev._has_estimated_m_values, u_estimated=lev._has_estimated_u_values)
                levels.append(row)
        audit.update(fit_status="complete", model=model, levels=levels, sessions=sessions)
        # Apply the exact saved comparison SQL and effective m/u on the frozen candidate union.
        db._con.register("org_candidates", pd.DataFrame(candidates)[["query_id", "record_id"]])
        db._con.register("org_queries", q)
        db._con.register("org_registry", r)
        columns = ["name", "names", "city", "city_names", "country"]
        base = "SELECT c.query_id,c.record_id," + ",".join(f"l.{x} AS {x}_l,r.{x} AS {x}_r" for x in columns) + " FROM org_candidates c JOIN org_queries l ON c.query_id=l.unique_id JOIN org_registry r ON c.record_id=r.unique_id"
        expressions = []
        for comp in linker._settings_obj.core_model_settings.comparisons:
            parts = []
            for lev in comp.comparison_levels:
                if lev.is_null_level:
                    w = 0.
                else:
                    if not lev.m_probability or not lev.u_probability:
                        raise ValueError("Zero effective parameter prevents finite scoring; fit retained without predictions")
                    w = math.log2(lev.m_probability / lev.u_probability)
                literal = f"CAST('{w}' AS DOUBLE)"
                parts.append(f"ELSE {literal}" if lev.sql_condition == "ELSE" else f"WHEN {lev.sql_condition} THEN {literal}")
            expressions.append("CASE " + " ".join(parts) + " END")
        result = db._con.execute("SELECT query_id,record_id," + "+".join("(" + x + ")" for x in expressions) + " AS log2_likelihood_ratio FROM (" + base + ") p").df()
        output = []
        for row in result.to_dict(orient="records"):
            posteriors = []
            for s in scenarios:
                if not s["feasible"]:
                    continue
                logodds = row["log2_likelihood_ratio"] + math.log2(s["prior"] / (1. - s["prior"]))
                prob = 1. / (1. + 2. ** (-max(-1000., min(1000., logodds))))
                posteriors.append({"assumed_anchor_recall": s["assumed_anchor_recall"], "probability": prob})
            row["prior_sensitivity"] = posteriors
            row["match_probability"] = next(v["probability"] for v in posteriors if v["assumed_anchor_recall"] == main["assumed_anchor_recall"])
            output.append(row)
        return output, audit
    except Exception as exc:
        audit.update(fit_status="failed", error_type=type(exc).__name__, error=str(exc))
        return [], audit
    finally:
        audit["fit_seconds"] = time.perf_counter() - started
        audit["warnings"] = stream.getvalue().splitlines()
        log.removeHandler(handler)


def run():
    started = time.perf_counter()
    queries, roster, partitions = inference_inputs()
    candidates, lexical, anchors, index = retrieve(queries, roster, partitions)
    fs, audit = fit_and_score(queries, roster, candidates, anchors)
    results = HERE / "results"
    outputs = [freeze_lines(results / "candidates.jsonl.gz", candidates),
        freeze_lines(results / "lexical-predictions.jsonl", lexical), freeze_lines(results / "splink-pair-predictions.jsonl.gz", fs),
        freeze_json(results / "splink-fit-audit.json", audit)]
    freeze_json(HERE / "protocol.json", PROTOCOL)
    manifest = {"schema": 1, "protocol_sha256": digest((HERE / "protocol.json").read_bytes()),
        "code_sha256": digest(Path(__file__).read_bytes()), "normalizer_code_sha256": digest((HERE / "prepare.py").read_bytes()),
        "input_hashes": {name: digest((DATA / name).read_bytes()) for name in ["cordis-observed.jsonl", "ror-v1.41-observed.jsonl.gz", "partitions.json"]},
        "query_count": len(lexical), "candidate_pairs": len(candidates), "registry_records": len(roster), "index": index,
        "fit_status": audit["fit_status"], "outcomes_opened": "none", "paid_calls": 0,
        "wall_seconds": time.perf_counter() - started, "files": outputs,
        "environment": {x: importlib.metadata.version(x) for x in ["splink", "duckdb", "pandas", "numpy", "scipy", "scikit-learn"]}}
    freeze_json(results / "freeze-manifest.json", manifest)
    return {k: manifest[k] for k in ["query_count", "candidate_pairs", "registry_records", "fit_status", "wall_seconds", "paid_calls"]}


if __name__ == "__main__":
    def label_guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = str(args[0]).replace("\\", "/")
            if "/organizations/data/gold/" in path or "/organizations/sources/" in path:
                raise RuntimeError("Inference process may not open source labels or gold files")
    sys.addaudithook(label_guard)
    print(json.dumps(run(), indent=2))
