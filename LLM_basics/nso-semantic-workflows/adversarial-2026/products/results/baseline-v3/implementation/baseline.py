"""Freeze label-free product candidates and scores. This program never loads truth."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from importlib.metadata import version
import io
import json
import logging
from pathlib import Path
import platform
import time
import warnings

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from splink import DuckDBAPI, Linker, SettingsCreator
import splink.comparison_library as cl
import splink.internals.expectation_maximisation as splink_em

from product_features import features, features_from_extraction, OBSERVED_FIELDS

HERE = Path(__file__).resolve().parent
CURATION = "v2"
RUN = "baseline-v3"
PROBABILITY_MASS = 1e-6
SEED = 20261007
THRESHOLDS = [0.5, 0.9, 0.99, 0.999]
PRIMARY_THRESHOLD = 0.9
LEXICAL_THRESHOLDS = [0.5, 0.65, 0.8, 0.9]
PRIMARY_LEXICAL_THRESHOLD = 0.65
CANDIDATE_K = 20
EM_BLOCKS = [
    "l.brand = r.brand AND substr(l.title_norm, 1, 4) = substr(r.title_norm, 1, 4)",
    "l.primary_model = r.primary_model",
    "array_length(list_intersect(l.title_words, r.title_words)) >= 3 AND jaro_winkler_similarity(l.title_norm, r.title_norm) >= 0.80",
]


class AuditedDuckDB(DuckDBAPI):
    """Observe the actual u-estimation node sample before Splink releases its table."""
    def __init__(self):
        super().__init__()
        self.u_samples = []

    def _sql_to_splink_dataframe(self, sql, templated_name, physical_name):
        result = super()._sql_to_splink_dataframe(sql, templated_name, physical_name)
        if templated_name == "__splink__df_concat_sample":
            rows = self._con.execute(f'SELECT source_dataset,unique_id FROM "{physical_name}" ORDER BY source_dataset,unique_id').fetchall()
            counts = {}
            for source, _ in rows:
                counts[source] = counts.get(source, 0) + 1
            self.u_samples.append({"source_record_counts": counts,
                "actual_cross_source_pairs": int(np.prod(list(counts.values()))) if len(counts)==2 else None,
                "sample_ids_sha256": sha256(json.dumps(rows).encode()).hexdigest(), "seed": SEED,
                "selection": "Splink 5 seeded observed-row hash sample; all cross-source pairs of sampled rows"})
        return result


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"Frozen file differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(raw)


def load_observed(dataset, split):
    location = HERE / "data" / dataset / CURATION
    records = json.loads((location / "records.json").read_text())
    partitions = json.loads((location / "partitions.json").read_text())
    if any(set(row) != set(OBSERVED_FIELDS) for row in records):
        raise ValueError("Unexpected inference fields")
    index = {r["record_id"]: r for r in records}
    ids = partitions[split]
    if set(ids["left"]) & set(ids["right"]):
        raise ValueError("Left and right opaque IDs overlap")
    return [index[i] for i in ids["left"]], [index[i] for i in ids["right"]]


def token_distribution(records):
    import tiktoken
    encoding = tiktoken.get_encoding("o200k_base")
    lengths = [len(encoding.encode(json.dumps(r, ensure_ascii=False))) for r in records]
    return {"count": len(lengths), "total": sum(lengths), "min": min(lengths, default=0),
            "p50": float(np.quantile(lengths, .5)) if lengths else 0,
            "p90": float(np.quantile(lengths, .9)) if lengths else 0,
            "p99": float(np.quantile(lengths, .99)) if lengths else 0,
            "max": max(lengths, default=0), "encoding": "o200k_base", "boundary": "Observed JSON token count; provider message/image overhead is not included."}


def baseline_contract(dataset, split, left, right):
    location = HERE / "data" / dataset / CURATION
    return {
        "schema": 1, "dataset": dataset, "split": split, "run": RUN,
        "input_counts": {"left": len(left), "right": len(right), "all_pairs": len(left) * len(right)},
        "fitting_scope": "Transductive unlabeled fitting on the observed left/right records in this split. No identity labels, family metadata, source identifiers or outcome scores are loaded. Development and test fits are separate.",
        "curation": CURATION, "seed": SEED,
        "source_sha256": {name: file_hash(location / name) for name in ["records.json", "partitions.json"]},
        "implementation_sha256": {name: file_hash(HERE / name) for name in ["baseline.py", "product_features.py"]},
        "lexical": {"word_title": "TF-IDF word 1-2 grams", "char_title": "TF-IDF char_wb 3-5 grams", "description": "TF-IDF word 1-2 grams; all observed description text",
                    "score": "clip(0.55*title_word+0.30*title_char+0.15*description+0.07*brand_agreement+0.18*model_intersection-0.15*quantity_conflict-0.10*primary_model_conflict,0,1)",
                    "thresholds": LEXICAL_THRESHOLDS, "primary_threshold": PRIMARY_LEXICAL_THRESHOLD, "calibrated_probability": False,
                    "candidate_union": "Union top 10 each by title word cosine, title character cosine and explicit combined score; include any shared model identifier; order union by combined score then opaque ID; retain first 20.", "candidate_k": CANDIDATE_K},
        "splink": {"version": "5.0.0", "comparisons": ["model tokens", "brand", "remaining title token Jaccard", "remaining description token Jaccard", "explicit title quantities", "observed numeric price"],
                   "u": {"method": "random unlabeled cross-source pairs", "max_pairs": 1_000_000, "seed": SEED, "min_count_per_level": None},
                   "prior": {"rule": "exact nonmissing brand and primary model OR exact nonempty normalized title", "assumed_strict_rule_recall": [0.5, 0.8, 1.0], "main_recall": 0.8,
                             "bounds": "Exclude probability outside (0,1); no one-to-one cap imposed on a many-target relation. These assumptions are not measured recall.",
                             "zero_rule_fallback": "1/max(left_count,right_count), an explicitly assumed sparse-overlap prior, with multipliers0.5/1/2."},
                   "em_blocks": EM_BLOCKS, "max_iterations": 100, "tolerance": 0.0001,
                   "fix_u_probabilities": True, "fix_probability_two_random_records_match": False,
                   "prior_boundary": "EM may update each blocked sample's match fraction; populate-global-prior is false, so the separately specified population prior stays unchanged.",
                   "term_frequency_adjustment": "exact brand agreement only; fit EM without TF then apply at prediction",
                   "prediction": "all cross-source pairs", "thresholds": THRESHOLDS, "primary_threshold": PRIMARY_THRESHOLD,
                   "calibrated_probability_established": False},
        "decision_policy": "Pair thresholds emit sets; never force one-to-one or a fixed number of targets. Every full-universe and candidate-restricted decision is preserved.",
        "unestimated_level_policy": "Baseline-v3 regularizes nonnull categorical vectors at initialization, after random-u, each EM M-step, and final aggregation. Add fixed 1e-6 probability mass per level then normalize; levels marked unobserved get zero raw mass first. All-zero vectors become uniform. Restore the exact normalized random-u snapshot after every Splink aggregation. Save pre-recovery defaults/flags. Require every nonempty EM block to converge; failure emits failed_fit without ordinary predictions. No truth-guided retries.",
        "prior_attribution_policy": "Recordwise extraction must reuse conventional split numeric prior and sensitivity grid exactly; agreement counts from extracted data never set its prior.",
        "stage_boundary": "Predictions saved before evaluation. This runner has no scoring command and does not load truth.json, metadata.json or source-map.json.",
        "versions": {p: version(p) for p in ["splink", "duckdb", "numpy", "pandas", "scikit-learn"]},
        "python": platform.python_version(),
    }


def vector_similarities(left_text, right_text, *, chars=False):
    kwargs = {"analyzer": "char_wb", "ngram_range": (3, 5)} if chars else {"ngram_range": (1, 2)}
    vectorizer = TfidfVectorizer(dtype=np.float32, lowercase=True, strip_accents="unicode", sublinear_tf=True, **kwargs)
    try:
        matrix = vectorizer.fit_transform(left_text + right_text)
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            return np.zeros((len(left_text), len(right_text)), dtype=np.float32)
        raise
    return (matrix[:len(left_text)] @ matrix[len(left_text):].T).toarray().astype(np.float32)


def lexical(left, right, left_features, right_features, output):
    start = time.perf_counter()
    title_word = vector_similarities([r["name"] for r in left], [r["name"] for r in right])
    title_char = vector_similarities([r["name"] for r in left], [r["name"] for r in right], chars=True)
    description = vector_similarities([r["description"] for r in left], [r["description"] for r in right])
    score = .55 * title_word + .30 * title_char + .15 * description
    candidates, decisions = [], {str(t): [] for t in LEXICAL_THRESHOLDS}
    right_models = [set(f["model_tokens"]) for f in right_features]
    right_quantities = [set(f["quantity_tokens"]) for f in right_features]
    right_dimensions = [set(f["quantity_dimensions"]) for f in right_features]
    for i, feature in enumerate(left_features):
        models, quantities, dimensions = (set(feature[n]) for n in ["model_tokens", "quantity_tokens", "quantity_dimensions"])
        shared_models = set()
        for j, other in enumerate(right_features):
            if feature["brand"] and feature["brand"] == other["brand"]:
                score[i, j] += .07
            if models & right_models[j]:
                score[i, j] += .18
                shared_models.add(j)
            elif feature["primary_model"] and other["primary_model"]:
                score[i, j] -= .10
            if dimensions & right_dimensions[j] and not quantities & right_quantities[j]:
                score[i, j] -= .15
        score[i] = np.clip(score[i], 0, 1)
        pool = shared_models
        for values in [title_word[i], title_char[i], score[i]]:
            pool.update(np.argsort(-values, kind="stable")[:10].tolist())
        order = sorted(pool, key=lambda j: (-float(score[i, j]), right[j]["record_id"]))[:CANDIDATE_K]
        candidates.append({"record_id": left[i]["record_id"], "candidate_ids": [right[j]["record_id"] for j in order]})
        for threshold in LEXICAL_THRESHOLDS:
            decisions[str(threshold)].append({"record_id": left[i]["record_id"], "target_ids": [right[j]["record_id"] for j in np.flatnonzero(score[i] >= threshold)]})
    # Numeric matrices contain no joined reference labels and are compact full-pair evidence.
    np.savez_compressed(output / "lexical-all-pair-scores.npz", score=score, title_word=title_word, title_char=title_char,
                        description=description, left_ids=np.array([r["record_id"] for r in left]), right_ids=np.array([r["record_id"] for r in right]))
    freeze(output / "candidates.json", candidates)
    freeze(output / "lexical-decisions.json", {"thresholds": decisions, "primary_threshold": PRIMARY_LEXICAL_THRESHOLD})
    index = {r["record_id"]: r for r in right}
    bundled = [{"query": r, "candidates": [index[i] for i in c["candidate_ids"]]} for r, c in zip(left, candidates)]
    freeze(output / "input-lengths.json", {"left": token_distribution(left), "right": token_distribution(right),
        "selection_bundles_all_fields": token_distribution(bundled), "truncation": "none"})
    return {"wall_seconds": time.perf_counter() - start, "candidate_pairs": sum(len(c["candidate_ids"]) for c in candidates)}


def comparisons():
    def custom(name, levels):
        return cl.CustomComparison(comparison_levels=levels, output_column_name=name)
    def level(sql, label, null=False):
        out = {"sql_condition": sql, "label_for_charts": label}
        if null:
            out["is_null_level"] = True
        return out
    def arrays(name, thresholds):
        union = f"array_length(list_distinct(list_concat({name}_l,{name}_r)))"
        overlap = f"array_length(list_intersect({name}_l,{name}_r))"
        return custom(name, [level(f"array_length({name}_l)=0 OR array_length({name}_r)=0", "missing", True)] +
            [level(f"{overlap}::DOUBLE/nullif({union},0)>={t}", f"Jaccard >= {t}") for t in thresholds] + [level("ELSE", "lower similarity")])
    return [
        custom("model", [level("array_length(model_tokens_l)=0 OR array_length(model_tokens_r)=0", "missing", True),
                         level("array_length(list_intersect(model_tokens_l,model_tokens_r))>=1", "shared explicit model identifier"),
                         level("length(primary_model_l)>=5 AND length(primary_model_r)>=5 AND damerau_levenshtein(primary_model_l,primary_model_r)=1", "primary model edit distance one"),
                         level("ELSE", "no shared model identifier")]),
        cl.ExactMatch("brand").configure(term_frequency_adjustments=True),
        arrays("title_words", [.9, .7, .4]), arrays("description_words", [.8, .5, .2]),
        custom("quantity", [level("array_length(list_intersect(quantity_dimensions_l,quantity_dimensions_r))=0", "no comparable title quantity", True),
                            level("array_length(list_intersect(quantity_tokens_l,quantity_tokens_r))>=1", "a stated quantity agrees"),
                            level("ELSE", "stated comparable quantities differ")]),
        custom("price", [level("price_value_l IS NULL OR price_value_r IS NULL OR price_value_l<=0 OR price_value_r<=0", "missing or zero price", True),
                         level("abs(price_value_l-price_value_r)/greatest(price_value_l,price_value_r)<=0.10", "price within ten percent"),
                         level("abs(price_value_l-price_value_r)/greatest(price_value_l,price_value_r)<=0.50", "price within fifty percent"),
                         level("ELSE", "larger price difference")]),
    ]


def parameter_audit(linker):
    rows = []
    for comparison in linker._settings_obj.core_model_settings.comparisons:
        for level in comparison.comparison_levels:
            row = {"comparison": comparison.output_column_name, "level": level.label_for_charts, "null": level.is_null_level}
            if not level.is_null_level:
                row.update({"m_raw": level._m_probability, "u_raw": level._u_probability,
                            "m_effective": level.m_probability, "u_effective": level.u_probability,
                            "m_estimated": level._has_estimated_m_values, "u_estimated": level._has_estimated_u_values})
            rows.append(row)
    return rows


def regularize(core, stage, events, parameters=("m", "u")):
    """Give every nonnull comparison a finite, positive categorical distribution."""
    for comparison in core.comparisons:
        levels = comparison._comparison_levels_excluding_null
        for parameter in parameters:
            raw = [getattr(level, f"_{parameter}_probability") for level in levels]
            effective = [getattr(level, f"{parameter}_probability") for level in levels]
            masses = np.array([0.0 if isinstance(value, str) else float(fallback if value is None else value)
                               for value, fallback in zip(raw, effective)], dtype=float)
            if not np.all(np.isfinite(masses)) or np.any(masses < 0):
                raise ValueError(f"Invalid {parameter} mass for {comparison.output_column_name}")
            repaired = (masses + PROBABILITY_MASS) / (masses.sum() + PROBABILITY_MASS * len(masses))
            for level, value in zip(levels, repaired):
                setattr(level, f"{parameter}_probability", float(value))
            events.append({"stage": stage, "comparison": comparison.output_column_name, "parameter": parameter,
                           "raw": raw, "effective_before": effective, "mass_used": masses.tolist(),
                           "normalized": repaired.tolist(), "additive_probability_mass": PROBABILITY_MASS})
    return validate_vectors(core)


def validate_vectors(core):
    result = []
    for comparison in core.comparisons:
        for parameter in ["m", "u"]:
            values = [getattr(level, f"{parameter}_probability") for level in comparison._comparison_levels_excluding_null]
            if not values or not all(np.isfinite(p) and 0 < p < 1 for p in values) or abs(sum(values)-1) > 1e-12:
                raise ValueError(f"Invalid categorical vector {comparison.output_column_name}/{parameter}: {values}")
            result.append({"comparison": comparison.output_column_name, "parameter": parameter,
                           "sum": sum(values), "minimum": min(values), "level_count": len(values)})
    return result


def probability_consistency(db, table, core):
    """Validate scores against the final distributions without reading any labels."""
    errors = {}
    columns = []
    for comparison in core.comparisons:
        name = comparison.output_column_name
        column = f"mw_{name}"
        columns.append(column)
        weights = [0.0] + [float(np.log2(level.m_probability/level.u_probability))
                          for level in comparison._comparison_levels_excluding_null]
        expr = "least(" + ",".join(f"abs({column}-({w!r}))" for w in weights) + ")"
        errors[name] = db._con.execute(f'SELECT max({expr}) FROM "{table}"').fetchone()[0]
    columns.append("mw_tf_adj_brand")
    prior = core.probability_two_random_records_match
    weight_expr = f"({float(np.log2(prior/(1-prior)))!r}) + " + " + ".join(columns)
    errors["total_weight"] = db._con.execute(f'SELECT max(abs(match_weight-({weight_expr}))) FROM "{table}"').fetchone()[0]
    errors["posterior"] = db._con.execute(f'SELECT max(abs(match_probability-1.0/(1.0+pow(2.0,-match_weight)))) FROM "{table}"').fetchone()[0]
    bad = db._con.execute(f'SELECT count(*) FROM "{table}" WHERE NOT isfinite(match_weight) OR NOT isfinite(match_probability) OR match_probability<0 OR match_probability>1').fetchone()[0]
    if bad or any(e is None or not np.isfinite(e) or e > 1e-9 for e in errors.values()):
        raise ValueError(f"Prediction/model inconsistency: errors={errors}; invalid_rows={bad}")
    return {"status": "passed", "maximum_absolute_errors": errors, "invalid_rows": bad,
            "boundary": "Every contribution equals a final categorical likelihood weight or null weight zero; total includes saved brand term-frequency adjustment and prior; posterior agrees with total. No truth read."}


def splink(left_features, right_features, candidates, output, *, prior_bundle=None):
    start = time.perf_counter()
    a, b = pd.DataFrame(left_features), pd.DataFrame(right_features)
    db = AuditedDuckDB()
    # Exclude the long retrieval-only field from the SQL pair materialization.
    a = a.drop(columns=["search_text"]); b = b.drop(columns=["search_text"])
    db._con.register("observed_left", a)
    db._con.register("observed_right", b)
    strict = db._con.execute("SELECT count(*) FROM observed_left l CROSS JOIN observed_right r WHERE (l.brand=r.brand AND l.primary_model=r.primary_model) OR (l.title_norm=r.title_norm AND l.title_norm<>'')").fetchone()[0]
    total = len(a) * len(b)
    priors = {str(recall): strict / (total * recall) for recall in [.5, .8, 1.0]} if strict else {
        str(recall): multiplier / max(len(a), len(b)) for recall, multiplier in [(.5, 2), (.8, 1), (1., .5)]}
    if prior_bundle is not None:
        if set(prior_bundle["priors"]) != {"0.5", "0.8", "1.0"}:
            raise ValueError("Required conventional prior grid is incomplete")
        priors = dict(prior_bundle["priors"])
        if priors["0.8"] != prior_bundle["main_prior"]:
            raise ValueError("Conventional main prior does not match sensitivity grid")
    main_prior = priors["0.8"]
    if not all(0 < p < 1 for p in priors.values()):
        raise ValueError("Infeasible declared prior; stop instead of silently clipping")
    settings = SettingsCreator(link_type="link_only", comparisons=comparisons(),
        blocking_rules_to_generate_predictions=["1=1"], probability_two_random_records_match=main_prior,
        retain_matching_columns=False, retain_intermediate_calculation_columns=True,
        max_iterations=100, em_convergence=.0001)
    tables = [db.register(frame, dataset_display_name=name) for frame,name in [(a,"product_left"),(b,"product_right")]]
    linker = Linker(tables, settings, log_level="WARNING")
    log = io.StringIO()
    handler = logging.StreamHandler(log)
    logger = logging.getLogger("splink"); logger.addHandler(handler)
    sessions, regularization_events = [], []
    failure = None
    fitted = []
    model_checks = None
    raw_fitted = []
    original_maximisation_step = splink_em.maximisation_step
    def normalized_maximisation_step(*args, **kwargs):
        core = original_maximisation_step(*args, **kwargs)
        regularize(core, "EM M-step", regularization_events, parameters=("m",))
        return core
    with warnings.catch_warnings(record=True) as caught, redirect_stderr(log), redirect_stdout(log):
        warnings.simplefilter("always")
        try:
            regularize(linker._settings_obj.core_model_settings, "initialization", regularization_events)
            linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000, seed=SEED, min_count_per_level=None, num_chunks=1)
            regularize(linker._settings_obj.core_model_settings, "random-u", regularization_events, parameters=("u",))
            fixed_u = {c.output_column_name: [lev.u_probability for lev in c._comparison_levels_excluding_null]
                       for c in linker._settings_obj.core_model_settings.comparisons}
            splink_em.maximisation_step = normalized_maximisation_step
            for block in EM_BLOCKS:
                pair_count = db._con.execute("SELECT count(*) FROM observed_left l CROSS JOIN observed_right r WHERE " + block).fetchone()[0]
                if pair_count == 0:
                    sessions.append({"block": block, "pairs": 0, "status": "empty_block"})
                    continue
                try:
                    session = linker.training.estimate_parameters_using_expectation_maximisation(block,
                        estimate_without_term_frequencies=True, fix_u_probabilities=True,
                        fix_m_probabilities=False, fix_probability_two_random_records_match=False,
                        populate_probability_two_random_records_match_from_trained_values=False)
                    history = []
                    for core in session._core_model_settings_history:
                        values = [core.probability_two_random_records_match]
                        for comparison in core.comparisons:
                            for lev in comparison.comparison_levels:
                                if not lev.is_null_level:
                                    values.extend([lev.m_probability, lev.u_probability])
                        history.append(values)
                    delta = max(abs(x-y) for x,y in zip(history[-1], history[-2])) if len(history)>1 else None
                    sessions.append({"block": block, "pairs": pair_count, "status": "fit_returned", "iterations": len(history)-1,
                        "parameter_tolerance_met": delta is not None and delta<.0001, "last_max_parameter_delta": delta,
                        "excluded_comparisons": [c.output_column_name for c in session._comparisons_that_cannot_be_estimated], "numeric_parameter_history": history})
                    for comparison in linker._settings_obj.core_model_settings.comparisons:
                        for lev, value in zip(comparison._comparison_levels_excluding_null, fixed_u[comparison.output_column_name]):
                            lev.u_probability = value
                    regularize(linker._settings_obj.core_model_settings, "session aggregation", regularization_events, parameters=("m",))
                except Exception as exc:
                    sessions.append({"block": block, "pairs": pair_count, "status": "failed", "exception": f"{type(exc).__name__}: {exc}"})
            if not sessions or not any(s["status"] == "fit_returned" for s in sessions) or any(s["status"] == "failed" or (s["status"] == "fit_returned" and not s["parameter_tolerance_met"]) for s in sessions):
                raise ValueError("At least one nonempty EM block failed or did not converge")
            raw_fitted = parameter_audit(linker)
            regularize(linker._settings_obj.core_model_settings, "final aggregation", regularization_events, parameters=("m",))
            fitted = parameter_audit(linker)
            model = linker.misc.save_model_to_json()
            if model["probability_two_random_records_match"] != main_prior:
                raise ValueError("Blocked EM changed the separately specified global prior")
            freeze(output / "splink-model.json", model)
            frame = linker.inference.predict(threshold_match_probability=0)
            table = frame.physical_name
            model_checks = probability_consistency(db, table, linker._settings_obj.core_model_settings)
            prediction_path = output / "splink-all-pairs.parquet"
            if prediction_path.exists():
                raise FileExistsError("Do not overwrite frozen predictions")
            # Paths are generated locally, not derived from a product description.
            db._con.execute(f'COPY (SELECT * FROM "{table}") TO ? (FORMAT PARQUET, COMPRESSION ZSTD)', [str(prediction_path)])
            count = db._con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            if count != total:
                raise ValueError(f"Incomplete all-pairs prediction: {count} != {total}")
            candidate_frame = pd.DataFrame([{"unique_id_l": c["record_id"], "unique_id_r": rid} for c in candidates for rid in c["candidate_ids"]])
            db._con.register("declared_candidates", candidate_frame)
            candidate_scores = db._con.execute(f'SELECT p.unique_id_l,p.unique_id_r,p.match_probability,p.match_weight FROM "{table}" p INNER JOIN declared_candidates c USING(unique_id_l,unique_id_r)').df()
            if len(candidate_scores) != len(candidate_frame):
                raise ValueError("Candidate score orientation/count mismatch")
            candidate_scores.to_parquet(output / "splink-candidate-scores.parquet", index=False)
            decisions = {}
            main_log_odds = np.log2(main_prior / (1-main_prior))
            # Prior-only sensitivity holds fitted likelihood ratios fixed; no hidden refit.
            for recall, prior in priors.items():
                offset = float(np.log2(prior/(1-prior)) - main_log_odds)
                for threshold in THRESHOLDS:
                    cutoff = float(np.log2(threshold/(1-threshold)))
                    pairs = db._con.execute(f'SELECT unique_id_l,unique_id_r FROM "{table}" WHERE match_weight + ? >= ? ORDER BY unique_id_l,unique_id_r', [offset, cutoff]).fetchall()
                    by_left = {rid: [] for rid in a.unique_id}
                    for rid, target in pairs:
                        by_left[rid].append(target)
                    decisions[f"assumed_recall={recall};threshold={threshold}"] = [{"record_id": rid,"target_ids": targets} for rid,targets in by_left.items()]
            freeze(output / "splink-decisions.json", {"main_assumed_recall": .8,"main_threshold": PRIMARY_THRESHOLD,"decisions":decisions,
                "sensitivity_boundary": "Prior-only shift at fixed fitted m/u, not a complete refit of each prior."})
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            splink_em.maximisation_step = original_maximisation_step
            logger.removeHandler(handler)
            (output / "splink-training.log").write_text(log.getvalue())
    warning_messages = [f"{w.category.__name__}: {w.message}" for w in caught]
    (output / "splink-training.log").write_text(log.getvalue())
    audit = {"strict_rule_pairs": strict, "all_pair_count": total, "priors": priors,"main_prior": main_prior,
        "status": "failed_fit" if failure else "predictions_frozen_without_truth_scoring", "failure": failure,
        "prior_source": "caller-supplied conventional numeric priors" if prior_bundle is not None else "conventional observed-agreement rule with assumed recall",
        "prior_bundle": prior_bundle,
        "prior_implied_link_counts": {recall:p*total for recall,p in priors.items()},
        "prior_evidence_boundary": "Strict observed-agreement rule counts and assumed recall only; no reference identity labels or true overlap count. No one-to-one bound applies.",
        "em_sessions": sessions, "comparison_levels": fitted,"pre_final_regularization_levels": raw_fitted, "python_warnings": warning_messages,
        "regularization_events": regularization_events, "prediction_consistency": model_checks,
        "categorical_vector_checks": validate_vectors(linker._settings_obj.core_model_settings) if not failure else None,
        "u_sampling": db.u_samples,
        "wall_seconds": time.perf_counter()-start,"held_out_truth_loaded":False,
        "unestimated_m_levels":sum(not r.get("m_estimated",True) for r in fitted),
        "unestimated_u_levels":sum(not r.get("u_estimated",True) for r in fitted)}
    freeze(output / "splink-fit-audit.json", audit)
    if failure:
        freeze(output / "failed-fit-decisions.json", {"status":"failed_fit", "reason":failure,
            "decisions":[{"record_id": rid,"status":"failed", "target_ids":[]} for rid in a.unique_id]})
    return {k:audit[k] for k in ["status","failure","all_pair_count","unestimated_m_levels","unestimated_u_levels","wall_seconds"]}


def run(dataset, split):
    left, right = load_observed(dataset, split)
    if not left or not right:
        return {"dataset": dataset,"split":split,"status":"empty_partition"}
    output = HERE / "results" / RUN / dataset / split
    output.mkdir(parents=True,exist_ok=True)
    contract = baseline_contract(dataset, split, left, right)
    freeze(output / "protocol.json",contract)
    if (output / "completion.json").exists():
        return json.loads((output / "completion.json").read_text())
    if any(output.glob("*.parquet")):
        raise FileExistsError("A partial prediction batch exists; inspect the failure without overwriting its evidence")
    left_features,right_features = ([features(r) for r in rows] for rows in [left,right])
    lex = lexical(left,right,left_features,right_features,output)
    candidates = json.loads((output / "candidates.json").read_text())
    fs = splink(left_features,right_features,candidates,output)
    files = {p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}
    complete = {"dataset":dataset,"split":split,"status":fs["status"], "lexical":lex,"splink":fs,"files":files}
    freeze(output / "completion.json",complete)
    return complete


def run_recordwise(left_rows, right_rows, extractions_by_id, candidates, output_dir, *, prior_bundle):
    """Fresh unlabeled fit from caller-supplied inference rows and frozen candidates."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expected = {row["record_id"] for row in left_rows + right_rows}
    if set(extractions_by_id) != expected:
        raise ValueError("Each observed record needs an extraction entry; failures must be explicit nulls")
    if len(expected) != len(left_rows) + len(right_rows):
        raise ValueError("Record IDs must be globally unique")
    freeze(output / "recordwise-input-manifest.json", {
        "left_count": len(left_rows), "right_count": len(right_rows),
        "failed_extractions": sum(value is None for value in extractions_by_id.values()),
        "failed_extraction_ids": sorted(rid for rid,value in extractions_by_id.items() if value is None),
        "observed_sha256": sha256(json.dumps(left_rows + right_rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "extractions_sha256": sha256(json.dumps(extractions_by_id, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "candidates_sha256": sha256(json.dumps(candidates, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "conventional_prior_bundle": prior_bundle,
        "implementation_sha256": {name:file_hash(HERE / name) for name in ["baseline.py","product_features.py","schema.json"]},
        "scope": "Same comparisons and fresh transductive unlabeled EM. Full-pair and unchanged-candidate results are separate. No truth is read.",
    })
    left = [features_from_extraction(row, extractions_by_id[row["record_id"]]) for row in left_rows]
    right = [features_from_extraction(row, extractions_by_id[row["record_id"]]) for row in right_rows]
    return splink(left, right, candidates, output, prior_bundle=prior_bundle)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["abt-buy","amazon-google"])
    parser.add_argument("--split", choices=["dev","test"],required=True)
    args=parser.parse_args()
    result=run(args.dataset,args.split)
    # Only shape, timing and fitting diagnostics are printed, never test accuracy.
    print(json.dumps({k:v for k,v in result.items() if k!="files"},indent=2))


if __name__=="__main__":
    main()
