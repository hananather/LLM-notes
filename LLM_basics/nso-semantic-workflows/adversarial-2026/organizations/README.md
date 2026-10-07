# Historical organization-linkage control

This package curates AffilGood's 3,329 structured CORDIS organization records against the complete ROR v1.41 registry. The reference target is one historically annotated research-organization identifier or no registry target. It does not establish legal-unit or establishment continuity.

The three native affiliation populations remain separate: 614 French, 322 multilingual and 168 unrelated-multi-organization records. Together they contain 1,081 distinct raw strings. Their target is the complete set of explicitly named organizations. No structured CORDIS row is counted as a native-text extraction example.

## Data and label boundary

| Location | Contents |
|---|---|
| `sources/` | Four unmodified publisher CSVs, including original annotations |
| `notices/` | Source licences and attribution |
| `data/cordis-observed.jsonl` | Opaque query ID and original `ORG`, `CITY`, `COUNTRY` only |
| `data/native-observed.jsonl` | Opaque query ID, source population and original affiliation string only |
| `data/ror-v1.41-observed.jsonl.gz` | All 108,476 historical registry entries with opaque IDs and observed names/location/status |
| `data/partitions.json` | Query IDs in development, evaluation and remainder |
| `data/gold/` | Original labels, registry identifiers, sampling groups, overlap and policy audit |
| `data/source-manifest.json` | Pinned sources, file hashes, schemas and reconstruction boundary |
| `protocol.json` | Fixed retrieval, fitting, prior and decision rules for the offline run |
| `results/` | Frozen candidates, predictions, fit diagnostics and development-only scoring |

The entire source `label` column is withheld from inference, including the organization names inside it. All positive reference IDs belong to the complete pinned registry. Inactive and withdrawn entries remain eligible because historical labels use them. Six repeated-input groups have inconsistent reference target sets; the audit records them and preserves every source row. No labels are silently corrected or removed.

CORDIS source records are grouped by shared positive ROR targets and normalized duplicate input triples. A seeded random permutation selects 100 groups for development and 1,000 groups for evaluation: 109 and 1,065 queries respectively. The remaining 2,155 records remain available as unscored input. Every source row is preserved. The sampling file records group inclusion probabilities; a record-level binomial interval would ignore the grouped design.

Inference and fitting use no training labels. Preparation uses published identities solely to keep duplicate/organization groups in one partition. The baseline is explicitly transductive: it may fit on all observed CORDIS input fields, including held-out input records. Native strings have no exact-text overlap with the earlier full S2AFF source; shared organizations and normalized text matches are recorded separately. Absence of exact overlap does not prove absence from model pretraining.

## Reconstruct and run

Use the experiment's Python 3.11 environment with Splink 5.0.0, DuckDB, NumPy, Pandas, SciPy and scikit-learn. The exact versions used appear in the frozen run manifest. These commands use no model API.

Download the [pinned ROR archive](https://github.com/ror-community/ror-data/raw/main/v1.41-2024-02-13-ror-data.zip) outside Git, then verify its SHA-256 is `0c532997867d205097f0be3ba13250ec3f3b322d580b1e88c23d0e9f4a25eafe`.

```bash
python prepare.py --ror-archive /path/to/ror-v1.41-2024-02-13.zip \
  --old-s2aff /path/to/pinned/gold_affiliation_annotations.csv
python baseline.py
python export_predictions.py
python score_development.py
```

Preparation refuses mismatched source hashes and changed frozen outputs. The optional old-S2AFF argument supplies the overlap audit; use the original pinned source when reproducing the supplied manifest. Baseline execution never opens `sources/` or `data/gold/`; an audit hook rejects those accesses. Development scoring verifies the frozen code, protocol, inputs and predictions before reading labels. It produces no evaluation metrics.

The supplied result files are immutable. Re-scoring them requires no refit. Repeating fitting produces run-time diagnostics that differ; reproduce a fit in a separate copy without its derived `results/` directory, while retaining the supplied results for comparison.

The baseline searches the full registry using normalized aliases, character TF-IDF and organization word TF-IDF. Exact aliases and weak city/country evidence enter a fixed score. The shared candidate union is determined without gold insertion. Linked, no-target and review decisions remain distinct; a separate forced top-1 result describes ranking performance.

The compact Splink model compares joint primary/alias name evidence, city and country. It estimates nonmatch probabilities from random cross-source pairs and match probabilities through complementary name/location blocks. The prior uses unique exact-alias/country anchors and a declared recall assumption; infeasible prior scenarios are rejected. Diagnostics retain fit warnings, sparse levels, exclusions and convergence. The learned probabilities are not established as calibrated. A capable lexical result remains available if EM is unstable or unidentifiable.

Predictions apply the saved Splink comparison SQL and effective likelihood ratios to the frozen candidate union. Prior sensitivity changes only posterior odds after fitting; it is not a refit sensitivity study. The fixed posterior grid does not select a best evaluation threshold.

The native-text sources are prepared for a distinct future test; this offline run does not fit or score a multi-organization selector. No paid organization sweep is included.

## Development evidence

The frozen development panel contains 109 queries in 100 source groups: 88 with a historical registry target and 21 without one. The rules below were fixed before development outcomes were opened. Evaluation predictions for 1,065 queries were frozen before the separate free holdout scoring described below.

| Fixed conventional method | Correct complete decisions | Automatic decisions | False links | False NIL | Review |
|---|---:|---:|---:|---:|---:|
| Lexical score and margin | 60 | 60 | 0 | 0 | 49 |
| Forced lexical top-1 | 77 | 109 | 32 | 0 | 0 |
| Splink EM posterior policy | 41 | 50 | 0 | 9 | 59 |

A review decision is unresolved and never counts as a correct NIL. Candidate recall is 77/88 at rank 1, 83/88 at rank 5, 86/88 at rank 10 and 87/88 at ranks 25 and 40. One positive development target is absent from the final candidate set. Retrieval and decision quality therefore have distinct denominators.

The lexical policy covers 55.0% of development queries with no observed false links. That count is descriptive evidence from this grouped historical sample, not a population error guarantee. The compact EM comparator makes nine false NIL decisions and has lower complete-decision accuracy. Both EM passes converged, but convergence does not establish identification or calibration; the exact-name match probability estimate is 0.0718 and requires scrutiny before making claims about the method class. No rule was changed after these scores were opened.

The offline run used no model API and took 39.93 seconds in the recorded environment. A native Splink inference check on 200 candidate pairs agreed with the saved likelihood calculations within `2.23e-16`. The fit audit preserves every estimated comparison level, EM history, prior scenario and library warning. Splink's `max_pairs` setting is an approximate sampled-pair target, not a strict realized-count bound.

`results/conventional-development.json` and `results/conventional-evaluation.json` use the common interface `method -> query_id -> {status, target_ids}`. Status is `linked`, `nil`, `review` or `failed`. `results/development-score.json` includes the common evaluator's complete-decision, false-assignment and coverage metrics; its provenance is in `development-evaluator-freeze.json` and `export-manifest.json`.

## Frozen free holdout evaluation

One complete evaluation used 1,065 queries in 1,000 source groups. The fixed lexical acceptance rule made 633 correct complete decisions, 640 automatic decisions and six false links; 425 queries remained for review. The frozen Splink policy made 323 correct decisions and 450 automatic decisions, with three false links, 124 false NIL decisions and 615 reviews. Forced top-1 made 820 correct decisions and 245 false links across all 1,065 automatic outputs; it remains a ranking diagnostic, not an operational acceptance rule.

The catalog shortlist contained 868 of 873 positive targets at rank 40. All 192 reference-NIL queries, four queries with flagged inconsistent source labels and 12 inactive/withdrawn-target queries remain in the denominator. No retrieval, fitting, threshold or prior changed. The [complete summary](results/free-evaluation-v1/summary.json) reports source-group intervals, paired differences and source-policy strata; the [audit](results/free-evaluation-v1/audit.json) records prediction sealing before outcome access. These conventional results contain no semantic-model comparison.
