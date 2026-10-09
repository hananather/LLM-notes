# Public product linkage: frozen inputs and offline baselines

This experiment asks whether interpreting product text improves linkage over declared conventional comparators without task-specific training labels. It preserves two complete historical source graphs and supports separate recordwise extraction and candidate-controlled selection. Development and the completed amended holdout show a tradeoff: semantic selection recovers more complete published target sets than the fixed primary lexical rule and adds false links. The original continuation gates remain failed; recordwise extraction stays development-only.

| Source | Complete left / right records | Supplied links | Development left / right | Test left / right |
|---|---:|---:|---:|---:|
| Amazon–GoogleProducts | 1,363 / 3,226 | 1,300 | 144 / 352 | 1,219 / 2,874 |
| Abt–Buy | 1,081 / 1,092 | 1,097 | 0 / 0 | 1,081 / 1,092 |

Amazon–Google is the prospective primary text source. Its test partition contains 231 left records and 1,728 right records without a supplied counterpart. Those are unmatched under the publisher's mapping convention. Abt–Buy is an intact offline replication source; every record has a supplied counterpart. Several records in each source have multiple counterparts. No one-to-one assignment or invented unmatched case is imposed.

## Source and split boundary

The [source notices](SOURCE-NOTICES.md) identify the publisher, source ZIPs and CC BY 4.0 licence. ZIP bytes are preserved in `sources/`. Their SHA-256 hashes appear in each curation manifest.

The current curation is `data/{dataset}/v2/`. Each `records.json` contains exactly `record_id`, `name`, `description`, `manufacturer` and `price`. Opaque IDs have no source-name or entity-label meaning. `partitions.json` holds split and table membership. `truth.json`, `metadata.json` and `source-map.json` are evaluator/provenance files; the baseline does not load them.

Whole published-link components, duplicate observed texts and identifiable title families stay in one partition. Family grouping uses same-brand model prefixes and the first two substantive title words after generic version, platform and licence qualifiers are removed. This grouping is intentionally broader than identity and is incomplete for undocumented lineages. Development targets 125 positive components. Unmapped family groups use the same seeded sampling fraction. Actual development and test counts are retained without forcing exact table sizes.

Initial curation remains in each dataset's parent folder. Its explicitly saved 12-record development source audit showed that software edition and licence families needed grouping. Curation v2 made that structural correction before any predictions. The original curation implementation bytes are preserved in `data/curation-v2-source/`, with the hashes recorded in the manifests.

## Conventional comparisons

`product_features.py` normalizes Unicode and punctuation, parses explicit brands, model identifiers, title quantities and observed numeric prices, and preserves source descriptions. Shared deterministic brand aliases and unit conversions apply to conventional and model-extracted fields. Model identifiers and their punctuation-separated components are removed from the remaining title/description word comparisons to reduce double counting.

The lexical comparator combines word and character title TF-IDF, description TF-IDF, brand/model agreement and explicit quantity/model conflicts. Its score is not a probability. Fixed thresholds are 0.5, 0.65, 0.8 and 0.9, with 0.65 primary. The candidate pool unions the top ten word-title, character-title and combined-score results with shared-identifier matches. It ranks the union by the combined score and opaque ID, then retains at most 20. A small union is not padded with additional records.

Splink 5 uses six comparisons: explicit model identifiers, brand, remaining title words, remaining description words, explicit title quantities and price. Fitting is transductive and unlabeled within each complete split. It uses observed test covariates, but no test identities, threshold tuning or reference-label-derived aliases. The development and test fits are separate. A label-free fit is not a claim that the probabilities are calibrated.

Random-pair estimation supplies nonmatch probabilities. The runner records the actual sampled source counts, sample-ID hash and seed before the sample table is released. Complementary EM blocks estimate match probabilities; blocked match fractions may change while the separately specified population prior stays fixed. All warnings, convergence histories, unestimated levels and effective default parameters are retained.

The current `baseline-v6` uses an explicit probability regularization extension to Splink's EM. At initialization, random-pair estimation and each EM update, it adds a fixed 10⁻⁶ mass per nonnull level and normalizes. A sampling-unobserved level contributes zero raw mass before smoothing. A comparison with no observed match levels copies its fixed nonmatch distribution, making that comparison uninformative. An entirely unobserved nonmatch vector becomes uniform. Session and final aggregation normalize already-positive values without adding further mass. The exact normalized random-pair estimates are restored after each session. Every used distribution must be finite, positive and sum to one; every nonempty session must converge within 500 iterations at a maximum parameter-change tolerance of 0.0001. Failure produces explicit failed-fit decisions. Full-pair numerical checks verify final likelihood contributions, total match weights and posterior conversion without accessing truth.

The prior uses strict observed agreement with an assumed recall of 0.8. The strict rule's recall is unknown. Original sensitivities use 0.5 and 1.0. A broader descriptive grid assumes 0.1, 0.5, 1 or 2 links per left record, giving pair prior equal to that rate divided by the right-table size. These are many-target prior assumptions, not overlap estimates or cardinality caps. They shift prior odds while holding fitted likelihood ratios fixed; they are not complete alternative-prior refits. The recordwise semantic arm reuses the conventional split's exact numeric prior and sensitivity grid. Implied link counts and the evidence boundary are saved. A one-to-one capacity bound does not apply to these graphs.

The primary Splink threshold is 0.9. Descriptive thresholds are 0.5, 0.99 and 0.999. Every threshold emits target sets without forcing a best candidate. Full-universe and candidate-restricted decisions must be reported separately; candidate misses remain in end-to-end denominators.

## Interfaces and reproduction

Run from the repository root using the existing Python 3.11 environment with Splink 5.0.0, DuckDB 1.5.6, NumPy 1.26.4, Pandas 2.3.3 and scikit-learn 1.9.1:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/products/baseline.py amazon-google --split dev
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/products/baseline.py amazon-google --split test
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/products/baseline.py abt-buy --split test
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/products/verify.py
```

Completed commands replay their completion manifests after checking the code/input contract. They do not overwrite different protocols or partial prediction batches. New methodology requires a new run name. These baseline commands neither call a paid model nor open identity labels to compute accuracy. Separate score scripts read reference labels only after their required prediction freeze.

`results/baseline-v6/amazon-google/{dev,test}/candidates.json` is the current shared listwise interface: a list of `{record_id, candidate_ids}`. Its generating implementation is preserved beside the outputs. `input-lengths.json` gives observed JSON token distributions using `o200k_base`, excluding provider overhead. No source text was truncated for these files.

`results/baseline-v6/{dataset}/{split}/` preserves the current protocol, all-pair numerical lexical arrays, all-pair Splink Parquet, candidate scores, threshold decisions, complete fitted model, warnings, fit audit and file-hash completion manifest. Prediction status is distinct from outcome evaluation. The baseline runner has no truth-scoring command.

The 2026-10-07 offline run completed Amazon development (50,688 pairs), Amazon test (3,503,406 pairs) and Abt test (1,180,452 pairs). All nine EM sessions met the declared tolerance; Amazon test required 114, 39 and 43 iterations. Every final categorical distribution and all-pair numerical check passed. `results/integrity-verification-v6.json` records source, curation, implementation and output-hash checks; `results/numerical-audit-v6.json` records convergence and sampling diagnostics. This evidence establishes successful computation and numerical consistency, not linkage accuracy or calibration. No identity-outcome scoring was performed by this runner.

Earlier runs remain as numerical validation evidence. `baseline-v1` exposed unnormalized sparse-level defaults and must not be used as the valid comparator. `baseline-v2` stopped on Splink's restoration of raw nonmatch estimates. Development-only `baseline-v3` and `baseline-v4` diagnosed initialized values for sampling-unobserved levels and repeated smoothing during aggregation. These corrections used parameter invariants, before outcome inspection. `baseline-v5` preserved a failed Amazon test fit: its first block reached 100 iterations with maximum change 0.000242675. Before identity-outcome inspection, v6 raised the finite cap to 500 for all splits and representations, preserving the algorithm, tolerance, primary prior and thresholds. The broad prior grid was declared at the same boundary; it must be reported without selecting the best result. Their exact code snapshots are preserved. Synthetic `structural-smoke-*` files test interfaces only and are not linkage results.

The [extraction contract](extraction-contract.md) and `schema.json` define source-supported attributes and limits. `features_from_extraction(observed_record, extraction_or_null)` produces the shared comparison representation. `baseline.run_recordwise(left_rows, right_rows, extractions_by_id, candidates, output_dir, prior_bundle=...)` fits fresh unlabeled EM and scores caller-supplied records without reading labels. The required prior bundle contains `main_prior` and the complete `priors` dictionary copied from that conventional split's `splink-fit-audit.json`. Its keys are `0.5`, `0.8`, `1.0` and `links_per_left=0.1`, `links_per_left=0.5`, `links_per_left=1.0`, `links_per_left=2.0`. Every source record needs an extraction entry; null failures are preserved by ID. The caller must retain failed-query status and must not present a decision using failed required extraction as successful. Run fits serially within a Python process because the scoped Splink EM extension is restored after each fit.

## Evidence limits

The datasets are public, historical and possibly present in pretrained models' data. Publisher mappings are external references, not a new complete independent adjudication. A reference NIL does not certify real-world absence. Family grouping is a reproducible observable rule, not exhaustive product genealogy. Prices may differ for legitimate offers and are supporting evidence only.

These experiments measure product identity under the stated source policy. They do not measure equivalent-product substitution, current prices, staff time saved, national-statistical-office production accuracy or a general failure of Fellegi–Sunter linkage.

## Scored comparisons

[Development results](results/model-development-v1/semantic-primary.json) retain all 144 queries. The fixed lexical rule makes 56 correct complete decisions with eight false links; primary Splink makes 21 with none; extraction plus fresh Splink makes 27 with one; semantic selection makes 112 with 20. Semantic selection also leaves eight queries for review. Both model arms fail the original spending rule. Extraction differences change across the complete fixed prior grid, so its primary gain is not a general representational advantage.

The [explicit amendment](../contracts/product-tradeoff-amendment-v1.json) overrode the spending stop once to characterize the unchanged selector on the reserved 1,219-query partition. It preserved the failed gate, all fixed comparisons and full failure denominator. The provider completed all 1,219 requests, and the [complete prediction seal](../results/product-holdout-v1/prediction-seal.json) was written at 2026-10-08 05:37:56 UTC. The frozen evaluator [verified that seal](results/model-evaluation-v1/seal-verification.json) before opening reference identities. The [completion manifest](results/model-evaluation-v1/completion.json) binds the saved result hashes. Notebook replay never initiates a paid run.

<a id="access-chronology"></a>
On 7 October 2026, after evaluation requests and scoring rules were frozen and the requests submitted, editorial inspection exposed reference-derived counterpart-presence and grouping metadata for two held-out records: one query and one catalog record. Prediction collection and sealing were incomplete. The inspection did not open the test reference-link file or test prediction/outcome files. No prompts, candidates, methods, requests, thresholds or scoring rules changed following this access. This deviated from the intended pre-seal access restriction. All 1,219 predictions were sealed on 8 October before the frozen evaluator read the full outcome labels and scored the complete query set. The [report disclosure](../../../record-linkage-with-semantic-operators.ipynb#product-access-chronology) preserves this distinction.

All methods below use the complete 1,219-query denominator across 916 observed source families. A correct complete decision returns the entire published reference set or a valid no-match decision when that set is empty. False and missed links count individual query–catalog edges; a query can contribute several errors.

| Fixed primary method | Correct complete decisions / 1,219 | False links | Missed links | Queries with false assignment | Automatic decisions | Review / failed |
|---|---:|---:|---:|---:|---:|---:|
| Lexical, all pairs, threshold 0.65 | 382 (31.3%) | 264 | 788 | 210 | 1,219 | 0 / 0 |
| Splink, all pairs, primary prior and threshold 0.9 | 275 (22.6%) | 45 | 1,082 | 39 | 1,219 | 0 / 0 |
| Lexical, same candidates, threshold 0.65 | 382 (31.3%) | 264 | 788 | 210 | 1,219 | 0 / 0 |
| Splink, same candidates, primary prior and threshold 0.9 | 275 (22.6%) | 45 | 1,082 | 39 | 1,219 | 0 / 0 |
| Semantic candidate selection | 702 (57.6%) | 508 | 151 | 388 | 1,125 | 68 / 26 |

The lexical all-pairs rule remains the development-selected reference. The selector leaves 94 queries unresolved: 68 reviews and 26 failed responses. Its automatic coverage is 92.3%, with 423 wrong decisions among 1,125 automatic decisions (37.6%). Positive reference links on unresolved queries remain missed; unresolved no-match queries are not correct decisions. Of the 1,219 provider responses, 1,193 met the parsing contract and 26 failed. Valid parsing includes requests for review and does not imply a correct linkage decision.

The [paired comparison](results/model-evaluation-v1/paired-family-uncertainty.json) against that lexical reference shows 320 more correct complete decisions, 244 more false links and 637 fewer missed links. Corresponding differences are +26.3 percentage points (95% interval 22.0 to 30.6), +0.200 false links per query (0.150 to 0.247) and −0.523 missed links per query (−0.565 to −0.481). The saved file reports all seven endpoints for all 64 conventional rules. Its 2,000 paired bootstrap replicates resample the 916 observed source families. Intervals are pointwise and descriptive, with no multiplicity correction or NSO-population interpretation.

The [candidate audit](results/model-evaluation-v1/candidate-audit.json) records 17,943 candidate pairs. Retrieval included 1,137 of 1,155 reference links and every reference target for 971 of 988 matched queries. Including the 231 reference no-match queries, complete reference sets were available for 1,202 of 1,219 queries. No missing target was inserted for scoring.

The reference-cardinality groups expose the no-match errors alongside recovery of single and multiple targets. Every cell reports correct complete decisions over the full group size; the [full subgroup file](results/model-evaluation-v1/subgroup-summaries.json) retains review, failure and edge counts.

| Reference group | Queries | Lexical primary | Splink primary | Selector |
|---|---:|---:|---:|---:|
| No supplied target (NIL) | 231 | 152 / 231 | 216 / 231 | 27 / 231 |
| One supplied target | 859 | 221 / 859 | 58 / 859 | 605 / 859 |
| Multiple supplied targets | 129 | 9 / 129 | 1 / 129 | 70 / 129 |

The [full fixed-rule summaries](results/model-evaluation-v1/all-fixed-rule-summaries.json) preserve eight lexical and 56 Splink policies plus the selector, with all-pairs and same-candidate scopes separate. No held-out threshold or prior is selected. The [loss grid](results/model-evaluation-v1/descriptive-loss-grid.json) evaluates `lambda × false links + missed links` at the predeclared ratios 0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50 and 100. These are hypothetical edge-cost ratios; staff time, review costs and ROI were not measured. The edge loss must be read alongside the selector's 63 unresolved positive and 31 unresolved NIL queries, without assuming successful or free review or selecting an operational winner. The [query audit](results/model-evaluation-v1/common-query-outcomes.json) retains every method's decision and outcome.

The [execution audit](results/model-evaluation-v1/primary-summary.json) records `gpt-6-luna` for every returned model identifier. The batch reserved US$0.963497421875 against the amendment's US$0.964 cap. Its nominal token-price estimate is US$0.119846250, with zero unpriced results. This estimate is not an invoice and excludes local compute and labour. The [request ledger](../results/api-ledger.jsonl), [batch plan](../results/product-holdout-v1/batch-plan.json) and [terminal transport payload](../results/product-holdout-v1/batch-plan-evidence/chunk-0000/output.jsonl) retain request and response evidence.

From the repository root, verify and replay the completed evaluation without provider requests:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/products/score_evaluation.py \
  --manifest LLM_basics/nso-semantic-workflows/adversarial-2026/contracts/products-tradeoff-test-v1.json \
  --seal LLM_basics/nso-semantic-workflows/adversarial-2026/results/product-holdout-v1/prediction-seal.json \
  --score-sealed
```

With the saved completion manifest present, this checks the bound inputs and output hashes and replays the completed summary. Omitting `--score-sealed` verifies the seal only, without parsing reference labels. Different evidence stops replay. The amendment ends after this one run; it does not authorize another sample or a revived extraction arm.

The independent [Abt–Buy conventional reference](results/free-abt-reference-v1/reference-summary.md) has 1,081 queries in 459 source families. The primary lexical rule makes 341 correct complete sets with 13 false links; primary Splink makes 310 with 455. All rules, same-candidate comparisons and family intervals are retained. No semantic model ran on Abt–Buy.
