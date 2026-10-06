# Canadian industry and occupation coding pilot

This experiment measures classification on a fixed, synthetic case set. It does not link people or businesses and does not estimate a national statistical office's accuracy or workload savings. A direct JSON classifier and an optional extraction-then-classification arm use the same query-driven retrieval implementation. The runner uses the provider API directly; it does not call native LOTUS operators.

## Data and reference answers

The reference resources are Statistics Canada's NAICS Canada 2022 Version 1.0 and NOC 2021 Version 1.0 structures and element files. Exact URLs, acquisition date, byte hashes and licence links are in `data/coding/sources.json`. The runner downloads missing source CSVs into the ignored `cache/coding/` directory and rejects changed bytes. The CSV cache is not required in Git.

There are 48 synthetic NAICS cases and 48 synthetic NOC cases. Each scheme has 12 boundary families with four cases: one complete description on each side, a second wording of one complete fact bundle, and an explicitly unresolved description. The separate `facts-gold.json` ledger specifies facts, labels, compatible alternatives and source-evidence keys. `source-evidence.json` preserves the relevant official definitions, exclusions and duties with source-record positions. `cases.json` contains observable narratives and metadata without gold codes.

The source-derived facts and reference decisions were specified before narrative materialization; the build writes the facts ledger before the narrative file. The same research agent authored both the fact specifications and prose. This is not blinded independent human annotation. Construction was independent of the evaluated model, GPT-6 Luna. A fresh independent label audit is required before live evaluation. Any audit changes create a new documented freeze before model calls. Synthetic descriptions are not actual survey returns.

Four families per scheme are development and eight are test, using seed 61006. The split was materialized before baseline scoring. All related variants stay together. Development has 32 total cases and test has 64; each has 75% uniquely coded cases and 25% review cases. The selected families are a convenience sample, not a probability sample of occupations, industries or difficult operational records.

The 48 official-verbatim controls contain the first published All examples entry for each of the 48 selected classes. Every official alias remains in the full lookup index. These controls measure correct ingestion and exact lookup. They offer no evidence of generative-model value and are not included in synthetic-case accuracy.

## Full-resource baseline and candidate construction

Every method has access to the same complete official reference index: all six-digit NAICS classes and five-digit NOC unit groups, their definitions, all example phrases, inclusions and NOC duties. A full-length destination referenced by an exclusion contributes its activity phrase to the destination's positive retrieval evidence. The exclusion remains attached to the original class as a negative constraint. Broad destination references are not converted into invented detailed-class labels.

The comparator includes normalized exact lookup, word 1–2-gram TF–IDF and character-within-word 3–5-gram TF–IDF. It retrieves individual official entries, takes the maximum score for each class in each representation and combines scores with fixed weights 0.55 word/0.45 character. Explicitly negated clauses are removed from the positive retrieval query by a generic syntactic rule; the complete original narrative remains available for classification. All classes are eligible. Ten highest-scoring classes form the candidate pool. Ties break by code. No target label or family-specific code list enters retrieval.

The baseline selects the highest-scoring class unless the description explicitly marks missing information or the score margin is below its threshold. Thresholds are chosen separately for NAICS and NOC from 0, .02, .05, .10 and .15 on development decision accuracy, with ties favouring the smaller threshold. Test labels do not choose thresholds. An explicit missing-information detector is included because these authored review cases signal missing information in text. Its success must not be interpreted as demonstrated understanding of naturally ambiguous survey responses.

This is a strong local retrieval baseline for a bounded pilot, not a reproduction of StatCan G-Code or a trained operational classifier. A word/character supervised classifier becomes relevant if using a held-out-dictionary experiment; the present complete-index test deliberately withholds no aliases. Candidate recall at ten is reported separately from downstream code accuracy, including whether both stated review alternatives are present. Retrieval failures count in end-to-end evaluation.

## Model arms

The primary arm receives the original narrative, classification version and ten query-derived candidates with definitions, matched evidence and exclusions. It must return `code` or `review`, a candidate code or null, compatible candidates, exact source quotations, missing information and a short reason. The model may not invent a code outside the candidates. A missing true class should produce review rather than a fabricated confident choice.

The optional extraction arm first extracts stated facts and unknowns with verbatim supporting quotations. It assigns no codes. Extracted fact values become the query for the same full-resource retriever. The selector sees the extraction and original narrative for verification. This arm requires two calls per case. It tests the effect of an evidence-structuring stage; it does not measure downstream reuse savings. Do not run it automatically if the primary arm and baseline already settle the decision.

JSON schemas validate output structure. Additional checks reject unavailable codes, a non-null review code or quotations absent from the source. Quote presence is necessary but does not establish that the quoted text entails the interpreted fact. Any invalid result remains a recorded failure. Provider requests, failures, settings and token costs are recorded by the shared runtime. There are no automatic retries. Labels are read for scoring after predictions are returned, never inserted into prompts or retrieval.

## Measurements and decision boundary

Report exact detailed-code accuracy for complete cases, correct review decisions, unsupported single-code decisions for review cases, candidate recall, paired corrections and regressions versus the full baseline, latency and all model token cost. Summarize by family, scheme and development/test split. Case variants within families are related; do not attach an independent-binomial population confidence interval to the pooled convenience sample. Official controls and synthetic scores remain separate.

Review-case compatible sets describe the alternatives explicitly left unresolved in the authored return. Review correctness primarily requires avoiding an unjustified single class; compatible-set retrieval is a separate diagnostic. Generic vague job titles are not assumed to admit exactly two classes.

Before model calls, freeze this development gate separately for each scheme: run the complete 16-case development panel, and proceed to its test families only if the direct model produces at least two net additional correct decisions over the baseline, without increasing unsupported single-code answers on the four review cases. The baseline has zero such review errors, so a passing model must also have zero. Otherwise stop that scheme and report its official controls and development results. Failed or invalid calls count as incorrect; they are not retried or removed. The gate is a spending decision for this pilot, not proof of operational reliability. A broader follow-up still requires an independently motivated residual workload. Source-label repairs found in the pre-model independent audit require an explicit changelog and new freeze.

## Reproduce and preserve

From the repository root, using the existing Python environment:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/coding.py --baseline --jobs --freeze --run-id initial
```

The command verifies pinned sources, runs only local computations, saves full baseline decisions and prepares model jobs without sending them. It refuses to overwrite different existing evidence. A new run ID is required for a revised protocol. `--live` is separate, requires a non-default run ID, and is only used after the independent audit clears the frozen labels. Shared runtime budget checks cover every attempted request.

The construction script `data/coding/build_cases.py` materializes the authored specifications; do not rerun it over an accepted dataset to hide a change. Freeze files record byte hashes of input narratives, reference labels, source manifests, construction code, runner, this protocol and the initial baseline snapshot. The `initial` freeze predates the subsequent development-gate clarification and safe exception reporting; `gated-v1` preserves that amendment without changing cases, labels or baseline decisions. No existing snapshot is overwritten by baseline or model execution.

## Attribution

Source: Statistics Canada, North American Industry Classification System (NAICS) Canada 2022 Version 1.0 and National Occupational Classification (NOC) 2021 Version 1.0, accessed 6 October 2026. Reproduced and distributed on an "as is" basis with the permission of Statistics Canada.

Adapted from Statistics Canada, North American Industry Classification System (NAICS) Canada 2022 Version 1.0 and National Occupational Classification (NOC) 2021 Version 1.0, accessed 6 October 2026. This does not constitute an endorsement by Statistics Canada of this product.

The official files are governed by the [Statistics Canada Open Licence](https://www.statcan.gc.ca/en/terms-conditions/open-licence). Synthetic case wording and analysis are additions to those sources. No confidential microdata or identifiable person/business records are used.
