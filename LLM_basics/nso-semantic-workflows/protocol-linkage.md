# Public identity-linkage studies

This protocol tests historical affiliation-to-organization matching and screens school-directory continuity for possible model value. These are separate populations and relations. Neither is an evaluation on confidential national statistical office records.

The protocol, input selection, reference labels and model contract were saved before any model calls. Baseline decisions are saved before test metrics are computed. Current commands make no paid calls.

## Reproduce preparation and conventional baselines

From the repository root, using the environment described by the main workflow guide:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/linkage.py prepare --download
.venv/bin/python LLM_basics/nso-semantic-workflows/linkage.py baseline --lane s2aff
.venv/bin/python LLM_basics/nso-semantic-workflows/linkage.py baseline --lane nces
.venv/bin/python LLM_basics/nso-semantic-workflows/linkage.py verify
```

Preparation verifies pinned SHA-256 hashes. Downloads go under `cache/linkage/` beside the runner and stay out of Git. After the cache is populated, preparation and baseline rebuilding need no network. `--import-cache DIRECTORY` can seed the cache from an existing download directory; it can be repeated. The runner uses NumPy, Pandas, SciPy and scikit-learn, with no NER model or embedding download.

Prepared artifacts are immutable: an existing file with different content raises an error. A completed baseline command returns its saved result; it does not silently overwrite it. A new methodological experiment needs new artifact names and a fresh protocol. Default replay reads compact data and results; rebuilding the affiliation index requires the complete pinned registry.

## Affiliation linkage

### Population and reference labels

The source is S2AFF's [manual affiliation annotations at commit ac8d6b5](https://raw.githubusercontent.com/allenai/S2AFF/ac8d6b58f42b253821c26fff8b013913317c5c72/data/gold_affiliation_annotations.csv): 1,132 training, 588 validation and 644 test rows. Keep all **644 official test rows** and their labels, in source order. Development uses 100 validation rows selected by the smallest SHA-256 values of the fixed seed and original row index. The remaining training/validation data remain available but are not used by these lexical baselines.

Truth is the set of ROR URLs in each annotation cell. Other label values are organization-name strings, not a standardized NIL token. Preserve that text separately and score those rows as an empty ROR target set under the published historical annotation policy. The test includes 38 such rows and 16 rows with multiple ROR targets. Empty output may reflect either historical no-target truth or retrieval failure; report both errors through independent labels.

The registry is [ROR v1.1, 16 June 2022](https://zenodo.org/records/6657125), the snapshot named by the [same S2AFF source revision](https://raw.githubusercontent.com/allenai/S2AFF/ac8d6b58f42b253821c26fff8b013913317c5c72/s2aff/consts.py). It contains **102,742 organizations**. All 286 distinct test gold ROR IDs are present. Registry membership does not independently adjudicate the labels.

`data/linkage/sources.json` pins every source URL, byte count and hash. The downloaded ZIP's MD5 also matches the publisher's `045f3edcb72e323a742e88bf81361a26`. The full source annotations and registry stay in the cache. Compact evaluation rows, independent labels, candidate records and predictions support replay.

This is a historical research-organization task. It does not measure current ROR coverage, legal-unit continuity, establishment linkage or accuracy on an NSO register. A current-registry extension needs separate treatment of new organizations, historical names, mergers, status and no-target drift. Public benchmark exposure by pretrained models is possible. Duplicate text and training-seen organization strata are recorded; the supplied split does not guarantee unseen organizations.

### Conventional alternatives

Every registry organization participates in retrieval. The code never constructs a candidate universe from gold institutions or inserts a true target.

1. Normalize names, aliases, acronyms and multilingual labels. Search full-registry alias character TF-IDF and organization word TF-IDF separately. Character grams have lengths 3–5; word grams have lengths 1–2. Each vocabulary is capped at 300,000 features.
2. Union the top 100 results from each signal with contained alias matches. Country and city matches contribute a weak reranking signal on this pool, rather than a hard geographic block.
3. Rank by `0.60 × character + 0.30 × word + 0.10 × location + 1.0 × contained-alias`. Keep 25 candidates per affiliation, using ROR ID as a deterministic tie break.
4. Report exact contained-alias sets, character top-1, word top-1, fused top-1 and fused selective sets. Select the selective-set threshold/gap only on the declared 100 validation cases. Candidate recall is reported before semantic selection.

The vocabulary is learned from registry text without labels. Selective decision thresholds use validation labels, so that complete procedure is not entirely label-free. The baseline is not the trained S2AFF NER/LightGBM pipeline. That package and its large model artifacts are absent from this environment; no comparison against it is claimed. There was no tuning against test errors.

### Frozen semantic selection

`data/linkage/model-contract.json` contains the system text and JSON schema. `data/linkage/s2aff-model-jobs.json` contains 100 validation and 644 test jobs. Each job receives source affiliation text, the historical registry version and the same top-25 candidate records. The fields contain no query labels. Candidate ROR IDs are legitimate public output choices.

The model must return a set of supplied IDs, exact source spans supporting each choice and a review flag. It may return an empty set. It may not invent an ID, infer a parent solely from a relationship, or use subject similarity as identity evidence. A span check verifies that quoted evidence occurs in the source; it does not itself prove correct identity. This is a custom blocked semantic selector when run through the shared API runtime, not a measured native LOTUS optimizer.

`model_jobs(split)` supplies frozen jobs without reading reference labels. `validate_model_output(job, answer)` checks IDs and evidence. `score_model_answers(answers, split)` scores every frozen case; absent/invalid outputs remain failures, including on no-target rows. The shared runtime owns model, token, retry and aggregate-budget controls. There is no live API command in this runner.

Report exact target-set accuracy, micro precision/recall/F1, false NIL, false assignment on NIL, review rate, accepted coverage, invalid responses and token/latency cost. Report target-level and all-targets-per-row recall at 1, 5, 10 and 25. A model cannot repair targets excluded by retrieval. Top-1 accuracy alone is incomplete for multi-target rows.

The model continuation gate is fixed before calls in `s2aff-development-gate.json`. The conventional selective-set result on the 100 validation rows has 74 exact sets and 22 false target assignments. Run the full 100 model validation jobs once. Continue to the unchanged 644-row official test only with **at least 79 exact sets and at most 22 false-assignment penalties**. Each emitted false target counts once; each failed response adds one separate conservative gate penalty. Report emitted false assignments and failed responses separately. This preserves the original gate without describing failed calls as invented organization assignments. Missing or invalid responses remain incorrect even on no-target cases; review flags do not remove cases from this gate. Do not revise prompts from these outcomes or run only favorable test strata. If the gate fails, stop the model lane and retain the development result. `development_gate(score)` applies this rule.

The recorded full-registry conventional test result is 456/644 exact sets (70.8%) for selective fusion. Top-25 candidates retain 614/623 annotated targets (98.6%) and all targets for 597/606 positive rows (98.5%). These figures describe the implemented lexical procedure, not the trained S2AFF system. They were not used to change the frozen candidate contract or prompt.

## NCES school-continuity feasibility screen

### Sources and administrative target

Sources are the [2021–22 preliminary directory, updated release](https://nces.ed.gov/ccd/Data/zip/ccd_sch_029_2122_w_0b_042922.zip), with 102,067 rows, and [2022–23 preliminary directory](https://nces.ed.gov/ccd/Data/zip/ccd_sch_029_2223_w_0a_051023.zip), with 102,229 rows. There are 101,149 shared unique NCESSCH values. Source ZIPs and year-specific documentation hashes are in the manifest.

The target is the **same continuing administrative school**, conditional on a shared unique identifier. Restrict source queries to schools with `SY_STATUS=1`, `UPDATED_STATUS=1` and `RECON_STATUS=No` in both years: 98,055 schools. The year-specific companion workbooks identify status 1 as Open and distinguish Closed, New, Added, Changed Boundary/Agency, Inactive, Future and Reopened. They define `RECON_STATUS` as a reconstituted flag. These fields were read directly from the downloaded year documentation.

This restriction avoids asserting physical-campus continuity across closures, reorganizations or changed identifiers. It does not establish that every unchanged administrative ID represents an unchanged institution. Those cases require a different labeling policy. Status distributions and excluded populations remain in `nces-selection.json`.

The target candidate universe is **all 102,229 newer-directory rows**, including records outside the query population. NCESSCH, SCHID, ST_SCHID, LEAID and derivative identifiers never enter scoring features. Source keys stay in the ignored cache; saved gold uses unrelated per-year opaque IDs. Matching uses school name, LEA name, address, city/state/ZIP, phone and grades. Displayed school type/status/charter fields preserve context.

### Samples and estimands

Use seed `2026100607`. Draw 100 development schools. Their 99 LEAs are excluded from the evaluation frame, leaving **94,586 schools**. Draw an independent simple random sample of 300 from this conditional frame, with inclusion probability `300/94586`. This is the probability-sample result; do not call it a national-population estimate.

Before scoring, define a changed-field stratum by a difference in both normalized school name and normalized first location street line. There are **161 such schools** in the evaluation frame. A separate sample contains 100, with inclusion probability `100/161`. It happens to overlap the 300-school sample in zero cases. Its result describes the enriched stratum and must not be pooled with the probability sample without a correct design analysis.

The conventional scorer uses address abbreviation normalization, exact name/state, name/city, name/LEA, address/ZIP and phone indexes, plus full-universe character TF-IDF. It unions candidate routes and applies frozen name/address/LEA/location/phone/grade weights. It does not fit on school truth. It saves predictions before checking hidden administrative labels.

### Prespecified stop and observed result

Stop paid model work in this lane if conventional top-1 accuracy is at least 99% in the 300-school probability sample. This is a resource-allocation gate, not a rare-error guarantee.

The recorded conventional result is **300/300 correct** in that sample, with 300/300 true targets in the top ten. Development is 100/100. The separate both-changed sample is **72/100 correct at top one and 96/100 at top ten**. The gate therefore stops model work for the whole lane this round. Do not use the enriched sample for a post-hoc paid model experiment under this protocol.

The negative result is that the defined ordinary continuing-school sample leaves no observed top-1 improvement for an LLM. The enriched result shows that this conclusion does not extend to every changed-field case. Three hundred successes do not prove perfect population performance. No human review time or operational cost saving was measured.

## Reuse terms and audit artifacts

S2AFF's exact-commit Apache-2.0 notice is retained as `data/linkage/S2AFF-LICENSE.txt`. [ROR documentation](https://ror.readme.io/docs/data-dump) assigns CC0 to IDs and registry metadata; preserve applicable GeoNames attribution when using locations. Keep original source references with extracted material.

NCES publishes the directories and their companion documentation as public-use data on the [official CCD page](https://nces.ed.gov/ccd/psu_rev.asp). Compact records contain public institutional directory fields, not private student/person records. The cached year-specific documentation is preserved with source hashes. The [NCES geographic service](https://nces.ed.gov/arcgis/rest/services/CCD/CCD_Data/MapServer/layers) labels its own information public domain; it is not substituted for the provenance of these exact CSV releases. No affiliation or endorsement is implied.

`prepared-manifest.json` binds input selection, labels and contract. Baseline results record source/code hashes and elapsed time. Candidate documents are a compact **replay subset after full-registry retrieval**, not the retrieval universe. No held-out errors were used to change model prompts, lexical rules or sample selection. Source and checksum verification can be repeated without model calls.
