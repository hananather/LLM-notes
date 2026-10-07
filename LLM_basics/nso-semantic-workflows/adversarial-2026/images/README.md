# Package image to catalog association

This study compares extraction and matching on 1,000 real grocery package images and 1,000 native catalog records from [Amazon Berkeley Objects (ABO)](https://registry.opendata.aws/amazon-berkeley-objects/). Each query uses the publisher's main image. The target is its originating catalog record, identified by the canonical item/domain relation. The source does not independently establish physical SKU identity.

All 1,096 selected development/test images downloaded successfully. OCR and model extraction receive the same prepared JPEG. No query was replaced because its image was unreadable, its OCR was empty, or a method failed. Test predictions are saved separately from evaluator truth. After the image-model development gate failed, only the frozen free test baselines were scored. Image-model test arms remain unrun.

## Data and identity boundary

The downloaded metadata contains 147,702 listing records and 145,615 distinct item IDs. The preparation rule finds 3,673 grocery items with an English or language-neutral name/brand and at least two globally item-exclusive image IDs. Requiring the canonical native `main_image_id` to be one of those images leaves **3,319 eligible items**. Canonical records prefer `amazon.com`, then domain and serialized record, and must satisfy eligibility themselves.

A main image's global item exclusivity is a source-association property. It does not establish that different item IDs represent different physical products. Punctuation/accent-normalized name+brand produces 211 repeated groups containing 444 IDs; the earlier case/whitespace-only screen found 203 groups containing 427 IDs. The frozen family rule also groups very similar names under the same canonical brand and numeric-token signature. Its 239 non-singleton components contain 511 IDs. These are input-similarity groups, not adjudicated product equivalence classes.

The sampler reserves whole family components to one split. Exact source IDs remain evaluator-only. Model-facing query and catalog IDs are separate opaque values, and row order is independently shuffled. Neither file paths nor source image/item IDs are matching features.

| View | Query rows | Catalog rows | Target associations |
|---|---:|---:|---:|
| Development source sample | 96 | 96 | 96 |
| Primary development comparison | 80 | Same 96 | 80 |
| Closed test | 1,000 | 1,000 | 1,000 |
| Designed NIL test | Same 1,000 | 1,000 | 800; 200 queries have no target |

The 80-query development subset was selected by fixed hash before paid predictions to fit a shared development budget. Its contract is `../contracts/image-development-subset.json`. Full 96-query offline outputs remain auxiliary. The development source sample has 83 families; test queries have 912. The NIL view removes whole selected-query families until exactly 200 queries have no target, then adds 200 prospectively selected catalog distractors from other families. Closed and NIL views are correlated evaluations of the same queries.

All 38 previously inspected IDs and their families are reserved for development. Three IDs have non-exclusive native main images and are outside the query frame: `B074MGW29M`, `B07RM3WJZN`, and `B07VZWS11S`. They remain reserved exclusions. The other 35 inspected IDs are among the 96 development queries.

## Reproduction and cache

Run from the repository root with its Python environment:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/images/prepare.py prepare
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/images/prepare.py download
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/images/baseline.py ocr --split dev80
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/images/baseline.py lexical --split dev80
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/images/baseline.py splink --split dev80
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/images/baseline.py clip --split dev80
```

The heavy cache defaults to `~/.cache/nso-adversarial-2026/abo/`; set `NSO_IMAGE_CACHE` to change it. Images and model weights are not repository artifacts. Measured image sizes are 334,189,016 original bytes and 505,939,391 prepared JPEG bytes. CLIP weights add approximately 605 MB. Metadata, embedding score matrices, fetch records and OCR caches also remain outside the repository. Native descriptive catalog fields, saved OCR, prediction rows, code and hash manifests are sufficient to inspect or replay the analysis with public downloads.

Preparation uses six download workers, a 45-second timeout and at most two attempts per URL. It never substitutes another image or item. A failed image remains in the frozen denominator. Existing fetch records are retained on restart, including recorded failures.

`prepare` preserves an existing selection freeze. To reconstruct selection independently, use a separate copy of this directory, retain the same protocol and compare the resulting model-facing data hashes with `evaluator/freeze.json`. Do not delete or overwrite the maintained study's frozen records. `prepare --source-dir PATH` can reuse previously downloaded metadata without changing selection.

### Exact image input

The prepared image is EXIF-transposed RGB, resized with LANCZOS while preserving aspect ratio, capped at a 2,048-pixel long side and `ceil(width/32) × ceil(height/32) ≤ 2,500`, then saved as JPEG quality 95. OCR and pixel extraction use this exact file. `data/input-manifest.jsonl` records its hash, dimensions, patch count, status and cache key. `baseline.resolve_image(query)` resolves that cache key. Originals remain available for audit.

The complete image screen found zero identical byte hashes or decoded-pixel hashes. Fifty pairs have pHash distance at most four, including five pairs crossing development and test. These are unadjudicated near-duplicate flags; the cutoff has false positives and false negatives. No flagged image was removed from the primary sample. Any later clean-stratum result must disclose its input-only rule and denominator. The existence of these flags prevents a claim that every test image is visually independent of development.

## Extraction interface

The bounded extraction object is defined in `schema.json`: brand, product name, variant, net quantity value/unit, pack count, model number, barcode and short supporting text. Unknown fields are null; supporting text is at most 240 characters. Quantity is explicit net contents, not nutrient amounts or a multiplication of approximate servings. A displayed jar count is not automatically a printed package-count claim.

Supply extraction results to a comparator as JSONL:

```json
{"record_id":"q_<opaque value>","status":"ok","fields":{"brand":null,"product_name":null,"variant":null,"net_quantity_value":null,"net_quantity_unit":null,"pack_count":null,"model_number":null,"barcode":null,"evidence_text":""}}
```

The opaque ID joins saved outputs; it is not part of the extraction object or package evidence. An `ok` object with no evidence becomes review. A missing/failed extraction becomes failed. The parser also accepts `extracted` or `output` as the object wrapper. No gold-derived barcode dictionary is built: native catalog fields in this study do not supply a verified barcode counterpart.

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/images/baseline.py lexical --split dev80 --extracted path/to/rows.jsonl --tag model-ocr
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/images/baseline.py splink --split dev80 --extracted path/to/rows.jsonl --tag model-ocr
```

The same commands support pixel-derived extraction. Both use the unchanged catalog, candidates, comparison rules, numerical prior and fitting validity criteria. A model extractor changes the evidence fields; it does not receive catalog identity or evaluator labels.

## Offline comparators

| Comparator | Input and fixed rule |
|---|---|
| Tesseract plus lexical | English Tesseract 5.5.3, PSM 11 then 6; ordered unique nonempty lines. Weighted title-token coverage, character/word TF-IDF, brand, quantity/count and model agreement. TF-IDF fits catalog title text only. |
| Tesseract plus Splink | Same ordinary extracted features; all query/catalog pairs; unlabeled u estimation and one global EM fit with fixed pair prior. Exact posterior ties use the frozen lexical score, followed by opaque catalog ID. |
| Supplied extraction plus lexical/Splink | The same matching code with supplied bounded fields. Empty evidence remains review or failed according to extraction status. |
| Frozen CLIP | Prepared query pixels and native catalog title/brand text. No catalog-side images. Cosine ranking from a pinned pretrained image/text encoder. |

The lexical score uses 0.35 IDF-weighted title-token coverage, 0.30 character cosine, 0.15 word cosine, 0.10 brand agreement, 0.06 quantity/count agreement and 0.04 model agreement. Its normalizers are deterministic. OCR quantities prefer explicit net-weight lines; other OCR quantities are only candidate evidence. Catalog quantity comes from title text, not the generic publisher `item_weight` field. This ordinary extraction is itself an imperfect baseline and its fields are saved.

Splink uses token-containment levels 0.9/0.7/0.4, exact brand, quantity agreement within 5%, and exact model where supported. Quantity/model comparisons require at least ten nonmissing records on each side. u is estimated from the available cross-pair sample with a fixed seed, without labels. Each main fit uses all pairs and fixes the designed prior: `1/catalog_size` for closed views, `0.8/catalog_size` for the designed NIL view. This prior follows the constructed benchmark overlap; it is not an estimated NSO population prevalence.

Each fit has a fixed 200-iteration cap. A valid fit must meet a final parameter change of at most `1e-4`, and every used m/u level must be strictly positive, finite and sum to one within `1e-6`. Invalid fits retain diagnostics and emit failed decisions. An initial blocked development fit and a 40-iteration predecessor remain as diagnostic artifacts. The repair was triggered by probability normalization and convergence defects; test outcomes remained unopened. Valid primary development fitting converged in 73 iterations. Closed and NIL test fitting converged in 50 and 39 iterations. These numerical checks do not establish calibrated probabilities or conditional independence.

Posterior ties are common: 631 of 1,000 closed-test top scores and 649 of 1,000 NIL top scores are tied before lexical resolution. This limits interpretation of an apparent advantage over Splink alone. Strong lexical and visual comparators remain necessary.

Prior sensitivity holds fitted likelihood ratios fixed and reweights posterior odds to `[0.25, 0.5, 0.8, 1.0]/catalog_size`. Saved sensitivity decisions include thresholds 0.8/0.9/0.95/0.99. They do not refit EM or select a threshold from test labels; primary scores and thresholds remain fixed.

CLIP is `openai/clip-vit-base-patch32` at revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. It uses 512-dimensional embeddings, a 224-pixel model-specific image crop and a 77-token text limit. The text template is `A product photo of {title}. Brand: {brand}.` The largest catalog inputs were 46 tokens closed and 48 NIL, so no title was truncated in these runs. The model-specific crop differs from the full prepared image available to OCR and pixel extraction. Its pretraining may include product imagery; absence of training overlap is not established. CLIP is a frozen representation comparator, not ground truth.

## Decisions, scoring and frozen policies

Every prediction has `status`/`decision_status` and `target_ids`:

- `linked`: a valid automatic catalog assignment.
- `nil`: valid evidence falls below the frozen NIL score threshold.
- `review`: evidence is successfully extracted but empty, or a sufficiently high score has an insufficient margin.
- `failed`: missing/failed input, empty OCR, failed extraction or invalid numerical fit.

In the closed view, evidence-bearing queries rank the full catalog and select top one. In NIL, lexical uses score 0.30 and margin 0.03; CLIP uses cosine 0.25 and margin 0.02; Splink uses posterior 0.95 with no additional margin. Scores are not mutually calibrated across methods. Review and failure never count as correct NIL or as automatic decisions.

`../evaluation.py` supplies common complete-decision metrics. `results/common-dev80-decisions.json` contains ready-to-score decisions; `common-dev80-baselines.json` contains the shared summaries. The primary 80-query development results are lexical **61/80**, Splink **46/80**, and CLIP **63/80**, each with 80 automatic decisions. This is a development comparison against 96 candidates, not a test result or evidence of deployment safety.

`evaluate.py` also reports ranking recall and a truncated MRR@20 lower bound, source-family strata and family-cluster bootstrap intervals. Its test scoring requires `--allow-test-outcomes`; use that switch only after all methods and experiment policies are frozen. Ranking metrics cannot replace complete-decision accuracy or failure accounting. The seven test queries with empty/failed OCR stay in the denominator.

`policy-freeze.json` binds implementation, schema, protocol and evaluator hashes. `evaluator/freeze.json` binds selected records and public-source hashes. Amendments retain technical reasons and predecessor hashes. `check_contracts.py` verifies empty OCR, invalid fit, low-margin review and all-null extraction against NIL truth using synthetic fixtures, without reading real test outcomes.

Test execution commands use `--split test` and optionally `--view nil`. All candidates are scored; top-20 outputs and full score matrices are saved, with matrices in the external cache. Do not rerun preparation or alter thresholds after examining test results.

## Observed results and stopping decision

Both primary model-extraction/Splink arms failed the prospective development spending gate. The two lexical normalization comparisons also failed. No image-model test calls were made, and no retry, output salvage, prompt repair or threshold change followed these outcomes.

| Development method | Correct complete decisions / 80 | Automatic decisions | False links | Review | Failed |
|---|---:|---:|---:|---:|---:|
| Ordinary OCR → lexical | 61 | 80 | 19 | 0 | 0 |
| Ordinary OCR → Splink | 46 | 80 | 34 | 0 | 0 |
| Frozen CLIP | 63 | 80 | 17 | 0 | 0 |
| Model on OCR text → lexical | 58 | 74 | 16 | 5 | 1 |
| Model on OCR text → Splink | 55 | 74 | 19 | 5 | 1 |
| Model on pixels → lexical | 63 | 71 | 8 | 2 | 7 |
| Model on pixels → Splink | 58 | 71 | 13 | 2 | 7 |

The strongest conventional development reference was CLIP. The gate required at least four additional correct decisions, no increase in false links or false-assignment queries, and at most four fewer automatic decisions. The pixel/lexical pipeline equalled CLIP's correct count while making nine fewer automatic decisions. Its lower false-link count came with lower coverage; it did not increase complete-decision accuracy or establish production value.

There were 160 extraction requests. The frozen parser accepted 152 outputs and rejected eight: six schema-validation failures caused by extra output keys and two length-terminated invalid outputs. Responses were received for all eight; they are not established HTTP failures or direct evidence of poor text comprehension. OCR-text extraction had one failure and pixel extraction seven. The extraction objects were not repaired. Reported nominal token-price usage was $0.0694894 against a $0.3180990625 reservation; these values describe this development run, not a deployed-system cost estimate.

Model extraction improved over the ordinary OCR/Splink pipeline: the OCR-text/Splink arm had 14 corrections and five regressions, while pixel/Splink had 17 corrections and five regressions. Against CLIP, those arms had five corrections/13 regressions and seven corrections/12 regressions. The comparisons resample the same 71 development source families. Full paired intervals, common decisions and gate conditions are in `results/model-development-v1/comparison.json`. This supports a bounded extraction/matcher interaction observation; it does not support replacing the strongest conventional workflow.

After gate closure, the already frozen free test predictions were scored on all 1,000 queries:

| Test view | Method | Correct complete decisions / 1,000 | Automatic decisions | Review | Failed | Family-bootstrap 95% interval for accuracy |
|---|---|---:|---:|---:|---:|---|
| Closed | OCR → lexical | 545 | 993 | 0 | 7 | 51.2–57.6% |
| Closed | OCR → Splink | 483 | 993 | 0 | 7 | 45.1–51.4% |
| Closed | CLIP | 603 | 1,000 | 0 | 0 | 57.1–63.6% |
| Designed NIL | OCR → lexical | 420 | 772 | 221 | 7 | 38.9–45.3% |
| Designed NIL | OCR → Splink | 276 | 993 | 0 | 7 | 24.7–30.7% |
| Designed NIL | CLIP | 260 | 312 | 688 | 0 | 23.2–28.7% |

The NIL view contains 200 true NIL queries. At the frozen thresholds, false NIL decisions were 231 for lexical, 690 for Splink and one for CLIP. CLIP instead sent 688 queries to review. These outcomes expose different threshold/coverage limitations; thresholds were not selected again from these results. Closed and NIL tables share the same query images and must not be treated as independent replications.

The pHash audit flagged five test queries through possible cross-split near-duplicate pairs. Closed-view correct counts on those five were 4/5 lexical, 4/5 Splink and 3/5 CLIP. Counts on the 995 unflagged queries were 541, 479 and 600. Seventy-one test queries participate in any pHash flag. `results/free-test-summary.json` retains all flagged/unflagged strata, source-family intervals, source-similarity strata and paired comparisons; no primary record was dropped. The five-query stratum is too small to establish a stable performance claim.

The image-model direction stops at this development gate. A later study needs a separately specified operational question and independently justified source/identity evidence, with fresh evaluation data. It should not continue adapting to this failed development/test selection. The benchmark remains useful as a reproducible native-catalog association experiment and as evidence about missing image text, extraction failure, matching and abstention.

## Evidence and environment

- `data/`: opaque queries, descriptive catalogs, prepared-image manifest.
- `evaluator/`: source identity, gold association, source-file hashes, source/family maps, ambiguity panels, selection and duplicate/fetch evidence.
- `results/`: OCR, derived features, ranked predictions, explicit decisions, fit/CLIP diagnostics, prior sensitivity, development summaries and contract fixtures.
- `source-notices-manifest.json`: public documentation URLs and downloaded hashes.

The implementation was exercised on macOS with Python 3.11, Tesseract 5.5.3, torch 2.14.1, transformers 4.57.6, scikit-learn 1.9.1, Pillow 12.3.0, Splink 5.0.0 and DuckDB 1.5.6. CLIP used MPS; its code falls back to CPU. Other platforms have not been validated by this run. No `open_clip` dependency is required. Public image/data rights and model notices are recorded in [NOTICES.md](NOTICES.md).
