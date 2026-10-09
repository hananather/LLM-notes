# Native text, package images and unlabeled record linkage

Semantic product selection recovered more complete reference sets than the fixed primary lexical rule in development and in the amended holdout, while adding false links. Recordwise extraction and package-image extraction did not meet the rule for a larger paid evaluation. This folder preserves those outcomes, the conventional comparisons, the source data and the explicit post-development amendment used for one unchanged product selector holdout.

The reader-facing report is the [complete report notebook](../../record-linkage-with-semantic-operators.ipynb). These files provide its reproducible evidence. They extend the earlier affiliation, NAICS, NOC, school and statistical-query experiments without replacing their results.

The complete report shows [selected input/decision examples](../../record-linkage-with-semantic-operators.ipynb#dataset-showcase). The earlier product, image and control gallery remains in [the source-linked showcase file](dataset-showcase.json) and its adjacent assets. These illustrations explain the data; the complete evaluations supply the performance estimates.

## Why these tasks matter

Statistics Canada uses retail scanner prices, sales and volumes in the Consumer Price Index and Monthly Retail Trade Survey. Product association can therefore matter when integrating alternative price sources. This study tests historical catalog identity; it does not estimate CPI effects or decide whether two products are suitable replacements. [Statistics Canada source description](https://www.statcan.gc.ca/en/our-data/data-sources/prices-price-indexes)

Statistics Netherlands linked company websites to its business register when studying AI-producing companies. That motivates source-to-register association, but does not make research-affiliation labels equivalent to legal-unit identities. [CBS methods](https://www.cbs.nl/en-gb/longread/aanvullende-statistische-diensten/2025/ai-monitor-2024/3-ai-producing-companies)

| Lane | Reference relation and scale | Evidence available |
|---|---|---|
| [Product text](products/README.md) | Amazon–Google publisher product mapping; 144 × 352 development, 1,219 × 2,874 holdout | Conventional all-pairs and candidate-restricted comparisons; two development model arms; completed amended selector holdout |
| [Package images](images/README.md) | ABO native main image to catalog item; 80 paid development queries against 96 catalog records; 1,000 held-out queries against 1,000 catalog records | OCR, Splink and CLIP; paired text/image extraction in development; conventional holdout results only |
| [Organizations](organizations/README.md) | AffilGood CORDIS annotations against all 108,476 historical ROR records; 109 development and 1,065 reserved evaluation queries | Conventional development and sealed holdout results; no paid semantic arm |
| [Document controls](controls/README.md) | Literal product facts rendered in four forms; 16 development and 64 holdout source records | 320 synthetic images and real OCR; prepared controls, not a measured linkage comparison |
| Abt–Buy | Complete publisher graph: 1,081 × 1,092 records | Frozen conventional results and numerical checks; no model result |

These public proxy datasets have different identity definitions. They are not samples of confidential NSO production records. Public benchmark exposure during model pretraining is unknown. Each source's licence, version and transformations are documented in its linked folder.

## Development results and the stop-rule amendment

All 144 product queries remain in each denominator. A correct complete decision returns the entire reference set, or a valid no-match decision when that set is empty. Review and failed outputs are unresolved; neither counts as a correct no-match decision.

| Product method | Correct complete decisions | False edges | Missed edges | Automatic decisions |
|---|---:|---:|---:|---:|
| Lexical, fixed primary rule | 56 | 8 | 99 | 144 |
| Regularized unlabeled Splink, fixed primary prior and rule | 21 | 0 | 143 | 144 |
| Recordwise extraction + fresh Splink | 27 | 1 | 136 | 141 |
| Semantic candidate selection | 112 | 20 | 18 | 136 |

The product methods and both image extraction pipelines failed the [original continuation rule](protocol.json). The [product development gates](products/results/model-development-v1/development-gates.json) remain unchanged. The primary Splink prior is an assumption derived from a strict observed agreement rule; the full prior grid must accompany its result. On broader fixed prior assumptions, the extraction gain disappears. This is not evidence of a general failure of Fellegi–Sunter linkage.

The [dated amendment](contracts/product-tradeoff-amendment-v1.json) allowed one unchanged product selector holdout to characterize the development tradeoff. It explicitly overrode the original spending stop after development. The endpoints are not newly discovered questions, and the amended run is not a gate pass. Prompts, candidates, thresholds and failure rules stayed fixed. The complete predictions were sealed before outcome scoring. No extraction or image model holdout was authorized by this amendment.

The [access chronology](../../record-linkage-with-semantic-operators.ipynb#product-access-chronology) records limited reference-derived metadata exposure after request freeze and before prediction sealing.

On the 80 image development queries, CLIP made 63 correct decisions out of 80 automatic decisions. Pixel extraction followed by lexical matching also made 63, with 71 automatic decisions; seven schema/completion failures and two reviews reduced coverage. That is not an improvement in complete-decision accuracy. On 1,000 held-out images, CLIP made 603 correct decisions, OCR plus lexical matching 545, and OCR plus Splink 483. No image-model holdout result exists. The designed missing-counterpart view and similarity-flagged strata remain in the [complete free results](images/results/free-test-summary.json).

## Completed product holdout

The provider returned all 1,219 attempted queries. The [prediction seal](results/product-holdout-v1/prediction-seal.json) was written at 2026-10-08 05:37:56 UTC, and the frozen evaluator checked it before opening reference identities. The [completed evaluation](products/results/model-evaluation-v1/completion.json) retains every query across 916 observed source families.

| Fixed primary method | Correct complete decisions / 1,219 | False links | Missed links | Automatic decisions | Review / failed |
|---|---:|---:|---:|---:|---:|
| Lexical, threshold 0.65 | 382 (31.3%) | 264 | 788 | 1,219 | 0 / 0 |
| Regularized unlabeled Splink, primary prior and threshold 0.9 | 275 (22.6%) | 45 | 1,082 | 1,219 | 0 / 0 |
| Semantic candidate selection | 702 (57.6%) | 508 | 151 | 1,125 | 68 / 26 |

The lexical rule remains the development-selected reference. The two conventional primary rules give identical results over all pairs and over the selector's candidates. The selector has 94 unresolved decisions and 92.3% automatic coverage; 423 of its 1,125 automatic decisions are wrong (37.6%). Reviews and failures do not count as correct no-match decisions, and their positive reference links remain missed.

The [product evidence](products/README.md#scored-comparisons) links all 64 fixed conventional policies, paired family intervals, reference-cardinality groups, retrieval availability and the complete query audit. Its hypothetical edge-cost grid describes the false-link/missed-link tradeoff without estimating staff time, ROI or an operational winner. The original development gates remain failed, and the amendment ends after this one run.

## Reproduction and cost

Recorded execution used Python 3.11.15 on macOS. Product fitting used Splink 5.0.0 and DuckDB 1.5.6. Image preparation used Pillow 12.3 and Tesseract 5.5.3; its additional dependencies and pinned CLIP model are in the [image instructions](images/README.md). Synthetic rendering uses the documented macOS font files. No other platform has been verified for this revision.

From the repository root, replay the notebook from saved evidence:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/run_notebook.py
```

Check frozen scientific contracts and request accounting without making provider requests:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/verify_replay.py
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/check_runtime.py
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/check_batch_runtime.py
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/check_scoring.py
```

The [product replay command](products/README.md#scored-comparisons) verifies the seal and saved evaluation hashes without making provider requests.

The development round made 800 calls: 763 met the runtime response contract and 37 failed. Four further product extraction responses failed literal-evidence checks. Failed outputs are retained, not retried or repaired. Reported token usage gives an estimated development price of US$0.1491891; the conservative reservation was US$0.985343125. The completed product holdout reserved US$0.963497421875, within the amendment's US$0.964 cap; its nominal token-price estimate is US$0.119846250, with no unpriced results. All 1,219 returned model identifiers are `gpt-6-luna`. Estimates are not invoices and exclude local compute and labour. Earlier experiments reserved US$0.930303125 and remain included in the shared ceiling.

The [append-only request ledger](results/api-ledger.jsonl) and frozen request manifests preserve each admitted attempt. Asynchronous transport reserves the whole batch before submission, joins out-of-order results by request identifier and keeps expired or missing responses in the denominator. It never substitutes a synchronous call or retries an unknown submission.

Heavy ABO rasters, model weights and materialized Batch input files are external caches, not repository artifacts. Published source records, OCR, contracts, predictions and hash manifests support inspection and reconstruction. Earlier numerical diagnostics remain distinguishable from the final valid Splink fits; they must not be substituted as outcome comparators.
