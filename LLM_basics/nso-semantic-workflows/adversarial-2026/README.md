# Native text, package images and unlabeled record linkage

Semantic product selection improved complete decisions in development while adding false links. Recordwise extraction and package-image extraction did not meet the rule for a larger paid evaluation. This folder preserves those outcomes, the conventional comparisons, the source data and the explicit amendment allowing one unchanged product holdout run.

The reader-facing report is the [single NSO notebook](../../nso-semantic-workflows.ipynb). These files provide its reproducible evidence. They extend the earlier affiliation, NAICS, NOC, school and statistical-query experiments without replacing their results.

## Why these tasks matter

Statistics Canada uses retail scanner prices, sales and volumes in the Consumer Price Index and Monthly Retail Trade Survey. Product association can therefore matter when integrating alternative price sources. This study tests historical catalog identity; it does not estimate CPI effects or decide whether two products are suitable replacements. [Statistics Canada source description](https://www.statcan.gc.ca/en/our-data/data-sources/prices-price-indexes)

Statistics Netherlands linked company websites to its business register when studying AI-producing companies. That motivates source-to-register association, but does not make research-affiliation labels equivalent to legal-unit identities. [CBS methods](https://www.cbs.nl/en-gb/longread/aanvullende-statistische-diensten/2025/ai-monitor-2024/3-ai-producing-companies)

| Lane | Reference relation and scale | Evidence available |
|---|---|---|
| [Product text](products/README.md) | Amazon–Google publisher product mapping; 144 × 352 development, 1,219 × 2,874 holdout | Conventional all-pairs and candidate-restricted comparisons; two development model arms; one amended selector holdout |
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

The [dated amendment](contracts/product-tradeoff-amendment-v1.json) allows one unchanged product selector holdout to characterize the large development tradeoff. It explicitly overrides the original spending stop after development. The endpoints are not newly discovered questions, and the amended run is not a gate pass. Prompts, candidates, thresholds and failure rules remain fixed. Predictions must be sealed before outcome scoring. No extraction or image model holdout is authorized by this amendment.

On the 80 image development queries, CLIP made 63 correct decisions out of 80 automatic decisions. Pixel extraction followed by lexical matching also made 63, with 71 automatic decisions; seven schema/completion failures and two reviews reduced coverage. That is not an improvement in complete-decision accuracy. On 1,000 held-out images, CLIP made 603 correct decisions, OCR plus lexical matching 545, and OCR plus Splink 483. No image-model holdout result exists. The designed missing-counterpart view and similarity-flagged strata remain in the [complete free results](images/results/free-test-summary.json).

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

The development round made 800 calls: 763 met the runtime response contract and 37 failed. Four further product extraction responses failed literal-evidence checks. Failed outputs are retained, not retried or repaired. Reported token usage gives an estimated development price of US$0.1491891; the conservative reservation was US$0.985343125. Estimates are not invoices and exclude local compute and labour. The amendment permits a further reservation of at most US$0.964. Earlier experiments reserved US$0.930303125 and remain included in the shared ceiling.

The [append-only request ledger](results/api-ledger.jsonl) and frozen request manifests preserve each admitted attempt. Asynchronous transport reserves the whole batch before submission, joins out-of-order results by request identifier and keeps expired or missing responses in the denominator. It never substitutes a synchronous call or retries an unknown submission.

Heavy ABO rasters, model weights and materialized Batch input files are external caches, not repository artifacts. Published source records, OCR, contracts, predictions and hash manifests support inspection and reconstruction. Earlier numerical diagnostics remain distinguishable from the final valid Splink fits; they must not be substituted as outcome comparators.
