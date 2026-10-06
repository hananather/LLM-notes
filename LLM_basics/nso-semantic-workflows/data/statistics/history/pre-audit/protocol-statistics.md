# Statistical discovery and exact-query protocol

This experiment measures whether a language model can select a statistical source and translate a request into the correct filters and calculation. It compares a conventional parser, a single model-generated plan and a bounded tool controller. Source discovery and numerical planning are scored separately.

The data are real public Statistics Canada observations. The requests are authored scenarios, not a sample of actual analyst tickets. The question author also implemented the rule baseline; this is a reproducible engineering pilot, not an externally blinded benchmark. No evaluated model generated the requests or reference answers. Test predictions do not select examples, rules or prompts.

## Frozen public data

The source vintage was downloaded on 6 October 2026 through Statistics Canada's [Web Data Service](https://www.statcan.gc.ca/en/developers/wds). The full catalog contains **8,270 table records**. The numerical corpus retains **every observation whose reference year is 2023, 2024 or 2025** from four tables:

| Table | Retained rows | Source series in retained rows | Compressed local file |
|---|---:|---:|---:|
| [14-10-0287-01, monthly labour force characteristics](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410028701) | 323,676 | 8,991 | 2,541,552 bytes |
| [18-10-0004-01, monthly CPI, not seasonally adjusted](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000401) | 71,905 | 2,006 | 426,401 bytes |
| [18-10-0005-01, annual average CPI, not seasonally adjusted](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000501) | 5,967 | 2,005 | 69,927 bytes |
| [18-10-0006-01, monthly CPI, seasonally adjusted](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000601) | 396 | 11 | 12,486 bytes |
| Total | **401,944** | — | **3,050,366 bytes** |

The [source manifest](data/statistics/manifest.json) records original ZIP URLs, hashes, byte counts, full-file row counts, extracted-file hashes, date, status counts and licence. It preserves all 56 suppressed LFS observations. No rows were selected according to model success. Files use the [Statistics Canada Open Licence](https://www.statcan.gc.ca/en/terms-conditions/open-licence); retain attribution to Statistics Canada and distinguish derived experimental results from official statistics.

Source ZIPs are a local cache under `cache/statistics/`. The shared Parquet files allow offline replay. If extracted files are absent, `prepare` can download the four original ZIPs; a changed current source must not silently replace the frozen corpus. Each observation has a stable certificate key `PID:VECTOR:REF_DATE`. CSV dimensions, units, scales, status and precision remain available.

The LFS table contains several adjustment and uncertainty variants despite its title. A complete query must specify Geography, Labour force characteristic, Gender, Age group, Statistics and Data type. CPI tables have different geographic and product coverage. Numeric coordinate codes are never assumed equivalent across tables.

## Requests and reference answers

[Questions](data/statistics/questions.json), [reference plans and certificates](data/statistics/reference.json), and [prompts](data/statistics/prompts.json) are frozen by [the evaluation lock](data/statistics/evaluation-lock.json). There are **12 development requests** and **32 test requests**. Nine development and 24 test requests have numerical answers; the others require review because the requested definition, geography, period, population or aggregation is unsupported or unresolved.

The development set includes Canada/Ontario July–August 2024 unemployment examples and Canada's December 2023–December 2024 CPI change inspected during source research. They are excluded from test scoring. Test scenario wording was authored separately before any predictions; no random sampling of real requests or independent human authorship is claimed. Some operations and statistical concepts recur across splits, so this is a wording/scenario holdout, not a holdout of entire statistical methods.

Each reference contains an exact plan, ordered source rows, original source values, scale, units, Decimal calculation and rounded result. A second checker uses boolean dataframe filters instead of the executor's indexed lookup and recomputes arithmetic from `SCALAR_ID`. It verifies all **33 answer references**. This checks numerical implementation independently; it does not replace external review of the conceptual plans. Reference data and answers are excluded from model inputs.

Queries include lookups, changes in percentage points, changes in persons, percent changes in price indexes, annual average versus endpoint distinctions, province comparisons and means over months. One test computes a mean across all 12 monthly observations in a calendar year using a compact window. Each request touches 1–12 numerical observations. Corpus size is the size available to query, not the number of rows a model reasons over in one answer.

## Separate comparisons

### 1. Source discovery

A fixed combination of word/bigram TF-IDF and character 3–5-gram TF-IDF ranks all 8,270 English/French title records. The query uses a short declared synonym dictionary. Word and character cosine scores receive weights 0.6 and 0.4; the weights were not selected on test predictions. Ties resolve by product ID.

The conventional source result is the top-ranked table. The model receives only the top 12 candidates, without the numerical-corpus availability flag, plus the original request. It selects a candidate or requests review. It does not receive the four numerical dictionaries or answer labels. Score designated-source recovery at rank 1 and designated-source inclusion at ranks 5 and 12. These are not complete relevance judgments: an unadjudicated alternative table is not automatically irrelevant. The 24 answerable test requests have a single authored source-table reference. That narrow reference tests the intended published table; alternative genuinely valid tables would require adjudication before being treated as equivalent.

The fixed conventional test result is **9/24 top-1**, **12/24 recall at 5**, and **15/24 recall at 12**. Candidate coverage limits a top-12 reranker to at most 15/24 on these references. This result is retained; the candidate count or query expansion is not enlarged after seeing it. A model that selects outside its supplied candidates fails.

### 2. Numerical planning with four known tables

All numerical arms receive the same four-table coverage, dimension labels, member values, units, date restrictions and operation language. They have access to the same deterministic executor. This task is explicitly scoped to the four-table corpus; it is not counted as successful catalog discovery.

- **Conventional:** a finite synonym/slot parser selects table, dimensions, dates and operation. It requests review when it cannot resolve a unique supported request. It does not read the reference plans.
- **One-shot:** one model response returns a JSON plan; the executor calculates once. A failed plan is a failure, with no corrective model call.
- **Tool controller:** the same model and initial evidence can issue up to three execution requests and must return a final plan within four model turns. Each tool result contains source rows or an explicit error. A final plan passes through the same executor.

The controller is implemented directly in Python using JSON actions. It is **not a native LOTUS `Corpus.agent` run**. Its value over one-shot planning is adaptive inspection and correction using the same available data and operations. Because the controller observes query results before its final answer, the comparison is an adaptive workflow comparison, not proof of a pure reasoning advantage with identical observations.

The operation language supports lookup, second-minus-first difference, `100*(second/first-1)` percent change, second/first ratio, arithmetic mean and sum. An operand has a table, all required dimension filters and a reference period. For monthly means/sums it can use `YYYY-MM/YYYY-MM`, so a 12-month operation needs one compact selector. No model code is executed. Exact filters must yield one observation per period. Missing, suppressed or unsupported observations raise explicit errors. Units must match; scalar multipliers are applied before Decimal arithmetic. Final rounding uses decimal half-up rules. A scaled LFS count is labelled Persons, while a difference between rates is labelled percentage points.

The executor's unit and uniqueness checks do not prove conceptual comparability. The reference certificate checks the requested table, population, adjustment, dates, rows, operation, units and rounding. This catches a plausible answer calculated from an inappropriate series.

## Scoring and cost

A numerical answer is completely correct only if its status, source rows, operation, unit and rounded value match the reference. Source-row order matters for subtraction and growth; means and sums treat source order as immaterial. A correct number with the wrong source fails. A required review is correct only when a nonempty reason accompanies review; the present automatic score does not validate the quality of that reason. Case-level review should distinguish sound reasons from coincidental abstention.

Report complete correctness, errors, review outputs, wrong non-abstaining answers and source/value components, alongside each prediction. Do not average catalog and numerical scores. The conventional numerical baseline achieved **12/12 development** and **29/32 test**, with one wrong non-abstaining test answer and nine review outputs. No post-test rule tuning was applied. These counts are descriptive for this authored workload and do not certify rare-error performance.

Model calls use the shared bounded caller: the declared model, deterministic settings, output caps, no retries, immutable request tags and a shared price-reservation ledger. A plan/action response has a 1,536-token output cap; catalog selection has 384. The maximum planned test calls are 32 one-shot, 128 controller and 24 discovery calls, before early final answers. API failures and invalid responses remain in the results and cost ledger. No human-time or national savings claims follow from API costs alone. Review counts and evidence items are proxies until reviewers are timed under the same correctness standard.

## Stopping and evidence boundary

The questions, reference plans, prompts, candidate depth and scoring rules are fixed before model execution. Run each planned test arm once; resumed execution skips journaled cases and the shared caller rejects duplicate call tags. A later prompt or rule revision must have a separately named protocol and fresh questions. Preserve this experiment's failed calls and adverse results.

The continuation threshold is at least three additional complete answers on the 32 test requests (29/32 to 32/32), with no increase above the baseline's one wrong non-abstaining answer. A workload reduction would require separate measured review evidence, not an after-the-fact replacement of this threshold. Given the baseline's 29/32 score, only three correct cases remain available to improve; report the numerator rather than imply precision from a percentage. The iterative controller must improve complete correctness over one-shot by at least 5 percentage points (at least two cases out of 32) to justify continued development when it consumes more than twice the tokens. An agent with no additional correct answers and higher cost has no demonstrated benefit here. A favorable result would justify testing real analyst requests, not production deployment.

Catalog failure may be candidate retrieval failure rather than model reasoning. Exact-label tasks are a control for unnecessary model use. Unsupported geography, uncertain age/adjustment, out-of-vintage dates, invalid rate aggregation and annual-versus-endpoint choices are built into the frozen workload. No selected anecdotes replace the complete table of outcomes.

## Reproduce

From the repository root, using the pinned environment:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/statistics.py verify
.venv/bin/python LLM_basics/nso-semantic-workflows/statistics.py baseline
```

Both commands are offline when the frozen files are present. The baseline command reuses recorded predictions after their first execution. Live model execution is explicit and uses the existing environment configuration and shared budget:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/statistics.py live --split test --arms one_shot agent discovery
```

`statistics.run(call_json, split="test", arms=(...))` exposes the same runner to a notebook. `statistics.jobs()` returns one-shot request specifications without sending them. Results are under `results/statistics/`; the ledger records raw provider requests, responses, usage, timing and failures. Source data, reference certificates and model predictions remain separate.
