# NSO semantic-workflow pilot protocol

This pilot tests whether language interpretation improves a defined data task after a conventional method has access to the same reference information. It separates classification, organization linkage and statistical queries. Their scores are not pooled into one accuracy number.

## Questions and comparators

| Task | Reference information | Comparison | Main decision |
|---|---|---|---|
| NAICS industry coding | Full Canadian 2022 classification and examples; source-grounded synthetic establishment descriptions | Full exact lookup, word/character retrieval and qualifier rules versus model selection | Does the model resolve unfamiliar activity descriptions without introducing boundary errors? |
| NOC occupation coding | Full Canadian 2021 classification, titles and duties; source-grounded synthetic job descriptions | The same full-resource controls versus model selection | Can duty evidence resolve a job beyond its title? |
| Affiliation linkage | Published S2AFF annotations and the historical full ROR registry | Full-registry lexical retrieval versus model selection from retrieved candidates | Does language interpretation improve identity selection, including multiple organizations and unmatched text? |
| School linkage | Two annual NCES public directories | Normalized names, addresses and conventional candidate matching | Does the remaining ambiguity justify any model calls? |
| Statistical queries | Frozen Statistics Canada catalog, metadata and public observations | Lexical discovery and rules; one-shot model plan; bounded iterative tool controller | Does the language layer improve complete answers, and does iteration add value beyond one shot? |

Each task has a separate protocol, source manifest, development split, frozen evaluation split and result record. Synthetic descriptions and authored questions are identified as such. Public observations do not make authored requests representative of actual analyst workloads.

## Freeze and development

Source versions, cases, reference answers, exclusions and split rules are fixed before live model scoring. Reference answers are checked independently of the evaluated model. Development permits one bounded prompt or rule revision. Evaluation results do not authorize edits to the same test or repeated attempts to improve its score. Corrections to a defective reference are recorded with their reason and original result; they do not disappear from the audit trail.

Exact official aliases remain available in every classification index. Candidate generation searches the complete eligible reference universe, and candidate recall is reported separately from downstream selection. No reference answer is injected into a candidate set. Data splitting and metrics follow the task protocols rather than treating related variants as independent observations.

## Execution and cost

The live model is `gpt-6-luna`, with temperature 0, reasoning effort `none`, JSON output, explicit output limits and no automatic retries. The [model page](https://developers.openai.com/api/docs/models/gpt-6-luna), checked on 6 October 2026, lists US$0.10 per million input tokens and US$0.50 per million output tokens. These rates give a token-price estimate, not an invoice. Input reservations also allow for cache-write premiums and estimation error.

All tasks share a US$10 token-price reservation ceiling. The append-only ledger records full public requests, prompt hashes, raw responses, provider usage, failures and elapsed time. Reservations persist after failures. A failed or invalid call remains an unsuccessful attempted case; it is not replaced silently. A unique request tag prevents accidental reruns. Credentials are read only from the existing environment and never saved in results.

The query controller uses a bounded JSON action loop and deterministic tools. It is evaluated against a one-shot plan with the same allowed operations. This suite records the execution method explicitly; its custom calls are not labeled as native LOTUS operators. The companion semantic-operators tutorial contains the native LOTUS demonstrations.

## Decisions and interpretation

Continue a line only when it resolves a material uncertainty or adds correct decisions beyond its credible baseline. The task protocols specify screening thresholds before model results. A stopped line retains its results and rationale. A better score on a small synthetic panel supports a larger workflow study; it does not establish national coding accuracy, production readiness or staff hours saved.

Report complete-answer counts, corrections and regressions, candidate recall, abstentions, unsupported decisions, calls, tokens, latency and estimated token price. Workload proxies such as review flags remain proxies until a timed human study measures them. Related synthetic cases are summarized by family; any resampling describes the observed families and does not turn them into a probability sample of NSO work.

The final report keeps adverse results and identifies the strongest conventional explanation for a model gain. Source and label defects, untested alternatives and incomplete evidence remain visible.
