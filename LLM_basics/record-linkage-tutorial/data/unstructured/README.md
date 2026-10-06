# Small unstructured-data reference sets

These frozen slices illustrate two workloads from the
[LOTUS paper](https://www.vldb.org/pvldb/vol18/p4171-patel.pdf): report-to-category
joins (BioDEX, §5.2) and claim verification with Wikipedia evidence (FEVER,
§5.1). They complement the FEBRL identity-linkage experiment. They do not
estimate accuracy on national statistical office data or reproduce the paper's
optimization experiments.

## BioDEX: join reports to reaction categories

The source is the authors' [BioDEX-Reactions dataset](https://huggingface.co/datasets/BioDEX/BioDEX-Reactions),
pinned to revision `01a5dacdabd144a120af04931a11a99febd48432`. The official
[LOTUS benchmark](https://github.com/lotus-data/lotus/tree/main/benchmarks/biodex)
uses this dataset and a semantic join against reaction names. The
[BioDEX paper](https://arxiv.org/abs/2305.13395) describes its reporting labels.

Eight reaction categories are selected by frequency in the **training** split,
with alphabetical tie-breaking. Eligible test reports have a CC BY license,
1,000–8,000 characters in `fulltext_processed`, and at least one of those
categories in their published reaction list. Of 4,249 test reports, 23 qualify.
The eight smallest hashes of `lotus-small-v1:` plus PMID select the reports.
No prediction is used to choose reports or categories.

The resulting 8 × 8 join has **64 pairs, including 11 annotation-positive
pairs**. This is an enriched convenience slice. Precision depends on this
selection; it is not an estimate for the complete dataset or an NSO workflow.
Published reaction-list membership defines the reference label. A missing label
is a dataset-negative pair, not a clinical finding that the condition is absent.
Some source reports mention events omitted from their reaction lists.

`biodex-reports.json` preserves the source's complete `fulltext_processed`
field without local truncation or rewriting. This field is already processed
upstream; it is not the original publication file. `biodex-truth.json` keeps the
published labels separate from the model's inputs. `biodex-attribution.json`
records each article's authors, title, DOI, PubMed URL, text length and exact
CC BY 3.0 or 4.0 license URL extracted from its source frontmatter. Those
article licenses continue to apply to the text. Reaction names and reporting
annotations retain their BioDEX provenance; no blanket repository license is
claimed for third-party content.

## FEVER: decide whether supplied evidence supports a claim

The sources are the [official FEVER shared-task development set and June 2017
Wikipedia dump](https://fever.ai/dataset/fever.html). The prepared claims retain
their original SUPPORTS/REFUTES labels. A singleton evidence annotation supplies
one sufficient evidence sentence according to FEVER's annotators.

Preparation examines original Wikipedia shards in lexicographic order and uses
the first with six distinct singleton-evidence pages in each label class:
`wiki-pages/wiki-006.jsonl`. Within each class, claim-ID hashes select six
claims, retaining at most one per evidence page. The final **12 claims use
10 distinct sentences**. FEVER IDs and original evidence sets are stored in
`fever-truth.json`; the model receives only claim, page title and sentence.
Wikipedia's tokenized titles and sentences are preserved. Attribution links
decode the title's parenthesis placeholders.

The filter evaluates support **conditional on supplied gold evidence**.
Ordinary background knowledge may clarify a term or named reference, but may
not replace missing evidence. Public benchmark contamination remains possible.
This excludes NOT ENOUGH INFO claims and cases requiring multiple sentences.
No unannotated claim–sentence pair is assigned a false label.

BM25 retrieval reports the selected annotated sentence's rank within this
10-sentence, gold-derived corpus. It is a closed-corpus illustration, not a
search evaluation over Wikipedia. A separate fixed lexical support heuristic
accepts claims with at least 80% content-word coverage in the supplied title
and sentence. Its stopword list and threshold are fixed in the runner; neither
is tuned on these 12 labels. Similar vocabulary does not imply support.

FEVER annotations and Wikipedia text are reused under the
[original FEVER license notice](FEVER-LICENSE.html) and applicable Wikipedia
licenses, including [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
Page URLs provide attribution to Wikipedia contributors. Local transformations
are subsetting, field separation and JSON serialization, as specified above.

## Reproduce and inspect

Run `unstructured_prepare.py` to rebuild the frozen slices. It downloads source
data into the ignored `.cache/unstructured-sources/` directory. Source versions,
SHA-256 hashes, ZIP identity, selection rules, counts and prepared-file hashes
are recorded in [manifest.json](manifest.json). Subsequent preparation checks
the committed manifest and refuses changed sources or outputs. Only selected
text is included in the repository; no full source corpus is committed.

`unstructured_benchmark.py` defaults to replay of two models released in 2026:
[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
(September 22) and
[GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
(July 9). Both use `reasoning_effort="none"`, temperature zero and a 32-token
completion limit. The prompts and frozen data are identical to the historical
runs. Their current catalog snapshot identifiers have no dated suffix, so each
run retains the exact requested and returned model identifiers and execution date.
Future live behavior may change; saved responses remain available for replay.

`--live` uses an environment key and writes new timestamped snapshots. It
preflights every prompt and the response limit using the 1.25× cache-write
input rate plus a 20% margin, requests zero retries, and refuses a planned cost
above its budget. The standard
[prices](https://developers.openai.com/api/docs/pricing), checked on October 6,
2026, are $0.10/$0.50 per million input/output tokens for GPT-6 Luna and
$0.20/$1.20 for GPT-5.6 Luna. The default suite budget is $0.10; the maximum
allowed is $1. This bounds planned requests, not the provider's invoice.

The earlier GPT-4.1 nano and GPT-4o mini results remain optional historical
baselines. Use `--model-group historical` to replay them or `--model-group all`
to inspect both generations. The default tables and new live runs use only the
2026 models. Historical snapshot files are preserved without modification.

Snapshots retain exact predicates, model snapshots, input and annotation-manifest
hashes, prompt hashes, raw answers, provider usage and elapsed time. Invalid
answers or incomplete runs are preserved and rejected for scoring. Replay makes
no model calls. The literal-phrase and all-label-words BioDEX baselines use the
same report text and categories as the LLMs; no Splink probabilities are fitted
to this different relation.

LOTUS 1.2.4 parses referenced columns through a Python set. For a multi-column
filter, this can change prompt field order between processes. The live helper
sorts that parser's output in a temporary scope and restores the original
function afterward, including on failure. This workaround is for the serial
tutorial runner; do not use it with concurrent LOTUS jobs. Offline prompt
capture under Python hash seeds 0, 1 and 2 verifies the same saved message hashes.

The installed LOTUS release also requests permissive parameter handling from
LiteLLM. An [offline request-payload check](../../results/unstructured-request-settings-verification.json)
intercepted all 152 current-model requests before transport and verified that
temperature zero, the 32-token limit, `reasoning_effort="none"` and zero SDK
retries reached the provider interface with the recorded message hashes.
No model requests were made by that check. The helper now rejects unsupported
parameters with `drop_params=False`; the same offline check passed for this
stricter path. The original live snapshots remain unchanged.

The 2026 suite contains **152 scored comparisons** with an uncached-price
estimate of **$0.0343137**. The provider returned the requested model IDs;
all responses finished normally and passed strict Boolean parsing. Its datasets,
category selection, support predicate and lexical baselines are unchanged from
the historical experiment. No prompt or example was selected using the new
models' outcomes.

The historical suite contains 152 scored model comparisons. One earlier
12-comparison FEVER GPT-4.1 nano attempt failed the prompt-hash audit and remains in an
explicitly named incomplete snapshot. It is excluded from accuracy tables.
Correcting predicate formatting and freezing field order changed the audit
implementation; the task predicate and source labels were unchanged. The final
historical runs' uncached-price estimate is **$0.02819575**; all 164 historical attempted comparisons,
including that rejected attempt, total **$0.02846805** at the dated list rates.
These estimates price every input token as uncached. Actual charges also depend
on cache reads, cache writes and the account's billing terms.
