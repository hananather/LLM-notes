# Semantic joins in NSO workflows

[Read the complete report](../record-linkage-with-semantic-operators.ipynb).
This is the single notebook for the methodological explanation, all completed
experiment families, source examples, adverse results, costs, research lessons
and next decisions. This directory preserves NSO experiment code and evidence;
[native LOTUS and Splink support](../record-linkage-tutorial/README.md) is adjacent.

The report's [selected examples](../record-linkage-with-semantic-operators.ipynb#dataset-showcase)
include both a correction and an incorrect automatic affiliation link. It also
contains the product, conventional-only, synthetic and unscored branches with
their distinct evidence boundaries. Detailed grids and complete source records
remain supporting evidence.

The completed native-affiliation study tests research-output allocation on
1,104 curated French, multilingual and multi-organization source rows. Its
strongest control trains conventional models on 195 native rows, selects on
212 and tests the remaining 697. Semantic selection returns 423 correct
automatic organization sets versus 263 for the selected tree, with 44 versus
163 false allocations and 416 versus 483 misses. It leaves 135 rows for review.
Genuine supervised Splink, extraction followed by Splink, lexical methods and
trained trees all retain their complete outcomes. These are public challenge
cohorts; the experiment does not estimate national publication counts or an
operational acceptance policy.

The completed clustering study retains its null and adverse findings.
Diagnostic acquisition policies tie a strong Splink baseline at 779 true links,
zero false links and two misses on 781 held-out synthetic entities. A student
trained on 300 LLM judgments loses recall. On a separate multi-record synthetic
corpus, three false edges create 27 false co-clustered relationships. The
notebook shows both the measured acquisition comparison and these actual graph
bridges.

On the frozen tests, GPT-6 Luna produced 593 valid annotated affiliation sets out of 644, compared with 456 for the lexical baseline. It produced 28/32 correct synthetic industry decisions versus 18/32, and 27/32 occupation decisions versus 23/32. The original affiliation comparison uses implemented lexical methods. The [9 October trained comparison](affiliation-value/README.md) adds logistic regression and a boosted tree on the same candidates: the validation-selected tree gets 492/644 annotated sets, compared with 593 for the saved LLM selections, or 546 correct LLM automatic decisions. It uses no new model API calls. The official S2AFF pipeline remains unmeasured. Coding cases are a convenience sample of authored classification boundaries, not observed survey returns.

The school screen stopped model work: conventional matching was correct for all 300 records in its probability sample. A separate changed-name-and-street sample scored 72/100. These populations are reported separately. The statistical-query comparison uses 401,944 frozen observations and separates catalog discovery from planning over four known tables. Rules, one-shot planning and the iterative controller each score 29/32; the controller uses 1.85 times the one-shot tokens and returns three wrong answers without review instead of one. Catalog selection recovers 15/24 designated sources versus 9/24 for lexical first choice, reaching the fixed shortlist ceiling.

Every scored model attempt is preserved, including invalid evidence quotations. The affiliation review flag passed through 30 incorrect cases among 576 automatic decisions. Selection gains therefore do not establish a reliable acceptance policy or measured staff-time savings.

The later [product holdout](adversarial-2026/products/README.md#scored-comparisons) retained all 1,219 queries. Semantic selection made 702 correct complete decisions versus 382 for the development-selected lexical rule, with 508 versus 264 false links and 151 versus 788 missed links. It left 68 queries for review and 26 failed responses unresolved. This run followed an explicit post-development spending amendment; the original continuation gate remains failed. The [access chronology](../record-linkage-with-semantic-operators.ipynb#product-access-chronology) discloses limited test metadata exposure after request freeze and before prediction sealing.

## Read and verify without paid calls

The notebook replays saved predictions and checks their evidence. It needs no API key or downloads. Python **3.11.15** on macOS is the tested execution environment. Statistics verification binds that exact Python patch version and the recorded dependency versions; another 3.11 patch release fails the frozen execution-contract check. Live-request budget locking uses POSIX file locks; Windows execution is not supported by this runner. Other platforms have not been tested in this revision.

From the repository root, use an isolated environment:

```bash
python3.11 --version  # Must report Python 3.11.15.
python3.11 -m venv .venv
.venv/bin/python -m pip install -r LLM_basics/record-linkage-tutorial/requirements-lock.txt
.venv/bin/python LLM_basics/nso-semantic-workflows/verify.py
.venv/bin/python LLM_basics/nso-semantic-workflows/run_selection.py linkage
.venv/bin/python LLM_basics/nso-semantic-workflows/run_selection.py coding
.venv/bin/python LLM_basics/nso-semantic-workflows/statistics.py verify
.venv/bin/python LLM_basics/nso-semantic-workflows/analyze.py
.venv/bin/python LLM_basics/nso-semantic-workflows/native-affiliation-study-20261009/verification.py
.venv/bin/python LLM_basics/nso-semantic-workflows/native-affiliation-adaptation-20261009/verification.py
.venv/bin/python LLM_basics/nso-semantic-workflows/clustering-study-20261009/verification.py
.venv/bin/python LLM_basics/nso-semantic-workflows/run_notebook.py --html nso-results.html
```

If `.venv` already belongs to another project, use a different environment directory. The full report uses the adjacent pinned `requirements-lock.txt`, which includes
LOTUS, Splink and every direct dependency listed in this directory's `requirements.txt`. The latter remains sufficient only for the standalone NSO scripts. The runner creates a temporary kernel that uses its launching Python interpreter, so it requires no global kernel registration. When opening the notebook interactively, select a kernel from the same environment. The HTML export retains figure descriptions and source links.

The [research-completion verification](../record-linkage-tutorial/results/research-completion-notebook-verification-20261009.json)
records successful complete-notebook execution, described figures and mathematics,
preserved original cells and unchanged prior scientific evidence. The earlier
[complete-report verification](../record-linkage-tutorial/results/final-notebook-verification-20261009.json)
is preserved as dated evidence for the preceding revision. The new study
[manifest](results/research-completion-20261009.json) binds the recorded outcomes
and figures to their independently checked files. Default replay makes no model calls.

`run_selection.py` checks complete saved batches without changing their measured wall time. `--split dev` checks the development runs. Its `--live` flag permits only unfinished declared cases; completed results remain unchanged. The shared ledger prevents repeating an attempted request tag. A new experimental revision requires new artifact names and an explicit protocol rather than overwriting this test.

## Follow the evidence

| Evidence | Location |
|---|---|
| Portfolio, scope and common cost controls | [Protocol](protocol.md) |
| Synthetic NAICS/NOC facts, sources and full-index comparison | [Coding protocol](protocol-coding.md) |
| Historical affiliations and school continuity | [Linkage protocol](protocol-linkage.md) |
| Trained same-candidate affiliation comparison | [Frozen extension and independent scoring](affiliation-value/README.md) |
| Genuine supervised Splink on historical affiliation text | [Fellegi–Sunter control](affiliation-splink-control-20261009/README.md) |
| Native affiliation allocation, extraction and full-universe retrieval | [Frozen comparison and offline verification](native-affiliation-study-20261009/README.md) |
| Same-policy native training as an alternative explanation | [Matched adaptation control](native-affiliation-adaptation-20261009/README.md) |
| Diagnostic acquisition, learned similarity and entity-cluster transfer | [Completed clustering study](clustering-study-20261009/README.md) |
| Catalog discovery and exact statistical queries | [Statistics protocol](protocol-statistics.md) |
| Complete paired results, corrections and regressions | [Offline analysis](results/analysis/README.md) |
| Completed amended product holdout, costs and offline replay | [Product evidence](adversarial-2026/products/README.md#scored-comparisons) |
| Raw public requests, responses, usage and failures | [API ledger](results/api-ledger.jsonl) |

The classification and affiliation systems implement blocked semantic selection through the provider API. The statistics system uses a bounded Python controller with JSON actions and deterministic tools. These are measurements of those implementations; they are not native LOTUS optimizer or `Corpus.agent` benchmarks. The same report contains the native LOTUS examples, with their implementation boundaries stated.

## Rebuild sources and conventional baselines

The task protocols provide the exact download and baseline commands. Large original archives remain in the ignored `cache/` directory. Downloaded bytes must match their pinned hashes. Compact candidate records support replay after full-universe retrieval; they are not the original candidate universe.

Source and licence records accompany each dataset. Statistics Canada data retain their required attribution; synthetic descriptions and experimental conclusions are separate from official statistics. S2AFF labels and ROR metadata use the recorded historical versions. NCES labels concern continuing administrative school identifiers within the stated open-school population.

Pre-model audit repairs remain visible. One synthetic cook vignette was clarified, all nonempty occupation duties were included in retrieval, missing responses were kept in scoring denominators, and three CPI difference units were corrected to index points. Original snapshots remain beside the accepted revisions. No test prediction was used to change these labels or prompts.
