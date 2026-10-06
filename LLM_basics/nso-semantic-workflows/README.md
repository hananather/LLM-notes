# Semantic joins in NSO workflows

[Read the notebook](../nso-semantic-workflows.ipynb) for measured comparisons of language-based selection, conventional matching and statistical-query tools. This extends the [Splink and LOTUS tutorial](../record-linkage-with-semantic-operators.ipynb) with public-source and explicitly synthetic experiments.

On the frozen tests, GPT-6 Luna produced 593 valid annotated affiliation sets out of 644, compared with 456 for the lexical baseline. It produced 28/32 correct synthetic industry decisions versus 18/32, and 27/32 occupation decisions versus 23/32. The affiliation comparison is against the implemented lexical methods, not the trained S2AFF pipeline. Coding cases are a convenience sample of authored classification boundaries, not observed survey returns.

The school screen stopped model work: conventional matching was correct for all 300 records in its probability sample. A separate changed-name-and-street sample scored 72/100. These populations are reported separately. The statistical-query comparison uses 401,944 frozen observations and separates catalog discovery from planning over four known tables. Rules, one-shot planning and the iterative controller each score 29/32; the controller uses 1.85 times the one-shot tokens and returns three wrong answers without review instead of one. Catalog selection recovers 15/24 designated sources versus 9/24 for lexical first choice, reaching the fixed shortlist ceiling.

Every scored model attempt is preserved, including invalid evidence quotations. The affiliation review flag passed through 30 incorrect cases among 576 automatic decisions. Selection gains therefore do not establish a reliable acceptance policy or measured staff-time savings.

## Read and verify without paid calls

The notebook replays saved predictions and checks their evidence. It needs no API key or downloads. Python **3.11.15** on macOS is the tested execution environment. Statistics verification binds that exact Python patch version and the recorded dependency versions; another 3.11 patch release fails the frozen execution-contract check. Live-request budget locking uses POSIX file locks; Windows execution is not supported by this runner. Other platforms have not been tested in this revision.

From the repository root, use an isolated environment:

```bash
python3.11 --version  # Must report Python 3.11.15.
python3.11 -m venv .venv
.venv/bin/python -m pip install -r LLM_basics/nso-semantic-workflows/requirements.txt
.venv/bin/python LLM_basics/nso-semantic-workflows/verify.py
.venv/bin/python LLM_basics/nso-semantic-workflows/run_selection.py linkage
.venv/bin/python LLM_basics/nso-semantic-workflows/run_selection.py coding
.venv/bin/python LLM_basics/nso-semantic-workflows/statistics.py verify
.venv/bin/python LLM_basics/nso-semantic-workflows/analyze.py
.venv/bin/python LLM_basics/nso-semantic-workflows/run_notebook.py --html nso-results.html
```

If `.venv` already belongs to another project, use a different environment directory. `requirements.txt` records tested direct versions; it is not a complete transitive dependency lock. The runner creates a temporary kernel that uses its launching Python interpreter, so it requires no global kernel registration. When opening the notebook interactively, select a kernel from the same environment. The HTML export retains figure descriptions and source links.

`run_selection.py` checks complete saved batches without changing their measured wall time. `--split dev` checks the development runs. Its `--live` flag permits only unfinished declared cases; completed results remain unchanged. The shared ledger prevents repeating an attempted request tag. A new experimental revision requires new artifact names and an explicit protocol rather than overwriting this test.

## Follow the evidence

| Evidence | Location |
|---|---|
| Portfolio, scope and common cost controls | [Protocol](protocol.md) |
| Synthetic NAICS/NOC facts, sources and full-index comparison | [Coding protocol](protocol-coding.md) |
| Historical affiliations and school continuity | [Linkage protocol](protocol-linkage.md) |
| Catalog discovery and exact statistical queries | [Statistics protocol](protocol-statistics.md) |
| Complete paired results, corrections and regressions | [Offline analysis](results/analysis/README.md) |
| Raw public requests, responses, usage and failures | [API ledger](results/api-ledger.jsonl) |

The classification and affiliation systems implement blocked semantic selection through the provider API. The statistics system uses a bounded Python controller with JSON actions and deterministic tools. These are measurements of those implementations; they are not native LOTUS optimizer or `Corpus.agent` benchmarks. The earlier tutorial contains the native LOTUS examples.

## Rebuild sources and conventional baselines

The task protocols provide the exact download and baseline commands. Large original archives remain in the ignored `cache/` directory. Downloaded bytes must match their pinned hashes. Compact candidate records support replay after full-universe retrieval; they are not the original candidate universe.

Source and licence records accompany each dataset. Statistics Canada data retain their required attribution; synthetic descriptions and experimental conclusions are separate from official statistics. S2AFF labels and ROR metadata use the recorded historical versions. NCES labels concern continuing administrative school identifiers within the stated open-school population.

Pre-model audit repairs remain visible. One synthetic cook vignette was clarified, all nonempty occupation duties were included in retrieval, missing responses were kept in scoring denominators, and three CPI difference units were corrected to index points. Original snapshots remain beside the accepted revisions. No test prediction was used to change these labels or prompts.
