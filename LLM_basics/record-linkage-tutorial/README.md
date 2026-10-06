# Record linkage with Splink and LOTUS

[Open the notebook](../record-linkage-with-semantic-operators.ipynb) for a
computational companion to the LOTUS paper. It connects Fellegi–Sunter linkage
with semantic joins, then examines how text interpretation could support
national statistical offices. One notebook contains eight chapters, fourteen
figures, three small experiments, an agentic dataset workflow, worked
calculations and a source-linked research guide.

The experiments answer different questions:

| Dataset | Target | Comparison |
| --- | --- | --- |
| Splink FEBRL4 | Same-person identity across two tables | Supervised Splink versus a direct LOTUS join on 225 pairs |
| BioDEX | Article–reaction category relation | Literal phrase and word baselines versus two cheap LOTUS models on 64 pairs |
| FEVER / Wikipedia | Annotated evidence retrieval and claim support | BM25 retrieval, word-overlap support and two cheap LOTUS models on 12 claims |
| Synthetic field-office returns | Revision selection, unit normalization and pooled completion rate | Tool-using LOTUS map/reduce checked against fixed source-derived counts |

The primary models are **GPT-6 Luna** (released 22 September 2026) and
**GPT-5.6 Luna** (released 9 July 2026). The text suite uses both; the FEBRL
identity comparison and agentic workflow use GPT-6 Luna. Earlier GPT-4 results appear only in
collapsed historical-baseline sections. [Release dates](https://developers.openai.com/api/docs/changelog)
and [prices](https://developers.openai.com/api/docs/pricing) were checked on
6 October 2026.

The text experiments examine different relations; their scores do not rank
identity linkage systems. Native-text identity linkage with extraction followed
by Splink remains a proposed comparison.

## Run or read

From the repository root, create an isolated Python 3.11 environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r LLM_basics/record-linkage-tutorial/requirements-lock.txt
.venv/bin/jupyter lab LLM_basics/record-linkage-with-semantic-operators.ipynb
```

If an environment already serves another project, use a separate environment
directory. Select its Python kernel and run the notebook from top to bottom.
Both the repository root and `LLM_basics/` are supported working directories.

The default fits Splink locally and replays recorded, real model responses.
It needs no API key, downloads or TeX installation. To execute and export a
reading copy without machine-wide Jupyter configuration:

```bash
.venv/bin/python LLM_basics/record-linkage-tutorial/run_notebook.py --html reading-copy.html
```

The HTML reading copy omits the setup cell; the notebook retains it for execution.
The opening and dataset diagrams are embedded so they display without running code.

Live requests are explicit and independent:

```bash
# Repeat the 225-pair FEBRL join.
.venv/bin/python LLM_basics/record-linkage-tutorial/run_notebook.py --live

# Repeat only the two small text tasks with both cheap models.
.venv/bin/python LLM_basics/record-linkage-tutorial/run_notebook.py --live-unstructured

# Repeat the agentic map/reduce workflow over three synthetic office returns.
.venv/bin/python LLM_basics/record-linkage-tutorial/run_notebook.py --live-agentic
```

These modes require `OPENAI_API_KEY` in the environment. Never paste a key into
a cell. The text suite checks a conservative token-price estimate against a
$0.10 budget before sending requests; retries are disabled for that suite.
The check bounds planned request sizes at the recorded prices, not an invoice.
Saved notebooks return to replay mode for their next execution.

Every successful live rerun writes new timestamped snapshots. Existing results
are never overwritten. Dated prose and Figure 5 identify the saved GPT-6 Luna
comparison; result tables identify the responses used by the current execution.
Live defaults use only the two 2026 models, with reasoning disabled. The
Boolean experiments cap answers at 32 tokens. The agentic workflow permits
1,500 output tokens per request, at most 21 requests and a separate $0.10
token-price budget, with no automatic retries. Its ledger includes planning,
map and reduce calls; the native LOTUS usage object excludes planning.

## Follow the argument

Chapters 1–5 define the entity relation, explain comparison evidence, introduce
LOTUS, and run the structured and text examples. Chapter 5 also maps the
workflows to NSO classification, catalogue search, extraction and survey
feedback, with primary institutional references and conventional comparators.
Section 5.6 then demonstrates the current `Corpus.agent` API: a model plans
instructions, three agents read source returns and call an exact calculator,
and a reducer combines their findings. Recorded tool observations and an
independent check make revision selection and unit conversion inspectable.
The synthetic workflow illustrates execution; it does not estimate accuracy
on operational NSO documents or savings over a purpose-built parser.

The recorded agentic run selected every source and count correctly and returned
1,130 completed cases out of 1,600 eligible cases (70.625%). Twelve model calls
and seven tool calls took 13.40 seconds and an estimated $0.001594 in API tokens,
including planning. These are measurements of one execution on invented data.
The helper preserves full model settings through `completer_factory`, records
the final provider payloads, and checks each output against both its tool
observation and the independent reference. The notebook shows the plan fields
used here; the snapshot also retains native metadata. In LOTUS 1.2.4,
`reduce_strategy` is unused: the reducer is a single agent over all findings.

Chapters 6–8 develop proposed research on diagnostic repairs, reusable LLM
supervision and scale. The illustrative arithmetic is executed; the repair,
distillation and 100M-by-100M studies have not been run. The supporting
[research notes](../../research/error-guided-record-linkage/README.md) retain
the fuller evidence review and experimental protocol.

## Inspect the evidence

| File | Purpose |
| --- | --- |
| `data/README.md` | FEBRL source, licences, entity splits and fixed evaluation slice |
| `data/unstructured/README.md` | BioDEX/FEVER source revisions, selection, licences and annotation boundaries |
| `tutorial_helpers.py` | Strict Boolean parsing, metrics and FEBRL replay integrity |
| `unstructured_benchmark.py` | Actual LOTUS joins/filters, lexical baselines, budget checks and text replay |
| `agentic_demo.py` | Native LOTUS agentic execution, bounded tools, full-call accounting and validated replay |
| `data/agentic-office-returns.json` | Synthetic office returns, revision states, units and definitions |
| `prepare_data.py`, `unstructured_prepare.py` | Rebuild the frozen datasets from checked sources |
| `handbook_examples.py` | Deterministic calculations, figures and replay displays |
| `diagrams/` | Editable TikZ sources, PDF/PNG assets and their hashes |
| `results/lotus-febrl4-gpt-6-luna.json` | Current FEBRL inputs, responses, prompt hashes, provider usage and timing |
| `results/lotus-febrl4.json` | Preserved GPT-4.1 mini historical baseline |
| `results/unstructured-*.json` | Current and historical text snapshots, failed-attempt evidence and request-settings verification |
| `results/agentic-*.json` | Agentic plan, tool calls and observations, provider usage and source checks |
| `results/review-verification.json` | Execution, integrity and rendering checks for the reviewed revision |
| `results/review-verification-20261006-text.json` | Preserved verification before the agentic section was added |
| `results/verification.json`, `results/handbook-verification.json` | Preserved verification of earlier revisions |

FEBRL truth is separate from observed fields. Both records for each entity
stay in one split; test labels do not fit the model or choose its threshold.
Splink uses supervised training entities, whereas the LLM receives a zero-shot
predicate. The teaching slice is enriched and contains only ten true links.

The text suite selects records before model scoring. BioDEX evaluates
agreement with published annotations, whose omissions remain visible in the
notebook. FEVER support decisions receive annotated evidence; its retrieval
result concerns the bundled miniature corpus. Neither example estimates
national-scale accuracy, and both public datasets may have appeared in model
training. Plain semantic operations are measured; no LOTUS cascade or optimizer
speedup is claimed.

## Tested environment

The pinned environment uses LOTUS 1.2.4, Splink 5.0.0 and Python 3.11.15.
`requirements.in` lists direct dependencies; `requirements-lock.txt` records
the installed versions. Splink 5 registers data with `DuckDBAPI.register`
before constructing a `Linker`.

The [current verification record](results/review-verification.json) identifies
the exact artifact hashes and checks on macOS 26.6.2, arm64. Linux and Windows
have not been tested for this tutorial. Earlier verification records describe
their own dated revisions; they do not certify subsequent edits.
