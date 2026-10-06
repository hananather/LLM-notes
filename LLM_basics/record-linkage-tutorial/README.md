# Record-linkage handbook

[Open the notebook](../record-linkage-with-semantic-operators.ipynb) for eight
chapters on record linkage, beginning with a Fellegi–Sunter baseline and a LOTUS
semantic join on Splink's standard FEBRL4a/FEBRL4b data. Later chapters examine
representation, error diagnosis, learned similarity and computation at scale.
Nine editable TikZ figures accompany the explanations. The measured benchmark,
constructed examples and proposed experiments are identified separately.

The chapters follow one sequence:

1. Define the entity and relation.
2. Combine comparison evidence.
3. Express a relation with semantic operators.
4. Compare Splink and LOTUS on the same records.
5. Separate representation from additional information.
6. Connect diagnostic patterns to testable repairs.
7. Learn reusable similarity from selected LLM judgments.
8. Account for candidate recall, computation and evaluation at scale.

The notebook executes locally in replay mode without a model key or network
access. A recorded, real LOTUS run supplies its LLM predictions. Set
`RUN_LIVE = True` in the notebook to make new paid requests using an
`OPENAI_API_KEY` already set in your environment. Never paste a key into a cell.

## Start here

From the repository root, create an isolated Python 3.11 environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r LLM_basics/record-linkage-tutorial/requirements-lock.txt
.venv/bin/jupyter lab LLM_basics/record-linkage-with-semantic-operators.ipynb
```

If this repository already has an environment for another project, use a new
environment directory and substitute its path in these commands. Select its
Python kernel, then run the notebook from top to bottom. The notebook supports
the repository root and `LLM_basics/` as working directories.

The latest stable releases checked on October 6, 2026 were
[LOTUS 1.2.4](https://pypi.org/project/lotus-ai/1.2.4/) and
[Splink 5.0.0](https://pypi.org/project/splink/5.0.0/).
`requirements.in` lists direct dependencies; `requirements-lock.txt` records
the complete installed environment. Splink 5 requires registering data with
`DuckDBAPI.register` before creating a `Linker`.

The original benchmark was executed with Python 3.11.15 on macOS 26.6.2, arm64.
The [handbook execution record](results/handbook-verification.json) describes
the expanded notebook and its checks separately.
Linux and Windows have not been tested for this tutorial. The lock file records
versions; it does not establish portability to every platform.

## What the first run establishes

The recorded live comparison on October 6, 2026 evaluated 225 pairs containing
10 true matches and 215 nonmatches:

| Method | True links found | False links | Missed links | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Splink 5.0.0 | 10 | 0 | 0 | 1.00 | 1.00 |
| LOTUS 1.2.4 + `gpt-4.1-mini-2025-04-14` | 8 | 0 | 2 | 1.00 | 0.80 |

The LOTUS run took 18.34 seconds, used 57,525 input tokens and 675 output tokens,
and reported an estimated API cost of **$0.02409 USD**. This is the model client's
estimate, not an invoice. It excludes a separate four-pair installation smoke
test. The notebook reports newly measured local Splink timings separately from
the recorded API timing.

These results support the conventional baseline on this slice. Ten true links
are too few to infer a general accuracy ranking. The notebook includes a
falsifiable hypothesis about unstructured records; it does not run a native
unstructured benchmark or LOTUS optimization experiment.

## What is held constant

Both methods see the same six observed fields and score the same Cartesian
pairs. Original record IDs encode identity, so preparation replaces them with
opaque IDs and keeps truth in a separate file. The LLM receives only the six
observed fields. Splink gets supervised parameter estimates from training
entities and a threshold chosen on separate validation entities. No test labels
are used to fit the model, tune the threshold or select favorable pairs.

Data preparation and provenance are documented in [data/README.md](data/README.md).
Source hashes, output hashes, split rules and counts are in
[data/manifest.json](data/manifest.json). Both upstream license notices are
preserved. The public benchmark may have appeared in model training; an entity
split does not rule out that possibility.

## Files and verification

| File | Purpose |
| --- | --- |
| `tutorial_helpers.py` | Strict Boolean parsing, pair metrics, prompt auditing and replay checks |
| `handbook_examples.py` | Deterministic calculations and display of the compiled diagrams |
| `diagrams/` | Nine TikZ sources, compiled PDF/PNG figures, build instructions and hashes |
| `run_notebook.py` | Execute and optionally render without machine-wide Jupyter configuration |
| `prepare_data.py` | Rebuild the pinned official benchmark with Python's standard library |
| `data/` | Prepared observations, separate truth, fixed slices, licenses and provenance |
| `results/lotus-febrl4.json` | Real model responses, ordered pair IDs, prompt/input hashes, versions, usage and timing |
| `results/verification.json` | Dated verification of the original benchmark revision |
| `results/handbook-verification.json` | Execution and figure checks for the expanded notebook |

Replay validates input, prompt, serialization, model and package metadata,
regenerates the exact LOTUS message hashes, and checks complete pair coverage.
Changing inputs or the prompt requires a corresponding new live run. Each
completed live run replaces the recorded snapshot; preserve a copy before
comparing different configurations. An incomplete or malformed run raises an
error and leaves the previous successful snapshot intact.
The benchmark interpretation and its diagram describe the dated original run;
revise them before presenting a new live run as a replacement result.

Run the default replay headlessly from the repository root:

```bash
.venv/bin/python LLM_basics/record-linkage-tutorial/run_notebook.py
```

Add `--html tutorial.html` to export a reading copy. Add `--live` only to make
new model calls. The runner saves the notebook with replay as its next default.
It uses the notebook libraries directly so unrelated global Jupyter extensions
do not affect execution.

The reading guide covers Fellegi and Sunter, LOTUS, Splink, Ather's
*LLM-Assisted Record Linkage*, active learning, diagnostic grouping,
LLM-guided clustering and distillation. The later experiments specify controls,
evaluation labels and stopping criteria; they have not been run at scale.
