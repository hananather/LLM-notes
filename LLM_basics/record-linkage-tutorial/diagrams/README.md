# Record-linkage figures

Fourteen active TikZ figures accompany the record-linkage notebook. The former chapter map is retained as a historical asset and is no longer displayed. Each source uses the shared `diagram-style.tex`; the PDF and PNG are exported from that source.

## Build

Requires an installed TeX distribution with `standalone`, TikZ, Helvetica, and AMS math, plus Poppler (`pdftoppm`). No Python packages are required.

```sh
python3 build.py
python3 build.py 07-teacher-student
```

Select the intended TeX distribution through `PATH` if more than one is installed. PNGs are rendered at 220 dpi. For notebooks, display the PNG at the full content width; use the PDF for zooming or publication. The script writes only the named PDFs and PNGs and keeps compiler intermediates in a temporary directory.

## Visual notation

Each box names its role. Slate boxes usually contain observed data. Lavender boxes contain statistical models or execution settings. Pink boxes contain language-model steps. Green boxes contain labels or evaluation. White boxes contain fixed operations or explanations. Color supports these labels; it does not certify correctness.

Figures omit a common role legend. This avoids listing components that are absent from a figure. Dark arrows show data or evidence flow. A labeled dashed return shows a repeated update.

The record-pair, EM, overview and dataset figures use 9-point text. The other workflow figures use 7-point text. The source includes an editable layout; PDF and PNG files provide portable reading copies.

## Captions and evidence

### 00-chapter-map (historical; not displayed)

From linkage evidence to testable semantic workflows. The first four chapters establish the linkage task and recorded structured-data comparison. Chapter 5 examines source representation; Chapters 6–8 develop proposed repairs, learning methods, and scale tests.

**Evidence boundary.** Navigation diagram; no result claim.

[PNG](00-chapter-map.png) · [PDF](00-chapter-map.pdf) · [Editable source](00-chapter-map.tex)

### 01-identity-relations

Identity, shared context, and diagnostic similarity answer different questions. The constructed records show how two people can share a clinic while one person appears under different spellings. Here, Y denotes identity and H denotes shared context.

**Evidence boundary.** Illustrative records and identities are defined by construction. Diagnostic groups do not establish identity.

[PNG](01-identity-relations.png) · [PDF](01-identity-relations.pdf) · [Editable source](01-identity-relations.tex)

### 02-fs-evidence

The same constructed pair appears as record cards, a comparison vector, field evidence and a match probability. The name, birth date and postcode agree. Under the stated probabilities, the posterior is 98.7%. Changing only the second record's birth date lowers it to 0.4%.

**Evidence boundary.** Records and probabilities are stipulated for teaching. The calculation assumes conditional independence within matches and within nonmatches. It uses the exact values in `evidence_example` in `handbook_examples.py`. Displayed weights are rounded. The prior is 0.01%. A decision still needs a chosen threshold.

The layout follows the explanatory sequence in Splink's [Fellegi–Sunter guide](https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html) and [waterfall chart](https://moj-analytical-services.github.io/splink/charts/waterfall_chart.html): observed values, comparison evidence, prior and result. The composition and example are original.

[PNG](02-fs-evidence.png) · [PDF](02-fs-evidence.pdf) · [Editable source](02-fs-evidence.tex)

### 03-semantic-execution

A semantic predicate fixes the requested relation; the chosen execution plan determines how candidate pairs are processed. Output validation and evaluation against identity labels are separate steps.

**Evidence boundary.** Schematic direct path, with optional execution optimization shown separately. No optimized LOTUS cascade was executed in the recorded benchmark. Reference-model agreement is not identity accuracy.

[PNG](03-semantic-execution.png) · [PDF](03-semantic-execution.pdf) · [Editable source](03-semantic-execution.tex)

### 04-controlled-comparison

The comparison on 6 October 2026 uses the same 225 candidate pairs. Supervised Splink and Splink with EM each find all 10 true links and make no false links. LOTUS 1.2.4 with gpt-6-luna finds 8 true links, misses 2 and makes no false links. Each arm correctly rejects 215 nonmatches. The supervised threshold is 26%; the fixed EM threshold is 50%. The [EM record](../results/unlabeled-febrl4-20261006T190913831251Z.json) preserves the protocol, fitted parameters and decisions. The [LLM snapshot](../results/lotus-febrl4-gpt-6-luna.json) preserves all 225 responses and provider usage. Notebook §4.6 reports the current execution.

**Evidence boundary.** Recorded decisions on a small enriched synthetic slice. EM and the fixed LLM instruction use no task-specific fitting or threshold-selection labels. The supervised reference uses training and validation labels. The existing partitions and slice were constructed using reference identities and had already been inspected. Prior knowledge and computation differ between procedures.

[PNG](04-controlled-comparison.png) · [PDF](04-controlled-comparison.pdf) · [Editable source](04-controlled-comparison.tex)

### 05-representation-paths

A representation study can compare structured-field linkage, direct judgment over native text, and recordwise extraction followed by validated field comparisons. The target identity relation, held-out evaluation, and accounting rules remain fixed.

**Evidence boundary.** Proposed study. Reformatting existing fields alone cannot establish a benefit from additional native information. Extraction costs must be counted.

[PNG](05-representation-paths.png) · [PDF](05-representation-paths.pdf) · [Editable source](05-representation-paths.tex)

### 06-diagnosis-repair

Diagnostic signatures guide review of several cases and counterexamples. A repair has an explicit applicability region; capped validation precedes a frozen evaluation on unqueried entities.

**Evidence boundary.** Proposed mechanism. The address-truncation example is illustrative, and groups may overlap. No repair-transfer result is asserted.

[PNG](06-diagnosis-repair.png) · [PDF](06-diagnosis-repair.pdf) · [Editable source](06-diagnosis-repair.tex)

### 07-teacher-student

Teacher judgments and student predictions meet in a training objective. In a separate held-out path, frozen linkage predictions and independent identity labels enter evaluation as distinct inputs.

**Evidence boundary.** Proposed teacher-guided learning mechanism. A teacher score is a fallible signal, and the choice of loss depends on its interpretation. No student or clustering gain is reported.

[PNG](07-teacher-student.png) · [PDF](07-teacher-student.pdf) · [Editable source](07-teacher-student.tex)

### 08-sparse-scale

Candidate generation reduces the cross product to a sparse graph; cheap comparisons and selective language-model work feed the final decision procedure. A separate audit measures true links omitted from the candidate set.

**Evidence boundary.** Symbolic cost and recall accounting, not a throughput claim. The candidate bound assumes an overall cap of k per left record. The recall factorization applies to final pair decisions restricted to a fixed candidate set and counts omitted true links as misses.

[PNG](08-sparse-scale.png) · [PDF](08-sparse-scale.pdf) · [Editable source](08-sparse-scale.tex)

### 09-agentic-pipeline

Agentic map–reduce separates planning, tool use and validation. The operator order is specified; the optional planner derives instructions and execution settings. Map agents process units or bounded batches and retain per-unit outputs. One reducer combines those outputs with access to the same tools. The green check represents this notebook's independent source validation.

**Evidence boundary.** Conceptual API flow, checked against LOTUS 1.2.4 and source commit `136ae4f4a344a2f75d89f811e516dfcb0de30e46`. The three lanes illustrate parallel work; they do not establish a speedup. Source validation is outside the native agentic operators.

[PNG](09-agentic-pipeline.png) · [PDF](09-agentic-pipeline.pdf) · [Editable source](09-agentic-pipeline.tex)

### 10-agentic-evidence-loop

The recorded Coastal trace shows GPT-6 Luna reading three revisions, submitting the highest approved row to a calculator, and returning its result with source identifiers. The calculator reports 200 eligible cases, 110 completed cases and 55% completion. Independent checks verify the selected rows, copied values, units, citations and reducer output across all three offices. Pooling their counts gives 1,130 completed cases out of 1,600 eligible cases, or 70.6% after rounding.

**Evidence boundary.** Abbreviated observable calls and results from the validated [6 October 2026 run](../results/agentic-office-returns-gpt-6-luna.json), plus the notebook's independent checks. The source returns are synthetic. This count-based completion rate is neither a survey-weighted estimate nor an agency response-rate standard. The full arguments, tool observations and returned outputs remain in the snapshot.

[PNG](10-agentic-evidence-loop.png) · [PDF](10-agentic-evidence-loop.pdf) · [Editable source](10-agentic-evidence-loop.tex)

### 11-linkage-semantic-overview

Two methods answer the same identity question about the Amina Patel records. The figure shows the comparison vector and the effect of name agreement. The semantic join uses a written condition and a specified model procedure. The lower panels summarize LOTUS’s operator interface and FDJ’s candidate screening and reference-LLM refinement.

**Evidence boundary.** The records and probabilities are constructed. The LLM return is conditional, not a recorded prediction. This figure reports no FDJ execution or comparative performance result.

Sources: [Fellegi–Sunter](https://doi.org/10.1080/01621459.1969.10501049), [Patel et al., LOTUS §§2.2–2.4](https://www.vldb.org/pvldb/vol18/p4171-patel.pdf#page=3) and [Zeighami, Shankar and Parameswaran, FDJ §§2–3](https://arxiv.org/html/2512.05399v1#S2).

[PNG](11-linkage-semantic-overview.png) · [PDF](11-linkage-semantic-overview.pdf) · [Editable source](11-linkage-semantic-overview.tex)

### 12-biodex-workflow

The inputs and output rows come from the saved BioDEX experiment. Model predictions and published reaction annotations are shown separately.

**Evidence boundary.** Eight selected articles crossed with eight categories. The two displayed accepted pairs are actual GPT-6 Luna outputs. An absent published annotation is a dataset-negative label, not proof of clinical absence; this is category assignment, not entity linkage.

Article excerpts: [Garcia et al.](https://doi.org/10.1016/j.abd.2020.07.008), CC BY 4.0; dataset: [BioDEX](https://arxiv.org/abs/2305.13395).

[PNG](12-biodex-workflow.png) · [PDF](12-biodex-workflow.pdf) · [Editable source](12-biodex-workflow.tex)

### 13-fever-workflow

The two displayed claims share the same evidence but receive opposite decisions. The recorded filter uses supplied evidence; BM25 retrieval is a separate illustration over the ten-sentence corpus.

**Evidence boundary.** Actual FEVER claims 70373 and 143488 and their saved GPT-6 Luna decisions. Annotated evidence is supplied directly to the support filter. The separate retrieval illustration uses a gold-derived miniature corpus; it is not open-Wikipedia retrieval or full FEVER evaluation.

Sources: [FEVER](https://aclanthology.org/N18-1074/) and its archived Wikipedia evidence, CC BY-SA 3.0.

[PNG](13-fever-workflow.png) · [PDF](13-fever-workflow.pdf) · [Editable source](13-fever-workflow.tex)

The machine-readable [manifest](manifest.json) records source pointers and asset hashes.

### 14-em-learning

EM repeats two steps over many training pairs. The E step computes match probabilities. The M step uses fractional counts to update the parameters. This diagram shows a common Splink route with separately estimated, fixed `u` values. After fitting, the model scores new record pairs.

**Evidence boundary.** This is a training schematic. It reports no numerical convergence trace. The 80% probability only illustrates fractional counts. The notebook's EM arm uses this route. The supervised reference estimates `m` from known matching pairs. Training blocks restrict which comparisons an EM session can estimate.

Sources: [Splink parameter estimation](https://moj-analytical-services.github.io/splink/demos/tutorials/04_Estimating_model_parameters.html) and [the Splink author's EM explanation](https://www.robinlinacre.com/em_intuition/).

[PNG](14-em-learning.png) · [PDF](14-em-learning.pdf) · [Editable source](14-em-learning.tex)
