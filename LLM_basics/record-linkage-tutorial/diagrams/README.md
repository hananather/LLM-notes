# Record-linkage figures

Fourteen editable TikZ figures accompany the chaptered record-linkage notebook. Each source uses the shared `diagram-style.tex`; the PDF and PNG are exported from that source.

## Build

Requires an installed TeX distribution with `standalone`, TikZ, Helvetica, and AMS math, plus Poppler (`pdftoppm`). No Python packages are required.

```sh
python3 build.py
python3 build.py 07-teacher-student
```

Select the intended TeX distribution through `PATH` if more than one is installed. PNGs are rendered at 220 dpi. For notebooks, display the PNG at the full content width; use the PDF for zooming or publication. The script writes only the named PDFs and PNGs and keeps compiler intermediates in a temporary directory.

## Visual notation

- Slate: records and data.
- Lavender: learned models, methods, and policies.
- Pink: language-model components and their fallible signals.
- Green: labels and evaluation against them.
- White: fixed operations, equations, and explanatory annotations.
- Dark arrows: data or evidence flow. A dashed lavender return in the teacher–student figure updates the student.

The chapter figures use 7-point text and math; the overview and dataset walkthroughs use 9-point text. All use strong dark connectors, role-based pastel fills, and aligned comparative paths. Captions provide the figure-level claims; the diagrams carry local stage labels.

## Captions and evidence

### 00-chapter-map

From linkage evidence to testable semantic workflows. The first four chapters establish the linkage task and recorded structured-data comparison. Chapter 5 examines source representation; Chapters 6–8 develop proposed repairs, learning methods, and scale tests.

**Evidence boundary.** Navigation diagram; no result claim.

[PNG](00-chapter-map.png) · [PDF](00-chapter-map.pdf) · [Editable source](00-chapter-map.tex)

### 01-identity-relations

Identity, shared context, and diagnostic similarity answer different questions. The constructed records show how two people can share a clinic while one person appears under different spellings. Here, Y denotes identity and H denotes shared context.

**Evidence boundary.** Illustrative records and identities are defined by construction. Diagnostic groups do not establish identity.

[PNG](01-identity-relations.png) · [PDF](01-identity-relations.pdf) · [Editable source](01-identity-relations.tex)

### 02-fs-evidence

Fellegi–Sunter converts a comparison vector into match evidence. The joint log-likelihood ratio decomposes into field contributions under conditional independence; a decision rule applies lower and upper thresholds.

**Evidence boundary.** Lowercase w denotes likelihood evidence. Posterior log odds additionally include prior log odds. The figure asserts no calibration result.

[PNG](02-fs-evidence.png) · [PDF](02-fs-evidence.pdf) · [Editable source](02-fs-evidence.tex)

### 03-semantic-execution

A semantic predicate fixes the requested relation; the chosen execution plan determines how candidate pairs are processed. Output validation and evaluation against identity labels are separate steps.

**Evidence boundary.** Schematic direct path, with optional execution optimization shown separately. No optimized LOTUS cascade was executed in the recorded benchmark. Reference-model agreement is not identity accuracy.

[PNG](03-semantic-execution.png) · [PDF](03-semantic-execution.pdf) · [Editable source](03-semantic-execution.tex)

### 04-controlled-comparison

The comparison on 6 October 2026 uses the same 225 candidate pairs, including 10 true matches. Splink 5.0.0 returns 10 TP, 0 FP, 0 FN and 215 TN; LOTUS 1.2.4 with `gpt-6-luna` returns 8 TP, 0 FP, 2 FN and 215 TN. Splink uses supervised match parameters and the language model is zero-shot. The [model snapshot](../results/lotus-febrl4-gpt-6-luna.json) preserves all 225 decisions and provider usage.

**Evidence boundary.** Recorded result on a small structured-data slice, with unequal training-label access. It does not measure end-to-end blocking recall or a language-model treatment effect.

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

The recorded Coastal trace shows GPT-6 Luna reading three revisions, submitting the highest approved row to a calculator, and returning its result with source identifiers. The calculator reports 200 eligible cases, 110 completed cases and 55% completion. Independent checks verify the selected rows, copied values, units, citations and reducer output across all three offices. Pooling their counts gives 1,130 completed cases out of 1,600 eligible cases, or 70.625%.

**Evidence boundary.** Abbreviated observable calls and results from the validated [6 October 2026 run](../results/agentic-office-returns-gpt-6-luna.json), plus the notebook's independent checks. The source returns are synthetic. This count-based completion rate is neither a survey-weighted estimate nor an agency response-rate standard. The full arguments, tool observations and returned outputs remain in the snapshot.

[PNG](10-agentic-evidence-loop.png) · [PDF](10-agentic-evidence-loop.pdf) · [Editable source](10-agentic-evidence-loop.tex)

### 11-linkage-semantic-overview

Both paths can address the same identity question. The diagram distinguishes their evidence, decision procedures and evaluation targets.

**Evidence boundary.** Conceptual comparison on a shared identity relation. General semantic joins may target other relations. Restricted candidates, reference-relative optimization and independent identity evaluation have separate error targets; no comparative performance claim.

Sources: [Fellegi–Sunter](https://doi.org/10.1080/01621459.1969.10501049), [LOTUS §§2.2–2.4](https://www.vldb.org/pvldb/vol18/p4171-patel.pdf#page=3) and [FDJ §2](https://arxiv.org/html/2512.05399v1#S2).

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
