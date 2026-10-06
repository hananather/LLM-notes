# Record-linkage figures

Nine editable TikZ figures accompany the chaptered record-linkage notebook. Each source uses the shared `diagram-style.tex`; the PDF and PNG are exported from that source.

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

The figures use 7-point text and math, strong dark connectors, role-based pastel fills, and aligned comparative paths. Captions provide the figure-level claims; the diagrams carry local stage labels.

## Captions and evidence

### 00-chapter-map

From linkage evidence to a testable research programme. The first four chapters establish the task and recorded comparison; the later chapters develop proposed mechanisms and tests.

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

The recorded comparison uses the same 225 candidate pairs, including 10 true matches. Splink returns 10 TP, 0 FP, 0 FN and 215 TN; LOTUS returns 8 TP, 0 FP, 2 FN and 215 TN. Splink uses supervised match parameters and the language model is zero-shot.

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

The machine-readable [manifest](manifest.json) records source pointers and asset hashes.
