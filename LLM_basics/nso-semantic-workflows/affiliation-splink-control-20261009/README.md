# Supervised Splink control on historical affiliation text

The unchanged 644-row S2AFF test benchmark gives **356 exact annotated sets** for the validation-selected supervised Splink control, **492** for the retained trained tree and **593** for the saved semantic selector including review-flagged outputs. The semantic selector supplies **546 correct automatic decisions** over all 644 cases, with 576 automatic outputs. These are distinct acceptance policies.

| Method | Exact annotated sets / 644 | False edges | Missed edges |
|---|---:|---:|---:|
| Compact supervised Splink | 305 | 96 | 284 |
| Expanded supervised Splink, selected on validation | 356 | 173 | 202 |
| Retained trained tree | 492 | 51 | 126 |
| Saved semantic selector, valid sets including review | 593 | 30 | 27 |
| Saved semantic selector, automatic resolved sets | 546 | 29 | 44 |

This adds an actual Fellegi–Sunter comparison to the earlier trained control. It is a retrospective extension on an inspected public benchmark. The stronger tree remains the main conventional comparison. The result concerns the two declared Splink representations; it does not establish superiority to every possible Splink configuration or an official specialist affiliation linker.

## Method and information

All methods use the same frozen 25 candidates per query and the same permitted source affiliation and registry names, aliases, acronyms and locations. Candidate retrieval originally searched the complete historical 102,742-organization registry. True targets are never inserted. Original training labels supervise the parameters; all 588 validation rows select the representation and threshold. Five observed duplicate training strings were excluded under the earlier unchanged rule, leaving 1,127 training rows. The complete 644-row test remains unchanged.

Each comparison categorizes source/candidate evidence. The compact model combines joint name evidence, registry character TF-IDF and location evidence. The expanded model adds word TF-IDF, registry-only token-rarity coverage and the position of contained names. Their definitions and levels are in [protocol.json](protocol.json). Additive smoothing of one count per level estimates match/nonmatch frequencies from the original training labels. The prior describes the retrieved candidate population, rather than random pairs from the full registry.

Native Splink 5 performs all likelihood scoring. A fixed candidate pair is encoded as one observed feature row and one opaque same-key anchor row; a blocking equality permits precisely that comparison. The anchor implements categorical comparisons of precomputed pair diagnostics. Its opaque key contains no identity information and is never a match feature. This is supervised Splink scoring of the fixed diagnostic representation, rather than a raw string-matching configuration. Native scores are checked against independent likelihood arithmetic; the maximum absolute differences are saved.

The Fellegi–Sunter model multiplies comparison likelihood ratios. Some lexical comparisons are dependent, so its scores are not established as calibrated probabilities. The expanded representation and threshold 0.925 were selected by validation exact-set agreement, with fewer false/missed edges as tie breaks. The compact representation selected 0.475. Every candidate above the selected threshold is emitted; empty and multiple sets are permitted.

Test scores, native model settings and predictions were hashed and sealed before the evaluation-stage reference read. Every declared representation remains saved, and there was no adjustment after test scoring. The original artifacts remain read-only inputs. Grouped bootstrap summaries account for repeated positive reference organizations and normalized duplicate strings; these are descriptive intervals for the historical source, not NSO population intervals. Reference ROR alternatives are preserved as original annotated sets rather than reinterpreted as simultaneous institutional identities.

## Reproduce

The prerequisite is the bundled [trained-control feature packet](../affiliation-value/README.md). It contains observed source/registry features for all original partitions and is verified against its own seal. No download or model API call is needed. Use the repository's pinned Python 3.11 environment with Splink 5.0.0.

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/affiliation-splink-control-20261009/run.py verify
.venv/bin/python LLM_basics/nso-semantic-workflows/affiliation-splink-control-20261009/independent_verify.py
```

The `freeze`, `fit` and `score` modes implement the staged experiment in a separate output copy. Existing evidence is immutable; rerunning `fit` verifies and reuses its sealed outputs. `results/evaluation.json` retains exact-set counts, edge errors, NIL errors, review accounting, paired corrections/regressions and descriptive false-link-cost sensitivity at 1, 2 and 5. The experiment made no paid calls.

Original source attribution and licences remain in [the maintained linkage protocol](../protocol-linkage.md) and [the trained-control documentation](../affiliation-value/README.md).
