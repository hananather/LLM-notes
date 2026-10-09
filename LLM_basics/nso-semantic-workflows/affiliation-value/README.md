# A trained conventional control for affiliation selection

This experiment tests whether supervised string and field matching closes the saved language-model selector's advantage on historical affiliation text. It uses the same 25 candidates per test query, the same observed affiliation and registry fields, and the same 644 reference annotations. It is a retrospective extension on an inspected public benchmark, not a new blind holdout or an official S2AFF implementation.

The [protocol](protocol.json) was saved before fitting. Two fixed models—scaled logistic regression and histogram gradient boosting—learn from 1,127 original training rows. Five training rows were excluded because their normalized text duplicates validation or test text; this is cross-split overlap removal, not complete deduplication. All 588 validation rows select the model family and set threshold. All 644 test rows remain unchanged. Feature definitions and both model configurations were recorded before this extension was fitted. Training labels fit the parameters; validation labels select the family and threshold. The historical test had already been inspected, and no new test result triggered retuning.

Validation selected gradient boosting at a score threshold of 0.95. It agrees with 492/644 exact annotated sets, versus 456 for the original lexical rule and 593 for the saved LLM selections including review-flagged outputs. The LLM makes 118 corrections and 17 regressions relative to the trained model. Its automatic-only policy supplies 546 correct resolved decisions over the full 644 cases, with 576 automatic decisions; the trained model supplies 492 correct decisions and resolves all 644. These policies have different coverage.

| Method | Exact annotated sets / 644 | False edges | Missed edges |
|---|---:|---:|---:|
| Original lexical rule | 456 | 174 | 158 |
| Logistic regression | 454 | 140 | 80 |
| Gradient boosting, selected on validation | 492 | 51 | 126 |
| Saved LLM, valid sets including review flags | 593 | 30 | 27 |

The trained model improves the original conventional result while leaving a 101-set gap to the saved LLM. This supports a contextual-selection advantage over these declared conventional implementations; it does not establish superiority to every trained affiliation system. Preparation took 66.0 seconds and fitting/validation selection took 1.63 seconds in the recorded environment, with no paid calls.

Each learner predicts a set with zero, one or multiple ROR IDs. The reference is the **exact annotated set**. Multiple annotations can include alternatives and should not automatically be read as simultaneous affiliations. Historical non-ROR annotation strings become empty ROR sets under the unchanged original scoring policy; those annotations can be incomplete or ambiguous.

## Information and procedure

The models receive 32 numeric features computed from query text, registry names/aliases/acronyms/location and the original retrieval scores. Features include token and character similarities, name coverage, abbreviation evidence, city/country matches, and candidate ranks/margins. Token rarity uses the registry alone. ROR identifiers, learned organization identities, LLM predictions, LLM embeddings and external descriptions are excluded.

Candidate generation searches the complete pinned registry of 102,742 organizations. Training and validation reference targets are never inserted into candidates. The existing 744 candidate lists are reused, including all test lists; the 100 previously prepared validation lists must reproduce exactly when the larger development pool is generated. The remaining 488 validation rows and retained training rows use the same retrieval code. Retrieval misses remain errors in the complete task.

The learner emits automatic decisions on every case. Its fitted scores are not asserted to be calibrated probabilities. Forty declared thresholds and the two model configurations are compared on validation exact-set agreement, then fewer false and missed edges. Every arm and threshold is retained. The selected model and both test prediction sets are saved and hashed before the scoring command reads test reference labels.

## Reproduce or verify

Use the repository's existing Python 3.11 environment with NumPy, scikit-learn and joblib. The recorded execution environment is in the preparation artifact. No command makes model API calls or downloads model weights.

From the repository root:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/affiliation-value/run.py freeze
.venv/bin/python LLM_basics/nso-semantic-workflows/affiliation-value/run.py fit
.venv/bin/python LLM_basics/nso-semantic-workflows/affiliation-value/run.py score
.venv/bin/python LLM_basics/nso-semantic-workflows/affiliation-value/run.py verify
```

`fit` prepares features if needed. With a saved seal it verifies and reuses predictions. Files are immutable: conflicting content raises an error. Re-execution that intentionally changes a method needs a new version and protocol, rather than replacement of this evidence. Fitted scikit-learn objects are cached outside Git under `../cache/affiliation-value/`; the stored scores and predictions remain inspectable without deserializing those models.

Preparing or verifying the complete source chain requires the pinned annotation CSV and registry ZIP already used by the original study in `../cache/linkage/`. Their exact public URLs, byte counts and SHA-256 hashes are in [the original source manifest](../data/linkage/sources.json). A fresh checkout must populate those source caches before these commands. Saved summary tables can be read directly without those downloads or fitted model files.

### Independent check without source downloads

The separate [independent verifier](independent_verify.py) reads committed compact labels, raw LLM responses, candidate lists, scores and predictions. It imports no implementation code and needs only Python and NumPy. From the repository root:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/affiliation-value/independent_verify.py
```

This command reproduces [the independent verification](results/independent-verification.json): 1,224 metric values, 36 paired case comparisons, the frozen validation selection and all 1,288 learned-model test decisions. It verifies the committed prediction seal without requiring the raw-source cache. The full validation labels are not reloaded by this check; the complete frozen validation sweep determines the selected model and thresholds.

Uncertainty uses 10,000 paired bootstrap resamples of 301 connected components, the largest containing 37 rows. Rows share a component when they share a reference ROR, normalized affiliation text or historical non-ROR label; these connections are transitive. Each resample retains all rows in its sampled components and estimates the row-weighted difference. The LLM's valid-set advantage over boosting is **15.7 percentage points**, with descriptive percentile bounds of **11.0–20.9 points**. For correct automatic decisions per input row, the difference is **8.4 points**, with bounds of **1.9–14.3**. Models and policies remain fixed: these bounds describe sensitivity to organization composition under an exchangeable-component assumption, exclude fitting/selection uncertainty and do not establish transfer to a new population.

The same command reproduces a fixed sensitivity excluding the four test rows whose normalized text also appears in validation. It leaves all models, thresholds and predictions unchanged. Among the remaining 640 rows, the LLM has 591 valid exact sets versus 491 for boosting: **15.6 points**, with bounds of **10.9–20.9**. The primary result retains all 644 rows. The excluded IDs, their validation counterparts and the pinned annotation-source hash are recorded in the verification file; reproducing this fixed sensitivity does not rerun the original source-overlap discovery.

## Evidence files

- [Preparation](results/preparation.json): retained counts, candidate coverage for training/validation, feature names, source hashes, reproduction checks and environment.
- [Validation selection](results/validation-selection.json): complete threshold grid, primary model, class counts and fit times.
- [Predictions](results/test-predictions.json) and [prediction seal](results/prediction-seal.json): both models' target sets, saved before scoring.
- [Evaluation](results/evaluation.json): all 644 rows, 590 single-label, 38 empty-ROR and 16 multiple-label rows, plus candidate-complete/missing strata and paired corrections/regressions.
- [Independent verification](results/independent-verification.json): reconstructed scores, separate review/invalid views, paired component-bootstrap bounds and the fixed four-row overlap sensitivity.
- [Source-consistency check](results/source-consistency-audit.json): exploratory reading of all 17 trained-correct/LLM-disagreement cases; no new adjudication or score changes.
- [Examples](results/examples.json): the first case ID where the selected learner is correct and the saved LLM is wrong, and vice versa. A missing category is recorded explicitly.
- Compressed candidate lists, 32-dimensional pair features, model scores and per-case outcomes preserve the complete comparison without placing raw tables in the reader notebook.

Three evaluation views keep review distinct from no target. **Valid annotated-set agreement** includes a valid LLM selection that carries a review flag. **Automatic resolved agreement** treats review and invalid outputs as unresolved and retains all 644 rows in the denominator. **Conditional automatic accuracy** scores only automatic outputs and displays that smaller denominator and coverage. An invalid or review output is never silently converted into a correct empty-set decision. Both learned arms are fully automatic, so their conditional and complete denominators coincide.

The original frozen LLM outputs should reproduce 593 valid annotated-set agreements; 546 correct among 576 automatic outputs; 65 reviews; and three task-invalid outputs. These are comparison checks, not new model calls. Counts are descriptive: repeated organizations and earlier benchmark inspection limit independent-trial and transfer claims. A favorable result does not establish administrative-data performance or staff-time savings.

The 1,127 retained training rows contain 1,109 distinct normalized strings. The unchanged official validation/test splits share three normalized strings, covering five validation and four test rows. No outcome-driven exclusions or revised thresholds were applied. The first trained-correct/LLM-wrong example explicitly names both Cambridge and Regina, while the reference contains Cambridge only; selecting both is reference disagreement, not clear evidence of an invented organization. The original score is retained.

## Sources and licences

The annotation CSV is from [S2AFF commit ac8d6b5](https://github.com/allenai/S2AFF/tree/ac8d6b58f42b253821c26fff8b013913317c5c72), under [Apache-2.0](../data/linkage/S2AFF-LICENSE.txt). Registry data are [ROR v1.1, 16 June 2022](https://zenodo.org/records/6657125), released under CC0. Source versions and content hashes are retained. This experiment uses the original local lexical retriever; it does not download or claim results for the official S2AFF NER/LightGBM pipeline.
