# Experiment comparison diagram

[Vector SVG](product-and-image-comparisons.svg) · [PNG](product-and-image-comparisons.png) · [Exact caption](product-and-image-comparisons-caption.md) · [Render manifest](product-and-image-comparisons.json)

This single two-panel figure introduces the native product experiment and the package-image comparison. It is a design diagram; it contains no performance chart. The SVG retains selectable text. The PNG is 2,592 × 2,116 pixels, rendered at 144 dpi from an 18 × 14.7 inch figure. Every label is at least 14 points in the source figure. Use the full available notebook width, or open the SVG for a larger view.

Reproduce from the repository root:

```bash
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/figures.py
.venv/bin/python LLM_basics/nso-semantic-workflows/adversarial-2026/figures.py --check
```

The renderer reads no experimental data or outcomes. It records Matplotlib and font provenance, its source hash, and the SVG/PNG hashes. `--check` rerenders in memory and compares bytes; the renderer also checks text against its enclosing node and enforces the 14-point minimum. Reproduction was checked with Matplotlib 3.11.2 and the recorded DejaVu Sans font. Other software/font versions are not verified.

## Scientific and visual audit

- The product input is shared observed text and the full catalog. Conventional lexical/Splink matching retains all-pair and same-shortlist comparisons; the semantic selector receives the frozen shortlist of at most 20 candidates. Lists are not padded to 20. The caption distinguishes the candidate boundary and the comparison on the same shortlist.
- Recordwise product extraction has no candidate context. Its fresh, unlabeled Splink fit uses the conventional numeric prior. This arm is explicitly DEV only; direct selection is labeled as one amended holdout.
- Every image arm starts from the same prepared query image. The OCR-text LLM receives the actual fixed Tesseract output through a separate arrow from the OCR node. The pixel LLM receives pixels directly.
- The two image extraction arms share their model/schema and downstream catalog/comparators. Both are explicitly DEV only. The free OCR and CLIP paths are labeled as 1,000-query test baselines.
- CLIP compares query images with catalog text. Slashes separate comparator runs; they do not indicate an ensemble or a serial lexical-to-Splink transformation.
- External reference links have one outgoing arrow, to scoring. Unlabeled fitting and declared prior assumptions are stated separately from evaluation labels. The identity targets are native publisher links for products and catalog-item association for ABO.
- Both panels retain target-set, NIL, review, and failure states. The scoring denominator explicitly retains candidate misses and failed calls.
- The final PNG was visually inspected on 7 October 2026. Labels fit their nodes, branches and arrows are legible, and no arrows cross unrelated processing nodes. Repeated rendering produced identical SVG/PNG bytes.

Method descriptions were checked against [product fitting code](../products/baseline.py), [image comparator code](../images/baseline.py), and the [image study description](../images/README.md). Figure scope does not establish parameter calibration, independent physical SKU identity, deployment performance, or a representative frequency of image conditions.
