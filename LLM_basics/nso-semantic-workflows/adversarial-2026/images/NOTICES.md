# Data and model notices

## Amazon Berkeley Objects

The product metadata and images are from **Amazon Berkeley Objects (ABO)**, published by Amazon. Cite the [dataset registry](https://registry.opendata.aws/amazon-berkeley-objects/) and [dataset documentation](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/README.md). The registry specifies [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). It notes that the assets changed from CC BY-NC to CC BY in 2023; older announcements may show the former license.

This study selects grocery records, retains native title/brand/bullet/model text, assigns opaque record IDs, derives image-to-catalog association labels, and resizes images for evaluation. These selections, transforms and annotations are study additions, not publisher endorsements. Preserve attribution and identify these changes when redistributing derivatives. The data license does not establish identity validity, model suitability, or rights to use trademarks beyond the applicable license and law.

Original image files and model weights remain in an external cache. The repository contains descriptive catalog subsets, derived OCR, decisions and provenance manifests. `evaluator/source-files.jsonl` pins every metadata file by URL and SHA-256; `evaluator/fetch-manifest.jsonl` records original image URLs, SHA-256, dimensions and prepared-image hashes.

## CLIP

The frozen visual comparator uses OpenAI CLIP ViT-B/32. See the [official repository](https://github.com/openai/CLIP), [MIT license](https://github.com/openai/CLIP/blob/main/LICENSE), [model card](https://github.com/openai/CLIP/blob/main/model-card.md), and [Hugging Face model repository](https://huggingface.co/openai/clip-vit-base-patch32). The pinned model revision is `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Its research-use model card describes limitations in fine-grained classification, counting and domain transfer. This study is a bounded research evaluation; the model supplies no identity annotation or deployment acceptance criterion.

No model weights are redistributed here. Model download and inference use Transformers. Keep the applicable upstream licenses with any later redistributed code or weights. Metadata about the documentation downloads, including SHA-256, is in `source-notices-manifest.json`.

## OCR and linkage software

Tesseract OCR is provided under Apache License 2.0; see its [official repository](https://github.com/tesseract-ocr/tesseract). Splink is provided under MIT; see its [official repository](https://github.com/moj-analytical-services/splink). Their output scores and extracted fields are algorithmic evidence, not source truth. Dependency versions used for this run are recorded in the README and result diagnostics.
