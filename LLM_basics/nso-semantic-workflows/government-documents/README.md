# Government text: context linked to quoted precedent

This folder contains three real civil and tax examples from
[LePaRD](https://huggingface.co/datasets/rmahari/LePaRD), a dataset of judicial
contexts linked to quotations from earlier cases. Each example preserves the
complete `destination_context` and `quote` fields returned by the source API.
These are text extracts, not complete opinions, PDFs, or new OCR outputs.

The collection makes the input relation inspectable. It is a purposive showcase,
with no model predictions, accuracy estimate, or measured advantage for semantic
matching. It illustrates context-to-precedent linkage relevant to citation-oriented
semantic joins; it is not a replication of the FDJ Citations benchmark.

## Three observed links

| Example | Later opinion and date | Earlier opinion and date | Source row / passage |
|---|---|---|---|
| Consent-decree interpretation | United States v. Associated Credit Bureaus, Inc.; 1972-07-14 | United States v. Atlantic Refining Co.; 1959-06-08 | `12` / `8895_1` |
| University admissions-tax collection | Regents of University System of Georgia v. Page; 1936-01-20 | Pool v. Walsh; 1922-08-07 | `23` / `5706389_1` |
| Ambiguity: a generic statutory phrase | Hubbard Inv. Co. v. Brast; 1932-06-13 | Pool v. Walsh; 1922-08-07 | `24` / `5706389_0` |

The first context discusses government acquiescence in an interpretation of a
consent decree. Its linked quotation describes when an accepted interpretation
should stand. The second context distinguishes a taxpayer from an alleged debtor
who collected admissions taxes. Its quotation concerns executive enforcement.

The third quotation is only “assessment or collection of any tax.” Its wording
does not identify a unique precedent. Keep the observed positive link, but do not
turn every uncited pair into a negative. Multiple opinions can quote a statutory
phrase; multiple passages may support an argument. This collection does not
enumerate all relevant precedents or independently adjudicate the extracted links.

## Exact field mapping

Each `examples[i].raw_row` in [samples.json](samples.json) contains all 13 original
source fields. Annotations outside `raw_row` are explanatory interpretations.

| Source field | Meaning in this showcase |
|---|---|
| `destination_context` | Complete available context before the quotation in the later opinion; the retrieval query |
| `quote` | The quoted text as extracted from the later opinion; an observed target excerpt |
| `dest_id` | Caselaw Access Project ID for the later, citing opinion |
| `dest_name`, `dest_date`, `dest_court`, `dest_cite` | Later opinion's name, date, court, and citation |
| `source_id` | Caselaw Access Project ID for the earlier, quoted opinion |
| `source_name`, `source_date`, `source_court`, `source_cite` | Earlier opinion's name, date, court, and citation |
| `passage_id` | Source-assigned passage identifier, incorporating `source_id` and a passage index |

The source card distinguishes a row's quotation from the canonical passage text
in `passage_dict.json`. Slightly different quotations can share a passage ID.
This showcase preserves the quotation and ID. It does not download that large
dictionary or claim that the quotation is the complete canonical passage.

The authors' [extraction code](https://github.com/rmahari/LePaRD/blob/cf9d84534a24c118041061eae7bc3f4f9b69e2a9/src/data/precedent_data_extraction.py)
extracts quotations, links them to cited opinions, matches quoted wording to
source sentences, and collects preceding context. The relation here is therefore
labelled `dataset_extracted_quotation_link`, not an independently reviewed legal
relevance judgment.

## Separate extraction from semantic retrieval

Start with the information available in the intended task:

1. If full opinions contain citations, parse and normalize those citations, then
   resolve them against case metadata. If a quotation is supplied, try exact and
   normalized quotation lookup in the candidate texts. Those are conventional
   extraction and retrieval baselines.
2. If only the preceding context is available, rank candidate passages with a
   lexical baseline such as BM25 before comparing semantic retrieval. The three
   observed target excerpts here are not a defensible candidate universe.
3. Hide `source_id`, `source_name`, `source_cite`, and `passage_id` from query
   inputs: they disclose the target. Audit case names and citations inside the
   context, too. Supplying `quote` to a context-to-passage prediction task also
   discloses its target text. Retain these fields in a separate provenance view.
4. Decide the target relation before labelling: recovering an observed citation
   differs from finding every legally relevant passage. Freeze candidates and
   adjudicate ambiguous pairs before reporting precision or recall.

No baseline or semantic model has been evaluated on these examples. An exact
substring check between a preceding context and its following quotation would
test the wrong input relation. The university-tax context also cites *United
States v. Johnston*. That is part of its argument; the recorded target for this
particular quotation is *Pool v. Walsh*.

## Read and reproduce

No API token or model service is required. The script uses Python's standard
library. Offline verification and a fresh re-fetch passed on Python 3.11.8 on
macOS on 2026-10-08 UTC. Other environments have not been tested. From the
repository root:

```bash
python3 LLM_basics/nso-semantic-workflows/government-documents/retrieve_lepard.py verify
```

This checks response hashes, row hashes, all six text fields, and agreement
between the selected rows and their preserved API responses. It uses no network.
To re-fetch only the same three rows into a new directory:

```bash
python3 LLM_basics/nso-semantic-workflows/government-documents/retrieve_lepard.py fetch \
  --output /tmp/lepard-three-row-check
```

The script uses the documented Hugging Face
[`/rows` endpoint](https://huggingface.co/docs/dataset-viewer/en/rows), requesting
one row at offsets 12, 23, and 24. It checks the dataset revision before and after
retrieval, checks each response's `x-revision` header, and rejects changed row
content. Existing output directories are preserved. Each response has a 2 MB
limit. It does not download the corpus or call a model.

For a notebook display:

```python
import json
from pathlib import Path

folder = Path("nso-semantic-workflows/government-documents")
examples = json.loads((folder / "samples.json").read_text())["examples"]
for example in examples:
    row = example["raw_row"]
    print(example["title"])
    print(row["dest_name"], row["dest_date"], "→", row["source_name"], row["source_date"])
    print("Full available context:", row["destination_context"])
    print("Observed quotation:", row["quote"])
    print(example["caution"])
```

The path above assumes the notebook runs in `LLM_basics/`. Each saved API response
reports no truncated cells. The viewer reports `partial=true`; its accessible
row count is not the full dataset size. Selection was limited to inspecting rows
0–29 and retaining three civil/tax examples. It is not a random sample.

Data retain [their separate CC BY-NC-SA 4.0 source notice](SOURCE-NOTICE.md).
[provenance.json](provenance.json) records request URLs, revision, retrieval time,
and hashes. [source-dataset-card.md](source-dataset-card.md) preserves the card
retrieved at that revision.
