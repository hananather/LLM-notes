# LePaRD source notice

The LePaRD rows in `samples.json` and `source-responses/` come from
[rmahari/LePaRD](https://huggingface.co/datasets/rmahari/LePaRD).
The dataset card identifies its license as
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
That source license remains attached to these data. The surrounding repository's
license does not replace it. This selected data collection is distributed under
the same CC BY-NC-SA 4.0 terms.

Credit: Robert Mahari, Dominik Stammbach, Elliott Ash, and Alex Pentland.
2024. [LePaRD: A Large-Scale Dataset of Judicial Citations to Precedent](https://aclanthology.org/2024.acl-long.532/).
Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics, pages 9863–9877.

Changes: three rows were selected for illustration, wrapped in a JSON collection,
and accompanied by explanatory annotations. The 13 source fields in each
`raw_row` are unchanged. Whitespace, punctuation, spelling, dates, and names are
preserved. `source-responses/` preserves the bytes returned by the dataset API.
`source-dataset-card.md` preserves the downloaded dataset card.

The source material consists of U.S. judicial opinion text distributed through
LePaRD, whose authors used the Caselaw Access Project. These are historical text
extracts. No complete opinion, scanned page, or newly produced OCR is included.
The examples do not establish the current legal status of a rule.

See [provenance.json](provenance.json) for the dataset revision, exact request
URLs, source-code reference, retrieval time, and SHA-256 hashes.
