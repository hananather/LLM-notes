# Official Splink FEBRL4 benchmark data

The tutorial uses the standard **FEBRL4a/FEBRL4b two-table person linkage
benchmark supplied by Splink**. It has 5,000 original records in table A and
5,000 corrupted duplicates in table B: one record per person in each table,
with 5,000 true cross-table links. These are synthetic benchmark records.

The [official Splink link-only example](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/febrl4.html)
uses these tables and derives identity labels from their record IDs. The
[Python Record Linkage Toolkit dataset documentation](https://recordlinkage.readthedocs.io/en/latest/ref-datasets.html)
describes how FEBRL4 was generated. The
[original FEBRL manual](https://users.cecs.anu.edu.au/~Peter.Christen/Febrl/febrl-0.3/febrldoc-0.3/node4.html)
describes the Australian National University project and its dataset generator.
Some Splink dataset descriptions conflate FEBRL with the German cancer-registry
comparison-vector dataset; that description does not identify the provenance
of these person records.

## Source, license, and modifications

The CSV sources come from the official
[Splink datasets repository](https://github.com/moj-analytical-services/splink_datasets),
pinned to commit `75876d806d9eff72072d21878150ee50a96d5f41`, dated July 26, 2023.
The public [Splink dataset API](https://moj-analytical-services.github.io/splink/api_docs/datasets.html)
loads the same benchmark as `splink_datasets.febrl4a` and
`splink_datasets.febrl4b`. In Splink 5 these properties return PyArrow tables;
`.to_pandas()` converts a table to a pandas DataFrame.

These files derive indirectly from FEBRL provided by the **Australian National
University**, developed by Peter Christen and Tim Churches. The original FEBRL
documentation places its documentation and data files under the
[ANU Open Source License 1.2](https://users.cecs.anu.edu.au/~Peter.Christen/Febrl/febrl-0.3/febrldoc-0.3/node105.html).
The Splink datasets repository also supplies an MIT license, copyright 2023
MoJ Analytical Services. Copies of both notices are retained here as
`LICENSE-ANUOS-1.2.txt` and `LICENSE-splink-datasets.txt`; the repository's notice
does not establish that it replaces the original FEBRL terms.

Modification date: **October 6, 2026**. Preparation strips padding from CSV
headers and values, preserves observed fields as strings, replaces source record
IDs with opaque IDs, moves benchmark identity labels to a separate truth file,
and adds deterministic split and slice metadata. It does not alter observed
values numerically, generate new records, or infer new identity labels. The
modified data remain subject to their original notices, available alongside the
reproducible preparation source.

| Upstream file | SHA-256 |
| --- | --- |
| `dataset4a.csv` | `797ffc5072b5d6cbfdcd0bcb603b92be8916fa67fe8e180db366994c5389d3a0` |
| `dataset4b.csv` | `af0330f0c355b7b12006ac0b8a2de23539d3bc868153f9416fc5ae43cc5f401c` |

Exact download URLs, source/license hashes, output hashes, and counts are in
`manifest.json`. Raw input downloads are cached locally in ignored `upstream/`.
The notebook reads the bundled prepared CSVs without downloading data.

## Observed records and separate truth

| File | Columns | Rows |
| --- | --- | ---: |
| `records.csv` | `source, unique_id, given_name, surname, street_number, address_1, address_2, suburb, postcode, state, date_of_birth, soc_sec_id` | 10,000 |
| `record_truth.csv` | `unique_id, entity_id, split` | 10,000 |
| `evaluation_slice.csv` | `unique_id, source` | 30 |
| `validation_slice.csv` | `unique_id, source` | 200 |

`source` is `febrl4a` or `febrl4b`. `unique_id` is the first 24 hexadecimal
characters of SHA-256 of `<source>:<original rec_id>`, used to locate records.
The original `rec_id` contains the entity identity; it is removed from observed
records. Identity is recovered only for `record_truth.csv`, from the entity
digits in `rec-<digits>-org` and `rec-<digits>-dup-0`.

Names, address components, dates, and identifiers remain text. Leading zeros
are retained. Empty source cells remain empty CSV cells and can be loaded as
nullable strings with `pd.read_csv(path, dtype="string")`. Labels, split values,
source names, and opaque IDs are metadata; matching prompts and comparisons
should use only the native observed fields. The synthetic `soc_sec_id` field is
an observed benchmark identifier, separate from its hidden identity labels.

## Fixed entity split

Format the entity digits as `entity_<four-digit number>`. Compute SHA-256 of
`lotus-febrl4-split-v1:<entity_id>`, convert the full hexadecimal digest to an
integer, and take it modulo 100. Buckets 0–69 are training, 70–84 validation,
and 85–99 test. Both records for a person stay in the same split.

| Split | Entities | Table A records | Table B records | Total records |
| --- | ---: | ---: | ---: | ---: |
| Train | 3,486 | 3,486 | 3,486 | 6,972 |
| Validation | 733 | 733 | 733 | 1,466 |
| Test | 781 | 781 | 781 | 1,562 |
| Total | 5,000 | 5,000 | 5,000 | 10,000 |

The nominal 70/15/15 proportions are hash buckets rather than exact quotas.
Benchmark identity is used to prevent entity overlap across splits and evaluate
predictions. It is excluded from matching features. This separation does not
establish whether a pretrained language model encountered this public benchmark.

## Evaluation and validation slices

Order test entities by SHA-256 of
`lotus-febrl4-evaluation-v1:<entity_id>` and take the first 20. Table A uses the
first 15; table B uses the last 15. The shared ten entities yield **225 possible
pairs: 10 matches and 215 nonmatches**. Selection is fixed before predictions
and uses no matching scores or errors.

For threshold selection, order validation entities independently by SHA-256 of
`lotus-febrl4-validation-v1:<entity_id>` and take the first 100. Include both
source records for each entity. The 100-by-100 validation slice has **10,000
pairs: 100 matches and 9,900 nonmatches**. It shares no entities with training
or evaluation.

These slices control runtime and model-call cost while keeping both methods on
the same observations. Their match proportions differ from the full 5,000-by-
5,000 cross product and from operational datasets. Slice accuracy and cost are
teaching results on a synthetic benchmark; they do not establish performance on
large survey, household, or business linkage jobs.

## Rebuild and verify

Preparation needs only Python's standard library. From the repository root:

```bash
.venv/bin/python LLM_basics/record-linkage-tutorial/prepare_data.py
```

The script downloads missing pinned sources, verifies source hashes, and
rewrites the prepared CSVs, license copies, and manifest deterministically.
`--refresh` redownloads sources and rejects any hash change. It checks two
5,000-record tables, one record per entity per source, unique opaque IDs, split
separation, and the fixed slice composition. Independent CSV checks verify
entity counts, exact match counts, missing-value handling, leading-zero
preservation, and reproducible output hashes. These checks concern data
preparation; linker accuracy requires running the tutorial.
