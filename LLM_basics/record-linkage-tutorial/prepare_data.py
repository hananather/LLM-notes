"""Rebuild bundled FEBRL4 CSVs from hash-verified official Splink sources.

The notebook needs only the bundled CSVs. Preparation uses Python's standard
library and downloads inputs only when absent from the local upstream cache.
Data derive from the Australian National University's FEBRL benchmark; see the
data directory's provenance, modification description, and license notices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import urllib.request


DATA_DIR = Path(__file__).resolve().parent / "data"
UPSTREAM_DIR = DATA_DIR / "upstream"
UPSTREAM_COMMIT = "75876d806d9eff72072d21878150ee50a96d5f41"
SPLIT_SEED = "lotus-febrl4-split-v1"
EVALUATION_SEED = "lotus-febrl4-evaluation-v1"
VALIDATION_SEED = "lotus-febrl4-validation-v1"
FEATURE_COLUMNS = [
    "given_name", "surname", "street_number", "address_1", "address_2",
    "suburb", "postcode", "state", "date_of_birth", "soc_sec_id",
]
SOURCE_BASE = f"https://raw.githubusercontent.com/moj-analytical-services/splink_datasets/{UPSTREAM_COMMIT}"
SOURCES = {
    "dataset4a.csv": {
        "url": f"{SOURCE_BASE}/data/febrl/dataset4a.csv",
        "sha256": "797ffc5072b5d6cbfdcd0bcb603b92be8916fa67fe8e180db366994c5389d3a0",
    },
    "dataset4b.csv": {
        "url": f"{SOURCE_BASE}/data/febrl/dataset4b.csv",
        "sha256": "af0330f0c355b7b12006ac0b8a2de23539d3bc868153f9416fc5ae43cc5f401c",
    },
    "LICENSE-splink-datasets.txt": {
        "url": f"{SOURCE_BASE}/LICENSE",
        "sha256": "5fc8625710eab2a1e924057ddd8bd5e840d71b4ccc897ad3a21c1a95c2b8763c",
    },
    "LICENSE-ANUOS-1.2.html": {
        "url": "https://users.cecs.anu.edu.au/~Peter.Christen/Febrl/febrl-0.3/febrldoc-0.3/node105.html",
        "sha256": "48a8ef85dcd6a1ba75d7aa9834a4107de00ad088bb0d2ff688a8398d177c78a5",
    },
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rank(seed: str, entity_id: str) -> str:
    return digest(f"{seed}:{entity_id}".encode("utf-8"))


def split_for(entity_id: str) -> str:
    bucket = int(rank(SPLIT_SEED, entity_id), 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def ensure_sources(refresh: bool) -> None:
    UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
    for filename, metadata in SOURCES.items():
        path = UPSTREAM_DIR / filename
        if refresh or not path.exists():
            request = urllib.request.Request(metadata["url"], headers={"User-Agent": "febrl4-tutorial"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            if digest(content) != metadata["sha256"]:
                raise ValueError(f"Upstream content changed: {filename}")
            path.write_bytes(content)
        if digest(path.read_bytes()) != metadata["sha256"]:
            raise ValueError(f"Cached source hash differs: {filename}")


def write_csv(filename: str, columns: list[str], rows: list[dict[str, str]]) -> None:
    with (DATA_DIR / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class LicenseText(HTMLParser):
    """Keep the complete original license text, including image alt text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.in_body = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "body":
            self.in_body = True
        if self.in_body and tag in {"p", "br", "h1", "h2", "h3", "li", "dt", "dd"}:
            self.parts.append("\n")
        if self.in_body and tag == "img":
            self.parts.append(dict(attrs).get("alt") or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "body":
            self.in_body = False
        if self.in_body and tag in {"p", "h1", "h2", "h3", "li", "dt", "dd"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_body:
            self.parts.append(data)


def prepare(refresh: bool = False) -> dict:
    ensure_sources(refresh)
    records: list[dict[str, str]] = []
    truth: list[dict[str, str]] = []
    entity_records: dict[str, dict[str, str]] = {}
    for filename, source, suffix in (("dataset4a.csv", "febrl4a", "org"),
                                     ("dataset4b.csv", "febrl4b", "dup-0")):
        text = (UPSTREAM_DIR / filename).read_text(encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(text), skipinitialspace=True))
        if len(rows) != 5000:
            raise ValueError(f"Unexpected row count: {filename}")
        for original in rows:
            # No numeric conversion: zeros in dates, postcodes and IDs survive.
            row = {key.strip(): value.strip() for key, value in original.items()}
            if set(row) != {"rec_id", *FEATURE_COLUMNS}:
                raise ValueError(f"Unexpected columns: {filename}")
            match = re.fullmatch(rf"rec-(\d+)-{suffix}", row.pop("rec_id"))
            if match is None:
                raise ValueError("Unexpected FEBRL record ID")
            entity_id = f"entity_{int(match.group(1)):04d}"
            rec_id = f"rec-{match.group(1)}-{suffix}"
            unique_id = digest(f"{source}:{rec_id}".encode("utf-8"))[:24]
            if source in entity_records.setdefault(entity_id, {}):
                raise ValueError("Duplicate source record for entity")
            entity_records[entity_id][source] = unique_id
            records.append({"source": source, "unique_id": unique_id, **row})
            truth.append({"unique_id": unique_id, "entity_id": entity_id,
                          "split": split_for(entity_id)})
    assert len(entity_records) == 5000
    assert all(set(ids) == {"febrl4a", "febrl4b"} for ids in entity_records.values())
    assert len({row["unique_id"] for row in records}) == 10000

    test_entities = sorted([entity for entity in entity_records if split_for(entity) == "test"],
                           key=lambda entity: rank(EVALUATION_SEED, entity))
    assert len(test_entities) >= 20
    selected = test_entities[:20]
    evaluation = [{"unique_id": entity_records[entity][source], "source": source}
                  for source, entities in (("febrl4a", selected[:15]), ("febrl4b", selected[5:]))
                  for entity in entities]
    validation_entities = sorted(
        [entity for entity in entity_records if split_for(entity) == "validation"],
        key=lambda entity: rank(VALIDATION_SEED, entity),
    )[:100]
    assert len(validation_entities) == 100
    validation = [{"unique_id": entity_records[entity][source], "source": source}
                  for source in ("febrl4a", "febrl4b") for entity in validation_entities]

    truth_by_id = {row["unique_id"]: row for row in truth}
    assert all(truth_by_id[row["unique_id"]]["split"] == "test" for row in evaluation)
    assert all(truth_by_id[row["unique_id"]]["split"] == "validation" for row in validation)
    assert {row["unique_id"] for row in evaluation}.isdisjoint(
        row["unique_id"] for row in validation)
    left_entities = {truth_by_id[row["unique_id"]]["entity_id"] for row in evaluation
                     if row["source"] == "febrl4a"}
    right_entities = {truth_by_id[row["unique_id"]]["entity_id"] for row in evaluation
                      if row["source"] == "febrl4b"}
    assert len(left_entities) == len(right_entities) == 15
    assert len(left_entities & right_entities) == 10

    records.sort(key=lambda row: (row["source"], row["unique_id"]))
    truth.sort(key=lambda row: row["unique_id"])
    write_csv("records.csv", ["source", "unique_id", *FEATURE_COLUMNS], records)
    write_csv("record_truth.csv", ["unique_id", "entity_id", "split"], truth)
    write_csv("evaluation_slice.csv", ["unique_id", "source"], evaluation)
    write_csv("validation_slice.csv", ["unique_id", "source"], validation)
    (DATA_DIR / "LICENSE-splink-datasets.txt").write_bytes(
        (UPSTREAM_DIR / "LICENSE-splink-datasets.txt").read_bytes())
    license_parser = LicenseText()
    license_parser.feed((UPSTREAM_DIR / "LICENSE-ANUOS-1.2.html").read_text(encoding="latin-1"))
    license_text = "\n".join(line.strip() for line in "".join(license_parser.parts).splitlines()
                             if line.strip()) + "\n"
    (DATA_DIR / "LICENSE-ANUOS-1.2.txt").write_text(license_text, encoding="utf-8")

    counts = {
        "records": 10000, "febrl4a": 5000, "febrl4b": 5000, "entities": 5000,
        "true_cross_source_links": 5000,
        "splits": {split: {"entities": sum(split_for(entity) == split for entity in entity_records),
                           "records": sum(row["split"] == split for row in truth)}
                   for split in ("train", "validation", "test")},
        "evaluation_slice": {"left_records": 15, "right_records": 15, "entities": 20,
                             "pairs": 225, "matches": 10, "nonmatches": 215},
        "validation_slice": {"left_records": 100, "right_records": 100, "entities": 100,
                             "pairs": 10000, "matches": 100, "nonmatches": 9900},
        "empty_values_by_field": {field: sum(row[field] == "" for row in records)
                                  for field in FEATURE_COLUMNS},
    }
    outputs = ["records.csv", "record_truth.csv", "evaluation_slice.csv", "validation_slice.csv",
               "LICENSE-splink-datasets.txt", "LICENSE-ANUOS-1.2.txt"]
    manifest = {
        "dataset": "Official Splink FEBRL4a / FEBRL4b; standard synthetic persons benchmark",
        "upstream_repository": "https://github.com/moj-analytical-services/splink_datasets",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_commit_date": "2023-07-26T16:46:32Z",
        "sources": SOURCES,
        "dataset_api": "from splink import splink_datasets; splink_datasets.febrl4a; splink_datasets.febrl4b",
        "license_provenance": "Splink dataset repository: MIT; original FEBRL documentation/data files: ANUOS 1.2. Both notices retained.",
        "modifications_date": "2026-10-06",
        "modifications": ["Strip CSV header and value padding without numeric conversion",
                          "Replace record IDs with opaque source-specific SHA256 prefixes",
                          "Store identity labels separately; add deterministic split/slice metadata"],
        "truth_rule": "Entity digits in rec-<digits>-org / rec-<digits>-dup-0 determine benchmark truth; not inferred from features.",
        "split_rule": {"seed": SPLIT_SEED, "hash": "SHA256(seed + ':' + entity_id), hexadecimal integer modulo 100",
                       "buckets": "0-69 train; 70-84 validation; 85-99 test"},
        "evaluation_selection": {"seed": EVALUATION_SEED,
                                 "rule": "First 20 test entities by SHA256(seed + ':' + entity_id); left first 15, right last 15."},
        "validation_selection": {"seed": VALIDATION_SEED,
                                 "rule": "First 100 validation entities by SHA256(seed + ':' + entity_id); all records on both sides."},
        "counts": counts,
        "output_sha256": {filename: digest((DATA_DIR / filename).read_bytes()) for filename in outputs},
        "checks": ["Pinned input hashes", "5000 unique entities and records in each source",
                   "Exactly one original and one duplicate per entity", "10000 unique opaque IDs",
                   "All matching fields retained as text; empty missing values retained",
                   "Entities assigned to one split", "Evaluation and validation subsets disjoint",
                   "225 evaluation pairs with 10 matches; 10000 validation pairs with 100 matches"],
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Redownload sources and verify expected hashes.")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.refresh), indent=2))
