"""Curate licensed source graphs; this module alone reads identity mapping files."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

from product_features import OBSERVED_FIELDS, family_keys, normalize

HERE = Path(__file__).resolve().parent
SEED = "products-20261007-v1"
CURATION_VERSION = "v2"
SOURCES = {
    "abt-buy": {
        "archive": "Abt-Buy.zip", "sha256": "b9ff2937d97371b7a00b0a97033213bfbdee24e8f1a0718a41ea8364472a8ff8",
        "left": "Abt.csv", "right": "Buy.csv", "mapping": "abt_buy_perfectMapping.csv",
        "left_key": "idAbt", "right_key": "idBuy", "title": "name", "development_components": 0,
    },
    "amazon-google": {
        "archive": "Amazon-GoogleProducts.zip", "sha256": "fcd72e223d51ce5b0d6b1d68034a438ba95cf2940431e209a57910b067cc4811",
        "left": "Amazon.csv", "right": "GoogleProducts.csv", "mapping": "Amzon_GoogleProducts_perfectMapping.csv",
        "left_key": "idAmazon", "right_key": "idGoogleBase", "title": "title", "development_components": 125,
    },
}


def digest_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def stable_id(*parts):
    return sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"Different frozen content already exists: {path}")
        return
    with path.open("xb") as stream:
        stream.write(raw)


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def groups(self):
        out = defaultdict(list)
        for i in sorted(self.parent):
            out[self.find(i)].append(i)
        return list(out.values())


def read_csv(archive, name):
    data = archive.read(name)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin1")
    return list(csv.DictReader(io.StringIO(text)))


def curate(dataset, source_directory):
    spec = SOURCES[dataset]
    source = source_directory / spec["archive"]
    if digest_file(source) != spec["sha256"]:
        raise ValueError("Source archive hash mismatch")
    archive_destination = HERE / "sources" / source.name
    archive_destination.parent.mkdir(exist_ok=True)
    if archive_destination.exists() and digest_file(archive_destination) != spec["sha256"]:
        raise ValueError("Preserved source differs")
    if not archive_destination.exists():
        shutil.copyfile(source, archive_destination)
    records, metadata, source_map = [], {}, []
    lookup = {}
    with ZipFile(source) as archive:
        for side in ["left", "right"]:
            raw = read_csv(archive, spec[side])
            for row in raw:
                record_id = "p_" + stable_id(SEED, dataset, side, row["id"])
                if (side, row["id"]) in lookup:
                    raise ValueError("Nonunique source identifier")
                lookup[(side, row["id"])] = record_id
                observed = {
                    "record_id": record_id,
                    "name": row.get("name", row.get("title", "")) or "",
                    "description": row.get("description", "") or "",
                    "manufacturer": row.get("manufacturer", "") or "",
                    "price": row.get("price", "") or "",
                }
                assert set(observed) == set(OBSERVED_FIELDS)
                records.append(observed)
                metadata[record_id] = {"side": side}
                source_map.append({"record_id": record_id, "source_file": spec[side], "source_id": row["id"]})
        mapping = read_csv(archive, spec["mapping"])
    edges = sorted({(lookup[("left", r[spec["left_key"]])], lookup[("right", r[spec["right_key"]])]) for r in mapping})
    if len(edges) != len(mapping):
        raise ValueError("Repeated source mapping edges")
    truth_graph = UnionFind(metadata)
    for a, b in edges:
        truth_graph.join(a, b)
    positive_ids = {i for pair in edges for i in pair}
    positive_components = 0
    for component in truth_graph.groups():
        cid = "c_" + stable_id(*component)
        positive = bool(set(component) & positive_ids)
        positive_components += int(positive)
        for rid in component:
            metadata[rid]["component_id"] = cid
            metadata[rid]["has_published_partner"] = rid in positive_ids
    families = UnionFind(metadata)
    for a, b in edges:
        families.join(a, b)
    by_exact, by_family = defaultdict(list), defaultdict(list)
    for row in records:
        exact = "|".join(normalize(row[k]) for k in ["name", "description", "manufacturer"])
        if exact.strip("|"):
            by_exact[exact].append(row["record_id"])
        for key in family_keys(row):
            by_family[key].append(row["record_id"])
    for group in [*by_exact.values(), *by_family.values()]:
        for rid in group[1:]:
            families.join(group[0], rid)
    family_groups = families.groups()
    group_components = {}
    for group in family_groups:
        fid = "f_" + stable_id(*group)
        positive = {metadata[rid]["component_id"] for rid in group if rid in positive_ids}
        group_components[fid] = len(positive)
        for rid in group:
            metadata[rid]["family_id"] = fid
    ordered_families = sorted(group_components, key=lambda f: stable_id(SEED, "development", f))
    dev_families, dev_components = set(), 0
    target = spec["development_components"]
    for fid in ordered_families:
        count = group_components[fid]
        if count and dev_components + count <= target:
            dev_families.add(fid)
            dev_components += count
        if dev_components >= target:
            break
    # Preserve natural unmapped groups using a seeded inclusion fraction, not invented NILs.
    fraction = dev_components / positive_components if positive_components else 0
    for fid in ordered_families:
        if group_components[fid] == 0 and int(stable_id(SEED, "nil-development", fid), 16) / 16 ** 24 < fraction:
            dev_families.add(fid)
    partitions = {split: {side: [] for side in ["left", "right"]} for split in ["dev", "test"]}
    for rid, meta in metadata.items():
        split = "dev" if meta["family_id"] in dev_families else "test"
        meta["split"] = split
        partitions[split][meta["side"]].append(rid)
    for split in partitions:
        for side in partitions[split]:
            partitions[split][side].sort()
    if any(metadata[a]["split"] != metadata[b]["split"] for a, b in edges):
        raise ValueError("A true edge crosses partitions")
    records.sort(key=lambda row: row["record_id"])
    location = HERE / "data" / dataset / CURATION_VERSION
    files = {
        "records.json": records, "partitions.json": partitions,
        "truth.json": {"target": "same product under publisher complete-mapping convention", "edges": [{"left_id": a, "right_id": b} for a, b in edges]},
        "metadata.json": metadata, "source-map.json": source_map,
    }
    for filename, value in files.items():
        write_once(location / filename, value)
    index = {r["record_id"]: r for r in records}
    # One explicit small source audit is the only permitted development inspection surface.
    audit_ids = partitions["dev"]["left"][:12]
    write_once(location / "development-source-audit.json", {
        "selection": "first 12 opaque-sorted development left records; no predictions or labels",
        "records": [index[i] for i in audit_ids],
    })
    counts = {}
    for split in partitions:
        ids = set(partitions[split]["left"]) | set(partitions[split]["right"])
        counts[split] = {
            "left": len(partitions[split]["left"]), "right": len(partitions[split]["right"]),
            "positive_components": len({metadata[i]["component_id"] for i in ids if i in positive_ids}),
            "family_groups": len({metadata[i]["family_id"] for i in ids}),
            "gold_links": sum(a in ids for a, _ in edges),
            "unmapped_left": sum(i not in positive_ids for i in partitions[split]["left"]),
            "unmapped_right": sum(i not in positive_ids for i in partitions[split]["right"]),
        }
    manifest = {
        "schema": 1, "dataset": dataset, "seed": SEED, "curation_version": CURATION_VERSION,
        "source": {"url": "https://dbs.uni-leipzig.de/files/datasets/" + source.name, "bytes": source.stat().st_size, "sha256": spec["sha256"]},
        "licence": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/", "publisher_statement": "https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution", "statement_date": "January 2019", "checked": "2026-10-07"},
        "split_policy": "Whole published-link components, exact duplicate normalized observed texts, same-brand title model prefixes of 2-8 letters followed by digits, and same-brand first two substantive title words after generic version/platform/licence removal are grouped. Seeded family ordering greedily fills the fixed positive-component development target without splitting a family. Unmapped families use the same component sampling fraction. All other records remain test.",
        "revision_note": "v1 is preserved in the parent directory. Inspection of its explicitly saved 12-record development source audit identified software edition/platform/licence families missed by model-prefix grouping. v2 adds a generic observed-title family key before any predictions or test outcome inspection.",
        "target_development_positive_components": target,
        "full_source": {"left": sum(m["side"] == "left" for m in metadata.values()), "right": sum(m["side"] == "right" for m in metadata.values()), "links": len(edges), "positive_components": positive_components, "family_groups": len(family_groups), "largest_family_records": max(map(len, family_groups)), "families_spanning_multiple_positive_components": sum(v > 1 for v in group_components.values())},
        "partitions": counts,
        "label_caveats": ["Published mappings are the scoring convention; unmapped does not independently prove real-world absence.", "Product matching is many-to-many; one-to-one assignment is prohibited.", "Public historical benchmarks may have appeared in model pretraining.", "Title-derived family detection is incomplete, particularly for software versions and undocumented product lineages.", "Greedy family-group sampling is a reproducible development screen, not a probability sample of the product market.", "Truth is used only for leakage-resistant curation and later independent evaluation, never by candidate generation or baseline fitting."],
        "file_sha256": {name: digest_file(location / name) for name in [*files, "development-source-audit.json"]},
        "curation_implementation_sha256": {name: digest_file(HERE / name) for name in ["prepare_products.py", "product_features.py"]},
    }
    write_once(location / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path, required=True)
    args = parser.parse_args()
    for dataset in SOURCES:
        result = curate(dataset, args.source_directory)
        print(json.dumps({"dataset": dataset, "full_source": result["full_source"], "partitions": result["partitions"]}))


if __name__ == "__main__":
    main()
