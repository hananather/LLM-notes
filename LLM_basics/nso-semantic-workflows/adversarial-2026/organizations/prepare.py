"""Prepare pinned public organization sources; labels never enter inference files."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import re
import unicodedata
import zipfile

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 2026100721
COMMIT = "9be394d5a8b11360b877ce59fb2355b0a7fa867e"
ARCHIVE_SHA = "0c532997867d205097f0be3ba13250ec3f3b322d580b1e88c23d0e9f4a25eafe"
SOURCES = {
    "cordis": ("CORDIS.csv", "a402aa3e2f798d7cc0acf29e1502c504fad2178bfc68ca1b812f4742ae03c7e3"),
    "french": ("French affiliations.csv", "a83a7ded6fccc324a018c5c5534bb08417549212bbefa84f80c777a396cc61fe"),
    "multilingual": ("Multilingual affiliations.csv", "4953ba2b4400abd389fa7ba31fffe91f60eff8e64f3da97e9ff93e6d0a6a4f92"),
    "multi-org": ("Non-related multi-organizations.csv", "2394bc7efcc1cce16f2e88c4bb8af74a06af93fef1d6deea6c38e56a67a9b9d9"),
}


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def norm(value):
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def opaque(namespace, value):
    return hashlib.sha256((namespace + ":" + value).encode()).hexdigest()[:24]


def freeze_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Refusing to replace frozen artifact: {path}")
    if not path.exists():
        path.write_bytes(content)
    return {"path": str(path.relative_to(HERE)), "bytes": len(content), "sha256": digest(content)}


def freeze_json(path, value):
    return freeze_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def freeze_lines(path, rows):
    content = b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode() for r in rows)
    if path.suffix == ".gz":
        content = gzip.compress(content, mtime=0)
    return freeze_bytes(path, content)


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def prepare(archive, old_s2aff=None):
    from urllib.parse import quote
    raw_archive = archive.read_bytes()
    if digest(raw_archive) != ARCHIVE_SHA:
        raise ValueError("Unexpected ROR archive hash")
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as z:
        registry = json.loads(z.read("v1.41-2024-02-13-ror-data.json"))
    assert len(registry) == len({r["id"] for r in registry}) == 108476
    rid = {r["id"]: opaque("ror-1.41", r["id"]) for r in registry}
    status = {r["id"]: r["status"] for r in registry}
    observed_roster = []
    for r in sorted(registry, key=lambda r: rid[r["id"]]):
        observed_roster.append({"record_id": rid[r["id"]], "name": r["name"],
            "aliases": r.get("aliases", []), "acronyms": r.get("acronyms", []),
            "labels": r.get("labels", []), "country_code": r["country"]["country_code"],
            "country": r["country"]["country_name"],
            "cities": sorted({a["city"] for a in r.get("addresses", []) if a.get("city")}),
            "status": r["status"]})
    outputs = [freeze_lines(DATA / "ror-v1.41-observed.jsonl.gz", observed_roster)]
    outputs.append(freeze_json(DATA / "gold" / "registry-identifiers.json", rid))
    sources, observed, truth, profiles, conflicts = [], {}, [], {}, []
    duplicates = defaultdict(list)
    for key, (name, expected) in SOURCES.items():
        path = HERE / "sources" / f"{key}.csv"
        raw = path.read_bytes()
        if digest(raw) != expected:
            raise ValueError(f"Unexpected source hash: {key}")
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        expected_fields = {"ORG", "CITY", "COUNTRY", "label"} if key == "cordis" else {"raw_affiliation_string", "label"}
        if any(set(r) != expected_fields or any(v is None for v in r.values()) for r in rows):
            raise ValueError(f"Invalid source CSV schema: {key}")
        current = []
        current_gold = []
        for index, r in enumerate(rows):
            qid = opaque(key, str(index))
            rec = {"query_id": qid, **{k: v for k, v in r.items() if k != "label"}}
            if key != "cordis":
                rec["source_population"] = key
            ids = sorted(set(re.findall(r"https://ror.org/[a-zA-Z0-9]+", r["label"])))
            absent = [x for x in ids if x not in rid]
            if absent:
                raise ValueError(f"Positive annotation absent from full registry: {key}:{index}: {absent}")
            gold = {"query_id": qid, "source_population": key, "source_row_zero_based": index,
                "source_label": r["label"], "ror_ids": ids, "target_ids": [rid[x] for x in ids],
                "target_statuses": [status[x] for x in ids], "historical_nil": not ids}
            current.append(rec)
            current_gold.append(gold)
            if key == "cordis":
                sig = ("cordis", *(norm(r[k]) for k in ["ORG", "CITY", "COUNTRY"]))
            else:
                sig = ("native", r["raw_affiliation_string"])
            duplicates[sig].append(gold)
        observed[key] = current
        truth.extend(current_gold)
        profiles[key] = {"source_rows": len(rows), "distinct_positive_ror_ids": len({i for g in current_gold for i in g["ror_ids"]}),
            "nil_rows": sum(g["historical_nil"] for g in current_gold), "multi_target_rows": sum(len(g["target_ids"]) > 1 for g in current_gold),
            "positive_target_status_counts": dict(Counter(s for g in current_gold for s in g["target_statuses"]))}
        sources.append({"file": str(path.relative_to(HERE)), "sha256": expected, "bytes": len(raw),
            "url": f"https://raw.githubusercontent.com/sirisacademic/affilgood/{COMMIT}/data/entity%20linking/{quote(name)}",
            "licence": "Apache-2.0; owning repository notice retained"})
    for sig, group in duplicates.items():
        if len({tuple(g["ror_ids"]) for g in group}) > 1:
            conflicts.append({"type": "same-input-different-source-targets", "query_ids": [g["query_id"] for g in group],
                "source_populations": [g["source_population"] for g in group], "target_sets": [g["ror_ids"] for g in group]})
    native = sum([observed[k] for k in ["french", "multilingual", "multi-org"]], [])
    outputs.extend([freeze_lines(DATA / "cordis-observed.jsonl", observed["cordis"]),
        freeze_lines(DATA / "native-observed.jsonl", native), freeze_lines(DATA / "gold" / "reference.jsonl", truth)])
    gold_by_id = {g["query_id"]: g for g in truth}
    uf = UnionFind(r["query_id"] for r in observed["cordis"])
    first = {}
    for r in observed["cordis"]:
        keys = [("input", *(norm(r[k]) for k in ["ORG", "CITY", "COUNTRY"]))]
        keys += [("target", x) for x in gold_by_id[r["query_id"]]["ror_ids"]]
        for key in keys:
            if key in first:
                uf.union(r["query_id"], first[key])
            else:
                first[key] = r["query_id"]
    groups = defaultdict(list)
    for r in observed["cordis"]:
        groups[uf.find(r["query_id"])].append(r["query_id"])
    ordered = sorted(groups)
    random.Random(SEED).shuffle(ordered)
    memberships = {"development": ordered[:100], "evaluation": ordered[100:1100], "remainder": ordered[1100:]}
    partitions = {k: sorted(q for g in gs for q in groups[g]) for k, gs in memberships.items()}
    assert len(partitions["evaluation"]) >= 1000
    assert sum(map(len, partitions.values())) == 3329
    outputs.append(freeze_json(DATA / "partitions.json", partitions))
    sampling = {"seed": SEED, "unit": "connected group of shared positive ROR targets or normalized duplicate input triples",
        "source_rows": 3329, "group_count": len(groups), "group_size_counts": dict(Counter(map(len, groups.values()))),
        "construction_uses_labels": "Only the data preparation custodian uses published ROR targets for group split isolation; inference and fitting do not.",
        "design": "Seeded random permutation of all groups; first 100 groups development, next 1000 evaluation, remainder unscored background.",
        "inclusion_probability_unconditional": {k: len(v) / len(groups) for k, v in memberships.items()},
        "evaluation_inclusion_probability_conditional_on_dev": 1000 / (len(groups) - 100),
        "row_counts": {k: len(v) for k, v in partitions.items()}, "group_members": groups, "group_partitions": memberships,
        "fitting": "Transductive unlabeled fitting may read all 3329 observed input triples; labels, group links and metrics remain unavailable to the fitter."}
    outputs.append(freeze_json(DATA / "gold" / "sampling-design.json", sampling))
    overlap = {"status": "not provided"}
    if old_s2aff is not None:
        raw = old_s2aff.read_bytes()
        if digest(raw) != "2385f28c6c1275f0114cf15c33d164672af7a2a6a88d0c03178e04236c3de8c3":
            raise ValueError("Unexpected historical S2AFF source")
        s2 = list(csv.DictReader(io.StringIO(raw.decode())))
        by_text = defaultdict(list)
        by_norm = defaultdict(list)
        s2_targets = set()
        for i, r in enumerate(s2):
            by_text[r["original_affiliation"]].append({"source_row": i, "split": r["split"]})
            by_norm[norm(r["original_affiliation"])].append({"source_row": i, "split": r["split"]})
            s2_targets.update(x for x in ast.literal_eval(r["labels"]) if x.startswith("https://ror.org/"))
        overlap = {"source_sha256": digest(raw), "source_rows": len(s2),
            "exact_text": [{"query_id": r["query_id"], "s2aff": by_text[r["raw_affiliation_string"]]} for r in native if r["raw_affiliation_string"] in by_text],
            "normalized_text": [{"query_id": r["query_id"], "s2aff": by_norm[norm(r["raw_affiliation_string"])]} for r in native if norm(r["raw_affiliation_string"]) in by_norm],
            "cordis_queries_sharing_any_s2aff_target": [r["query_id"] for r in observed["cordis"] if set(gold_by_id[r["query_id"]]["ror_ids"]) & s2_targets],
            "native_queries_sharing_any_s2aff_target": [r["query_id"] for r in native if set(gold_by_id[r["query_id"]]["ror_ids"]) & s2_targets]}
    duplicate_groups = [{"query_ids": [g["query_id"] for g in gs], "source_populations": [g["source_population"] for g in gs]} for sig, gs in duplicates.items() if len(gs) > 1]
    audit = {"profiles": profiles, "registry_rows": len(registry), "registry_status_counts": dict(Counter(status.values())),
        "native_source_rows": len(native), "native_distinct_raw_strings": len({r["raw_affiliation_string"] for r in native}),
        "duplicate_groups": duplicate_groups, "source_label_policy_flags": conflicts, "old_s2aff_overlap": overlap,
        "missing_positive_registry_ids": [], "exclusions": [], "policy": "Preserve all historical labels, duplicates, inactive/withdrawn records and flagged inconsistencies. No silent adjudication."}
    outputs.append(freeze_json(DATA / "gold" / "curation-audit.json", audit))
    manifest = {"schema": 1, "sources": sources, "registry": {"sha256": ARCHIVE_SHA, "bytes": len(raw_archive),
        "url": "https://github.com/ror-community/ror-data/raw/main/v1.41-2024-02-13-ror-data.zip", "rows": len(registry), "licence": "ROR CC0; GeoNames attribution retained"},
        "inference_fields": {"cordis": ["query_id", "ORG", "CITY", "COUNTRY"], "native": ["query_id", "source_population", "raw_affiliation_string"]},
        "label_boundary": "Entire source label column, ROR IDs, gold target links and grouping links are confined to sources/ and data/gold/. Baseline reads only inference files and query-ID partitions.",
        "prepared_files": outputs, "code_sha256": digest(Path(__file__).read_bytes())}
    freeze_json(DATA / "source-manifest.json", manifest)
    return {"profiles": profiles, "registry_rows": len(registry), "partition_rows": sampling["row_counts"],
        "native_distinct_raw_strings": audit["native_distinct_raw_strings"], "policy_flags": len(conflicts),
        "old_s2aff_exact_text_overlap_rows": len(overlap.get("exact_text", []))}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ror-archive", type=Path, required=True)
    p.add_argument("--old-s2aff", type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.ror_archive, a.old_s2aff), indent=2))
