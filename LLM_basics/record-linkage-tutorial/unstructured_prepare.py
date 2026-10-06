"""Freeze small, openly licensed slices of the paper's BioDEX and FEVER tasks.

Network access is needed only for this preparation command. The notebook reads
the frozen output. Selection never uses a model's predictions.
"""

from collections import Counter
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import urllib.request
import urllib.parse
import zipfile

import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "unstructured"
CACHE = HERE / ".cache" / "unstructured-sources"
HF_REVISION = "01a5dacdabd144a120af04931a11a99febd48432"
HF_BASE = f"https://huggingface.co/datasets/BioDEX/BioDEX-Reactions/resolve/{HF_REVISION}/data/"
PARQUETS = {
    "test": ["test-00000-of-00001-2449439e9130546c.parquet"],
    "train": ["train-00000-of-00002-3a4af29b480220dc.parquet",
              "train-00001-of-00002-91a4d2de97566883.parquet"],
}
FEVER_BASE = "https://fever.ai/download/fever/"
SEED = "lotus-small-v1:"


def hash_bytes(value):
    return sha256(value).hexdigest()


def rank_id(value):
    return hash_bytes((SEED + str(value)).encode())


def download(url, filename):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / filename
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    verify_source(path)
    return path


def verify_source(path):
    """A committed manifest is the source pin on every later preparation run."""
    manifest_path = DATA / "manifest.json"
    if not manifest_path.exists():
        return
    pinned = json.loads(manifest_path.read_text())
    expected = pinned["biodex"]["source_files_sha256"].get(path.name)
    if path.name == "shared_task_dev.jsonl":
        expected = pinned["fever"]["claims_sha256"]
    if path.name == Path(pinned["fever"]["wiki_member"]).name:
        expected = pinned["fever"]["wiki_member_sha256"]
    if expected and hash_bytes(path.read_bytes()) != expected:
        raise ValueError(f"Source does not match the committed hash: {path.name}")


class RemoteZip(io.RawIOBase):
    """Read one ZIP member using checked HTTP byte ranges, not a full download."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        response = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(response.headers["Content-Length"])
        self.etag = response.headers["ETag"]

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset if whence == 0 else self.pos + offset
                    if whence == 1 else self.size + offset)
        return self.pos

    def read(self, size=-1):
        end = self.size - 1 if size < 0 else min(self.pos + size - 1, self.size - 1)
        if end < self.pos:
            return b""
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={self.pos}-{end}", "If-Match": self.etag,
        })
        with urllib.request.urlopen(request) as response:
            if response.status != 206:
                raise ValueError("The source server did not honor the byte range.")
            content = response.read()
        if len(content) != end - self.pos + 1:
            raise ValueError("Incomplete ZIP byte range.")
        self.pos += len(content)
        return content


def write_json(name, value):
    path = DATA / name
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return hash_bytes(path.read_bytes())


def prepare_biodex():
    source_hashes = {}
    train_counts = Counter()
    for filename in PARQUETS["train"]:
        path = download(HF_BASE + filename, filename)
        source_hashes[filename] = hash_bytes(path.read_bytes())
        for labels in pd.read_parquet(path, columns=["reactions"]).reactions:
            train_counts.update(set(x.strip() for x in labels.split(",")))
    targets = sorted(train_counts, key=lambda x: (-train_counts[x], x))[:8]
    filename = PARQUETS["test"][0]
    path = download(HF_BASE + filename, filename)
    source_hashes[filename] = hash_bytes(path.read_bytes())
    frame = pd.read_parquet(path)
    eligible = []
    for row in frame.to_dict("records"):
        labels = sorted(set(x.strip() for x in row["reactions"].split(",")))
        text = row["fulltext_processed"]
        if (row["fulltext_license"] == "CC BY" and 1000 <= len(text) <= 8000
                and set(targets).intersection(labels)):
            row["labels"] = labels
            eligible.append(row)
    chosen = sorted(eligible, key=lambda x: (rank_id(x["pmid"]), x["pmid"]))[:8]
    if len(chosen) != 8 or len({x["pmid"] for x in chosen}) != 8:
        raise ValueError("BioDEX requires eight distinct eligible reports.")
    observations, truth, attribution = [], [], []
    for row in chosen:
        uid = "bio-" + rank_id(row["pmid"])[:12]
        observations.append({"unique_id": uid, "report": row["fulltext_processed"]})
        truth.append({"unique_id": uid, "reactions": row["labels"]})
        attribution.append({k: row[k] for k in [
            "pmid", "pmc", "doi", "title", "authors", "fulltext_license"]} | {
            "unique_id": uid, "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/",
            "fulltext_characters": len(row["fulltext_processed"]), "local_truncation": False,
            "license_url": sorted(set(re.findall(r"https?://creativecommons.org/licenses/by/[\d.]+/", row["fulltext"])))[0],
        })
    files = {
        "biodex-reports.json": observations,
        "biodex-truth.json": truth,
        "biodex-categories.json": [{"unique_id": f"category-{i:02d}", "reaction": label}
                                   for i, label in enumerate(targets)],
        "biodex-attribution.json": attribution,
    }
    return files, {
        "source": "BioDEX/BioDEX-Reactions", "revision": HF_REVISION,
        "source_files_sha256": source_hashes,
        "original_test_reports": len(frame), "eligible_test_reports": len(eligible),
        "category_rule": "Eight most frequent training reaction labels; ties alphabetic.",
        "train_category_frequencies": {label: train_counts[label] for label in targets},
        "report_rule": "CC BY; 1000–8000 processed-text characters; at least one target label; first eight by SHA256(seed + PMID).",
        "seed": SEED, "selected_reports": 8, "selected_categories": 8,
        "pairs": 64, "positive_pairs": sum(len(set(x["reactions"]) & set(targets)) for x in truth),
        "interpretation": "Agreement with published report-level reaction annotations in an enriched slice; labels may be incomplete. No clinical adjudication.",
    }


def prepare_fever():
    dev_path = download(FEVER_BASE + "shared_task_dev.jsonl", "shared_task_dev.jsonl")
    dev = [json.loads(line) for line in dev_path.read_text().splitlines()]
    remote = RemoteZip(FEVER_BASE + "wiki-pages.zip")
    if (DATA / "manifest.json").exists():
        expected = json.loads((DATA / "manifest.json").read_text())["fever"]
        if remote.etag != expected["wiki_zip_etag"] or remote.size != expected["wiki_zip_bytes"]:
            raise ValueError("The original Wikipedia ZIP changed.")
    examined_shards = []
    with zipfile.ZipFile(remote) as archive:
        members = sorted(x for x in archive.namelist()
                         if x.startswith("wiki-pages/") and x.endswith(".jsonl"))
        for member in members:
            wiki_path = CACHE / Path(member).name
            if not wiki_path.exists():
                wiki_path.write_bytes(archive.read(member))
            verify_source(wiki_path)
            wiki = {x["id"]: x for x in map(json.loads, wiki_path.read_text().splitlines())}
            candidates = {"SUPPORTS": [], "REFUTES": []}
            for claim in dev:
                if claim["label"] not in candidates:
                    continue
                singleton = sorted({(group[0][2], group[0][3]) for group in claim["evidence"]
                                    if len(group) == 1 and group[0][2] in wiki})
                if singleton:
                    claim = dict(claim, selected_evidence=singleton[0])
                    candidates[claim["label"]].append(claim)
            examined_shards.append({"member": member,
                                    "counts": {k: len(v) for k, v in candidates.items()}})
            diverse = {}
            for label, rows in candidates.items():
                seen, selected = set(), []
                for row in sorted(rows, key=lambda x: rank_id(x["id"])):
                    page = row["selected_evidence"][0]
                    if page not in seen:
                        selected.append(row)
                        seen.add(page)
                diverse[label] = selected
            if min(map(len, diverse.values())) >= 6:
                break
        license_text = archive.read("license.html").decode()
    chosen = [row for label in ("SUPPORTS", "REFUTES") for row in diverse[label][:6]]
    if len(chosen) != 12:
        raise ValueError("The fixed FEVER shard lacks six eligible claims per class.")
    chosen.sort(key=lambda x: rank_id(x["id"]))
    claims, evidence, truth = [], {}, []
    for row in chosen:
        page, line = row["selected_evidence"]
        sentences = {int(parts[0]): parts[1] for parts in
                     (x.split("\t") for x in wiki[page]["lines"].splitlines())
                     if len(parts) >= 2 and parts[0].isdigit()}
        sentence = sentences[line]
        eid = "wiki-" + rank_id(f"{page}:{line}")[:12]
        uid = "claim-" + rank_id(row["id"])[:12]
        claims.append({"unique_id": uid, "claim": row["claim"]})
        evidence[eid] = {"unique_id": eid, "title": page, "sentence": sentence,
                         "sentence_id": line,
                         "source_url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(page.replace("-LRB-", "(").replace("-RRB-", ")"))}
        truth.append({"unique_id": uid, "fever_id": row["id"], "label": row["label"],
                      "evidence_id": eid, "evidence_sets": row["evidence"]})
    files = {"fever-claims.json": claims, "fever-evidence.json": list(evidence.values()),
             "fever-truth.json": truth}
    (DATA / "FEVER-LICENSE.html").write_text(license_text)
    return files, {
        "claims_source": FEVER_BASE + "shared_task_dev.jsonl",
        "claims_sha256": hash_bytes(dev_path.read_bytes()), "original_dev_claims": len(dev),
        "wiki_source": FEVER_BASE + "wiki-pages.zip", "wiki_zip_etag": remote.etag,
        "wiki_zip_bytes": remote.size, "wiki_member": member,
        "wiki_member_sha256": hash_bytes(wiki_path.read_bytes()), "shard_pages": len(wiki),
        "eligible_claims_by_label": {k: len(v) for k, v in candidates.items()},
        "examined_shards": examined_shards,
        "selection_rule": "First lexicographic original Wikipedia shard with at least six distinct eligible evidence pages per class; singleton annotated evidence in that shard; six claims per supports/refutes class by SHA256(seed + claim ID), retaining at most one claim per evidence page per class. First evidence by (page, sentence).",
        "seed": SEED, "claims": len(claims), "evidence_sentences": len(evidence),
        "positive_claims": 6, "negative_claims": 6,
        "interpretation": "Support filtering conditional on supplied gold evidence. Retrieval checks only this gold-derived miniature sentence corpus. Excludes NotEnoughInfo and multi-sentence-only evidence.",
    }


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    previous = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else None
    manifest = {"schema": 1, "paper": "https://www.vldb.org/pvldb/vol18/p4171-patel.pdf",
                "prepared_date": "2026-10-06", "files_sha256": {}}
    prepared = {}
    for name, prepare in (("biodex", prepare_biodex), ("fever", prepare_fever)):
        files, metadata = prepare()
        for filename, contents in files.items():
            prepared[filename] = contents
            serialized = json.dumps(contents, indent=2, ensure_ascii=False) + "\n"
            manifest["files_sha256"][filename] = hash_bytes(serialized.encode())
        manifest[name] = metadata
    manifest["files_sha256"]["FEVER-LICENSE.html"] = hash_bytes((DATA / "FEVER-LICENSE.html").read_bytes())
    if previous is not None and manifest != previous:
        raise ValueError("Sources, selection, or output hashes differ from the frozen manifest; existing data were not replaced.")
    for filename, contents in prepared.items():
        write_json(filename, contents)
    write_json("manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
