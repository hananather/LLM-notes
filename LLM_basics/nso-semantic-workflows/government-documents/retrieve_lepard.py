"""Verify or re-fetch three pinned, public LePaRD rows. Standard library only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
MAX_RESPONSE_BYTES = 2_000_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    """Canonical encoding used for row hashes in provenance.json."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def download(url: str, expected_revision: str | None = None) -> bytes:
    request = Request(url, headers={"User-Agent": "LePaRD-small-source-showcase/1.0"})
    with urlopen(request, timeout=60) as response:
        if expected_revision and response.headers.get("x-revision") != expected_revision:
            raise ValueError("Viewer response revision differs from the pinned dataset")
        data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("Response exceeded the 2 MB per-request limit")
    return data


def check_row(raw: bytes, source: dict) -> dict:
    response = json.loads(raw)
    rows = response["rows"]
    if len(rows) != 1 or rows[0]["row_idx"] != source["row_idx"]:
        raise ValueError("The viewer returned a different row selection")
    if rows[0]["truncated_cells"]:
        raise ValueError("The viewer truncated a cell")
    row = rows[0]["row"]
    if sha256(canonical_bytes(row)) != source["row_sha256"]:
        raise ValueError(f"Source row {source['row_idx']} changed")
    return row


def verify(directory: Path) -> None:
    provenance = json.loads((directory / "provenance.json").read_text())
    samples = json.loads((directory / "samples.json").read_text())
    card = provenance["dataset"]["card"]
    if sha256((directory / card["file"]).read_bytes()) != card["sha256"]:
        raise ValueError("Saved source dataset card changed")
    examples = samples["examples"]
    sources = provenance["rows"]
    if len(examples) != 3 or len(sources) != 3:
        raise ValueError("Expected exactly three curated examples")
    for example, source in zip(examples, sources, strict=True):
        raw = (directory / source["response_file"]).read_bytes()
        if sha256(raw) != source["response_sha256"]:
            raise ValueError(f"Saved response changed: {source['response_file']}")
        row = check_row(raw, source)
        if example["row_idx"] != source["row_idx"] or example["raw_row"] != row:
            raise ValueError("Showcase text differs from the source response")
        for field in ("quote", "destination_context"):
            if sha256(row[field].encode("utf-8")) != source[f"{field}_sha256"]:
                raise ValueError(f"Source field {field} changed")
        if example["observed_relation"] != "dataset_extracted_quotation_link":
            raise ValueError("Unexpected relation label")
    print("Verified 3 source rows and all 6 complete text fields; no model calls.")


def fetch(output: Path) -> None:
    provenance = json.loads((HERE / "provenance.json").read_text())
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing files are preserved")
    revision_url = provenance["dataset"]["metadata_url"]
    expected_revision = provenance["dataset"]["revision"]
    if json.loads(download(revision_url))["sha"] != expected_revision:
        raise ValueError("Dataset revision changed; inspect it before selecting new rows")
    downloaded = []
    for source in provenance["rows"]:
        raw = download(source["request_url"], expected_revision)
        check_row(raw, source)
        downloaded.append((source, raw))
    card = provenance["dataset"]["card"]
    card_raw = download(card["url"])
    if sha256(card_raw) != card["sha256"]:
        raise ValueError("Pinned source dataset card changed")
    if json.loads(download(revision_url))["sha"] != expected_revision:
        raise ValueError("Dataset revision changed during retrieval")
    output.mkdir(parents=True)
    for source, raw in downloaded:
        path = output / source["response_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        # API envelope metadata or serialization may change without a row change.
        source["response_sha256"] = sha256(raw)
        source["response_bytes"] = len(raw)
    provenance["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / card["file"]).write_bytes(card_raw)
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "samples.json").write_bytes((HERE / "samples.json").read_bytes())
    (output / "SOURCE-NOTICE.md").write_bytes((HERE / "SOURCE-NOTICE.md").read_bytes())
    verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify_parser = sub.add_parser("verify", help="Check saved data without network access")
    verify_parser.add_argument("--directory", type=Path, default=HERE)
    fetch_parser = sub.add_parser("fetch", help="Re-fetch only the three selected rows")
    fetch_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.directory)
    else:
        fetch(args.output)


if __name__ == "__main__":
    main()
