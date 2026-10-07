"""Offline integrity checks only; never joins prediction scores to reference truth."""
from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []
    failed_fits = []
    for dataset in ["abt-buy", "amazon-google"]:
        location = HERE / "data" / dataset / "v2"
        manifest = json.loads((location / "manifest.json").read_text())
        for name, expected in manifest["file_sha256"].items():
            assert digest(location / name) == expected, name
        archive = HERE / "sources" / manifest["source"]["url"].rsplit("/", 1)[-1]
        assert digest(archive) == manifest["source"]["sha256"]
        for name, expected in manifest["curation_implementation_sha256"].items():
            assert digest(HERE / "data" / "curation-v2-source" / name) == expected, name
        records = json.loads((location / "records.json").read_text())
        assert all(set(r) == {"record_id","name","description","manufacturer","price"} for r in records)
        ids = {r["record_id"] for r in records}
        assert len(ids) == len(records)
        partitions = json.loads((location / "partitions.json").read_text())
        joined = [rid for split in partitions.values() for side in split.values() for rid in side]
        assert len(joined) == len(set(joined)) and set(joined) == ids
        checks.append({"dataset": dataset,"source_and_curation_hashes": "verified","records":len(ids)})
    # Replay curation with its exact preserved source bytes, including family isolation.
    with tempfile.TemporaryDirectory(prefix="curation-check-", dir=HERE) as folder:
        work = Path(folder)
        for file in (HERE / "data" / "curation-v2-source").glob("*.py"):
            shutil.copyfile(file, work / file.name)
        import subprocess,sys
        subprocess.run([sys.executable,str(work / "prepare_products.py"),"--source-directory",str(HERE / "sources")],check=True,capture_output=True,text=True)
        for dataset in ["abt-buy","amazon-google"]:
            for expected in (HERE / "data" / dataset / "v2").glob("*.json"):
                assert digest(work / "data" / dataset / "v2" / expected.name) == digest(expected), expected.name
        checks.append({"curation_replay": "byte-identical with frozen implementation"})
    for split in ["dev","test"]:
        location = HERE / "results" / "candidates-v1" / "amazon-google" / split
        protocol = json.loads((location / "protocol.json").read_text())
        for name, expected in protocol["implementation_sha256"].items():
            assert digest(HERE / "results" / "candidates-v1" / "implementation" / name) == expected
        partitions = json.loads((HERE / "data/amazon-google/v2/partitions.json").read_text())[split]
        candidates = json.loads((location / "candidates.json").read_text())
        assert [c["record_id"] for c in candidates] == partitions["left"]
        universe = set(partitions["right"])
        for row in candidates:
            assert len(row["candidate_ids"]) <= 20
            assert len(row["candidate_ids"]) == len(set(row["candidate_ids"]))
            assert set(row["candidate_ids"]) <= universe
        checks.append({"candidate_split":split,"rows":len(candidates),"candidate_references":"valid"})
    for completion in sorted((HERE / "results").glob("baseline-v*/*/*/completion.json")):
        manifest = json.loads(completion.read_text())
        for name, expected in manifest["files"].items():
            assert digest(completion.parent / name) == expected, name
        protocol = json.loads((completion.parent / "protocol.json").read_text())
        for name, expected in protocol["implementation_sha256"].items():
            snapshot = completion.parents[2] / "implementation" / name
            assert digest(snapshot if snapshot.exists() else HERE / name) == expected, name
        if manifest["dataset"] == "amazon-google":
            candidate_reference = HERE / "results/candidates-v1/amazon-google" / manifest["split"] / "candidates.json"
            assert digest(candidate_reference) == digest(completion.parent / "candidates.json")
        checks.append({"predictions":str(completion.parent.relative_to(HERE)),"complete_hash_manifest":"verified", "fit_status":manifest["status"]})
        if completion.parents[2].name == "baseline-v6":
            audit = json.loads((completion.parent / "splink-fit-audit.json").read_text())
            if audit["status"] == "failed_fit":
                failed_fits.append(str(completion.parent.relative_to(HERE)))
                assert (completion.parent / "failed-fit-decisions.json").exists()
                assert not (completion.parent / "splink-decisions.json").exists()
            else:
                assert audit["prediction_consistency"]["status"] == "passed"
                assert all(row["minimum"] > 0 and abs(row["sum"]-1)<1e-12 for row in audit["categorical_vector_checks"])
                assert all(row["status"] == "empty_block" or row["parameter_tolerance_met"] for row in audit["em_sessions"])
    print(json.dumps({"status":"integrity_checks_passed","checks":checks,"current_failed_fits":failed_fits,"truth_scoring":"not performed"},indent=2))


if __name__ == "__main__":
    main()
