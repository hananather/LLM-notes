"""Verify saved scientific contracts and accounting without provider requests."""
from __future__ import annotations

import json
from pathlib import Path

import model_jobs
import runtime

ROOT = Path(__file__).resolve().parent


def verify():
    manifests = []
    for path in sorted((ROOT / "contracts").glob("*.json")):
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or "job_bounds" not in value:
            continue
        runtime.verify_manifest(value)
        model_jobs.verify_bindings(value)
        manifests.append({"path": str(path.relative_to(ROOT)),
                          "sha256": runtime.digest(value),
                          "jobs": len(value["jobs"]),
                          "stage": value["stage"],
                          "transport": value["transport"]})
    rows = runtime.events()
    admitted = [row for row in rows if row["event"] == "admission"]
    known = {item["sha256"] for item in manifests}
    if any(row["manifest_sha256"] not in known for row in admitted):
        raise ValueError("The ledger contains an admission without a verified manifest.")
    attempts = [row["tag"] for row in rows if row["event"] == "attempt"]
    results = [row for row in rows if row["event"] == "result"]
    if len(attempts) != len(set(attempts)):
        raise ValueError("Duplicate request attempts.")
    tags = [row["tag"] for row in results]
    if len(tags) != len(set(tags)) or not set(tags) <= set(attempts):
        raise ValueError("Results do not identify unique attempted requests.")
    if any(row.get("reservation_exceeded") for row in rows):
        raise ValueError("An unresolved financial accounting alert is present.")
    summary = runtime.summarize()
    return {"status": "verified", "provider_requests_made_by_verifier": 0,
            "contracts": manifests, "ledger": summary,
            "boundary": "Hash and accounting verification; not a new model run or an independent validation of reference labels."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
