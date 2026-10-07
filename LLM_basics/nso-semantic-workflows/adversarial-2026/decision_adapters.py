"""Translate saved model/fit output to explicit complete-decision states."""
from __future__ import annotations

from model_jobs import extraction_valid, selection_decision


def product_decisions(rows, *, failed_extraction_ids=(), fit_valid=True):
    failed = set(failed_extraction_ids)
    result = {}
    for row in rows:
        rid = row["record_id"]
        targets = list(row["target_ids"])
        failure = (not fit_valid or rid in failed or bool(set(targets) & failed))
        result[rid] = {"status": "failed" if failure else "linked" if targets else "nil",
                       "target_ids": [] if failure else targets}
        if failure:
            result[rid]["raw_target_ids_before_failure_check"] = targets
    return result


def image_decisions(rows):
    result = {}
    for row in rows:
        status = row.get("decision_status", "failed")
        if row.get("fit_valid") is False:
            status = "failed"
        target = row.get("prediction")
        result[row["query_id"]] = {"status": status,
                                   "target_ids": [target] if status == "linked" and target else []}
    return result


def collect_results(manifest, ledger_events):
    """A missing or unfinished provider result remains a failed observation."""
    tags = [job["tag"] for job in manifest["jobs"]]
    rows = [r for r in ledger_events if r["event"] == "result" and r["tag"] in set(tags)]
    if len({r["tag"] for r in rows}) != len(rows):
        raise ValueError("Duplicate results violate the one-attempt contract.")
    by_tag = {r["tag"]: r for r in rows}
    return {tag: by_tag.get(tag, {"status": "failed", "failure": "missing_or_unresolved_result"}) for tag in tags}


def product_extractions(manifest, ledger_events):
    by_tag = collect_results(manifest, ledger_events)
    fields, audit = {}, []
    for job in manifest["jobs"]:
        if "/extraction/" not in job["tag"]:
            continue
        row = by_tag[job["tag"]]
        rid = job["tag"].split("/")[-1]
        valid = row["status"] == "valid"
        if valid:
            valid = extraction_valid(row["parsed"], job["user"])
        fields[rid] = row["parsed"] if valid else None
        audit.append({"record_id": rid, "status": "valid" if valid else "failed",
                      "provider_status": row["status"],
                      "evidence_quote_presence_valid": valid if row["status"] == "valid" else None})
    return fields, audit


def product_selections(manifest, ledger_events, candidates):
    by_tag = collect_results(manifest, ledger_events)
    pool = {r["record_id"]: r["candidate_ids"] for r in candidates}
    result = {}
    for job in manifest["jobs"]:
        if "/selection/" not in job["tag"]:
            continue
        rid = job["tag"].split("/")[-1]
        row = by_tag[job["tag"]]
        result[rid] = (selection_decision(row["parsed"], pool[rid]) if row["status"] == "valid"
                       else {"status": "failed", "target_ids": []})
    return result


def image_extractions(manifest, ledger_events):
    by_tag = collect_results(manifest, ledger_events)
    result = {"ocr-extraction": [], "pixel-extraction": []}
    for job in manifest["jobs"]:
        _, _, arm, rid = job["tag"].split("/")
        row = by_tag[job["tag"]]
        valid = row["status"] == "valid"
        result[arm].append({"record_id": rid, "status": "ok" if valid else "failed",
                            "fields": row["parsed"] if valid else {}})
    return result
