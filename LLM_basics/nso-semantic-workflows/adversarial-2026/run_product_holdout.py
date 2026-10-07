"""Explicit preparation, submission and collection of one amended product run.

Only ``submit`` creates provider work. ``status`` reads its state; ``collect``
downloads terminal evidence and seals complete predictions without labels.
Notebook replay does not invoke this program.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path

import batch_runtime as batch
import decision_adapters as adapters
import model_jobs
import runtime as rt

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "contracts/products-tradeoff-test-v1.json"
OUTPUT = ROOT / "results/product-holdout-v1"
PLAN = OUTPUT / "batch-plan.json"
BASELINE = ROOT / "products/results/baseline-v6/amazon-google/test"
AMENDMENT = ROOT / "contracts/product-tradeoff-amendment-v1.json"
EVALUATOR = ROOT / "products/score_evaluation.py"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    batch._save(path, (rt.canonical(value) + "\n").encode())


def prepare():
    jobs = model_jobs.product_jobs("test", BASELINE / "candidates.json", arms=("selection",))
    assert len(jobs) == 1219
    paths = [ROOT / name for name in [
        "runtime.py", "evaluation.py", "model_jobs.py", "decision_adapters.py",
        "batch_runtime.py", "protocol.json", "run_product_holdout.py",
        "products/baseline.py", "products/product_features.py",
        "products/prepare_products.py",
        "products/results/model-development-v1/development-gates.json",
        "products/results/model-development-v1/development-reference.json"]]
    paths += [AMENDMENT, EVALUATOR]
    paths += [p for p in (ROOT / "products/data/amazon-google/v2").glob("*.json")]
    paths += [p for p in BASELINE.iterdir() if p.is_file()]
    audit = read(BASELINE / "splink-fit-audit.json")
    contract = {
        "status": "frozen_for_execution", "stage": "evaluation",
        "study": "Post-development amended product tradeoff characterization",
        "query_count": 1219, "catalog_count": 2874,
        "curation_version": "v2", "baseline_version": "v6",
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "original_gate": "failed; this run explicitly overrides the spending stop once",
        "arms": ["same-candidate-semantic-selection"],
        "primary_reference": "lexical_all_pairs",
        "prior_bundle": {"priors": audit["priors"], "main_prior": audit["main_prior"]},
        "files": model_jobs.file_bindings(paths)}
    manifest = rt.make_manifest("products-tradeoff-test-v1", "evaluation", jobs,
                                contract, transport="batch")
    if Decimal(manifest["reserved_usd"]) > Decimal(read(AMENDMENT)["allowed_run"]["maximum_reserved_usd"]):
        raise ValueError("Complete run exceeds its amendment reservation.")
    model_jobs.save_manifest(manifest, MANIFEST)
    plan = batch.prepare(manifest, plan_path=PLAN,
                         cache_dir=Path.home() / ".cache/nso-adversarial-2026/batch")
    if len(plan["chunks"]) != 1:
        raise ValueError("This sealed study expects one complete provider batch.")
    return {"manifest_sha256": rt.digest(manifest), "jobs": len(jobs),
            "reserved_usd": manifest["reserved_usd"], "chunks": len(plan["chunks"]),
            "request_bytes": sum(c["bytes"] for c in plan["chunks"]),
            "provider_requests": 0}


def submit():
    manifest, plan = read(MANIFEST), read(PLAN)
    batch.admit(manifest, plan)
    return [compact(batch.submit(manifest, plan, i)) for i in range(len(plan["chunks"]))]


def compact(snapshot):
    return {k: snapshot.get(k) for k in ["id", "status", "request_counts", "created_at", "completed_at", "expires_at"]}


def status():
    manifest, plan = read(MANIFEST), read(PLAN)
    return [compact(batch.poll(manifest, plan, i)) for i in range(len(plan["chunks"]))]


def collect():
    manifest, plan = read(MANIFEST), read(PLAN)
    for i in range(len(plan["chunks"])):
        batch.retrieve(manifest, plan, i)
    rows = rt.events()
    tags = {j["tag"] for j in manifest["jobs"]}
    results = [row for row in rows if row["event"] == "result" and row["tag"] in tags]
    if len(results) != len(tags) or len({row["tag"] for row in results}) != len(tags):
        raise ValueError("Do not seal an incomplete or duplicate denominator.")
    decisions = adapters.product_selections(manifest, rows, read(BASELINE / "candidates.json"))
    path = OUTPUT / "selection-decisions.json"
    save(path, decisions)
    execution_audit = {
        "attempted_requests": len(tags), "terminal_result_count": len(results),
        "parsed_result_status_counts": dict(Counter(row["status"] for row in results)),
        "decision_status_counts": dict(Counter(row["status"] for row in decisions.values())),
        "provider_model_id_counts": dict(Counter(row["response"]["model"] for row in results
                                              if (row.get("response") or {}).get("model"))),
        "terminal_status": next(row["terminal_status"] for row in rows
                                if row["event"] == "batch_retrieved" and row.get("plan_sha256") == plan["plan_sha256"]),
        "reserved_usd": manifest["reserved_usd"],
        "nominal_token_price_usd": str(sum((Decimal(row.get("estimated_token_price_usd", "0")) for row in results), Decimal(0))),
        "unpriced_results": sum("estimated_token_price_usd" not in row for row in results),
        "boundary": "Usage-price estimates, not invoices. All failures retained. No identity labels loaded."}
    seal_path = OUTPUT / "prediction-seal.json"
    seal = {"status": "predictions_sealed", "sealed_at_utc": rt.utc_now(),
            "manifest_sha256": rt.digest(manifest), "evaluator_sha256": rt.file_hash(EVALUATOR),
            "predictions_path": str(path.relative_to(ROOT)), "predictions_sha256": rt.file_hash(path),
            "query_count": len(decisions), "terminal_result_count": len(results),
            "execution_audit": execution_audit}
    if seal_path.exists():
        old = read(seal_path)
        seal["sealed_at_utc"] = old["sealed_at_utc"]
    save(seal_path, seal)
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "status", "collect"])
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
