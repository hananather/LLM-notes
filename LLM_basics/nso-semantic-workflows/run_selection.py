"""Replay or complete the declared affiliation and coding selection batches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from batch import run_jobs
import coding
import linkage
from verify import verify_manifest

ROOT = Path(__file__).resolve().parent


def save_or_check(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Reconstructed score differs from saved evidence: {path.name}")
    else:
        path.write_text(json.dumps(value, indent=2) + "\n")


def completed_or_run(jobs, path, run_id, live):
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["status"] == "complete":
            from runtime import digest
            if saved["spec"]["jobs_sha256"] != digest(jobs):
                raise ValueError("The completed batch has a different frozen request specification.")
            actual = [r["case_id"] for r in saved["rows"]]
            if len(actual) != len(set(actual)) or set(actual) != {j["case_id"] for j in jobs}:
                raise ValueError("The completed batch is missing or duplicates expected cases.")
            return saved
    if not live:
        raise RuntimeError("This batch is incomplete. --live explicitly permits its remaining paid calls.")
    return run_jobs(jobs, path, run_id, workers=4, max_output_tokens=768)


def run_linkage(split, live):
    folder = ROOT / "results/linkage"
    verify_manifest(folder / "pre-model-manifest.json")
    if split == "test":
        development = json.loads((folder / "model-val-v1-score.json").read_text())
        if not linkage.development_gate(development)["continue_to_locked_test"]:
            raise RuntimeError("The frozen development gate did not pass.")
    phase = "val" if split == "dev" else "test"
    jobs = linkage.model_jobs(phase)
    result = completed_or_run(jobs, folder / f"model-{phase}-v1.json", f"s2aff/v1/{phase}", live)
    answers = {row["case_id"]: row["answer"] for row in result["rows"] if row["status"] == "valid"}
    score = linkage.score_model_answers(answers, phase)
    if split == "dev":
        score["gate"] = linkage.development_gate(score)
    save_or_check(folder / f"model-{phase}-v1-score.json", score)
    return {k: v for k, v in score.items() if k != "predictions"}


def run_coding(split, live):
    folder = ROOT / "results/coding"
    frozen = json.loads((folder / "audited-v2-freeze.json").read_text())
    for name, expected in frozen["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen coding artifact changed: {name}")
    bound = frozen["baseline_evidence"]
    if hashlib.sha256((ROOT / bound["path"]).read_bytes()).hexdigest() != bound["sha256"]:
        raise ValueError("Frozen baseline changed.")
    if split == "test":
        development = json.loads((folder / "model-dev-v2-score.json").read_text())
        if not all(development["summaries"][f"{scheme}_dev"]["development_gate"] == "pass" for scheme in ("naics", "noc")):
            raise RuntimeError("Both frozen scheme gates must pass for the complete test panel.")
    jobs = coding.model_jobs("reviewed-v2", split=split)
    by_id = {job["case_id"]: job for job in jobs}
    result = completed_or_run(jobs, folder / f"model-{split}-v2.json", f"coding/v2/{split}", live)
    records = []
    for row in result["rows"]:
        item = {"id": row["case_id"], "status": "failed"}
        job = by_id[row["case_id"]]
        if row["status"] == "valid":
            try:
                coding.validate_selection(row["answer"], job["user"]["candidates"], job["user"]["source_description"])
                item.update(status="success", prediction=row["answer"])
            except (ValueError, KeyError, TypeError) as error:
                item["error_type"] = type(error).__name__
        records.append(item)
    score = coding.score_model_records(records, split=split)
    save_or_check(folder / f"model-{split}-v2-score.json", score)
    return score["summaries"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["linkage", "coding"])
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--live", action="store_true", help="Permit paid calls only for unfinished declared cases; completed runs stay unchanged.")
    args = parser.parse_args()
    result = run_linkage(args.split, args.live) if args.task == "linkage" else run_coding(args.split, args.live)
    print(json.dumps(result, indent=2))
