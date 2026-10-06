"""Run frozen independent JSON jobs and retain successes and failed attempts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

from runtime import call_json, canonical, digest, events


def run_jobs(jobs, destination, run_id, *, workers=4, max_output_tokens=768):
    """Resume only saved completions; never silently repeat an attempted call.

    Each job has case_id, system, user and schema. A retry requires a different
    declared run_id; all original results remain in the shared API ledger.
    """
    jobs = list(jobs)
    if len({j["case_id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate case IDs in one batch.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    spec = {"run_id": run_id, "jobs_sha256": digest(jobs), "case_count": len(jobs),
            "default_max_output_tokens": max_output_tokens, "workers": workers}
    spec_path = destination.with_suffix(".spec.json")
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError("A saved batch cannot be resumed with altered jobs or settings.")
    else:
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    prior = events()
    attempts = {r["tag"]: r for r in prior if r["event"] == "attempt"}
    results = {r["tag"]: r for r in prior if r["event"] == "result"}

    def run(job):
        tag = f"{run_id}/{job['case_id']}"
        if tag in attempts:
            saved = attempts[tag]["payload"]
            expected_messages = [
                {"role": "system", "content": job["system"] + "\nReturn only JSON conforming to this schema:\n" + canonical(job["schema"])},
                {"role": "user", "content": job["user"] if isinstance(job["user"], str) else canonical(job["user"])},
            ]
            if (saved["messages"] != expected_messages
                    or saved["max_completion_tokens"] != job.get("max_output_tokens", max_output_tokens)):
                raise ValueError("The existing attempt tag belongs to a different request.")
            prior_result = results.get(tag)
            if prior_result and prior_result["status"] == "valid":
                return {"case_id": job["case_id"], "status": "valid", "answer": prior_result["parsed"], "tag": tag}
            return {"case_id": job["case_id"], "status": "failed", "error_type": (prior_result or {}).get("error_type", "UnresolvedAttempt"), "tag": tag}
        try:
            answer = call_json(job["system"], job["user"], job["schema"],
                               job.get("max_output_tokens", max_output_tokens), tag)
            return {"case_id": job["case_id"], "status": "valid", "answer": answer, "tag": tag}
        except Exception as error:
            return {"case_id": job["case_id"], "status": "failed", "error_type": type(error).__name__, "tag": tag}

    start = time.perf_counter()
    completed = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run, job): job["case_id"] for job in jobs}
        for future in as_completed(pending):
            row = future.result()
            completed[row["case_id"]] = row
            # The ledger is the durable record; this compact snapshot is replaced
            # atomically as the same batch progresses, never used as new truth.
            snapshot = {"spec": spec, "status": "complete" if len(completed) == len(jobs) else "running",
                        "rows": [completed[j["case_id"]] for j in jobs if j["case_id"] in completed],
                        "wall_seconds": time.perf_counter() - start}
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(destination)
            if len(completed) % 20 == 0 or len(completed) == len(jobs):
                print(f"{run_id}: {len(completed)}/{len(jobs)} completed", flush=True)
    return json.loads(destination.read_text())
