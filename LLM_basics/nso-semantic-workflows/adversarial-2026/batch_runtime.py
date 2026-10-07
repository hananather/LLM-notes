"""Explicit asynchronous Batch transport with whole-manifest financial admission.

``prepare`` only writes local files. Call ``admit`` on its complete plan before
explicitly calling ``submit`` for each chunk. ``poll`` makes one read-only status
request; ``retrieve`` saves terminal evidence and joins results by custom_id.
No function loops until completion or falls back to synchronous inference.

Unknown upload/creation outcomes cannot be submitted again. Reservations are
never released, including for expired, failed or missing responses. The shared
runtime ledger remains the financial authority. Request JSONL belongs in an
external cache; the compact plan and response evidence belong beside results.

Contract: https://developers.openai.com/api/docs/guides/batch (2026-10-07).
The documented limit is 200 MB; this transport caps each file at 80,000,000
bytes and uses one model per file. The verified Batch rate is half the standard
rate, including the input cache-write upper bound. SDK retries are disabled.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import jsonschema
from openai import OpenAI

import runtime as rt

MAX_CHUNK_BYTES = 80_000_000
ENDPOINT = "/v1/chat/completions"
TERMINAL = {"completed", "expired", "failed", "cancelled"}
REQUIRED_BINDINGS = {"runtime.py", "evaluation.py", "model_jobs.py",
                     "decision_adapters.py", "protocol.json", "batch_runtime.py"}


def _bytes(value):
    return (rt.canonical(value) + "\n").encode("utf-8")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _strict_json(data):
    """Decode JSON whose numeric values can be persisted in the shared ledger.

    Python accepts NaN/Infinity extensions by default and turns large finite
    number literals such as 1e309 into infinity. Reject both at any nesting
    depth, before a provider row or parsed model output enters a result event.
    Raw provider bytes are preserved separately by the download evidence file.
    """
    def reject_constant(_):
        raise ValueError("Nonfinite JSON constant.")

    def finite_float(literal):
        value = float(literal)
        if not math.isfinite(value):
            raise ValueError("JSON number exceeds finite representation.")
        return value

    return json.loads(data, parse_constant=reject_constant, parse_float=finite_float)


def _save(path, data):
    """Publish complete bytes atomically, refusing to replace different bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Frozen artifact changed: " + str(path))
        return
    fd, temporary = tempfile.mkstemp(prefix=".batch-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("Concurrent artifact conflict: " + str(path))
    finally:
        os.unlink(temporary)


def _sources(manifest):
    rt.verify_manifest(manifest)
    if manifest["transport"] != "batch":
        raise ValueError("Batch transport requires its own discounted manifest.")
    contract = manifest["contract"]
    files = contract.get("files", {})
    if contract.get("status") != "frozen_for_execution" or not REQUIRED_BINDINGS <= set(files):
        raise ValueError("Frozen scientific and Batch transport source bindings are required.")
    root = rt.ROOT.resolve()
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or rt.file_hash(path) != expected:
            raise ValueError("Bound experiment file changed: " + name)
    if files["batch_runtime.py"] != rt.file_hash(__file__):
        raise ValueError("The executing Batch transport differs from its source binding.")


def _custom_id(index, job):
    return f"j-{index:06d}-{rt.digest(job)[:16]}"


def _chunks(manifest, image_root, limit):
    """Deterministic model order, original within-model order, exact payloads."""
    for model in sorted({job["model"] for job in manifest["jobs"]}):
        data, jobs = bytearray(), []
        for index, job in enumerate(manifest["jobs"]):
            if job["model"] != model:
                continue
            request = {"custom_id": _custom_id(index, job), "method": "POST",
                       "url": ENDPOINT, "body": rt.materialize(job, image_root)}
            line = _bytes(request)
            if len(line) > limit:
                raise ValueError("One exact request exceeds the chunk byte limit.")
            if data and len(data) + len(line) > limit:
                yield model, bytes(data), jobs
                data, jobs = bytearray(), []
            data.extend(line)
            jobs.append({"index": index, "custom_id": request["custom_id"],
                         "job_sha256": rt.digest(job), "request_sha256": _hash(line)})
        if data:
            yield model, bytes(data), jobs


def prepare(manifest, *, plan_path, cache_dir, image_root=None,
            max_chunk_bytes=MAX_CHUNK_BYTES):
    """Unpaid preflight of ALL exact requests; never admit or use the SDK.

    Existing equal files are reusable; changed files fail closed. ``cache_dir``
    must be outside the experiment tree so embedded image bytes cannot be added
    with the result directory. An ignored external cache is also appropriate.
    """
    _sources(manifest)
    if type(max_chunk_bytes) is not int or not 1 <= max_chunk_bytes <= MAX_CHUNK_BYTES:
        raise ValueError("Invalid conservative chunk limit.")
    cache = Path(cache_dir).resolve()
    if cache.is_relative_to(rt.ROOT.resolve()):
        raise ValueError("Keep bulky requests outside the experiment tree.")
    path = Path(plan_path).resolve()
    sha = rt.digest(manifest)
    plan = {"version": 1, "manifest_sha256": sha,
            "batch_runtime_sha256": rt.file_hash(__file__),
            "max_chunk_bytes": max_chunk_bytes, "endpoint": ENDPOINT,
            "completion_window": "24h", "cache_dir": str(cache),
            "image_root": str(Path(image_root).resolve()) if image_root is not None else None,
            "plan_path": str(path), "evidence_dir": str(path.parent / (path.stem + "-evidence")),
            "job_count": len(manifest["jobs"]), "chunks": []}
    for number, (model, data, jobs) in enumerate(_chunks(manifest, image_root, max_chunk_bytes)):
        request_path = cache / sha / f"chunk-{number:04d}-{model}.jsonl"
        _save(request_path, data)
        plan["chunks"].append({"index": number, "model": model,
                               "path": str(request_path), "sha256": _hash(data),
                               "bytes": len(data), "jobs": jobs})
    plan["plan_sha256"] = rt.digest(plan)
    _save(path, _bytes(plan))
    return plan


def verify_plan(manifest, plan, *, materialize=False):
    """Verify source bindings, complete coverage, persisted plan and input bytes."""
    _sources(manifest)
    core = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (plan["plan_sha256"] != rt.digest(core)
            or plan["manifest_sha256"] != rt.digest(manifest)
            or plan["batch_runtime_sha256"] != rt.file_hash(__file__)
            or plan["endpoint"] != ENDPOINT or plan["completion_window"] != "24h"
            or plan["job_count"] != len(manifest["jobs"])
            or type(plan["max_chunk_bytes"]) is not int
            or not 1 <= plan["max_chunk_bytes"] <= MAX_CHUNK_BYTES):
        raise ValueError("Manifest or Batch plan changed.")
    if Path(plan["plan_path"]).read_bytes() != _bytes(plan):
        raise ValueError("Persisted plan differs from supplied plan.")
    indices = []
    cache = Path(plan["cache_dir"]).resolve()
    if cache.is_relative_to(rt.ROOT.resolve()):
        raise ValueError("Request cache overlaps experiment tree.")
    for number, chunk in enumerate(plan["chunks"]):
        expected_path = cache / plan["manifest_sha256"] / f"chunk-{number:04d}-{chunk['model']}.jsonl"
        if (chunk["index"] != number or Path(chunk["path"]) != expected_path
                or not 0 < chunk["bytes"] <= plan["max_chunk_bytes"]
                or expected_path.stat().st_size != chunk["bytes"]
                or rt.file_hash(expected_path) != chunk["sha256"]):
            raise ValueError("Chunk path, size or bytes changed.")
        with expected_path.open(encoding="utf-8") as handle:
            requests = [json.loads(line) for line in handle]
        if len(requests) != len(chunk["jobs"]) or not requests:
            raise ValueError("Chunk request coverage changed.")
        for request, reference in zip(requests, chunk["jobs"]):
            index = reference["index"]
            if type(index) is not int or not 0 <= index < len(manifest["jobs"]):
                raise ValueError("Invalid manifest job index.")
            job = manifest["jobs"][index]
            if (reference["job_sha256"] != rt.digest(job)
                    or reference["custom_id"] != _custom_id(index, job)
                    or request["custom_id"] != reference["custom_id"]
                    or reference["request_sha256"] != _hash(_bytes(request))
                    or request["method"] != "POST" or request["url"] != ENDPOINT
                    or request["body"]["model"] != job["model"]
                    or request["body"]["model"] != chunk["model"]):
                raise ValueError("Request and frozen job disagree.")
            if materialize and request["body"] != rt.materialize(job, plan["image_root"]):
                raise ValueError("Exact materialized request differs from frozen input.")
            indices.append(index)
    if sorted(indices) != list(range(len(manifest["jobs"]))):
        raise ValueError("Plan does not cover every job exactly once.")


def _related(rows, plan, event=None, chunk_index=None):
    return [r for r in rows if r.get("plan_sha256") == plan["plan_sha256"]
            and (event is None or r["event"] == event)
            and (chunk_index is None or r.get("chunk_index") == chunk_index)]


def _event(event_type, manifest, plan, chunk_index=None, **fields):
    row = {"event": event_type, "batch_id": manifest["batch_id"],
           "manifest_sha256": plan["manifest_sha256"],
           "plan_sha256": plan["plan_sha256"], "time_utc": rt.utc_now(), **fields}
    if chunk_index is not None:
        row["chunk_index"] = chunk_index
    return row


def _admitted(rows, manifest, plan):
    admissions = [r for r in rows if r["event"] == "admission"
                  and r["manifest_sha256"] == plan["manifest_sha256"]]
    plans = [r for r in rows if r["event"] == "batch_plan"
             and r["manifest_sha256"] == plan["manifest_sha256"]]
    if (len(admissions) != 1 or len(plans) != 1
            or plans[0]["plan_sha256"] != plan["plan_sha256"]
            or admissions[0]["tags"] != [j["tag"] for j in manifest["jobs"]]):
        raise RuntimeError("Admit the exact COMPLETE manifest and Batch plan first.")


def admit(manifest, plan, *, ledger=rt.LEDGER):
    """Pin one plan and reserve all jobs, including chunks not yet uploaded."""
    verify_plan(manifest, plan, materialize=True)
    sha = plan["manifest_sha256"]
    if not any(r["event"] == "admission" and r["manifest_sha256"] == sha for r in rt.events(ledger)):
        try:
            rt.admit(manifest, ledger=ledger)
        except ValueError:
            # Another caller may have admitted the identical manifest.
            if not any(r["event"] == "admission" and r["manifest_sha256"] == sha for r in rt.events(ledger)):
                raise
    with rt.locked_ledger(ledger) as (handle, rows):
        plans = [r for r in rows if r["event"] == "batch_plan" and r["manifest_sha256"] == sha]
        if plans and (len(plans) != 1 or plans[0]["plan_sha256"] != plan["plan_sha256"]):
            raise RuntimeError("This manifest is already bound to a different Batch plan.")
        if not plans:
            rt.append_locked(handle, _event("batch_plan", manifest, plan,
                                           chunk_count=len(plan["chunks"])))
    return sha


@contextmanager
def _client(client):
    owned = client is None
    if owned:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        client = OpenAI(max_retries=0, timeout=90)
    if getattr(client, "max_retries", None) != 0:
        if owned:
            client.close()
        raise ValueError("SDK client must explicitly disable retries.")
    try:
        yield client
    finally:
        if owned:
            client.close()


def _dump(obj):
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else dict(obj)


def _metadata(plan, chunk):
    return {"manifest_sha256": plan["manifest_sha256"],
            "chunk_sha256": chunk["sha256"], "plan_sha256": plan["plan_sha256"]}


def _check_provider_batch(snapshot, plan, chunk, input_file_id, expected_id=None):
    if (not isinstance(snapshot.get("id"), str) or not snapshot["id"]
            or (expected_id is not None and snapshot["id"] != expected_id)
            or snapshot.get("input_file_id") != input_file_id
            or snapshot.get("endpoint") != ENDPOINT
            or snapshot.get("completion_window") != "24h"
            or snapshot.get("metadata") != _metadata(plan, chunk)):
        raise ValueError("Provider Batch identity does not match the frozen chunk.")


def submit(manifest, plan, chunk_index, *, ledger=rt.LEDGER, client=None):
    """Explicit provider writes: one upload and at most one creation attempt.

    Once the submission event is fsynced, this entry point never retries the
    chunk, even when the provider outcome is unknown. Known batches use poll.
    """
    verify_plan(manifest, plan, materialize=True)
    if type(chunk_index) is not int or not 0 <= chunk_index < len(plan["chunks"]):
        raise ValueError("Invalid chunk index.")
    chunk = plan["chunks"][chunk_index]
    with _client(client) as api:
        with rt.locked_ledger(ledger) as (handle, rows):
            _admitted(rows, manifest, plan)
            if _related(rows, plan, "batch_submission_attempt", chunk_index):
                raise RuntimeError("Never resubmit an attempted chunk, including unknown outcomes.")
            if any(r.get("reservation_exceeded") for r in rows):
                raise RuntimeError("Accounting alert stops further submissions.")
            tags = {manifest["jobs"][j["index"]]["tag"] for j in chunk["jobs"]}
            if any(r["event"] == "attempt" and r["tag"] in tags for r in rows):
                raise RuntimeError("One or more chunk jobs were already attempted.")
            rt.append_locked(handle, _event("batch_submission_attempt", manifest, plan, chunk_index,
                                           chunk_sha256=chunk["sha256"], sdk_max_retries=0))
        phase = "upload"
        try:
            # Send an in-memory snapshot of verified bytes, avoiding path reopen
            # races between hashing and upload; at most 80 MB per chunk.
            data = Path(chunk["path"]).read_bytes()
            if _hash(data) != chunk["sha256"]:
                raise ValueError("Request bytes changed before upload.")
            uploaded = _dump(api.files.create(file=(Path(chunk["path"]).name, data, "application/jsonl"), purpose="batch"))
            file_id = uploaded.get("id")
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("Provider returned no input file ID.")
            with rt.locked_ledger(ledger) as (handle, rows):
                rt.append_locked(handle, _event("batch_uploaded", manifest, plan, chunk_index,
                                               input_file_id=file_id, chunk_sha256=chunk["sha256"]))
                if any(r.get("reservation_exceeded") for r in rows):
                    raise RuntimeError("Accounting alert stops creation after upload.")
                for reference in chunk["jobs"]:
                    index = reference["index"]
                    job = manifest["jobs"][index]
                    if any(r["event"] == "attempt" and r["tag"] == job["tag"] for r in rows):
                        raise RuntimeError("Job attempt collision.")
                    rt.append_locked(handle, _event("attempt", manifest, plan, chunk_index,
                        tag=job["tag"], job_index=index, job_sha256=reference["job_sha256"],
                        custom_id=reference["custom_id"], sdk_max_retries=0, transport="batch"))
                rt.append_locked(handle, _event("batch_create_attempt", manifest, plan, chunk_index,
                    input_file_id=file_id, chunk_sha256=chunk["sha256"], sdk_max_retries=0,
                    completion_window="24h", endpoint=ENDPOINT))
            phase = "create"
            snapshot = _dump(api.batches.create(input_file_id=file_id, endpoint=ENDPOINT,
                              completion_window="24h", metadata=_metadata(plan, chunk)))
            _check_provider_batch(snapshot, plan, chunk, file_id)
            with rt.locked_ledger(ledger) as (handle, _):
                rt.append_locked(handle, _event("batch_created", manifest, plan, chunk_index,
                    provider_batch_id=snapshot["id"], input_file_id=file_id, snapshot=snapshot))
            return snapshot
        except Exception as error:
            with rt.locked_ledger(ledger) as (handle, _):
                rt.append_locked(handle, _event("batch_submission_failed", manifest, plan, chunk_index,
                    phase=phase, outcome="unknown_or_failed_no_resubmit", error_type=type(error).__name__,
                    http_status=getattr(error, "status_code", None)))
            raise


def _created(manifest, plan, chunk_index, ledger):
    if type(chunk_index) is not int or not 0 <= chunk_index < len(plan["chunks"]):
        raise ValueError("Invalid chunk index.")
    rows = rt.events(ledger)
    _admitted(rows, manifest, plan)
    created = _related(rows, plan, "batch_created", chunk_index)
    if len(created) != 1:
        raise RuntimeError("No uniquely known created batch; automatic resubmission is forbidden.")
    return created[0]


def poll(manifest, plan, chunk_index, *, ledger=rt.LEDGER, client=None):
    """One read-only status request for a ledger-recorded provider batch."""
    verify_plan(manifest, plan)
    created = _created(manifest, plan, chunk_index, ledger)
    with _client(client) as api:
        snapshot = _dump(api.batches.retrieve(created["provider_batch_id"]))
    _check_provider_batch(snapshot, plan, plan["chunks"][chunk_index],
                          created["input_file_id"], created["provider_batch_id"])
    with rt.locked_ledger(ledger) as (handle, _):
        rt.append_locked(handle, _event("batch_status", manifest, plan, chunk_index,
                                        provider_batch_id=snapshot["id"], snapshot=snapshot))
    return snapshot


def _download(api, file_id, kind, manifest, plan, chunk_index, ledger):
    path = Path(plan["evidence_dir"]) / f"chunk-{chunk_index:04d}" / (kind + ".jsonl")
    with rt.locked_ledger(ledger) as (handle, rows):
        prior = [r for r in _related(rows, plan, "batch_downloaded", chunk_index) if r["kind"] == kind]
        if prior:
            if len(prior) != 1 or prior[0]["file_id"] != file_id or rt.file_hash(path) != prior[0]["sha256"]:
                raise ValueError("Saved provider evidence changed.")
            return path.read_bytes()
        # A crashed download may be repeated read-only, never a model request.
        rt.append_locked(handle, _event("batch_download_attempt", manifest, plan, chunk_index,
                                        file_id=file_id, kind=kind))
    response = api.files.content(file_id)
    data = response.content
    if not isinstance(data, bytes):
        raise TypeError("Files content response must expose bytes.")
    _save(path, data)
    with rt.locked_ledger(ledger) as (handle, rows):
        prior = [r for r in _related(rows, plan, "batch_downloaded", chunk_index) if r["kind"] == kind]
        if not prior:
            rt.append_locked(handle, _event("batch_downloaded", manifest, plan, chunk_index,
                file_id=file_id, kind=kind, path=str(path), sha256=_hash(data), bytes=len(data)))
    return data


def _result(manifest, plan, chunk, reference, entries, provider_id):
    index = reference["index"]
    job = manifest["jobs"][index]
    result = _event("result", manifest, plan, chunk["index"], tag=job["tag"],
                    job_index=index, custom_id=reference["custom_id"], transport="batch",
                    provider_batch_id=provider_id, status="failed")
    if len(entries) != 1:
        result["failure"] = "missing_batch_result" if not entries else "duplicate_custom_id"
        if entries:
            result["reservation_exceeded"] = True  # Unknown duplicate usage: stop new requests.
        return result
    entry = entries[0]
    response = entry.get("response")
    result["provider_request_id"] = response.get("request_id") if isinstance(response, dict) else None
    if entry.get("error") is not None or not isinstance(response, dict) or response.get("status_code") != 200:
        result.update(failure="provider_request_error", provider_error=entry.get("error"),
                      http_status=response.get("status_code") if isinstance(response, dict) else None)
        return result
    body = response.get("body")
    result["response"] = body
    try:
        usage = body["usage"]
        prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
        if any(type(v) is not int or v < 0 for v in (prompt, completion)):
            raise ValueError("Invalid usage counts.")
        ri, ro = rt.PRICES[job["model"]]
        discount = Decimal("0.5") / Decimal(1_000_000)
        nominal = (Decimal(prompt) * ri + Decimal(completion) * ro) * discount
        upper = (Decimal(prompt) * ri * Decimal("1.25") + Decimal(completion) * ro) * discount
        result.update(estimated_token_price_usd=str(nominal), usage_price_upper_usd=str(upper),
                      reservation_exceeded=upper > Decimal(manifest["job_bounds"][index]["reserved_usd"]))
        choices = body["choices"]
        if (body["model"] != job["model"] or len(choices) != 1
                or choices[0]["finish_reason"] != "stop" or completion > job["max_output_tokens"]
                or result["reservation_exceeded"]):
            raise ValueError("Response violates model, completion or accounting contract.")
        details = usage.get("completion_tokens_details")
        if details and details.get("reasoning_tokens"):
            raise ValueError("Unexpected reasoning-token use.")
        parsed = _strict_json(choices[0]["message"]["content"])
        jsonschema.validate(parsed, job["schema"])
        result.update(status="valid", parsed=parsed)
    except Exception as error:
        result.update(failure="invalid_batch_response", error_type=type(error).__name__)
    return result


def retrieve(manifest, plan, chunk_index, *, ledger=rt.LEDGER, client=None):
    """Save terminal files once, join by custom_id, and retain every failed job.

    Call poll first. Repeated retrieval reuses hash-verified saved files and
    appends no duplicate results; interrupted local ingestion can be resumed.
    Read-only file retrieval may be called again after a failed download.
    """
    verify_plan(manifest, plan)
    created = _created(manifest, plan, chunk_index, ledger)
    chunk = plan["chunks"][chunk_index]
    rows = rt.events(ledger)
    statuses = _related(rows, plan, "batch_status", chunk_index)
    snapshot = statuses[-1]["snapshot"] if statuses else created["snapshot"]
    if snapshot.get("status") not in TERMINAL:
        raise RuntimeError("Batch is not terminal; poll once later.")
    _check_provider_batch(snapshot, plan, chunk, created["input_file_id"], created["provider_batch_id"])
    evidence_path = Path(plan["evidence_dir"]) / f"chunk-{chunk_index:04d}" / "terminal.json"
    _save(evidence_path, _bytes(snapshot))
    known_ids = {j["custom_id"] for j in chunk["jobs"]}
    by_id, anomalies = defaultdict(list), []
    saved = _related(rows, plan, "batch_downloaded", chunk_index)
    needs_download = any(snapshot.get(kind + "_file_id")
                         and not any(r["kind"] == kind for r in saved)
                         for kind in ("output", "error"))
    # Fully cached ingestion is unpaid and does not need provider credentials.
    with (_client(client) if needs_download else nullcontext(None)) as api:
        for kind in ("output", "error"):
            file_id = snapshot.get(kind + "_file_id")
            if not file_id:
                continue
            data = _download(api, file_id, kind, manifest, plan, chunk_index, ledger)
            for line_number, line in enumerate(data.splitlines(), 1):
                try:
                    entry = _strict_json(line)
                    custom_id = entry.get("custom_id")
                    if not isinstance(custom_id, str) or custom_id not in known_ids:
                        raise ValueError("Unknown or absent custom_id.")
                    by_id[custom_id].append(entry)
                except Exception as error:
                    anomalies.append({"kind": kind, "line": line_number,
                                      "error_type": type(error).__name__, "line_sha256": _hash(line)})
    results = [_result(manifest, plan, chunk, reference, by_id[reference["custom_id"]], snapshot["id"])
               for reference in chunk["jobs"]]
    with rt.locked_ledger(ledger) as (handle, rows):
        for result in results:
            prior = [r for r in rows if r["event"] == "result" and r["tag"] == result["tag"]]
            if prior:
                # Time is local ingestion metadata; all scientific evidence must agree.
                stable = lambda value: {k: v for k, v in value.items() if k != "time_utc"}
                if len(prior) != 1 or stable(prior[0]) != stable(result):
                    raise ValueError("Existing result conflicts with provider evidence.")
            else:
                rt.append_locked(handle, result)
        if not _related(rows, plan, "batch_retrieved", chunk_index):
            rt.append_locked(handle, _event("batch_retrieved", manifest, plan, chunk_index,
                provider_batch_id=snapshot["id"], terminal_status=snapshot["status"],
                expected_jobs=len(chunk["jobs"]), valid_jobs=sum(r["status"] == "valid" for r in results),
                failed_jobs=sum(r["status"] == "failed" for r in results), anomalies=anomalies,
                reservation_exceeded=bool(anomalies)))
    return results
