"""Audited, whole-batch admission for the prospective NSO experiments.

The notebook replays saved files. This module makes paid requests only through
an explicit ``run_batch`` call. All jobs are reserved before the first request;
failures and unused reservations stay charged against the experiment envelope.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import threading

import jsonschema
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "results" / "api-ledger.jsonl"
NEW_CEILING = Decimal("9.00")
DEVELOPMENT_CEILING = Decimal("1.00")
PRIOR_RESERVED = Decimal("0.9303031250000006")
PRICES = {
    "gpt-6-luna": (Decimal("0.10"), Decimal("0.50")),
    "gpt-5.6-luna": (Decimal("0.20"), Decimal("1.20")),
}
PRICE_DATE = "2026-10-07"
PRICE_URL = "https://developers.openai.com/api/docs/pricing"
VISION_URL = "https://developers.openai.com/api/docs/guides/images-vision"
MAX_JOBS = 15000
_VERIFIED = set()
_VERIFY_LOCK = threading.Lock()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@contextmanager
def locked_ledger(path=LEDGER):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        # A corrupt or partial entry fails closed. Never ignore a reservation.
        rows = [json.loads(line) for line in handle if line.strip()]
        yield handle, rows


def append_locked(handle, row):
    handle.seek(0, 2)
    handle.write(canonical(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def events(path=LEDGER):
    if not Path(path).exists():
        return []
    with locked_ledger(path) as (_, rows):
        return rows


def image_reference(path, *, reference):
    """Bind an image to bytes and dimensions without putting base64 in a log."""
    with Image.open(path) as im:
        width, height = im.size
        mime = Image.MIME.get(im.format)
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("Only static PNG, JPEG and WebP inputs are supported.")
    return {"reference": reference, "sha256": file_hash(path),
            "width": width, "height": height, "mime": mime, "detail": "high"}


def compact_payload(job):
    """Create the request with image placeholders for review and accounting."""
    required = {"tag", "model", "system", "user", "schema", "max_output_tokens"}
    if set(job) - required - {"images"} or not required <= set(job):
        raise ValueError("Unexpected or missing job fields.")
    if not isinstance(job["tag"], str) or not job["tag"]:
        raise ValueError("Every job needs an immutable nonempty tag.")
    if job["model"] not in PRICES:
        raise ValueError("Model has no verified rate contract.")
    if (type(job["max_output_tokens"]) is not int or
            not 1 <= job["max_output_tokens"] <= 2048):
        raise ValueError("Output cap must be an integer in [1, 2048].")
    jsonschema.Draft202012Validator.check_schema(job["schema"])
    if not isinstance(job["system"], str):
        raise ValueError("System instruction must be text.")
    user = job["user"] if isinstance(job["user"], str) else canonical(job["user"])
    content = [{"type": "text", "text": user}]
    images = job.get("images", [])
    if images and (job["model"] != "gpt-5.6-luna" or len(images) > 1):
        raise ValueError("Vision is restricted to one gpt-5.6-luna image per job.")
    for im in images:
        if (set(im) != {"reference", "sha256", "width", "height", "mime", "detail"}
                or im["detail"] != "high"
                or not isinstance(im["reference"], str)
                or len(im["sha256"]) != 64
                or any(type(im[k]) is not int or im[k] <= 0 for k in ("width", "height"))):
            raise ValueError("Invalid bound image reference.")
        content.append({"type": "image_reference", **im})
    return {"model": job["model"], "temperature": 0, "reasoning_effort": "none",
            "store": False, "service_tier": "default",
            "max_completion_tokens": job["max_output_tokens"],
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": job["system"] +
                          "\nReturn only JSON conforming to this schema:\n" + canonical(job["schema"])},
                         {"role": "user", "content": content}]}


def price_bound(job, transport="standard"):
    """Conservative token-price reservation, not an account billing limit.

    UTF-8 byte length bounds byte-level text tokenization and includes schema and
    request framing. Reserve 512 additional framing tokens and the documented
    3,000-token maximum for a high-detail gpt-5.6-luna image. A 25% input premium
    covers cache writes; a further 25% margin covers accounting uncertainty.
    Every output is reserved at its hard cap. Cached-input savings are ignored.
    """
    payload = compact_payload(job)
    if transport not in {"standard", "batch"}:
        raise ValueError("Unknown processing transport.")
    text_bound = len(canonical(payload).encode("utf-8")) + 512
    if text_bound > 60000:
        raise ValueError("Input exceeds the 60,000-byte text reservation limit.")
    image_bound = 3000 * len(job.get("images", []))
    rate_in, rate_out = PRICES[job["model"]]
    reserved = ((Decimal(text_bound + image_bound) * rate_in * Decimal("1.25")
                 + Decimal(job["max_output_tokens"]) * rate_out)
                * Decimal("1.25") / Decimal(1000000))
    if transport == "batch":
        reserved *= Decimal("0.5")
    return {"reserved_usd": str(reserved), "text_token_bound": text_bound,
            "image_token_bound": image_bound}


def make_manifest(batch_id, stage, jobs, contract, *, transport="standard"):
    """Pure preflight: do not charge or call the provider."""
    if stage not in {"development", "evaluation"} or not batch_id:
        raise ValueError("Specify batch identity and development/evaluation stage.")
    if not jobs or len(jobs) > MAX_JOBS:
        raise ValueError("Batch size is out of bounds.")
    tags = [x["tag"] for x in jobs]
    if len(tags) != len(set(tags)):
        raise ValueError("Duplicate job tags within batch.")
    bounds = [price_bound(job, transport) for job in jobs]
    return {"version": 1, "batch_id": batch_id, "stage": stage,
            "transport": transport,
            "runtime_sha256": file_hash(__file__), "contract": contract,
            "contract_sha256": digest(contract), "jobs": jobs,
            "job_bounds": bounds,
            "reserved_usd": str(sum((Decimal(x["reserved_usd"]) for x in bounds), Decimal(0))),
            "price_date": PRICE_DATE, "price_source": PRICE_URL,
            "vision_source": VISION_URL}


def verify_manifest(manifest):
    # Revalidate if ANY byte-equivalent field or the executing source changes.
    # Avoid repeating thousands of schema checks for every request in a batch.
    key = (digest(manifest), file_hash(__file__))
    with _VERIFY_LOCK:
        if key in _VERIFIED:
            return
        expected = make_manifest(manifest["batch_id"], manifest["stage"],
                                 manifest["jobs"], manifest["contract"],
                                 transport=manifest["transport"])
        if manifest != expected:
            raise ValueError("Manifest, runtime, contract or reservation has changed.")
        _VERIFIED.add(key)


def admit(manifest, *, ledger=LEDGER):
    """Reserve the COMPLETE batch atomically; no partial admission."""
    verify_manifest(manifest)
    sha = digest(manifest)
    with locked_ledger(ledger) as (handle, rows):
        batches = [r for r in rows if r["event"] == "admission"]
        if any(r["batch_id"] == manifest["batch_id"] for r in batches):
            raise ValueError("Batch identity is already admitted.")
        tags = {tag for r in batches for tag in r["tags"]}
        if any(j["tag"] in tags for j in manifest["jobs"]):
            raise ValueError("Job tag was admitted in another batch.")
        total = sum((Decimal(r["reserved_usd"]) for r in batches), Decimal(0))
        dev = sum((Decimal(r["reserved_usd"]) for r in batches
                   if r["stage"] == "development"), Decimal(0))
        cost = Decimal(manifest["reserved_usd"])
        if (total + cost > NEW_CEILING
                or (manifest["stage"] == "development" and dev + cost > DEVELOPMENT_CEILING)
                or len(tags) + len(manifest["jobs"]) > MAX_JOBS):
            raise RuntimeError("Whole batch exceeds the reserved experiment envelope.")
        append_locked(handle, {"event": "admission", "batch_id": manifest["batch_id"],
                               "stage": manifest["stage"], "manifest_sha256": sha,
                               "reserved_usd": manifest["reserved_usd"],
                               "tags": [j["tag"] for j in manifest["jobs"]],
                               "time_utc": utc_now()})
    return sha


def materialize(job, image_root=None):
    payload = compact_payload(job)
    content = payload["messages"][1]["content"]
    for i, part in enumerate(content):
        if part["type"] != "image_reference":
            continue
        if image_root is None:
            raise ValueError("An image cache root is required.")
        root = Path(image_root).resolve()
        path = (root / part["reference"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image reference escapes the supplied cache root.")
        actual = image_reference(path, reference=part["reference"])
        if actual != {k: v for k, v in part.items() if k != "type"}:
            raise ValueError("Image bytes or dimensions changed after freezing.")
        content[i] = {"type": "image_url", "image_url": {
            "url": "data:" + part["mime"] + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii"),
            "detail": "high"}}
    return payload


def call_job(manifest, index, *, ledger=LEDGER, image_root=None, client=None):
    """Run one admitted job exactly once. Invalid/failed output stays recorded."""
    verify_manifest(manifest)
    if manifest["transport"] != "standard":
        raise ValueError("Discounted Batch reservations cannot use synchronous requests.")
    job = manifest["jobs"][index]
    # Missing files/auth are setup defects: stop before consuming an attempt.
    payload = materialize(job, image_root)
    if client is None and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    sha = digest(manifest)
    with locked_ledger(ledger) as (handle, rows):
        admitted = [r for r in rows if r["event"] == "admission" and r["manifest_sha256"] == sha]
        if len(admitted) != 1:
            raise RuntimeError("The exact whole batch must be admitted before requests.")
        if any(r["event"] == "attempt" and r["tag"] == job["tag"] for r in rows):
            raise RuntimeError("No retries: this immutable job has already been attempted.")
        if any(r.get("reservation_exceeded") for r in rows):
            raise RuntimeError("Accounting exceeded a reservation; further requests are stopped.")
        append_locked(handle, {"event": "attempt", "tag": job["tag"],
                               "batch_id": manifest["batch_id"], "manifest_sha256": sha,
                               "job_index": index, "job_sha256": digest(job),
                               "time_utc": utc_now(), "sdk_max_retries": 0,
                               "timeout_seconds": 90})
    started = time.perf_counter()
    result = {"event": "result", "tag": job["tag"], "batch_id": manifest["batch_id"]}
    owned_client = client is None
    try:
        if client is None:
            client = OpenAI(max_retries=0, timeout=90)
        response = client.chat.completions.create(**payload)
        result["response"] = response.model_dump(mode="json")
        usage = response.usage
        if usage:
            ri, ro = PRICES[job["model"]]
            nominal = (Decimal(usage.prompt_tokens) * ri + Decimal(usage.completion_tokens) * ro) / Decimal(1000000)
            upper = (Decimal(usage.prompt_tokens) * ri * Decimal("1.25") + Decimal(usage.completion_tokens) * ro) / Decimal(1000000)
            result.update(estimated_token_price_usd=str(nominal), usage_price_upper_usd=str(upper),
                          reservation_exceeded=upper > Decimal(manifest["job_bounds"][index]["reserved_usd"]))
        if (response.model != job["model"] or len(response.choices) != 1
                or response.choices[0].finish_reason != "stop" or not usage
                or usage.completion_tokens > job["max_output_tokens"]
                or result.get("reservation_exceeded")):
            raise ValueError("Response violates model, completion or accounting contract.")
        if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens:
            raise ValueError("Unexpected reasoning-token use.")
        parsed = json.loads(response.choices[0].message.content)
        jsonschema.validate(parsed, job["schema"])
        result.update(status="valid", parsed=parsed)
    except Exception as error:
        # Error messages can include credentials/request internals. Retain types
        # and status codes; the public manifest already records request content.
        result.update(status="failed", error_type=type(error).__name__,
                      http_status=getattr(error, "status_code", None))
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        with locked_ledger(ledger) as (handle, _):
            append_locked(handle, result)
        if owned_client and client is not None:
            client.close()
    return result


def run_batch(manifest, *, ledger=LEDGER, image_root=None, workers=4):
    """Resume unattempted jobs, never retry an attempted unknown/failed outcome."""
    from concurrent.futures import ThreadPoolExecutor
    verify_manifest(manifest)
    if manifest["transport"] != "standard":
        raise ValueError("Use the Batch API runner for a Batch manifest.")
    rows = events(ledger)
    sha = digest(manifest)
    if not any(r["event"] == "admission" and r["manifest_sha256"] == sha for r in rows):
        raise RuntimeError("Call admit on the complete frozen manifest first.")
    attempted = {r["tag"] for r in rows if r["event"] == "attempt"}
    indices = [i for i, j in enumerate(manifest["jobs"]) if j["tag"] not in attempted]
    if not 1 <= workers <= 8:
        raise ValueError("Use one to eight workers.")
    def run(index):
        return call_job(manifest, index, ledger=ledger, image_root=image_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, indices))


def summarize(ledger=LEDGER):
    rows = events(ledger)
    batches = [r for r in rows if r["event"] == "admission"]
    attempts = [r for r in rows if r["event"] == "attempt"]
    results = [r for r in rows if r["event"] == "result"]
    result_tags = {r["tag"] for r in results}
    reserved = sum((Decimal(r["reserved_usd"]) for r in batches), Decimal(0))
    return {"admitted_jobs": sum(len(r["tags"]) for r in batches),
            "attempted_calls": len(attempts),
            "valid_calls": sum(r["status"] == "valid" for r in results),
            "failed_calls": sum(r["status"] == "failed" for r in results),
            "unresolved_attempts": sum(r["tag"] not in result_tags for r in attempts),
            "unattempted_jobs": sum(len(r["tags"]) for r in batches) - len(attempts),
            "new_reserved_usd": str(reserved),
            "development_reserved_usd": str(sum((Decimal(r["reserved_usd"]) for r in batches if r["stage"] == "development"), Decimal(0))),
            "prior_reserved_usd": str(PRIOR_RESERVED),
            "combined_reserved_usd": str(PRIOR_RESERVED + reserved),
            "nominal_token_price_usd": str(sum((Decimal(r.get("estimated_token_price_usd", "0")) for r in results), Decimal(0))),
            "new_ceiling_usd": str(NEW_CEILING),
            "all_prices_are_estimates_not_invoices": True}


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
