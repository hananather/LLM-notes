"""Bounded, auditable JSON model calls for the NSO workflow experiments.

Offline replay is the default in the notebook. Live calls require an explicit
runner invocation and the existing OPENAI_API_KEY environment variable.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import jsonschema
from openai import OpenAI
import tiktoken

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-6-luna"
INPUT_USD_PER_MILLION = 0.10
OUTPUT_USD_PER_MILLION = 0.50
BUDGET_USD = 10.0
MAX_CALLS = 3000
PRICE_CHECK_DATE = "2026-10-06"
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-6-luna"
LEDGER = ROOT / "results" / "api-ledger.jsonl"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def events(path=LEDGER):
    if not Path(path).exists():
        return []
    with Path(path).open() as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        return [json.loads(line) for line in handle if line.strip()]


def reserve(payload, tag, ledger=LEDGER):
    """Keep conservative reservations permanently, including failed calls.

    A cross-process file lock serializes all experiment reservations. The input
    estimate includes the complete schema, 256 framing tokens and a 25% margin;
    a further 25% input-price allowance covers cache-write premiums. Output is
    reserved at its hard limit. This is a token-price cap, not a billing receipt.
    """
    encoder = tiktoken.get_encoding("o200k_base")
    n_input = len(encoder.encode(canonical(payload))) + 256
    if n_input > 60000:
        raise ValueError("Request exceeds the experiment's 60,000 input-token estimate limit.")
    reservation = (n_input * INPUT_USD_PER_MILLION * 1.25
                   + payload["max_completion_tokens"] * OUTPUT_USD_PER_MILLION) / 1e6 * 1.25
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        prior = [json.loads(line) for line in handle if line.strip()]
        attempts = [x for x in prior if x["event"] == "attempt"]
        if len(attempts) >= MAX_CALLS or sum(x["reserved_usd"] for x in attempts) + reservation > BUDGET_USD:
            raise RuntimeError("Shared experiment call or token-price budget exhausted.")
        if any(x["tag"] == tag for x in attempts):
            raise RuntimeError("Duplicate attempt tag; use an explicitly named new experiment for a rerun.")
        item = {"event": "attempt", "id": str(uuid.uuid4()), "tag": tag,
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "payload": payload, "payload_sha256": digest(payload),
                "input_tokens_estimated": n_input, "reserved_usd": reservation,
                "model": MODEL, "sdk_max_retries": 0, "timeout_seconds": 90,
                "price_check_date": PRICE_CHECK_DATE, "price_source": PRICE_SOURCE}
        handle.write(canonical(item) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return item


def append(item, ledger=LEDGER):
    with Path(ledger).open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(canonical(item) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def call_json(system, user, schema, max_output_tokens=768, tag=None, *, ledger=LEDGER):
    """Return a validated JSON object; preserve failures without retrying.

    The schema is supplied in the actual system message and validated locally.
    JSON mode enforces JSON syntax; it does not enforce semantic correctness.
    No gold labels or evaluation feedback may enter these messages.
    """
    if not tag:
        raise ValueError("An immutable experiment/case/arm/step tag is required.")
    if not 1 <= max_output_tokens <= 4096:
        raise ValueError("Output-token limit must be between 1 and 4096.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Existing OPENAI_API_KEY environment configuration is required.")
    jsonschema.Draft202012Validator.check_schema(schema)
    payload = {"model": MODEL, "temperature": 0, "reasoning_effort": "none",
               "max_completion_tokens": max_output_tokens,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": system + "\nReturn only JSON conforming to this schema:\n" + canonical(schema)},
                            {"role": "user", "content": user if isinstance(user, str) else canonical(user)}]}
    attempt = reserve(payload, tag, ledger)
    start = time.perf_counter()
    item = {"event": "result", "id": attempt["id"], "tag": tag}
    try:
        with OpenAI(max_retries=0, timeout=90) as client:
            response = client.chat.completions.create(**payload)
        item["response"] = response.model_dump(mode="json")
        usage = response.usage
        if usage:
            item["estimated_token_price_usd"] = (usage.prompt_tokens * INPUT_USD_PER_MILLION
                                                 + usage.completion_tokens * OUTPUT_USD_PER_MILLION) / 1e6
        if response.model != MODEL:
            raise ValueError("Provider returned an unexpected model identifier.")
        if len(response.choices) != 1 or response.choices[0].finish_reason != "stop":
            raise ValueError("The response did not finish normally.")
        if not usage or usage.completion_tokens > max_output_tokens:
            raise ValueError("Missing or invalid provider token accounting.")
        if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens:
            raise ValueError("Unexpected reasoning-token use.")
        parsed = json.loads(response.choices[0].message.content)
        jsonschema.validate(parsed, schema)
        item.update(status="valid", parsed=parsed)
        return parsed
    except Exception as error:
        # Exception messages can embed request details; record only safe types
        # and status codes. Public request/response evidence is already saved.
        item.update(status="failed", error_type=type(error).__name__,
                    http_status=getattr(error, "status_code", None))
        raise
    finally:
        item["elapsed_seconds"] = time.perf_counter() - start
        append(item, ledger)


def summarize(ledger=LEDGER):
    rows = events(ledger)
    attempted = [x for x in rows if x["event"] == "attempt"]
    results = [x for x in rows if x["event"] == "result"]
    by_id = {x["id"]: x for x in results}
    usage = [x.get("response", {}).get("usage") or {} for x in results]
    return {"model": MODEL, "attempted_calls": len(attempted),
            "valid_calls": sum(x["status"] == "valid" for x in results),
            "failed_calls": sum(x["status"] == "failed" for x in results),
            "unresolved_attempts": sum(x["id"] not in by_id for x in attempted),
            "reserved_usd": sum(x["reserved_usd"] for x in attempted),
            "estimated_token_price_usd": sum(x.get("estimated_token_price_usd", 0) for x in results),
            "prompt_tokens": sum(x.get("prompt_tokens", 0) for x in usage),
            "completion_tokens": sum(x.get("completion_tokens", 0) for x in usage),
            "call_elapsed_seconds_sum": sum(x["elapsed_seconds"] for x in results),
            "budget_usd": BUDGET_USD}


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
