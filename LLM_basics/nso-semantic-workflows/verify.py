"""Check saved experiment provenance without network access or model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jsonschema

import runtime


def verify_ledger(path=runtime.LEDGER):
    records = runtime.events(path)
    attempts = [row for row in records if row["event"] == "attempt"]
    results = [row for row in records if row["event"] == "result"]
    if len(attempts) + len(results) != len(records):
        raise ValueError("Unrecognized audit event.")
    if len({r["id"] for r in attempts}) != len(attempts) or len({r["tag"] for r in attempts}) != len(attempts):
        raise ValueError("Duplicate attempt identifier or tag.")
    if len({r["id"] for r in results}) != len(results):
        raise ValueError("Duplicate response.")
    by_id = {row["id"]: row for row in attempts}
    for row in attempts:
        payload = row["payload"]
        if runtime.digest(payload) != row["payload_sha256"]:
            raise ValueError("Request hash mismatch.")
        expected = {"model": runtime.MODEL, "temperature": 0, "reasoning_effort": "none",
                    "response_format": {"type": "json_object"}}
        if any(payload.get(k) != v for k, v in expected.items()) or row["sdk_max_retries"] != 0:
            raise ValueError("Unexpected model settings.")
        if not 1 <= payload["max_completion_tokens"] <= 4096:
            raise ValueError("Unexpected output-token limit.")
    for row in results:
        if row["id"] not in by_id or row["tag"] != by_id[row["id"]]["tag"]:
            raise ValueError("Result without its matching attempt.")
        if row["status"] not in {"valid", "failed"}:
            raise ValueError("Invalid call status.")
        if row["status"] == "valid":
            attempt = by_id[row["id"]]
            response = row["response"]
            if response["model"] != runtime.MODEL or response["choices"][0]["finish_reason"] != "stop":
                raise ValueError("Unexpected completed response.")
            usage = response["usage"]
            if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                raise ValueError("Provider token sum does not reconcile.")
            if usage["completion_tokens"] > attempt["payload"]["max_completion_tokens"]:
                raise ValueError("Provider output exceeded the request limit.")
            if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0):
                raise ValueError("Unexpected reasoning tokens.")
            parsed = json.loads(response["choices"][0]["message"]["content"])
            if parsed != row["parsed"]:
                raise ValueError("Parsed output differs from the saved raw response.")
            schema = json.loads(attempt["payload"]["messages"][0]["content"].rsplit(
                "\nReturn only JSON conforming to this schema:\n", 1)[1])
            jsonschema.validate(parsed, schema)
        if "response" in row and row["response"].get("usage"):
            usage = row["response"]["usage"]
            expected_price = (usage["prompt_tokens"] * runtime.INPUT_USD_PER_MILLION
                              + usage["completion_tokens"] * runtime.OUTPUT_USD_PER_MILLION) / 1e6
            if abs(expected_price - row["estimated_token_price_usd"]) > 1e-12:
                raise ValueError("Token-price estimate does not reconcile.")
            if expected_price > by_id[row["id"]]["reserved_usd"]:
                raise ValueError("Actual token-price estimate exceeded its reservation.")
    summary = runtime.summarize(path)
    if summary["reserved_usd"] > runtime.BUDGET_USD or summary["attempted_calls"] > runtime.MAX_CALLS:
        raise ValueError("The shared experiment limit was exceeded.")
    return summary


def verify_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["files"]:
        path = runtime.ROOT / row["path"]
        if not path.resolve().is_relative_to(runtime.ROOT.resolve()):
            raise ValueError("Manifest path escapes the experiment directory.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Frozen file changed: {row['path']}")
    return {"files_verified": len(manifest["files"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = {"ledger": verify_ledger()}
    if args.manifest:
        result["manifest"] = verify_manifest(args.manifest)
    print(json.dumps(result, indent=2))
