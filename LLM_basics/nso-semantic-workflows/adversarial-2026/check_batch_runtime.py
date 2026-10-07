"""Offline Batch lifecycle and financial checks; no provider operations occur."""
from concurrent.futures import ThreadPoolExecutor
import copy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import batch_runtime as br
import decision_adapters
import runtime as rt


def job(tag="case", **changes):
    return {"tag": tag, "model": "gpt-6-luna", "system": "Extract observed text.",
            "user": {"name": "Example"}, "schema": {"type": "object",
            "properties": {"name": {"type": "string"}}, "required": ["name"],
            "additionalProperties": False}, "max_output_tokens": 128, **changes}


def response(custom_id, text='{"name":"Example"}', *, model="gpt-6-luna", **body_changes):
    body = {"model": model, "usage": {"prompt_tokens": 20, "completion_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 0}},
            "choices": [{"finish_reason": "stop", "message": {"content": text}}], **body_changes}
    return {"custom_id": custom_id, "response": {"status_code": 200,
            "request_id": "request-fixture", "body": body}, "error": None}


class FakeClient:
    """In-memory SDK-shaped client. No HTTP implementation is present."""
    max_retries = 0

    def __init__(self, *, create_failure=False, upload_failure=False, before_create=None):
        self.create_failure, self.upload_failure = create_failure, upload_failure
        self.before_create = before_create
        self.uploads, self.creations, self.polls, self.downloads = [], [], [], []
        self.objects, self.contents = {}, {}
        self.files = SimpleNamespace(create=self.upload, content=self.content)
        self.batches = SimpleNamespace(create=self.create, retrieve=self.status)

    def upload(self, **kwargs):
        self.uploads.append(kwargs)
        if self.upload_failure:
            raise TimeoutError("fixture upload outcome is unknown")
        return {"id": f"file-{len(self.uploads)}", "purpose": "batch"}

    def create(self, **kwargs):
        self.creations.append(kwargs)
        if self.before_create:
            self.before_create()
        if self.create_failure:
            raise TimeoutError("fixture creation outcome is unknown")
        row = {"id": f"batch-{len(self.creations)}", "status": "validating",
               "output_file_id": None, "error_file_id": None, **kwargs}
        self.objects[row["id"]] = row
        return copy.deepcopy(row)

    def status(self, provider_id):
        self.polls.append(provider_id)
        return copy.deepcopy(self.objects[provider_id])

    def content(self, file_id):
        self.downloads.append(file_id)
        return SimpleNamespace(content=self.contents[file_id])

    def terminal(self, provider_id, output=(), errors=(), status="completed"):
        row = self.objects[provider_id]
        row["status"] = status
        for kind, values in (("output", output), ("error", errors)):
            if values:
                fid = provider_id + "-" + kind
                row[kind + "_file_id"] = fid
                self.contents[fid] = b"".join(br._bytes(value) for value in values)


class BatchChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "experiment"
        self.root.mkdir()
        for name in br.REQUIRED_BINDINGS:
            source = Path(br.__file__) if name == "batch_runtime.py" else rt.ROOT / name
            (self.root / name).write_bytes(source.read_bytes())
        self.root_patch = patch.object(rt, "ROOT", self.root)
        self.root_patch.start()
        self.ledger = self.base / "ledger.jsonl"
        self.api = FakeClient()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def manifest(self, jobs=None, *, name="batch", stage="development", transport="batch"):
        contract = {"status": "frozen_for_execution", "files": {
            name: rt.file_hash(self.root / name) for name in br.REQUIRED_BINDINGS}}
        return rt.make_manifest(name, stage, jobs or [job()], contract, transport=transport)

    def prepare(self, manifest, **kwargs):
        return br.prepare(manifest, plan_path=self.root / (manifest["batch_id"] + "-plan.json"),
                          cache_dir=self.base / "request-cache", **kwargs)

    def start(self, manifest=None, **kwargs):
        manifest = manifest or self.manifest()
        plan = self.prepare(manifest, **kwargs)
        br.admit(manifest, plan, ledger=self.ledger)
        snapshot = br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        return manifest, plan, snapshot

    def finish(self, manifest, plan, snapshot, output=(), errors=(), status="completed"):
        self.api.terminal(snapshot["id"], output, errors, status)
        br.poll(manifest, plan, 0, ledger=self.ledger, client=self.api)
        return br.retrieve(manifest, plan, 0, ledger=self.ledger, client=self.api)

    def test_prepare_is_pure_deterministic_model_grouped_and_exact(self):
        path = self.base / "image.png"
        Image.new("RGB", (12, 13), "white").save(path)
        jobs = [job("a"), job("image", model="gpt-5.6-luna",
                images=[rt.image_reference(path, reference="image.png")]), job("b")]
        manifest = self.manifest(jobs)
        plan = self.prepare(manifest, image_root=self.base)
        self.assertEqual(plan, self.prepare(manifest, image_root=self.base))
        self.assertEqual(rt.events(self.ledger), [])
        self.assertEqual(len(plan["chunks"]), 2)
        for chunk in plan["chunks"]:
            requests = [json.loads(line) for line in Path(chunk["path"]).read_text().splitlines()]
            self.assertEqual({r["body"]["model"] for r in requests}, {chunk["model"]})
            for request, reference in zip(requests, chunk["jobs"]):
                self.assertEqual(request["body"], rt.materialize(jobs[reference["index"]], self.base))
        br.verify_plan(manifest, plan, materialize=True)

    def test_chunk_limit_and_complete_denominator(self):
        manifest = self.manifest([job(str(i)) for i in range(5)])
        plan = self.prepare(manifest, max_chunk_bytes=1700)
        self.assertGreater(len(plan["chunks"]), 1)
        self.assertTrue(all(c["bytes"] <= 1700 for c in plan["chunks"]))
        br.admit(manifest, plan, ledger=self.ledger)
        self.assertEqual(rt.summarize(self.ledger)["admitted_jobs"], 5)
        self.assertEqual(rt.summarize(self.ledger)["unattempted_jobs"], 5)
        with self.assertRaises(ValueError):
            self.prepare(manifest, max_chunk_bytes=1)

    def test_attempts_and_creation_intent_are_durable_before_create(self):
        manifest = self.manifest([job("a"), job("b")])
        plan = self.prepare(manifest)
        br.admit(manifest, plan, ledger=self.ledger)
        def inspect():
            rows = rt.events(self.ledger)
            self.assertEqual(sum(r["event"] == "attempt" for r in rows), 2)
            self.assertEqual(sum(r["event"] == "batch_create_attempt" for r in rows), 1)
        self.api.before_create = inspect
        br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        self.assertEqual(self.api.uploads[0]["purpose"], "batch")
        self.assertEqual(self.api.creations[0]["completion_window"], "24h")
        self.assertEqual(self.api.creations[0]["metadata"]["chunk_sha256"], plan["chunks"][0]["sha256"])

    def test_unknown_creation_has_no_retry_and_keeps_full_reservation(self):
        self.api.create_failure = True
        manifest = self.manifest([job("a"), job("b")])
        plan = self.prepare(manifest)
        br.admit(manifest, plan, ledger=self.ledger)
        with self.assertRaises(TimeoutError):
            br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        with self.assertRaises(RuntimeError):
            br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        with self.assertRaises(RuntimeError):
            br.poll(manifest, plan, 0, ledger=self.ledger, client=self.api)
        self.assertEqual(len(self.api.creations), 1)
        summary = rt.summarize(self.ledger)
        self.assertEqual(summary["unresolved_attempts"], 2)
        self.assertEqual(summary["new_reserved_usd"], manifest["reserved_usd"])
        results = decision_adapters.collect_results(manifest, rt.events(self.ledger))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["status"] == "failed" for r in results.values()))

    def test_upload_unknown_cannot_repeat(self):
        self.api.upload_failure = True
        manifest = self.manifest()
        plan = self.prepare(manifest)
        br.admit(manifest, plan, ledger=self.ledger)
        with self.assertRaises(TimeoutError):
            br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        with self.assertRaises(RuntimeError):
            br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        self.assertEqual(len(self.api.uploads), 1)
        self.assertEqual(len(self.api.creations), 0)
        self.assertEqual(rt.summarize(self.ledger)["new_reserved_usd"], manifest["reserved_usd"])

    def test_concurrent_submission_creates_once(self):
        manifest = self.manifest()
        plan = self.prepare(manifest)
        br.admit(manifest, plan, ledger=self.ledger)
        def submit(_):
            try:
                br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
                return True
            except RuntimeError:
                return False
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(submit, range(4)))
        self.assertEqual(sum(results), 1)
        self.assertEqual(len(self.api.creations), 1)

    def test_reordered_results_join_custom_id_and_retrieve_is_idempotent(self):
        manifest, plan, snapshot = self.start(self.manifest([job("a"), job("b")]))
        ids = [j["custom_id"] for j in plan["chunks"][0]["jobs"]]
        results = self.finish(manifest, plan, snapshot,
            [response(ids[1], '{"name":"Second"}'), response(ids[0], '{"name":"First"}')])
        self.assertEqual({r["tag"]: r["parsed"]["name"] for r in results}, {"a": "First", "b": "Second"})
        with patch.object(br, "OpenAI", side_effect=AssertionError("Cached replay must not construct SDK client")):
            br.retrieve(manifest, plan, 0, ledger=self.ledger)
        self.assertEqual(len(self.api.downloads), 1)
        self.assertEqual(rt.summarize(self.ledger)["valid_calls"], 2)
        self.assertEqual(len(decision_adapters.collect_results(manifest, rt.events(self.ledger))), 2)

    def test_interrupted_result_ingestion_resumes_without_duplicate_or_download(self):
        manifest, plan, snapshot = self.start(self.manifest([job("a"), job("b")]))
        ids = [j["custom_id"] for j in plan["chunks"][0]["jobs"]]
        self.finish(manifest, plan, snapshot, [response(ids[0]), response(ids[1])])
        rows = rt.events(self.ledger)
        # Simulate process loss after the first durable per-job result.
        first = next(i for i, row in enumerate(rows) if row["event"] == "result")
        self.ledger.write_bytes(b"".join(br._bytes(row) for row in rows[:first+1]))
        with patch.object(br, "OpenAI", side_effect=AssertionError("Saved files must suffice")):
            results = br.retrieve(manifest, plan, 0, ledger=self.ledger)
        self.assertEqual(len(results), 2)
        self.assertEqual(rt.summarize(self.ledger)["valid_calls"], 2)
        self.assertEqual(len(self.api.downloads), 1)

    def test_partial_expiry_error_and_missing_retained(self):
        manifest, plan, snapshot = self.start(self.manifest([job("a"), job("b"), job("c")]))
        ids = [j["custom_id"] for j in plan["chunks"][0]["jobs"]]
        results = self.finish(manifest, plan, snapshot, [response(ids[0])],
            [{"custom_id": ids[1], "response": None, "error": {"code": "batch_expired"}}], "expired")
        self.assertEqual([r["status"] for r in results], ["valid", "failed", "failed"])
        self.assertEqual(results[2]["failure"], "missing_batch_result")
        summary = rt.summarize(self.ledger)
        self.assertEqual((summary["valid_calls"], summary["failed_calls"], summary["unresolved_attempts"]), (1, 2, 0))
        self.assertEqual(summary["new_reserved_usd"], manifest["reserved_usd"])

    def test_terminal_validation_failure_preserves_all_jobs(self):
        manifest, plan, snapshot = self.start(self.manifest([job("a"), job("b")]))
        results = self.finish(manifest, plan, snapshot, status="failed")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["failure"] == "missing_batch_result" for r in results))
        self.assertEqual(self.api.downloads, [])

    def test_invalid_schema_model_usage_reasoning_and_output_cap_stay_failed(self):
        jobs = [job(str(i)) for i in range(5)]
        manifest, plan, snapshot = self.start(self.manifest(jobs))
        ids = [j["custom_id"] for j in plan["chunks"][0]["jobs"]]
        output = [response(ids[0], '{"wrong":1}'), response(ids[1], model="other-model"),
                  response(ids[2], usage={"prompt_tokens": -1, "completion_tokens": 8}),
                  response(ids[3], usage={"prompt_tokens": 20, "completion_tokens": 8,
                           "completion_tokens_details": {"reasoning_tokens": 1}}),
                  response(ids[4], usage={"prompt_tokens": 20, "completion_tokens": 129})]
        results = self.finish(manifest, plan, snapshot, output)
        self.assertTrue(all(r["status"] == "failed" for r in results))
        self.assertTrue(all("response" in r for r in results))
        self.assertEqual(rt.summarize(self.ledger)["failed_calls"], 5)

    def test_nonfinite_model_content_fails_individually_and_cached_replay_succeeds(self):
        literals = ("NaN", "Infinity", "-Infinity", "1e309", "-1e309")
        schema = {"type": "object", "properties": {"quantity": {"type": "number"}},
                  "required": ["quantity"], "additionalProperties": False}
        jobs = [job(str(i), schema=schema) for i in range(len(literals) + 1)]
        manifest, plan, snapshot = self.start(self.manifest(jobs))
        ids = [j["custom_id"] for j in plan["chunks"][0]["jobs"]]
        output = [response(cid, '{"quantity":' + literal + '}')
                  for cid, literal in zip(ids, literals)]
        output.append(response(ids[-1], '{"quantity":2.5}'))
        results = self.finish(manifest, plan, snapshot, output)
        self.assertEqual([r["status"] for r in results], ["failed"] * len(literals) + ["valid"])
        self.assertTrue(all(r["failure"] == "invalid_batch_response" and "parsed" not in r
                            for r in results[:-1]))
        self.assertEqual(results[-1]["parsed"], {"quantity": 2.5})
        # Literal nonfinite strings in raw message content are safe evidence;
        # no nonfinite Python numeric value may reach the ledger.
        rt.canonical(rt.events(self.ledger))
        with patch.object(br, "OpenAI", side_effect=AssertionError("Cached replay must use no SDK")):
            replay = br.retrieve(manifest, plan, 0, ledger=self.ledger)
        self.assertEqual([r["status"] for r in replay], [r["status"] for r in results])
        self.assertEqual(len(self.api.downloads), 1)
        summary = rt.summarize(self.ledger)
        self.assertEqual((summary["valid_calls"], summary["failed_calls"], summary["unresolved_attempts"]),
                         (1, len(literals), 0))

    def test_nonfinite_outer_lines_preserve_raw_anomalies_and_valid_neighbors(self):
        literals = ("NaN", "Infinity", "-Infinity", "1e309", "-1e309")
        jobs = [job(str(i)) for i in range(len(literals) + 2)]
        manifest, plan, snapshot = self.start(self.manifest(jobs))
        ids = [j["custom_id"] for j in plan["chunks"][0]["jobs"]]
        self.api.terminal(snapshot["id"], [response(ids[-1])],
                          [{"custom_id": ids[-2], "response": None, "error": {"code": "fixture"}}])
        output_id = self.api.objects[snapshot["id"]]["output_file_id"]
        error_id = self.api.objects[snapshot["id"]]["error_file_id"]
        bad_lines = []
        for cid, literal in zip(ids, literals):
            entry = response(cid)
            entry["response"]["body"]["audit_numeric"] = 0
            bad_lines.append(br._bytes(entry).replace(b'"audit_numeric":0',
                              ('"audit_numeric":' + literal).encode()))
        raw_output = b"".join(bad_lines) + self.api.contents[output_id]
        raw_error = self.api.contents[error_id].replace(b'"code":"fixture"', b'"code":"fixture","detail":NaN')
        self.api.contents[output_id], self.api.contents[error_id] = raw_output, raw_error
        br.poll(manifest, plan, 0, ledger=self.ledger, client=self.api)
        results = br.retrieve(manifest, plan, 0, ledger=self.ledger, client=self.api)
        self.assertEqual([r["status"] for r in results], ["failed"] * (len(literals) + 1) + ["valid"])
        self.assertTrue(all(r["failure"] == "missing_batch_result" for r in results[:-1]))
        evidence = Path(plan["evidence_dir"]) / "chunk-0000"
        self.assertEqual((evidence / "output.jsonl").read_bytes(), raw_output)
        self.assertEqual((evidence / "error.jsonl").read_bytes(), raw_error)
        terminal = [r for r in rt.events(self.ledger) if r["event"] == "batch_retrieved"][0]
        self.assertEqual(len(terminal["anomalies"]), len(literals) + 1)
        self.assertTrue(terminal["reservation_exceeded"])
        self.assertEqual(terminal["anomalies"][0]["line_sha256"], br._hash(bad_lines[0].rstrip(b"\n")))
        rt.canonical(rt.events(self.ledger))
        with patch.object(br, "OpenAI", side_effect=AssertionError("Cached replay must use no SDK")):
            br.retrieve(manifest, plan, 0, ledger=self.ledger)
        self.assertEqual(len(self.api.downloads), 2)
        self.assertEqual(rt.summarize(self.ledger)["failed_calls"], len(literals) + 1)
        next_manifest = self.manifest([job("next")], name="after-anomaly")
        next_plan = self.prepare(next_manifest)
        br.admit(next_manifest, next_plan, ledger=self.ledger)
        with self.assertRaises(RuntimeError):
            br.submit(next_manifest, next_plan, 0, ledger=self.ledger, client=self.api)
        self.assertEqual(len(self.api.uploads), 1)

    def test_duplicate_unknown_and_corrupt_rows_do_not_select_favorable_response(self):
        manifest, plan, snapshot = self.start(self.manifest([job("a"), job("b")]))
        ids = [j["custom_id"] for j in plan["chunks"][0]["jobs"]]
        self.api.terminal(snapshot["id"], [response(ids[0]), response(ids[0]), response("unknown")])
        output_id = self.api.objects[snapshot["id"]]["output_file_id"]
        self.api.contents[output_id] += b"not json\n"
        br.poll(manifest, plan, 0, ledger=self.ledger, client=self.api)
        results = br.retrieve(manifest, plan, 0, ledger=self.ledger, client=self.api)
        self.assertEqual([r["failure"] for r in results], ["duplicate_custom_id", "missing_batch_result"])
        terminal = [r for r in rt.events(self.ledger) if r["event"] == "batch_retrieved"][0]
        self.assertEqual(len(terminal["anomalies"]), 2)
        self.assertTrue(terminal["reservation_exceeded"])

    def test_changed_source_input_image_and_plan_rejected_before_provider(self):
        manifest = self.manifest()
        plan = self.prepare(manifest)
        br.admit(manifest, plan, ledger=self.ledger)
        original = (self.root / "protocol.json").read_bytes()
        (self.root / "protocol.json").write_text("changed")
        with self.assertRaises(ValueError):
            br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        (self.root / "protocol.json").write_bytes(original)
        request_path = Path(plan["chunks"][0]["path"])
        original_request = request_path.read_bytes()
        request_path.write_bytes(original_request + b" ")
        with self.assertRaises(ValueError):
            br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        request_path.write_bytes(original_request)
        bad = copy.deepcopy(plan)
        bad["chunks"][0]["jobs"][0]["index"] = 99
        with self.assertRaises(ValueError):
            br.submit(manifest, bad, 0, ledger=self.ledger, client=self.api)
        self.assertEqual(self.api.uploads, [])
        image = self.base / "image.png"
        Image.new("RGB", (10, 10), "white").save(image)
        m2 = self.manifest([job("image", model="gpt-5.6-luna",
              images=[rt.image_reference(image, reference="image.png")])], name="image-batch")
        p2 = self.prepare(m2, image_root=self.base)
        br.admit(m2, p2, ledger=self.ledger)
        Image.new("RGB", (10, 10), "black").save(image)
        with self.assertRaises(ValueError):
            br.submit(m2, p2, 0, ledger=self.ledger, client=self.api)
        self.assertEqual(self.api.uploads, [])

    def test_total_budget_rejection_before_any_chunk_upload(self):
        manifest = self.manifest([job(str(i), user="x" * 5000, max_output_tokens=2048) for i in range(60)])
        plan = self.prepare(manifest, max_chunk_bytes=16000)
        self.assertGreater(len(plan["chunks"]), 1)
        with patch.object(rt, "DEVELOPMENT_CEILING", Decimal(manifest["reserved_usd"]) / 2):
            with self.assertRaises(RuntimeError):
                br.admit(manifest, plan, ledger=self.ledger)
        self.assertEqual(rt.events(self.ledger), [])
        with self.assertRaises(RuntimeError):
            br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        self.assertEqual(self.api.uploads, [])

    def test_discount_accounting_and_no_standard_fallback(self):
        manifest, plan, snapshot = self.start()
        standard = self.manifest(transport="standard")
        self.assertEqual(Decimal(manifest["reserved_usd"]) * 2, Decimal(standard["reserved_usd"]))
        cid = plan["chunks"][0]["jobs"][0]["custom_id"]
        result = self.finish(manifest, plan, snapshot, [response(cid)])[0]
        ri, ro = rt.PRICES["gpt-6-luna"]
        self.assertEqual(Decimal(result["estimated_token_price_usd"]), (20*ri + 8*ro) / 2_000_000)
        self.assertEqual(Decimal(result["usage_price_upper_usd"]), (20*ri*Decimal("1.25") + 8*ro) / 2_000_000)
        with self.assertRaises(ValueError):
            br.prepare(standard, plan_path=self.root/"standard.json", cache_dir=self.base/"cache")
        with self.assertRaises(ValueError):
            rt.call_job(manifest, 0, ledger=self.ledger, client=self.api)
        with self.assertRaises(ValueError):
            rt.run_batch(manifest, ledger=self.ledger)

    def test_batch_and_standard_share_global_reservation_cap(self):
        prior = self.manifest([job("prior")], name="standard", stage="evaluation", transport="standard")
        rt.admit(prior, ledger=self.ledger)
        manifest = self.manifest([job("new")], stage="evaluation")
        plan = self.prepare(manifest)
        ceiling = Decimal(prior["reserved_usd"]) + Decimal(manifest["reserved_usd"]) - Decimal("0.00000001")
        with patch.object(rt, "NEW_CEILING", ceiling):
            with self.assertRaises(RuntimeError):
                br.admit(manifest, plan, ledger=self.ledger)
        self.assertEqual(rt.summarize(self.ledger)["admitted_jobs"], 1)
        self.assertEqual(rt.summarize(self.ledger)["new_reserved_usd"], prior["reserved_usd"])
        self.assertEqual(self.api.uploads, [])

    def test_accounting_breach_stops_remaining_chunks(self):
        manifest = self.manifest([job("a"), job("b", model="gpt-5.6-luna")])
        plan = self.prepare(manifest)
        br.admit(manifest, plan, ledger=self.ledger)
        snapshot = br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        chunk = plan["chunks"][0]
        cid = chunk["jobs"][0]["custom_id"]
        self.finish(manifest, plan, snapshot, [response(cid, model=chunk["model"],
                    usage={"prompt_tokens": 1_000_000, "completion_tokens": 8})])
        with self.assertRaises(RuntimeError):
            br.submit(manifest, plan, 1, ledger=self.ledger, client=self.api)
        self.assertEqual(len(self.api.creations), 1)

    def test_clients_with_sdk_retries_are_rejected(self):
        manifest = self.manifest()
        plan = self.prepare(manifest)
        br.admit(manifest, plan, ledger=self.ledger)
        self.api.max_retries = 2
        with self.assertRaises(ValueError):
            br.submit(manifest, plan, 0, ledger=self.ledger, client=self.api)
        self.assertEqual(self.api.uploads, [])
        self.assertEqual(rt.summarize(self.ledger)["attempted_calls"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
