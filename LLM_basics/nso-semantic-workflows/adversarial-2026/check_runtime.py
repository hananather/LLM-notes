"""Offline financial/integrity checks. No provider requests are made."""
from concurrent.futures import ThreadPoolExecutor
import copy
from decimal import Decimal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import runtime as rt


def job(tag="case", **updates):
    row = {"tag": tag, "model": "gpt-6-luna", "system": "Extract observed text.",
           "user": {"name": "Example"}, "schema": {"type": "object",
           "properties": {"name": {"type": "string"}}, "required": ["name"],
           "additionalProperties": False}, "max_output_tokens": 128}
    return {**row, **updates}


class FakeResponse:
    model = "gpt-6-luna"
    usage = SimpleNamespace(prompt_tokens=20, completion_tokens=8,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=0))

    def __init__(self, text):
        self.choices = [SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=text))]

    def model_dump(self, **kwargs):
        return {"model": self.model, "usage": {"prompt_tokens": 20, "completion_tokens": 8},
                "choices": [{"message": {"content": self.choices[0].message.content}}]}


class FakeClient:
    def __init__(self, text='{"name":"Example"}', failure=False):
        self.text, self.failure, self.calls = text, failure, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        if self.failure:
            raise TimeoutError("deliberately not logged")
        return FakeResponse(self.text)


class RuntimeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger = self.root / "ledger.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, jobs=None, stage="development", name="batch"):
        return rt.make_manifest(name, stage, jobs or [job()], {"fixed_rule": "test fixture"})

    def test_complete_batch_rejection_does_not_reserve_prefix(self):
        many = [job(str(i), user="x" * 10000, max_output_tokens=2048) for i in range(500)]
        m = self.manifest(many)
        self.assertGreater(Decimal(m["reserved_usd"]), rt.DEVELOPMENT_CEILING)
        with self.assertRaises(RuntimeError):
            rt.admit(m, ledger=self.ledger)
        self.assertEqual(rt.events(self.ledger), [])

    def test_evaluation_cap_rejects_complete_batch(self):
        many = [job(str(i), user="x" * 30000, max_output_tokens=2048) for i in range(2000)]
        m = self.manifest(many, stage="evaluation")
        self.assertGreater(Decimal(m["reserved_usd"]), rt.NEW_CEILING)
        with self.assertRaises(RuntimeError):
            rt.admit(m, ledger=self.ledger)
        self.assertEqual(rt.events(self.ledger), [])

    def test_concurrent_admission_is_atomic(self):
        m = self.manifest()
        def admit(_):
            try:
                rt.admit(m, ledger=self.ledger)
                return True
            except ValueError:
                return False
        with ThreadPoolExecutor(8) as pool:
            accepted = list(pool.map(admit, range(20)))
        self.assertEqual(sum(accepted), 1)
        self.assertEqual(rt.summarize(self.ledger)["admitted_jobs"], 1)

    def test_modified_manifest_or_duplicate_tag_is_rejected(self):
        m = self.manifest()
        rt.admit(m, ledger=self.ledger)
        changed = copy.deepcopy(m)
        changed["jobs"][0]["user"] = "different evidence"
        with self.assertRaises(ValueError):
            rt.call_job(changed, 0, ledger=self.ledger, client=FakeClient())
        with self.assertRaises(ValueError):
            rt.admit(self.manifest(name="another"), ledger=self.ledger)

    def test_failure_retains_reservation_and_cannot_retry(self):
        m = self.manifest()
        rt.admit(m, ledger=self.ledger)
        client = FakeClient(failure=True)
        result = rt.call_job(m, 0, ledger=self.ledger, client=client)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "TimeoutError")
        with self.assertRaises(RuntimeError):
            rt.call_job(m, 0, ledger=self.ledger, client=client)
        summary = rt.summarize(self.ledger)
        self.assertEqual(client.calls, 1)
        self.assertEqual(summary["new_reserved_usd"], m["reserved_usd"])
        self.assertEqual(summary["failed_calls"], 1)

    def test_invalid_schema_output_stays_failed(self):
        m = self.manifest()
        rt.admit(m, ledger=self.ledger)
        result = rt.call_job(m, 0, ledger=self.ledger, client=FakeClient('{"wrong":1}'))
        self.assertEqual(result["status"], "failed")
        self.assertIn("response", result)

    def test_unknown_attempt_is_not_retried(self):
        m = self.manifest()
        rt.admit(m, ledger=self.ledger)
        with rt.locked_ledger(self.ledger) as (handle, _):
            rt.append_locked(handle, {"event": "attempt", "tag": "case"})
        self.assertEqual(rt.run_batch(m, ledger=self.ledger), [])
        self.assertEqual(rt.summarize(self.ledger)["unresolved_attempts"], 1)

    def test_changed_image_fails_before_attempt(self):
        path = self.root / "image.png"
        Image.new("RGB", (32, 32), "white").save(path)
        reference = rt.image_reference(path, reference="image.png")
        m = self.manifest([job(model="gpt-5.6-luna", images=[reference])])
        rt.admit(m, ledger=self.ledger)
        Image.new("RGB", (32, 32), "black").save(path)
        with self.assertRaises(ValueError):
            rt.call_job(m, 0, ledger=self.ledger, image_root=self.root, client=FakeClient())
        self.assertEqual(rt.summarize(self.ledger)["attempted_calls"], 0)

    def test_accounting_breach_stops_next_job(self):
        m = self.manifest([job("a"), job("b")])
        rt.admit(m, ledger=self.ledger)
        with rt.locked_ledger(self.ledger) as (handle, _):
            rt.append_locked(handle, {"event": "accounting_alert", "reservation_exceeded": True})
        client = FakeClient()
        with self.assertRaises(RuntimeError):
            rt.call_job(m, 1, ledger=self.ledger, client=client)
        self.assertEqual(client.calls, 0)

    def test_valid_call_records_reconstructable_evidence(self):
        m = self.manifest()
        rt.admit(m, ledger=self.ledger)
        result = rt.call_job(m, 0, ledger=self.ledger, client=FakeClient())
        self.assertEqual(result["parsed"], {"name": "Example"})
        self.assertEqual(rt.summarize(self.ledger)["valid_calls"], 1)
        attempt = next(r for r in rt.events(self.ledger) if r["event"] == "attempt")
        self.assertEqual(attempt["job_sha256"], rt.digest(m["jobs"][0]))

    def test_batch_discount_cannot_be_used_for_standard_calls(self):
        m = rt.make_manifest("batch-discount", "evaluation", [job()], {}, transport="batch")
        standard = self.manifest(stage="evaluation")
        self.assertEqual(Decimal(m["reserved_usd"]) * 2, Decimal(standard["reserved_usd"]))
        rt.admit(m, ledger=self.ledger)
        client = FakeClient()
        with self.assertRaises(ValueError):
            rt.call_job(m, 0, ledger=self.ledger, client=client)
        with self.assertRaises(ValueError):
            rt.run_batch(m, ledger=self.ledger)
        self.assertEqual(client.calls, 0)

    def test_forged_processing_discount_is_rejected(self):
        m = self.manifest()
        m["transport"] = "batch"
        with self.assertRaises(ValueError):
            rt.admit(m, ledger=self.ledger)


if __name__ == "__main__":
    unittest.main(verbosity=2)
