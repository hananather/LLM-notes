"""Offline checks for wrong-link, abstention and failed-call accounting."""
import unittest
from decision_adapters import product_decisions, image_decisions
from evaluation import outcomes, summarize, development_gate


class ScoringChecks(unittest.TestCase):
    def test_missing_and_malformed_are_never_correct_nil(self):
        for value in [None, [], "nil", 7, {"status": "linked", "target_ids": []}]:
            row = outcomes({"q": []}, {"q": value})[0]
            self.assertEqual(row["status"], "failed")
            self.assertFalse(row["correct"])
        self.assertFalse(outcomes({"q": []}, {})[0]["correct"])

    def test_review_is_not_nil(self):
        row = outcomes({"q": []}, {"q": {"status": "review", "target_ids": []}})[0]
        self.assertFalse(row["correct"])
        self.assertFalse(row["automatic"])

    def test_failed_product_source_cannot_make_automatic_decision(self):
        rows = [{"record_id": "left", "target_ids": ["right"]}]
        for failures in [["left"], ["right"]]:
            p = product_decisions(rows, failed_extraction_ids=failures)
            self.assertEqual(p["left"]["status"], "failed")
            self.assertFalse(outcomes({"left": ["right"]}, p)[0]["correct"])

    def test_failed_fit_cannot_make_correct_nil(self):
        p = product_decisions([{"record_id": "q", "target_ids": []}], fit_valid=False)
        self.assertFalse(outcomes({"q": []}, p)[0]["correct"])
        p = image_decisions([{"query_id": "q", "decision_status": "nil", "prediction": None, "fit_valid": False}])
        self.assertFalse(outcomes({"q": []}, p)[0]["correct"])

    def test_complete_set_not_partial_match(self):
        p = {"q": {"status": "linked", "target_ids": ["a", "x"]}}
        s = summarize(outcomes({"q": ["a", "b"]}, p))
        self.assertEqual(s["correct_complete_decisions"], 0)
        self.assertEqual(s["false_links"], 1)
        self.assertEqual(s["missed_links"], 1)

    def test_gate_uses_frozen_operational_set_and_exact_queries(self):
        good = summarize(outcomes({"a": []}, {"a": {"status": "nil", "target_ids": []}}))
        bad = summarize(outcomes({"a": []}, {}))
        gate = development_gate({"operational": bad, "diagnostic": good}, good, eligible_methods=["operational"])
        self.assertEqual(gate["reference"], "operational")
        wrong_ids = summarize(outcomes({"b": []}, {}))
        with self.assertRaises(ValueError):
            development_gate({"operational": wrong_ids}, good, eligible_methods=["operational"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
