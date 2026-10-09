"""Independent count and provenance checks on sealed retrospective FS outputs."""
from pathlib import Path
import gzip
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


seal = read(HERE / "results/prediction-seal.json")
for name, expected in seal["artifact_sha256"].items():
    assert sha(HERE / name) == expected, name
for name, expected in seal["source_sha256"].items():
    assert sha(ROOT / name) == expected, name
report = read(HERE / "results/evaluation.json")
assert report["prediction_seal_sha256"] == sha(HERE / "results/prediction-seal.json")
assert report["prediction_sealed_at"] <= report["evaluated_at"]
assert not seal["test_labels_parsed_during_fit"]
with gzip.open(HERE / "results/case-outcomes.json.gz", "rt") as stream:
    rows = json.load(stream)
assert len(rows) == len({r["case_id"] for r in rows}) == 644
assert sha(HERE / "results/case-outcomes.json.gz") == report["case_outcomes_sha256"]
for method in report["scores"]:
    for view in ["valid_set", "automatic_resolved", "automatic_conditional"]:
        values = {"rows": 0, "exact_sets": 0, "tp": 0, "fp": 0, "fn": 0,
                  "automatic_rows": 0, "review_rows": 0, "invalid_rows": 0,
                  "nil_rows": 0, "nil_correct": 0, "false_nil_rows": 0,
                  "false_assignment_on_nil_rows": 0}
        for row in rows:
            state = row["methods"][method]
            if view == "automatic_conditional" and state["status"] != "automatic":
                continue
            truth = set(row["truth"])
            resolved = state["status"] != "invalid" if view == "valid_set" else state["status"] == "automatic"
            targets = set(state["targets"]) if resolved else set()
            values["rows"] += 1
            values["exact_sets"] += int(resolved and targets == truth)
            values["tp"] += len(targets & truth)
            values["fp"] += len(targets - truth)
            values["fn"] += len(truth - targets)
            values[f"{state['status']}_rows"] += 1
            values["nil_rows"] += int(not truth)
            values["nil_correct"] += int(resolved and not truth and not targets)
            values["false_nil_rows"] += int(resolved and bool(truth) and not targets)
            values["false_assignment_on_nil_rows"] += int(not truth and bool(targets))
        for key, actual in values.items():
            assert report["scores"][method][view][key] == actual, (method, view, key)
for name in ["compact", "expanded"]:
    parameters = read(HERE / f"results/{name}-training.json")
    n, positive = parameters["training_pairs"], parameters["training_positive_pairs"]
    assert n == 1127 * 25
    assert parameters["prior"] == (positive + 1) / (n + 2)
    for comparison in parameters["comparisons"].values():
        assert sum(comparison["positive_counts"]) == positive
        assert sum(comparison["negative_counts"]) == n - positive
        levels = len(comparison["m"])
        for index in range(levels):
            assert comparison["m"][index] == (comparison["positive_counts"][index] + 1) / (positive + levels)
            assert comparison["u"][index] == (comparison["negative_counts"][index] + 1) / (n - positive + levels)
selection = read(HERE / "results/validation-selection.json")
assert max(selection["native_max_absolute_difference"].values()) < 1e-10
out = {"all_original_source_hashes_unchanged": True, "sealed_prediction_artifacts": len(seal["artifact_sha256"]),
       "independently_recomputed_rows": len(rows), "methods": len(report["scores"]),
       "m_u_class_counts_verified": True, "native_arithmetic_agreement_max": max(selection["native_max_absolute_difference"].values()),
       "paid_calls": 0}
payload = json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n"
path = HERE / "results/independent-verification.json"
if path.exists():
    assert path.read_text() == payload
else:
    path.write_text(payload)
print(payload)
