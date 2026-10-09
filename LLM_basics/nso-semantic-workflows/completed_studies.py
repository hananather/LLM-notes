"""Read verified October research results without fitting or model requests."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "results" / "research-completion-20261009.json"


def checked_study(name):
    """Bind notebook tables and figures to the final verification manifest."""
    manifest = json.loads(MANIFEST.read_text())
    if manifest["status"] != "verified":
        raise ValueError("Research completion has not been verified.")
    evidence = manifest["studies"][name]
    directory = ROOT / name
    for relative, expected in evidence["files_sha256"].items():
        actual = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Changed research evidence: {name}/{relative}")
    return directory


def clustering_results():
    directory = checked_study("clustering-study-20261009")
    result = json.loads((directory / "results.json").read_text())
    outcomes = pd.read_csv(directory / "results.csv")
    summary = pd.read_csv(directory / "summary.csv")
    teacher = json.loads((directory / "teacher-audit.json").read_text())
    ledger = pd.DataFrame(json.loads((directory / "label-ledger.json").read_text()))
    transfer = json.loads((directory / "febrl3-transfer/results.json").read_text())
    transfer_outcomes = pd.read_csv(directory / "febrl3-transfer/results.csv")
    if result["test_entities"] != 781 or result["test_records"] != 1562:
        raise ValueError("Unexpected FEBRL4 evaluation denominator.")
    if (teacher["valid_identity_decisions"] + teacher["insufficient_evidence"]
            + teacher["failed_calls"] != teacher["total_jobs"]):
        raise ValueError("Teacher outcome counts do not reconcile.")
    if not outcomes.loc[outcomes.fp_cost.eq(2) & outcomes.arm.eq("baseline"), "tp"].eq(779).all():
        raise ValueError("Baseline table differs from the verified result.")
    return {"directory": directory, "results": result, "outcomes": outcomes,
            "summary": summary, "teacher": teacher, "ledger": ledger, "transfer": transfer,
            "transfer_outcomes": transfer_outcomes}


def native_results():
    directory = checked_study("native-affiliation-study-20261009")
    adaptation_directory = checked_study("native-affiliation-adaptation-20261009")
    evaluation = json.loads((directory / "results/evaluation.json").read_text())
    adaptation = json.loads((adaptation_directory / "results.json").read_text())
    if evaluation["rows"] != 1104 or adaptation["test_rows"] != 697:
        raise ValueError("Unexpected native-affiliation evaluation denominator.")
    return {"directory": directory, "evaluation": evaluation,
            "adaptation_directory": adaptation_directory, "adaptation": adaptation}
