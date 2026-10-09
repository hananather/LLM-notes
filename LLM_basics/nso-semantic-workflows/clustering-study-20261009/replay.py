"""Load scientific result tables without fitting or model inference."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_report():
    return {
        "results": json.loads((HERE / "results.json").read_text()),
        "outcomes": pd.read_csv(HERE / "results.csv"),
        "summary": pd.read_csv(HERE / "summary.csv"),
        "teacher": json.loads((HERE / "teacher-audit.json").read_text()),
        "ledger": pd.DataFrame(json.loads((HERE / "label-ledger.json").read_text())),
        "febrl3": json.loads((HERE / "febrl3-transfer/results.json").read_text()),
        "febrl3_outcomes": pd.read_csv(HERE / "febrl3-transfer/results.csv"),
    }
