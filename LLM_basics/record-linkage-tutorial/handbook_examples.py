"""Small, deterministic calculations used by the record-linkage notebook."""

from itertools import combinations
import json
from math import comb
from pathlib import Path

from IPython.display import Image, display
import numpy as np
import pandas as pd


def diagram(name, width=900):
    """Display a compiled TikZ figure; no TeX installation is needed for replay."""
    path = Path(__file__).parent / "diagrams" / f"{name}.png"
    display(Image(filename=str(path), width=width))


def evidence_example(prior=0.0001):
    """Evaluate two comparison vectors under stated illustrative m/u values."""
    levels = pd.DataFrame(
        [
            ("Name agreement", 0.85, 0.01),
            ("Birth-date agreement", 0.90, 0.002),
            ("Birth-date contradiction", 0.02, 0.90),
            ("Postcode agreement", 0.60, 0.03),
        ], columns=["comparison outcome", "m", "u"],
    ).set_index("comparison outcome")
    levels["likelihood ratio"] = levels.m / levels.u
    levels["weight (bits)"] = np.log2(levels["likelihood ratio"])
    prior_odds = prior / (1 - prior)
    rows = []
    for name, birth in [("Three agreements", "Birth-date agreement"),
                        ("Birth date contradicts", "Birth-date contradiction")]:
        selected = levels.loc[["Name agreement", birth, "Postcode agreement"]]
        odds = prior_odds * selected["likelihood ratio"].prod()
        rows.append({"comparison pattern": name, "prior probability": prior,
                     "sum of weights (bits)": selected["weight (bits)"].sum(),
                     "posterior probability": odds / (1 + odds)})
    return levels, pd.DataFrame(rows).set_index("comparison pattern")


def duplicated_evidence_example(prior=0.01, likelihood_ratio=9):
    """Contrast one signal with the incorrect product of its exact duplicate."""
    odds = prior / (1 - prior)
    one = odds * likelihood_ratio
    duplicate = one * likelihood_ratio
    return pd.DataFrame({
        "calculation": ["Signal counted once", "Exact copy incorrectly counted independently"],
        "posterior probability": [one / (1 + one), duplicate / (1 + duplicate)],
    }).set_index("calculation")


def reference_example():
    """Construct perfect reference agreement with imperfect identity decisions."""
    pairs = pd.DataFrame({"pair": ["a", "b", "c", "d"],
                          "entity truth": [True, False, True, False],
                          "reference decision": [True, True, False, False]})
    pairs["reference copy"] = pairs["reference decision"]
    scores = pd.Series({
        "Agreement with reference": pairs["reference copy"].eq(pairs["reference decision"]).mean(),
        "Accuracy against entity truth": pairs["reference copy"].eq(pairs["entity truth"]).mean(),
    }, name="fraction")
    return pairs.set_index("pair"), scores


def controlled_prose(record):
    """Render a known grammar and recover it exactly; this is not free-text extraction."""
    parts = [f'The {field.replace("_", " ")} is {json.dumps(value)}.'
             for field, value in record.items()]
    text = " ".join(parts)
    remaining = text
    recovered = {}
    decoder = json.JSONDecoder()
    for field in record:
        prefix = f'The {field.replace("_", " ")} is '
        if not remaining.startswith(prefix):
            raise ValueError("Controlled grammar changed.")
        value, end = decoder.raw_decode(remaining[len(prefix):])
        remaining = remaining[len(prefix) + end:]
        if not remaining.startswith("."):
            raise ValueError("Missing sentence boundary.")
        remaining = remaining[1:].lstrip()
        recovered[field] = value
    if remaining or record != recovered:
        raise ValueError("The representation did not preserve all fields.")
    return text, recovered


def repair_benefit_example():
    """Calculate benefits from explicitly constructed predictions, not fitted repairs."""
    cases = pd.DataFrame({
        "case": ["A", "B", "C", "D"],
        "truth": [True, True, True, False],
        "baseline": [False, False, False, False],
        "repair 1": [True, True, False, True],
        "repair 2": [False, True, True, False],
    }).set_index("case")
    loss_before = cases.baseline.ne(cases.truth).astype(int)
    benefit = pd.DataFrame(index=cases.index)
    for repair in ["repair 1", "repair 2"]:
        benefit[repair] = loss_before - cases[repair].ne(cases.truth).astype(int)
    return cases, benefit


def bridge_example(size_a=50, size_b=50):
    """Compute and independently enumerate the damage from one false bridge."""
    within = comb(size_a, 2) + comb(size_b, 2)
    after = comb(size_a + size_b, 2)
    # Enumeration verifies the pair denominator without relying on the closed form.
    labels = [0] * size_a + [1] * size_b
    false_pairs = sum(labels[i] != labels[j]
                      for i, j in combinations(range(len(labels)), 2))
    assert false_pairs == size_a * size_b == after - within
    return pd.DataFrame([
        {"stage": "Accepted edges", "predicted matching pairs": within + 1,
         "false matching pairs": 1, "precision": within / (within + 1), "recall": 1.0},
        {"stage": "After connected components", "predicted matching pairs": after,
         "false matching pairs": false_pairs, "precision": within / after, "recall": 1.0},
    ]).set_index("stage")


def scale_example(rows_per_source=100_000_000, candidates_per_left=20,
                  dimensions=384, teacher_queries=5_000):
    """Return workload arithmetic, without a hardware or throughput forecast."""
    quantities = [
        ("Cross-file pairs", rows_per_source**2, "pairs"),
        ("Retained candidate pairs", rows_per_source * candidates_per_left, "pairs"),
        ("Dense affinity (float32)", 4 * rows_per_source**2, "bytes"),
        ("Both files' vectors (float16)", 2 * rows_per_source * dimensions * 2, "bytes"),
        ("Candidate edge payload (24 bytes/edge)", rows_per_source * candidates_per_left * 24, "bytes"),
        ("32 float32 diagnostics per candidate", rows_per_source * candidates_per_left * 32 * 4, "bytes"),
        ("Direct review of 0.01% of candidates", rows_per_source * candidates_per_left // 10_000, "pairs"),
        ("All-record extraction input (500 tokens/record)", 2 * rows_per_source * 500, "tokens"),
        ("All-record extraction output (50 tokens/record)", 2 * rows_per_source * 50, "tokens"),
        ("Teacher input (1,200 tokens/query)", teacher_queries * 1_200, "tokens"),
        ("Teacher output (20 tokens/query)", teacher_queries * 20, "tokens"),
    ]
    return pd.DataFrame(quantities, columns=["quantity", "value", "unit"]).set_index("quantity")


def recall_example(true_links=100, retained_true_links=80, accepted_true_links=60):
    """Factor recall for final pair decisions restricted to the candidate set."""
    assert 0 < accepted_true_links <= retained_true_links <= true_links
    candidate = retained_true_links / true_links
    conditional = accepted_true_links / retained_true_links
    final = accepted_true_links / true_links
    assert np.isclose(candidate * conditional, final)
    return pd.DataFrame({
        "measure": ["Candidate recall", "Recall conditional on candidacy", "End-to-end recall"],
        "numerator": [retained_true_links, accepted_true_links, accepted_true_links],
        "denominator": [true_links, retained_true_links, true_links],
        "recall": [candidate, conditional, final],
    }).set_index("measure")
