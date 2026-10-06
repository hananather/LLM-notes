"""Small descriptive summaries for paired, clustered pilot experiments."""
from __future__ import annotations

import math
import random


def wilson(successes: int, trials: int, z: float = 1.959963984540054):
    """Wilson interval for an independent Bernoulli sample, when appropriate.

    Do not use this interval for related synthetic variants or as a population
    interval for a convenience sample. A census of a fixed test has no sampling
    uncertainty about that test's measured score.
    """
    if not 0 <= successes <= trials:
        raise ValueError("Require 0 <= successes <= trials.")
    if trials == 0:
        return None
    p = successes / trials
    divisor = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / divisor
    width = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / divisor
    return [max(0.0, center - width), min(1.0, center + width)]


def paired_summary(rows, baseline_key="baseline_correct", model_key="model_correct", family_key="family"):
    """Describe net corrections and variability across observed task families.

    The bootstrap resamples complete families and weights each sampled family
    equally. It is a sensitivity summary of these authored families, not a
    confidence interval for national workflows or all future model responses.
    """
    if not rows:
        return {"n": 0}
    groups = {}
    for row in rows:
        if type(row[baseline_key]) is not bool or type(row[model_key]) is not bool:
            raise TypeError("Correctness fields must be Boolean.")
        groups.setdefault(row[family_key], []).append(int(row[model_key]) - int(row[baseline_key]))
    changes = [sum(v) / len(v) for v in groups.values()]
    rng = random.Random(20261006)
    samples = sorted(sum(rng.choices(changes, k=len(changes))) / len(changes) for _ in range(10000))
    return {"n": len(rows), "families": len(groups),
            "baseline_correct": sum(r[baseline_key] for r in rows),
            "model_correct": sum(r[model_key] for r in rows),
            "corrected": sum(not r[baseline_key] and r[model_key] for r in rows),
            "regressed": sum(r[baseline_key] and not r[model_key] for r in rows),
            "family_mean_difference": sum(changes) / len(changes),
            "family_bootstrap_95_percentile": [samples[249], samples[9749]],
            "interpretation": "Descriptive resampling of observed families; no national-population coverage claim."}
