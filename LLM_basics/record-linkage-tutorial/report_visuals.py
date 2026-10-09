"""Plot saved comparison outcomes without inference or evidence-file writes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


INK = "#1E3342"
MUTED = "#536777"
GRID = "#DFE5EA"
CORRECT = "#267A63"
REVIEW_CORRECT = "#B1DECE"
WRONG = "#BC6254"
UNRESOLVED = "#CED7E1"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _checked_hash(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Saved figure input differs from its recorded hash: {path.name}")


def _clean_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for side in ["left", "top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0)


def affiliation_outcomes(nso_root: str | Path):
    """Return a figure separating annotated-set agreement from automatic decisions.

    Counts and paired intervals come from the frozen evaluation and are checked
    against its independent reconstruction. Correct review-flagged proposals
    count in annotation agreement, but stay unresolved in the automatic policy.
    """
    root = Path(nso_root)
    directory = root / "affiliation-value" / "results"
    evaluation = _read(directory / "evaluation.json")
    verification = _read(directory / "independent-verification.json")
    _checked_hash(directory / "evaluation.json", verification["input_sha256"][
        "affiliation-value/results/evaluation.json"])
    if verification["status"] != "independently_verified":
        raise ValueError("Affiliation inputs lack a completed independent verification.")
    if evaluation["scores"] != verification["full_644"]["scores"]:
        raise ValueError("Affiliation scores disagree with independent reconstruction.")
    if evaluation["group_sizes"] != verification["full_644"]["group_sizes"]:
        raise ValueError("Affiliation denominators disagree with independent reconstruction.")

    model = evaluation["primary_model"]
    if model != "hist_gradient_boosting":
        raise ValueError("This figure describes the saved validation-selected tree.")
    methods = [model, "saved_llm"]
    names = ["Trained tree", "GPT-6 Luna"]
    total = evaluation["group_sizes"]["all"]
    automatic_correct, reviewed_correct, automatic_wrong, unresolved = [], [], [], []
    for method in methods:
        scores = evaluation["scores"][method]["all"]
        auto = scores["automatic_resolved"]
        valid = scores["valid_set"]
        if auto["rows"] != total or valid["rows"] != total:
            raise ValueError("Affiliation views must retain every input row.")
        correct = auto["exact_sets"]
        wrong = auto["automatic_rows"] - correct
        pending = auto["review_rows"] + auto["invalid_rows"]
        reviewed = valid["exact_sets"] - correct
        if correct + wrong + pending != total or not 0 <= reviewed <= auto["review_rows"]:
            raise ValueError("Affiliation review and automatic counts do not reconcile.")
        automatic_correct.append(correct)
        reviewed_correct.append(reviewed)
        automatic_wrong.append(wrong)
        unresolved.append(pending)

    contrast = verification["full_644"]["paired_comparisons"][f"saved_llm_minus_{model}"]
    exact_difference = contrast["exact_sets_difference_per_input"]
    automatic_difference = contrast["automatic_correct_difference_per_input"]
    group_count = verification["full_644"]["grouping"]["components"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), dpi=130)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.75, bottom=0.43, wspace=0.43)
        fig.text(0.025, 0.95, "Affiliation selection: gains and review burden", fontsize=19, weight="bold")
        fig.text(0.025, 0.89, f"Two views of the same {total:,} historical S2AFF inputs and 25-candidate lists", color=MUTED)
        y = np.arange(len(names))
        for ax in axes:
            _clean_axis(ax)
            ax.set_xlim(0, total)
            ax.set_xticks([0, 200, 400, total])
            ax.set_yticks(y, names)
            ax.invert_yaxis()
            ax.set_ylim(1.5, -0.6)
            ax.set_xlabel(f"Inputs out of {total:,}", labelpad=10)

        left, right = axes
        left.set_title("Exact annotated sets", loc="left", fontsize=13, weight="bold", pad=12)
        left.barh(y, automatic_correct, color=CORRECT, height=0.48)
        left.barh(y, reviewed_correct, left=automatic_correct, color=REVIEW_CORRECT,
                  hatch="///", edgecolor=CORRECT, linewidth=0.7, height=0.48)
        for row, (correct, reviewed) in enumerate(zip(automatic_correct, reviewed_correct)):
            left.text(correct / 2, row, f"{correct:,}", ha="center", va="center", color="white", weight="bold")
            if reviewed:
                left.text(correct + reviewed / 2, row, str(reviewed), ha="center", va="center", fontsize=10, weight="bold")
                left.text(correct + reviewed, row - 0.39, f"Total {correct + reviewed:,}", ha="right", fontsize=10, weight="bold")

        right.set_title("Automatic decision policy", loc="left", fontsize=13, weight="bold", pad=12)
        offset = np.zeros(len(methods), dtype=int)
        for counts, color in [(automatic_correct, CORRECT), (automatic_wrong, WRONG), (unresolved, UNRESOLVED)]:
            right.barh(y, counts, left=offset, color=color, height=0.48, edgecolor="white", linewidth=0.8)
            for row, count in enumerate(counts):
                if count:
                    right.text(offset[row] + count / 2, row, str(count), ha="center", va="center",
                               color=INK if color == UNRESOLVED else "white", fontsize=10, weight="bold")
            offset += np.asarray(counts)

        fig.legend(handles=[
            Patch(facecolor=CORRECT, label="Correct automatic decision"),
            Patch(facecolor=REVIEW_CORRECT, edgecolor=CORRECT, hatch="///", label="Correct proposal flagged for review"),
            Patch(facecolor=WRONG, label="Wrong automatic decision"),
            Patch(facecolor=UNRESOLVED, label="Unresolved: review or invalid"),
        ], loc="lower left", bbox_to_anchor=(0.025, 0.21), ncol=2, frameon=False, fontsize=10,
            columnspacing=2.4, handlelength=1.7)

        def interval_text(metric):
            lo, hi = metric["paired_cluster_percentile_95"]
            return f"+{100 * metric['point']:.1f} points (95% interval {100 * lo:.1f}–{100 * hi:.1f})"

        fig.text(0.025, 0.16, f"LLM gain over tree: annotated sets {interval_text(exact_difference)};", fontsize=10)
        fig.text(0.025, 0.115, f"correct automatic decisions {interval_text(automatic_difference)}.", fontsize=10)
        fig.text(0.025, 0.06, f"Paired bootstrap over {group_count} source groups; fixed models. Inspected historical test, not an NSO population estimate.", fontsize=9.5, color=MUTED)
    return fig


def product_tradeoff(nso_root: str | Path):
    """Return four saved product outcomes, keeping query and pair units distinct."""
    root = Path(nso_root)
    directory = root / "adversarial-2026" / "products" / "results" / "model-evaluation-v1"
    completion = _read(directory / "completion.json")
    for name in ["primary-summary.json", "common-query-outcomes.json"]:
        _checked_hash(directory / name, completion["files"][name])
    summary = _read(directory / "primary-summary.json")
    outcomes = _read(directory / "common-query-outcomes.json")
    if summary["results"] != completion["primary_summary"]:
        raise ValueError("Product scores disagree with the saved completion record.")
    methods = [
        ("Lexical", "lexical_all_pairs", "lexical/threshold=0.65/all_pairs"),
        ("Splink", "splink_all_pairs", "splink/assumed_recall=0.8;threshold=0.9/all_pairs"),
        ("GPT-6 Luna", "semantic_selection", "semantic_selection"),
    ]
    total = summary["queries"]
    if len(outcomes) != total:
        raise ValueError("Product outcomes do not retain the full query denominator.")
    exact, false_links, missed_links, correct_nil = [], [], [], []
    for _, key, policy in methods:
        records = [row["methods"][policy]["outcome"] for row in outcomes]
        score = summary["results"][key]
        exact_count = sum(row["correct"] for row in records)
        false_count = sum(row["false_links"] for row in records)
        missed_count = sum(row["missed_links"] for row in records)
        nil_count = sum(row["true_nil"] and row["correct"] for row in records)
        if (exact_count, false_count, missed_count) != (
                score["correct_complete_decisions"], score["false_links"], score["missed_links"]):
            raise ValueError("Product plotted counts disagree with per-query outcomes.")
        if sum(row["true_nil"] for row in records) != score["true_nil_queries"]:
            raise ValueError("Product no-target denominators disagree.")
        exact.append(exact_count)
        false_links.append(false_count)
        missed_links.append(missed_count)
        correct_nil.append(nil_count)
    nil_total = summary["results"]["semantic_selection"]["true_nil_queries"]
    semantic = summary["results"]["semantic_selection"]
    colors = ["#7A8590", "#567AA3", "#BC6254"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=130)
        fig.subplots_adjust(left=0.13, right=0.97, top=0.77, bottom=0.18, wspace=0.45, hspace=0.68)
        fig.text(0.025, 0.95, "Products: more recovery, weaker rejection", fontsize=19, weight="bold")
        fig.text(0.025, 0.895, f"Amazon–Google · {total:,} queries · unchanged primary policies", color=MUTED)
        panels = [
            (axes[0, 0], exact, f"Complete correct decisions / {total:,}", "Queries · higher is better", total),
            (axes[0, 1], correct_nil, f"Correct no-target decisions / {nil_total:,}", "Queries · higher is better", nil_total),
            (axes[1, 0], false_links, "False links", "Record pairs · lower is better", max(false_links) * 1.15),
            (axes[1, 1], missed_links, "Missed reference links", "Record pairs · lower is better", max(missed_links) * 1.15),
        ]
        names = [name for name, _, _ in methods]
        for ax, counts, title, xlabel, limit in panels:
            _clean_axis(ax)
            y = np.arange(len(names))
            ax.barh(y, counts, color=colors, height=0.57)
            ax.set_yticks(y, names)
            ax.invert_yaxis()
            ax.set_xlim(0, limit)
            ax.set_title(title, loc="left", fontsize=12.5, weight="bold", pad=11)
            ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
            for row, count in enumerate(counts):
                ax.text(count + limit * 0.016, row, f"{count:,}", va="center", fontsize=11, weight="bold")
        fig.text(0.025, 0.075, f"LLM: {semantic['review_queries']} reviews and {semantic['failed_queries']} invalid responses stay in all {total:,} inputs.", fontsize=10)
        fig.text(0.025, 0.035, "No target means absent from the publisher mapping. Query decisions and pair errors have different units.", fontsize=9.5, color=MUTED)
    return fig
