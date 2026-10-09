"""Reconstruct and draw every measured false-merge component in FEBRL3."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent


def components(nso_root):
    folder = Path(nso_root) / "clustering-study-20261009" / "febrl3-transfer"
    source = folder / "source-dataset3.csv"
    predictions = folder / "sealed-predictions.parquet"
    records = pd.read_csv(source, dtype=str)
    identity = {
        hashlib.sha256(("febrl3-transfer:" + row).encode()).hexdigest()[:24]:
        re.match(r"rec-(\d+)-", row).group(1)
        for row in records.rec_id
    }
    pairs = pd.read_parquet(predictions)
    accepted = pairs.loc[pairs.teacher_student_c2]
    parent = {node: node for node in set(pairs.unique_id_l) | set(pairs.unique_id_r)}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for row in accepted.itertuples():
        parent[find(row.unique_id_l)] = find(row.unique_id_r)
    groups = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    output = []
    for nodes in groups.values():
        counts = Counter(identity[node] for node in nodes)
        if len(counts) <= 1:
            continue
        sizes = list(counts.values())
        edges = [
            {"left": row.unique_id_l, "right": row.unique_id_r,
             "false": identity[row.unique_id_l] != identity[row.unique_id_r]}
            for row in accepted.loc[
                accepted.unique_id_l.isin(nodes) & accepted.unique_id_r.isin(nodes)
            ].itertuples()
        ]
        output.append({
            "entity_members": sorted(
                [sorted(node for node in nodes if identity[node] == entity)
                 for entity in counts], key=lambda members: (-len(members), members)),
            "false_relationships": sum(a * b for i, a in enumerate(sizes)
                                       for b in sizes[i + 1:]),
            "edges": edges,
        })
    output.sort(key=lambda row: -row["false_relationships"])
    assert len(output) == 3 and sum(row["false_relationships"] for row in output) == 27
    assert sum(edge["false"] for row in output for edge in row["edges"]) == 3
    return {
        "arm": "teacher_student_c2", "test_records": 733, "test_entities": 296,
        "selection": "Every false-merge component after evaluation; no omitted false merges",
        "input_sha256": {str(p.relative_to(Path(nso_root))): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [source, predictions]},
        "components": output,
    }


def figure(nso_root):
    evidence = components(nso_root)
    ink, muted, false = "#1E3342", "#536777", "#BC6254"
    colors = ["#276C9A", "#267A63"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": ink, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.4), dpi=150)
        fig.subplots_adjust(left=.03, right=.99, top=.73, bottom=.32, wspace=.19)
        fig.text(.03, .95, "Measured false bridges: 20 + 6 + 1 wrong relationships", fontsize=20, weight="bold")
        fig.text(.03, .87, "All three false-merge components from the LLM-trained matcher on held-out FEBRL3", fontsize=12, color=muted)
        for number, (ax, component) in enumerate(zip(axes, evidence["components"]), 1):
            locations = {}
            for side, members in enumerate(component["entity_members"]):
                center = np.array([-1.05 if side == 0 else 1.05, 0.0])
                angles = np.linspace(0, 2 * np.pi, len(members), endpoint=False) + np.pi / 2
                positions = center + .58 * np.column_stack([np.cos(angles), np.sin(angles)])
                if len(members) == 1:
                    positions = np.array([center])
                for node, position in zip(members, positions):
                    locations[node] = position
            for edge in component["edges"]:
                a, b = locations[edge["left"]], locations[edge["right"]]
                ax.plot([a[0], b[0]], [a[1], b[1]],
                        color=false if edge["false"] else "#80949F",
                        linewidth=2.5 if edge["false"] else 1.1,
                        linestyle="--" if edge["false"] else "-", zorder=1)
            for side, members in enumerate(component["entity_members"]):
                for i, node in enumerate(members, 1):
                    x, y = locations[node]
                    ax.scatter(x, y, s=450, color=colors[side], edgecolors="white", linewidth=1.5, zorder=2)
                    ax.text(x, y, f"{'A' if side == 0 else 'B'}{i}",
                            color="white", fontsize=9, ha="center", va="center", weight="bold", zorder=3)
            sizes = [len(members) for members in component["entity_members"]]
            ax.set_title(f"Component {number}: {sizes[0]} + {sizes[1]} records", loc="left", pad=18,
                         color=ink, weight="bold", fontsize=13)
            ax.text(0, -1.05, f"1 false edge → {sizes[0]} × {sizes[1]} = {component['false_relationships']} false pairs",
                    ha="center", weight="bold", fontsize=11, color=false)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-1.4, 1.1)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.legend(handles=[
            Line2D([], [], marker="o", linestyle="", color=colors[0], label="Reference entity A"),
            Line2D([], [], marker="o", linestyle="", color=colors[1], label="Reference entity B"),
            Line2D([], [], color=false, linestyle="--", linewidth=2.5, label="False accepted bridge"),
        ], loc="lower left", bbox_to_anchor=(.03, .18), ncol=3, frameon=False)
        fig.text(.03, .12, "Nodes and solid edges are reconstructed from the saved predictions; record labels are relabeled within each panel.", color=muted, fontsize=10)
        fig.text(.03, .065, "Connected-component closure asserts every cross-entity pair. The three panels contain all 27 false relationships in the full test.", color=muted, fontsize=10)
    return fig, evidence


if __name__ == "__main__":
    plot, evidence = figure(HERE.parent / "nso-semantic-workflows")
    for extension in ["png", "svg", "pdf"]:
        plot.savefig(HERE / "diagrams" / f"16-measured-cluster-bridges.{extension}")
    path = HERE / "results" / "measured-febrl3-bridges-20261009.json"
    if path.exists():
        assert json.loads(path.read_text()) == evidence
    else:
        path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
