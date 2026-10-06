"""Render the experiment workflow as a reproducible vector figure."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent


def workflow():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set(xlim=(0, 12), ylim=(0, 6.5))
    ax.axis("off")
    ink, blue, teal = "#182d40", "#eaf1fa", "#e5f3ef"

    def box(x, y, text, color=blue, width=2.3, height=1.0):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                   boxstyle="round,pad=0.08,rounding_size=0.10",
                                   linewidth=1.2, edgecolor="#7890a4", facecolor=color))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                color=ink, fontsize=11.5, linespacing=1.45)

    def arrow(start, end, **kwargs):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.6, color="#536a7e", **kwargs))

    ax.text(.15, 6.08, "Where language interpretation enters the workflow", fontsize=19,
            color=ink, fontweight="bold")
    ax.text(.15, 5.48, "A   Classification and affiliation linkage", fontsize=13,
            color=ink, fontweight="bold")
    xs = [.2, 3.25, 6.3, 9.35]
    texts = ["Description or\naffiliation text", "Full reference index\n→ candidate shortlist",
             "Compare rules with\nmodel selection", "Code, organization set\nor review decision"]
    for x, text in zip(xs, texts):
        box(x, 3.98, text, teal if x == xs[-1] else blue)
    for x in xs[:-1]:
        arrow((x + 2.4, 4.48), (x + 2.92, 4.48))
    ax.text(.2, 3.52, "The same source universe supports retrieval. Missing candidates remain end-to-end errors.",
            color="#435d72", fontsize=11)

    ax.text(.15, 2.95, "B   Questions about published statistics", fontsize=13,
            color=ink, fontweight="bold")
    texts = ["Analyst request\n+ source metadata", "Rules or model\nproduce a query plan",
             "Frozen observations\n+ exact calculator", "Value, unit, sources\nor review decision"]
    for x, text in zip(xs, texts):
        box(x, 1.48, text, teal if x == xs[-1] else blue)
    for x in xs[:-1]:
        arrow((x + 2.4, 1.98), (x + 2.92, 1.98))
    arrow((7.45, 1.35), (4.4, 1.35), connectionstyle="arc3,rad=-0.25")
    ax.text(5.92, .47, "Iterative arm only: inspect tool results, then revise the plan", ha="center",
            color="#435d72", fontsize=11)
    ax.text(.2, .06, "Separate reference answers score decisions after prediction; they never enter model requests.",
            color=ink, fontsize=10.5)
    fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.98)
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "workflow.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(folder / "workflow.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def selection_results():
    """Show each fixed test separately; never pool scores across relations."""
    read = lambda p: json.loads((ROOT / p).read_text())
    link_base = read("results/linkage/s2aff-baseline.json")
    link_model = read("results/linkage/model-test-v1-score.json")
    codes = read("results/coding/model-test-v2-score.json")["summaries"]
    tests = [
        ("Affiliation → organization set", "Public historical text · exact annotation-set agreement",
         644, link_base["metrics"]["test"]["arms"]["fused_selective_set"]["exact_sets"],
         link_model["all_cases"]["exact_sets"]),
        ("Establishment description → industry code", "Synthetic NAICS cases · correct code or required review",
         codes["naics_test"]["n"], codes["naics_test"]["baseline_correct"], codes["naics_test"]["decision_correct"]),
        ("Job description → occupation code", "Synthetic NOC cases · correct code or required review",
         codes["noc_test"]["n"], codes["noc_test"]["baseline_correct"], codes["noc_test"]["decision_correct"]),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=180)
    colors = ["#9bafc3", "#176d79"]
    for ax, (title, subtitle, n, baseline, model) in zip(axes, tests):
        ax.set_xlim(0, 105)
        ax.set_ylim(-.65, 1.65)
        ax.barh([1, 0], [100 * baseline / n, 100 * model / n], color=colors, height=.46)
        ax.set_yticks([1, 0], ["Conventional", "Model selection"])
        ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e2e8ec", linewidth=.8)
        for y, count in zip([1, 0], [baseline, model]):
            ax.text(100 * count / n + 1.2, y, f"{count}/{n}", va="center", color="#182d40", fontsize=12)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=13, pad=25, color="#182d40")
        ax.text(0, 1.07, subtitle, transform=ax.transAxes, color="#435d72", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, labelcolor="#435d72")
    fig.suptitle("Model selection improved agreement on three fixed tests", x=.025, ha="left",
                 fontsize=18, fontweight="bold", color="#182d40", y=.995)
    fig.text(.025, .015, "Read each comparison within its task. Synthetic cases do not estimate operational coding accuracy.\n"
             "Affiliation agreement includes review-flagged outputs; it is not an automatic-acceptance accuracy.",
             fontsize=10, color="#435d72", linespacing=1.5)
    fig.subplots_adjust(left=.17, right=.94, top=.88, bottom=.12, hspace=.9)
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "selection-results.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(folder / "selection-results.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    workflow()
    if (ROOT / "results/linkage/model-test-v1-score.json").exists():
        selection_results()
