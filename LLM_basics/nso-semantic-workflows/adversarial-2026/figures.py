"""Render the scientific comparison design without loading labels or outcomes.

Run with the repository Python environment. ``--check`` renders in memory and
compares the SVG/PNG bytes with the saved artifacts. The SVG keeps selectable
text. All figure text is at least 14 points in an 18-inch-wide source figure.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
STEM = "product-and-image-comparisons"
WIDTH, HEIGHT = 18, 14.7
INK, MUTED, LINE = "#172E40", "#506474", "#637A8C"
COLORS = {
    "input": ("#F6F8FA", "#8296A5"),
    "conventional": ("#EAF1F8", "#6487A6"),
    "semantic": ("#EAF5F1", "#568F82"),
    "decision": ("#F0F3F6", "#7B8D9C"),
    "truth": ("#FFF4D9", "#B79852"),
}
CAPTION = (
    "Two comparisons separate changes in observed information, representation, and matching. "
    "(A) Native Amazon–GoogleProducts queries and the full catalog feed conventional features and recordwise "
    "large language model (LLM) extraction. Lexical matching and regularized Splink retain both all-pair and same-shortlist comparisons. "
    "Direct semantic selection uses a frozen shortlist of at most 20 candidates and continues to one amended holdout. Recordwise "
    "extraction stays development-only and fits Splink afresh with the conventional numeric prior. "
    "(B) Fixed Tesseract optical character recognition (OCR), pixel extraction, and the pretrained CLIP "
    "image–text encoder use identical prepared query pixels. The OCR-text and pixel LLM arms share a schema, "
    "model, and downstream comparators; both stay development-only. Free OCR and CLIP baselines cover "
    "1,000 test queries. CLIP uses catalog title and brand; lexical and Splink can also use native model numbers. Slashes denote separate comparator runs. Fitting uses "
    "unlabeled observed records and declared prior assumptions. External reference links enter scoring only; "
    "candidate misses and failures remain in the denominator. Each query retains its target set, NIL "
    "(no counterpart), review, or failure state. Product evaluation uses publisher links; ABO evaluation uses "
    "catalog-item association, with physical SKU equivalence unverified. The diagram shows design and "
    "execution scope, without performance results."
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_figure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "svg.fonttype": "none", "svg.hashsalt": "nso-comparison-design-v1",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "text.color": INK,
    })
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=144)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
    ax.set_axis_off()
    texts, bounds = [], []

    def text(x, y, value, size=15, *, bold=False, color=INK, ha="left", va="center"):
        artist = ax.text(x, y, value, fontsize=size, fontweight="bold" if bold else "normal",
                         color=color, ha=ha, va=va, linespacing=1.35, zorder=4)
        texts.append(artist)
        return artist

    def box(x, center, width, title, body, *, height=1.0, kind="conventional", body_color=MUTED):
        fill, stroke = COLORS[kind]
        y = center - height/2
        patch = FancyBboxPatch((x, y), width, height,
            boxstyle="round,pad=0.025,rounding_size=0.095", linewidth=1.35,
            edgecolor=stroke, facecolor=fill, zorder=2)
        ax.add_patch(patch)
        title_text = text(x+width/2, center+height*0.19, title, 16, bold=True, ha="center")
        body_text = text(x+width/2, center-height*0.20, body, 14, color=body_color, ha="center")
        bounds.extend(((title_text, patch), (body_text, patch)))
        return patch

    def line(points, *, color=LINE, width=1.55):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=width, solid_capstyle="round", zorder=1)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                    color=LINE, linewidth=1.55, shrinkA=0, shrinkB=0, zorder=3))

    def decision(center):
        box(13.10, center, 3.3, "One state per query", "Target set or NIL\nReview or failure",
            height=2.05, kind="decision")

    def fan_in(ys, center):
        for y in ys:
            line([(11.98,y), (12.48,y)])
        line([(12.48,min(ys)), (12.48,max(ys))])
        arrow((12.48,center), (13.04,center))

    text(.55, 14.23, "A  Product text", 24, bold=True)
    text(4.00, 14.23, "Native Amazon–GoogleProducts", 18, color=MUTED)
    text(.55, 13.75, "One observed source population; compare matching, candidate selection, and representation.", 15, color=MUTED)

    product_ys = [12.88, 11.50, 10.12]
    box(.55, 11.50, 2.75, "Observed text", "Queries + full catalog\nSame source fields", height=1.8, kind="input")
    line([(3.33,11.50), (3.58,11.50)])
    line([(3.58,min(product_ys)), (3.58,max(product_ys))])
    for y in product_ys:
        arrow((3.58,y), (3.85,y))
        arrow((7.53,y), (8.09,y))
    box(3.90, product_ys[0], 3.60, "Explicit features", "Text · brand · model · size")
    box(8.15, product_ys[0], 3.80, "Lexical / regularized Splink", "All pairs + same shortlist")
    box(3.90, product_ys[1], 3.60, "Fixed shortlist (≤20)", "Frozen candidate list")
    box(8.15, product_ys[1], 3.80, "Direct semantic selection", "One amended holdout", kind="semantic")
    box(3.90, product_ys[2], 3.60, "Recordwise LLM extraction", "Each record; no candidate context", kind="semantic")
    box(8.15, product_ys[2], 3.80, "Fresh regularized Splink", "Same prior · DEV only", kind="semantic")
    decision(11.50)
    fan_in(product_ys, 11.50)
    text(3.90, 9.34, "Recordwise extraction changes fields; its Splink parameters are fitted afresh without labels.", 14, color=MUTED)
    line([(.55,8.88), (16.50,8.88)], color="#D4DCE3", width=1.1)

    text(.55, 8.35, "B  Package images", 24, bold=True)
    text(4.72, 8.35, "ABO catalog-item association", 18, color=MUTED)
    text(.55, 7.87, "Same catalog records; CLIP uses title and brand. Both semantic extraction arms remain DEV only.", 15, color=MUTED)

    image_ys = [7.08, 5.74, 4.40, 3.06]
    image_center = 5.07
    box(.55, image_center, 2.75, "Prepared pixels", "Same bytes in every arm\nOne image per query", height=1.8, kind="input")
    line([(3.33,image_center), (3.58,image_center)])
    line([(3.58,image_ys[-1]), (3.58,image_ys[0])])
    # The semantic OCR arm receives Tesseract text, not an independent pixel input.
    for y in (image_ys[0],image_ys[2],image_ys[3]):
        arrow((3.58,y), (3.85,y))
    box(3.90, image_ys[0], 3.60, "Fixed Tesseract OCR", "PSM 11 + 6 line union")
    box(8.15, image_ys[0], 3.80, "Lexical / Splink", "1,000-query test baselines")
    arrow((5.70, image_ys[0]-.54), (5.70,image_ys[1]+.56))
    box(3.90, image_ys[1], 3.60, "LLM attributes from OCR", "Same schema and model", kind="semantic")
    box(8.15, image_ys[1], 3.80, "Lexical / fresh Splink", "Same catalog · DEV only", kind="semantic")
    box(3.90, image_ys[2], 3.60, "LLM attributes from pixels", "Same schema and model", kind="semantic")
    box(8.15, image_ys[2], 3.80, "Lexical / fresh Splink", "Same catalog · DEV only", kind="semantic")
    for y in image_ys[:3]:
        arrow((7.53,y), (8.09,y))
    box(3.90, image_ys[3], 8.05, "Frozen CLIP: image → catalog-text similarity",
        "Direct image comparator · 1,000-query test baseline")
    decision(image_center)
    fan_in(image_ys, image_center)

    # Saved decisions feed only the shared evaluation endpoint. The external
    # reference-link arrow has no connection to fitting or extraction nodes.
    for center in (11.50,image_center):
        line([(16.43,center), (17.13,center)])
    line([(17.13,11.50), (17.13,2.10), (14.80,2.10)])
    arrow((14.80,2.10), (14.80,1.72))
    box(8.10, 1.02, 3.80, "External reference links", "Evaluation only", height=1.30, kind="truth")
    box(12.65, 1.02, 4.30, "Score every attempted query", "Target-set errors + review/failure", height=1.30, kind="decision")
    arrow((11.93,1.02), (12.59,1.02))
    text(.55, 1.38, "Fitting uses observed records, without identity labels.", 15, bold=True)
    text(.55, .90, "Candidate misses and failed calls remain in the denominator.", 14, color=MUTED)
    text(.55, .45, "Declared prior assumptions are separate from fitted parameters.", 14, color=MUTED)

    # Layout checks concern the visual artifact, never data or empirical results.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, patch in bounds:
        text_extent = artist.get_window_extent(renderer)
        box_extent = patch.get_window_extent(renderer)
        if not (box_extent.x0 <= text_extent.x0 and text_extent.x1 <= box_extent.x1
                and box_extent.y0 <= text_extent.y0 and text_extent.y1 <= box_extent.y1):
            raise RuntimeError("Text exceeds its node: " + artist.get_text())
    assert min(t.get_fontsize() for t in texts) >= 14
    return fig, texts


def render(check=False):
    fig, texts = make_figure()
    assets = {}
    for extension in ("svg", "png"):
        buffer = io.BytesIO()
        metadata = {"Date": None, "Creator": "NSO semantic workflow diagrams"} if extension == "svg" else {"Software": "NSO semantic workflow diagrams"}
        fig.savefig(buffer, format=extension, dpi=144, metadata=metadata)
        assets[extension] = buffer.getvalue()
    plt.close(fig)
    font = Path(font_manager.findfont("DejaVu Sans"))
    manifest = {
        "figure": STEM, "source_sha256": sha(Path(__file__).read_bytes()),
        "matplotlib_version": matplotlib.__version__, "font": font.name,
        "font_sha256": sha(font.read_bytes()), "size_inches": [WIDTH,HEIGHT],
        "png_dpi": 144, "minimum_font_points": min(t.get_fontsize() for t in texts),
        "artifact_sha256": {ext:sha(data) for ext,data in assets.items()},
        "scope": "Scientific design and execution scope; no result chart or outcome data read",
    }
    if check:
        for extension, data in assets.items():
            assert (OUT/f"{STEM}.{extension}").read_bytes() == data, extension + " render differs"
        assert json.loads((OUT/f"{STEM}.json").read_text()) == manifest
        print("Verified deterministic SVG/PNG bytes, font minimum and node text bounds.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for extension, data in assets.items():
        (OUT/f"{STEM}.{extension}").write_bytes(data)
    (OUT/f"{STEM}.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (OUT/f"{STEM}-caption.md").write_text(CAPTION+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    render(parser.parse_args().check)
