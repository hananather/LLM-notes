"""Render frozen observed product facts and run real OCR, without identity labels.

Only records.json and partitions.json are read from the product data directory.
Model predictions, evaluator metadata, source mappings and identity truth are not
inputs. All writes are confined to this controls directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, __version__ as PIL_VERSION

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "products" / "data" / "amazon-google" / "v2"
SEED = "controls-20261007-v1"
SIZES = {"dev": 16, "test": 64}
FIELDS = ("record_id", "name", "description", "manufacturer", "price")
VIEWS = ("clean_structured", "compact_reordered", "compact_degraded", "missing_identity")
EXPECTED = {
    "records.json": "348231e9bf0ec59acf9a13e65e74c5cb1d1e80344c7b9a897c8469d17ba93a5c",
    "partitions.json": "65112f0240dd25ca9e23c4e1655bd18c1d9f9cb6e6b63dd685f9f9cc8632fab8",
}
FONT_ROOT = Path("/System/Library/Fonts/Supplemental")
FONT_PATHS = {
    "sans": FONT_ROOT / "Arial.ttf",
    "sans_bold": FONT_ROOT / "Arial Bold.ttf",
    "serif": FONT_ROOT / "Georgia.ttf",
    "serif_bold": FONT_ROOT / "Georgia Bold.ttf",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def freeze(path: Path, value) -> None:
    data = canonical(value)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f"Refusing to overwrite changed frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def frozen_bytes(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f"Refusing to overwrite changed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def font(which: str, size: int):
    return ImageFont.truetype(str(FONT_PATHS[which]), size)


def literal_lines(value: str, face, width: int) -> list[str]:
    """Insert visual line breaks without deleting, changing or adding characters."""
    result = []
    for paragraph in value.split("\n"):
        if not paragraph:
            result.append("")
            continue
        current = ""
        for character in paragraph:
            if current and face.getlength(current + character) > width:
                result.append(current)
                current = ""
            current += character
        result.append(current)
    assert "".join(result) == value.replace("\n", "")
    return result


def draw_lines(draw, position, lines, face, *, fill="#151b25", step=40):
    x, y = position
    for line in lines:
        draw.text((x, y), line, font=face, fill=fill)
        y += step
    return y


def render_clean(record: dict, split: str) -> Image.Image:
    if split == "dev":
        face, label = font("sans", 34), font("sans_bold", 21)
        wrapped = {key: literal_lines(record[key], face, 896) for key in ("name", "manufacturer", "price")}
        height = 134 + sum(42 + 43 * len(wrapped[key]) + 25 for key in wrapped)
        image = Image.new("RGB", (1000, height), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1000, 86), fill="#e9eff7")
        draw.text((42, 24), "CATALOG OBSERVATION", font=font("sans_bold", 28), fill="#253c5b")
        y = 116
        for key in ("name", "manufacturer", "price"):
            draw.text((44, y), key.upper(), font=label, fill="#4b5b6c")
            y = draw_lines(draw, (44, y + 34), wrapped[key], face, step=43) + 25
        return image
    face, label = font("serif", 32), font("serif_bold", 23)
    wrapped = {key: literal_lines(record[key], face, 880) for key in ("name", "manufacturer", "price")}
    heights = {key: 52 + 42 * len(wrapped[key]) for key in wrapped}
    image = Image.new("RGB", (1240, 132 + sum(heights.values())), "#fffdf7")
    draw = ImageDraw.Draw(image)
    draw.text((48, 26), "Catalog observation", font=font("serif_bold", 32), fill="#3d3b31")
    y = 109
    for key in ("name", "manufacturer", "price"):
        draw.line((48, y, 1192, y), fill="#aaa58e", width=2)
        draw.text((52, y + 20), key.capitalize(), font=label, fill="#555145")
        draw_lines(draw, (284, y + 18), wrapped[key], face, step=42)
        y += heights[key]
    return image


def render_compact(record: dict, split: str) -> Image.Image:
    if split == "dev":
        face, label = font("sans", 30), font("sans_bold", 20)
        fields = ("price", "manufacturer", "name")
        edges = (28, 190, 520, 1152)
        wrapped = {key: literal_lines(record[key], face, edges[i+1]-edges[i]-30) for i,key in enumerate(fields)}
        height = 142 + max(len(value) for value in wrapped.values()) * 38
        image = Image.new("RGB", (1180, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((30, 18), "CATALOG OBSERVATION", font=font("sans_bold", 24), fill="#253c5b")
        draw.rectangle((28, 65, 1152, height-24), outline="#536272", width=2)
        draw.rectangle((30, 67, 1150, 103), fill="#e9eff7")
        for i,key in enumerate(fields):
            if i:
                draw.line((edges[i], 65, edges[i], height-24), fill="#536272", width=2)
            draw.text((edges[i]+14, 72), key.upper(), font=label, fill="#273c53")
            draw_lines(draw, (edges[i]+14, 112), wrapped[key], face, step=38)
        return image
    face, label = font("serif", 28), font("serif_bold", 21)
    fields = ("manufacturer", "price", "name")
    wrapped = {key: literal_lines(record[key], face, 860) for key in fields}
    heights = {key: 25 + 36 * len(wrapped[key]) for key in fields}
    image = Image.new("RGB", (1130, 99 + sum(heights.values())), "#fffdf7")
    draw = ImageDraw.Draw(image)
    draw.text((28, 17), "Catalog observation", font=font("serif_bold", 25), fill="#3d3b31")
    y = 72
    for key in fields:
        bottom = y + heights[key]
        draw.rectangle((24,y,1106,bottom), outline="#77715e", width=2)
        draw.rectangle((26,y+2,217,bottom-2), fill="#eeeadd")
        draw.line((219,y,219,bottom), fill="#77715e", width=2)
        draw.text((35,y+12), key.capitalize(), font=label, fill="#3d3b31")
        draw_lines(draw, (237,y+10), wrapped[key], face, step=36)
        y = bottom
    return image


def render_missing(record: dict, split: str) -> Image.Image:
    # Only price plus generic document labels. Neither identity nor IDs are drawn.
    serif = split == "test"
    image = Image.new("RGB", (720 if serif else 660, 260), "#fffdf7" if serif else "white")
    draw = ImageDraw.Draw(image)
    draw.text((35,28), "Catalog observation", font=font("serif_bold" if serif else "sans_bold",27), fill="#333333")
    draw.text((35,102), "Price", font=font("serif_bold" if serif else "sans_bold",23), fill="#555555")
    draw.text((35,155), record["price"], font=font("serif" if serif else "sans",38), fill="#151b25")
    return image


def degradation(record_id: str) -> dict:
    polarity = 1 if int(digest((SEED + "|rotation|" + record_id).encode())[:2], 16) % 2 else -1
    return {"downsample_factor": 0.55, "downsample_resampling": "LANCZOS", "blur_radius_output_pixels": 0.6,
            "rotation_degrees": 1.5 * polarity, "rotation_resampling": "BICUBIC", "expand": True, "fill": "white"}


def degrade(image: Image.Image, recipe: dict) -> Image.Image:
    small = image.resize(tuple(round(v * recipe["downsample_factor"]) for v in image.size), Image.Resampling.LANCZOS)
    small = small.filter(ImageFilter.GaussianBlur(recipe["blur_radius_output_pixels"]))
    return small.rotate(recipe["rotation_degrees"], resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")


def png_bytes(image: Image.Image) -> bytes:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def no_description(row: dict) -> dict:
    return {key: "" if key == "description" else row[key] for key in FIELDS}


def ocr_row(record_id: str, text: str) -> dict:
    # All observed OCR text goes into name; no source facts silently bypass OCR.
    return {"record_id": record_id, "name": text, "description": "", "manufacturer": "", "price": ""}


def inputs(split: str, view: str):
    """Return five-field left/right records for ONE condition; never pool views."""
    if split not in SIZES or view not in (*VIEWS, "source_structured"):
        raise ValueError("Unknown split or view")
    left_name = "left-source-no-description.json" if view == "source_structured" else f"{view}/ocr-five-field-rows.json"
    return json.loads((HERE / "data" / split / left_name).read_text()), json.loads((HERE / "data" / split / "right-no-description.json").read_text())


def select_fixed_candidates(candidates: list[dict], selected_ids: list[str]) -> list[dict]:
    """Filter a caller's frozen candidate list without adding or reordering targets.

    Generate that shared list from the source_structured no-description arm if
    isolating layout/OCR. The original description-enabled list changes the
    retrieval-information boundary and must be disclosed separately.
    """
    index = {row["record_id"]: row for row in candidates}
    if len(index) != len(candidates) or any(rid not in index for rid in selected_ids):
        raise ValueError("Duplicate or absent query in fixed candidate list")
    return [index[rid] for rid in selected_ids]


def run_ocr(path: Path, executable: str) -> tuple[str,str]:
    completed = subprocess.run([executable, str(path), "stdout", "-l", "eng", "--oem", "1", "--psm", "6"],
                               capture_output=True, check=True, timeout=90)
    return completed.stdout.decode("utf-8"), completed.stderr.decode("utf-8")


def prepare():
    executable = shutil.which("tesseract")
    if not executable:
        raise RuntimeError("Installed Tesseract is required; no package is installed by this script")
    for filename, expected in EXPECTED.items():
        if digest((SOURCE/filename).read_bytes()) != expected:
            raise RuntimeError(f"Source hash differs: {filename}")
    for path in FONT_PATHS.values():
        if not path.exists():
            raise RuntimeError(f"Pinned render font is unavailable: {path}")
    records = json.loads((SOURCE/"records.json").read_text())
    if any(set(row) != set(FIELDS) or any(not isinstance(row[k],str) for k in FIELDS) for row in records):
        raise ValueError("Unexpected observed source schema")
    index = {row["record_id"]:row for row in records}
    parts = json.loads((SOURCE/"partitions.json").read_text())
    selected = {split: sorted(parts[split]["left"], key=lambda rid: digest(f"{SEED}|{split}|{rid}".encode()))[:count]
                for split,count in SIZES.items()}
    assert not set(selected["dev"]) & set(selected["test"])
    version = subprocess.check_output([executable,"--version"], text=True).strip()
    protocol = {
        "version": "source-fact-ocr-controls-v1", "seed": SEED,
        "status": "Frozen observed-data controls; no model outcomes or linkage scores generated",
        "source": {"dataset": "Amazon-GoogleProducts", "curation": "v2", "file_sha256": EXPECTED,
                   "url": "https://dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip",
                   "licence": "CC BY 4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                   "attribution": "Leipzig University Database Group; H. Köpcke, A. Thor and E. Rahm (2010), Evaluation of Entity Resolution Approaches on Real-World Match Problems"},
        "selection": {"rule": "Take the smallest SHA256(seed|split|opaque LEFT record_id) values within frozen partitions", "counts": SIZES,
                      "uses": ["partition membership","opaque record_id"], "outcomes_or_gold_used": False,
                      "grouping": "All four views of each original source record share one control_family_id; existing source family split inherited"},
        "observed_fields": ["name","manufacturer","price"],
        "ablation": "Description omitted from every rendered/OCR view and both left/right inference tables; immutable source copies retain original description in audit-only provenance",
        "views": {"clean_structured": "Labeled catalog-observation card; original literal source values",
                  "compact_reordered": "Identical source facts with compact reordered table layout",
                  "compact_degraded": "Same compact image downsampled 0.55, blurred radius0.6, rotated hash-selected +/-1.5 degrees, with expansion to avoid clipping",
                  "missing_identity": "Only original price and generic Catalog observation/Price labels; expected action review for insufficient identity evidence"},
        "layout_families": {"dev": {"clean": "dev-stacked-sans", "compact": "dev-three-column-sans", "missing": "dev-price-sans"},
                            "test": {"clean": "test-label-column-serif", "compact": "test-two-column-serif", "missing": "test-price-serif"}},
        "layout_boundary": "DEV and TEST use distinct fixed layout/font families; variation is a designed mechanism control, not natural deployment frequency",
        "ocr": {"engine": "Tesseract", "language": "eng", "oem": 1, "psm": 6, "executable": executable, "version": version,
                "same_configuration_every_view": True, "text_postprocessing": "none; literal UTF8 stdout retained", "retry": False},
        "missing_identity_scoring": {"expected_state": "review", "reason": "insufficient_identity_evidence", "do_not_score_as": ["forced exact link","true NIL","independent entity"],
                                     "note": "Original source association remains provenance only; price alone does not establish a unique identity"},
        "interfaces": {"structured": "data/{split}/left-source-no-description.json and right-no-description.json",
                       "ocr": "data/{split}/{view}/ocr-five-field-rows.json", "raw_ocr": "data/{split}/{view}/ocr.jsonl",
                       "identity": "record_id remains original source opaque ID; view_id and control_family_id are audit-only manifest fields",
                       "pooling": "Run each view separately against the same right roster and shared no-description candidates; never concatenate repeated record_ids"},
        "assets": {"generated_images": ".cache/rendered/{split}/{view}/{record_id}.png", "git_policy": ".cache excluded; small DEV contact sheet and OCR/recipe metadata retained",
                   "regeneration": "Pinned source/font/software hashes; renderer checks generated bytes against frozen image hashes"},
        "limits": ["Synthetic catalog-observation cards, not receipts or original merchant documents", "80 source records yield320 dependent views, not320 independent entities", "No counterpart labels or outcome files opened", "No paid/model calls or confidence claims"]
    }
    freeze(HERE/"protocol.json",protocol)
    freeze(HERE/"data"/"selection.json",{"seed":SEED,"selected":selected,"partition_sha256":EXPECTED["partitions.json"]})
    freeze(HERE/"data"/"source-records-audit-only.json",[index[rid] for split in SIZES for rid in selected[split]])
    freeze(HERE/"schema.json",{
        "five_field_row": {"additional_fields":False, "fields":{key:"string" for key in FIELDS}, "description":"Always empty in inference tables"},
        "ocr_wrapper": {"record_id":"Original opaque source ID", "text":"Unmodified Tesseract stdout", "view_id":"Audit-only unique condition ID"},
        "manifest_record": {"control_family_id":"Source record ID shared by every view", "original_query_id":"Original opaque source ID", "view":"Predeclared condition", "split":"Inherited partition", "image_sha256":"SHA256 PNG bytes", "pixel_sha256":"SHA256 raw RGB pixels", "ocr_sha256":"SHA256 raw OCR UTF8", "rendered_source_fields":"Literal original field values; audit-only"}
    })
    manifest = []
    for split,ids in selected.items():
        freeze(HERE/"data"/split/"left-source-no-description.json",[no_description(index[rid]) for rid in ids])
        freeze(HERE/"data"/split/"right-no-description.json",[no_description(index[rid]) for rid in parts[split]["right"]])
        for view in VIEWS:
            rows,raw = [],[]
            for rid in ids:
                row = index[rid]
                recipe = degradation(rid) if view == "compact_degraded" else None
                if view == "clean_structured":image = render_clean(row,split)
                elif view == "missing_identity":image = render_missing(row,split)
                else:
                    image = render_compact(row,split)
                    if recipe:image = degrade(image,recipe)
                image_path = HERE/".cache"/"rendered"/split/view/(rid+".png")
                frozen_bytes(image_path,png_bytes(image))
                text,stderr = run_ocr(image_path,executable)
                view_id = f"{rid}:{view}"
                entry = {"record_id":rid,"view_id":view_id,"text":text}
                rows.append(ocr_row(rid,text));raw.append(entry)
                fields = ["price"] if view == "missing_identity" else ["name","manufacturer","price"]
                manifest.append({"original_query_id":rid,"control_family_id":rid,"view_id":view_id,"split":split,"view":view,
                    "source_record_sha256":digest(canonical(row)),"image_path":str(image_path.relative_to(HERE)),
                    "image_sha256":digest(image_path.read_bytes()),"pixel_sha256":digest(image.tobytes()),"image_size":list(image.size),
                    "ocr_sha256":digest(text.encode()),"ocr_characters":len(text),"ocr_words":len(text.split()),"ocr_stderr":stderr,
                    "rendered_source_fields":{key:row[key] for key in fields},"rendered_generic_labels":["Catalog observation"]+[key.capitalize() for key in fields],
                    "description_exposed":False,"source_id_rendered":False,"degradation":recipe,
                    "expected_evidence_state":"insufficient_identity_evidence" if view == "missing_identity" else "source_identity_fields_present",
                    "ground_truth": "Source-record association only; no counterpart label loaded"})
            freeze(HERE/"data"/split/view/"ocr-five-field-rows.json",rows)
            frozen_bytes(HERE/"data"/split/view/"ocr.jsonl",b"".join(json.dumps(row,ensure_ascii=False,sort_keys=True).encode()+b"\n" for row in raw))
        print(f"Prepared {split}: {len(ids)} source records, {len(ids)*len(VIEWS)} views",flush=True)
    freeze(HERE/"data"/"manifest.json",manifest)
    # DEV-only visual inspection sheet: first four prospectively selected IDs.
    panel_w,panel_h = 390,275
    contact = Image.new("RGB",(panel_w*4,panel_h*4),"#edf0f3")
    draw = ImageDraw.Draw(contact)
    for r,rid in enumerate(selected["dev"][:4]):
        for c,view in enumerate(VIEWS):
            path = HERE/".cache"/"rendered"/"dev"/view/(rid+".png")
            picture = Image.open(path).convert("RGB")
            picture.thumbnail((panel_w-16,panel_h-42),Image.Resampling.LANCZOS)
            x,y = c*panel_w,r*panel_h
            draw.text((x+8,y+7),f"DEV {r+1}: {view}",font=font("sans",16),fill="black")
            contact.paste(picture,(x+8,y+34))
    frozen_bytes(HERE/"data"/"dev-contact-sheet.png",png_bytes(contact))
    summaries = {}
    for split in SIZES:
        summaries[split] = {}
        for view in VIEWS:
            group = [row for row in manifest if row["split"]==split and row["view"]==view]
            lengths = [row["ocr_characters"] for row in group]
            summaries[split][view] = {"views":len(group),"ocr_characters_min":min(lengths),"ocr_characters_median":statistics.median(lengths),"ocr_characters_max":max(lengths),"empty_ocr":sum(v==0 for v in lengths)}
    runtime = {"python":platform.python_version(),"platform":platform.platform(),"Pillow":PIL_VERSION,"tesseract":version,
               "fonts":{k:{"path":str(v),"sha256":digest(v.read_bytes())} for k,v in FONT_PATHS.items()}}
    freeze(HERE/"data"/"runtime.json",runtime)
    freeze(HERE/"data"/"summary.json",{"source_records":sum(SIZES.values()),"distinct_real_entity_count":"Not asserted; counterpart labels were not read", "rendered_views":len(manifest),"ocr_summary":summaries,
        "unique_image_hashes":len({row["image_sha256"] for row in manifest}),"image_bytes":sum((HERE/row["image_path"]).stat().st_size for row in manifest),
        "manifest_sha256":digest((HERE/"data"/"manifest.json").read_bytes()),"renderer_sha256":digest(Path(__file__).read_bytes()),
        "scope":"No model outputs, counterparty truth, counterpart decisions, baseline performance or entity matching scores generated"})
    print(json.dumps(summaries,indent=2))


def verify():
    manifest = json.loads((HERE/"data"/"manifest.json").read_text())
    selection = json.loads((HERE/"data"/"selection.json").read_text())["selected"]
    source = {row["record_id"]:row for row in json.loads((HERE/"data"/"source-records-audit-only.json").read_text())}
    assert len(source)==80 and len(manifest)==320
    for filename, expected in EXPECTED.items():
        assert digest((SOURCE / filename).read_bytes()) == expected
    for split,ids in selection.items():
        assert len(ids)==SIZES[split]
        for view in VIEWS:
            left,right=inputs(split,view)
            assert [row["record_id"] for row in left]==ids
            assert all(set(row)==set(FIELDS) and row["description"]=="" for row in left+right)
            assert not {row["record_id"] for row in left}&{row["record_id"] for row in right}
    for row in manifest:
        rid=row["original_query_id"]
        expected={key:source[rid][key] for key in (["price"] if row["view"]=="missing_identity" else ["name","manufacturer","price"])}
        assert row["rendered_source_fields"]==expected
        assert not row["description_exposed"] and not row["source_id_rendered"]
        if row["view"] == "missing_identity":
            probe = dict(source[rid], name="MUST NOT APPEAR", manufacturer="MUST NOT APPEAR", record_id="MUST NOT APPEAR", description="MUST NOT APPEAR")
            assert digest(render_missing(probe, row["split"]).tobytes()) == row["pixel_sha256"]
        image_path=HERE/row["image_path"]
        if image_path.exists():
            assert digest(image_path.read_bytes())==row["image_sha256"]
            assert digest(Image.open(image_path).convert("RGB").tobytes())==row["pixel_sha256"]
    for split in SIZES:
        for view in VIEWS:
            raw=[json.loads(line) for line in (HERE/"data"/split/view/"ocr.jsonl").read_text().splitlines()]
            entries={row["original_query_id"]:row for row in manifest if row["split"]==split and row["view"]==view}
            assert all(digest(row["text"].encode())==entries[row["record_id"]]["ocr_sha256"] for row in raw)
    print("Verified: 80 source records,320 views; source-field, split, ablation, OCR and available image hashes agree.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    verify() if args.verify else prepare()
