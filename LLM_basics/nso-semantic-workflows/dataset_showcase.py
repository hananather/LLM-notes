"""Render selected linkage records and decisions from saved local evidence."""
from __future__ import annotations

import base64
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path

from IPython.display import Image, Markdown, display


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
DATA = ROOT / "adversarial-2026/dataset-showcase.json"


def load_verified():
    """Verify every retained source artifact and exact gallery image byte hash."""
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for source in data["source_files"]:
        path = REPO / source["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != source["sha256"]:
            raise ValueError(f"Showcase source changed: {source['path']}")
    return data


def _value(value):
    if value is None:
        return "—"
    if value == "":
        return "(empty source field)"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _html(markup, fallback):
    display({"text/html": markup, "text/plain": fallback}, raw=True)


def _table(rows):
    if not rows:
        return
    keys = list(rows[0])
    alignment = "text-align:left!important;vertical-align:top;white-space:normal;padding:9px 12px"
    header = "".join(f"<th scope='col' style='{alignment}'>{html.escape(k)}</th>" for k in keys)
    body = "".join(
        "<tr>" + "".join(f"<td style='{alignment}'>{html.escape(_value(row[k]))}</td>" for k in keys) + "</tr>"
        for row in rows
    )
    _html("<div class='dataset-showcase'><table><thead><tr>" + header
          + "</tr></thead><tbody>" + body + "</tbody></table></div>",
          "\n".join(" | ".join(f"{key}: {_value(row[key])}" for key in keys) for row in rows))


def _details(label, value):
    text = html.escape(json.dumps(value, indent=2, ensure_ascii=False))
    _html(f"<details><summary>{html.escape(label)}</summary>"
          f"<pre style='white-space:pre-wrap;overflow-wrap:anywhere'>{text}</pre></details>",
          label + "\n" + json.dumps(value, indent=2, ensure_ascii=False))


def _image(asset, alt, caption, width=330):
    path = REPO / asset["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != asset["sha256"]:
        raise ValueError(f"Image hash mismatch: {asset['path']}")
    # Native image output embeds the exact file bytes; alt is image MIME metadata.
    bundle, metadata = Image(data=raw, format=path.suffix.lstrip("."), width=width, alt=alt)._repr_mimebundle_()
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(raw).decode("ascii")
    bundle["text/html"] = (
        f'<img src="data:{mime};base64,{encoded}" alt="{html.escape(alt, quote=True)}" '
        f'width="{width}" style="max-width:100%;height:auto">'
    )
    bundle["text/plain"] = alt
    display(bundle, metadata=metadata, raw=True)
    display(Markdown(caption))


def inventory(data):
    _html("<style>.dataset-showcase {overflow-x:auto;margin:10px 0 18px}"
                 ".dataset-showcase table {border-collapse:collapse;width:100%;font-size:14px}"
                 ".dataset-showcase th,.dataset-showcase td {text-align:left;vertical-align:top;"
                 "padding:9px 12px;border-bottom:1px solid #d6dfe8;white-space:normal}"
                 ".dataset-showcase th {background:#eef3f7;color:#19334a}"
                 "details {margin:8px 0 20px} summary {cursor:pointer}</style>", "Dataset inventory and saved-record gallery.")
    _table(data["inventory"])
    display(Markdown("The product tables preserve the five observed fields shown above; "
                     "missing values remain empty. Opaque IDs join saved files. "
                     "Reference labels are separate evaluation data. "
                     "[Selected full records and provenance](nso-semantic-workflows/adversarial-2026/dataset-showcase.json) "
                     "· [Image attribution and hashes](nso-semantic-workflows/adversarial-2026/figures/dataset-showcase/source-attribution.json)."))


IMAGE_METHODS = {
    "ocr_lexical": "OCR → lexical", "ocr_splink": "OCR → Splink", "clip": "Frozen CLIP",
    "luna_ocr_lexical": "Model on OCR → lexical", "luna_ocr_splink": "Model on OCR → Splink",
    "luna_pixel_lexical": "Model on pixels → lexical", "luna_pixel_splink": "Model on pixels → Splink",
}


def grocery(data):
    cases = {row["query_id"]: row for row in data["images"]["cases"]}
    order = ["q_c81cc1b200047efe130d", "q_f26d1d718ef4889b071e", "q_552df6bae11e5a2914fa"]
    alts = [
        "Vedaka rice bag with large vertical Vedaka lettering and a Rice label beside a burlap sack and scoop of rice.",
        "365 Whole Foods Market Organic Ground Seed Blend package, Chia and Flax, Cocoa Coconut Flavor, net weight 12 oz (340 g).",
        "Fresh tray labelled Artichoke Stuffed Crimini Mushrooms, net weight 9 oz (255 g).",
    ]
    credit = ("Image and catalog: [Amazon Berkeley Objects](https://registry.opendata.aws/amazon-berkeley-objects/) "
              "(Amazon), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
              "The exact prepared evaluation JPEG is shown; the original photograph was curated, not generated.")
    for index, (query_id, alt) in enumerate(zip(order, alts)):
        case = cases[query_id]
        display(Markdown(f"#### {case['title']}\n\nDevelopment illustration: 80 compared queries, 96 catalog candidates."))
        _image(case["asset"], alt, credit, width=230 if index == 2 else 320)
        rows = []
        catalog = {row["record_id"]: row for row in case["catalog_rows"]}
        reference = case["reference"]["catalog_id"]
        shown_ids = [reference] + [key for key in catalog if key != reference]
        for record_id in shown_ids:
            row = catalog[record_id]
            rows.append({"Catalog role": "Reference source row" if record_id == reference else "Selected alternative",
                         "Native title": row["title"], "Brand": row["brand"], "Model number": row["model_number"]})
        _table(rows)
        display(Markdown("**Exact OCR excerpt** (uncorrected; full OCR is linked in the record details):"))
        _html("<pre>" + html.escape(case["ocr"]["exact_excerpt"]) + "</pre>", case["ocr"]["exact_excerpt"])
        if index == 0:
            _table([{"Saved pixel extraction": "brand", "Value": "Vedaka"},
                    {"Saved pixel extraction": "product_name", "Value": "Rice"},
                    {"Saved pixel extraction": "variant / net quantity", "Value": "null / null"}])
        grouped = defaultdict(list)
        for result in case["decisions"]:
            key = (result["status"], tuple(result["target_ids"]), result["correct_complete_decision"])
            grouped[key].append(IMAGE_METHODS[result["method"]])
        decisions = []
        for (status, targets, correct), methods in grouped.items():
            selected = "; ".join(catalog[target]["title"] for target in targets) or "—"
            if targets and targets[0] != reference and catalog[targets[0]]["title"] == catalog[reference]["title"]:
                selected += " [different source record]"
            decisions.append({"Saved method(s)": "; ".join(methods), "Decision": status,
                              "Assigned catalog row": selected,
                              "Reference agreement": "Correct" if correct else ("Unresolved" if status in {"review", "failed"} else "Wrong source association")})
        _table(decisions)
        display(Markdown(case["caption"]))
        if index == 0:
            display(Markdown("The pixel extraction leaves the Ponni variety and 5 kg quantity unknown. "
                             "The correct assignment in this roster does not mean every catalog attribute is visible."))
        if index == 1:
            display(Markdown("The reference row has no model number; the same-title alternative has `180097`. "
                             "These labels establish originating catalog association, not independently adjudicated physical-product identity. "
                             "The rejected pixel response was received; this is an output-schema failure, not an established transport failure."))
        _details("Full source IDs, exact records, extractions, decisions and image provenance", case)


def products(data):
    for case in data["products"]["cases"]:
        dataset = data["products"]["datasets"][case["dataset"]]
        display(Markdown(f"#### {case['title']}\n\n{dataset['name']} · {case['split']} illustration."))
        records = [case["left_record"], *case["right_records"]]
        names = {row["record_id"]: row["name"] for row in records}
        _table([{"Source / role": (dataset["left_name"] + " query") if index == 0 else
                 (dataset["right_name"] + (" reference target" if row["record_id"] in case["reference_target_ids"] else " candidate")),
                 "Exact name": row["name"], "Manufacturer": row["manufacturer"], "Historical price": row["price"]}
                for index, row in enumerate(records)])
        for excerpt in case["recommended_exact_excerpts"]:
            display(Markdown("**Exact description excerpt:** “" + excerpt["text"] + "”"))
        reference = "; ".join(names[key] for key in case["reference_target_ids"]) or "No supplied counterpart (publisher NIL)"
        display(Markdown("**Reference:** " + reference + f". The retrieved shortlist contained {case['candidate_count']} candidates."))
        rows = []
        for method, result in case["observed_methods"].items():
            label = "Semantic selection" if method == "semantic_selection" else ("Lexical" if method.startswith("lexical") else "Splink")
            decision = result["decision"]
            rows.append({"Saved method": label, "Decision": decision["status"],
                         "Selected record": "; ".join(names[key] for key in decision["target_ids"]) or "—",
                         "Published-reference agreement": "Correct" if result["outcome"]["correct"] else "Incorrect"})
        if case["dataset"] == "abt-buy":
            rows.append({"Saved method": "Semantic selection", "Decision": "Not run", "Selected record": "—", "Published-reference agreement": "Unscored"})
        _table(rows)
        display(Markdown(case["caption"]))
        _details("Full exact source descriptions, opaque IDs, saved outcomes and file hashes", case)
    display(Markdown("Source: [Leipzig entity-resolution benchmarks](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution), "
                     "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
                     "Credit H. Köpcke, A. Thor and E. Rahm, *Evaluation of Entity Resolution Approaches on Real-World Match Problems*, PVLDB (2010). "
                     "We curated the published rows into a shared schema; the products and descriptions were not synthetically authored."))


def organizations(data):
    for case in data["organizations"]["examples"]:
        display(Markdown(f"#### {case['title']}"))
        query = case["source_query"]
        if "raw_affiliation_string" in query:
            _table([{"Original native text": query["raw_affiliation_string"],
                     "Published reference organizations": "; ".join(row["name"] for row in case["reference_registry_records"]),
                     "Inference / evaluation": "Not run"}])
            display(Markdown("This one original string has a two-organization reference set. The displayed relation is a published annotation, not a model decision."))
        else:
            _table([{key: query[key] for key in ["ORG", "CITY", "COUNTRY"]}])
            chosen = case["candidate_records"][:2] if not case["source_reference"]["historical_nil"] else [case["candidate_records"][0], case["candidate_records"][2]]
            _table([{"Registry name": row["observed_registry_record"]["name"],
                     "Native registry labels": "; ".join(label["label"] for label in row["observed_registry_record"]["labels"]) or "—",
                     "City / country": "; ".join(row["observed_registry_record"]["cities"]) + " / " + row["observed_registry_record"]["country_code"],
                     "Lexical rank": row["retrieval_evidence"]["rank"]} for row in chosen])
            names = {row["observed_registry_record"]["record_id"]: row["observed_registry_record"]["name"] for row in case["candidate_records"]}
            labels = {"lexical_fixed_policy": "Lexical acceptance policy", "splink_fixed_policy": "Splink acceptance policy", "lexical_top1_forced": "Forced lexical first choice"}
            _table([{"Saved method": labels[method], "Decision": decision["status"],
                     "Target": "; ".join(names[key] for key in decision["target_ids"]) or "—"}
                    for method, decision in case["frozen_decisions"].items()])
            display(Markdown(case["caption"]))
            if case["source_reference"]["historical_nil"]:
                display(Markdown("Review remains unresolved; it does not count as a correct NIL prediction. "
                                 "Splink ranks **Climate and Energy Fund** first, then sends the query to review. "
                                 "A blank historical annotation does not establish absence from the current registry."))
            display(Markdown("**Semantic-model decision: not run** for this organization dataset."))
        _details("Exact observed records, historical reference, saved decisions and score details", case)
    display(Markdown("Source: [AffilGood](https://github.com/sirisacademic/affilgood/tree/9be394d5a8b11360b877ce59fb2355b0a7fa867e/data/entity%20linking) "
                     "(Apache-2.0 notice retained) and [ROR v1.41](https://github.com/ror-community/ror-data/raw/main/v1.41-2024-02-13-ror-data.zip) "
                     "(CC0; GeoNames attribution retained). These are historical research organizations; the labels do not establish enterprise or legal-unit continuity."))


def controls(data):
    control = data["controls"]
    _table([control["shown_source_fields"]])
    descriptions = {
        "clean_structured": ("1. Clean, labelled card", "Catalog observation card: microsoft word 2004 (mac), manufacturer microsoft, price 229.99.", "Original name, manufacturer and price appear in stacked fields."),
        "compact_reordered": ("2. Same values, new order", "Compact table with price 229.99, manufacturer microsoft and name microsoft word 2004 (mac).", "The same three source values appear in a compact table with reordered fields."),
        "compact_degraded": ("3. Same compact card, degraded pixels", "Smaller, blurred and slightly rotated version of the compact Microsoft Word 2004 source card.", "The existing recipe downsamples by 0.55, blurs and rotates the compact rendering. It does not guarantee worse OCR."),
        "missing_identity": ("4. Identity fields removed", "Catalog observation card showing only Price 229.99; no name or manufacturer is present.", "Only the source price remains. Appropriate interpretation: insufficient identity evidence; review or uncertainty."),
    }
    for view in control["views"]:
        title, alt, caption = descriptions[view["view"]]
        display(Markdown(f"#### {title}"))
        _image(view["asset"], alt, caption, width=580)
        display(Markdown("**Exact saved OCR:**"))
        _html("<pre style='white-space:pre-wrap'>" + html.escape(view["ocr_text"]) + "</pre>", view["ocr_text"])
    display(Markdown("We generated these **catalog-observation cards** from observed Amazon–GoogleProducts fields, then ran the same fixed OCR procedure on all views. "
                     "Descriptions are omitted from every inference view; the fourth also omits name and manufacturer. "
                     "The historical price is copied from the source. All four images belong to one source record, not four independent entities. "
                     "**Status: 320 images prepared and OCR-run; model responses, linkage decisions and comparative accuracy are unscored.**\n\n"
                     "Source fields: [Leipzig Amazon–GoogleProducts](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution), "
                     "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); credit Köpcke, Thor and Rahm (PVLDB 2010). "
                     "Changes: deterministic field omission, document rendering, declared image degradation and OCR. These are controlled illustrations, not real receipts."))
    _details("Four exact image hashes, rendering recipes and source-record provenance", control)
