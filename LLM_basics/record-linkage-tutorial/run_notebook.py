"""Execute the complete semantic-join report without machine-wide Jupyter configuration."""

import argparse
import importlib.util
from html import escape, unescape
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor


class StaticMathHTMLExporter(HTMLExporter):
    """Embed mathematical notation so a reading copy needs no remote renderer."""

    def from_notebook_node(self, notebook, resources=None, **kwargs):
        html, resources = super().from_notebook_node(notebook, resources, **kwargs)
        return render_static_math(html), resources


def render_static_math(html):
    """Render Markdown math as described SVG images, preserving its TeX source."""
    import base64
    from io import BytesIO

    from bs4 import BeautifulSoup
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser, math_to_image

    pattern = re.compile(r"\$\$([\s\S]*?)\$\$|(?<![\\$])\$([^$\n]+?)\$(?!\$)")
    soup = BeautifulSoup(html, "html.parser")
    parser = MathTextParser("path")
    font = FontProperties(size=11)
    for block in soup.select(".jp-RenderedMarkdown"):
        for text in list(block.find_all(string=True)):
            if text.parent.name in {"script", "style", "code", "pre"}:
                continue
            value = str(text)
            matches = list(pattern.finditer(value))
            if not matches:
                continue
            pieces, offset = [], 0
            for match in matches:
                pieces.append(value[offset:match.start()])
                tex = (match.group(1) or match.group(2)).replace("\n", " ")
                expression = "$" + tex + "$"
                geometry = parser.parse(expression, dpi=72, prop=font)
                buffer = BytesIO()
                math_to_image(expression, buffer, prop=font, format="svg", color="#1E3342")
                image = soup.new_tag("img", alt=tex)
                image["src"] = "data:image/svg+xml;base64," + base64.b64encode(buffer.getvalue()).decode()
                image["class"] = "report-static-math"
                if match.group(1) is not None:
                    image["style"] = (f"display:block;margin:1em auto;max-width:100%;"
                                      f"width:{geometry.width / 11:.3f}em;height:auto")
                else:
                    image["style"] = (f"width:{geometry.width / 11:.3f}em;height:auto;"
                                      f"vertical-align:{-geometry.depth / 11:.3f}em")
                pieces.append(image)
                offset = match.end()
            pieces.append(value[offset:])
            text.replace_with(*pieces)
    for script in list(soup.find_all("script")):
        if "mathjax" in script.get("src", "").lower() or script.get("type") == "text/x-mathjax-config":
            script.decompose()
    return str(soup)


def reading_exporter():
    """Retain image descriptions when the Lab template renders PNG outputs."""
    exporter = StaticMathHTMLExporter(template_name="lab", raw_template='''
{% extends 'lab/index.html.j2' %}
{% block data_png scoped %}
{%- set description = (output | get_metadata('alt', 'image/png')) or (cell | get_metadata('alt')) or '' -%}
{{ super() | replace('<img ', '<img alt="' ~ (description | escape_html) ~ '" ') }}
{% endblock data_png %}
''')
    exporter.register_preprocessor(
        TagRemovePreprocessor(remove_cell_tags={"reader-hide-setup"},
                              remove_input_tags={"reader-hide-input"}), enabled=True)
    return exporter


def rebase_file_links(html, notebook_dir, output_dir):
    """Keep notebook-relative references valid when exporting elsewhere."""
    def replace(match):
        url = urlsplit(unescape(match.group(1)))
        if url.scheme or url.netloc or not url.path:
            return match.group(0)
        source = (notebook_dir / unquote(url.path)).resolve()
        relative = Path(os.path.relpath(source, output_dir)).as_posix()
        href = urlunsplit(("", "", quote(relative, safe="/"), url.query, url.fragment))
        return 'href="' + escape(href, quote=True) + '"'

    return re.sub(r'href="([^"]*)"', replace, html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make new paid model calls; requires OPENAI_API_KEY.")
    parser.add_argument("--live-unstructured", action="store_true",
                        help="Rerun the small BioDEX and FEVER model batches; requires OPENAI_API_KEY.")
    parser.add_argument("--live-agentic", action="store_true",
                        help="Rerun the three-office agentic workflow; requires OPENAI_API_KEY.")
    parser.add_argument("--html", type=Path, help="Also export the executed notebook to this HTML path.")
    args = parser.parse_args()
    notebook_path = Path(__file__).resolve().parent.parent / "record-linkage-with-semantic-operators.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    config_cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "RUN_LIVE = " in cell.source)
    config_cell.source = re.sub(r"^RUN_LIVE = (?:True|False)$", f"RUN_LIVE = {args.live}", config_cell.source, flags=re.M)
    config_cell.source = re.sub(r"^RUN_UNSTRUCTURED_LIVE = (?:True|False)$",
                               f"RUN_UNSTRUCTURED_LIVE = {args.live_unstructured}",
                               config_cell.source, flags=re.M)
    config_cell.source = re.sub(r"^RUN_AGENTIC_LIVE = (?:True|False)$",
                               f"RUN_AGENTIC_LIVE = {args.live_agentic}",
                               config_cell.source, flags=re.M)
    runner_path = notebook_path.parent / "nso-semantic-workflows" / "run_notebook.py"
    spec = importlib.util.spec_from_file_location("report_replay", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.execute_saved(notebook, notebook_path.parent)
    # A saved notebook always starts in the free replay mode on its next execution.
    config_cell.source = re.sub(r"^RUN_LIVE = (?:True|False)$", "RUN_LIVE = False", config_cell.source, flags=re.M)
    config_cell.source = re.sub(r"^RUN_UNSTRUCTURED_LIVE = (?:True|False)$", "RUN_UNSTRUCTURED_LIVE = False",
                               config_cell.source, flags=re.M)
    config_cell.source = re.sub(r"^RUN_AGENTIC_LIVE = (?:True|False)$", "RUN_AGENTIC_LIVE = False",
                               config_cell.source, flags=re.M)
    nbformat.write(notebook, notebook_path)
    if args.html:
        exporter = reading_exporter()
        html, _ = exporter.from_notebook_node(
            notebook, resources={"metadata": {"name": "Semantic joins and record linkage"}})
        args.html.parent.mkdir(parents=True, exist_ok=True)
        html = rebase_file_links(html, notebook_path.parent, args.html.resolve().parent)
        args.html.write_text(html, encoding="utf-8")
    live_tasks = [name for name, enabled in (("FEBRL", args.live),
                  ("unstructured", args.live_unstructured), ("agentic", args.live_agentic)) if enabled]
    mode = "live: " + ", ".join(live_tasks) if live_tasks else "replay"
    print(f"Executed {len(notebook.cells)} cells successfully ({mode}).")
    print(notebook_path)


if __name__ == "__main__":
    main()
