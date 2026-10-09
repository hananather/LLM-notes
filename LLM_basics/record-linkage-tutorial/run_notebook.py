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


def reading_exporter():
    """Retain image descriptions when the Lab template renders PNG outputs."""
    exporter = HTMLExporter(template_name="lab", raw_template='''
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
