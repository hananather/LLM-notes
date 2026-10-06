"""Execute the tutorial without loading machine-wide Jupyter configuration."""

import argparse
from pathlib import Path
import re

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make new paid model calls; requires OPENAI_API_KEY.")
    parser.add_argument("--html", type=Path, help="Also export the executed notebook to this HTML path.")
    args = parser.parse_args()
    notebook_path = Path(__file__).resolve().parent.parent / "record-linkage-with-semantic-operators.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    config_cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "RUN_LIVE = " in cell.source)
    config_cell.source = re.sub(r"^RUN_LIVE = (?:True|False)$", f"RUN_LIVE = {args.live}", config_cell.source, flags=re.M)
    NotebookClient(notebook, timeout=240, kernel_name="python3",
                   resources={"metadata": {"path": str(notebook_path.parent)}}).execute()
    # A saved notebook always starts in the free replay mode on its next execution.
    config_cell.source = re.sub(r"^RUN_LIVE = (?:True|False)$", "RUN_LIVE = False", config_cell.source, flags=re.M)
    nbformat.write(notebook, notebook_path)
    if args.html:
        html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(html)
    print(f"Executed {len(notebook.cells)} cells successfully ({'live' if args.live else 'replay'}).")
    print(notebook_path)


if __name__ == "__main__":
    main()
