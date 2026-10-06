"""Execute the NSO experiment report from saved evidence, without model calls."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


def execute_saved(notebook, notebook_dir):
    """Use the launching interpreter without installing a global kernel."""
    with TemporaryDirectory(prefix="nso-notebook-kernel-") as directory:
        kernel_dir = Path(directory) / "nso-report"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "NSO report replay", "language": "python",
        }), encoding="utf-8")
        manager = AsyncKernelManager(
            kernel_name="nso-report",
            kernel_spec_manager=KernelSpecManager(
                kernel_dirs=[directory], ensure_native_kernel=False),
        )
        NotebookClient(notebook, km=manager, timeout=240,
                       resources={"metadata": {"path": str(notebook_dir)}}).execute(cleanup_kc=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, help="Export an HTML reading copy.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    notebook_path = root.parent / "nso-semantic-workflows.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    execute_saved(notebook, root.parent)
    nbformat.write(notebook, notebook_path)
    if args.html:
        helper = root.parent / "record-linkage-tutorial" / "run_notebook.py"
        spec = importlib.util.spec_from_file_location("tutorial_export", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        html, _ = module.reading_exporter().from_notebook_node(notebook)
        args.html.parent.mkdir(parents=True, exist_ok=True)
        html = module.rebase_file_links(html, notebook_path.parent, args.html.resolve().parent)
        args.html.write_text(html, encoding="utf-8")
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {code_cells} code cells from saved evidence; no API calls.")
    print(notebook_path)


if __name__ == "__main__":
    main()
