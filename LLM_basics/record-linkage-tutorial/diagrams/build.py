"""Build the editable TikZ figures with an installed TeX distribution and Poppler.

Run from any directory: python3 path/to/diagrams/build.py [figure-stem ...]
Requires pdflatex and pdftoppm on PATH. PNGs are rendered at 220 dpi.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent

def build(stem):
    source = ROOT / f'{stem}.tex'
    if not source.is_file() or stem == 'diagram-style':
        raise ValueError(f'Unknown figure: {stem}')
    with tempfile.TemporaryDirectory(prefix=f'{stem}-') as tmp:
        proc = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
             f'-output-directory={tmp}', source.name],
            cwd=ROOT, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f'{stem}:\n{proc.stdout[-5000:]}')
        log = (Path(tmp) / f'{stem}.log').read_text()
        warnings = [x for x in log.splitlines() if 'Overfull' in x or 'Warning' in x]
        shutil.copy2(Path(tmp) / f'{stem}.pdf', ROOT / f'{stem}.pdf')
        subprocess.run(['pdftoppm', '-png', '-r', '220', '-singlefile',
                        str(ROOT / f'{stem}.pdf'), str(ROOT / stem)], check=True,
                       capture_output=True)
        return stem, warnings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figures', nargs='*')
    args = parser.parse_args()
    stems = args.figures or [p.stem for p in sorted(ROOT.glob('[0-9][0-9]-*.tex'))]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for stem, warnings in pool.map(build, stems):
            print(f'Built {stem}.pdf and {stem}.png')
            for warning in warnings:
                print(f'  {warning}')
