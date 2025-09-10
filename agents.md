# Repository Guidelines

## Project Structure & Module Organization
- Topic-first layout; each folder groups related notes, notebooks, and small code:
  - `Agents/`, `LLM_basics/`, `Systems/`, `LangGraph/`, `DSPy/`, `RL/`, `PPI/`, `MCP/`, `Theory/`, `Presentations/`.
- Root holds high-level docs (e.g., `README.md`, `AGENTS.md`). Keep content docs and images in topic folders (e.g., `Systems/`, `LLM_basics/`, or `Presentations/`).
- Place images next to the doc/notebook that uses them, or under `Presentations/`; reference via relative paths.

## Build, Test, and Development Commands
- Python: repo targets 3.11.x (see `.python-version`).
- Create env: `python -m venv .venv && source .venv/bin/activate`.
- Launch notebooks: `jupyter lab` (or `jupyter notebook`).
- Execute a notebook (smoke test):
  `jupyter nbconvert --to notebook --execute path/to/notebook.ipynb --inplace`.

## Coding Style & Naming Conventions
- Docs/notebooks/images use kebab-case (e.g., `structured-extraction.ipynb`, `function-calling.png`).
- Python follows PEP 8, 4-space indents, type hints; use snake_case for module files and identifiers (hyphens aren’t importable).
- Markdown: concise sections, fenced code blocks, relative links.

## Testing Guidelines
- No formal unit tests; validate by running notebooks end-to-end before committing.
- Add brief example cells with expected outputs; avoid hidden environment assumptions.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood (e.g., `docs: add policy gradient notes`).
- PRs include: short description, impacted folders, screenshots (if visual), and links to related notes/issues.
- Ensure notebooks execute cleanly, outputs trimmed, assets referenced via relative paths.

## Agent & Demos
- For agent demos (e.g., LangGraph, MCP), include a short folder `README.md` with setup, run commands, and an example I/O trace. Include function/tool schemas for reproducibility.
