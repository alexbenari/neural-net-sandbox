# Repository Guidelines

## Project Structure & Module Organization
- `sandbox.py` is the main runnable script with a small PyTorch model and training loop.
- `neural_nets_sandbox.ipynb` is a notebook sandbox for experiments and exploration.
- `config/` holds local runtime configuration (e.g., `config/hypercorn.toml`) and secrets (`config/secret_key`).
- No dedicated `src/` or `tests/` directories exist yet; add them if the project grows.

## Build, Test, and Development Commands
- Run the script: `python sandbox.py` (prints the model and trains on dummy data).
- Use the notebook: `jupyter notebook neural_nets_sandbox.ipynb` (interactive exploration).
- If you add dependencies, document them in a `requirements.txt` or `pyproject.toml`.

## Coding Style & Naming Conventions
- Python with 4-space indentation; keep functions and classes small and readable.
- Prefer descriptive, lowercase `snake_case` for variables/functions and `PascalCase` for classes.
- No formatter or linter is configured; follow PEP 8 conventions and keep imports grouped.

## Testing Guidelines
- There is no automated test suite yet.
- If you introduce tests, place them under `tests/` and name files `test_*.py`.
- Consider `pytest` as the default runner: `pytest -q` once added.

## Commit & Pull Request Guidelines
- Existing commits are short, imperative messages (e.g., "Initial commit").
- Keep commit titles concise and action-oriented; add context in the body if needed.
- PRs should include: a short description, steps to run/reproduce, and screenshots or notebook outputs when they affect visuals or results.

## Security & Configuration Tips
- Treat `config/secret_key` as sensitive; do not share it outside the repo.
- Keep runtime settings in `config/hypercorn.toml` and document any new config keys in this file.

## Agent-Specific Instructions
- Keep changes minimal and scoped; update this guide if you add new folders or tooling.
