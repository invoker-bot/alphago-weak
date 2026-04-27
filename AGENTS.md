# Repository Guidelines

## Agent-Specific Instructions
Always reply in Chinese unless the user explicitly requests another language.

## Project Structure & Module Organization
`alphago_weak/` is the main package: `board/` has Go rules, `dataset/` parses SGF data, `model/` contains PyTorch neural-network and MCTS code, `gtp/` contains bots, `gui/` contains the interface, and `utils/` has helpers. `main.py` exposes dataset download, supervised training, experimental MCTS self-play, evaluation, and GTP play modes. Tests are in `test/`, fixtures in `test/sgf/`, and images in `img/`. Generated datasets, logs, weights, `.pt`, `.h5`, and `.data/` stay ignored.

## Build, Test, and Development Commands
- `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`: create and activate a Windows virtual environment.
- `pip install -e .`: install PyTorch, Typer, pytest, SGF parsing, and the `alphago` console command.
- `pytest`: run the full test suite, including benchmark-backed board tests.
- `pytest test/test_board.py`: run board rules and SGF replay tests only.
- `alphago --help`: list packaged CLI commands.
- `alphago download --type u-go`: download training data into `.data/`.
- `alphago train --init 4096 --epochs 1`: smoke-test preprocessing and training with a short run.
- `alphago self-play --steps 1 --epochs 1 --simulations 1 --max-moves 2 --record-games 1 --replay-buffer-size 128`: smoke-test the experimental AlphaGo Zero-style self-play loop, replay-buffer training, and sample SGF output; do not treat it as a validated full reinforcement-learning pipeline.

## Coding Style & Naming Conventions
Use Python with 4-space indentation. Keep modules, functions, and variables in `snake_case`; classes in `CamelCase`; constants uppercase where practical. Preserve domain prefixes such as `Go*`, `GTP*`, and `AlphaGoWeak*`. Prefer type hints for public helpers. No formatter or linter is configured, so group imports as standard library, third-party, then local package imports.

## Testing Guidelines
Tests use `pytest` and `pytest-benchmark`. Name files `test_*.py` and functions `test_*`. Add board regressions to `test/test_board.py`, model encoding tests to `test/test_alphagoweak.py`, and reusable fixtures to `test/sgf/`. Keep fixtures small and deterministic. Run `pytest` before opening a PR.

## Commit & Pull Request Guidelines
History uses short imperative commit messages such as `Add MCTS self-play logic.` or `Edit some docs.` Keep each commit focused. Pull requests should describe behavior changes, list tests run, note training-data or model-weight implications, link issues, and include screenshots when GUI or documentation images change.

## Security & Configuration Tips
Do not commit datasets, trained weights, logs, virtual environments, or generated `.pt`/`.h5` files. Keep large experiments under ignored paths such as `.data/` or `weights/`.
